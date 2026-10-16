// Ported from Ceres Solver 2.0.0: internal/ceres/trust_region_minimizer.cc,
// internal/ceres/minimizer.cc and internal/ceres/callbacks.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/trust_region_minimizer.cc).
//
// Copyright 2016 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Generic trust region minimization algorithm.
//!
//! For example usage, see SolverImpl::Minimize.
//!
//! RUST PORT NOTE: inner iterations and trust region problem dumping are not
//! ported.

use std::time::Instant;

use crate::line_search::{ArmijoLineSearch, LineSearchFunction, LineSearchOptions};
use crate::program::{BlockJacobian, Evaluator};
use crate::solver::{IterationSummary, SolverOptions, SolverSummary};
use crate::stringprintf::{fmt_d, fmt_e};
use crate::trust_region::{LevenbergMarquardtStrategy, TrustRegionStepEvaluator};
use crate::types::{CallbackReturnType, LinearSolverTerminationType, LoggingType, TerminationType};

/// The trust region minimizer state (C++ `TrustRegionMinimizer`).
pub(crate) struct TrustRegionMinimizer<'e, 'p, 'a, 'o> {
    options: &'o SolverOptions,
    evaluator: &'e mut Evaluator<'p, 'a>,
    strategy: LevenbergMarquardtStrategy,
    jacobian: BlockJacobian,
    step_evaluator: Option<TrustRegionStepEvaluator>,

    is_not_silent: bool,
    // Use a bounds constrained optimization algorithm (C++
    // `Minimizer::Options::is_constrained`).
    is_constrained: bool,
    num_parameters: usize,

    start_time: Instant,
    iteration_start_time: Instant,
    preprocessor_time_in_seconds: f64,

    // Current point.
    x: Vec<f64>,
    // Residuals at x.
    residuals: Vec<f64>,
    // Gradient at x.
    gradient: Vec<f64>,
    // Solution to the trust region problem.
    trust_region_step: Vec<f64>,
    // delta = trust_region_step * jacobian_scaling
    delta: Vec<f64>,
    // candidate_x = x + delta
    candidate_x: Vec<f64>,
    // model_residuals = J * trust_region_step
    model_residuals: Vec<f64>,
    negative_gradient: Vec<f64>,
    // projected_gradient_step = Plus(x, -gradient), an intermediate quantity
    // used to compute the projected gradient norm.
    projected_gradient_step: Vec<f64>,
    // The jacobian scaling vector.
    jacobian_scaling: Vec<f64>,

    // Euclidean norm of x.
    x_norm: f64,
    // Cost at x.
    x_cost: f64,
    // Minimum cost encountered up till now.
    minimum_cost: f64,
    // How much did the trust region solver reduce the cost of the
    // linearization/the trust region model.
    model_cost_change: f64,
    // Cost at candidate_x.
    candidate_cost: f64,

    // Number of consecutive steps where the minimizer loop computed a
    // numerically invalid step.
    num_consecutive_invalid_steps: i32,

    iteration_summary: IterationSummary,
}

impl<'e, 'p, 'a, 'o> TrustRegionMinimizer<'e, 'p, 'a, 'o> {
    /// Initialize the minimizer, allocate working space and set some of the
    /// fields in the solver_summary.
    pub(crate) fn new(
        options: &'o SolverOptions,
        evaluator: &'e mut Evaluator<'p, 'a>,
        strategy: LevenbergMarquardtStrategy,
        is_constrained: bool,
        preprocessor_time_in_seconds: f64,
    ) -> Self {
        let num_parameters = evaluator.program.num_parameters;
        let num_effective_parameters = evaluator.program.num_effective_parameters;
        let num_residuals = evaluator.program.num_residuals;
        let jacobian = evaluator.create_jacobian();
        let now = Instant::now();
        Self {
            options,
            evaluator,
            strategy,
            jacobian,
            step_evaluator: None,
            is_not_silent: options.logging_type != LoggingType::Silent,
            is_constrained,
            num_parameters,
            start_time: now,
            iteration_start_time: now,
            preprocessor_time_in_seconds,
            x: Vec::new(),
            residuals: vec![0.0; num_residuals],
            gradient: vec![0.0; num_effective_parameters],
            trust_region_step: vec![0.0; num_effective_parameters],
            delta: vec![0.0; num_effective_parameters],
            candidate_x: vec![0.0; num_parameters],
            model_residuals: vec![0.0; num_residuals],
            negative_gradient: vec![0.0; num_effective_parameters],
            projected_gradient_step: vec![0.0; num_parameters],
            // By default scaling is one, if the user requests Jacobi scaling
            // of the Jacobian, we will compute and overwrite this vector.
            jacobian_scaling: vec![1.0; num_effective_parameters],
            x_norm: -1.0, // Invalid value
            x_cost: f64::MAX,
            minimum_cost: f64::MAX,
            model_cost_change: 0.0,
            candidate_cost: 0.0,
            num_consecutive_invalid_steps: 0,
            iteration_summary: IterationSummary::default(),
        }
    }

    /// Minimizes starting from `parameters`; on return `parameters` holds
    /// the best point found.
    pub(crate) fn minimize(&mut self, parameters: &mut [f64], solver_summary: &mut SolverSummary) {
        self.start_time = Instant::now();
        self.iteration_start_time = self.start_time;
        solver_summary.termination_type = TerminationType::NoConvergence;
        solver_summary.num_successful_steps = 0;
        solver_summary.num_unsuccessful_steps = 0;
        solver_summary.is_constrained = self.is_constrained;
        self.x = parameters[..self.num_parameters].to_vec();

        if !self.iteration_zero(solver_summary) {
            log::error!("Terminating: {}", solver_summary.message);
            return;
        }

        // Create the TrustRegionStepEvaluator. The construction needs to be
        // delayed to this point because we need the cost for the starting
        // point to initialize the step evaluator.
        self.step_evaluator = Some(TrustRegionStepEvaluator::new(
            self.x_cost,
            if self.options.use_nonmonotonic_steps {
                self.options.max_consecutive_nonmonotonic_steps
            } else {
                0
            },
        ));

        while self
            .finalize_iteration_and_check_if_minimizer_can_continue(parameters, solver_summary)
        {
            self.iteration_start_time = Instant::now();

            let previous_gradient_norm = self.iteration_summary.gradient_norm;
            let previous_gradient_max_norm = self.iteration_summary.gradient_max_norm;

            self.iteration_summary = IterationSummary {
                iteration: solver_summary
                    .iterations
                    .last()
                    .map_or(0, |s| s.iteration + 1),
                ..IterationSummary::default()
            };

            if !self.compute_trust_region_step(solver_summary) {
                log::error!("Terminating: {}", solver_summary.message);
                return;
            }
            if !self.iteration_summary.step_is_valid {
                if !self.handle_invalid_step(solver_summary) {
                    log::error!("Terminating: {}", solver_summary.message);
                    return;
                }
                continue;
            }

            if self.is_constrained && self.options.max_num_line_search_step_size_iterations > 0 {
                // Use a projected line search to enforce the bounds
                // constraints and improve the quality of the step.
                self.do_line_search(solver_summary);
            }

            self.compute_candidate_point_and_evaluate_cost();

            if self.parameter_tolerance_reached(solver_summary) {
                return;
            }

            if self.function_tolerance_reached(solver_summary) {
                return;
            }

            if self.is_step_successful() {
                if !self.handle_successful_step(solver_summary) {
                    log::error!("Terminating: {}", solver_summary.message);
                    return;
                }
            } else {
                // Declare the step unsuccessful and inform the trust region
                // strategy.
                self.iteration_summary.step_is_successful = false;
                self.iteration_summary.cost = self.candidate_cost + solver_summary.fixed_cost;

                // When the step is unsuccessful, we do not compute the
                // gradient (or update x), so we preserve its value from the
                // last successful iteration.
                self.iteration_summary.gradient_norm = previous_gradient_norm;
                self.iteration_summary.gradient_max_norm = previous_gradient_max_norm;
                self.strategy
                    .step_rejected(self.iteration_summary.relative_decrease);
            }
        }
    }

    /// Number of linear solves and the time spent in them.
    pub(crate) fn linear_solver_statistics(&self) -> (i32, f64) {
        (
            self.strategy.num_linear_solves,
            self.strategy.linear_solver_time_in_seconds,
        )
    }

    /// 1. Project the initial solution onto the feasible set if needed.
    /// 2. Compute the initial cost, jacobian & gradient.
    ///
    /// Return true if all computations can be performed successfully.
    fn iteration_zero(&mut self, solver_summary: &mut SolverSummary) -> bool {
        self.iteration_summary = IterationSummary {
            iteration: 0,
            step_is_valid: false,
            step_is_successful: false,
            cost_change: 0.0,
            gradient_max_norm: 0.0,
            gradient_norm: 0.0,
            step_norm: 0.0,
            relative_decrease: 0.0,
            eta: self.options.eta,
            linear_solver_iterations: 0,
            step_solver_time_in_seconds: 0.0,
            ..IterationSummary::default()
        };

        if self.is_constrained {
            self.delta.fill(0.0);
            if !self
                .evaluator
                .plus(&self.x, &self.delta, &mut self.candidate_x)
            {
                solver_summary.message =
                    "Unable to project initial point onto the feasible set.".to_string();
                solver_summary.termination_type = TerminationType::Failure;
                return false;
            }

            self.x.copy_from_slice(&self.candidate_x);
            self.x_norm = self.x.iter().map(|v| v * v).sum::<f64>().sqrt();
        }

        if !self.evaluate_gradient_and_jacobian(solver_summary) {
            return false;
        }

        solver_summary.initial_cost = self.x_cost + solver_summary.fixed_cost;
        self.iteration_summary.step_is_valid = true;
        self.iteration_summary.step_is_successful = true;
        true
    }

    /// For the current x, compute
    ///
    ///  1. Cost
    ///  2. Jacobian
    ///  3. Gradient
    ///  4. Scale the Jacobian if needed (and compute the scaling if we are
    ///     in iteration zero).
    ///  5. Compute the 2 and max norm of the gradient.
    ///
    /// Returns true if all computations could be performed successfully. Any
    /// failures are considered fatal and the Solver::Summary is updated to
    /// indicate this.
    fn evaluate_gradient_and_jacobian(&mut self, solver_summary: &mut SolverSummary) -> bool {
        if !self.evaluator.evaluate(
            &self.x,
            true,
            &mut self.x_cost,
            Some(&mut self.residuals),
            Some(&mut self.gradient),
            Some(&mut self.jacobian),
        ) {
            solver_summary.message = "Residual and Jacobian evaluation failed.".to_string();
            solver_summary.termination_type = TerminationType::Failure;
            return false;
        }

        self.iteration_summary.cost = self.x_cost + solver_summary.fixed_cost;

        if self.options.jacobi_scaling {
            if self.iteration_summary.iteration == 0 {
                // Compute a scaling vector that is used to improve the
                // conditioning of the Jacobian.
                //
                // jacobian_scaling_ = diag(J'J)^{-1}
                self.jacobian
                    .squared_column_norm(&mut self.jacobian_scaling);
                for s in &mut self.jacobian_scaling {
                    // Add one to the denominator to prevent division by zero.
                    *s = 1.0 / (1.0 + s.sqrt());
                }
            }

            // jacobian = jacobian * diag(J'J) ^{-1}
            self.jacobian.scale_columns(&self.jacobian_scaling);
        }

        // The gradient exists in the local tangent space. To account for the
        // bounds constraints correctly, instead of just computing the norm
        // of the gradient vector, we compute
        //
        // |Plus(x, -gradient) - x|
        //
        // Where the Plus operator lifts the negative gradient to the ambient
        // space, adds it to x and projects it on the hypercube defined by
        // the bounds.
        for (n, g) in self.negative_gradient.iter_mut().zip(&self.gradient) {
            *n = -*g;
        }
        if !self.evaluator.plus(
            &self.x,
            &self.negative_gradient,
            &mut self.projected_gradient_step,
        ) {
            solver_summary.message =
                "projected_gradient_step = Plus(x, -gradient) failed.".to_string();
            solver_summary.termination_type = TerminationType::Failure;
            return false;
        }

        let mut max_norm: f64 = 0.0;
        let mut squared_norm = 0.0;
        for (x, p) in self.x.iter().zip(&self.projected_gradient_step) {
            let d = x - p;
            max_norm = max_norm.max(d.abs());
            squared_norm += d * d;
        }
        self.iteration_summary.gradient_max_norm = max_norm;
        self.iteration_summary.gradient_norm = squared_norm.sqrt();
        true
    }

    /// 1. Add the final timing information to the iteration summary.
    /// 2. Run the callbacks
    /// 3. Check for termination based on
    ///    a. Run time
    ///    b. Iteration count
    ///    c. Max norm of the gradient
    ///    d. Size of the trust region radius.
    ///
    /// Returns true if user did not terminate the solver and none of these
    /// termination criterion are met.
    fn finalize_iteration_and_check_if_minimizer_can_continue(
        &mut self,
        parameters: &mut [f64],
        solver_summary: &mut SolverSummary,
    ) -> bool {
        if self.iteration_summary.step_is_successful {
            solver_summary.num_successful_steps += 1;
            if self.x_cost < self.minimum_cost {
                self.minimum_cost = self.x_cost;
                parameters[..self.num_parameters].copy_from_slice(&self.x);
                self.iteration_summary.step_is_nonmonotonic = false;
            } else {
                self.iteration_summary.step_is_nonmonotonic = true;
            }
        } else {
            solver_summary.num_unsuccessful_steps += 1;
        }

        self.iteration_summary.trust_region_radius = self.strategy.radius();
        self.iteration_summary.iteration_time_in_seconds =
            self.iteration_start_time.elapsed().as_secs_f64();
        self.iteration_summary.cumulative_time_in_seconds =
            self.start_time.elapsed().as_secs_f64() + self.preprocessor_time_in_seconds;
        solver_summary
            .iterations
            .push(self.iteration_summary.clone());

        if !self.run_callbacks(solver_summary) {
            return false;
        }

        if self.max_solver_time_reached(solver_summary) {
            return false;
        }

        if self.max_solver_iterations_reached(solver_summary) {
            return false;
        }

        if self.gradient_tolerance_reached(solver_summary) {
            return false;
        }

        if self.min_trust_region_radius_reached(solver_summary) {
            return false;
        }

        true
    }

    /// C++ `Minimizer::RunCallbacks`, with the `LoggingCallback` first.
    fn run_callbacks(&self, summary: &mut SolverSummary) -> bool {
        if self.is_not_silent {
            log_iteration(
                &self.iteration_summary,
                self.options.minimizer_progress_to_stdout,
            );
        }
        let mut status = CallbackReturnType::SolverContinue;
        for callback in &self.options.callbacks {
            status = callback.call(&self.iteration_summary);
            if status != CallbackReturnType::SolverContinue {
                break;
            }
        }
        match status {
            CallbackReturnType::SolverContinue => true,
            CallbackReturnType::SolverTerminateSuccessfully => {
                summary.termination_type = TerminationType::UserSuccess;
                summary.message =
                    "User callback returned SOLVER_TERMINATE_SUCCESSFULLY.".to_string();
                if self.is_not_silent {
                    log::debug!("Terminating: {}", summary.message);
                }
                false
            }
            CallbackReturnType::SolverAbort => {
                summary.termination_type = TerminationType::UserFailure;
                summary.message = "User callback returned SOLVER_ABORT.".to_string();
                if self.is_not_silent {
                    log::debug!("Terminating: {}", summary.message);
                }
                false
            }
        }
    }

    /// Compute the trust region step using the TrustRegionStrategy chosen by
    /// the user.
    ///
    /// If the strategy returns with LINEAR_SOLVER_FATAL_ERROR, which
    /// indicates an unrecoverable error, return false. This is the only
    /// condition that returns false.
    ///
    /// If the strategy returns with LINEAR_SOLVER_FAILURE, which indicates a
    /// numerical failure that could be recovered from by retrying (e.g. by
    /// increasing the strength of the regularization), we set
    /// iteration_summary_.step_is_valid to false and return true.
    ///
    /// In all other cases, we compute the decrease in the trust region model
    /// problem. In exact arithmetic, this should always be positive, but due
    /// to numerical problems in the TrustRegionStrategy or round off error
    /// when computing the decrease it may be negative. In which case again,
    /// we set iteration_summary_.step_is_valid to false.
    fn compute_trust_region_step(&mut self, solver_summary: &mut SolverSummary) -> bool {
        let strategy_start_time = Instant::now();
        self.iteration_summary.step_is_valid = false;
        self.iteration_summary.eta = self.options.eta;

        let strategy_summary = self.strategy.compute_step(
            &self.jacobian,
            &self.residuals,
            &mut self.trust_region_step,
        );

        if strategy_summary.termination_type == LinearSolverTerminationType::FatalError {
            solver_summary.message = "Linear solver failed due to unrecoverable non-numeric \
                                      causes. Please see the error log for clues. "
                .to_string();
            solver_summary.termination_type = TerminationType::Failure;
            return false;
        }

        self.iteration_summary.step_solver_time_in_seconds =
            strategy_start_time.elapsed().as_secs_f64();
        self.iteration_summary.linear_solver_iterations = strategy_summary.num_iterations;

        if strategy_summary.termination_type == LinearSolverTerminationType::Failure {
            return true;
        }

        // new_model_cost
        //  = 1/2 [f + J * step]^2
        //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
        // model_cost_change
        //  = cost - new_model_cost
        //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
        //  = -f'J * step - step' * J' * J * step / 2
        //  = -(J * step)'(f + J * step / 2)
        self.model_residuals.fill(0.0);
        self.jacobian
            .right_multiply(&self.trust_region_step, &mut self.model_residuals);
        let mut dot = 0.0;
        for (m, r) in self.model_residuals.iter().zip(&self.residuals) {
            dot += m * (r + m / 2.0);
        }
        self.model_cost_change = -dot;

        // TODO(sameeragarwal)
        //
        //  1. What happens if model_cost_change_ = 0
        //  2. What happens if -epsilon <= model_cost_change_ < 0 for some
        //     small epsilon due to round off error.
        self.iteration_summary.step_is_valid = self.model_cost_change > 0.0;
        if self.iteration_summary.step_is_valid {
            // Undo the Jacobian column scaling.
            for ((d, s), j) in self
                .delta
                .iter_mut()
                .zip(&self.trust_region_step)
                .zip(&self.jacobian_scaling)
            {
                *d = s * j;
            }
            self.num_consecutive_invalid_steps = 0;
        }

        if self.is_not_silent && !self.iteration_summary.step_is_valid {
            log::debug!(
                "Invalid step: current_cost: {} absolute model cost change: {} relative model cost change: {}",
                self.x_cost,
                self.model_cost_change,
                self.model_cost_change / self.x_cost
            );
        }
        true
    }

    /// Invalid steps can happen due to a number of reasons, and we allow a
    /// limited number of consecutive failures, and return false if this
    /// limit is exceeded.
    fn handle_invalid_step(&mut self, solver_summary: &mut SolverSummary) -> bool {
        // TODO(sameeragarwal): Should we be returning FAILURE or
        // NO_CONVERGENCE? The solution value is still usable in many cases,
        // it is not clear if we should declare the solver a failure
        // entirely. For example the case where model_cost_change ~ 0.0, but
        // just slightly negative.
        self.num_consecutive_invalid_steps += 1;
        if self.num_consecutive_invalid_steps >= self.options.max_num_consecutive_invalid_steps {
            solver_summary.message = format!(
                "Number of consecutive invalid steps more than \
                 Solver::Options::max_num_consecutive_invalid_steps: {}",
                self.options.max_num_consecutive_invalid_steps
            );
            solver_summary.termination_type = TerminationType::Failure;
            return false;
        }

        self.strategy.step_is_invalid();

        // We are going to try and reduce the trust region radius and solve
        // again. To do this, we are going to treat this iteration as an
        // unsuccessful iteration. Since the various callbacks are still
        // executed, we are going to fill the iteration summary with data that
        // assumes a step of length zero and no progress.
        let last = solver_summary.iterations.last().expect("iteration zero");
        self.iteration_summary.cost = self.x_cost + solver_summary.fixed_cost;
        self.iteration_summary.cost_change = 0.0;
        self.iteration_summary.gradient_max_norm = last.gradient_max_norm;
        self.iteration_summary.gradient_norm = last.gradient_norm;
        self.iteration_summary.step_norm = 0.0;
        self.iteration_summary.relative_decrease = 0.0;
        self.iteration_summary.eta = self.options.eta;
        true
    }

    /// Use an Armijo line search along `delta` (C++ `DoLineSearch`, called
    /// with `x_`, `gradient_`, `x_cost_` and `&delta_`) to project the step
    /// onto the bounds constraints and improve its quality: on success
    /// `delta` is scaled by the step size found.
    ///
    /// TODO(sameeragarwal): The current implementation does not do anything
    /// illegal but is incorrect and not terribly effective.
    ///
    /// https://github.com/ceres-solver/ceres-solver/issues/187
    fn do_line_search(&mut self, solver_summary: &mut SolverSummary) {
        let options = self.options;
        let line_search_options = LineSearchOptions {
            is_silent: true,
            interpolation_type: options.line_search_interpolation_type,
            min_step_size: options.min_line_search_step_size,
            sufficient_decrease: options.line_search_sufficient_function_decrease,
            max_step_contraction: options.max_line_search_step_contraction,
            min_step_contraction: options.min_line_search_step_contraction,
            max_num_iterations: options.max_num_line_search_step_size_iterations,
            sufficient_curvature_decrease: options.line_search_sufficient_curvature_decrease,
            max_step_expansion: options.max_line_search_step_expansion,
        };

        let cost = self.x_cost;
        let initial_gradient: f64 = self
            .gradient
            .iter()
            .zip(&self.delta)
            .map(|(g, d)| g * d)
            .sum();
        let mut line_search_function = LineSearchFunction::new(&mut *self.evaluator);
        let line_search = ArmijoLineSearch::new(line_search_options);
        line_search_function.init(&self.x, &self.delta);
        let line_search_summary =
            line_search.search(&mut line_search_function, 1.0, cost, initial_gradient);

        solver_summary.num_line_search_steps += line_search_summary.num_iterations;
        solver_summary.line_search_cost_evaluation_time_in_seconds +=
            line_search_summary.cost_evaluation_time_in_seconds;
        solver_summary.line_search_gradient_evaluation_time_in_seconds +=
            line_search_summary.gradient_evaluation_time_in_seconds;
        solver_summary.line_search_polynomial_minimization_time_in_seconds +=
            line_search_summary.polynomial_minimization_time_in_seconds;
        solver_summary.line_search_total_time_in_seconds +=
            line_search_summary.total_time_in_seconds;

        if line_search_summary.success {
            let step = line_search_summary.optimal_point.x;
            for d in &mut self.delta {
                *d *= step;
            }
        }
    }

    /// Check if the maximum amount of time allowed by the user for the
    /// solver has been exceeded, and if so return false after updating
    /// Solver::Summary::message.
    fn max_solver_time_reached(&self, solver_summary: &mut SolverSummary) -> bool {
        let total_solver_time =
            self.start_time.elapsed().as_secs_f64() + self.preprocessor_time_in_seconds;
        if total_solver_time < self.options.max_solver_time_in_seconds {
            return false;
        }

        solver_summary.message = format!(
            "Maximum solver time reached. Total solver time: {} >= {}.",
            fmt_e(total_solver_time, 0, 6, false),
            fmt_e(self.options.max_solver_time_in_seconds, 0, 6, false)
        );
        solver_summary.termination_type = TerminationType::NoConvergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Check if the maximum number of iterations allowed by the user for the
    /// solver has been exceeded, and if so return false after updating
    /// Solver::Summary::message.
    fn max_solver_iterations_reached(&self, solver_summary: &mut SolverSummary) -> bool {
        if self.iteration_summary.iteration < self.options.max_num_iterations {
            return false;
        }

        solver_summary.message = format!(
            "Maximum number of iterations reached. Number of iterations: {}.",
            self.iteration_summary.iteration
        );
        solver_summary.termination_type = TerminationType::NoConvergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Check convergence based on the max norm of the gradient (only for
    /// iterations where the step was declared successful).
    fn gradient_tolerance_reached(&self, solver_summary: &mut SolverSummary) -> bool {
        if !self.iteration_summary.step_is_successful
            || self.iteration_summary.gradient_max_norm > self.options.gradient_tolerance
        {
            return false;
        }

        solver_summary.message = format!(
            "Gradient tolerance reached. Gradient max norm: {} <= {}",
            fmt_e(self.iteration_summary.gradient_max_norm, 0, 6, false),
            fmt_e(self.options.gradient_tolerance, 0, 6, false)
        );
        solver_summary.termination_type = TerminationType::Convergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Check convergence based the size of the trust region radius.
    fn min_trust_region_radius_reached(&self, solver_summary: &mut SolverSummary) -> bool {
        if self.iteration_summary.trust_region_radius > self.options.min_trust_region_radius {
            return false;
        }

        solver_summary.message = format!(
            "Minimum trust region radius reached. Trust region radius: {} <= {}",
            fmt_e(self.iteration_summary.trust_region_radius, 0, 6, false),
            fmt_e(self.options.min_trust_region_radius, 0, 6, false)
        );
        solver_summary.termination_type = TerminationType::Convergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Solver::Options::parameter_tolerance based convergence check.
    fn parameter_tolerance_reached(&mut self, solver_summary: &mut SolverSummary) -> bool {
        // Compute the norm of the step in the ambient space.
        self.iteration_summary.step_norm = self
            .x
            .iter()
            .zip(&self.candidate_x)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let step_size_tolerance =
            self.options.parameter_tolerance * (self.x_norm + self.options.parameter_tolerance);

        if self.iteration_summary.step_norm > step_size_tolerance {
            return false;
        }

        solver_summary.message = format!(
            "Parameter tolerance reached. Relative step_norm: {} <= {}.",
            fmt_e(
                self.iteration_summary.step_norm / (self.x_norm + self.options.parameter_tolerance),
                0,
                6,
                false
            ),
            fmt_e(self.options.parameter_tolerance, 0, 6, false)
        );
        solver_summary.termination_type = TerminationType::Convergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Solver::Options::function_tolerance based convergence check.
    fn function_tolerance_reached(&mut self, solver_summary: &mut SolverSummary) -> bool {
        self.iteration_summary.cost_change = self.x_cost - self.candidate_cost;
        let absolute_function_tolerance = self.options.function_tolerance * self.x_cost;

        if self.iteration_summary.cost_change.abs() > absolute_function_tolerance {
            return false;
        }

        solver_summary.message = format!(
            "Function tolerance reached. |cost_change|/cost: {} <= {}",
            fmt_e(
                self.iteration_summary.cost_change.abs() / self.x_cost,
                0,
                6,
                false
            ),
            fmt_e(self.options.function_tolerance, 0, 6, false)
        );
        solver_summary.termination_type = TerminationType::Convergence;
        if self.is_not_silent {
            log::debug!("Terminating: {}", solver_summary.message);
        }
        true
    }

    /// Compute candidate_x_ = Plus(x_, delta_)
    /// Evaluate the cost of candidate_x_ as candidate_cost_.
    ///
    /// Failure to compute the step or the cost mean that candidate_cost_ is
    /// set to std::numeric_limits<double>::max(). Unlike
    /// EvaluateGradientAndJacobian, failure in this function is not fatal as
    /// we are only computing and evaluating a candidate point, and if for
    /// some reason we are unable to evaluate it, we consider it to be a
    /// point with very high cost. This allows the user to deal with edge
    /// cases/constraints as part of the LocalParameterization and
    /// CostFunction objects.
    fn compute_candidate_point_and_evaluate_cost(&mut self) {
        if !self
            .evaluator
            .plus(&self.x, &self.delta, &mut self.candidate_x)
        {
            if self.is_not_silent {
                log::warn!(
                    "x_plus_delta = Plus(x, delta) failed. Treating it as a step with infinite cost"
                );
            }
            self.candidate_cost = f64::MAX;
            return;
        }

        if !self.evaluator.evaluate(
            &self.candidate_x,
            true,
            &mut self.candidate_cost,
            None,
            None,
            None,
        ) {
            if self.is_not_silent {
                log::warn!("Step failed to evaluate. Treating it as a step with infinite cost");
            }
            self.candidate_cost = f64::MAX;
        }
    }

    fn is_step_successful(&mut self) -> bool {
        self.iteration_summary.relative_decrease = self
            .step_evaluator
            .as_ref()
            .expect("step evaluator")
            .step_quality(self.candidate_cost, self.model_cost_change);

        // In most cases, boosting the model_cost_change by the improvement
        // caused by the inner iterations is fine, but inner iterations are
        // not ported, see the module documentation.
        self.iteration_summary.relative_decrease > self.options.min_relative_decrease
    }

    /// Declare the step successful, move to candidate_x, update the
    /// derivatives and let the trust region strategy and the step evaluator
    /// know that the step has been accepted.
    fn handle_successful_step(&mut self, solver_summary: &mut SolverSummary) -> bool {
        self.x.copy_from_slice(&self.candidate_x);
        self.x_norm = self.x.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Since the step was successful, this point has already had the
        // residual evaluated (but not the jacobian). So indicate that to the
        // evaluator.
        if !self.evaluate_gradient_and_jacobian(solver_summary) {
            return false;
        }

        self.iteration_summary.step_is_successful = true;
        self.strategy
            .step_accepted(self.iteration_summary.relative_decrease);
        self.step_evaluator
            .as_mut()
            .expect("step evaluator")
            .step_accepted(self.candidate_cost, self.model_cost_change);
        true
    }
}

/// C++ `LoggingCallback` for the trust region minimizer.
fn log_iteration(summary: &IterationSummary, log_to_stdout: bool) {
    let mut output = String::new();
    if summary.iteration == 0 {
        output.push_str(
            "iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time\n",
        );
    }
    // "% 4d % 8e   % 3.2e   % 3.2e  % 3.2e  % 3.2e % 3.2e     % 4d   % 3.2e   % 3.2e"
    output.push_str(&format!(
        "{} {}   {}   {}  {}  {} {}     {}   {}   {}",
        fmt_d(i64::from(summary.iteration), 4, true),
        fmt_e(summary.cost, 8, 6, true),
        fmt_e(summary.cost_change, 3, 2, true),
        fmt_e(summary.gradient_max_norm, 3, 2, true),
        fmt_e(summary.step_norm, 3, 2, true),
        fmt_e(summary.relative_decrease, 3, 2, true),
        fmt_e(summary.trust_region_radius, 3, 2, true),
        fmt_d(i64::from(summary.linear_solver_iterations), 4, true),
        fmt_e(summary.iteration_time_in_seconds, 3, 2, true),
        fmt_e(summary.cumulative_time_in_seconds, 3, 2, true),
    ));
    if log_to_stdout {
        println!("{output}");
    } else {
        log::debug!("{output}");
    }
}
