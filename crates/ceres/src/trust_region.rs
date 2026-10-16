// Ported from Ceres Solver 2.0.0: internal/ceres/levenberg_marquardt_strategy.cc,
// internal/ceres/trust_region_strategy.h and
// internal/ceres/trust_region_step_evaluator.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/levenberg_marquardt_strategy.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Levenberg-Marquardt trust region strategy and the (non-monotonic) trust
//! region step evaluator.

use crate::linear_solver::LinearSolver;
use crate::program::BlockJacobian;
use crate::residual_block::{invalidate_array, is_array_valid};
use crate::types::LinearSolverTerminationType;

/// Options of the trust region strategy (C++ `TrustRegionStrategy::Options`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TrustRegionStrategyOptions {
    pub initial_radius: f64,
    pub max_radius: f64,
    /// Minimum and maximum values of the diagonal damping matrix used by
    /// LevenbergMarquardtStrategy.
    pub min_lm_diagonal: f64,
    pub max_lm_diagonal: f64,
}

/// Summary of a step computation (C++ `TrustRegionStrategy::Summary`).
#[derive(Clone, Debug)]
pub(crate) struct TrustRegionStrategySummary {
    pub num_iterations: i32,
    pub termination_type: LinearSolverTerminationType,
}

/// Levenberg-Marquardt step computation and trust region radius update
/// strategy based on a algorithm 6.18 in Madsen, Nielsen & Tingleff's
/// "Methods for Non-linear least squares problems".
pub(crate) struct LevenbergMarquardtStrategy {
    linear_solver: LinearSolver,
    radius: f64,
    max_radius: f64,
    min_diagonal: f64,
    max_diagonal: f64,
    decrease_factor: f64,
    reuse_diagonal: bool,
    /// diagonal_ = diag(J'J)
    diagonal: Vec<f64>,
    /// lm_diagonal_ = sqrt(diagonal_ / radius_);
    lm_diagonal: Vec<f64>,
    /// Number of linear solves and the time spent in them (C++
    /// "LinearSolver::Solve" call statistics).
    pub num_linear_solves: i32,
    pub linear_solver_time_in_seconds: f64,
}

impl LevenbergMarquardtStrategy {
    pub(crate) fn new(options: TrustRegionStrategyOptions, linear_solver: LinearSolver) -> Self {
        assert!(options.min_lm_diagonal > 0.0);
        assert!(options.min_lm_diagonal <= options.max_lm_diagonal);
        assert!(options.max_radius > 0.0);
        Self {
            linear_solver,
            radius: options.initial_radius,
            max_radius: options.max_radius,
            min_diagonal: options.min_lm_diagonal,
            max_diagonal: options.max_lm_diagonal,
            decrease_factor: 2.0,
            reuse_diagonal: false,
            diagonal: Vec::new(),
            lm_diagonal: Vec::new(),
            num_linear_solves: 0,
            linear_solver_time_in_seconds: 0.0,
        }
    }

    /// Use the current radius to solve for the trust region step.
    pub(crate) fn compute_step(
        &mut self,
        jacobian: &BlockJacobian,
        residuals: &[f64],
        step: &mut [f64],
    ) -> TrustRegionStrategySummary {
        let num_parameters = jacobian.num_cols;
        if !self.reuse_diagonal {
            self.diagonal.resize(num_parameters, 0.0);
            jacobian.squared_column_norm(&mut self.diagonal);
            for d in &mut self.diagonal {
                *d = d.max(self.min_diagonal).min(self.max_diagonal);
            }
        }

        self.lm_diagonal.clear();
        self.lm_diagonal
            .extend(self.diagonal.iter().map(|d| (d / self.radius).sqrt()));

        // Invalidate the output array lm_step, so that we can detect if the
        // linear solver generated numerical garbage. This is known to happen
        // for the DENSE_QR and then DENSE_SCHUR solver when the Jacobin is
        // severely rank deficient and mu is too small.
        invalidate_array(&mut step[..num_parameters]);

        // Instead of solving Jx = -r, solve Jy = r. Then x can be found as
        // x = -y, but the inputs jacobian and residuals do not need to be
        // modified.
        let solve_start = std::time::Instant::now();
        let mut summary =
            self.linear_solver
                .solve(jacobian, residuals, Some(&self.lm_diagonal), step);
        self.num_linear_solves += 1;
        self.linear_solver_time_in_seconds += solve_start.elapsed().as_secs_f64();

        match summary.termination_type {
            LinearSolverTerminationType::FatalError => {
                log::warn!("Linear solver fatal error: {}", summary.message);
            }
            LinearSolverTerminationType::Failure => {
                log::warn!(
                    "Linear solver failure. Failed to compute a step: {}",
                    summary.message
                );
            }
            _ if !is_array_valid(&step[..num_parameters]) => {
                log::warn!("Linear solver failure. Failed to compute a finite step.");
                summary.termination_type = LinearSolverTerminationType::Failure;
            }
            _ => {
                for s in &mut step[..num_parameters] {
                    *s = -*s;
                }
            }
        }

        self.reuse_diagonal = true;

        TrustRegionStrategySummary {
            num_iterations: summary.num_iterations,
            termination_type: summary.termination_type,
        }
    }

    /// Inform the strategy that the current step has been accepted, and that
    /// the ratio of the decrease in the non-linear objective to the decrease
    /// in the trust region model is step_quality.
    pub(crate) fn step_accepted(&mut self, step_quality: f64) {
        assert!(step_quality > 0.0);
        self.radius /= (1.0f64 / 3.0).max(1.0 - (2.0 * step_quality - 1.0).powf(3.0));
        self.radius = self.max_radius.min(self.radius);
        self.decrease_factor = 2.0;
        self.reuse_diagonal = false;
    }

    /// Inform the strategy that the current step has been rejected.
    pub(crate) fn step_rejected(&mut self, _step_quality: f64) {
        self.radius /= self.decrease_factor;
        self.decrease_factor *= 2.0;
        self.reuse_diagonal = true;
    }

    /// Treat the current step as a rejected step with no increase in
    /// solution quality. Since rejected steps lead to decrease in the size of
    /// the trust region, the next time ComputeStep is called, this will lead
    /// to a better conditioned system.
    pub(crate) fn step_is_invalid(&mut self) {
        self.step_rejected(0.0);
    }

    /// Current trust region radius.
    pub(crate) fn radius(&self) -> f64 {
        self.radius
    }
}

/// The job of the TrustRegionStepEvaluator is to evaluate the quality of a
/// step, i.e., how the cost of a step compares with the reduction in the
/// objective of the trust region problem.
///
/// Classic trust region methods are descent methods, in that they only
/// accept a point if it strictly reduces the value of the objective
/// function. They do this by measuring the quality of a step as
///
///   cost_change / model_cost_change.
///
/// Relaxing the monotonic descent requirement allows the algorithm to be
/// more efficient in the long term at the cost of some local increase in
/// the value of the objective function. This is implemented following
/// Algorithm 10.1.2 in "Trust Region Methods" by Conn, Gould & Toint.
pub(crate) struct TrustRegionStepEvaluator {
    max_consecutive_nonmonotonic_steps: i32,
    /// The minimum cost encountered up till now.
    minimum_cost: f64,
    /// The current cost of the trust region minimizer as informed by the
    /// last call to StepAccepted.
    current_cost: f64,
    reference_cost: f64,
    candidate_cost: f64,
    /// Accumulated model cost since the last time the reference model cost
    /// was updated, i.e., when a step with cost less than the current known
    /// minimum cost is accepted.
    accumulated_reference_model_cost_change: f64,
    /// Accumulated model cost since the last time the candidate model cost
    /// was updated, i.e., a non-monotonic step was taken with a cost that was
    /// greater than the current candidate cost.
    accumulated_candidate_model_cost_change: f64,
    /// Number of steps taken since the last time minimum_cost was updated.
    num_consecutive_nonmonotonic_steps: i32,
}

impl TrustRegionStepEvaluator {
    /// `initial_cost` is as the name implies the cost of the starting state
    /// of the trust region minimizer.
    ///
    /// `max_consecutive_nonmonotonic_steps` controls the window size used by
    /// the step selection algorithm to accept non-monotonic steps. Setting
    /// this parameter to zero, recovers the classic monotonic descent
    /// algorithm.
    pub(crate) fn new(initial_cost: f64, max_consecutive_nonmonotonic_steps: i32) -> Self {
        Self {
            max_consecutive_nonmonotonic_steps,
            minimum_cost: initial_cost,
            current_cost: initial_cost,
            reference_cost: initial_cost,
            candidate_cost: initial_cost,
            accumulated_reference_model_cost_change: 0.0,
            accumulated_candidate_model_cost_change: 0.0,
            num_consecutive_nonmonotonic_steps: 0,
        }
    }

    /// Return the quality of the step given its cost and the decrease in the
    /// cost of the model. model_cost_change has to be positive.
    pub(crate) fn step_quality(&self, cost: f64, model_cost_change: f64) -> f64 {
        // If the function evaluation for this step was a failure, in which
        // case the TrustRegionMinimizer would have set the cost to
        // std::numeric_limits<double>::max(). In this case, the division by
        // model_cost_change can result in an overflow. To prevent that from
        // happening, we will deal with this case explicitly.
        if cost >= f64::MAX {
            return f64::MIN;
        }

        let relative_decrease = (self.current_cost - cost) / model_cost_change;
        let historical_relative_decrease = (self.reference_cost - cost)
            / (self.accumulated_reference_model_cost_change + model_cost_change);
        relative_decrease.max(historical_relative_decrease)
    }

    /// Inform the step evaluator that a step with the given cost and
    /// model_cost_change has been accepted by the trust region minimizer.
    pub(crate) fn step_accepted(&mut self, cost: f64, model_cost_change: f64) {
        // Algorithm 10.1.2 from Trust Region Methods by Conn, Gould & Toint.
        //
        // Step 3a
        self.current_cost = cost;
        self.accumulated_candidate_model_cost_change += model_cost_change;
        self.accumulated_reference_model_cost_change += model_cost_change;

        // Step 3b.
        if self.current_cost < self.minimum_cost {
            self.minimum_cost = self.current_cost;
            self.num_consecutive_nonmonotonic_steps = 0;
            self.candidate_cost = self.current_cost;
            self.accumulated_candidate_model_cost_change = 0.0;
        } else {
            // Step 3c.
            self.num_consecutive_nonmonotonic_steps += 1;
            if self.current_cost > self.candidate_cost {
                self.candidate_cost = self.current_cost;
                self.accumulated_candidate_model_cost_change = 0.0;
            }
        }

        // Step 3d.
        //
        // At this point we have made too many non-monotonic steps and we are
        // going to reset the value of the reference iterate so as to force
        // the algorithm to descend.
        //
        // Note: In the original algorithm by Toint, this step was only
        // executed if the step was non-monotonic, but that would not handle
        // the case of max_consecutive_nonmonotonic_steps = 0. The small
        // modification of doing this always handles that corner case
        // correctly.
        if self.num_consecutive_nonmonotonic_steps == self.max_consecutive_nonmonotonic_steps {
            self.reference_cost = self.candidate_cost;
            self.accumulated_reference_model_cost_change =
                self.accumulated_candidate_model_cost_change;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ceres 2.0.0 has no dedicated test for the step evaluator; these check
    // Algorithm 10.1.2 (Conn, Gould & Toint) by hand.
    #[test]
    fn monotonic_step_quality() {
        let mut e = TrustRegionStepEvaluator::new(10.0, 0);
        assert_eq!(e.step_quality(9.0, 2.0), 0.5);
        e.step_accepted(9.0, 2.0);
        assert_eq!(e.step_quality(8.0, 1.0), 1.0);
        assert_eq!(e.step_quality(f64::MAX, 1.0), f64::MIN);
    }

    #[test]
    fn nonmonotonic_reference_cost() {
        let mut e = TrustRegionStepEvaluator::new(10.0, 2);
        e.step_accepted(5.0, 5.0);
        // A non-monotonic step: cost goes up but the reference is still 10.
        e.step_accepted(6.0, 1.0);
        // Relative to current cost 6, decrease 1 over model 1 = 1; relative
        // to reference 10 with accumulated model 6+1: (10-5)/7.
        assert_eq!(e.step_quality(5.0, 1.0), 1.0);
        assert!((e.step_quality(7.0, 1.0) - 3.0 / 7.0).abs() < 1e-15);
    }
}
