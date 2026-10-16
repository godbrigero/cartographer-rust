// Ported from Ceres Solver 2.0.0: internal/ceres/line_search.{h,cc}
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/line_search.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Line search: `LineSearchFunction`, the interpolation based step size
//! selection shared by all line searches, and the Armijo line search.
//!
//! Line search is another name for a one dimensional optimization
//! algorithm. The name "line search" comes from the fact one dimensional
//! optimization problems that arise as subproblems of general
//! multidimensional optimization problems.
//!
//! While finding the exact minimum of a one dimensional function is hard,
//! instances of LineSearch find a point that satisfies a sufficient
//! decrease condition. Depending on the particular condition used, we get a
//! variety of different line search algorithms, e.g., Armijo, Wolfe etc.
//!
//! RUST PORT NOTE: only what the trust region minimizer uses for bounds
//! constrained problems is ported (`TrustRegionMinimizer::DoLineSearch`
//! always creates an `ARMIJO` line search). The Wolfe line search and the
//! line search minimizer are omitted. `LineSearch::Create` +
//! `ArmijoLineSearch::DoSearch` become [`ArmijoLineSearch::search`].

use std::time::Instant;

use crate::polynomial::{FunctionSample, minimize_interpolating_polynomial};
use crate::program::Evaluator;
use crate::stringprintf::fmt_e;
use crate::types::LineSearchInterpolationType;

/// Line search options (C++ `LineSearch::Options`, without the `function`
/// pointer, which is passed to [`ArmijoLineSearch::search`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct LineSearchOptions {
    /// Degree of the polynomial used to approximate the objective function.
    pub interpolation_type: LineSearchInterpolationType,

    // Armijo and Wolfe line search parameters.
    /// Solving the line search problem exactly is computationally
    /// prohibitive. Fortunately, line search based optimization algorithms
    /// can still guarantee convergence if instead of an exact solution, the
    /// line search algorithm returns a solution which decreases the value of
    /// the objective function sufficiently. More precisely, we are looking
    /// for a step_size s.t.
    ///
    ///  f(step_size) <= f(0) + sufficient_decrease * f'(0) * step_size
    pub sufficient_decrease: f64,
    /// In each iteration of the Armijo / Wolfe line search,
    ///
    /// new_step_size >= max_step_contraction * step_size
    ///
    /// Note that by definition, for contraction:
    ///
    ///  0 < max_step_contraction < min_step_contraction < 1
    pub max_step_contraction: f64,
    /// In each iteration of the Armijo / Wolfe line search,
    ///
    /// new_step_size <= min_step_contraction * step_size
    pub min_step_contraction: f64,
    /// If during the line search, the step_size falls below this value, it
    /// is truncated to zero.
    pub min_step_size: f64,
    /// Maximum number of trial step size iterations during each line search,
    /// if a step size satisfying the search conditions cannot be found
    /// within this number of trials, the line search will terminate.
    pub max_num_iterations: i32,

    // Wolfe-specific line search parameters (unused by Armijo, kept so that
    // DoLineSearch copies all options like the C++ code).
    /// |f'(step_size)| <= sufficient_curvature_decrease * |f'(0)|
    #[allow(dead_code)]
    pub sufficient_curvature_decrease: f64,
    /// new_step_size <= max_step_expansion * step_size.
    #[allow(dead_code)]
    pub max_step_expansion: f64,

    /// Do not log warnings.
    pub is_silent: bool,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        Self {
            interpolation_type: LineSearchInterpolationType::Cubic,
            sufficient_decrease: 1e-4,
            max_step_contraction: 1e-3,
            min_step_contraction: 0.9,
            min_step_size: 1e-9,
            max_num_iterations: 20,
            sufficient_curvature_decrease: 0.9,
            max_step_expansion: 10.0,
            is_silent: false,
        }
    }
}

/// Result of the line search (C++ `LineSearch::Summary`).
#[derive(Clone, Debug, Default)]
pub(crate) struct LineSearchSummary {
    pub success: bool,
    pub optimal_point: FunctionSample,
    pub num_function_evaluations: i32,
    pub num_gradient_evaluations: i32,
    pub num_iterations: i32,
    /// Cumulative time spent evaluating the value of the cost function
    /// across all iterations.
    pub cost_evaluation_time_in_seconds: f64,
    /// Cumulative time spent evaluating the gradient of the cost function
    /// across all iterations.
    pub gradient_evaluation_time_in_seconds: f64,
    /// Cumulative time spent minimizing the interpolating polynomial to
    /// compute the next candidate step size across all iterations.
    pub polynomial_minimization_time_in_seconds: f64,
    pub total_time_in_seconds: f64,
    pub error: String,
}

/// An object used by the line search to access the function values and
/// gradient of the one dimensional function being optimized (C++
/// `LineSearchFunction`).
///
/// In practice, this object provides access to the objective function value
/// and the directional derivative of the underlying optimization problem
/// along a specific search direction.
pub(crate) struct LineSearchFunction<'e, 'p, 'a> {
    evaluator: &'e mut Evaluator<'p, 'a>,
    position: Vec<f64>,
    direction: Vec<f64>,
    // scaled_direction = x * direction_;
    scaled_direction: Vec<f64>,

    // We may not exclusively own the evaluator (e.g. in the Trust Region
    // minimizer), hence we need to save the initial evaluation durations for
    // the value & gradient to accurately determine the duration of the
    // evaluations we invoked. These are reset by a call to
    // reset_time_statistics().
    initial_evaluator_residual_time_in_seconds: f64,
    initial_evaluator_jacobian_time_in_seconds: f64,
}

impl<'e, 'p, 'a> LineSearchFunction<'e, 'p, 'a> {
    pub(crate) fn new(evaluator: &'e mut Evaluator<'p, 'a>) -> Self {
        let num_parameters = evaluator.program.num_parameters;
        let num_effective_parameters = evaluator.program.num_effective_parameters;
        Self {
            evaluator,
            position: vec![0.0; num_parameters],
            direction: vec![0.0; num_effective_parameters],
            scaled_direction: vec![0.0; num_effective_parameters],
            initial_evaluator_residual_time_in_seconds: 0.0,
            initial_evaluator_jacobian_time_in_seconds: 0.0,
        }
    }

    pub(crate) fn init(&mut self, position: &[f64], direction: &[f64]) {
        self.position = position.to_vec();
        self.direction = direction.to_vec();
        self.scaled_direction.resize(direction.len(), 0.0);
    }

    /// Evaluate the line search objective
    ///
    ///   f(x) = p(position + x * direction)
    ///
    /// Where, p is the objective function of the general optimization
    /// problem.
    ///
    /// evaluate_gradient controls whether the gradient will be evaluated or
    /// not.
    ///
    /// On return output.*_is_valid indicate indicate whether the
    /// corresponding fields have numerically valid values or not.
    pub(crate) fn evaluate(
        &mut self,
        x: f64,
        evaluate_gradient: bool,
        output: &mut FunctionSample,
    ) {
        output.x = x;
        output.vector_x_is_valid = false;
        output.value_is_valid = false;
        output.gradient_is_valid = false;
        output.vector_gradient_is_valid = false;

        for (s, d) in self.scaled_direction.iter_mut().zip(&self.direction) {
            *s = output.x * d;
        }
        output.vector_x.resize(self.position.len(), 0.0);
        if !self
            .evaluator
            .plus(&self.position, &self.scaled_direction, &mut output.vector_x)
        {
            return;
        }
        output.vector_x_is_valid = true;

        if evaluate_gradient {
            output.vector_gradient.resize(self.direction.len(), 0.0);
        }
        let gradient = evaluate_gradient.then_some(&mut output.vector_gradient[..]);
        let eval_status = self.evaluator.evaluate(
            &output.vector_x,
            true,
            &mut output.value,
            None,
            gradient,
            None,
        );

        if !eval_status || !output.value.is_finite() {
            return;
        }

        output.value_is_valid = true;
        if !evaluate_gradient {
            return;
        }

        output.gradient = self
            .direction
            .iter()
            .zip(&output.vector_gradient)
            .map(|(d, g)| d * g)
            .sum();
        if !output.gradient.is_finite() {
            return;
        }

        output.gradient_is_valid = true;
        output.vector_gradient_is_valid = true;
    }

    pub(crate) fn direction_infinity_norm(&self) -> f64 {
        self.direction.iter().fold(0.0, |m: f64, d| m.max(d.abs()))
    }

    /// Resets to now, the start point for the results from
    /// time_statistics().
    pub(crate) fn reset_time_statistics(&mut self) {
        let stats = self.evaluator.statistics;
        self.initial_evaluator_residual_time_in_seconds = stats.residual_time;
        self.initial_evaluator_jacobian_time_in_seconds = stats.jacobian_time;
    }

    /// Returns `(cost_evaluation_time_in_seconds,
    /// gradient_evaluation_time_in_seconds)` since the last reset.
    ///
    /// Strictly speaking this will slightly underestimate the time spent
    /// evaluating the gradient of the line search univariate cost function
    /// as it does not count the time spent performing the dot product with
    /// the direction vector. However, this will typically be small by
    /// comparison, and also allows direct subtraction of the timing
    /// information from the totals for the evaluator returned in the solver
    /// summary.
    pub(crate) fn time_statistics(&self) -> (f64, f64) {
        let stats = self.evaluator.statistics;
        (
            stats.residual_time - self.initial_evaluator_residual_time_in_seconds,
            stats.jacobian_time - self.initial_evaluator_jacobian_time_in_seconds,
        )
    }

    pub(crate) fn position(&self) -> &[f64] {
        &self.position
    }
}

/// Returns step_size \in [min_step_size, max_step_size] which minimizes the
/// polynomial of degree defined by interpolation_type which interpolates
/// all of the provided samples with valid values (C++
/// `LineSearch::InterpolatingPolynomialMinimizingStepSize`).
pub(crate) fn interpolating_polynomial_minimizing_step_size(
    interpolation_type: LineSearchInterpolationType,
    lowerbound: &FunctionSample,
    previous: &FunctionSample,
    current: &FunctionSample,
    min_step_size: f64,
    max_step_size: f64,
) -> f64 {
    if !current.value_is_valid
        || (interpolation_type == LineSearchInterpolationType::Bisection
            && max_step_size <= current.x)
    {
        // Either: sample is invalid; or we are using BISECTION and
        // contracting the step size.
        return (current.x * 0.5).max(min_step_size).min(max_step_size);
    } else if interpolation_type == LineSearchInterpolationType::Bisection {
        assert!(max_step_size > current.x);
        // We are expanding the search (during a Wolfe bracketing phase) using
        // BISECTION interpolation. Using BISECTION when trying to expand is
        // strictly speaking an oxymoron, but we define this to mean always
        // taking the maximum step size so that the Armijo & Wolfe
        // implementations are agnostic to the interpolation type.
        return max_step_size;
    }
    // Only check if lower-bound is valid here, where it is required to avoid
    // replicating current.value_is_valid == false behaviour in
    // WolfeLineSearch.
    assert!(
        lowerbound.value_is_valid,
        "Ceres bug: lower-bound sample for interpolation is invalid, please contact the \
         developers!, interpolation_type: {interpolation_type}, lowerbound: {}, previous: {}, \
         current: {}",
        lowerbound.to_debug_string(),
        previous.to_debug_string(),
        current.to_debug_string()
    );

    // Select step size by interpolating the function and gradient values and
    // minimizing the corresponding polynomial.
    let mut samples = vec![lowerbound.clone()];

    if interpolation_type == LineSearchInterpolationType::Quadratic {
        // Two point interpolation using function values and the gradient at
        // the lower bound.
        samples.push(FunctionSample::with_value(current.x, current.value));

        if previous.value_is_valid {
            // Three point interpolation, using function values and the
            // gradient at the lower bound.
            samples.push(FunctionSample::with_value(previous.x, previous.value));
        }
    } else {
        // CUBIC: Two point interpolation using the function values and the
        // gradients.
        samples.push(current.clone());

        if previous.value_is_valid {
            // Three point interpolation using the function values and the
            // gradients.
            samples.push(previous.clone());
        }
    }

    let (step_size, _unused_min_value) =
        minimize_interpolating_polynomial(&samples, min_step_size, max_step_size);
    step_size
}

/// Backtracking and interpolation based Armijo line search (C++
/// `ArmijoLineSearch`). This implementation is based on the Armijo line
/// search that ships in the minFunc package by Mark Schmidt.
///
/// For more details: http://www.di.ens.fr/~mschmidt/Software/minFunc.html
pub(crate) struct ArmijoLineSearch {
    options: LineSearchOptions,
}

impl ArmijoLineSearch {
    pub(crate) fn new(options: LineSearchOptions) -> Self {
        Self { options }
    }

    /// Perform the line search (C++ `LineSearch::Search`).
    ///
    /// step_size_estimate must be a positive number.
    ///
    /// initial_cost and initial_gradient are the values and gradient of the
    /// function at zero. The returned summary contains the result of the
    /// line search; `success` is true if a non-zero step size is found.
    pub(crate) fn search(
        &self,
        function: &mut LineSearchFunction<'_, '_, '_>,
        step_size_estimate: f64,
        initial_cost: f64,
        initial_gradient: f64,
    ) -> LineSearchSummary {
        let start_time = Instant::now();
        let mut summary = LineSearchSummary::default();
        function.reset_time_statistics();
        self.do_search(
            function,
            step_size_estimate,
            initial_cost,
            initial_gradient,
            &mut summary,
        );
        let (cost_time, gradient_time) = function.time_statistics();
        summary.cost_evaluation_time_in_seconds = cost_time;
        summary.gradient_evaluation_time_in_seconds = gradient_time;
        summary.total_time_in_seconds = start_time.elapsed().as_secs_f64();
        summary
    }

    /// C++ `ArmijoLineSearch::DoSearch`.
    fn do_search(
        &self,
        function: &mut LineSearchFunction<'_, '_, '_>,
        step_size_estimate: f64,
        initial_cost: f64,
        initial_gradient: f64,
        summary: &mut LineSearchSummary,
    ) {
        let options = &self.options;
        assert!(step_size_estimate >= 0.0);
        assert!(options.sufficient_decrease > 0.0);
        assert!(options.sufficient_decrease < 1.0);
        assert!(options.max_num_iterations > 0);

        // Note initial_cost & initial_gradient are evaluated at step_size =
        // 0, not step_size_estimate, which is our starting guess.
        let mut initial_position =
            FunctionSample::with_value_and_gradient(0.0, initial_cost, initial_gradient);
        initial_position.vector_x = function.position().to_vec();
        initial_position.vector_x_is_valid = true;

        let descent_direction_max_norm = function.direction_infinity_norm();
        let mut previous = FunctionSample::default();
        let mut current = FunctionSample::default();

        // As the Armijo line search algorithm always uses the initial point,
        // for which both the function value and derivative are known, when
        // fitting a minimizing polynomial, we can fit up to a quadratic
        // without requiring the gradient at the current query point.
        let evaluate_gradient = options.interpolation_type == LineSearchInterpolationType::Cubic;

        summary.num_function_evaluations += 1;
        if evaluate_gradient {
            summary.num_gradient_evaluations += 1;
        }

        function.evaluate(step_size_estimate, evaluate_gradient, &mut current);
        while !current.value_is_valid
            || current.value
                > (initial_cost + options.sufficient_decrease * initial_gradient * current.x)
        {
            // If current.value_is_valid is false, we treat it as if the cost
            // at that point is not large enough to satisfy the sufficient
            // decrease condition.
            summary.num_iterations += 1;
            if summary.num_iterations >= options.max_num_iterations {
                summary.error = format!(
                    "Line search failed: Armijo failed to find a point satisfying the \
                     sufficient decrease condition within specified max_num_iterations: {}.",
                    options.max_num_iterations
                );
                if !options.is_silent {
                    log::warn!("{}", summary.error);
                }
                return;
            }

            let polynomial_minimization_start_time = Instant::now();
            let step_size = interpolating_polynomial_minimizing_step_size(
                options.interpolation_type,
                &initial_position,
                &previous,
                &current,
                options.max_step_contraction * current.x,
                options.min_step_contraction * current.x,
            );
            summary.polynomial_minimization_time_in_seconds +=
                polynomial_minimization_start_time.elapsed().as_secs_f64();

            if step_size * descent_direction_max_norm < options.min_step_size {
                summary.error = format!(
                    "Line search failed: step_size too small: {} with \
                     descent_direction_max_norm: {}.",
                    fmt_e(step_size, 0, 5, false),
                    fmt_e(descent_direction_max_norm, 0, 5, false)
                );
                if !options.is_silent {
                    log::warn!("{}", summary.error);
                }
                return;
            }

            previous = current.clone();

            summary.num_function_evaluations += 1;
            if evaluate_gradient {
                summary.num_gradient_evaluations += 1;
            }

            function.evaluate(step_size, evaluate_gradient, &mut current);
        }

        summary.optimal_point = current;
        summary.success = true;
    }
}

#[cfg(test)]
mod tests {
    //! Ceres 2.0.0 has no line_search_test.cc; these exercise the Armijo
    //! search on small problems through a real `Evaluator`.
    use super::*;
    use crate::autodiff_cost_function::{AutoDiffCostFunction, CostFunctor};
    use crate::problem::Problem;
    use crate::program::Program;
    use crate::scalar::Scalar;

    /// r(x) = x - 1 (cost 0.5 (x - 1)^2).
    struct Linear;
    impl CostFunctor for Linear {
        fn evaluate<T: Scalar>(&self, p: &[&[T]], r: &mut [T]) -> bool {
            r[0] = p[0][0] - T::constant(1.0);
            true
        }
    }

    /// r(x) = exp(x) - 1: strongly non-quadratic along +x.
    struct Exp;
    impl CostFunctor for Exp {
        fn evaluate<T: Scalar>(&self, p: &[&[T]], r: &mut [T]) -> bool {
            r[0] = p[0][0].exp() - T::constant(1.0);
            true
        }
    }

    fn problem_with<F: CostFunctor + 'static>(f: F, x0: f64) -> Problem<'static> {
        let mut problem = Problem::new();
        let x = problem.add_parameter_block(&[x0]);
        problem.add_residual_block(
            Box::new(AutoDiffCostFunction::<_, 1>::new(f, 1, &[1])),
            None,
            &[x],
        );
        problem
    }

    fn run(
        problem: &Problem<'_>,
        options: LineSearchOptions,
        x0: f64,
        direction: f64,
        step_size_estimate: f64,
    ) -> LineSearchSummary {
        let (program, _) = Program::create_reduced_program(problem).unwrap();
        let mut evaluator = Evaluator::new(problem, program, 1);
        let mut cost = 0.0;
        let mut gradient = [0.0];
        assert!(evaluator.evaluate(&[x0], true, &mut cost, None, Some(&mut gradient), None));
        let mut function = LineSearchFunction::new(&mut evaluator);
        function.init(&[x0], &[direction]);
        ArmijoLineSearch::new(options).search(
            &mut function,
            step_size_estimate,
            cost,
            gradient[0] * direction,
        )
    }

    #[test]
    fn full_step_is_accepted_when_sufficient() {
        // From x = 3 the Newton direction -2 reaches the minimum exactly.
        let problem = problem_with(Linear, 3.0);
        let summary = run(&problem, LineSearchOptions::default(), 3.0, -2.0, 1.0);
        assert!(summary.success);
        assert_eq!(summary.optimal_point.x, 1.0);
        assert_eq!(summary.num_iterations, 0);
        assert_eq!(summary.num_function_evaluations, 1);
        assert_eq!(summary.num_gradient_evaluations, 1);
        assert_eq!(summary.optimal_point.vector_x, vec![1.0]);
    }

    #[test]
    fn overshooting_step_is_contracted_by_interpolation() {
        // A step 10 times too long: f(10) = 0.5 * 18^2 is far above the
        // Armijo line. Cubic interpolation of the exact quadratic recovers
        // the minimizer 0.1 (clamped into [1e-3, 0.6] * 10 => [0.01, 6]).
        let problem = problem_with(Linear, 3.0);
        let options = LineSearchOptions {
            min_step_contraction: 0.6,
            ..LineSearchOptions::default()
        };
        for interpolation_type in [
            LineSearchInterpolationType::Cubic,
            LineSearchInterpolationType::Quadratic,
        ] {
            let summary = run(
                &problem,
                LineSearchOptions {
                    interpolation_type,
                    ..options
                },
                3.0,
                -20.0,
                1.0,
            );
            assert!(summary.success, "{interpolation_type}");
            assert!((summary.optimal_point.x - 0.1).abs() < 1e-12);
            assert_eq!(summary.num_iterations, 1);
            assert_eq!(
                summary.num_gradient_evaluations,
                if interpolation_type == LineSearchInterpolationType::Cubic {
                    2
                } else {
                    0
                }
            );
        }
    }

    #[test]
    fn bisection_halves_the_step() {
        let problem = problem_with(Linear, 3.0);
        let options = LineSearchOptions {
            interpolation_type: LineSearchInterpolationType::Bisection,
            max_step_contraction: 0.1,
            min_step_contraction: 0.6,
            ..LineSearchOptions::default()
        };
        // f(s) = 0.5 (2 - 8 s)^2: s = 1 fails, s = 0.5 fails (f = 2 > 2 -
        // eps), s = 0.25 is the minimizer.
        let summary = run(&problem, options, 3.0, -8.0, 1.0);
        assert!(summary.success);
        assert_eq!(summary.optimal_point.x, 0.25);
        assert_eq!(summary.num_iterations, 2);
    }

    #[test]
    fn non_quadratic_function_satisfies_armijo_condition() {
        // From x = -3 along +10, the full step lands at x = 7 where the cost
        // is ~6e5: the step must be contracted (several times) until the
        // Armijo condition holds.
        let problem = problem_with(Exp, -3.0);
        let options = LineSearchOptions::default();
        let summary = run(&problem, options, -3.0, 10.0, 1.0);
        assert!(summary.success);
        let x = summary.optimal_point.x;
        let e = (-3f64).exp();
        let f0 = 0.5 * (e - 1.0).powi(2);
        let g0 = (e - 1.0) * e * 10.0;
        assert!(summary.optimal_point.value <= f0 + options.sufficient_decrease * g0 * x);
        assert!(x > 0.0 && x < 1.0);
        assert!(summary.num_iterations >= 1);
        assert_eq!(summary.num_function_evaluations, summary.num_iterations + 1);
    }

    #[test]
    fn fails_after_max_num_iterations() {
        let problem = problem_with(Linear, 3.0);
        let options = LineSearchOptions {
            max_num_iterations: 1,
            is_silent: true,
            ..LineSearchOptions::default()
        };
        let summary = run(&problem, options, 3.0, -20.0, 1.0);
        assert!(!summary.success);
        assert_eq!(
            summary.error,
            "Line search failed: Armijo failed to find a point satisfying the sufficient \
             decrease condition within specified max_num_iterations: 1."
        );
    }

    #[test]
    fn fails_when_step_size_too_small() {
        // An ascent direction never satisfies the sufficient decrease
        // condition; the step shrinks until it is below min_step_size.
        let problem = problem_with(Linear, 3.0);
        let options = LineSearchOptions {
            is_silent: true,
            min_step_size: 1e-3,
            ..LineSearchOptions::default()
        };
        let summary = run(&problem, options, 3.0, 1.0, 1.0);
        assert!(!summary.success);
        assert!(
            summary
                .error
                .starts_with("Line search failed: step_size too small: "),
            "{}",
            summary.error
        );
    }

    #[test]
    fn invalid_sample_is_bisected() {
        let lower = FunctionSample::with_value_and_gradient(0.0, 1.0, -1.0);
        let invalid = FunctionSample {
            x: 1.0,
            ..FunctionSample::default()
        };
        let step = interpolating_polynomial_minimizing_step_size(
            LineSearchInterpolationType::Cubic,
            &lower,
            &FunctionSample::default(),
            &invalid,
            0.001,
            0.9,
        );
        assert_eq!(step, 0.5);
    }
}
