// Ported from Ceres Solver 2.0.0: include/ceres/solver.h,
// include/ceres/iteration_callback.h, internal/ceres/solver.cc,
// internal/ceres/solver_utils.{h,cc}, internal/ceres/preprocessor.cc and
// internal/ceres/trust_region_preprocessor.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/solver.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sameeragarwal@google.com (Sameer Agarwal),
// keir@google.com (Keir Mierle)

//! [`solve`], [`SolverOptions`], [`SolverSummary`] and iteration callbacks.
//!
//! RUST PORT NOTE: `Solver::Options` / `Solver::Summary` become
//! [`SolverOptions`] / [`SolverSummary`] (Rust has no nested types). Only
//! the trust region minimizer with Levenberg-Marquardt and the
//! `DENSE_QR`, `DENSE_NORMAL_CHOLESKY` and `SPARSE_NORMAL_CHOLESKY` linear
//! solvers are ported; options for the other algorithms (line search
//! minimizer, dogleg, Schur/iterative solvers, inner iterations, orderings,
//! gradient checking, problem dumping) are absent. The line search step
//! size options are present: they configure the projected Armijo line
//! search used for bounds constrained problems.

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::linear_solver::LinearSolver;
use crate::problem::Problem;
use crate::program::{Evaluator, Program, is_feasible};
use crate::stringprintf::{fmt_d, fmt_e, fmt_f};
use crate::trust_region::{LevenbergMarquardtStrategy, TrustRegionStrategyOptions};
use crate::trust_region_minimizer::TrustRegionMinimizer;
use crate::types::{
    CallbackReturnType, DenseLinearAlgebraLibraryType, LineSearchInterpolationType,
    LinearSolverType, LoggingType, MinimizerType, SparseLinearAlgebraLibraryType, TerminationType,
    TrustRegionStrategyType,
};

/// This struct describes the state of the optimizer after each iteration of
/// the minimization (C++ `IterationSummary`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IterationSummary {
    /// Current iteration number.
    pub iteration: i32,
    /// Step was numerically valid, i.e., all values are finite and the step
    /// reduces the value of the linearized model.
    ///
    /// Note: step_is_valid is always true when iteration = 0.
    pub step_is_valid: bool,
    /// Step did not reduce the value of the objective function sufficiently,
    /// but it was accepted because of the relaxed acceptance criterion used
    /// by the non-monotonic trust region algorithm.
    ///
    /// Note: step_is_nonmonotonic is always false when iteration = 0.
    pub step_is_nonmonotonic: bool,
    /// Whether or not the minimizer accepted this step or not. If the
    /// ordinary trust region algorithm is used, this means that the relative
    /// reduction in the objective function value was greater than
    /// Solver::Options::min_relative_decrease. However, if the non-monotonic
    /// trust region algorithm is used (Solver::Options:use_nonmonotonic_steps
    /// = true), then even if the relative decrease is not sufficient, the
    /// algorithm may accept the step and the step is declared successful.
    ///
    /// Note: step_is_successful is always true when iteration = 0.
    pub step_is_successful: bool,
    /// Value of the objective function.
    pub cost: f64,
    /// Change in the value of the objective function in this iteration. This
    /// can be positive or negative.
    pub cost_change: f64,
    /// Infinity norm of the gradient vector.
    pub gradient_max_norm: f64,
    /// 2-norm of the gradient vector.
    pub gradient_norm: f64,
    /// 2-norm of the size of the step computed by the optimization
    /// algorithm.
    pub step_norm: f64,
    /// For trust region algorithms, the ratio of the actual change in cost
    /// and the change in the cost of the linearized approximation.
    pub relative_decrease: f64,
    /// Size of the trust region at the end of the current iteration. For the
    /// Levenberg-Marquardt algorithm, the regularization parameter mu =
    /// 1.0 / trust_region_radius.
    pub trust_region_radius: f64,
    /// For the inexact step Levenberg-Marquardt algorithm, this is the
    /// relative accuracy with which the Newton(LM) step is solved. This
    /// number affects only the iterative solvers capable of solving linear
    /// systems inexactly. Factorization-based exact solvers ignore it.
    pub eta: f64,
    /// Step sized computed by the line search algorithm (unused here).
    pub step_size: f64,
    /// Number of function value evaluations used by the line search (unused).
    pub line_search_function_evaluations: i32,
    /// Number of function gradient evaluations used by the line search
    /// (unused).
    pub line_search_gradient_evaluations: i32,
    /// Number of iterations taken by the line search (unused).
    pub line_search_iterations: i32,
    /// Number of iterations taken by the linear solver to solve for the
    /// Newton step.
    pub linear_solver_iterations: i32,
    /// Time (in seconds) spent inside the minimizer loop in the current
    /// iteration.
    pub iteration_time_in_seconds: f64,
    /// Time (in seconds) spent inside the trust region step solver.
    pub step_solver_time_in_seconds: f64,
    /// Time (in seconds) since the user called Solve().
    pub cumulative_time_in_seconds: f64,
}

/// Interface for specifying callbacks that are executed at the end of each
/// iteration of the Minimizer. The solver uses the return value of `call`
/// to decide whether to continue solving or to terminate.
///
/// RUST PORT NOTE: callbacks are shared (`Arc`) so that [`SolverOptions`]
/// stays `Clone`; use interior mutability to record state.
pub trait IterationCallback: Send + Sync {
    /// Called at the end of each iteration.
    fn call(&self, summary: &IterationSummary) -> CallbackReturnType;
}

impl<F: Fn(&IterationSummary) -> CallbackReturnType + Send + Sync> IterationCallback for F {
    fn call(&self, summary: &IterationSummary) -> CallbackReturnType {
        self(summary)
    }
}

/// Solver options (C++ `Solver::Options`). The defaults are the Ceres 2.0.0
/// defaults.
#[derive(Clone)]
pub struct SolverOptions {
    /// Minimizer type (only `TRUST_REGION`).
    pub minimizer_type: MinimizerType,
    /// Trust region strategy (only `LEVENBERG_MARQUARDT`).
    pub trust_region_strategy_type: TrustRegionStrategyType,
    /// The classical trust region methods are descent methods, in that they
    /// only accept a point if it strictly reduces the value of the objective
    /// function. Relaxing this requirement allows the algorithm to be more
    /// efficient in the long term at the cost of some local increase in the
    /// value of the objective function.
    pub use_nonmonotonic_steps: bool,
    /// Window size used by the non-monotonic step acceptance.
    pub max_consecutive_nonmonotonic_steps: i32,
    /// Maximum number of iterations for the minimizer to run for.
    pub max_num_iterations: i32,
    /// Maximum time for which the minimizer should run for.
    pub max_solver_time_in_seconds: f64,
    /// Number of threads used by Ceres for evaluating the cost and jacobians.
    pub num_threads: i32,
    /// Trust region minimizer settings.
    pub initial_trust_region_radius: f64,
    /// Maximum trust region radius.
    pub max_trust_region_radius: f64,
    /// Minimizer terminates when the trust region radius becomes smaller
    /// than this value.
    pub min_trust_region_radius: f64,
    /// Lower bound for the relative decrease before a step is accepted.
    pub min_relative_decrease: f64,
    /// For the Levenberg-Marquadt algorithm, the scaled diagonal of the
    /// normal equations J'J is used to control the size of the trust region.
    /// Extremely small and large values along the diagonal can make this
    /// regularization scheme fail. max_lm_diagonal and min_lm_diagonal,
    /// clamp the values of diag(J'J) from above and below.
    pub min_lm_diagonal: f64,
    /// See `min_lm_diagonal`.
    pub max_lm_diagonal: f64,
    /// Sometimes due to numerical conditioning problems or linear solver
    /// flakiness, the trust region strategy may return a numerically invalid
    /// step that can be fixed by reducing the trust region size. So the
    /// TrustRegionMinimizer allows for a few successive invalid steps before
    /// it declares NUMERICAL_FAILURE.
    pub max_num_consecutive_invalid_steps: i32,
    /// Degree of the polynomial used to approximate the objective function
    /// in the line search. Valid values are BISECTION, QUADRATIC and CUBIC.
    ///
    /// RUST PORT NOTE: the line search options are used by the projected
    /// Armijo line search the trust region minimizer runs on bounds
    /// constrained problems (the line search minimizer is not ported).
    pub line_search_interpolation_type: LineSearchInterpolationType,
    /// If during the line search, the step_size falls below this value, it
    /// is truncated to zero.
    pub min_line_search_step_size: f64,
    /// Line search parameters.
    ///
    /// Solving the line search problem exactly is computationally
    /// prohibitive. Fortunately, line search based optimization algorithms
    /// can still guarantee convergence if instead of an exact solution, the
    /// line search algorithm returns a solution which decreases the value of
    /// the objective function sufficiently. More precisely, we are looking
    /// for a step_size s.t.
    ///
    ///   f(step_size) <= f(0) + sufficient_decrease * f'(0) * step_size
    pub line_search_sufficient_function_decrease: f64,
    /// In each iteration of the line search,
    ///
    ///  new_step_size >= max_line_search_step_contraction * step_size
    ///
    /// Note that by definition, for contraction:
    ///
    ///  0 < max_step_contraction < min_step_contraction < 1
    pub max_line_search_step_contraction: f64,
    /// In each iteration of the line search,
    ///
    ///  new_step_size <= min_line_search_step_contraction * step_size
    pub min_line_search_step_contraction: f64,
    /// Maximum number of trial step size iterations during each line
    /// search, if a step size satisfying the search conditions cannot be
    /// found within this number of trials, the line search will terminate.
    /// For the trust region minimizer, 0 disables the projected line search
    /// on bounds constrained problems.
    pub max_num_line_search_step_size_iterations: i32,
    /// The strong Wolfe conditions consist of the Armijo sufficient decrease
    /// condition, and an additional requirement that the step-size be chosen
    /// s.t. the _magnitude_ ('strong' Wolfe conditions) of the gradient
    /// along the search direction decreases sufficiently (validated only).
    pub line_search_sufficient_curvature_decrease: f64,
    /// During the bracketing phase of the Wolfe search, the step size is
    /// increased until either a point satisfying the Wolfe conditions is
    /// found, or an upper bound for a bracket containing a point satisfying
    /// the conditions is found (validated only).
    pub max_line_search_step_expansion: f64,
    /// Minimizer terminates when
    ///
    ///   (new_cost - old_cost) < function_tolerance * old_cost;
    pub function_tolerance: f64,
    /// Minimizer terminates when
    ///
    ///   max_i |x - Project(Plus(x, -g(x))| < gradient_tolerance
    ///
    /// This value should typically be 1e-4 * function_tolerance.
    pub gradient_tolerance: f64,
    /// Minimizer terminates when
    ///
    ///   |step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)
    pub parameter_tolerance: f64,
    /// Linear least squares solver options. Default `SPARSE_NORMAL_CHOLESKY`
    /// (sparse support is always available in this port).
    pub linear_solver_type: LinearSolverType,
    /// Dense linear algebra library (only `EIGEN`, i.e. nalgebra).
    pub dense_linear_algebra_library_type: DenseLinearAlgebraLibraryType,
    /// Sparse linear algebra library (only `FAER`).
    pub sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType,
    /// Minimum number of iterations for which the linear solver should run,
    /// even if the convergence criterion is satisfied (validated only).
    pub min_linear_solver_iterations: i32,
    /// Maximum number of iterations for which the linear solver should run
    /// (validated only).
    pub max_linear_solver_iterations: i32,
    /// Forcing sequence parameter for inexact solves (unused by the exact
    /// solvers ported here, but validated and reported).
    pub eta: f64,
    /// Normalize the jacobian using Jacobi scaling before calling the linear
    /// least squares solver.
    pub jacobi_scaling: bool,
    /// Logging options: per-iteration progress goes to `log::debug!`, or to
    /// stdout if `minimizer_progress_to_stdout`.
    pub logging_type: LoggingType,
    /// By default the Minimizer progress is logged to the log crate at debug
    /// level. Setting this to true prints it to stdout.
    pub minimizer_progress_to_stdout: bool,
    /// Callbacks that are executed at the end of each iteration of the
    /// Minimizer. They are executed in the order that they are specified in
    /// this vector.
    pub callbacks: Vec<Arc<dyn IterationCallback>>,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            minimizer_type: MinimizerType::TrustRegion,
            trust_region_strategy_type: TrustRegionStrategyType::LevenbergMarquardt,
            use_nonmonotonic_steps: false,
            max_consecutive_nonmonotonic_steps: 5,
            max_num_iterations: 50,
            max_solver_time_in_seconds: 1e9,
            num_threads: 1,
            initial_trust_region_radius: 1e4,
            max_trust_region_radius: 1e16,
            min_trust_region_radius: 1e-32,
            min_relative_decrease: 1e-3,
            min_lm_diagonal: 1e-6,
            max_lm_diagonal: 1e32,
            max_num_consecutive_invalid_steps: 5,
            line_search_interpolation_type: LineSearchInterpolationType::Cubic,
            min_line_search_step_size: 1e-9,
            line_search_sufficient_function_decrease: 1e-4,
            max_line_search_step_contraction: 1e-3,
            min_line_search_step_contraction: 0.6,
            max_num_line_search_step_size_iterations: 20,
            line_search_sufficient_curvature_decrease: 0.9,
            max_line_search_step_expansion: 10.0,
            function_tolerance: 1e-6,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            linear_solver_type: LinearSolverType::SparseNormalCholesky,
            dense_linear_algebra_library_type: DenseLinearAlgebraLibraryType::Eigen,
            sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType::Faer,
            min_linear_solver_iterations: 0,
            max_linear_solver_iterations: 500,
            eta: 1e-1,
            jacobi_scaling: true,
            logging_type: LoggingType::PerMinimizerIteration,
            minimizer_progress_to_stdout: false,
            callbacks: Vec::new(),
        }
    }
}

impl fmt::Debug for SolverOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SolverOptions")
            .field("use_nonmonotonic_steps", &self.use_nonmonotonic_steps)
            .field("max_num_iterations", &self.max_num_iterations)
            .field("num_threads", &self.num_threads)
            .field("linear_solver_type", &self.linear_solver_type)
            .field("function_tolerance", &self.function_tolerance)
            .field("gradient_tolerance", &self.gradient_tolerance)
            .field("parameter_tolerance", &self.parameter_tolerance)
            .field("num_callbacks", &self.callbacks.len())
            .finish_non_exhaustive()
    }
}

macro_rules! option_op {
    ($opts:ident, $field:ident, $op:tt, $value:expr) => {
        if !($opts.$field $op $value) {
            return Err(format!(
                "Invalid configuration. Solver::Options::{} = {}. Violated constraint: Solver::Options::{} {} {}",
                stringify!($field), $opts.$field, stringify!($field), stringify!($op), stringify!($value)
            ));
        }
    };
}

macro_rules! option_op_option {
    ($opts:ident, $x:ident, $op:tt, $y:ident) => {
        if !($opts.$x $op $opts.$y) {
            return Err(format!(
                "Invalid configuration. Solver::Options::{} = {}. Solver::Options::{} = {}. Violated constraint: Solver::Options::{} {} Solver::Options::{}.",
                stringify!($x), $opts.$x, stringify!($y), $opts.$y, stringify!($x), stringify!($op), stringify!($y)
            ));
        }
    };
}

impl SolverOptions {
    /// Returns `Ok(())` if the options struct has a valid configuration,
    /// otherwise the error message (C++ `IsValid`).
    // Negated comparisons are deliberate: NaN options must be rejected.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn is_valid(&self) -> Result<(), String> {
        let o = self;
        // CommonOptionsAreValid.
        option_op!(o, max_num_iterations, >=, 0);
        option_op!(o, max_solver_time_in_seconds, >=, 0.0);
        option_op!(o, function_tolerance, >=, 0.0);
        option_op!(o, gradient_tolerance, >=, 0.0);
        option_op!(o, parameter_tolerance, >=, 0.0);
        option_op!(o, num_threads, >, 0);

        // TrustRegionOptionsAreValid.
        option_op!(o, initial_trust_region_radius, >, 0.0);
        option_op!(o, min_trust_region_radius, >, 0.0);
        option_op!(o, max_trust_region_radius, >, 0.0);
        option_op_option!(o, min_trust_region_radius, <=, max_trust_region_radius);
        option_op_option!(o, min_trust_region_radius, <=, initial_trust_region_radius);
        option_op_option!(o, initial_trust_region_radius, <=, max_trust_region_radius);
        option_op!(o, min_relative_decrease, >=, 0.0);
        option_op!(o, min_lm_diagonal, >=, 0.0);
        option_op!(o, max_lm_diagonal, >=, 0.0);
        option_op_option!(o, min_lm_diagonal, <=, max_lm_diagonal);
        option_op!(o, max_num_consecutive_invalid_steps, >=, 0);
        option_op!(o, eta, >, 0.0);
        option_op!(o, min_linear_solver_iterations, >=, 0);
        option_op!(o, max_linear_solver_iterations, >=, 1);
        option_op_option!(o, min_linear_solver_iterations, <=, max_linear_solver_iterations);
        if o.use_nonmonotonic_steps {
            option_op!(o, max_consecutive_nonmonotonic_steps, >, 0);
        }

        if o.dense_linear_algebra_library_type == DenseLinearAlgebraLibraryType::Lapack
            && matches!(
                o.linear_solver_type,
                LinearSolverType::DenseNormalCholesky | LinearSolverType::DenseQr
            )
        {
            return Err(format!(
                "Can't use {} with Solver::Options::dense_linear_algebra_library_type = LAPACK \
                 because LAPACK was not enabled when Ceres was built.",
                o.linear_solver_type
            ));
        }

        if o.linear_solver_type == LinearSolverType::SparseNormalCholesky {
            match o.sparse_linear_algebra_library_type {
                SparseLinearAlgebraLibraryType::Faer => {}
                SparseLinearAlgebraLibraryType::NoSparse => {
                    return Err(format!(
                        "Can't use {} with Solver::Options::sparse_linear_algebra_library_type = {}.",
                        o.linear_solver_type, o.sparse_linear_algebra_library_type
                    ));
                }
                other => {
                    return Err(format!(
                        "Can't use {} with Solver::Options::sparse_linear_algebra_library_type = {}, \
                         because support was not enabled when Ceres Solver was built.",
                        o.linear_solver_type, other
                    ));
                }
            }
        }

        // LineSearchOptionsAreValid. We do not know if the problem is bounds
        // constrained or not, if it is then the trust region solver will
        // also use the line search solver to do a projection onto the box
        // constraints, so make sure that the line search options are checked
        // independent of what minimizer algorithm is being used.
        //
        // RUST PORT NOTE: `max_lbfgs_rank` and the (L)BFGS/WOLFE check do not
        // apply (no line search minimizer).
        option_op!(o, min_line_search_step_size, >, 0.0);
        option_op!(o, max_line_search_step_contraction, >, 0.0);
        option_op!(o, max_line_search_step_contraction, <, 1.0);
        option_op_option!(
            o,
            max_line_search_step_contraction,
            <,
            min_line_search_step_contraction
        );
        option_op!(o, min_line_search_step_contraction, <=, 1.0);
        // minimizer_type is always TRUST_REGION here.
        option_op!(o, max_num_line_search_step_size_iterations, >=, 0);
        option_op!(o, line_search_sufficient_function_decrease, >, 0.0);
        option_op_option!(
            o,
            line_search_sufficient_function_decrease,
            <,
            line_search_sufficient_curvature_decrease
        );
        option_op!(o, line_search_sufficient_curvature_decrease, <, 1.0);
        option_op!(o, max_line_search_step_expansion, >, 1.0);
        // Warn user if they have requested BISECTION interpolation, but
        // constraints on max/min step size change during line search prevent
        // bisection scaling from occurring. Warn only, as this is likely a
        // user mistake, but one which does not prevent us from continuing.
        if o.line_search_interpolation_type == LineSearchInterpolationType::Bisection
            && (o.max_line_search_step_contraction > 0.5
                || o.min_line_search_step_contraction < 0.5)
        {
            log::warn!(
                "Line search interpolation type is BISECTION, but specified \
                 max_line_search_step_contraction: {}, and min_line_search_step_contraction: {}, \
                 prevent bisection (0.5) scaling, continuing with solve regardless.",
                o.max_line_search_step_contraction,
                o.min_line_search_step_contraction
            );
        }
        Ok(())
    }
}

/// Summary of a solve (C++ `Solver::Summary`).
#[derive(Clone, Debug)]
pub struct SolverSummary {
    /// Minimizer summary.
    pub minimizer_type: MinimizerType,
    /// Why the solver terminated.
    pub termination_type: TerminationType,
    /// Reason why the solver terminated.
    pub message: String,
    /// Cost of the problem (value of the objective function) before the
    /// optimization.
    pub initial_cost: f64,
    /// Cost of the problem (value of the objective function) after the
    /// optimization.
    pub final_cost: f64,
    /// The part of the total cost that comes from residual blocks that were
    /// held fixed by the preprocessor because all the parameter blocks that
    /// they depend on were fixed.
    pub fixed_cost: f64,
    /// IterationSummary for each minimizer iteration in order.
    pub iterations: Vec<IterationSummary>,
    /// Number of minimizer iterations in which the step was accepted. Unless
    /// use_nonmonotonic_steps is true this is also the number of steps in
    /// which the objective function value/cost went down.
    pub num_successful_steps: i32,
    /// Number of minimizer iterations in which the step was rejected either
    /// because it did not reduce the cost enough or the step was not
    /// numerically valid.
    pub num_unsuccessful_steps: i32,
    /// Time (in seconds) spent in the preprocessor.
    pub preprocessor_time_in_seconds: f64,
    /// Time (in seconds) spent in the Minimizer.
    pub minimizer_time_in_seconds: f64,
    /// Time (in seconds) spent in the post processor.
    pub postprocessor_time_in_seconds: f64,
    /// Time (in seconds) spent in the solver.
    pub total_time_in_seconds: f64,
    /// Time (in seconds) spent in the linear solver computing the trust
    /// region step.
    pub linear_solver_time_in_seconds: f64,
    /// Number of times the Newton step was computed by solving a linear
    /// system.
    pub num_linear_solves: i32,
    /// Time (in seconds) spent evaluating the residual vector.
    pub residual_evaluation_time_in_seconds: f64,
    /// Number of residual only evaluations.
    pub num_residual_evaluations: i32,
    /// Time (in seconds) spent evaluating the jacobian matrix.
    pub jacobian_evaluation_time_in_seconds: f64,
    /// Number of Jacobian (and residual) evaluations.
    pub num_jacobian_evaluations: i32,
    /// Number of parameter blocks in the problem.
    pub num_parameter_blocks: i32,
    /// Number of parameters in the problem.
    pub num_parameters: i32,
    /// Dimension of the tangent space of the problem (or the number of
    /// columns in the Jacobian for the problem).
    pub num_effective_parameters: i32,
    /// Number of residual blocks in the problem.
    pub num_residual_blocks: i32,
    /// Number of residuals in the problem.
    pub num_residuals: i32,
    /// Number of parameter blocks in the problem after the inactive and
    /// constant parameter blocks have been removed.
    pub num_parameter_blocks_reduced: i32,
    /// Number of parameters in the reduced problem.
    pub num_parameters_reduced: i32,
    /// Dimension of the tangent space of the reduced problem.
    pub num_effective_parameters_reduced: i32,
    /// Number of residual blocks in the reduced problem.
    pub num_residual_blocks_reduced: i32,
    /// Number of residuals in the reduced problem.
    pub num_residuals_reduced: i32,
    /// Is the reduced problem bounds constrained.
    pub is_constrained: bool,
    /// Number of times the line search was used by the trust region
    /// minimizer for bounds constrained problems (sum of the line search
    /// iterations).
    pub num_line_search_steps: i32,
    /// Time (in seconds) spent evaluating the univariate cost function as
    /// part of a line search.
    pub line_search_cost_evaluation_time_in_seconds: f64,
    /// Time (in seconds) spent evaluating the gradient of the univariate
    /// cost function as part of a line search.
    pub line_search_gradient_evaluation_time_in_seconds: f64,
    /// Time (in seconds) spent minimizing the interpolating polynomial to
    /// compute the next candidate step size as part of a line search.
    pub line_search_polynomial_minimization_time_in_seconds: f64,
    /// Total time (in seconds) spent performing line searches.
    pub line_search_total_time_in_seconds: f64,
    /// Type of polynomial interpolation used by the line search.
    pub line_search_interpolation_type: LineSearchInterpolationType,
    /// Number of threads specified by the user for Jacobian and residual
    /// evaluation.
    pub num_threads_given: i32,
    /// Number of threads actually used by the solver.
    pub num_threads_used: i32,
    /// Type of the linear solver requested by the user.
    pub linear_solver_type_given: LinearSolverType,
    /// Type of the linear solver actually used.
    pub linear_solver_type_used: LinearSolverType,
    /// Type of trust region strategy.
    pub trust_region_strategy_type: TrustRegionStrategyType,
    /// Type of dense linear algebra library.
    pub dense_linear_algebra_library_type: DenseLinearAlgebraLibraryType,
    /// Type of sparse linear algebra library.
    pub sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType,
}

impl Default for SolverSummary {
    fn default() -> Self {
        Self {
            minimizer_type: MinimizerType::TrustRegion,
            termination_type: TerminationType::Failure,
            message: "ceres::Solve was not called.".to_string(),
            initial_cost: -1.0,
            final_cost: -1.0,
            fixed_cost: -1.0,
            iterations: Vec::new(),
            num_successful_steps: -1,
            num_unsuccessful_steps: -1,
            preprocessor_time_in_seconds: -1.0,
            minimizer_time_in_seconds: -1.0,
            postprocessor_time_in_seconds: -1.0,
            total_time_in_seconds: -1.0,
            linear_solver_time_in_seconds: -1.0,
            num_linear_solves: -1,
            residual_evaluation_time_in_seconds: -1.0,
            num_residual_evaluations: -1,
            jacobian_evaluation_time_in_seconds: -1.0,
            num_jacobian_evaluations: -1,
            num_parameter_blocks: -1,
            num_parameters: -1,
            num_effective_parameters: -1,
            num_residual_blocks: -1,
            num_residuals: -1,
            num_parameter_blocks_reduced: -1,
            num_parameters_reduced: -1,
            num_effective_parameters_reduced: -1,
            num_residual_blocks_reduced: -1,
            num_residuals_reduced: -1,
            is_constrained: false,
            num_line_search_steps: -1,
            line_search_cost_evaluation_time_in_seconds: -1.0,
            line_search_gradient_evaluation_time_in_seconds: -1.0,
            line_search_polynomial_minimization_time_in_seconds: -1.0,
            line_search_total_time_in_seconds: -1.0,
            line_search_interpolation_type: LineSearchInterpolationType::Cubic,
            num_threads_given: -1,
            num_threads_used: -1,
            linear_solver_type_given: LinearSolverType::SparseNormalCholesky,
            linear_solver_type_used: LinearSolverType::SparseNormalCholesky,
            trust_region_strategy_type: TrustRegionStrategyType::LevenbergMarquardt,
            dense_linear_algebra_library_type: DenseLinearAlgebraLibraryType::Eigen,
            sparse_linear_algebra_library_type: SparseLinearAlgebraLibraryType::Faer,
        }
    }
}

/// Version string reported in [`SolverSummary::full_report`].
pub const VERSION_STRING: &str = "2.0.0-rust-port-nalgebra-faer";

fn usize_to_i32(x: usize) -> i32 {
    i32::try_from(x).unwrap_or(i32::MAX)
}

impl SolverSummary {
    /// A brief one line description of the state of the solver after
    /// termination.
    pub fn brief_report(&self) -> String {
        format!(
            "Ceres Solver Report: Iterations: {}, Initial cost: {}, Final cost: {}, Termination: {}",
            self.num_successful_steps + self.num_unsuccessful_steps,
            fmt_e(self.initial_cost, 0, 6, false),
            fmt_e(self.final_cost, 0, 6, false),
            self.termination_type
        )
    }

    /// A full multiline description of the state of the solver after
    /// termination.
    pub fn full_report(&self) -> String {
        let d = |x: i32, w: usize| fmt_d(i64::from(x), w, true);
        let mut report = format!("\nSolver Summary (v {VERSION_STRING})\n\n");
        report += &format!("{:>45}    {:>21}\n", "Original", "Reduced");
        report += &format!(
            "Parameter blocks    {}{}\n",
            d(self.num_parameter_blocks, 25),
            d(self.num_parameter_blocks_reduced, 25)
        );
        report += &format!(
            "Parameters          {}{}\n",
            d(self.num_parameters, 25),
            d(self.num_parameters_reduced, 25)
        );
        if self.num_effective_parameters_reduced != self.num_parameters_reduced {
            report += &format!(
                "Effective parameters{}{}\n",
                d(self.num_effective_parameters, 25),
                d(self.num_effective_parameters_reduced, 25)
            );
        }
        report += &format!(
            "Residual blocks     {}{}\n",
            d(self.num_residual_blocks, 25),
            d(self.num_residual_blocks_reduced, 25)
        );
        report += &format!(
            "Residuals           {}{}\n",
            d(self.num_residuals, 25),
            d(self.num_residuals_reduced, 25)
        );

        report += &format!("\nMinimizer                 {:>19}\n", "TRUST_REGION");
        if matches!(
            self.linear_solver_type_used,
            LinearSolverType::DenseNormalCholesky | LinearSolverType::DenseQr
        ) {
            report += &format!(
                "\nDense linear algebra library  {:>15}\n",
                self.dense_linear_algebra_library_type.as_str()
            );
        }
        if self.linear_solver_type_used == LinearSolverType::SparseNormalCholesky {
            report += &format!(
                "\nSparse linear algebra library {:>15}\n",
                self.sparse_linear_algebra_library_type.as_str()
            );
        }
        report += &format!(
            "Trust region strategy     {:>19}",
            self.trust_region_strategy_type.as_str()
        );
        report += "\n";
        report += "\n";
        report += &format!("{:>45}    {:>21}\n", "Given", "Used");
        report += &format!(
            "Linear solver       {:>25}{:>25}\n",
            self.linear_solver_type_given.as_str(),
            self.linear_solver_type_used.as_str()
        );
        report += &format!(
            "Threads             {}{}\n",
            d(self.num_threads_given, 25),
            d(self.num_threads_used, 25)
        );
        report += &format!(
            "Linear solver ordering {:>22} {:>24}\n",
            "AUTOMATIC", "AUTOMATIC"
        );

        report += "\nCost:\n";
        report += &format!("Initial        {}\n", fmt_e(self.initial_cost, 30, 6, true));
        if self.termination_type != TerminationType::Failure
            && self.termination_type != TerminationType::UserFailure
        {
            report += &format!("Final          {}\n", fmt_e(self.final_cost, 30, 6, true));
            report += &format!(
                "Change         {}\n",
                fmt_e(self.initial_cost - self.final_cost, 30, 6, true)
            );
        }

        report += &format!(
            "\nMinimizer iterations         {}\n",
            d(self.num_successful_steps + self.num_unsuccessful_steps, 16)
        );
        report += &format!(
            "Successful steps               {}\n",
            d(self.num_successful_steps, 14)
        );
        report += &format!(
            "Unsuccessful steps             {}\n",
            d(self.num_unsuccessful_steps, 14)
        );

        // minimizer_type is always TRUST_REGION here.
        let line_search_used = self.is_constrained;

        if line_search_used {
            report += &format!(
                "Line search steps              {}\n",
                d(self.num_line_search_steps, 14)
            );
        }

        report += "\nTime (in seconds):\n";
        report += &format!(
            "Preprocessor        {}\n",
            fmt_f(self.preprocessor_time_in_seconds, 25, 6, false)
        );
        report += &format!(
            "\n  Residual only evaluation {} ({})\n",
            fmt_f(self.residual_evaluation_time_in_seconds, 18, 6, false),
            self.num_residual_evaluations
        );
        if line_search_used {
            report += &format!(
                "    Line search cost evaluation    {}\n",
                fmt_f(
                    self.line_search_cost_evaluation_time_in_seconds,
                    10,
                    6,
                    false
                )
            );
        }
        report += &format!(
            "  Jacobian & residual evaluation {} ({})\n",
            fmt_f(self.jacobian_evaluation_time_in_seconds, 12, 6, false),
            self.num_jacobian_evaluations
        );
        if line_search_used {
            report += &format!(
                "    Line search gradient evaluation   {}\n",
                fmt_f(
                    self.line_search_gradient_evaluation_time_in_seconds,
                    6,
                    6,
                    false
                )
            );
        }
        report += &format!(
            "  Linear solver       {} ({})\n",
            fmt_f(self.linear_solver_time_in_seconds, 23, 6, false),
            self.num_linear_solves
        );
        if line_search_used {
            report += &format!(
                "  Line search polynomial minimization  {}\n",
                fmt_f(
                    self.line_search_polynomial_minimization_time_in_seconds,
                    0,
                    6,
                    false
                )
            );
        }
        report += &format!(
            "Minimizer           {}\n\n",
            fmt_f(self.minimizer_time_in_seconds, 25, 6, false)
        );
        report += &format!(
            "Postprocessor        {}\n",
            fmt_f(self.postprocessor_time_in_seconds, 24, 6, false)
        );
        report += &format!(
            "Total               {}\n\n",
            fmt_f(self.total_time_in_seconds, 25, 6, false)
        );
        report += &format!(
            "Termination:        {:>25} ({})\n",
            self.termination_type.as_str(),
            self.message
        );
        report
    }

    /// Was the solver able to find a solution that can be used: true iff
    /// termination is `CONVERGENCE`, `NO_CONVERGENCE` or `USER_SUCCESS`.
    pub fn is_solution_usable(&self) -> bool {
        matches!(
            self.termination_type,
            TerminationType::Convergence
                | TerminationType::NoConvergence
                | TerminationType::UserSuccess
        )
    }
}

/// Minimizes the given problem using the options and fills `summary`
/// (C++ `ceres::Solve`). On a usable solution the problem's parameter blocks
/// hold the optimized values (read them with
/// [`Problem::parameter_block`]); otherwise they are left unchanged.
pub fn solve(options: &SolverOptions, problem: &mut Problem<'_>, summary: &mut SolverSummary) {
    let start_time = Instant::now();
    *summary = SolverSummary::default();

    if let Err(message) = options.is_valid() {
        summary.message = message;
        log::error!("Terminating: {}", summary.message);
        return;
    }

    // PreSolveSummarize.
    summary.num_parameter_blocks = usize_to_i32(problem.num_parameter_blocks());
    summary.num_parameters = usize_to_i32(problem.num_parameters());
    summary.num_effective_parameters = usize_to_i32(
        problem
            .parameter_blocks()
            .iter()
            .map(|&id| problem.parameter_block_local_size(id))
            .sum(),
    );
    summary.num_residual_blocks = usize_to_i32(problem.num_residual_blocks());
    summary.num_residuals = usize_to_i32(problem.num_residuals());
    summary.dense_linear_algebra_library_type = options.dense_linear_algebra_library_type;
    summary.linear_solver_type_given = options.linear_solver_type;
    summary.minimizer_type = options.minimizer_type;
    summary.num_threads_given = options.num_threads;
    summary.sparse_linear_algebra_library_type = options.sparse_linear_algebra_library_type;
    summary.trust_region_strategy_type = options.trust_region_strategy_type;
    summary.num_line_search_steps = 0;
    summary.line_search_cost_evaluation_time_in_seconds = 0.0;
    summary.line_search_gradient_evaluation_time_in_seconds = 0.0;
    summary.line_search_polynomial_minimization_time_in_seconds = 0.0;
    summary.line_search_total_time_in_seconds = 0.0;
    summary.line_search_interpolation_type = options.line_search_interpolation_type;

    // TrustRegionPreprocessor::Preprocess.
    let preprocessed = preprocess(problem);
    let mut solution: Option<Vec<f64>> = None;
    let mut reduced: Option<Program> = None;
    match preprocessed {
        Err(error) => {
            summary.fixed_cost = -1.0;
            summary.preprocessor_time_in_seconds = start_time.elapsed().as_secs_f64();
            summary.message = error;
        }
        Ok((program, fixed_cost)) => {
            summary.fixed_cost = fixed_cost;
            summary.linear_solver_type_used = options.linear_solver_type;
            summary.num_threads_used = options.num_threads;
            if program.num_parameter_blocks() == 0 {
                summary.preprocessor_time_in_seconds = start_time.elapsed().as_secs_f64();
                let minimizer_start_time = Instant::now();
                summary.message =
                    "Function tolerance reached. No non-constant parameter blocks found."
                        .to_string();
                summary.termination_type = TerminationType::Convergence;
                if options.logging_type != LoggingType::Silent {
                    log::debug!("{}", summary.message);
                }
                summary.initial_cost = summary.fixed_cost;
                summary.final_cost = summary.fixed_cost;
                summary.minimizer_time_in_seconds = minimizer_start_time.elapsed().as_secs_f64();
                summary.num_linear_solves = 0;
                summary.linear_solver_time_in_seconds = 0.0;
                summary.num_residual_evaluations = 0;
                summary.residual_evaluation_time_in_seconds = 0.0;
                summary.num_jacobian_evaluations = 0;
                summary.jacobian_evaluation_time_in_seconds = 0.0;
            } else {
                let mut reduced_parameters = vec![0.0; program.num_parameters];
                program.parameter_blocks_to_state_vector(problem, &mut reduced_parameters);
                let original_reduced_parameters = reduced_parameters.clone();

                let num_threads = usize::try_from(options.num_threads).unwrap_or(1);
                let mut evaluator = Evaluator::new(problem, program.clone(), num_threads);
                let strategy = LevenbergMarquardtStrategy::new(
                    TrustRegionStrategyOptions {
                        initial_radius: options.initial_trust_region_radius,
                        max_radius: options.max_trust_region_radius,
                        min_lm_diagonal: options.min_lm_diagonal,
                        max_lm_diagonal: options.max_lm_diagonal,
                    },
                    LinearSolver::new(options.linear_solver_type),
                );
                // SetupMinimizerOptions.
                let is_constrained = program.is_bounds_constrained(problem);
                summary.preprocessor_time_in_seconds = start_time.elapsed().as_secs_f64();

                let minimizer_start_time = Instant::now();
                let (linear_solves, linear_solver_time) = {
                    let mut minimizer = TrustRegionMinimizer::new(
                        options,
                        &mut evaluator,
                        strategy,
                        is_constrained,
                        summary.preprocessor_time_in_seconds,
                    );
                    minimizer.minimize(&mut reduced_parameters, summary);
                    minimizer.linear_solver_statistics()
                };
                summary.minimizer_time_in_seconds = minimizer_start_time.elapsed().as_secs_f64();

                let stats = evaluator.statistics;
                summary.residual_evaluation_time_in_seconds = stats.residual_time;
                summary.num_residual_evaluations = usize_to_i32(stats.residual_calls);
                summary.jacobian_evaluation_time_in_seconds = stats.jacobian_time;
                summary.num_jacobian_evaluations = usize_to_i32(stats.jacobian_calls);
                summary.num_linear_solves = linear_solves;
                summary.linear_solver_time_in_seconds = linear_solver_time;

                solution = Some(if summary.is_solution_usable() {
                    reduced_parameters
                } else {
                    original_reduced_parameters
                });
            }
            reduced = Some(program);
        }
    }

    // Postprocess.
    let postprocessor_start_time = Instant::now();
    if let (Some(program), Some(solution)) = (&reduced, &solution) {
        program.state_vector_to_parameter_blocks(problem, solution);
    }
    // SetSummaryFinalCost.
    // We need the loop here, instead of just looking at the last iteration
    // because the minimizer maybe making non-monotonic steps.
    summary.final_cost = summary.initial_cost;
    for it in &summary.iterations {
        summary.final_cost = summary.final_cost.min(it.cost);
    }
    if let Some(program) = &reduced {
        summary.num_parameter_blocks_reduced = usize_to_i32(program.num_parameter_blocks());
        summary.num_parameters_reduced = usize_to_i32(program.num_parameters);
        summary.num_effective_parameters_reduced = usize_to_i32(program.num_effective_parameters);
        summary.num_residual_blocks_reduced = usize_to_i32(program.num_residual_blocks());
        summary.num_residuals_reduced = usize_to_i32(program.num_residuals);
    }
    summary.postprocessor_time_in_seconds = postprocessor_start_time.elapsed().as_secs_f64();
    summary.total_time_in_seconds = start_time.elapsed().as_secs_f64();
}

/// Validates the program and builds the reduced program (C++
/// `TrustRegionPreprocessor::Preprocess`).
fn preprocess(problem: &Problem<'_>) -> Result<(Program, f64), String> {
    // IsProgramValid: all parameter values must be finite.
    for id in problem.parameter_blocks() {
        let values = problem.parameter_block(id);
        if let Some(invalid_index) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "ParameterBlock: {id:?} with size {} has at least one invalid value.\n\
                 First invalid value is at index: {invalid_index}.\nParameter block values: {values:?}",
                values.len()
            ));
        }
    }
    // IsProgramValid: the program must be feasible.
    is_feasible(problem)?;
    Program::create_reduced_program(problem)
}
