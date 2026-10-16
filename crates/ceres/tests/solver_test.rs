// Ported from Ceres Solver 2.0.0: internal/ceres/solver_test.cc,
// internal/ceres/trust_region_minimizer_test.cc, examples/powell.cc,
// examples/curve_fitting.cc and examples/robust_curve_fitting.cc, plus
// pose graph tests exercising the sparse solver, threading determinism and
// local parameterizations (no upstream counterpart).

// Index loops mirror the C++ test code.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use ceres::{
    AutoDiffCostFunction, AutoDiffLocalParameterization, CallbackReturnType, CauchyLoss,
    CostFunction, CostFunctor, HuberLoss, IterationSummary, LinearSolverType, LoggingType,
    ParameterBlockId, PlusFunctor, Problem, QuaternionParameterization, Scalar, SizedCostFunction,
    SolverOptions, SolverSummary, SparseLinearAlgebraLibraryType, SubsetParameterization,
    TerminationType, solve,
};
use nalgebra::{Rotation2, UnitQuaternion, Vector2, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn silent_options() -> SolverOptions {
    SolverOptions {
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    }
}

// ---------------------------------------------------------------------------
// solver_test.cc
// ---------------------------------------------------------------------------

#[test]
fn solver_options_default_trust_region_options_are_valid() {
    let options = SolverOptions::default();
    assert!(options.is_valid().is_ok(), "{:?}", options.is_valid());
}

// DefaultLineSearchOptionsAreValid: the line search minimizer is not ported.

struct QuadraticCostFunctor;
impl CostFunctor for QuadraticCostFunctor {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        residual[0] = T::constant(5.0) - p[0][0];
        true
    }
}

fn quadratic_cost_function() -> Box<dyn CostFunction> {
    Box::new(AutoDiffCostFunction::<_, 1>::new(
        QuadraticCostFunctor,
        1,
        &[1],
    ))
}

// UpdateStateEveryIteration*: `update_state_every_iteration` and evaluation
// callbacks are not ported. Instead check that callbacks see every
// iteration and can terminate the solver.
#[test]
fn solver_callbacks_are_called_every_iteration() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[50.0]);
    problem.add_residual_block(quadratic_cost_function(), None, &[x]);

    let seen = Arc::new(Mutex::new(Vec::<IterationSummary>::new()));
    let seen_in_callback = seen.clone();
    let mut options = silent_options();
    options
        .callbacks
        .push(Arc::new(move |s: &IterationSummary| {
            seen_in_callback.lock().unwrap().push(s.clone());
            CallbackReturnType::SolverContinue
        }));
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), summary.iterations.len());
    assert_eq!(*seen, summary.iterations);
    assert!((problem.parameter_block(x)[0] - 5.0).abs() < 1e-8);
}

#[test]
fn solver_callback_can_abort_and_terminate() {
    for (status, termination, usable) in [
        (
            CallbackReturnType::SolverAbort,
            TerminationType::UserFailure,
            false,
        ),
        (
            CallbackReturnType::SolverTerminateSuccessfully,
            TerminationType::UserSuccess,
            true,
        ),
    ] {
        let mut problem = Problem::new();
        let x = problem.add_parameter_block(&[50.0]);
        problem.add_residual_block(quadratic_cost_function(), None, &[x]);
        let mut options = silent_options();
        options
            .callbacks
            .push(Arc::new(move |s: &IterationSummary| {
                if s.iteration == 1 {
                    status
                } else {
                    CallbackReturnType::SolverContinue
                }
            }));
        let mut summary = SolverSummary::default();
        solve(&options, &mut problem, &mut summary);
        assert_eq!(summary.termination_type, termination);
        assert_eq!(summary.is_solution_usable(), usable);
        assert_eq!(summary.iterations.len(), 2);
        if usable {
            // The state after the first iteration is returned.
            assert!(problem.parameter_block(x)[0] < 50.0);
            assert_eq!(summary.final_cost, summary.iterations[1].cost);
        } else {
            // The user's parameter blocks are not updated.
            assert_eq!(problem.parameter_block(x)[0], 50.0);
        }
    }
}

struct UnaryIdentityCostFunction;
impl SizedCostFunction for UnaryIdentityCostFunction {
    const NUM_RESIDUALS: usize = 1;
    const PARAMETER_BLOCK_SIZES: &'static [usize] = &[1];
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        residuals[0] = parameters[0][0];
        if let Some(Some(j)) = jacobians.map(|j| j[0].as_deref_mut()) {
            j[0] = 1.0;
        }
        true
    }
}

#[test]
fn solver_trust_region_problem_has_no_parameter_blocks() {
    let mut problem = Problem::new();
    let options = SolverOptions::default();
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    assert_eq!(
        summary.message,
        "Function tolerance reached. No non-constant parameter blocks found."
    );
}

#[test]
fn solver_trust_region_problem_has_zero_residuals() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[1.0]);
    let options = SolverOptions::default();
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    assert_eq!(
        summary.message,
        "Function tolerance reached. No non-constant parameter blocks found."
    );
}

#[test]
fn solver_trust_region_problem_is_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(Box::new(UnaryIdentityCostFunction), None, &[x]);
    problem.set_parameter_block_constant(x);
    let options = SolverOptions::default();
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    assert_eq!(summary.initial_cost, 1.0 / 2.0);
    assert_eq!(summary.final_cost, 1.0 / 2.0);
}

// Sparse*No{SuiteSparse,CXSparse,AccelerateSparse,EigenSparse,SparseLibrary}:
// in this port only the faer backend is available.
#[test]
fn solver_sparse_normal_cholesky_unavailable_libraries() {
    for library in [
        SparseLinearAlgebraLibraryType::SuiteSparse,
        SparseLinearAlgebraLibraryType::CxSparse,
        SparseLinearAlgebraLibraryType::AccelerateSparse,
        SparseLinearAlgebraLibraryType::EigenSparse,
        SparseLinearAlgebraLibraryType::NoSparse,
    ] {
        let options = SolverOptions {
            linear_solver_type: LinearSolverType::SparseNormalCholesky,
            sparse_linear_algebra_library_type: library,
            ..SolverOptions::default()
        };
        assert!(options.is_valid().is_err(), "{library}");
    }
}

#[test]
fn solver_invalid_options_terminate_with_failure() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(Box::new(UnaryIdentityCostFunction), None, &[x]);
    let options = SolverOptions {
        num_threads: 0,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert_eq!(summary.termination_type, TerminationType::Failure);
    assert!(
        summary.message.contains("num_threads"),
        "{}",
        summary.message
    );
    assert_eq!(problem.parameter_block(x)[0], 1.0);
}

#[test]
fn solver_linear_solver_type_normal_operation() {
    for t in [
        LinearSolverType::DenseQr,
        LinearSolverType::DenseNormalCholesky,
        LinearSolverType::SparseNormalCholesky,
    ] {
        let options = SolverOptions {
            linear_solver_type: t,
            ..SolverOptions::default()
        };
        assert!(options.is_valid().is_ok());
    }
}

struct DummyCostFunction;
impl SizedCostFunction for DummyCostFunction {
    const NUM_RESIDUALS: usize = 2;
    const PARAMETER_BLOCK_SIZES: &'static [usize] = &[1];
    fn evaluate(
        &self,
        _: &[&[f64]],
        residuals: &mut [f64],
        _: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        for (i, r) in residuals.iter_mut().enumerate() {
            *r = (2 * 2 + i) as f64;
        }
        true
    }
}

#[test]
fn solver_fixed_cost_for_constant_problem() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(Box::new(DummyCostFunction), None, &[x]);
    problem.set_parameter_block_constant(x);
    let expected_cost = 41.0 / 2.0; // 1/2 * ((4 + 0)^2 + (4 + 1)^2)
    let options = SolverOptions::default();
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert!(summary.is_solution_usable());
    assert_eq!(summary.fixed_cost, expected_cost);
    assert_eq!(summary.initial_cost, expected_cost);
    assert_eq!(summary.final_cost, expected_cost);
    assert_eq!(summary.iterations.len(), 0);
}

struct LinearCostFunction;
impl CostFunctor for LinearCostFunction {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        residual[0] = T::constant(10.0) - p[0][0];
        residual[1] = T::constant(5.0) - p[1][0];
        true
    }
}

#[test]
fn solver_zero_sized_local_parameterization_holds_parameter_block_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(
        Box::new(AutoDiffCostFunction::<_, 2>::new(
            LinearCostFunction,
            2,
            &[1, 1],
        )),
        None,
        &[x, y],
    );
    problem.set_parameterization(y, Some(Box::new(SubsetParameterization::new(1, &[0]))));
    assert!(problem.is_parameter_block_constant(y));

    let options = SolverOptions {
        function_tolerance: 0.0,
        gradient_tolerance: 0.0,
        parameter_tolerance: 0.0,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);

    assert_eq!(summary.termination_type, TerminationType::Convergence);
    assert!((problem.parameter_block(x)[0] - 10.0).abs() <= 1e-7);
    assert_eq!(problem.parameter_block(y)[0], 1.0);
}

// ---------------------------------------------------------------------------
// trust_region_minimizer_test.cc
// ---------------------------------------------------------------------------

/// Powell's singular function with an analytic jacobian, as one residual
/// block over four 1-dimensional parameter blocks (the C++ test uses a
/// custom Evaluator; here inactive columns are constant parameter blocks).
struct PowellCostFunction;
impl SizedCostFunction for PowellCostFunction {
    const NUM_RESIDUALS: usize = 4;
    const PARAMETER_BLOCK_SIZES: &'static [usize] = &[1, 1, 1, 1];
    fn evaluate(
        &self,
        p: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let (x1, x2, x3, x4) = (p[0][0], p[1][0], p[2][0], p[3][0]);
        residuals[0] = x1 + 10.0 * x2;
        residuals[1] = 5.0f64.sqrt() * (x3 - x4);
        residuals[2] = (x2 - 2.0 * x3).powf(2.0);
        residuals[3] = 10.0f64.sqrt() * (x1 - x4).powf(2.0);
        if let Some(j) = jacobians {
            let cols = [
                [1.0, 0.0, 0.0, 10.0f64.sqrt() * 2.0 * (x1 - x4) * (1.0 - x4)],
                [10.0, 0.0, 2.0 * (x2 - 2.0 * x3) * (1.0 - 2.0 * x3), 0.0],
                [0.0, 5.0f64.sqrt(), 2.0 * (x2 - 2.0 * x3) * (x2 - 2.0), 0.0],
                [
                    0.0,
                    -(5.0f64.sqrt()),
                    0.0,
                    10.0f64.sqrt() * 2.0 * (x1 - x4) * (x1 - 1.0),
                ],
            ];
            for (b, col) in cols.iter().enumerate() {
                if let Some(jb) = j[b].as_deref_mut() {
                    jb.copy_from_slice(col);
                }
            }
        }
        true
    }
}

fn is_trust_region_solve_successful(cols: [bool; 4]) {
    let mut parameters = [3.0, -1.0, 0.0, 1.0];
    for i in 0..4 {
        if !cols[i] {
            parameters[i] = 0.0;
        }
    }
    let mut problem = Problem::new();
    let ids: Vec<ParameterBlockId> = parameters
        .iter()
        .map(|&p| problem.add_parameter_block(&[p]))
        .collect();
    problem.add_residual_block(Box::new(PowellCostFunction), None, &ids);
    for i in 0..4 {
        if !cols[i] {
            problem.set_parameter_block_constant(ids[i]);
        }
    }
    let options = SolverOptions {
        gradient_tolerance: 1e-26,
        function_tolerance: 1e-26,
        parameter_tolerance: 1e-26,
        linear_solver_type: LinearSolverType::DenseQr,
        initial_trust_region_radius: 1e4,
        max_trust_region_radius: 1e20,
        min_lm_diagonal: 1e-6,
        max_lm_diagonal: 1e32,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    for (i, &id) in ids.iter().enumerate() {
        assert!(
            problem.parameter_block(id)[0].abs() <= 0.001,
            "cols {cols:?}: x{} = {}",
            i + 1,
            problem.parameter_block(id)[0]
        );
    }
}

#[test]
fn trust_region_minimizer_powells_singular_function_using_levenberg_marquardt() {
    let t = true;
    let f = false;
    for cols in [
        [t, t, t, t],
        [t, t, t, f],
        [t, f, t, t],
        [f, t, t, t],
        [t, t, f, f],
        [t, f, t, f],
        [f, t, t, f],
        [t, f, f, t],
        [f, t, f, t],
        [f, f, t, t],
        [t, f, f, f],
        [f, t, f, f],
        [f, f, t, f],
        [f, f, f, t],
    ] {
        is_trust_region_solve_successful(cols);
    }
}

// PowellsSingularFunctionUsingDogleg: DOGLEG is not ported.

struct CurveCostFunction {
    num_vertices: usize,
    target_length: f64,
    sizes: Vec<usize>,
}

impl CostFunction for CurveCostFunction {
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let n = self.num_vertices;
        residuals[0] = self.target_length;
        for i in 0..n {
            let prev = (n + i - 1) % n;
            let mut length = 0.0;
            for dim in 0..2 {
                let diff = parameters[prev][dim] - parameters[i][dim];
                length += diff * diff;
            }
            residuals[0] -= length.sqrt();
        }
        let Some(jacobians) = jacobians else {
            return true;
        };
        for i in 0..n {
            if let Some(j) = jacobians[i].as_deref_mut() {
                let prev = (n + i - 1) % n;
                let next = (i + 1) % n;
                let mut u = [0.0; 2];
                let mut v = [0.0; 2];
                let mut norm_u = 0.0;
                let mut norm_v = 0.0;
                for dim in 0..2 {
                    u[dim] = parameters[i][dim] - parameters[prev][dim];
                    norm_u += u[dim] * u[dim];
                    v[dim] = parameters[next][dim] - parameters[i][dim];
                    norm_v += v[dim] * v[dim];
                }
                let norm_u: f64 = norm_u.sqrt();
                let norm_v: f64 = norm_v.sqrt();
                for dim in 0..2 {
                    j[dim] = 0.0;
                    if norm_u > f64::MIN_POSITIVE {
                        j[dim] -= u[dim] / norm_u;
                    }
                    if norm_v > f64::MIN_POSITIVE {
                        j[dim] += v[dim] / norm_v;
                    }
                }
            }
        }
        true
    }
    fn num_residuals(&self) -> usize {
        1
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.sizes
    }
}

#[test]
fn trust_region_minimizer_jacobi_scaling_test() {
    let n = 6;
    let mut problem = Problem::new();
    let pi = PI; // C++: 3.1415926535897932384626433
    let ids: Vec<ParameterBlockId> = (0..n)
        .map(|i| {
            let theta = i as f64 * 2.0 * pi / n as f64;
            problem.add_parameter_block(&[theta.cos(), theta.sin()])
        })
        .collect();
    problem.add_residual_block(
        Box::new(CurveCostFunction {
            num_vertices: n,
            target_length: 10.0,
            sizes: vec![2; n],
        }),
        None,
        &ids,
    );
    let options = SolverOptions {
        linear_solver_type: LinearSolverType::DenseQr,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    assert!(summary.final_cost <= 1e-10, "{}", summary.full_report());
}

// GradientToleranceConvergenceUpdatesStep: parameter bounds are not ported.

// ---------------------------------------------------------------------------
// examples/powell.cc
// ---------------------------------------------------------------------------

struct F1;
impl CostFunctor for F1 {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        // f1 = x1 + 10 * x2;
        residual[0] = p[0][0] + T::constant(10.0) * p[1][0];
        true
    }
}
struct F2;
impl CostFunctor for F2 {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        // f2 = sqrt(5) (x3 - x4)
        residual[0] = T::constant(5.0f64.sqrt()) * (p[0][0] - p[1][0]);
        true
    }
}
struct F3;
impl CostFunctor for F3 {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        // f3 = (x2 - 2 x3)^2
        let d = p[0][0] - T::constant(2.0) * p[1][0];
        residual[0] = d * d;
        true
    }
}
struct F4;
impl CostFunctor for F4 {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        // f4 = sqrt(10) (x1 - x4)^2
        let d = p[0][0] - p[1][0];
        residual[0] = T::constant(10.0f64.sqrt()) * d * d;
        true
    }
}

fn powell_problem() -> (Problem<'static>, [ParameterBlockId; 4]) {
    let mut problem = Problem::new();
    let x1 = problem.add_parameter_block(&[3.0]);
    let x2 = problem.add_parameter_block(&[-1.0]);
    let x3 = problem.add_parameter_block(&[0.0]);
    let x4 = problem.add_parameter_block(&[1.0]);
    // Add residual terms to the problem using the autodiff wrapper to get the
    // derivatives automatically.
    problem.add_residual_block(
        Box::new(AutoDiffCostFunction::<_, 2>::new(F1, 1, &[1, 1])),
        None,
        &[x1, x2],
    );
    problem.add_residual_block(
        Box::new(AutoDiffCostFunction::<_, 2>::new(F2, 1, &[1, 1])),
        None,
        &[x3, x4],
    );
    problem.add_residual_block(
        Box::new(AutoDiffCostFunction::<_, 2>::new(F3, 1, &[1, 1])),
        None,
        &[x2, x3],
    );
    problem.add_residual_block(
        Box::new(AutoDiffCostFunction::<_, 2>::new(F4, 1, &[1, 1])),
        None,
        &[x1, x4],
    );
    (problem, [x1, x2, x3, x4])
}

#[test]
fn example_powell() {
    for linear_solver_type in [
        LinearSolverType::DenseQr,
        LinearSolverType::DenseNormalCholesky,
        LinearSolverType::SparseNormalCholesky,
    ] {
        let (mut problem, ids) = powell_problem();
        let options = SolverOptions {
            max_num_iterations: 100,
            linear_solver_type,
            logging_type: LoggingType::Silent,
            ..SolverOptions::default()
        };
        let mut summary = SolverSummary::default();
        solve(&options, &mut problem, &mut summary);
        assert!(summary.is_solution_usable(), "{}", summary.full_report());
        // 1/2 (1 + 5 + 4 + 205), up to rounding of sqrt(5)^2 and sqrt(10)^2.
        assert!((summary.initial_cost - 107.5).abs() <= 1e-12);
        assert!(summary.final_cost < 1e-12, "{}", summary.full_report());
        for id in ids {
            assert!(problem.parameter_block(id)[0].abs() < 1e-3);
        }
        let brief = summary.brief_report();
        assert!(
            brief.starts_with("Ceres Solver Report: Iterations: "),
            "{brief}"
        );
        assert!(brief.contains("Initial cost: 1.075000e+02"), "{brief}");
    }
}

// ---------------------------------------------------------------------------
// examples/curve_fitting.cc and examples/robust_curve_fitting.cc
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const CURVE_FITTING_DATA: [[f64; 2]; 67] = [
    [0.000000e+00, 1.133898e+00],
    [7.500000e-02, 1.334902e+00],
    [1.500000e-01, 1.213546e+00],
    [2.250000e-01, 1.252016e+00],
    [3.000000e-01, 1.392265e+00],
    [3.750000e-01, 1.314458e+00],
    [4.500000e-01, 1.472541e+00],
    [5.250000e-01, 1.536218e+00],
    [6.000000e-01, 1.355679e+00],
    [6.750000e-01, 1.463566e+00],
    [7.500000e-01, 1.490201e+00],
    [8.250000e-01, 1.658699e+00],
    [9.000000e-01, 1.067574e+00],
    [9.750000e-01, 1.464629e+00],
    [1.050000e+00, 1.402653e+00],
    [1.125000e+00, 1.713141e+00],
    [1.200000e+00, 1.527021e+00],
    [1.275000e+00, 1.702632e+00],
    [1.350000e+00, 1.423899e+00],
    [1.425000e+00, 1.543078e+00],
    [1.500000e+00, 1.664015e+00],
    [1.575000e+00, 1.732484e+00],
    [1.650000e+00, 1.543296e+00],
    [1.725000e+00, 1.959523e+00],
    [1.800000e+00, 1.685132e+00],
    [1.875000e+00, 1.951791e+00],
    [1.950000e+00, 2.095346e+00],
    [2.025000e+00, 2.361460e+00],
    [2.100000e+00, 2.169119e+00],
    [2.175000e+00, 2.061745e+00],
    [2.250000e+00, 2.178641e+00],
    [2.325000e+00, 2.104346e+00],
    [2.400000e+00, 2.584470e+00],
    [2.475000e+00, 1.914158e+00],
    [2.550000e+00, 2.368375e+00],
    [2.625000e+00, 2.686125e+00],
    [2.700000e+00, 2.712395e+00],
    [2.775000e+00, 2.499511e+00],
    [2.850000e+00, 2.558897e+00],
    [2.925000e+00, 2.309154e+00],
    [3.000000e+00, 2.869503e+00],
    [3.075000e+00, 3.116645e+00],
    [3.150000e+00, 3.094907e+00],
    [3.225000e+00, 2.471759e+00],
    [3.300000e+00, 3.017131e+00],
    [3.375000e+00, 3.232381e+00],
    [3.450000e+00, 2.944596e+00],
    [3.525000e+00, 3.385343e+00],
    [3.600000e+00, 3.199826e+00],
    [3.675000e+00, 3.423039e+00],
    [3.750000e+00, 3.621552e+00],
    [3.825000e+00, 3.559255e+00],
    [3.900000e+00, 3.530713e+00],
    [3.975000e+00, 3.561766e+00],
    [4.050000e+00, 3.544574e+00],
    [4.125000e+00, 3.867945e+00],
    [4.200000e+00, 4.049776e+00],
    [4.275000e+00, 3.885601e+00],
    [4.350000e+00, 4.110505e+00],
    [4.425000e+00, 4.345320e+00],
    [4.500000e+00, 4.161241e+00],
    [4.575000e+00, 4.363407e+00],
    [4.650000e+00, 4.161576e+00],
    [4.725000e+00, 4.619728e+00],
    [4.800000e+00, 4.737410e+00],
    [4.875000e+00, 4.727863e+00],
    [4.950000e+00, 4.669206e+00],
];

#[rustfmt::skip]
const ROBUST_CURVE_FITTING_DATA: [[f64; 2]; 67] = [
    [0.000000e+00, 1.133898e+00],
    [7.500000e-02, 1.334902e+00],
    [1.500000e-01, 1.213546e+00],
    [2.250000e-01, 1.252016e+00],
    [3.000000e-01, 1.392265e+00],
    [3.750000e-01, 1.314458e+00],
    [4.500000e-01, 1.472541e+00],
    [5.250000e-01, 1.536218e+00],
    [6.000000e-01, 1.355679e+00],
    [6.750000e-01, 1.463566e+00],
    [7.500000e-01, 1.490201e+00],
    [8.250000e-01, 1.658699e+00],
    [9.000000e-01, 1.067574e+00],
    [9.750000e-01, 1.464629e+00],
    [1.050000e+00, 1.402653e+00],
    [1.125000e+00, 1.713141e+00],
    [1.200000e+00, 1.527021e+00],
    [1.275000e+00, 1.702632e+00],
    [1.350000e+00, 1.423899e+00],
    [1.425000e+00, 5.543078e+00],
    [1.500000e+00, 5.664015e+00],
    [1.575000e+00, 1.732484e+00],
    [1.650000e+00, 1.543296e+00],
    [1.725000e+00, 1.959523e+00],
    [1.800000e+00, 1.685132e+00],
    [1.875000e+00, 1.951791e+00],
    [1.950000e+00, 2.095346e+00],
    [2.025000e+00, 2.361460e+00],
    [2.100000e+00, 2.169119e+00],
    [2.175000e+00, 2.061745e+00],
    [2.250000e+00, 2.178641e+00],
    [2.325000e+00, 2.104346e+00],
    [2.400000e+00, 2.584470e+00],
    [2.475000e+00, 1.914158e+00],
    [2.550000e+00, 2.368375e+00],
    [2.625000e+00, 2.686125e+00],
    [2.700000e+00, 2.712395e+00],
    [2.775000e+00, 2.499511e+00],
    [2.850000e+00, 2.558897e+00],
    [2.925000e+00, 2.309154e+00],
    [3.000000e+00, 2.869503e+00],
    [3.075000e+00, 3.116645e+00],
    [3.150000e+00, 3.094907e+00],
    [3.225000e+00, 2.471759e+00],
    [3.300000e+00, 3.017131e+00],
    [3.375000e+00, 3.232381e+00],
    [3.450000e+00, 2.944596e+00],
    [3.525000e+00, 3.385343e+00],
    [3.600000e+00, 3.199826e+00],
    [3.675000e+00, 3.423039e+00],
    [3.750000e+00, 3.621552e+00],
    [3.825000e+00, 3.559255e+00],
    [3.900000e+00, 3.530713e+00],
    [3.975000e+00, 3.561766e+00],
    [4.050000e+00, 3.544574e+00],
    [4.125000e+00, 3.867945e+00],
    [4.200000e+00, 4.049776e+00],
    [4.275000e+00, 3.885601e+00],
    [4.350000e+00, 4.110505e+00],
    [4.425000e+00, 4.345320e+00],
    [4.500000e+00, 4.161241e+00],
    [4.575000e+00, 4.363407e+00],
    [4.650000e+00, 4.161576e+00],
    [4.725000e+00, 4.619728e+00],
    [4.800000e+00, 4.737410e+00],
    [4.875000e+00, 4.727863e+00],
    [4.950000e+00, 4.669206e+00],
];

struct ExponentialResidual {
    x: f64,
    y: f64,
}

impl CostFunctor for ExponentialResidual {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
        residual[0] = T::constant(self.y) - (p[0][0] * T::constant(self.x) + p[1][0]).exp();
        true
    }
}

fn fit_curve(
    data: &[[f64; 2]],
    robust: bool,
    max_num_iterations: i32,
) -> (f64, f64, SolverSummary) {
    let mut problem = Problem::new();
    let m = problem.add_parameter_block(&[0.0]);
    let c = problem.add_parameter_block(&[0.0]);
    for &[x, y] in data {
        problem.add_residual_block(
            Box::new(AutoDiffCostFunction::<_, 2>::new(
                ExponentialResidual { x, y },
                1,
                &[1, 1],
            )),
            if robust {
                Some(Box::new(CauchyLoss::new(0.5)))
            } else {
                None
            },
            &[m, c],
        );
    }
    let options = SolverOptions {
        max_num_iterations,
        linear_solver_type: LinearSolverType::DenseQr,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    (
        problem.parameter_block(m)[0],
        problem.parameter_block(c)[0],
        summary,
    )
}

#[test]
fn example_curve_fitting() {
    let (m, c, summary) = fit_curve(&CURVE_FITTING_DATA, false, 25);
    assert_eq!(
        summary.termination_type,
        TerminationType::Convergence,
        "{}",
        summary.full_report()
    );
    // Values reported by Ceres for this example (docs: "Final m: 0.291861
    // c: 0.131439"); the data was generated with m = 0.3, c = 0.1.
    assert!((m - 0.291861).abs() < 1e-5, "m = {m}");
    assert!((c - 0.131439).abs() < 1e-5, "c = {c}");
}

#[test]
fn example_robust_curve_fitting() {
    let (m_robust, c_robust, summary) = fit_curve(&ROBUST_CURVE_FITTING_DATA, true, 50);
    assert_eq!(
        summary.termination_type,
        TerminationType::Convergence,
        "{}",
        summary.full_report()
    );
    let (m_plain, c_plain, _) = fit_curve(&ROBUST_CURVE_FITTING_DATA, false, 50);
    // The data was generated with m = 0.3, c = 0.1 plus outliers; the robust
    // fit must be much closer to the truth than the plain least squares fit.
    let err_robust = (m_robust - 0.3).hypot(c_robust - 0.1);
    let err_plain = (m_plain - 0.3).hypot(c_plain - 0.1);
    assert!(err_robust < 0.06, "robust m = {m_robust}, c = {c_robust}");
    assert!(
        err_robust < err_plain,
        "robust {err_robust} vs plain {err_plain}"
    );
}

// ---------------------------------------------------------------------------
// Pose graph tests (no upstream counterpart): 2D SPA-style constraints as in
// Cartographer's spa_cost_function_2d, solved with all linear solvers.
// ---------------------------------------------------------------------------

fn normalize_angle_difference<T: Scalar>(mut difference: T) -> T {
    let pi = T::constant(PI);
    while difference > pi {
        difference -= T::constant(2.0 * PI);
    }
    while difference < -pi {
        difference += T::constant(2.0 * PI);
    }
    difference
}

/// Residual of the relative pose between `start` and `end` (x, y, theta)
/// against a measurement.
struct SpaCostFunction2D {
    measured: [f64; 3],
    translation_weight: f64,
    rotation_weight: f64,
}

impl CostFunctor for SpaCostFunction2D {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], e: &mut [T]) -> bool {
        let (start, end) = (p[0], p[1]);
        let rotation = Rotation2::new(start[2]);
        let delta = Vector2::new(end[0] - start[0], end[1] - start[1]);
        let h = rotation.inverse() * delta;
        e[0] = T::constant(self.translation_weight) * (T::constant(self.measured[0]) - h.x);
        e[1] = T::constant(self.translation_weight) * (T::constant(self.measured[1]) - h.y);
        e[2] = T::constant(self.rotation_weight)
            * normalize_angle_difference(T::constant(self.measured[2]) - (end[2] - start[2]));
        true
    }
}

fn relative_2d(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let r = Rotation2::new(a[2]).inverse() * Vector2::new(b[0] - a[0], b[1] - a[1]);
    [r.x, r.y, normalize_angle_difference(b[2] - a[2])]
}

fn compose_2d(a: &[f64; 3], d: &[f64; 3]) -> [f64; 3] {
    let t = Rotation2::new(a[2]) * Vector2::new(d[0], d[1]);
    [a[0] + t.x, a[1] + t.y, a[2] + d[2]]
}

struct PoseGraph2D {
    initial: Vec<[f64; 3]>,
    truth: Vec<[f64; 3]>,
    edges: Vec<(usize, usize, [f64; 3])>,
}

/// A noisy loop trajectory with odometry edges and loop closures.
fn make_pose_graph_2d(num_nodes: usize, seed: u64) -> PoseGraph2D {
    let mut rng = StdRng::seed_from_u64(seed);
    let laps = 3.0;
    let truth: Vec<[f64; 3]> = (0..num_nodes)
        .map(|i| {
            let t = laps * 2.0 * PI * i as f64 / num_nodes as f64;
            let radius = 10.0 + (i % 7) as f64 * 0.1;
            [
                radius * t.cos(),
                radius * t.sin(),
                normalize_angle_difference(t + PI / 2.0),
            ]
        })
        .collect();
    let mut noise = |sigma: f64| (rng.random::<f64>() - 0.5) * 2.0 * sigma;
    let mut edges = Vec::new();
    for i in 0..num_nodes - 1 {
        let mut m = relative_2d(&truth[i], &truth[i + 1]);
        m[0] += noise(0.02);
        m[1] += noise(0.02);
        m[2] += noise(0.01);
        edges.push((i, i + 1, m));
    }
    let per_lap = (num_nodes as f64 / laps) as usize;
    for i in (0..num_nodes).step_by(5) {
        let j = i + per_lap;
        if j < num_nodes {
            let mut m = relative_2d(&truth[i], &truth[j]);
            m[0] += noise(0.01);
            m[1] += noise(0.01);
            m[2] += noise(0.005);
            edges.push((i, j, m));
        }
    }
    // Initial guess: integrate odometry.
    let mut initial = vec![truth[0]];
    for (_, _, m) in edges.iter().take(num_nodes - 1) {
        let next = compose_2d(initial.last().unwrap(), m);
        initial.push(next);
    }
    PoseGraph2D {
        initial,
        truth,
        edges,
    }
}

fn build_pose_graph_problem_2d(
    graph: &PoseGraph2D,
    poses: &[[f64; 3]],
) -> (Problem<'static>, Vec<ParameterBlockId>) {
    let mut problem = Problem::new();
    let ids: Vec<ParameterBlockId> = poses
        .iter()
        .map(|p| problem.add_parameter_block(p))
        .collect();
    for (k, &(i, j, m)) in graph.edges.iter().enumerate() {
        let loss: Option<Box<dyn ceres::LossFunction>> = if k % 2 == 0 {
            Some(Box::new(HuberLoss::new(1.0)))
        } else {
            None
        };
        problem.add_residual_block(
            Box::new(AutoDiffCostFunction::<_, 6>::new(
                SpaCostFunction2D {
                    measured: m,
                    translation_weight: 1e2,
                    rotation_weight: 1e2,
                },
                3,
                &[3, 3],
            )),
            loss,
            &[ids[i], ids[j]],
        );
    }
    problem.set_parameter_block_constant(ids[0]);
    (problem, ids)
}

fn solve_pose_graph_2d(
    graph: &PoseGraph2D,
    linear_solver_type: LinearSolverType,
    num_threads: i32,
) -> (Vec<[f64; 3]>, SolverSummary) {
    let (mut problem, ids) = build_pose_graph_problem_2d(graph, &graph.initial);
    let options = SolverOptions {
        linear_solver_type,
        num_threads,
        max_num_iterations: 100,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    let poses = ids
        .iter()
        .map(|&id| {
            let p = problem.parameter_block(id);
            [p[0], p[1], p[2]]
        })
        .collect();
    (poses, summary)
}

#[test]
fn pose_graph_2d_all_linear_solvers_agree() {
    let graph = make_pose_graph_2d(90, 7);
    let (sparse, sparse_summary) =
        solve_pose_graph_2d(&graph, LinearSolverType::SparseNormalCholesky, 1);
    assert_eq!(
        sparse_summary.termination_type,
        TerminationType::Convergence,
        "{}",
        sparse_summary.full_report()
    );
    assert!(
        sparse_summary.final_cost < 2e-3 * sparse_summary.initial_cost,
        "{}",
        sparse_summary.full_report()
    );
    assert_eq!(sparse_summary.num_parameter_blocks_reduced, 89);
    assert_eq!(sparse_summary.fixed_cost, 0.0);
    for t in [
        LinearSolverType::DenseQr,
        LinearSolverType::DenseNormalCholesky,
    ] {
        let (dense, summary) = solve_pose_graph_2d(&graph, t, 1);
        assert_eq!(summary.termination_type, TerminationType::Convergence);
        assert!(
            (summary.final_cost - sparse_summary.final_cost).abs()
                <= 1e-6 * sparse_summary.final_cost.max(1e-12),
            "{t}: {} vs {}",
            summary.final_cost,
            sparse_summary.final_cost
        );
        for (a, b) in dense.iter().zip(&sparse) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-6, "{t}: {a:?} vs {b:?}");
            }
        }
    }
    // The optimum must not be worse than the ground truth trajectory (which
    // is not the optimum because of the measurement noise), and the
    // optimized trajectory is closer to the truth than the odometry.
    let (truth_problem, _) = build_pose_graph_problem_2d(&graph, &graph.truth);
    let mut truth_cost = 0.0;
    assert!(truth_problem.evaluate(
        &ceres::EvaluateOptions::default(),
        Some(&mut truth_cost),
        None,
        None,
        None
    ));
    assert!(
        sparse_summary.final_cost <= truth_cost,
        "{} vs {truth_cost}",
        sparse_summary.final_cost
    );
    let error = |poses: &[[f64; 3]]| -> f64 {
        poses
            .iter()
            .zip(&graph.truth)
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .sum::<f64>()
    };
    assert!(error(&sparse) < error(&graph.initial));
    // The first node was constant.
    assert_eq!(sparse[0], graph.initial[0]);
}

#[test]
fn pose_graph_2d_results_are_independent_of_num_threads() {
    let graph = make_pose_graph_2d(120, 11);
    let (single, s1) = solve_pose_graph_2d(&graph, LinearSolverType::SparseNormalCholesky, 1);
    for threads in [2, 3, 8] {
        let (multi, s) =
            solve_pose_graph_2d(&graph, LinearSolverType::SparseNormalCholesky, threads);
        // Bit-for-bit identical.
        assert_eq!(single, multi, "num_threads = {threads}");
        assert_eq!(s1.final_cost.to_bits(), s.final_cost.to_bits());
        assert_eq!(s1.iterations.len(), s.iterations.len());
        assert_eq!(s.num_threads_used, threads);
    }
}

#[test]
fn pose_graph_2d_large_sparse_problem() {
    // ~10^4 parameter blocks of size 3, as in Cartographer pose graphs.
    let graph = make_pose_graph_2d(10_000, 3);
    let start = std::time::Instant::now();
    let (poses, summary) = solve_pose_graph_2d(&graph, LinearSolverType::SparseNormalCholesky, 4);
    let elapsed = start.elapsed().as_secs_f64();
    assert!(summary.is_solution_usable(), "{}", summary.full_report());
    assert!(
        summary.final_cost < 1e-2 * summary.initial_cost,
        "{}",
        summary.full_report()
    );
    assert_eq!(poses.len(), 10_000);
    assert!(
        elapsed < 120.0,
        "took {elapsed}s\n{}",
        summary.full_report()
    );
    let report = summary.full_report();
    assert!(report.contains("SPARSE_NORMAL_CHOLESKY"));
    assert!(
        report.contains("Parameter blocks                        10000                     9999"),
        "{report}"
    );
}

// ---------------------------------------------------------------------------
// 3D: quaternion parameterization and autodiff local parameterization.
// ---------------------------------------------------------------------------

/// Residual between a rotated point and its measurement; quaternion in Ceres
/// [w, x, y, z] order.
struct RotatedPoint {
    point: Vector3<f64>,
    measured: Vector3<f64>,
}

impl CostFunctor for RotatedPoint {
    fn evaluate<T: Scalar>(&self, p: &[&[T]], e: &mut [T]) -> bool {
        let q = p[0];
        let rotation =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]));
        let rotated = rotation * self.point.map(T::constant);
        let r = rotated - self.measured.map(T::constant);
        e.copy_from_slice(r.as_slice());
        true
    }
}

fn solve_rotation(
    parameterization: Option<Box<dyn ceres::LocalParameterization>>,
) -> ([f64; 4], UnitQuaternion<f64>, SolverSummary) {
    let truth = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.7);
    let mut problem = Problem::new();
    let q =
        problem.add_parameter_block_with_parameterization(&[1.0, 0.0, 0.0, 0.0], parameterization);
    for point in [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 2.0, 0.5),
        Vector3::new(-1.0, 0.3, 1.0),
    ] {
        problem.add_residual_block(
            Box::new(AutoDiffCostFunction::<_, 4>::new(
                RotatedPoint {
                    point,
                    measured: truth * point,
                },
                3,
                &[4],
            )),
            None,
            &[q],
        );
    }
    let options = SolverOptions {
        linear_solver_type: LinearSolverType::DenseQr,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    let v = problem.parameter_block(q);
    ([v[0], v[1], v[2], v[3]], truth, summary)
}

#[test]
fn quaternion_parameterization_recovers_rotation() {
    let (q, truth, summary) = solve_rotation(Some(Box::new(QuaternionParameterization)));
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    assert_eq!(summary.num_effective_parameters_reduced, 3);
    let estimate =
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]));
    assert!(estimate.angle_to(&truth) < 1e-8);
    // The parameterization keeps the quaternion normalized.
    assert!((q.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
}

/// Quaternion plus via autodiff (same as QuaternionParameterization).
struct QuaternionPlus;
impl PlusFunctor for QuaternionPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        // As in Ceres' autodiff_local_parameterization_test.cc: away from zero
        // use the exact map, at zero the first order Taylor expansion (so that
        // the jacobian at delta = 0 is correct).
        let squared_norm_delta = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        let q_delta = if squared_norm_delta > T::constant(0.0) {
            let norm_delta = squared_norm_delta.sqrt();
            let sin_delta_by_delta = norm_delta.sin() / norm_delta;
            [
                norm_delta.cos(),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2],
            ]
        } else {
            [T::constant(1.0), delta[0], delta[1], delta[2]]
        };
        ceres::rotation::quaternion_product(&q_delta, x, x_plus_delta);
        true
    }
}

#[test]
fn autodiff_local_parameterization_recovers_rotation() {
    let (q, truth, summary) =
        solve_rotation(Some(Box::new(
            AutoDiffLocalParameterization::<_, 4, 3>::new(QuaternionPlus),
        )));
    assert_eq!(summary.termination_type, TerminationType::Convergence);
    let estimate =
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]));
    assert!(estimate.angle_to(&truth) < 1e-8);
}

#[test]
fn full_report_format() {
    let (mut problem, _) = powell_problem();
    let options = SolverOptions {
        linear_solver_type: LinearSolverType::DenseQr,
        logging_type: LoggingType::Silent,
        ..SolverOptions::default()
    };
    let mut summary = SolverSummary::default();
    solve(&options, &mut problem, &mut summary);
    let report = summary.full_report();
    for line in [
        "                                     Original                  Reduced",
        // Expected lines computed with the C++ printf formats of solver.cc.
        "Parameter blocks                            4                        4",
        "Parameters                                  4                        4",
        "Residual blocks                             4                        4",
        "Residuals                                   4                        4",
        "\nMinimizer                        TRUST_REGION",
        "\nDense linear algebra library            EIGEN",
        "Trust region strategy     LEVENBERG_MARQUARDT",
        "Linear solver                        DENSE_QR                 DENSE_QR",
        "Threads                                     1                        1",
        "Initial                          1.075000e+02",
        "Termination:                      CONVERGENCE (",
    ] {
        assert!(report.contains(line), "missing {line:?} in\n{report}");
    }
}
