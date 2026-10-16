//! A focused, pure-Rust port of the subset of [Ceres Solver] 2.0.0 used by
//! Google Cartographer.
//!
//! [Ceres Solver]: http://ceres-solver.org/
//!
//! Contents:
//!
//! * [`Jet`]: dual numbers for forward-mode automatic differentiation, with
//!   the `jet.h` math functions and the `num-traits` / `nalgebra` numeric
//!   traits, so nalgebra types and generic code (`T: `[`Scalar`]) work with
//!   jets like Eigen works with `ceres::Jet`.
//! * [`rotation`] (`rotation.h`) and [`cubic_interpolation`]
//!   (`cubic_interpolation.h`), generic over [`Scalar`].
//! * Cost functions: [`CostFunction`], [`SizedCostFunction`],
//!   [`AutoDiffCostFunction`] (with a [`CostFunctor`]).
//! * Loss functions ([`HuberLoss`], [`TolerantLoss`], ...) and local
//!   parameterizations ([`QuaternionParameterization`],
//!   [`SubsetParameterization`], [`AutoDiffLocalParameterization`], ...).
//! * [`Problem`] and [`solve`]: trust region minimization with
//!   Levenberg-Marquardt and `DENSE_QR`, `DENSE_NORMAL_CHOLESKY` or
//!   `SPARSE_NORMAL_CHOLESKY` linear solvers.
//!
//! # Example
//!
//! Fit `y = exp(m x + c)` to noisy data (Ceres' `curve_fitting.cc`):
//!
//! ```
//! use ceres::{
//!     AutoDiffCostFunction, CostFunctor, CauchyLoss, Problem, Scalar, SolverOptions,
//!     SolverSummary, LinearSolverType, solve,
//! };
//!
//! struct ExponentialResidual { x: f64, y: f64 }
//!
//! impl CostFunctor for ExponentialResidual {
//!     fn evaluate<T: Scalar>(&self, p: &[&[T]], residual: &mut [T]) -> bool {
//!         let (m, c) = (p[0][0], p[1][0]);
//!         residual[0] = T::constant(self.y) - (m * T::constant(self.x) + c).exp();
//!         true
//!     }
//! }
//!
//! let mut problem = Problem::new();
//! let m = problem.add_parameter_block(&[0.0]);
//! let c = problem.add_parameter_block(&[0.0]);
//! for i in 0..50 {
//!     let x = f64::from(i) / 10.0;
//!     let y = (0.3 * x + 0.1).exp();
//!     problem.add_residual_block(
//!         // 1 residual; two parameter blocks of size 1 => N = 2.
//!         Box::new(AutoDiffCostFunction::<_, 2>::new(ExponentialResidual { x, y }, 1, &[1, 1])),
//!         Some(Box::new(CauchyLoss::new(0.5))),
//!         &[m, c],
//!     );
//! }
//!
//! let mut options = SolverOptions::default();
//! options.linear_solver_type = LinearSolverType::DenseQr;
//! let mut summary = SolverSummary::default();
//! solve(&options, &mut problem, &mut summary);
//! println!("{}", summary.brief_report());
//! assert!(summary.is_solution_usable());
//! assert!((problem.parameter_block(m)[0] - 0.3).abs() < 1e-6);
//! assert!((problem.parameter_block(c)[0] - 0.1).abs() < 1e-6);
//! ```
//!
//! Generic code using nalgebra types runs unchanged on `f64` and on jets:
//!
//! ```
//! use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
//! use nalgebra::{Rotation2, Vector2};
//!
//! /// Residual between a 2D pose (x, y, theta) applied to `point` and `target`.
//! struct PointToPoint { point: Vector2<f64>, target: Vector2<f64> }
//!
//! impl CostFunctor for PointToPoint {
//!     fn evaluate<T: Scalar>(&self, p: &[&[T]], r: &mut [T]) -> bool {
//!         let pose = p[0];
//!         let rotation = Rotation2::new(pose[2]);
//!         let translation = Vector2::new(pose[0], pose[1]);
//!         let point: Vector2<T> = self.point.map(T::constant);
//!         let world = rotation * point + translation;
//!         let e = world - self.target.map(T::constant);
//!         r[0] = e.x;
//!         r[1] = e.y;
//!         true
//!     }
//! }
//!
//! let f = AutoDiffCostFunction::<_, 3>::new(
//!     PointToPoint { point: Vector2::new(1.0, 0.0), target: Vector2::new(0.0, 1.0) }, 2, &[3]);
//! let mut r = [0.0; 2];
//! let mut j = [0.0; 6];
//! assert!(f.evaluate(&[&[0.0, 0.0, 0.0]], &mut r, Some(&mut [Some(&mut j[..])])));
//! // d(world.y)/d(theta) at theta = 0 is point.x = 1.
//! assert_eq!(j[5], 1.0);
//! ```

// Index loops mirror the C++ numerical code one-to-one.
#![allow(clippy::needless_range_loop)]

pub mod autodiff_cost_function;
pub mod autodiff_local_parameterization;
mod corrector;
pub mod cost_function;
pub mod cubic_interpolation;
pub mod jet;
mod linear_solver;
pub mod local_parameterization;
pub mod loss_function;
pub mod problem;
mod program;
mod residual_block;
pub mod rotation;
pub mod scalar;
pub mod solver;
mod stringprintf;
mod trust_region;
mod trust_region_minimizer;
pub mod types;

pub use autodiff_cost_function::{AutoDiffCostFunction, CostFunctor};
pub use autodiff_local_parameterization::{AutoDiffLocalParameterization, PlusFunctor};
pub use cost_function::{CostFunction, SizedCostFunction};
pub use cubic_interpolation::{
    BiCubicInterpolator, CubicInterpolator, Grid1, Grid1D, Grid2, Grid2D, cubic_hermite_spline,
};
pub use jet::Jet;
pub use local_parameterization::{
    EigenQuaternionParameterization, HomogeneousVectorParameterization, IdentityParameterization,
    LocalParameterization, ProductParameterization, QuaternionParameterization,
    SubsetParameterization,
};
pub use loss_function::{
    ArctanLoss, CauchyLoss, ComposedLoss, HuberLoss, LossFunction, LossFunctionWrapper, ScaledLoss,
    SoftLOneLoss, TolerantLoss, TrivialLoss, TukeyLoss,
};
pub use problem::{
    CrsMatrix, EvaluateOptions, ParameterBlockId, Problem, ProblemOptions, ResidualBlockId,
};
pub use scalar::Scalar;
pub use solver::{IterationCallback, IterationSummary, SolverOptions, SolverSummary, solve};
pub use types::{
    CallbackReturnType, DenseLinearAlgebraLibraryType, LinearSolverType, LoggingType,
    MinimizerType, SparseLinearAlgebraLibraryType, TerminationType, TrustRegionStrategyType,
};
