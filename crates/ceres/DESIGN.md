# cartographer-ceres: design notes

Pure-Rust port of the subset of [Ceres Solver](http://ceres-solver.org/)
**2.0.0** used by Cartographer (2.0.0 because Cartographer still uses
`LocalParameterization`). Every ported Rust file cites its upstream source
at the top.

## Sparse linear solver: faer vs. a hand-written block LDLᵀ

`SPARSE_NORMAL_CHOLESKY` (the pose graph default) needs a fill-reducing
ordering and a sparse Cholesky factorization that scales to ~10⁴ parameter
blocks (3 or 4+3 parameters each). Options considered:

* **faer 0.24** (`faer::sparse::linalg::cholesky`): pure Rust, actively
  maintained (frequent releases in 2024–2026), AMD ordering built in,
  supernodal *and* simplicial LLᵀ chosen automatically by flop count,
  symbolic analysis separable from numeric factorization (so the analysis
  is done once per solve and reused every iteration, like CHOLMOD in Ceres).
  Supernodal factorization uses dense BLAS-3 kernels for the dense
  separators that appear in loop-closed pose graphs.
* **Writing a compact block-sparse LDLᵀ**: fewer dependencies, but a
  production-quality minimum degree ordering (AMD is ~1.5k lines in
  SuiteSparse) plus an efficient (ideally supernodal) factorization is a
  substantial amount of numerically delicate code to write and maintain,
  and a simplicial implementation would be markedly slower on graphs with
  many loop closures.

**Choice: faer** (`default-features = false`, features `std` +
`sparse-linalg`; no rayon). It is better maintained and faster than
anything reasonable to hand-write here. The normal equations are never
densified: the upper triangle of `JᵀJ + DᵀD` is assembled directly from the
block structure of the Jacobian into a CSC matrix whose pattern (and the
mapping of Jacobian block pairs to value positions) is computed once
per solve. A synthetic 2D pose graph with 10 000 nodes / 11 333 constraints
solves in ~0.2 s (29 LM iterations) in the test profile.

Ceres+SuiteSparse computes a block AMD ordering of the parameter blocks and
reorders the program; here faer computes a scalar AMD ordering of `JᵀJ`.
Both are fill-reducing orderings of the same matrix; results differ only
by floating point rounding.

`DENSE_QR` and `DENSE_NORMAL_CHOLESKY` use nalgebra (Householder QR of
`[J; D]`, and LLᵀ of `JᵀJ + D²`), mirroring the Eigen code paths of Ceres.

## Ported Ceres files

| Rust file | Ceres 2.0.0 source |
|---|---|
| `jet.rs` | `include/ceres/jet.h` (+ num-traits / simba / nalgebra trait impls) |
| `scalar.rs` | — (the `T: Scalar` bound replacing `template <typename T>`) |
| `rotation.rs` | `include/ceres/rotation.h` (complete) |
| `cubic_interpolation.rs` | `include/ceres/cubic_interpolation.h` (complete) |
| `cost_function.rs` | `include/ceres/cost_function.h`, `sized_cost_function.h` |
| `autodiff_cost_function.rs` | `include/ceres/autodiff_cost_function.h`, `internal/autodiff.h` |
| `loss_function.rs` | `include/ceres/loss_function.h`, `internal/ceres/loss_function.cc` |
| `local_parameterization.rs` | `include/ceres/local_parameterization.h`, `internal/ceres/local_parameterization.cc`, `internal/householder_vector.h` |
| `autodiff_local_parameterization.rs` | `include/ceres/autodiff_local_parameterization.h` |
| `corrector.rs` | `internal/ceres/corrector.{h,cc}` |
| `residual_block.rs` | `internal/ceres/residual_block.cc`, `residual_block_utils.cc`, `array_utils.cc` |
| `problem.rs` | `include/ceres/problem.h`, `internal/ceres/problem_impl.cc`, `parameter_block.h`, `include/ceres/crs_matrix.h` |
| `program.rs` | `internal/ceres/program.cc`, `program_evaluator.h`, `block_sparse_matrix.cc`, `block_jacobian_writer.cc` |
| `linear_solver.rs` | `internal/ceres/linear_solver.h`, `dense_qr_solver.cc`, `dense_normal_cholesky_solver.cc`, `sparse_normal_cholesky_solver.cc`, `inner_product_computer.cc` |
| `trust_region.rs` | `internal/ceres/levenberg_marquardt_strategy.cc`, `trust_region_strategy.h`, `trust_region_step_evaluator.cc` |
| `trust_region_minimizer.rs` | `internal/ceres/trust_region_minimizer.cc` (incl. `DoLineSearch` and the projection in `IterationZero`), `minimizer.cc`, `callbacks.cc` (logging callback) |
| `line_search.rs` | `internal/ceres/line_search.{h,cc}` (`LineSearchFunction`, `InterpolatingPolynomialMinimizingStepSize`, `ArmijoLineSearch`) |
| `polynomial.rs` | `internal/ceres/polynomial.{h,cc}`, `function_sample.{h,cc}` |
| `solver.rs` | `include/ceres/solver.h`, `iteration_callback.h`, `internal/ceres/solver.cc`, `solver_utils.{h,cc}`, `preprocessor.cc`, `trust_region_preprocessor.cc` |
| `types.rs` | `include/ceres/types.h`, `internal/ceres/types.cc` |
| `stringprintf.rs` | `internal/ceres/stringprintf.cc` (only the printf conversions needed for reports) |

Ported tests: `jet_test`, `rotation_test`, `cubic_interpolation_test`,
`loss_function_test`, `local_parameterization_test`,
`autodiff_local_parameterization_test`, `autodiff_cost_function_test`,
`corrector_test`, `problem_test`, `solver_test`, `polynomial_test`
(in `polynomial.rs`), the bound tests of `parameter_block_test`,
`program_test` and `trust_region_preprocessor_test`, and
`trust_region_minimizer_test` (where applicable), plus the `powell`,
`curve_fitting` and `robust_curve_fitting` examples, bounded curve fitting
and pose graph tests. Ceres 2.0.0 has no `line_search_test`; the Armijo
search has its own unit tests in `line_search.rs`.

## Parameter bounds

`Problem::set_parameter_{lower,upper}_bound` / `get_parameter_*_bound`
store bounds per parameter block, lazily allocated, default `∓f64::MAX`
(`parameter_block.h`). `ParameterBlock::plus` projects onto the box after
the local parameterization's `Plus`, and `Program::plus` (used by the
evaluator, the minimizer and the line search) goes through it. The
preprocessor runs `Program::IsFeasible` (constant blocks must satisfy their
bounds, variable blocks need `lower < upper`) and sets `is_constrained`
from `Program::IsBoundsConstrained` on the reduced program. The trust region
minimizer then projects the initial point and, if
`max_num_line_search_step_size_iterations > 0`, runs the projected Armijo
line search (`DoLineSearch`) on every valid step, with the 2.0.0 line search
option defaults (CUBIC interpolation, `min_line_search_step_contraction =
0.6`, ...). Summary fields `is_constrained`, `num_line_search_steps` and
the line search timings are filled, and `full_report` prints the line
search lines like Ceres.

As upstream notes in `DoLineSearch` ("incorrect and not terribly
effective", ceres-solver issue #187), this scheme converges slowly when a
bound is active and the variables are coupled: the step quality uses the
unprojected model decrease and the Armijo test the unprojected directional
derivative. The port reproduces that behaviour rather than improving on it
(see the bounded curve fitting tests).

Numerical differences from Eigen: `FindPolynomialRoots` uses nalgebra's
real Schur decomposition for the companion matrix eigenvalues and deflates
exact zero roots (trailing zero coefficients) first, since nalgebra does
not resolve the nilpotent companion matrix exactly; `FullPivLU` with
threshold 0 is reimplemented to match Eigen on rank deficient systems.

## Structural changes (RUST PORT NOTEs)

* **Handle-based `Problem`.** Ceres optimizes the user's `double*` memory
  in place and identifies blocks by pointer. Here `Problem` owns a copy of
  each parameter block: `add_parameter_block(&[f64]) -> ParameterBlockId`,
  residual blocks refer to ids, and results are read back with
  `problem.parameter_block(id)`. Parameter blocks must be added before the
  residual blocks using them. Ownership flags are replaced by `Box`/`Arc`.
  `Problem<'a>` carries a lifetime so cost functions may borrow data (point
  clouds, grids) like Cartographer's functors hold references.
* **Autodiff arity.** `AutoDiffCostFunction<F, N>` is parameterized by the
  total parameter count `N` (the jet dimension); residual count and block
  sizes are constructor arguments (so `ceres::DYNAMIC` is just a runtime
  count). Functors implement `CostFunctor::evaluate<T: Scalar>(&self,
  &[&[T]], &mut [T])`. The hot path does not allocate: parameters live in a
  stack array, ≤ 20 residual jets on the stack, larger residual vectors
  (scan matching) reuse a per-cost-function scratch buffer.
* **`AutoDiffLocalParameterization<F, GLOBAL, LOCAL>`** seeds only `delta`
  (jets of dimension `LOCAL`) instead of `x` and `delta`; the Jacobian is
  identical.
* **Determinism with threads.** `num_threads > 1` evaluates contiguous
  chunks of residual blocks with `std::thread::scope`. Per-block costs are
  stored separately and reduced sequentially, and the gradient is computed
  as `Jᵀr` afterwards, so results are bit-for-bit independent of the thread
  count (Ceres sums per-thread partial results, which is schedule
  dependent).
* **`Scalar` helpers.** `T::constant(x)` replaces C++ `T(x)`;
  `x.scalar_part()` replaces `x.a`. Cubic interpolators have one generic
  `evaluate` that reproduces the C++ `double`/`Jet` overloads exactly.
* **Nested types.** `Solver::Options`, `Solver::Summary`,
  `Problem::Options`, `Problem::EvaluateOptions` become `SolverOptions`,
  `SolverSummary`, `ProblemOptions`, `EvaluateOptions`. Enumerators are
  CamelCase; `Display` prints the Ceres names (`DENSE_QR`, ...).
* **Logging.** `VLOG(1)` progress goes to `log::debug!` (or stdout with
  `minimizer_progress_to_stdout`), warnings to `log::warn!`.
* **One `unsafe` token.** `nalgebra::RealField` requires `simba::SimdValue`,
  which declares `unsafe fn extract_unchecked/replace_unchecked`.
  Implementing them for `Jet` requires `#[allow(unsafe_code)]` on those two
  methods; their bodies contain no unsafe operation.

## Dependencies

`nalgebra` 0.35, `simba` 0.10 / `num-traits` / `approx` (the versions used
by nalgebra 0.35), `faer` 0.24 (sparse Cholesky), `libm` (pure Rust,
already a dependency of faer: `erf`, `erfc`, Bessel `j0/j1/jn` for
`jet.h`), `log`. Dev: `proptest`, `rand`.

## Intentionally omitted

* Line search minimizer (LBFGS/BFGS/NCG, Wolfe line search,
  `line_search_direction_type`, `max_lbfgs_rank`, ...), DOGLEG strategy, inner
  iterations (coordinate descent), Schur and iterative linear solvers
  (`DENSE_SCHUR`, `SPARSE_SCHUR`, `ITERATIVE_SCHUR`, `CGNR`),
  preconditioners, linear solver orderings, `dynamic_sparsity`, mixed
  precision / iterative refinement.
* `EvaluationCallback`, `update_state_every_iteration`, gradient checking
  (`check_gradients`), trust region problem dumping, `Context`.
* `NumericDiffCostFunction`, `DynamicAutoDiffCostFunction`,
  `CostFunctionToFunctor`, `ConditionedCostFunction`, `GradientProblem`,
  `Covariance`, `LineParameterization`, `Manifold` (Ceres ≥ 2.1).
* Nested jets (`Jet<Jet<..>>`): `Jet<N>` stores `f64` only.

None of these are used by Cartographer.
