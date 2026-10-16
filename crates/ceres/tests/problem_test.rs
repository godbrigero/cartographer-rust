// Ported from Ceres Solver 2.0.0: internal/ceres/problem_test.cc
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sameeragarwal@google.com (Sameer Agarwal),
// keir@google.com (Keir Mierle)
//
// RUST PORT NOTES:
// - The Rust `Problem` is handle based (`ParameterBlockId` /
//   `ResidualBlockId`) instead of identifying parameter blocks by `double*`.
//   Parameter blocks must therefore be added explicitly before the residual
//   blocks that use them, and "unknown pointer" death tests use a handle that
//   does not belong to the problem.
// - The `DynamicProblem` `TEST_P` fixture (parameterized on
//   `enable_fast_removal`) collapses to a single variant, since removal is
//   always supported. Checks of internal `ProblemImpl` / `Program` state
//   (back-pointers, residual block hash sets, `Program::IsValid`) are replaced
//   by the equivalent public queries.
// - EXPECT_DEATH tests become `#[should_panic]` tests (one per death).

use std::sync::Arc;

use ceres::{
    AutoDiffCostFunction, CostFunction, CostFunctor, CrsMatrix, EvaluateOptions,
    IdentityParameterization, LocalParameterization, LossFunction, ParameterBlockId, Problem,
    QuaternionParameterization, ResidualBlockId, Scalar, ScaledLoss, SubsetParameterization,
    TrivialLoss,
};
use nalgebra::{DMatrix, DVector};

// The following three cost functions are for the purposes of defining
// function signatures. They have dummy evaluate functions.

/// Trivial cost function that accepts a single argument.
struct UnaryCostFunction {
    num_residuals: usize,
    parameter_block_sizes: Vec<usize>,
}

impl UnaryCostFunction {
    fn new(num_residuals: usize, parameter_block_size: usize) -> Self {
        Self {
            num_residuals,
            parameter_block_sizes: vec![parameter_block_size],
        }
    }
}

impl CostFunction for UnaryCostFunction {
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        residuals: &mut [f64],
        _jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        residuals.fill(1.0);
        true
    }
    fn num_residuals(&self) -> usize {
        self.num_residuals
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }
}

/// Trivial cost function that accepts two arguments.
struct BinaryCostFunction {
    num_residuals: usize,
    parameter_block_sizes: Vec<usize>,
}

impl BinaryCostFunction {
    fn new(num_residuals: usize, size1: usize, size2: usize) -> Self {
        Self {
            num_residuals,
            parameter_block_sizes: vec![size1, size2],
        }
    }
}

impl CostFunction for BinaryCostFunction {
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        residuals: &mut [f64],
        _jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        residuals.fill(2.0);
        true
    }
    fn num_residuals(&self) -> usize {
        self.num_residuals
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }
}

/// Trivial cost function that accepts three arguments.
struct TernaryCostFunction {
    num_residuals: usize,
    parameter_block_sizes: Vec<usize>,
}

impl TernaryCostFunction {
    fn new(num_residuals: usize, size1: usize, size2: usize, size3: usize) -> Self {
        Self {
            num_residuals,
            parameter_block_sizes: vec![size1, size2, size3],
        }
    }
}

impl CostFunction for TernaryCostFunction {
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        residuals: &mut [f64],
        _jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        residuals.fill(3.0);
        true
    }
    fn num_residuals(&self) -> usize {
        self.num_residuals
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }
}

/// Returns a parameter block handle that does not belong to a problem with
/// fewer than two parameter blocks (stand-in for an unknown `double*`).
fn foreign_parameter_block_id() -> ParameterBlockId {
    let mut other = Problem::new();
    other.add_parameter_block(&[0.0]);
    other.add_parameter_block(&[0.0])
}

// Skipped: Problem.MoveConstructor / Problem.MoveAssignment (C++ move
// semantics; Rust moves are always valid).

// Skipped: Problem.AddResidualWithNullCostFunctionDies (a null cost function
// cannot be expressed with `Box<dyn CostFunction>`).

#[test]
#[should_panic(expected = "Number of blocks input is different")]
fn add_residual_with_incorrect_number_of_parameter_blocks_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);
    let _z = problem.add_parameter_block(&[0.0; 5]);

    // UnaryCostFunction takes only one parameter, but two are passed.
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 3)), None, &[x, y]);
}

#[test]
#[should_panic(expected = "expects parameter block 0 of size 4")]
fn add_residual_with_different_sizes_on_the_same_variable_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 3)), None, &[x]);
    problem.add_residual_block(
        Box::new(UnaryCostFunction::new(2, 4 /* 4 != 3 */)),
        None,
        &[x],
    );
}

#[test]
#[should_panic(expected = "Duplicate parameter blocks")]
fn add_residual_with_duplicate_parameters_dies_binary() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    problem.add_residual_block(Box::new(BinaryCostFunction::new(2, 3, 3)), None, &[x, x]);
}

#[test]
#[should_panic(expected = "Duplicate parameter blocks")]
fn add_residual_with_duplicate_parameters_dies_ternary() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let z = problem.add_parameter_block(&[0.0; 5]);
    problem.add_residual_block(
        Box::new(TernaryCostFunction::new(1, 5, 3, 5)),
        None,
        &[z, x, z],
    );
}

#[test]
#[should_panic(expected = "expects parameter block 1 of size 4")]
fn add_residual_with_incorrect_sizes_of_parameter_block_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let _y = problem.add_parameter_block(&[0.0; 4]);
    let z = problem.add_parameter_block(&[0.0; 5]);

    // The cost function expects the size of the second parameter, z, to be 4
    // instead of 5 as declared above. This is fatal.
    problem.add_residual_block(Box::new(BinaryCostFunction::new(2, 3, 4)), None, &[x, z]);
}

// Adapted to the handle API: using the same handle in several residual blocks
// does not create new parameter blocks.
#[test]
fn add_residual_adds_duplicated_parameters_only_once() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);
    let z = problem.add_parameter_block(&[0.0; 5]);
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 3)), None, &[x]);
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 3)), None, &[x]);
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 4)), None, &[y]);
    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 5)), None, &[z]);

    assert_eq!(problem.num_parameter_blocks(), 3);
    assert_eq!(problem.num_parameters(), 12);
}

// Skipped: Problem.AddParameterWithDifferentSizesOnTheSameVariableDies,
// Problem.AddParameterWithAliasedParametersDies and
// Problem.AddParameterIgnoresDuplicateCalls (they rely on `double*` identity /
// memory aliasing; in the handle API every add creates a distinct block).

#[test]
fn adding_parameters_and_residuals_results_in_expected_problem() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    assert_eq!(problem.num_parameter_blocks(), 1);
    assert_eq!(problem.num_parameters(), 3);

    let y = problem.add_parameter_block(&[0.0; 4]);
    assert_eq!(problem.num_parameter_blocks(), 2);
    assert_eq!(problem.num_parameters(), 7);

    let z = problem.add_parameter_block(&[0.0; 5]);
    assert_eq!(problem.num_parameter_blocks(), 3);
    assert_eq!(problem.num_parameters(), 12);

    // Add a parameter that has a local parameterization.
    let w = [1.0, 0.0, 0.0, 0.0];
    let _w = problem
        .add_parameter_block_with_parameterization(&w, Some(Box::new(QuaternionParameterization)));
    assert_eq!(problem.num_parameter_blocks(), 4);
    assert_eq!(problem.num_parameters(), 16);

    problem.add_residual_block(Box::new(UnaryCostFunction::new(2, 3)), None, &[x]);
    problem.add_residual_block(Box::new(BinaryCostFunction::new(6, 5, 4)), None, &[z, y]);
    problem.add_residual_block(Box::new(BinaryCostFunction::new(3, 3, 5)), None, &[x, z]);
    problem.add_residual_block(Box::new(BinaryCostFunction::new(7, 5, 3)), None, &[z, x]);
    problem.add_residual_block(
        Box::new(TernaryCostFunction::new(1, 5, 3, 4)),
        None,
        &[z, x, y],
    );

    let total_residuals = 2 + 6 + 3 + 7 + 1;
    assert_eq!(problem.num_residual_blocks(), 5);
    assert_eq!(problem.num_residuals(), total_residuals);
}

// Skipped: Problem.ReusedCostFunctionsAreOnlyDeletedOnce and
// Problem.CostFunctionsAreDeletedEvenWithRemovals (destructor counting /
// ownership; handled by Rust ownership and `Arc`).

#[test]
fn get_cost_function_for_residual_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let cost_function: Box<dyn CostFunction> = Box::new(UnaryCostFunction::new(2, 3));
    let cost_function_ptr: *const dyn CostFunction = &*cost_function;
    let residual_block = problem.add_residual_block(cost_function, None, &[x]);
    assert!(std::ptr::addr_eq(
        problem.cost_function_for_residual_block(residual_block) as *const dyn CostFunction,
        cost_function_ptr
    ));
    assert!(
        problem
            .loss_function_for_residual_block(residual_block)
            .is_none()
    );
}

#[test]
fn get_loss_function_for_residual_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let cost_function: Box<dyn CostFunction> = Box::new(UnaryCostFunction::new(2, 3));
    let cost_function_ptr: *const dyn CostFunction = &*cost_function;
    let loss_function: Box<dyn LossFunction> = Box::new(TrivialLoss);
    let loss_function_ptr: *const dyn LossFunction = &*loss_function;
    let residual_block = problem.add_residual_block(cost_function, Some(loss_function), &[x]);
    assert!(std::ptr::addr_eq(
        problem.cost_function_for_residual_block(residual_block) as *const dyn CostFunction,
        cost_function_ptr
    ));
    let loss = problem
        .loss_function_for_residual_block(residual_block)
        .expect("loss function should be set");
    assert!(std::ptr::addr_eq(
        loss as *const dyn LossFunction,
        loss_function_ptr
    ));
    // TrivialLoss is zero sized, so additionally check its behaviour.
    let mut rho = [0.0; 3];
    loss.evaluate(2.5, &mut rho);
    assert_eq!(rho, [2.5, 1.0, 0.0]);
}

#[test]
#[should_panic(expected = "Parameter block not found:")]
fn set_parameter_block_constant_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    problem.set_parameter_block_constant(foreign_parameter_block_id());
}

#[test]
#[should_panic(expected = "Parameter block not found:")]
fn set_parameter_block_variable_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    problem.set_parameter_block_variable(foreign_parameter_block_id());
}

#[test]
fn is_parameter_block_constant() {
    let mut problem = Problem::new();
    let x1 = problem.add_parameter_block(&[0.0; 3]);
    let x2 = problem.add_parameter_block(&[0.0; 3]);

    assert!(!problem.is_parameter_block_constant(x1));
    assert!(!problem.is_parameter_block_constant(x2));

    problem.set_parameter_block_constant(x1);
    assert!(problem.is_parameter_block_constant(x1));
    assert!(!problem.is_parameter_block_constant(x2));

    problem.set_parameter_block_constant(x2);
    assert!(problem.is_parameter_block_constant(x1));
    assert!(problem.is_parameter_block_constant(x2));

    problem.set_parameter_block_variable(x1);
    assert!(!problem.is_parameter_block_constant(x1));
    assert!(problem.is_parameter_block_constant(x2));
}

#[test]
#[should_panic(expected = "Parameter block not found:")]
fn is_parameter_block_constant_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    problem.is_parameter_block_constant(foreign_parameter_block_id());
}

#[test]
#[should_panic(expected = "Parameter block not found:")]
fn set_local_parameterization_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    problem.set_parameterization(
        foreign_parameter_block_id(),
        Some(Box::new(IdentityParameterization::new(3))),
    );
}

#[test]
#[should_panic(expected = "Parameter block not found:")]
fn remove_parameter_block_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    problem.remove_parameter_block(foreign_parameter_block_id());
}

#[test]
fn get_parameterization() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 2]);

    let parameterization: Box<dyn LocalParameterization> =
        Box::new(IdentityParameterization::new(3));
    let parameterization_ptr: *const dyn LocalParameterization = &*parameterization;
    problem.set_parameterization(x, Some(parameterization));
    let got = problem
        .get_parameterization(x)
        .expect("parameterization should be set");
    assert!(std::ptr::addr_eq(
        got as *const dyn LocalParameterization,
        parameterization_ptr
    ));
    assert!(problem.get_parameterization(y).is_none());
}

#[test]
fn parameter_block_query_test() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);

    let constant_parameters = [0];
    problem.set_parameterization(
        x,
        Some(Box::new(SubsetParameterization::new(
            3,
            &constant_parameters,
        ))),
    );
    assert_eq!(problem.parameter_block_size(x), 3);
    assert_eq!(problem.parameter_block_local_size(x), 2);
    assert_eq!(problem.parameter_block_local_size(y), 4);

    let parameter_blocks = problem.parameter_blocks();
    assert_eq!(parameter_blocks.len(), 2);
    assert_ne!(parameter_blocks[0], parameter_blocks[1]);
    assert!(parameter_blocks[0] == x || parameter_blocks[0] == y);
    assert!(parameter_blocks[1] == x || parameter_blocks[1] == y);

    assert!(problem.has_parameter_block(x));
    problem.remove_parameter_block(x);
    assert!(!problem.has_parameter_block(x));
    let parameter_blocks = problem.parameter_blocks();
    assert_eq!(parameter_blocks.len(), 1);
    assert_eq!(parameter_blocks[0], y);
}

// --- DynamicProblem ----------------------------------------------------------

/// Asserts that `actual` and `expected` contain the same elements, ignoring
/// order (C++ `ExpectVectorContainsUnordered`).
#[track_caller]
fn assert_contains_unordered<T: Ord + Copy + std::fmt::Debug>(actual: &[T], expected: &[T]) {
    let mut a = actual.to_vec();
    let mut b = expected.to_vec();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

/// The DynamicProblem fixture: y (size 4), z (size 5), w (size 3).
struct DynamicProblem {
    problem: Problem<'static>,
    y: ParameterBlockId,
    z: ParameterBlockId,
    w: ParameterBlockId,
}

/// All combinations of residual blocks over y, z, w.
struct AllResiduals {
    r_yzw: ResidualBlockId,
    r_yz: ResidualBlockId,
    r_yw: ResidualBlockId,
    r_zw: ResidualBlockId,
    r_y: ResidualBlockId,
    r_z: ResidualBlockId,
    r_w: ResidualBlockId,
}

impl DynamicProblem {
    fn new() -> Self {
        let mut problem = Problem::new();
        let y = problem.add_parameter_block(&[0.0; 4]);
        let z = problem.add_parameter_block(&[0.0; 5]);
        let w = problem.add_parameter_block(&[0.0; 3]);
        Self { problem, y, z, w }
    }

    fn has_residual_block(&self, id: ResidualBlockId) -> bool {
        self.problem.residual_blocks().contains(&id)
    }

    /// C++ `ExpectParameterBlockContains`: the set of residual blocks
    /// depending on `values` is exactly `expected`.
    #[track_caller]
    fn expect_parameter_block_contains(
        &self,
        values: ParameterBlockId,
        expected: &[ResidualBlockId],
    ) {
        assert_contains_unordered(
            &self.problem.residual_blocks_for_parameter_block(values),
            expected,
        );
    }

    fn add_all_residuals(&mut self) -> AllResiduals {
        let (y, z, w) = (self.y, self.z, self.w);
        let p = &mut self.problem;
        AllResiduals {
            r_yzw: p.add_residual_block(
                Box::new(TernaryCostFunction::new(1, 4, 5, 3)),
                None,
                &[y, z, w],
            ),
            r_yz: p.add_residual_block(Box::new(BinaryCostFunction::new(1, 4, 5)), None, &[y, z]),
            r_yw: p.add_residual_block(Box::new(BinaryCostFunction::new(1, 4, 3)), None, &[y, w]),
            r_zw: p.add_residual_block(Box::new(BinaryCostFunction::new(1, 5, 3)), None, &[z, w]),
            r_y: p.add_residual_block(Box::new(UnaryCostFunction::new(1, 4)), None, &[y]),
            r_z: p.add_residual_block(Box::new(UnaryCostFunction::new(1, 5)), None, &[z]),
            r_w: p.add_residual_block(Box::new(UnaryCostFunction::new(1, 3)), None, &[w]),
        }
    }
}

#[test]
fn dynamic_problem_remove_parameter_block_with_no_residuals() {
    let mut f = DynamicProblem::new();
    let (y, mut z, mut w) = (f.y, f.z, f.w);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![y, z, w]);

    // w is at the end, which might break the swapping logic so try adding and
    // removing it.
    f.problem.remove_parameter_block(w);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![y, z]);
    w = f.problem.add_parameter_block(&[0.0; 3]);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![y, z, w]);

    // Now remove z, which is in the middle, and add it back.
    f.problem.remove_parameter_block(z);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![y, w]);
    z = f.problem.add_parameter_block(&[0.0; 5]);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![y, w, z]);

    // Now remove everything.
    // y. C++ (swap-with-last removal) expects the order z, w here; the Rust
    // port keeps insertion order, so only the set is compared.
    f.problem.remove_parameter_block(y);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_contains_unordered(&f.problem.parameter_blocks(), &[z, w]);

    // z
    f.problem.remove_parameter_block(z);
    assert_eq!(f.problem.num_parameter_blocks(), 1);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![w]);
    assert_eq!(f.problem.parameter_block_size(w), 3);

    // w
    f.problem.remove_parameter_block(w);
    assert_eq!(f.problem.num_parameter_blocks(), 0);
    assert_eq!(f.problem.num_residual_blocks(), 0);
}

#[test]
fn dynamic_problem_remove_parameter_block_with_residuals() {
    let mut f = DynamicProblem::new();
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    assert_eq!(f.problem.parameter_blocks(), vec![f.y, f.z, f.w]);

    // Add all combinations of cost functions.
    let r = f.add_all_residuals();

    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 7);

    // Remove w, which should remove r_yzw, r_yw, r_zw, r_w.
    f.problem.remove_parameter_block(f.w);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 3);

    assert!(!f.has_residual_block(r.r_yzw));
    assert!(f.has_residual_block(r.r_yz));
    assert!(!f.has_residual_block(r.r_yw));
    assert!(!f.has_residual_block(r.r_zw));
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));
    assert!(!f.has_residual_block(r.r_w));

    // Remove z, which will remove almost everything else.
    f.problem.remove_parameter_block(f.z);
    assert_eq!(f.problem.num_parameter_blocks(), 1);
    assert_eq!(f.problem.num_residual_blocks(), 1);

    assert!(!f.has_residual_block(r.r_yzw));
    assert!(!f.has_residual_block(r.r_yz));
    assert!(!f.has_residual_block(r.r_yw));
    assert!(!f.has_residual_block(r.r_zw));
    assert!(f.has_residual_block(r.r_y));
    assert!(!f.has_residual_block(r.r_z));
    assert!(!f.has_residual_block(r.r_w));

    // Remove y; all gone.
    f.problem.remove_parameter_block(f.y);
    assert_eq!(f.problem.num_parameter_blocks(), 0);
    assert_eq!(f.problem.num_residual_blocks(), 0);
}

#[test]
fn dynamic_problem_remove_residual_block() {
    let mut f = DynamicProblem::new();
    let (y, z, w) = (f.y, f.z, f.w);
    let r = f.add_all_residuals();

    f.expect_parameter_block_contains(y, &[r.r_yzw, r.r_yz, r.r_yw, r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_yzw, r.r_yz, r.r_zw, r.r_z]);
    f.expect_parameter_block_contains(w, &[r.r_yzw, r.r_yw, r.r_zw, r.r_w]);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 7);

    // Remove each residual and check the state after each removal.

    // Remove r_yzw.
    f.problem.remove_residual_block(r.r_yzw);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 6);
    f.expect_parameter_block_contains(y, &[r.r_yz, r.r_yw, r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_yz, r.r_zw, r.r_z]);
    f.expect_parameter_block_contains(w, &[r.r_yw, r.r_zw, r.r_w]);
    assert!(f.has_residual_block(r.r_yz));
    assert!(f.has_residual_block(r.r_yw));
    assert!(f.has_residual_block(r.r_zw));
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));
    assert!(f.has_residual_block(r.r_w));

    // Remove r_yw.
    f.problem.remove_residual_block(r.r_yw);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 5);
    f.expect_parameter_block_contains(y, &[r.r_yz, r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_yz, r.r_zw, r.r_z]);
    f.expect_parameter_block_contains(w, &[r.r_zw, r.r_w]);
    assert!(f.has_residual_block(r.r_yz));
    assert!(f.has_residual_block(r.r_zw));
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));
    assert!(f.has_residual_block(r.r_w));

    // Remove r_zw.
    f.problem.remove_residual_block(r.r_zw);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 4);
    f.expect_parameter_block_contains(y, &[r.r_yz, r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_yz, r.r_z]);
    f.expect_parameter_block_contains(w, &[r.r_w]);
    assert!(f.has_residual_block(r.r_yz));
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));
    assert!(f.has_residual_block(r.r_w));

    // Remove r_w.
    f.problem.remove_residual_block(r.r_w);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 3);
    f.expect_parameter_block_contains(y, &[r.r_yz, r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_yz, r.r_z]);
    f.expect_parameter_block_contains(w, &[]);
    assert!(f.has_residual_block(r.r_yz));
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));

    // Remove r_yz.
    f.problem.remove_residual_block(r.r_yz);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 2);
    f.expect_parameter_block_contains(y, &[r.r_y]);
    f.expect_parameter_block_contains(z, &[r.r_z]);
    f.expect_parameter_block_contains(w, &[]);
    assert!(f.has_residual_block(r.r_y));
    assert!(f.has_residual_block(r.r_z));

    // Remove the last two.
    f.problem.remove_residual_block(r.r_z);
    f.problem.remove_residual_block(r.r_y);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 0);
    f.expect_parameter_block_contains(y, &[]);
    f.expect_parameter_block_contains(z, &[]);
    f.expect_parameter_block_contains(w, &[]);
}

/// The non-dying part of DynamicProblem.RemoveInvalidResidualBlockDies.
#[test]
fn dynamic_problem_remove_invalid_residual_block() {
    let mut f = DynamicProblem::new();
    let r = f.add_all_residuals();

    // Remove r_yzw.
    f.problem.remove_residual_block(r.r_yzw);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 6);

    // Remove a parameter block, which in turn removes the dependent residuals.
    f.problem.remove_parameter_block(f.z);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 3);

    f.problem.remove_residual_block(r.r_yw);
    f.problem.remove_residual_block(r.r_w);
    f.problem.remove_residual_block(r.r_y);
    assert_eq!(f.problem.num_residual_blocks(), 0);
}

#[test]
#[should_panic(expected = "not found")]
fn dynamic_problem_remove_invalid_residual_block_dies_removed_twice() {
    let mut f = DynamicProblem::new();
    let r = f.add_all_residuals();
    f.problem.remove_residual_block(r.r_yzw);
    assert_eq!(f.problem.num_parameter_blocks(), 3);
    assert_eq!(f.problem.num_residual_blocks(), 6);
    // Attempt to remove r_yzw again.
    f.problem.remove_residual_block(r.r_yzw);
}

#[test]
#[should_panic(expected = "not found")]
fn dynamic_problem_remove_invalid_residual_block_dies_never_added() {
    let mut f = DynamicProblem::new();
    f.add_all_residuals();
    // Attempt to remove a handle never added to this problem (C++: a cast
    // pointer). It comes from another problem with more residual blocks.
    let mut other = DynamicProblem::new();
    other.add_all_residuals();
    let invalid_residual = other.add_all_residuals().r_w;
    f.problem.remove_residual_block(invalid_residual);
}

#[track_caller]
fn remove_residual_after_parameter_removal(pick: fn(&AllResiduals) -> ResidualBlockId) {
    let mut f = DynamicProblem::new();
    let r = f.add_all_residuals();
    f.problem.remove_residual_block(r.r_yzw);
    // Remove a parameter block, which in turn removes the dependent residuals
    // then attempt to remove them directly.
    f.problem.remove_parameter_block(f.z);
    assert_eq!(f.problem.num_parameter_blocks(), 2);
    assert_eq!(f.problem.num_residual_blocks(), 3);
    f.problem.remove_residual_block(pick(&r));
}

#[test]
#[should_panic(expected = "not found")]
fn dynamic_problem_remove_invalid_residual_block_dies_r_yz() {
    remove_residual_after_parameter_removal(|r| r.r_yz);
}

#[test]
#[should_panic(expected = "not found")]
fn dynamic_problem_remove_invalid_residual_block_dies_r_zw() {
    remove_residual_after_parameter_removal(|r| r.r_zw);
}

#[test]
#[should_panic(expected = "not found")]
fn dynamic_problem_remove_invalid_residual_block_dies_r_z() {
    remove_residual_after_parameter_removal(|r| r.r_z);
}

#[test]
fn dynamic_problem_get_xxx_blocks_for_yyy_block() {
    let mut f = DynamicProblem::new();
    let (y, z, w) = (f.y, f.z, f.w);
    let p = &mut f.problem;

    // Add all combinations of cost functions.
    let r_yzw = p.add_residual_block(
        Box::new(TernaryCostFunction::new(1, 4, 5, 3)),
        None,
        &[y, z, w],
    );
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw]);
    let r_yz = p.add_residual_block(Box::new(BinaryCostFunction::new(1, 4, 5)), None, &[y, z]);
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw, r_yz]);
    let r_yw = p.add_residual_block(Box::new(BinaryCostFunction::new(1, 4, 3)), None, &[y, w]);
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw, r_yz, r_yw]);
    let r_zw = p.add_residual_block(Box::new(BinaryCostFunction::new(1, 5, 3)), None, &[z, w]);
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw, r_yz, r_yw, r_zw]);
    let r_y = p.add_residual_block(Box::new(UnaryCostFunction::new(1, 4)), None, &[y]);
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw, r_yz, r_yw, r_zw, r_y]);
    let r_z = p.add_residual_block(Box::new(UnaryCostFunction::new(1, 5)), None, &[z]);
    assert_contains_unordered(&p.residual_blocks(), &[r_yzw, r_yz, r_yw, r_zw, r_y, r_z]);
    let r_w = p.add_residual_block(Box::new(UnaryCostFunction::new(1, 3)), None, &[w]);
    assert_contains_unordered(
        &p.residual_blocks(),
        &[r_yzw, r_yz, r_yw, r_zw, r_y, r_z, r_w],
    );

    // Check residual_blocks_for_parameter_block() for all parameter blocks.
    let get_residual_blocks_cases: [(ParameterBlockId, Vec<ResidualBlockId>); 3] = [
        (y, vec![r_yzw, r_yz, r_yw, r_y]),
        (z, vec![r_yzw, r_yz, r_zw, r_z]),
        (w, vec![r_yzw, r_yw, r_zw, r_w]),
    ];
    for (parameter_block, expected) in &get_residual_blocks_cases {
        assert_contains_unordered(
            &p.residual_blocks_for_parameter_block(*parameter_block),
            expected,
        );
    }

    // Check parameter_blocks_for_residual_block() for all residual blocks.
    let get_parameter_blocks_cases: [(ResidualBlockId, Vec<ParameterBlockId>); 7] = [
        (r_yzw, vec![y, z, w]),
        (r_yz, vec![y, z]),
        (r_yw, vec![y, w]),
        (r_zw, vec![z, w]),
        (r_y, vec![y]),
        (r_z, vec![z]),
        (r_w, vec![w]),
    ];
    for (residual_block, expected) in &get_parameter_blocks_cases {
        assert_contains_unordered(
            p.parameter_blocks_for_residual_block(*residual_block),
            expected,
        );
    }
}

// --- Problem::Evaluate ---------------------------------------------------------

/// r_i = i - (j + 1) * x_ij^2
struct QuadraticCostFunction<const NUM_RESIDUALS: usize, const NUM_PARAMETER_BLOCKS: usize> {
    parameter_block_sizes: Vec<usize>,
}

impl<const NUM_RESIDUALS: usize, const NUM_PARAMETER_BLOCKS: usize>
    QuadraticCostFunction<NUM_RESIDUALS, NUM_PARAMETER_BLOCKS>
{
    fn new() -> Self {
        assert!(NUM_RESIDUALS > 0);
        assert!(NUM_PARAMETER_BLOCKS > 0);
        Self {
            parameter_block_sizes: vec![NUM_RESIDUALS; NUM_PARAMETER_BLOCKS],
        }
    }
}

impl<const NUM_RESIDUALS: usize, const NUM_PARAMETER_BLOCKS: usize> CostFunction
    for QuadraticCostFunction<NUM_RESIDUALS, NUM_PARAMETER_BLOCKS>
{
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        for (i, r) in residuals.iter_mut().enumerate().take(NUM_RESIDUALS) {
            *r = i as f64;
            for (j, p) in parameters.iter().enumerate().take(NUM_PARAMETER_BLOCKS) {
                *r -= (j as f64 + 1.0) * p[i] * p[i];
            }
        }

        let Some(jacobians) = jacobians else {
            return true;
        };

        for (j, jacobian) in jacobians.iter_mut().enumerate().take(NUM_PARAMETER_BLOCKS) {
            if let Some(jacobian) = jacobian {
                jacobian.fill(0.0);
                for i in 0..NUM_RESIDUALS {
                    jacobian[i * NUM_RESIDUALS + i] = -2.0 * (j as f64 + 1.0) * parameters[j][i];
                }
            }
        }
        true
    }
    fn num_residuals(&self) -> usize {
        NUM_RESIDUALS
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }
}

/// C++ `ExpectedEvaluation` (evaluator_test_utils.h); the jacobian is row
/// major.
struct ExpectedEvaluation<'e> {
    num_rows: usize,
    num_cols: usize,
    cost: f64,
    residuals: &'e [f64],
    gradient: &'e [f64],
    jacobian: &'e [f64],
}

struct ProblemEvaluateTest {
    problem: Problem<'static>,
    parameter_blocks: Vec<ParameterBlockId>,
    residual_blocks: Vec<ResidualBlockId>,
}

impl ProblemEvaluateTest {
    fn new() -> Self {
        let mut problem = Problem::new();
        // parameters_[i] = i + 1, split into blocks x, y, z of size 2.
        let x = problem.add_parameter_block(&[1.0, 2.0]);
        let y = problem.add_parameter_block(&[3.0, 4.0]);
        let z = problem.add_parameter_block(&[5.0, 6.0]);
        let parameter_blocks = vec![x, y, z];

        let cost_function = Arc::new(QuadraticCostFunction::<2, 2>::new());

        let residual_blocks = vec![
            // f(x, y)
            problem.add_residual_block(Box::new(cost_function.clone()), None, &[x, y]),
            // g(y, z)
            problem.add_residual_block(Box::new(cost_function.clone()), None, &[y, z]),
            // h(z, x)
            problem.add_residual_block(Box::new(cost_function), None, &[z, x]),
        ];
        Self {
            problem,
            parameter_blocks,
            residual_blocks,
        }
    }

    #[track_caller]
    fn evaluate_and_compare(
        &self,
        options: &EvaluateOptions,
        expected: &ExpectedEvaluation<'_>,
        want_residuals: bool,
        want_gradient: bool,
        want_jacobian: bool,
    ) {
        let mut cost = 0.0;
        let mut residuals = Vec::new();
        let mut gradient = Vec::new();
        let mut jacobian = CrsMatrix::default();

        assert!(self.problem.evaluate(
            options,
            Some(&mut cost),
            want_residuals.then_some(&mut residuals),
            want_gradient.then_some(&mut gradient),
            want_jacobian.then_some(&mut jacobian),
        ));

        // C++ CompareEvaluations uses exact comparisons.
        assert_eq!(cost, expected.cost);

        if want_residuals {
            assert_eq!(residuals.len(), expected.num_rows);
            assert_eq!(residuals.as_slice(), expected.residuals);
        }

        if want_gradient {
            assert_eq!(gradient.len(), expected.num_cols);
            assert_eq!(gradient.as_slice(), expected.gradient);
        }

        if want_jacobian {
            assert_eq!(jacobian.num_rows, expected.num_rows);
            assert_eq!(jacobian.num_cols, expected.num_cols);
            let dense = jacobian.to_dense();
            let expected_jacobian =
                DMatrix::from_row_slice(expected.num_rows, expected.num_cols, expected.jacobian);
            assert_eq!(
                dense, expected_jacobian,
                "Actual:\n{dense}\nExpected:\n{expected_jacobian}"
            );
        }
    }

    #[track_caller]
    fn check_all_evaluation_combinations(
        &self,
        options: &EvaluateOptions,
        expected: &ExpectedEvaluation<'_>,
    ) {
        for i in 0..8 {
            self.evaluate_and_compare(options, expected, i & 1 != 0, i & 2 != 0, i & 4 != 0);
        }
    }
}

#[rustfmt::skip]
const FULL_RESIDUALS: [f64; 6] = [
    -19.0, -35.0,  // f
    -59.0, -87.0,  // g
    -27.0, -43.0,  // h
];

#[rustfmt::skip]
const FULL_GRADIENT: [f64; 6] = [
     146.0,  484.0,  // x
     582.0, 1256.0,  // y
    1450.0, 2604.0,  // z
];

#[rustfmt::skip]
const FULL_JACOBIAN: [f64; 36] = [
    //                x             y             z
    /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                   0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
    /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                   0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
    /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                   0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
];

#[test]
fn problem_evaluate_test_multiple_parameter_and_residual_blocks() {
    let t = ProblemEvaluateTest::new();
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 6,
        cost: 7607.0,
        residuals: &FULL_RESIDUALS,
        gradient: &FULL_GRADIENT,
        jacobian: &FULL_JACOBIAN,
    };
    t.check_all_evaluation_combinations(&EvaluateOptions::default(), &expected);
}

#[test]
fn problem_evaluate_test_parameter_and_residual_blocks_passed_in_options() {
    let t = ProblemEvaluateTest::new();
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 6,
        cost: 7607.0,
        residuals: &FULL_RESIDUALS,
        gradient: &FULL_GRADIENT,
        jacobian: &FULL_JACOBIAN,
    };
    let evaluate_options = EvaluateOptions {
        parameter_blocks: t.parameter_blocks.clone(),
        residual_blocks: t.residual_blocks.clone(),
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_reordered_residual_blocks() {
    let t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 6,
        num_cols: 6,
        // Cost
        cost: 7607.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -27.0, -43.0,  // h
            -59.0, -87.0,  // g
        ],
        // Gradient
        gradient: &[
             146.0,  484.0,  // x
             582.0, 1256.0,  // y
            1450.0, 2604.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x             y             z
            /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
            /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
        ],
    };

    let evaluate_options = EvaluateOptions {
        parameter_blocks: t.parameter_blocks.clone(),
        // f, h, g
        residual_blocks: vec![
            t.residual_blocks[0],
            t.residual_blocks[2],
            t.residual_blocks[1],
        ],
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_reordered_residual_blocks_and_reordered_parameter_blocks() {
    let t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 6,
        num_cols: 6,
        // Cost
        cost: 7607.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -27.0, -43.0,  // h
            -59.0, -87.0,  // g
        ],
        // Gradient
        gradient: &[
            1450.0, 2604.0,  // z
             582.0, 1256.0,  // y
             146.0,  484.0,  // x
        ],
        // Jacobian
        jacobian: &[
            //                 z             y             x
            /* f(x, y) */   0.0,   0.0, -12.0,   0.0,  -2.0,   0.0,
                            0.0,   0.0,   0.0, -16.0,   0.0,  -4.0,
            /* h(z, x) */ -10.0,   0.0,   0.0,   0.0,  -4.0,   0.0,
                            0.0, -12.0,   0.0,   0.0,   0.0,  -8.0,
            /* g(y, z) */ -20.0,   0.0,  -6.0,   0.0,   0.0,   0.0,
                            0.0, -24.0,   0.0,  -8.0,   0.0,   0.0,
        ],
    };

    let evaluate_options = EvaluateOptions {
        // z, y, x
        parameter_blocks: vec![
            t.parameter_blocks[2],
            t.parameter_blocks[1],
            t.parameter_blocks[0],
        ],
        // f, h, g
        residual_blocks: vec![
            t.residual_blocks[0],
            t.residual_blocks[2],
            t.residual_blocks[1],
        ],
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_constant_parameter_block() {
    let mut t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 6,
        num_cols: 6,
        // Cost
        cost: 7607.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -59.0, -87.0,  // g
            -27.0, -43.0,  // h
        ],
        // Gradient
        gradient: &[
             146.0,  484.0,  // x
               0.0,    0.0,  // y
            1450.0, 2604.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x             y             z
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0,   0.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0,   0.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
        ],
    };

    let y = t.parameter_blocks[1];
    t.problem.set_parameter_block_constant(y);
    t.check_all_evaluation_combinations(&EvaluateOptions::default(), &expected);
}

#[test]
fn problem_evaluate_test_excluded_a_residual_block() {
    let t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 4,
        num_cols: 6,
        // Cost
        cost: 2082.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -27.0, -43.0,  // h
        ],
        // Gradient
        gradient: &[
            146.0, 484.0,  // x
            228.0, 560.0,  // y
            270.0, 516.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x             y             z
            /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
        ],
    };

    let evaluate_options = EvaluateOptions {
        residual_blocks: vec![t.residual_blocks[0], t.residual_blocks[2]],
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_excluded_parameter_block() {
    let t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 6,
        num_cols: 4,
        // Cost
        cost: 7607.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -59.0, -87.0,  // g
            -27.0, -43.0,  // h
        ],
        // Gradient
        gradient: &[
             146.0,  484.0,  // x
            1450.0, 2604.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x             z
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0, -12.0,
        ],
    };

    let evaluate_options = EvaluateOptions {
        // x, z
        parameter_blocks: vec![t.parameter_blocks[0], t.parameter_blocks[2]],
        residual_blocks: t.residual_blocks.clone(),
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_excluded_parameter_block_and_excluded_residual_block() {
    let t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 4,
        num_cols: 4,
        // Cost
        cost: 6318.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -59.0, -87.0,  // g
        ],
        // Gradient
        gradient: &[
              38.0,  140.0,  // x
            1180.0, 2088.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x             z
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0, -24.0,
        ],
    };

    let evaluate_options = EvaluateOptions {
        // x, z
        parameter_blocks: vec![t.parameter_blocks[0], t.parameter_blocks[2]],
        residual_blocks: vec![t.residual_blocks[0], t.residual_blocks[1]],
        ..EvaluateOptions::default()
    };
    t.check_all_evaluation_combinations(&evaluate_options, &expected);
}

#[test]
fn problem_evaluate_test_local_parameterization() {
    let mut t = ProblemEvaluateTest::new();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        // Rows/columns
        num_rows: 6,
        num_cols: 5,
        // Cost
        cost: 7607.0,
        // Residuals
        residuals: &[
            -19.0, -35.0,  // f
            -59.0, -87.0,  // g
            -27.0, -43.0,  // h
        ],
        // Gradient
        gradient: &[
             146.0,  484.0,  // x
            1256.0,          // y with SubsetParameterization
            1450.0, 2604.0,  // z
        ],
        // Jacobian
        jacobian: &[
            //                x      y             z
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,
                           0.0, -4.0, -16.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,  -8.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0, -12.0,
        ],
    };

    let constant_parameters = [0];
    let y = t.parameter_blocks[1];
    t.problem.set_parameterization(
        y,
        Some(Box::new(SubsetParameterization::new(
            2,
            &constant_parameters,
        ))),
    );
    t.check_all_evaluation_combinations(&EvaluateOptions::default(), &expected);
}

// --- Problem::EvaluateResidualBlock ------------------------------------------

struct IdentityFunctor;

impl CostFunctor for IdentityFunctor {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residuals: &mut [T]) -> bool {
        let (x, y) = (parameters[0], parameters[1]);
        residuals[0] = x[0];
        residuals[1] = x[1];
        residuals[2] = y[0];
        residuals[3] = y[1];
        residuals[4] = y[2];
        true
    }
}

impl IdentityFunctor {
    fn create() -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 5>::new(
            IdentityFunctor,
            5,
            &[2, 3],
        ))
    }
}

const K_APPLY_LOSS_FUNCTION: bool = true;
const K_DO_NOT_APPLY_LOSS_FUNCTION: bool = false;
const LOSS_FUNCTION_SCALE: f64 = 2.0;

/// The ProblemEvaluateResidualBlockTest fixture: x = {1, 2}, y = {1, 2, 3}.
fn residual_block_fixture(
    loss: Option<Box<dyn LossFunction>>,
) -> (
    Problem<'static>,
    ParameterBlockId,
    ParameterBlockId,
    ResidualBlockId,
) {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);
    let y = problem.add_parameter_block(&[1.0, 2.0, 3.0]);
    let residual_block_id = problem.add_residual_block(IdentityFunctor::create(), loss, &[x, y]);
    (problem, x, y, residual_block_id)
}

fn expected_f() -> DVector<f64> {
    DVector::from_row_slice(&[1.0, 2.0, 1.0, 2.0, 3.0])
}

/// A 5 x `cols` zero matrix with an identity block at (`row`, 0) of size `n`.
fn identity_block(cols: usize, row: usize, n: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(5, cols);
    m.view_mut((row, 0), (n, n)).fill_with_identity();
    m
}

#[track_caller]
fn expect_cost_near(expected_cost: f64, actual_cost: f64) {
    let rel = (expected_cost - actual_cost).abs() / actual_cost;
    assert!(rel <= f64::EPSILON, "{actual_cost} vs {expected_cost}");
}

#[track_caller]
fn expect_vector_near(expected: &DVector<f64>, actual: &[f64]) {
    let actual = DVector::from_row_slice(actual);
    let rel = (expected - &actual).norm() / actual.norm();
    assert!(rel <= f64::EPSILON, "{actual} vs {expected}");
}

/// `actual` is a row-major `expected.nrows() x expected.ncols()` buffer.
#[track_caller]
fn expect_matrix_near(expected: &DMatrix<f64>, actual: &[f64]) {
    let actual = DMatrix::from_row_slice(expected.nrows(), expected.ncols(), actual);
    let rel = (expected - &actual).norm() / actual.norm();
    assert!(rel <= f64::EPSILON, "{actual} vs {expected}");
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_full_eval() {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    let expected_f = expected_f();
    let expected_dfdx = identity_block(2, 0, 2);
    let expected_dfdy = identity_block(3, 2, 3);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];
    let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_null_eval() {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        None,
        None,
        None,
    ));
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_cost() {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    let expected_cost = expected_f().norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        None,
        None,
    ));

    expect_cost_near(expected_cost, actual_cost);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_cost_and_residual() {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    let expected_f = expected_f();
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        None,
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_cost_residual_and_one_jacobian()
 {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    let expected_f = expected_f();
    let expected_dfdx = identity_block(2, 0, 2);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut jacobians = [Some(&mut actual_dfdx[..]), None];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_no_loss_function_residual() {
    let (problem, _, _, residual_block_id) = residual_block_fixture(None);
    let expected_f = expected_f();
    let mut actual_f = [0.0; 5];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        None,
        Some(&mut actual_f),
        None,
    ));

    expect_vector_near(&expected_f, &actual_f);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_loss_function() {
    let (problem, _, _, residual_block_id) =
        residual_block_fixture(Some(Box::new(ScaledLoss::new(None, 2.0))));
    let expected_f = expected_f() * LOSS_FUNCTION_SCALE.sqrt();
    let expected_dfdx = identity_block(2, 0, 2) * LOSS_FUNCTION_SCALE.sqrt();
    let expected_dfdy = identity_block(3, 2, 3) * LOSS_FUNCTION_SCALE.sqrt();
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];
    let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_loss_function_disabled() {
    let (problem, _, _, residual_block_id) =
        residual_block_fixture(Some(Box::new(ScaledLoss::new(None, 2.0))));
    let expected_f = expected_f();
    let expected_dfdx = identity_block(2, 0, 2);
    let expected_dfdy = identity_block(3, 2, 3);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];
    let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_DO_NOT_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_one_local_parameterization() {
    let (mut problem, x, _, residual_block_id) = residual_block_fixture(None);
    problem.set_parameterization(x, Some(Box::new(SubsetParameterization::new(2, &[1]))));

    let expected_f = expected_f();
    let expected_dfdx = identity_block(1, 0, 1);
    let expected_dfdy = identity_block(3, 2, 3);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5];
    let mut actual_dfdy = [0.0; 5 * 3];
    let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_two_local_parameterizations() {
    let (mut problem, x, y, residual_block_id) = residual_block_fixture(None);
    problem.set_parameterization(x, Some(Box::new(SubsetParameterization::new(2, &[1]))));
    problem.set_parameterization(y, Some(Box::new(SubsetParameterization::new(3, &[2]))));

    let expected_f = expected_f();
    let expected_dfdx = identity_block(1, 0, 1);
    let expected_dfdy = identity_block(2, 2, 2);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5];
    let mut actual_dfdy = [0.0; 5 * 2];
    let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdx, &actual_dfdx);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_one_constant_parameter_block() {
    let (mut problem, x, _, residual_block_id) = residual_block_fixture(None);
    problem.set_parameter_block_constant(x);

    let expected_f = expected_f();
    let expected_dfdy = identity_block(3, 2, 3);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];

    // Try evaluating both Jacobians, this should fail.
    {
        let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
        assert!(!problem.evaluate_residual_block(
            residual_block_id,
            K_APPLY_LOSS_FUNCTION,
            Some(&mut actual_cost),
            Some(&mut actual_f),
            Some(&mut jacobians),
        ));
    }

    let mut jacobians = [None, Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_all_constant_parameter_blocks() {
    let (mut problem, x, y, residual_block_id) = residual_block_fixture(None);
    problem.set_parameter_block_constant(x);
    problem.set_parameter_block_constant(y);

    let expected_f = expected_f();
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];

    // Try evaluating with one or more Jacobians, this should fail.
    {
        let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
        assert!(!problem.evaluate_residual_block(
            residual_block_id,
            K_APPLY_LOSS_FUNCTION,
            Some(&mut actual_cost),
            Some(&mut actual_f),
            Some(&mut jacobians),
        ));
    }
    {
        let mut jacobians = [None, Some(&mut actual_dfdy[..])];
        assert!(!problem.evaluate_residual_block(
            residual_block_id,
            K_APPLY_LOSS_FUNCTION,
            Some(&mut actual_cost),
            Some(&mut actual_f),
            Some(&mut jacobians),
        ));
    }
    let mut jacobians: [Option<&mut [f64]>; 2] = [None, None];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));

    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
}

#[test]
fn problem_evaluate_residual_block_test_one_residual_block_with_one_parameter_block_constant_and_parameter_block_changed()
 {
    let (mut problem, x, y, residual_block_id) = residual_block_fixture(None);
    problem.set_parameter_block_constant(x);

    problem.parameter_block_mut(x)[0] = 2.0;
    problem.parameter_block_mut(y)[2] = 1.0;
    let expected_f = DVector::from_row_slice(&[2.0, 2.0, 1.0, 2.0, 1.0]);
    let expected_dfdy = identity_block(3, 2, 3);
    let expected_cost = expected_f.norm_squared() / 2.0;

    let mut actual_cost = 0.0;
    let mut actual_f = [0.0; 5];
    let mut actual_dfdx = [0.0; 5 * 2];
    let mut actual_dfdy = [0.0; 5 * 3];

    // Try evaluating with one or more Jacobians, this should fail.
    {
        let mut jacobians = [Some(&mut actual_dfdx[..]), Some(&mut actual_dfdy[..])];
        assert!(!problem.evaluate_residual_block(
            residual_block_id,
            K_APPLY_LOSS_FUNCTION,
            Some(&mut actual_cost),
            Some(&mut actual_f),
            Some(&mut jacobians),
        ));
    }

    let mut jacobians = [None, Some(&mut actual_dfdy[..])];
    assert!(problem.evaluate_residual_block(
        residual_block_id,
        K_APPLY_LOSS_FUNCTION,
        Some(&mut actual_cost),
        Some(&mut actual_f),
        Some(&mut jacobians),
    ));
    expect_cost_near(expected_cost, actual_cost);
    expect_vector_near(&expected_f, &actual_f);
    expect_matrix_near(&expected_dfdy, &actual_dfdy);
}

#[test]
fn set_and_get_parameter_lower_bound() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);

    assert_eq!(problem.get_parameter_lower_bound(x, 0), -f64::MAX);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);

    problem.set_parameter_lower_bound(x, 0, -1.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 0), -1.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);

    problem.set_parameter_lower_bound(x, 0, -2.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 0), -2.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);

    problem.set_parameter_lower_bound(x, 0, -f64::MAX);
    assert_eq!(problem.get_parameter_lower_bound(x, 0), -f64::MAX);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);
}

#[test]
fn set_and_get_parameter_upper_bound() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);

    assert_eq!(problem.get_parameter_upper_bound(x, 0), f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), f64::MAX);

    problem.set_parameter_upper_bound(x, 0, -1.0);
    assert_eq!(problem.get_parameter_upper_bound(x, 0), -1.0);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), f64::MAX);

    problem.set_parameter_upper_bound(x, 0, -2.0);
    assert_eq!(problem.get_parameter_upper_bound(x, 0), -2.0);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), f64::MAX);

    problem.set_parameter_upper_bound(x, 0, f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 0), f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), f64::MAX);
}

// ParameterBlock.DefaultBounds / ParameterBlock.SetBounds
// (parameter_block_test.cc), through the public Problem API.
#[test]
fn parameter_block_default_and_set_bounds() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0, 0.0]);
    assert_eq!(problem.get_parameter_upper_bound(x, 0), f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), f64::MAX);
    assert_eq!(problem.get_parameter_lower_bound(x, 0), -f64::MAX);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);

    problem.set_parameter_lower_bound(x, 0, 1.0);
    problem.set_parameter_upper_bound(x, 1, 1.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 0), 1.0);
    assert_eq!(problem.get_parameter_lower_bound(x, 1), -f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 0), f64::MAX);
    assert_eq!(problem.get_parameter_upper_bound(x, 1), 1.0);
}

#[test]
#[should_panic(expected = "Parameter block not found")]
fn set_parameter_lower_bound_on_unknown_block_dies() {
    let mut other = Problem::new();
    let x = other.add_parameter_block(&[1.0]);
    let mut problem = Problem::new();
    problem.set_parameter_lower_bound(x, 0, 0.0);
}

#[test]
#[should_panic(expected = "Parameter block not found")]
fn get_parameter_upper_bound_on_removed_block_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0]);
    problem.remove_parameter_block(x);
    problem.get_parameter_upper_bound(x, 0);
}

// CHECK_LT(index, size_) in ParameterBlock::SetLowerBound / SetUpperBound.
#[test]
#[should_panic(expected = "index < size")]
fn set_parameter_bound_with_out_of_range_index_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);
    problem.set_parameter_upper_bound(x, 2, 0.0);
}

#[test]
fn set_parameterization_twice() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0, 3.0]);
    problem.set_parameterization(x, Some(Box::new(SubsetParameterization::new(3, &[1]))));
    assert_eq!(problem.get_parameterization(x).unwrap().global_size(), 3);
    assert_eq!(problem.get_parameterization(x).unwrap().local_size(), 2);

    problem.set_parameterization(x, Some(Box::new(SubsetParameterization::new(3, &[0, 1]))));
    assert_eq!(problem.get_parameterization(x).unwrap().global_size(), 3);
    assert_eq!(problem.get_parameterization(x).unwrap().local_size(), 1);
}

#[test]
fn set_parameterization_and_then_clear_it_with_null() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0, 3.0]);
    problem.set_parameterization(x, Some(Box::new(SubsetParameterization::new(3, &[1]))));
    assert_eq!(problem.get_parameterization(x).unwrap().global_size(), 3);
    assert_eq!(problem.get_parameterization(x).unwrap().local_size(), 2);

    problem.set_parameterization(x, None);
    assert!(problem.get_parameterization(x).is_none());
    assert_eq!(problem.parameter_block_local_size(x), 3);
    assert_eq!(problem.parameter_block_size(x), 3);
}

#[test]
fn zero_sized_local_parameterization_means_parameter_block_is_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(Box::new(BinaryCostFunction::new(1, 1, 1)), None, &[x, y]);
    problem.set_parameterization(y, Some(Box::new(SubsetParameterization::new(1, &[0]))));
    assert!(problem.is_parameter_block_constant(y));
}

// Skipped: ProblemEvaluate.CallsEvaluationCallbackWithoutJacobian,
// ProblemEvaluate.CallsEvaluationCallbackWithJacobian,
// ProblemEvaluateResidualBlock.NewPointCallsEvaluationCallback and
// ProblemEvaluateResidualBlock.OldPointCallsEvaluationCallback
// (EvaluationCallback is not ported).
