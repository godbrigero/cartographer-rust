// Ported from Ceres Solver 2.0.0: internal/ceres/autodiff_cost_function_test.cc

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};

/// C++ `kImpossibleValue` (array_utils.cc), used by `InvalidateArray` and
/// `IsArrayValid`.
const IMPOSSIBLE_VALUE: f64 = 1e302;

/// C++ `InvalidateArray` (array_utils.h).
fn invalidate_array(x: &mut [f64]) {
    x.fill(IMPOSSIBLE_VALUE);
}

/// C++ `IsArrayValid` (array_utils.h).
fn is_array_valid(x: &[f64]) -> bool {
    x.iter().all(|&v| v.is_finite() && v != IMPOSSIBLE_VALUE)
}

struct BinaryScalarCost {
    a: f64,
}

impl BinaryScalarCost {
    fn new(a: f64) -> Self {
        Self { a }
    }
}

impl CostFunctor for BinaryScalarCost {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], cost: &mut [T]) -> bool {
        let (x, y) = (parameters[0], parameters[1]);
        cost[0] = x[0] * y[0] + x[1] * y[1] - T::constant(self.a);
        true
    }
}

#[test]
fn autodiff_cost_function_bilinear_differentiation_test() {
    let cost_function: Box<dyn CostFunction> = Box::new(AutoDiffCostFunction::<_, 4>::new(
        BinaryScalarCost::new(1.0),
        1,
        &[2, 2],
    ));

    let p0 = [1.0, 2.0];
    let p1 = [3.0, 4.0];
    let parameters: [&[f64]; 2] = [&p0, &p1];

    let mut jacobian0 = [0.0; 2];
    let mut jacobian1 = [0.0; 2];

    let mut residuals = [0.0];

    cost_function.evaluate(&parameters, &mut residuals, None);
    assert_eq!(10.0, residuals[0]);

    {
        let mut jacobians: [Option<&mut [f64]>; 2] =
            [Some(&mut jacobian0[..]), Some(&mut jacobian1[..])];
        cost_function.evaluate(&parameters, &mut residuals, Some(&mut jacobians));
    }
    assert_eq!(10.0, residuals[0]);

    assert_eq!(3.0, jacobian0[0]);
    assert_eq!(4.0, jacobian0[1]);
    assert_eq!(1.0, jacobian1[0]);
    assert_eq!(2.0, jacobian1[1]);
}

/// In C++ this checks that the variadic template instantiates with ten
/// parameter blocks; here it is ten blocks of size one (N = 10).
struct TenParameterCost;

impl CostFunctor for TenParameterCost {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], cost: &mut [T]) -> bool {
        cost[0] = parameters[0][0]
            + parameters[1][0]
            + parameters[2][0]
            + parameters[3][0]
            + parameters[4][0]
            + parameters[5][0]
            + parameters[6][0]
            + parameters[7][0]
            + parameters[8][0]
            + parameters[9][0];
        true
    }
}

#[test]
fn autodiff_cost_function_many_parameter_autodiff_instantiates() {
    let cost_function: Box<dyn CostFunction> = Box::new(AutoDiffCostFunction::<_, 10>::new(
        TenParameterCost,
        1,
        &[1; 10],
    ));

    let parameter_values: Vec<[f64; 1]> = (0..10).map(|i| [f64::from(i)]).collect();
    let parameters: Vec<&[f64]> = parameter_values.iter().map(|p| &p[..]).collect();
    let mut jacobian_values = vec![[0.0f64; 1]; 10];

    let mut residuals = [0.0];

    cost_function.evaluate(&parameters, &mut residuals, None);
    assert_eq!(45.0, residuals[0]);

    {
        let mut jacobians: Vec<Option<&mut [f64]>> = jacobian_values
            .iter_mut()
            .map(|j| Some(&mut j[..]))
            .collect();
        cost_function.evaluate(&parameters, &mut residuals, Some(&mut jacobians));
    }
    assert_eq!(residuals[0], 45.0);
    for jacobian in &jacobian_values {
        assert_eq!(1.0, jacobian[0]);
    }
}

struct OnlyFillsOneOutputFunctor;

impl CostFunctor for OnlyFillsOneOutputFunctor {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], output: &mut [T]) -> bool {
        output[0] = parameters[0][0];
        true
    }
}

#[test]
fn auto_diff_cost_function_partially_filled_residual_should_fail_evaluation() {
    let parameter = [1.0];
    let mut jacobian = [0.0; 2];
    let mut residuals = [0.0; 2];
    let parameters: [&[f64]; 1] = [&parameter];

    let cost_function: Box<dyn CostFunction> = Box::new(AutoDiffCostFunction::<_, 1>::new(
        OnlyFillsOneOutputFunctor,
        2,
        &[1],
    ));
    invalidate_array(&mut jacobian);
    invalidate_array(&mut residuals);
    {
        let mut jacobians: [Option<&mut [f64]>; 1] = [Some(&mut jacobian[..])];
        assert!(cost_function.evaluate(&parameters, &mut residuals, Some(&mut jacobians)));
    }
    assert!(!is_array_valid(&jacobian));
    assert!(!is_array_valid(&residuals));
}
