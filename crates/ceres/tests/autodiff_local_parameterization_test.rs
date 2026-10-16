// Ported from Ceres Solver 2.0.0: internal/ceres/autodiff_local_parameterization_test.cc

use ceres::rotation::quaternion_product;
use ceres::{
    AutoDiffLocalParameterization, LocalParameterization, PlusFunctor, QuaternionParameterization,
    Scalar,
};

fn assert_near(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected} +/- {tol}, got {actual} (diff {})",
        (actual - expected).abs()
    );
}

#[derive(Default)]
struct IdentityPlus;

impl PlusFunctor for IdentityPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        for i in 0..3 {
            x_plus_delta[i] = x[i] + delta[i];
        }
        true
    }
}

#[test]
fn auto_diff_local_parameterization_test_identity_parameterization() {
    let parameterization = AutoDiffLocalParameterization::<IdentityPlus, 3, 3>::default();

    let x = [1.0, 2.0, 3.0];
    let delta = [0.0, 1.0, 2.0];
    let mut x_plus_delta = [0.0; 3];
    parameterization.plus(&x, &delta, &mut x_plus_delta);

    assert_eq!(x_plus_delta[0], 1.0);
    assert_eq!(x_plus_delta[1], 3.0);
    assert_eq!(x_plus_delta[2], 5.0);

    let mut jacobian = [0.0; 9];
    parameterization.compute_jacobian(&x, &mut jacobian);
    let mut k = 0;
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(jacobian[k], if i == j { 1.0 } else { 0.0 });
            k += 1;
        }
    }
}

struct ScaledPlus {
    scale_factor: f64,
}

impl ScaledPlus {
    fn new(scale_factor: f64) -> Self {
        Self { scale_factor }
    }
}

impl PlusFunctor for ScaledPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        for i in 0..3 {
            x_plus_delta[i] = x[i] + T::constant(self.scale_factor) * delta[i];
        }
        true
    }
}

#[test]
fn auto_diff_local_parameterization_test_scaled_parameterization() {
    const TOLERANCE: f64 = 1e-14;

    let parameterization =
        AutoDiffLocalParameterization::<ScaledPlus, 3, 3>::new(ScaledPlus::new(1.2345));

    let x = [1.0, 2.0, 3.0];
    let delta = [0.0, 1.0, 2.0];
    let mut x_plus_delta = [0.0; 3];
    parameterization.plus(&x, &delta, &mut x_plus_delta);

    assert_near(x_plus_delta[0], 1.0, TOLERANCE);
    assert_near(x_plus_delta[1], 3.2345, TOLERANCE);
    assert_near(x_plus_delta[2], 5.469, TOLERANCE);

    let mut jacobian = [0.0; 9];
    parameterization.compute_jacobian(&x, &mut jacobian);
    let mut k = 0;
    for i in 0..3 {
        for j in 0..3 {
            assert_near(jacobian[k], if i == j { 1.2345 } else { 0.0 }, TOLERANCE);
            k += 1;
        }
    }
}

#[derive(Default)]
struct QuaternionPlus;

impl PlusFunctor for QuaternionPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        let squared_norm_delta = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

        let q_delta: [T; 4] = if squared_norm_delta > T::constant(0.0) {
            let norm_delta = squared_norm_delta.sqrt();
            let sin_delta_by_delta = norm_delta.sin() / norm_delta;
            [
                norm_delta.cos(),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2],
            ]
        } else {
            // We do not just use q_delta = [1,0,0,0] here because that is a
            // constant and when used for automatic differentiation will
            // lead to a zero derivative. Instead we take a first order
            // approximation and evaluate it at zero.
            [T::constant(1.0), delta[0], delta[1], delta[2]]
        };

        quaternion_product(&q_delta, x, x_plus_delta);
        true
    }
}

fn quaternion_parameterization_test_helper(x: &[f64], delta: &[f64]) {
    const TOLERANCE: f64 = 1e-14;
    let mut x_plus_delta_ref = [0.0; 4];
    let mut jacobian_ref = [0.0; 12];

    let ref_parameterization = QuaternionParameterization;
    ref_parameterization.plus(x, delta, &mut x_plus_delta_ref);
    ref_parameterization.compute_jacobian(x, &mut jacobian_ref);

    let mut x_plus_delta = [0.0; 4];
    let mut jacobian = [0.0; 12];
    let parameterization = AutoDiffLocalParameterization::<QuaternionPlus, 4, 3>::default();
    parameterization.plus(x, delta, &mut x_plus_delta);
    parameterization.compute_jacobian(x, &mut jacobian);

    for i in 0..4 {
        assert_near(x_plus_delta[i], x_plus_delta_ref[i], TOLERANCE);
    }

    let x_plus_delta_norm = (x_plus_delta[0] * x_plus_delta[0]
        + x_plus_delta[1] * x_plus_delta[1]
        + x_plus_delta[2] * x_plus_delta[2]
        + x_plus_delta[3] * x_plus_delta[3])
        .sqrt();

    assert_near(x_plus_delta_norm, 1.0, TOLERANCE);

    for i in 0..12 {
        assert!(jacobian[i].is_finite());
        assert!(
            (jacobian[i] - jacobian_ref[i]).abs() <= TOLERANCE,
            "Jacobian mismatch: i = {i}\n Expected \n{jacobian_ref:?}\n Actual \n{jacobian:?}"
        );
    }
}

#[test]
fn auto_diff_local_parameterization_quaternion_parameterization_zero_test() {
    let x = [0.5, 0.5, 0.5, 0.5];
    let delta = [0.0, 0.0, 0.0];
    quaternion_parameterization_test_helper(&x, &delta);
}

#[test]
fn auto_diff_local_parameterization_quaternion_parameterization_near_zero_test() {
    let mut x: [f64; 4] = [0.52, 0.25, 0.15, 0.45];
    let norm_x = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
    for xi in &mut x {
        *xi /= norm_x;
    }

    let mut delta = [0.24, 0.15, 0.10];
    for d in &mut delta {
        *d *= 1e-14;
    }

    quaternion_parameterization_test_helper(&x, &delta);
}

#[test]
fn auto_diff_local_parameterization_quaternion_parameterization_non_zero_test() {
    let mut x: [f64; 4] = [0.52, 0.25, 0.15, 0.45];
    let norm_x = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();

    for xi in &mut x {
        *xi /= norm_x;
    }

    let delta = [0.24, 0.15, 0.10];
    quaternion_parameterization_test_helper(&x, &delta);
}
