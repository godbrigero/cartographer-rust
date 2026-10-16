// Ported from Ceres Solver 2.0.0: internal/ceres/local_parameterization_test.cc

use std::sync::Arc;

use ceres::rotation::quaternion_product;
use ceres::{
    AutoDiffCostFunction, AutoDiffLocalParameterization, CostFunction, CostFunctor,
    EigenQuaternionParameterization, HomogeneousVectorParameterization, IdentityParameterization,
    LocalParameterization, PlusFunctor, ProductParameterization, QuaternionParameterization,
    Scalar, SubsetParameterization,
};
use nalgebra::Quaternion;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helpers (row-major matrices stored in Vec<f64>, like ceres' RowMajor
// `Matrix` typedef).
// ---------------------------------------------------------------------------

/// `a (rows x inner) * b (inner x cols)`, all row-major.
fn mat_mul(a: &[f64], rows: usize, inner: usize, b: &[f64], cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += a[r * inner + k] * b[k * cols + c];
            }
            out[r * cols + c] = sum;
        }
    }
    out
}

/// Frobenius norm of `a - b`.
fn diff_norm(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Eigen's `Matrix::Random`: uniform entries in [-1, 1].
fn random_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|_| rng.random_range(-1.0..=1.0))
        .collect()
}

/// ceres' `RandNormal()` (Marsaglia polar method).
fn rand_normal(rng: &mut StdRng) -> f64 {
    loop {
        let x1: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let x2: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let w = x1 * x1 + x2 * x2;
        if w < 1.0 && w > 0.0 {
            return x1 * (-2.0 * w.ln() / w).sqrt();
        }
    }
}

fn normalize<const N: usize>(x: &mut [f64; N]) {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in x.iter_mut() {
        *v /= norm;
    }
}

fn assert_near(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected} +/- {tol}, got {actual} (diff {})",
        (actual - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// IdentityParameterization
// ---------------------------------------------------------------------------

#[test]
fn identity_parameterization_everything_test() {
    let parameterization = IdentityParameterization::new(3);
    assert_eq!(parameterization.global_size(), 3);
    assert_eq!(parameterization.local_size(), 3);

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

    let global_matrix = vec![1.0; 10 * 3];
    let mut local_matrix = vec![0.0; 10 * 3];
    parameterization.multiply_by_jacobian(&x, 10, &global_matrix, &mut local_matrix);
    assert_eq!(diff_norm(&local_matrix, &global_matrix), 0.0);
}

// ---------------------------------------------------------------------------
// SubsetParameterization
// ---------------------------------------------------------------------------

#[test]
fn subset_parameterization_empty_constant_parameters() {
    let constant_parameters: Vec<usize> = Vec::new();
    let parameterization = SubsetParameterization::new(3, &constant_parameters);
    assert_eq!(parameterization.global_size(), 3);
    assert_eq!(parameterization.local_size(), 3);
    let x = [1.0, 2.0, 3.0];
    let delta = [4.0, 5.0, 6.0];
    let mut x_plus_delta = [-1.0, -2.0, -3.0];
    parameterization.plus(&x, &delta, &mut x_plus_delta);
    assert_eq!(x_plus_delta[0], x[0] + delta[0]);
    assert_eq!(x_plus_delta[1], x[1] + delta[1]);
    assert_eq!(x_plus_delta[2], x[2] + delta[2]);

    let mut jacobian = [0.0; 9];
    let expected_jacobian = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    parameterization.compute_jacobian(&x, &mut jacobian);
    assert_eq!(jacobian, expected_jacobian);

    // C++ uses a 3x5 random matrix interpreted as 5 rows of size 3.
    let mut rng = StdRng::seed_from_u64(5);
    let global_matrix = random_matrix(&mut rng, 3, 5);
    let mut local_matrix = vec![0.0; 3 * 5];
    parameterization.multiply_by_jacobian(&x, 5, &global_matrix, &mut local_matrix);
    assert_eq!(global_matrix, local_matrix);
}

// NegativeParameterIndexDeathTest: not portable, constant indices are `usize`
// in the Rust API so a negative index cannot be expressed.

#[test]
#[should_panic(expected = "less than the size")]
fn subset_parameterization_greater_than_size_parameter_index_death_test() {
    let constant_parameters = vec![2];
    let _parameterization = SubsetParameterization::new(2, &constant_parameters);
}

#[test]
#[should_panic(expected = "duplicates")]
fn subset_parameterization_duplicate_parameters_death_test() {
    let constant_parameters = vec![1, 1];
    let _parameterization = SubsetParameterization::new(2, &constant_parameters);
}

#[test]
fn subset_parameterization_product_parameterization_with_zero_local_size_subset_parameterization1()
{
    let constant_parameters = vec![0];
    let subset_param: Box<dyn LocalParameterization> =
        Box::new(SubsetParameterization::new(1, &constant_parameters));
    let identity_param: Box<dyn LocalParameterization> = Box::new(IdentityParameterization::new(2));
    let product_param = ProductParameterization::new(vec![subset_param, identity_param]);
    assert_eq!(product_param.global_size(), 3);
    assert_eq!(product_param.local_size(), 2);
    let x = [1.0, 1.0, 1.0];
    let delta = [2.0, 3.0];
    let mut x_plus_delta = [0.0; 3];
    assert!(product_param.plus(&x, &delta, &mut x_plus_delta));
    assert_eq!(x_plus_delta[0], x[0]);
    assert_eq!(x_plus_delta[1], x[1] + delta[0]);
    assert_eq!(x_plus_delta[2], x[2] + delta[1]);

    let mut actual_jacobian = [0.0; 3 * 2];
    assert!(product_param.compute_jacobian(&x, &mut actual_jacobian));
}

#[test]
fn subset_parameterization_product_parameterization_with_zero_local_size_subset_parameterization2()
{
    let constant_parameters = vec![0];
    let subset_param: Box<dyn LocalParameterization> =
        Box::new(SubsetParameterization::new(1, &constant_parameters));
    let identity_param: Box<dyn LocalParameterization> = Box::new(IdentityParameterization::new(2));
    let product_param = ProductParameterization::new(vec![identity_param, subset_param]);
    assert_eq!(product_param.global_size(), 3);
    assert_eq!(product_param.local_size(), 2);
    let x = [1.0, 1.0, 1.0];
    let delta = [2.0, 3.0];
    let mut x_plus_delta = [0.0; 3];
    assert!(product_param.plus(&x, &delta, &mut x_plus_delta));
    assert_eq!(x_plus_delta[0], x[0] + delta[0]);
    assert_eq!(x_plus_delta[1], x[1] + delta[1]);
    assert_eq!(x_plus_delta[2], x[2]);

    let mut actual_jacobian = [0.0; 3 * 2];
    assert!(product_param.compute_jacobian(&x, &mut actual_jacobian));
}

#[test]
fn subset_parameterization_normal_function_test() {
    const GLOBAL_SIZE: usize = 4;
    const LOCAL_SIZE: usize = 3;

    let x = [1.0, 2.0, 3.0, 4.0];
    for i in 0..GLOBAL_SIZE {
        let constant_parameters = vec![i];
        let parameterization = SubsetParameterization::new(GLOBAL_SIZE, &constant_parameters);
        let delta = [1.0, 2.0, 3.0];
        let mut x_plus_delta = [0.0; GLOBAL_SIZE];

        parameterization.plus(&x, &delta, &mut x_plus_delta);
        let mut k = 0;
        for j in 0..GLOBAL_SIZE {
            if j == i {
                assert_eq!(x_plus_delta[j], x[j]);
            } else {
                assert_eq!(x_plus_delta[j], x[j] + delta[k]);
                k += 1;
            }
        }

        let mut jacobian = [0.0; GLOBAL_SIZE * LOCAL_SIZE];
        parameterization.compute_jacobian(&x, &mut jacobian);
        let mut delta_cursor = 0;
        let mut jacobian_cursor = 0;
        for j in 0..GLOBAL_SIZE {
            if j != i {
                for k in 0..LOCAL_SIZE {
                    assert_eq!(
                        jacobian[jacobian_cursor],
                        if delta_cursor == k { 1.0 } else { 0.0 }
                    );
                    jacobian_cursor += 1;
                }
                delta_cursor += 1;
            } else {
                for _ in 0..LOCAL_SIZE {
                    assert_eq!(jacobian[jacobian_cursor], 0.0);
                    jacobian_cursor += 1;
                }
            }
        }

        let mut global_matrix = vec![1.0; 10 * GLOBAL_SIZE];
        for row in 0..GLOBAL_SIZE {
            for col in 0..GLOBAL_SIZE {
                global_matrix[row * GLOBAL_SIZE + col] = col as f64;
            }
        }

        let mut local_matrix = vec![0.0; 10 * LOCAL_SIZE];
        parameterization.multiply_by_jacobian(&x, 10, &global_matrix, &mut local_matrix);
        let expected_local_matrix = mat_mul(&global_matrix, 10, GLOBAL_SIZE, &jacobian, LOCAL_SIZE);
        assert_eq!(diff_norm(&local_matrix, &expected_local_matrix), 0.0);
    }
}

// ---------------------------------------------------------------------------
// Quaternion parameterizations
// ---------------------------------------------------------------------------

/// Functor needed to implement automatically differentiated Plus for
/// quaternions.
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

/// Functor needed to implement automatically differentiated Plus for
/// Eigen's quaternion (nalgebra's `Quaternion::coords` has the same
/// `[x, y, z, w]` layout).
#[derive(Default)]
struct EigenQuaternionPlus;

impl PlusFunctor for EigenQuaternionPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        let norm_delta = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();

        // Note: w is first in the constructor.
        let q_delta = if norm_delta > T::constant(0.0) {
            let sin_delta_by_delta = norm_delta.sin() / norm_delta;
            Quaternion::new(
                norm_delta.cos(),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2],
            )
        } else {
            // We do not just use q_delta = [0,0,0,1] here because that is a
            // constant and when used for automatic differentiation will
            // lead to a zero derivative. Instead we take a first order
            // approximation and evaluate it at zero.
            Quaternion::new(T::constant(1.0), delta[0], delta[1], delta[2])
        };

        let x_ref = Quaternion::new(x[3], x[0], x[1], x[2]);
        let result = q_delta * x_ref;
        x_plus_delta[..4].copy_from_slice(result.coords.as_slice());
        true
    }
}

/// Adapts a `PlusFunctor` into a two-block (x, delta) `CostFunctor`, standing
/// in for C++ `internal::AutoDifferentiate<kGlobalSize,
/// StaticParameterDims<kGlobalSize, kLocalSize>>(Plus(), ...)`.
struct PlusAsCostFunctor<F>(F);

impl<F: PlusFunctor> CostFunctor for PlusAsCostFunctor<F> {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residuals: &mut [T]) -> bool {
        self.0.plus(parameters[0], parameters[1], residuals)
    }
}

fn quaternion_parameterization_test_helper<P, F>(
    parameterization: P,
    plus: F,
    x: &[f64],
    delta: &[f64],
    x_plus_delta_ref: &[f64],
) where
    P: LocalParameterization,
    F: PlusFunctor,
{
    const GLOBAL_SIZE: usize = 4;
    const LOCAL_SIZE: usize = 3;

    const TOLERANCE: f64 = 1e-14;

    let mut x_plus_delta = [0.0; GLOBAL_SIZE];
    parameterization.plus(x, delta, &mut x_plus_delta);
    // Upstream compares `x_plus_delta[i]` with itself (a typo); the intended
    // comparison against the reference value is made here.
    for i in 0..GLOBAL_SIZE {
        assert_near(x_plus_delta[i], x_plus_delta_ref[i], TOLERANCE);
    }

    let x_plus_delta_norm = (x_plus_delta[0] * x_plus_delta[0]
        + x_plus_delta[1] * x_plus_delta[1]
        + x_plus_delta[2] * x_plus_delta[2]
        + x_plus_delta[3] * x_plus_delta[3])
        .sqrt();

    assert_near(x_plus_delta_norm, 1.0, TOLERANCE);

    let mut jacobian_ref = [0.0; 12];
    let zero_delta = [0.0; LOCAL_SIZE];
    let parameters: [&[f64]; 2] = [x, &zero_delta];

    // Autodiff jacobian at delta_x = 0.
    let autodiff = AutoDiffCostFunction::<_, { GLOBAL_SIZE + LOCAL_SIZE }>::new(
        PlusAsCostFunctor(plus),
        GLOBAL_SIZE,
        &[GLOBAL_SIZE, LOCAL_SIZE],
    );
    {
        let mut jacobian_array: [Option<&mut [f64]>; 2] = [None, Some(&mut jacobian_ref[..])];
        assert!(CostFunction::evaluate(
            &autodiff,
            &parameters,
            &mut x_plus_delta,
            Some(&mut jacobian_array),
        ));
    }

    let mut jacobian = [0.0; 12];
    parameterization.compute_jacobian(x, &mut jacobian);
    for i in 0..12 {
        assert!(jacobian[i].is_finite());
        assert!(
            (jacobian[i] - jacobian_ref[i]).abs() <= TOLERANCE,
            "Jacobian mismatch: i = {i}\n Expected \n{jacobian_ref:?}\n Actual \n{jacobian:?}"
        );
    }

    let mut rng = StdRng::seed_from_u64(42);
    let global_matrix = random_matrix(&mut rng, 10, GLOBAL_SIZE);
    let mut local_matrix = vec![0.0; 10 * LOCAL_SIZE];
    parameterization.multiply_by_jacobian(x, 10, &global_matrix, &mut local_matrix);
    let expected_local_matrix = mat_mul(&global_matrix, 10, GLOBAL_SIZE, &jacobian, LOCAL_SIZE);
    assert_near(
        diff_norm(&local_matrix, &expected_local_matrix),
        0.0,
        10.0 * f64::EPSILON,
    );
}

#[test]
fn quaternion_parameterization_zero_test() {
    let x = [0.5, 0.5, 0.5, 0.5];
    let delta = [0.0, 0.0, 0.0];
    let q_delta = [1.0, 0.0, 0.0, 0.0];
    let mut x_plus_delta = [0.0; 4];
    quaternion_product(&q_delta, &x, &mut x_plus_delta);
    quaternion_parameterization_test_helper(
        QuaternionParameterization,
        QuaternionPlus,
        &x,
        &delta,
        &x_plus_delta,
    );
}

#[test]
fn quaternion_parameterization_near_zero_test() {
    let mut x = [0.52, 0.25, 0.15, 0.45];
    normalize(&mut x);

    let mut delta = [0.24, 0.15, 0.10];
    for d in &mut delta {
        *d *= 1e-14;
    }

    let q_delta = [1.0, delta[0], delta[1], delta[2]];

    let mut x_plus_delta = [0.0; 4];
    quaternion_product(&q_delta, &x, &mut x_plus_delta);
    quaternion_parameterization_test_helper(
        QuaternionParameterization,
        QuaternionPlus,
        &x,
        &delta,
        &x_plus_delta,
    );
}

#[test]
fn quaternion_parameterization_away_from_zero_test() {
    let mut x = [0.52, 0.25, 0.15, 0.45];
    normalize(&mut x);

    let delta: [f64; 3] = [0.24, 0.15, 0.10];
    let delta_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let q_delta = [
        delta_norm.cos(),
        delta_norm.sin() / delta_norm * delta[0],
        delta_norm.sin() / delta_norm * delta[1],
        delta_norm.sin() / delta_norm * delta[2],
    ];

    let mut x_plus_delta = [0.0; 4];
    quaternion_product(&q_delta, &x, &mut x_plus_delta);
    quaternion_parameterization_test_helper(
        QuaternionParameterization,
        QuaternionPlus,
        &x,
        &delta,
        &x_plus_delta,
    );
}

#[test]
fn eigen_quaternion_parameterization_zero_test() {
    let x = Quaternion::new(0.5, 0.5, 0.5, 0.5);
    let delta = [0.0, 0.0, 0.0];
    let q_delta = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let x_plus_delta = q_delta * x;
    quaternion_parameterization_test_helper(
        EigenQuaternionParameterization,
        EigenQuaternionPlus,
        x.coords.as_slice(),
        &delta,
        x_plus_delta.coords.as_slice(),
    );
}

#[test]
fn eigen_quaternion_parameterization_near_zero_test() {
    let x = Quaternion::new(0.52, 0.25, 0.15, 0.45).normalize();

    let mut delta = [0.24, 0.15, 0.10];
    for d in &mut delta {
        *d *= 1e-14;
    }

    // Note: w is first in the constructor.
    let q_delta = Quaternion::new(1.0, delta[0], delta[1], delta[2]);

    let x_plus_delta = q_delta * x;
    quaternion_parameterization_test_helper(
        EigenQuaternionParameterization,
        EigenQuaternionPlus,
        x.coords.as_slice(),
        &delta,
        x_plus_delta.coords.as_slice(),
    );
}

#[test]
fn eigen_quaternion_parameterization_away_from_zero_test() {
    let x = Quaternion::new(0.52, 0.25, 0.15, 0.45).normalize();

    let delta: [f64; 3] = [0.24, 0.15, 0.10];
    let delta_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();

    // Note: w is first in the constructor.
    let q_delta = Quaternion::new(
        delta_norm.cos(),
        delta_norm.sin() / delta_norm * delta[0],
        delta_norm.sin() / delta_norm * delta[1],
        delta_norm.sin() / delta_norm * delta[2],
    );

    let x_plus_delta = q_delta * x;
    quaternion_parameterization_test_helper(
        EigenQuaternionParameterization,
        EigenQuaternionPlus,
        x.coords.as_slice(),
        &delta,
        x_plus_delta.coords.as_slice(),
    );
}

// ---------------------------------------------------------------------------
// HomogeneousVectorParameterization
// ---------------------------------------------------------------------------

/// Port of `internal::ComputeHouseholderVector` (householder_vector.h),
/// generic over the scalar type. Returns beta.
fn compute_householder_vector<T: Scalar>(x: &[T], v: &mut [T]) -> T {
    let n = x.len();
    assert!(n > 1);
    assert_eq!(n, v.len());

    let mut sigma = T::constant(0.0);
    for &xi in &x[..n - 1] {
        sigma += xi * xi;
    }
    v.copy_from_slice(x);
    v[n - 1] = T::constant(1.0);

    let mut beta = T::constant(0.0);
    let x_pivot = x[n - 1];

    if sigma <= T::constant(f64::EPSILON) {
        if x_pivot < T::constant(0.0) {
            beta = T::constant(2.0);
        }
        return beta;
    }

    let mu = (x_pivot * x_pivot + sigma).sqrt();
    let v_pivot = if x_pivot <= T::constant(0.0) {
        x_pivot - mu
    } else {
        -sigma / (x_pivot + mu)
    };

    beta = T::constant(2.0) * v_pivot * v_pivot / (sigma + v_pivot * v_pivot);

    for vi in &mut v[..n - 1] {
        *vi /= v_pivot;
    }
    beta
}

/// Functor needed to implement automatically differentiated Plus for
/// homogeneous vectors.
#[derive(Default)]
struct HomogeneousVectorParameterizationPlus<const DIM: usize>;

impl<const DIM: usize> PlusFunctor for HomogeneousVectorParameterizationPlus<DIM> {
    fn plus<T: Scalar>(&self, p_x: &[T], p_delta: &[T], p_x_plus_delta: &mut [T]) -> bool {
        let x = &p_x[..DIM];
        let delta = &p_delta[..DIM - 1];

        let mut squared_norm_delta = T::constant(0.0);
        for &d in delta {
            squared_norm_delta += d * d;
        }

        let mut y = [T::constant(0.0); DIM];
        let one_half = T::constant(0.5);
        if squared_norm_delta > T::constant(0.0) {
            let norm_delta = squared_norm_delta.sqrt();
            let norm_delta_div_2 = T::constant(0.5) * norm_delta;
            let sin_delta_by_delta = norm_delta_div_2.sin() / norm_delta_div_2;
            for i in 0..DIM - 1 {
                y[i] = sin_delta_by_delta * one_half * delta[i];
            }
            y[DIM - 1] = norm_delta_div_2.cos();
        } else {
            // We do not just use y = [0,0,0,1] here because that is a
            // constant and when used for automatic differentiation will
            // lead to a zero derivative. Instead we take a first order
            // approximation and evaluate it at zero.
            for i in 0..DIM - 1 {
                y[i] = delta[i] * one_half;
            }
            y[DIM - 1] = T::constant(1.0);
        }

        let mut v = [T::constant(0.0); DIM];
        let beta = compute_householder_vector(x, &mut v);

        let mut x_norm = T::constant(0.0);
        let mut v_dot_y = T::constant(0.0);
        for i in 0..DIM {
            x_norm += x[i] * x[i];
            v_dot_y += v[i] * y[i];
        }
        let x_norm = x_norm.sqrt();
        for i in 0..DIM {
            p_x_plus_delta[i] = x_norm * (y[i] - v[i] * (beta * v_dot_y));
        }
        true
    }
}

fn homogeneous_vector_parameterization_helper(x: &[f64], delta: &[f64]) {
    const TOLERANCE: f64 = 1e-14;

    let homogeneous_vector_parameterization = HomogeneousVectorParameterization::new(4);

    // Ensure the update maintains the norm.
    let mut x_plus_delta = [0.0; 4];
    homogeneous_vector_parameterization.plus(x, delta, &mut x_plus_delta);

    let x_plus_delta_norm = (x_plus_delta[0] * x_plus_delta[0]
        + x_plus_delta[1] * x_plus_delta[1]
        + x_plus_delta[2] * x_plus_delta[2]
        + x_plus_delta[3] * x_plus_delta[3])
        .sqrt();

    let x_norm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();

    assert_near(x_plus_delta_norm, x_norm, TOLERANCE);

    // Autodiff jacobian at delta_x = 0.
    let autodiff_jacobian =
        AutoDiffLocalParameterization::<HomogeneousVectorParameterizationPlus<4>, 4, 3>::default();

    let mut jacobian_autodiff = [0.0; 12];
    let mut jacobian_analytic = [0.0; 12];

    homogeneous_vector_parameterization.compute_jacobian(x, &mut jacobian_analytic);
    autodiff_jacobian.compute_jacobian(x, &mut jacobian_autodiff);

    for i in 0..12 {
        assert!(jacobian_analytic[i].is_finite());
        assert!(
            (jacobian_analytic[i] - jacobian_autodiff[i]).abs() <= TOLERANCE,
            "Jacobian mismatch: i = {i}, {} {}",
            jacobian_analytic[i],
            jacobian_autodiff[i]
        );
    }
}

#[test]
fn homogeneous_vector_parameterization_zero_test() {
    let mut x = [0.0, 0.0, 0.0, 1.0];
    normalize(&mut x);
    let delta = [0.0, 0.0, 0.0];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_near_zero_test1() {
    let mut x = [1e-5, 1e-5, 1e-5, 1.0];
    normalize(&mut x);
    let delta = [0.0, 1.0, 0.0];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_near_zero_test2() {
    let x = [0.001, 0.0, 0.0, 0.0];
    let delta = [0.0, 1.0, 0.0];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_away_from_zero_test1() {
    let mut x = [0.52, 0.25, 0.15, 0.45];
    normalize(&mut x);
    let delta = [0.0, 1.0, -0.5];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_away_from_zero_test2() {
    let mut x = [0.87, -0.25, -0.34, 0.45];
    normalize(&mut x);
    let delta = [0.0, 0.0, -0.5];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_away_from_zero_test3() {
    let x = [0.0, 0.0, 0.0, 2.0];
    let delta = [0.0, 0.0, 0.0];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_away_from_zero_test4() {
    let x = [0.2, -1.0, 0.0, 2.0];
    let delta = [1.4, 0.0, -0.5];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
fn homogeneous_vector_parameterization_away_from_zero_test5() {
    let x = [2.0, 0.0, 0.0, 0.0];
    let delta = [1.4, 0.0, -0.5];

    homogeneous_vector_parameterization_helper(&x, &delta);
}

#[test]
#[should_panic(expected = "size")]
fn homogeneous_vector_parameterization_death_tests() {
    let _x = HomogeneousVectorParameterization::new(1);
}

// LineParameterization tests (ZeroTest3D, ZeroTest4D, ZeroOriginPointTest3D,
// ZeroOriginPointTest4D, ZeroDirTest3D, ZeroDirTest4D, AwayFromZeroTest3D1,
// AwayFromZeroTest4D1, AwayFromZeroTest3D2, AwayFromZeroTest4D2) are skipped:
// LineParameterization is not ported in this crate.

// ---------------------------------------------------------------------------
// ProductParameterization
// ---------------------------------------------------------------------------

/// The `ProductParameterizationTest` fixture. The parameterizations are
/// shared through `Arc` so the test can keep using them after handing them
/// to the product (C++ keeps raw pointers after `release()`).
struct ProductParameterizationFixture {
    param1: Arc<SubsetParameterization>,
    param2: Arc<SubsetParameterization>,
    param3: Arc<SubsetParameterization>,
    param4: Arc<SubsetParameterization>,
}

impl ProductParameterizationFixture {
    fn set_up() -> Self {
        let global_size1 = 5;
        let constant_parameters1 = vec![2];
        let param1 = Arc::new(SubsetParameterization::new(
            global_size1,
            &constant_parameters1,
        ));

        let global_size2 = 3;
        let constant_parameters2 = vec![0, 1];
        let param2 = Arc::new(SubsetParameterization::new(
            global_size2,
            &constant_parameters2,
        ));

        let global_size3 = 4;
        let constant_parameters3 = vec![1];
        let param3 = Arc::new(SubsetParameterization::new(
            global_size3,
            &constant_parameters3,
        ));

        let global_size4 = 2;
        let constant_parameters4 = vec![1];
        let param4 = Arc::new(SubsetParameterization::new(
            global_size4,
            &constant_parameters4,
        ));

        Self {
            param1,
            param2,
            param3,
            param4,
        }
    }
}

fn boxed(p: &Arc<SubsetParameterization>) -> Box<dyn LocalParameterization> {
    Box::new(Arc::clone(p))
}

#[test]
fn product_parameterization_test_local_and_global_size2() {
    let f = ProductParameterizationFixture::set_up();
    let (param1, param2) = (&f.param1, &f.param2);

    let product_param = ProductParameterization::new(vec![boxed(param1), boxed(param2)]);
    assert_eq!(
        product_param.local_size(),
        param1.local_size() + param2.local_size()
    );
    assert_eq!(
        product_param.global_size(),
        param1.global_size() + param2.global_size()
    );
}

#[test]
fn product_parameterization_test_local_and_global_size3() {
    let f = ProductParameterizationFixture::set_up();
    let (param1, param2, param3) = (&f.param1, &f.param2, &f.param3);

    let product_param =
        ProductParameterization::new(vec![boxed(param1), boxed(param2), boxed(param3)]);
    assert_eq!(
        product_param.local_size(),
        param1.local_size() + param2.local_size() + param3.local_size()
    );
    assert_eq!(
        product_param.global_size(),
        param1.global_size() + param2.global_size() + param3.global_size()
    );
}

#[test]
fn product_parameterization_test_local_and_global_size4() {
    let f = ProductParameterizationFixture::set_up();
    let (param1, param2, param3, param4) = (&f.param1, &f.param2, &f.param3, &f.param4);

    let product_param = ProductParameterization::new(vec![
        boxed(param1),
        boxed(param2),
        boxed(param3),
        boxed(param4),
    ]);
    assert_eq!(
        product_param.local_size(),
        param1.local_size() + param2.local_size() + param3.local_size() + param4.local_size()
    );
    assert_eq!(
        product_param.global_size(),
        param1.global_size() + param2.global_size() + param3.global_size() + param4.global_size()
    );
}

#[test]
fn product_parameterization_test_plus() {
    let f = ProductParameterizationFixture::set_up();
    let params: [&Arc<SubsetParameterization>; 4] = [&f.param1, &f.param2, &f.param3, &f.param4];

    let product_param = ProductParameterization::new(params.iter().map(|p| boxed(p)).collect());
    let mut rng = StdRng::seed_from_u64(7);
    let x: Vec<f64> = (0..product_param.global_size())
        .map(|_| rand_normal(&mut rng))
        .collect();
    let delta: Vec<f64> = (0..product_param.local_size())
        .map(|_| rand_normal(&mut rng))
        .collect();
    let mut x_plus_delta_expected = vec![0.0; product_param.global_size()];
    let mut x_plus_delta = vec![0.0; product_param.global_size()];

    assert!(product_param.plus(&x, &delta, &mut x_plus_delta_expected));
    let mut x_cursor = 0;
    let mut delta_cursor = 0;

    for param in params {
        assert!(param.plus(
            &x[x_cursor..],
            &delta[delta_cursor..],
            &mut x_plus_delta[x_cursor..]
        ));
        x_cursor += param.global_size();
        delta_cursor += param.local_size();
    }

    for i in 0..x.len() {
        assert_eq!(x_plus_delta[i], x_plus_delta_expected[i]);
    }
}

#[test]
fn product_parameterization_test_compute_jacobian() {
    let f = ProductParameterizationFixture::set_up();
    let params: [&Arc<SubsetParameterization>; 4] = [&f.param1, &f.param2, &f.param3, &f.param4];

    let product_param = ProductParameterization::new(params.iter().map(|p| boxed(p)).collect());
    let mut rng = StdRng::seed_from_u64(11);
    let x: Vec<f64> = (0..product_param.global_size())
        .map(|_| rand_normal(&mut rng))
        .collect();

    let global_size = product_param.global_size();
    let local_size = product_param.local_size();
    let mut jacobian = random_matrix(&mut rng, global_size, local_size);
    assert!(product_param.compute_jacobian(&x, &mut jacobian));
    let mut x_cursor = 0;
    let mut delta_cursor = 0;

    for param in params {
        let (g, l) = (param.global_size(), param.local_size());
        let mut jacobian_i = vec![0.0; g * l];
        assert!(param.compute_jacobian(&x[x_cursor..], &mut jacobian_i));
        for r in 0..g {
            for c in 0..l {
                jacobian[(x_cursor + r) * local_size + delta_cursor + c] -= jacobian_i[r * l + c];
            }
        }
        x_cursor += g;
        delta_cursor += l;
    }

    let norm = jacobian.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_near(norm, 0.0, f64::EPSILON);
}
