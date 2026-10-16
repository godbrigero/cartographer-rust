// Ported from Ceres Solver 2.0.0: internal/ceres/rotation_test.cc
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)
//
// RUST PORT NOTES:
// - The gmock matchers (IsNormalizedQuaternion, IsNearQuaternion,
//   IsNearAngleAxis, IsOrthonormal, IsNear3x3Matrix) are plain functions
//   returning `Result<(), String>` (the `Err` carries the matcher's message).
// - `RandDouble()` / `srand(5)` become a seeded `StdRng`; the random streams
//   do not match the C++ ones.
// - Eigen vectors / matrices map to nalgebra (`Vector3`, `Matrix3`).

use ceres::Jet;
use ceres::rotation::{
    MatrixAdapter, angle_axis_rotate_point, angle_axis_to_quaternion,
    angle_axis_to_rotation_matrix, angle_axis_to_rotation_matrix_adapter, column_major_adapter_3x3,
    const_column_major_adapter_3x3, const_row_major_adapter_3x3, euler_angles_to_rotation_matrix,
    quaternion_product, quaternion_to_angle_axis, quaternion_to_rotation,
    quaternion_to_scaled_rotation, rotation_matrix_to_angle_axis,
    rotation_matrix_to_angle_axis_adapter, row_major_adapter_3x3, unit_quaternion_rotate_point,
};
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// C++: 3.14159265358979323846 and 0.707106781186547524401 (the same doubles).
const K_PI: f64 = std::f64::consts::PI;
const K_HALF_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// A tolerance value for floating-point comparisons.
const K_TOLERANCE: f64 = f64::EPSILON * 10.0;

/// Looser tolerance used for numerically unstable conversions.
const K_LOOSE_TOLERANCE: f64 = 1e-9;

const K_NUM_TRIALS: usize = 10000;

/// C++ `RandDouble()`: a uniform value in [0, 1].
fn rand_double(rng: &mut StdRng) -> f64 {
    rng.random::<f64>()
}

/// C++ `srand(5)`.
fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(5)
}

// ---------------------------------------------------------------------------
// Matchers.
// ---------------------------------------------------------------------------

/// MATCHER(IsNormalizedQuaternion).
fn is_normalized_quaternion(arg: &[f64]) -> Result<(), String> {
    let norm2 = arg[0] * arg[0] + arg[1] * arg[1] + arg[2] * arg[2] + arg[3] * arg[3];
    if (norm2 - 1.0).abs() > K_TOLERANCE {
        return Err(format!("squared norm is {norm2}"));
    }
    Ok(())
}

/// MATCHER_P(IsNearQuaternion, expected).
fn is_near_quaternion(arg: &[f64], expected: &[f64]) -> Result<(), String> {
    // Quaternions are equivalent upto a sign change. So we will compare
    // both signs before declaring failure.
    if (0..4).all(|i| (arg[i] - expected[i]).abs() <= K_TOLERANCE) {
        return Ok(());
    }
    if (0..4).all(|i| (arg[i] + expected[i]).abs() <= K_TOLERANCE) {
        return Ok(());
    }
    Err(format!(
        "expected : {} {} {} {} actual : {} {} {} {}",
        expected[0], expected[1], expected[2], expected[3], arg[0], arg[1], arg[2], arg[3]
    ))
}

/// MATCHER_P(IsNearAngleAxis, expected).
fn is_near_angle_axis(arg: &[f64], expected: &[f64]) -> Result<(), String> {
    let a = Vector3::new(arg[0], arg[1], arg[2]);
    let e = Vector3::new(expected[0], expected[1], expected[2]);
    let e_norm = e.norm();

    let delta_norm = if e_norm > 0.0 {
        // Deal with the sign ambiguity near PI. Since the sign can flip,
        // we take the smaller of the two differences.
        if (e_norm - K_PI).abs() < K_LOOSE_TOLERANCE {
            (a - e).norm().min((a + e).norm()) / e_norm
        } else {
            (a - e).norm() / e_norm
        }
    } else {
        a.norm()
    };

    if delta_norm <= K_LOOSE_TOLERANCE {
        return Ok(());
    }
    Err(format!(
        " arg: {} {} {} was expected to be: {} {} {}",
        arg[0], arg[1], arg[2], expected[0], expected[1], expected[2]
    ))
}

/// MATCHER(IsOrthonormal).
fn is_orthonormal(arg: &[f64]) -> Result<(), String> {
    for c1 in 0..3 {
        for c2 in 0..3 {
            let mut v = 0.0;
            for i in 0..3 {
                v += arg[i + 3 * c1] * arg[i + 3 * c2];
            }
            let expected = if c1 == c2 { 1.0 } else { 0.0 };
            if (expected - v).abs() > K_TOLERANCE {
                return Err(format!(
                    "Columns {c1} and {c2} should have dot product {expected} but have {v}"
                ));
            }
        }
    }
    Ok(())
}

/// MATCHER_P(IsNear3x3Matrix, expected).
fn is_near_3x3_matrix(arg: &[f64], expected: &[f64]) -> Result<(), String> {
    for i in 0..9 {
        if (arg[i] - expected[i]).abs() > K_TOLERANCE {
            return Err(format!(
                "component {i} should be {} but is {}",
                expected[i], arg[i]
            ));
        }
    }
    Ok(())
}

/// EXPECT_THAT / ASSERT_THAT.
#[track_caller]
fn expect_that(result: Result<(), String>) {
    if let Err(msg) = result {
        panic!("{msg}");
    }
}

/// EXPECT_NEAR.
#[track_caller]
fn expect_near(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected |{a} - {b}| = {} <= {tol}",
        (a - b).abs()
    );
}

/// test_util.cc `ExpectClose` (relative difference, absolute if either is 0).
fn expect_close(x: f64, y: f64, max_abs_relative_difference: f64) -> bool {
    if x.is_infinite() && y.is_infinite() {
        assert_eq!(x.is_sign_negative(), y.is_sign_negative());
        return true;
    }
    if x.is_nan() && y.is_nan() {
        return true;
    }
    let absolute_difference = (x - y).abs();
    let mut relative_difference = absolute_difference / x.abs().max(y.abs());
    if x == 0.0 || y == 0.0 {
        relative_difference = absolute_difference;
    }
    relative_difference.abs() <= max_abs_relative_difference
}

/// test_util.cc `ExpectArraysClose`.
#[track_caller]
fn expect_arrays_close(n: usize, p: &[f64], q: &[f64], tol: f64) {
    for i in 0..n {
        assert!(
            expect_close(p[i], q[i], tol),
            "p[{i}]{} q[{i}]{} tol: {tol}",
            p[i],
            q[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// Transforms a zero axis/angle to a quaternion.
#[test]
fn rotation_zero_angle_axis_to_quaternion() {
    let axis_angle = [0.0, 0.0, 0.0];
    let mut quaternion = [0.0; 4];
    let expected = [1.0, 0.0, 0.0, 0.0];
    angle_axis_to_quaternion(&axis_angle, &mut quaternion);
    expect_that(is_normalized_quaternion(&quaternion));
    expect_that(is_near_quaternion(&quaternion, &expected));
}

// Test that exact conversion works for small angles.
#[test]
fn rotation_small_angle_axis_to_quaternion() {
    // Small, finite value to test.
    let theta: f64 = 1.0e-2;
    let axis_angle = [theta, 0.0, 0.0];
    let mut quaternion = [0.0; 4];
    let expected = [(theta / 2.0).cos(), (theta / 2.0).sin(), 0.0, 0.0];
    angle_axis_to_quaternion(&axis_angle, &mut quaternion);
    expect_that(is_normalized_quaternion(&quaternion));
    expect_that(is_near_quaternion(&quaternion, &expected));
}

// Test that approximate conversion works for very small angles.
#[test]
fn rotation_tiny_angle_axis_to_quaternion() {
    // Very small value that could potentially cause underflow.
    let theta = f64::MIN_POSITIVE.powf(0.75);
    let axis_angle = [theta, 0.0, 0.0];
    let mut quaternion = [0.0; 4];
    let expected = [(theta / 2.0).cos(), (theta / 2.0).sin(), 0.0, 0.0];
    angle_axis_to_quaternion(&axis_angle, &mut quaternion);
    expect_that(is_normalized_quaternion(&quaternion));
    expect_that(is_near_quaternion(&quaternion, &expected));
}

// Transforms a rotation by pi/2 around X to a quaternion.
#[test]
fn rotation_x_rotation_to_quaternion() {
    let axis_angle = [K_PI / 2.0, 0.0, 0.0];
    let mut quaternion = [0.0; 4];
    let expected = [K_HALF_SQRT2, K_HALF_SQRT2, 0.0, 0.0];
    angle_axis_to_quaternion(&axis_angle, &mut quaternion);
    expect_that(is_normalized_quaternion(&quaternion));
    expect_that(is_near_quaternion(&quaternion, &expected));
}

// Transforms a unit quaternion to an axis angle.
#[test]
fn rotation_unit_quaternion_to_angle_axis() {
    let quaternion = [1.0, 0.0, 0.0, 0.0];
    let mut axis_angle = [0.0; 3];
    let expected = [0.0, 0.0, 0.0];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_that(is_near_angle_axis(&axis_angle, &expected));
}

// Transforms a quaternion that rotates by pi about the Y axis to an axis angle.
#[test]
fn rotation_y_rotation_quaternion_to_angle_axis() {
    let quaternion = [0.0, 0.0, 1.0, 0.0];
    let mut axis_angle = [0.0; 3];
    let expected = [0.0, K_PI, 0.0];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_that(is_near_angle_axis(&axis_angle, &expected));
}

// Transforms a quaternion that rotates by pi/3 about the Z axis to an axis
// angle.
#[test]
fn rotation_z_rotation_quaternion_to_angle_axis() {
    let quaternion = [3.0_f64.sqrt() / 2.0, 0.0, 0.0, 0.5];
    let mut axis_angle = [0.0; 3];
    let expected = [0.0, 0.0, K_PI / 3.0];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_that(is_near_angle_axis(&axis_angle, &expected));
}

// Test that exact conversion works for small angles.
#[test]
fn rotation_small_quaternion_to_angle_axis() {
    // Small, finite value to test.
    let theta: f64 = 1.0e-2;
    let quaternion = [(theta / 2.0).cos(), (theta / 2.0).sin(), 0.0, 0.0];
    let mut axis_angle = [0.0; 3];
    let expected = [theta, 0.0, 0.0];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_that(is_near_angle_axis(&axis_angle, &expected));
}

// Test that approximate conversion works for very small angles.
#[test]
fn rotation_tiny_quaternion_to_angle_axis() {
    // Very small value that could potentially cause underflow.
    let theta = f64::MIN_POSITIVE.powf(0.75);
    let quaternion = [(theta / 2.0).cos(), (theta / 2.0).sin(), 0.0, 0.0];
    let mut axis_angle = [0.0; 3];
    let expected = [theta, 0.0, 0.0];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_that(is_near_angle_axis(&axis_angle, &expected));
}

#[test]
fn rotation_quaternion_to_angle_axis_angle_is_less_than_pi() {
    let mut angle_axis = [0.0; 3];

    let half_theta = 0.75 * K_PI;

    let quaternion = [half_theta.cos(), half_theta.sin(), 0.0, 0.0];
    quaternion_to_angle_axis(&quaternion, &mut angle_axis);
    let angle = (angle_axis[0] * angle_axis[0]
        + angle_axis[1] * angle_axis[1]
        + angle_axis[2] * angle_axis[2])
        .sqrt();
    assert!(angle <= K_PI, "angle {angle} > pi");
}

/// Random axis (three numbers in [-1, 1), normalized) scaled by `theta`
/// produced from the second random draw via `theta_of`.
fn random_axis_angle(rng: &mut StdRng, theta_of: impl Fn(f64) -> f64) -> [f64; 3] {
    let mut axis_angle = [0.0; 3];
    let mut norm = 0.0;
    for v in axis_angle.iter_mut() {
        *v = rand_double(rng) * 2.0 - 1.0;
        norm += *v * *v;
    }
    let norm = f64::sqrt(norm);
    let theta = theta_of(rand_double(rng));
    for v in axis_angle.iter_mut() {
        *v = *v * theta / norm;
    }
    axis_angle
}

// Takes a bunch of random axis/angle values, converts them to quaternions,
// and back again.
#[test]
fn rotation_angle_axis_to_quaterion_and_back() {
    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        // Angle in [-pi, pi).
        let axis_angle = random_axis_angle(&mut rng, |r| K_PI * 2.0 * r - K_PI);

        let mut quaternion = [0.0; 4];
        let mut round_trip = [0.0; 3];
        angle_axis_to_quaternion(&axis_angle, &mut quaternion);
        expect_that(is_normalized_quaternion(&quaternion));
        quaternion_to_angle_axis(&quaternion, &mut round_trip);
        expect_that(is_near_angle_axis(&round_trip, &axis_angle));
    }
}

// Takes a bunch of random quaternions, converts them to axis/angle,
// and back again.
#[test]
fn rotation_quaterion_to_angle_axis_and_back() {
    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        let mut quaternion = [0.0; 4];
        // Choose four random numbers in [-1, 1) and normalize.
        let mut norm = 0.0;
        for q in quaternion.iter_mut() {
            *q = rand_double(&mut rng) * 2.0 - 1.0;
            norm += *q * *q;
        }
        let norm = f64::sqrt(norm);
        for q in quaternion.iter_mut() {
            *q /= norm;
        }

        let mut axis_angle = [0.0; 3];
        let mut round_trip = [0.0; 4];
        quaternion_to_angle_axis(&quaternion, &mut axis_angle);
        angle_axis_to_quaternion(&axis_angle, &mut round_trip);
        expect_that(is_normalized_quaternion(&round_trip));
        expect_that(is_near_quaternion(&round_trip, &quaternion));
    }
}

// Transforms a zero axis/angle to a rotation matrix.
#[test]
fn rotation_zero_angle_axis_to_rotation_matrix() {
    let axis_angle = [0.0, 0.0, 0.0];
    let mut matrix = [0.0; 9];
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
    expect_that(is_orthonormal(&matrix));
    expect_that(is_near_3x3_matrix(&matrix, &expected));
}

#[test]
fn rotation_near_zero_angle_axis_to_rotation_matrix() {
    let axis_angle = [1e-24, 2e-24, 3e-24];
    let mut matrix = [0.0; 9];
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
    expect_that(is_orthonormal(&matrix));
    expect_that(is_near_3x3_matrix(&matrix, &expected));
}

// Transforms a rotation by pi/2 around X to a rotation matrix and back.
#[test]
fn rotation_x_rotation_to_rotation_matrix() {
    let axis_angle = [K_PI / 2.0, 0.0, 0.0];
    let mut matrix = [0.0; 9];
    // The rotation matrices are stored column-major.
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
    expect_that(is_orthonormal(&matrix));
    expect_that(is_near_3x3_matrix(&matrix, &expected));
    let mut round_trip = [0.0; 3];
    rotation_matrix_to_angle_axis(&matrix, &mut round_trip);
    expect_that(is_near_angle_axis(&round_trip, &axis_angle));
}

// Transforms an axis angle that rotates by pi about the Y axis to a
// rotation matrix and back.
#[test]
fn rotation_y_rotation_to_rotation_matrix() {
    let axis_angle = [0.0, K_PI, 0.0];
    let mut matrix = [0.0; 9];
    let expected = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
    expect_that(is_orthonormal(&matrix));
    expect_that(is_near_3x3_matrix(&matrix, &expected));

    let mut round_trip = [0.0; 3];
    rotation_matrix_to_angle_axis(&matrix, &mut round_trip);
    expect_that(is_near_angle_axis(&round_trip, &axis_angle));
}

#[test]
fn rotation_near_pi_angle_axis_round_trip() {
    let mut matrix = [0.0; 9];
    let mut out_axis_angle = [0.0; 3];

    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        // Angle in [pi - kMaxSmallAngle, pi).
        const K_MAX_SMALL_ANGLE: f64 = 1e-8;
        let in_axis_angle = random_axis_angle(&mut rng, |r| K_PI - K_MAX_SMALL_ANGLE * r);
        angle_axis_to_rotation_matrix(&in_axis_angle, &mut matrix);
        rotation_matrix_to_angle_axis(&matrix, &mut out_axis_angle);
        expect_that(is_near_angle_axis(&in_axis_angle, &out_axis_angle));
    }
}

#[test]
fn rotation_at_pi_angle_axis_round_trip() {
    // A rotation of kPi about the X axis;
    #[rustfmt::skip]
    const K_MATRIX: [[f64; 3]; 3] = [
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ];

    let mut in_matrix = [0.0; 9];
    // Fill it from kMatrix in col-major order.
    let mut k = 0;
    for j in 0..3 {
        for row in &K_MATRIX {
            in_matrix[k] = row[j];
            k += 1;
        }
    }

    let expected_axis_angle = [K_PI, 0.0, 0.0];

    let mut out_matrix = [0.0; 9];
    let mut axis_angle = [0.0; 3];
    rotation_matrix_to_angle_axis(&in_matrix, &mut axis_angle);
    angle_axis_to_rotation_matrix(&axis_angle, &mut out_matrix);

    // (The C++ LOG(INFO) diagnostics are omitted.)
    expect_that(is_near_angle_axis(&axis_angle, &expected_axis_angle));
    expect_that(is_near_3x3_matrix(&out_matrix, &in_matrix));
}

// Transforms an axis angle that rotates by pi/3 about the Z axis to a
// rotation matrix.
#[test]
fn rotation_z_rotation_to_rotation_matrix() {
    let axis_angle = [0.0, 0.0, K_PI / 3.0];
    let mut matrix = [0.0; 9];
    // This is laid-out row-major on the screen but is actually stored
    // column-major.
    let s3 = 3.0_f64.sqrt();
    #[rustfmt::skip]
    let expected = [
        0.5, s3 / 2.0, 0.0,   // Column 1
        -s3 / 2.0, 0.5, 0.0,  // Column 2
        0.0, 0.0, 1.0,        // Column 3
    ];
    angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
    expect_that(is_orthonormal(&matrix));
    expect_that(is_near_3x3_matrix(&matrix, &expected));
    let mut round_trip = [0.0; 3];
    rotation_matrix_to_angle_axis(&matrix, &mut round_trip);
    expect_that(is_near_angle_axis(&round_trip, &axis_angle));
}

// Takes a bunch of random axis/angle values, converts them to rotation
// matrices, and back again.
#[test]
fn rotation_angle_axis_to_rotation_matrix_and_back() {
    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        // Angle in [-pi, pi).
        let axis_angle = random_axis_angle(&mut rng, |r| K_PI * 2.0 * r - K_PI);

        let mut matrix = [0.0; 9];
        let mut round_trip = [0.0; 3];
        angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
        expect_that(is_orthonormal(&matrix));
        rotation_matrix_to_angle_axis(&matrix, &mut round_trip);

        for i in 0..3 {
            expect_near(round_trip[i], axis_angle[i], K_LOOSE_TOLERANCE);
        }
    }
}

// Takes a bunch of random axis/angle values near zero, converts them
// to rotation matrices, and back again.
#[test]
fn rotation_angle_axis_to_rotation_matrix_and_back_near_zero() {
    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        // Tiny theta.
        let axis_angle = random_axis_angle(&mut rng, |r| 1e-16 * (K_PI * 2.0 * r - K_PI));

        let mut matrix = [0.0; 9];
        let mut round_trip = [0.0; 3];
        angle_axis_to_rotation_matrix(&axis_angle, &mut matrix);
        expect_that(is_orthonormal(&matrix));
        rotation_matrix_to_angle_axis(&matrix, &mut round_trip);

        for i in 0..3 {
            expect_near(round_trip[i], axis_angle[i], f64::EPSILON);
        }
    }
}

/// Transposes a 3x3 matrix.
fn transpose_3x3(m: &mut [f64; 9]) {
    m.swap(1, 3);
    m.swap(2, 6);
    m.swap(5, 7);
}

/// Convert Euler angles from radians to degrees.
fn to_degrees(euler_angles: &mut [f64; 3]) {
    for e in euler_angles.iter_mut() {
        *e *= 180.0 / K_PI;
    }
}

/// Compare the 3x3 rotation matrices produced by the axis-angle
/// rotation 'aa' and the Euler angle rotation 'ea' (in radians).
fn compare_euler_to_angle_axis(aa: &[f64; 3], ea: &mut [f64; 3]) {
    let mut aa_matrix = [0.0; 9];
    angle_axis_to_rotation_matrix(aa, &mut aa_matrix);
    transpose_3x3(&mut aa_matrix); // Column to row major order.

    let mut ea_matrix = [0.0; 9];
    to_degrees(ea); // Radians to degrees.
    const K_ROW_STRIDE: i32 = 3;
    euler_angles_to_rotation_matrix(ea, K_ROW_STRIDE, &mut ea_matrix);

    expect_that(is_orthonormal(&aa_matrix));
    expect_that(is_orthonormal(&ea_matrix));
    expect_that(is_near_3x3_matrix(&ea_matrix, &aa_matrix));
}

// Test with rotation axis along the x/y/z axes.
// Also test zero rotation.
#[test]
fn euler_angles_to_rotation_matrix_on_axis() {
    let mut n_tests = 0;
    let values = [-1.0, 0.0, 1.0];
    for &x in &values {
        for &y in &values {
            for &z in &values {
                if usize::from(x != 0.0) + usize::from(y != 0.0) + usize::from(z != 0.0) > 1 {
                    continue;
                }
                let axis_angle = [x, y, z];
                let mut euler_angles = [x, y, z];
                compare_euler_to_angle_axis(&axis_angle, &mut euler_angles);
                n_tests += 1;
            }
        }
    }
    assert_eq!(7, n_tests);
}

// Test that a random rotation produces an orthonormal rotation
// matrix.
#[test]
fn euler_angles_to_rotation_matrix_is_orthonormal() {
    let mut rng = seeded_rng();
    for _ in 0..K_NUM_TRIALS {
        let mut euler_angles_degrees = [0.0; 3];
        for e in euler_angles_degrees.iter_mut() {
            *e = rand_double(&mut rng) * 360.0 - 180.0;
        }
        let mut rotation_matrix = [0.0; 9];
        euler_angles_to_rotation_matrix(&euler_angles_degrees, 3, &mut rotation_matrix);
        expect_that(is_orthonormal(&rotation_matrix));
    }
}

// Tests using Jets for specific behavior involving auto differentiation
// near singularity points.

type J3 = Jet<3>;
type J4 = Jet<4>;

fn make_j3(a: f64, v0: f64, v1: f64, v2: f64) -> J3 {
    J3::new(a, [v0, v1, v2])
}

fn make_j4(a: f64, v0: f64, v1: f64, v2: f64, v3: f64) -> J4 {
    J4::new(a, [v0, v1, v2, v3])
}

/// is_close.cc `IsClose(x, y, kTolerance)` plus the NaN checks of the
/// test's `IsClose` wrapper.
#[track_caller]
fn is_close(x: f64, y: f64) -> bool {
    assert!(!x.is_nan(), "x is NaN");
    assert!(!y.is_nan(), "y is NaN");
    let absolute_error = (x - y).abs();
    let mut relative_error = absolute_error / x.abs().max(y.abs());
    if x == 0.0 || y == 0.0 {
        relative_error = absolute_error;
    }
    relative_error < K_TOLERANCE.abs()
}

#[track_caller]
fn is_close_jet<const N: usize>(x: &Jet<N>, y: &Jet<N>) -> bool {
    if !is_close(x.a, y.a) {
        return false;
    }
    (0..N).all(|i| is_close(x.v[i], y.v[i]))
}

#[track_caller]
fn expect_jet_arrays_close<const M: usize, const N: usize>(x: &[Jet<N>; M], y: &[Jet<N>; M]) {
    let mut failures = Vec::new();
    for i in 0..M {
        if !is_close_jet(&x[i], &y[i]) {
            failures.push(format!(
                "Jet {i}/{M} not equal: x[{i}]: {:?} y[{i}]: {:?}",
                x[i], y[i]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Log-10 of a value well below machine precision.
fn k_small_tiny_cutoff() -> i32 {
    (2.0 * f64::EPSILON.ln() / 10.0_f64.ln()) as i32
}

/// Log-10 of a value just below values representable by double.
fn k_tiny_zero_limit() -> i32 {
    (1.0 + f64::MIN_POSITIVE.ln() / 10.0_f64.ln()) as i32
}

// Test that exact conversion works for small angles when jets are used.
#[test]
fn rotation_small_angle_axis_to_quaternion_for_jets() {
    // Examine small x rotations that are still large enough
    // to be well within the range represented by doubles.
    let mut i = -2;
    while i >= k_small_tiny_cutoff() {
        let theta = 10.0_f64.powi(i);
        let axis_angle = [
            J3::variable(theta, 0),
            J3::variable(0.0, 1),
            J3::variable(0.0, 2),
        ];
        let mut quaternion = [J3::constant(0.0); 4];
        let expected = [
            make_j3((theta / 2.0).cos(), -(theta / 2.0).sin() / 2.0, 0.0, 0.0),
            make_j3((theta / 2.0).sin(), (theta / 2.0).cos() / 2.0, 0.0, 0.0),
            make_j3(0.0, 0.0, (theta / 2.0).sin() / theta, 0.0),
            make_j3(0.0, 0.0, 0.0, (theta / 2.0).sin() / theta),
        ];
        angle_axis_to_quaternion(&axis_angle, &mut quaternion);
        expect_jet_arrays_close(&quaternion, &expected);
        i -= 1;
    }
}

// Test that conversion works for very small angles when jets are used.
#[test]
fn rotation_tiny_angle_axis_to_quaternion_for_jets() {
    // Examine tiny x rotations that extend all the way to where
    // underflow occurs.
    let mut i = k_small_tiny_cutoff();
    while i >= k_tiny_zero_limit() {
        let theta = 10.0_f64.powi(i);
        let axis_angle = [
            J3::variable(theta, 0),
            J3::variable(0.0, 1),
            J3::variable(0.0, 2),
        ];
        let mut quaternion = [J3::constant(0.0); 4];
        // To avoid loss of precision in the test itself,
        // a finite expansion is used here, which will
        // be exact up to machine precision for the test values used.
        let expected = [
            make_j3(1.0, 0.0, 0.0, 0.0),
            make_j3(0.0, 0.5, 0.0, 0.0),
            make_j3(0.0, 0.0, 0.5, 0.0),
            make_j3(0.0, 0.0, 0.0, 0.5),
        ];
        angle_axis_to_quaternion(&axis_angle, &mut quaternion);
        expect_jet_arrays_close(&quaternion, &expected);
        i -= 1;
    }
}

// Test that derivatives are correct for zero rotation.
#[test]
fn rotation_zero_angle_axis_to_quaternion_for_jets() {
    let axis_angle = [
        J3::variable(0.0, 0),
        J3::variable(0.0, 1),
        J3::variable(0.0, 2),
    ];
    let mut quaternion = [J3::constant(0.0); 4];
    let expected = [
        make_j3(1.0, 0.0, 0.0, 0.0),
        make_j3(0.0, 0.5, 0.0, 0.0),
        make_j3(0.0, 0.0, 0.5, 0.0),
        make_j3(0.0, 0.0, 0.0, 0.5),
    ];
    angle_axis_to_quaternion(&axis_angle, &mut quaternion);
    expect_jet_arrays_close(&quaternion, &expected);
}

// Test that exact conversion works for small angles.
#[test]
fn rotation_small_quaternion_to_angle_axis_for_jets() {
    // Examine small x rotations that are still large enough
    // to be well within the range represented by doubles.
    let mut i = -2;
    while i >= k_small_tiny_cutoff() {
        let theta = 10.0_f64.powi(i);
        let s = theta.sin();
        let c = theta.cos();
        let quaternion = [
            J4::variable(c, 0),
            J4::variable(s, 1),
            J4::variable(0.0, 2),
            J4::variable(0.0, 3),
        ];
        let mut axis_angle = [J4::constant(0.0); 3];
        #[rustfmt::skip]
        let expected = [
            make_j4(2.0 * theta, -2.0 * s, 2.0 * c, 0.0,               0.0),
            make_j4(0.0,          0.0,     0.0,     2.0 * theta / s,   0.0),
            make_j4(0.0,          0.0,     0.0,     0.0,               2.0 * theta / s),
        ];
        quaternion_to_angle_axis(&quaternion, &mut axis_angle);
        expect_jet_arrays_close(&axis_angle, &expected);
        i -= 1;
    }
}

// Test that conversion works for very small angles.
#[test]
fn rotation_tiny_quaternion_to_angle_axis_for_jets() {
    // Examine tiny x rotations that extend all the way to where
    // underflow occurs.
    let mut i = k_small_tiny_cutoff();
    while i >= k_tiny_zero_limit() {
        let theta = 10.0_f64.powi(i);
        let s = theta.sin();
        let c = theta.cos();
        let quaternion = [
            J4::variable(c, 0),
            J4::variable(s, 1),
            J4::variable(0.0, 2),
            J4::variable(0.0, 3),
        ];
        let mut axis_angle = [J4::constant(0.0); 3];
        // To avoid loss of precision in the test itself,
        // a finite expansion is used here, which will
        // be exact up to machine precision for the test values used.
        #[rustfmt::skip]
        let expected = [
            make_j4(2.0 * theta, -2.0 * s, 2.0, 0.0, 0.0),
            make_j4(0.0,          0.0,     0.0, 2.0, 0.0),
            make_j4(0.0,          0.0,     0.0, 0.0, 2.0),
        ];
        quaternion_to_angle_axis(&quaternion, &mut axis_angle);
        expect_jet_arrays_close(&axis_angle, &expected);
        i -= 1;
    }
}

// Test that conversion works for no rotation.
#[test]
fn rotation_zero_quaternion_to_angle_axis_for_jets() {
    let quaternion = [
        J4::variable(1.0, 0),
        J4::variable(0.0, 1),
        J4::variable(0.0, 2),
        J4::variable(0.0, 3),
    ];
    let mut axis_angle = [J4::constant(0.0); 3];
    let expected = [
        make_j4(0.0, 0.0, 2.0, 0.0, 0.0),
        make_j4(0.0, 0.0, 0.0, 2.0, 0.0),
        make_j4(0.0, 0.0, 0.0, 0.0, 2.0),
    ];
    quaternion_to_angle_axis(&quaternion, &mut axis_angle);
    expect_jet_arrays_close(&axis_angle, &expected);
}

#[test]
fn quaternion_rotate_point_gives_same_answer_as_rotation_by_matrix_canned() {
    // Canned data generated in octave.
    let q: [f64; 4] = [
        0.195_683_047_175_407_4,
        -0.015_061_856_247_484_7,
        0.763_457_298_278_808_6,
        -0.301_945_477_724_075_3,
    ];
    #[rustfmt::skip]
    let scaled_q: [f64; 9] = [
        // Scaled rotation matrix (row-major).
        -0.635_519_403_347_725_2, 0.095_173_054_168_225_4, 0.307_887_019_791_118_6,
        -0.141_169_390_479_299_2, 0.529_760_970_215_390_5, -0.455_150_257_448_201_9,
        -0.289_695_582_270_886_2, -0.466_939_657_154_705, -0.453_630_979_338_924_8,
    ];
    #[rustfmt::skip]
    let r: [f64; 9] = [
        // With unit rows and columns (row-major).
        -0.891_885_916_405_308, 0.133_565_562_572_564_9, 0.432_087_667_739_474_5,
        -0.198_116_675_168_009_6, 0.743_464_866_544_439_9, -0.638_756_428_722_585_6,
        -0.406_557_861_980_601_3, -0.655_301_634_904_669_3, -0.636_624_278_639_316_4,
    ];

    // Compute R from q and compare to known answer.
    let mut rq = [0.0; 9];
    quaternion_to_scaled_rotation(&q, &mut rq);
    expect_arrays_close(9, &scaled_q, &rq, K_TOLERANCE);

    // Now do the same but compute R with normalization.
    quaternion_to_rotation(&q, &mut rq);
    expect_arrays_close(9, &r, &rq, K_TOLERANCE);
}

#[test]
fn quaternion_rotate_point_gives_same_answer_as_rotation_by_matrix() {
    // Rotation defined by a unit quaternion.
    let q: [f64; 4] = [
        0.231_816_021_609_710_9,
        -0.017_843_035_683_206_0,
        0.904_430_077_671_715_9,
        -0.357_699_864_139_459_7,
    ];
    let p: [f64; 3] = [0.11, -13.15, 1.17];

    let mut r = [0.0; 9];
    quaternion_to_rotation(&q, &mut r);

    let mut result1 = [0.0; 3];
    unit_quaternion_rotate_point(&q, &p, &mut result1);

    // C++: VectorRef(result2, 3) = ConstMatrixRef(R, 3, 3) * ConstVectorRef(p, 3)
    // where ConstMatrixRef is row-major.
    let result2 = Matrix3::from_row_slice(&r) * Vector3::from_column_slice(&p);
    expect_arrays_close(3, &result1, result2.as_slice(), K_TOLERANCE);
}

// Verify that (a * b) * c == a * (b * c).
#[test]
fn quaternion_multiplication_is_associative() {
    let mut rng = seeded_rng();
    let mut a = [0.0; 4];
    let mut b = [0.0; 4];
    let mut c = [0.0; 4];
    for i in 0..4 {
        a[i] = 2.0 * rand_double(&mut rng) - 1.0;
        b[i] = 2.0 * rand_double(&mut rng) - 1.0;
        c[i] = 2.0 * rand_double(&mut rng) - 1.0;
    }

    let mut ab = [0.0; 4];
    let mut ab_c = [0.0; 4];
    quaternion_product(&a, &b, &mut ab);
    quaternion_product(&ab, &c, &mut ab_c);

    let mut bc = [0.0; 4];
    let mut a_bc = [0.0; 4];
    quaternion_product(&b, &c, &mut bc);
    quaternion_product(&a, &bc, &mut a_bc);

    expect_near(ab_c[0], a_bc[0], K_TOLERANCE);
    expect_near(ab_c[1], a_bc[1], K_TOLERANCE);
    expect_near(ab_c[2], a_bc[2], K_TOLERANCE);
    expect_near(ab_c[3], a_bc[3], K_TOLERANCE);
}

/// Shared body of the two AngleAxis RotatePoint tests.
#[track_caller]
fn check_angle_axis_rotate_point_matches_matrix(angle_axis: &[f64; 3], p: &[f64; 3]) {
    let mut r = [0.0; 9];
    let mut angle_axis_rotated_p = [0.0; 3];

    angle_axis_to_rotation_matrix(angle_axis, &mut r);
    let rotation_matrix_rotated_p = [
        r[0] * p[0] + r[3] * p[1] + r[6] * p[2],
        r[1] * p[0] + r[4] * p[1] + r[7] * p[2],
        r[2] * p[0] + r[5] * p[1] + r[8] * p[2],
    ];

    angle_axis_rotate_point(angle_axis, p, &mut angle_axis_rotated_p);
    for k in 0..3 {
        assert!(
            (rotation_matrix_rotated_p[k] - angle_axis_rotated_p[k]).abs() <= K_TOLERANCE,
            "{} vs {} p: {} {} {} angle_axis: {} {} {}",
            rotation_matrix_rotated_p[k],
            angle_axis_rotated_p[k],
            p[0],
            p[1],
            p[2],
            angle_axis[0],
            angle_axis[1],
            angle_axis[2]
        );
    }
}

#[test]
fn angle_axis_rotate_point_gives_same_answer_as_rotation_matrix() {
    let mut rng = seeded_rng();
    let mut angle_axis = [0.0; 3];
    let mut p = [0.0; 3];

    for i in 0..10000 {
        let theta = (2.0 * f64::from(i) * 0.0011 - 1.0) * K_PI;
        for _ in 0..50 {
            let mut norm2 = 0.0;
            for k in 0..3 {
                angle_axis[k] = 2.0 * rand_double(&mut rng) - 1.0;
                p[k] = 2.0 * rand_double(&mut rng) - 1.0;
                // NOTE: `=` (not `+=`) as in the C++ test.
                norm2 = angle_axis[k] * angle_axis[k];
            }

            let inv_norm = theta / f64::sqrt(norm2);
            for v in angle_axis.iter_mut() {
                *v *= inv_norm;
            }

            check_angle_axis_rotate_point_matches_matrix(&angle_axis, &p);
        }
    }
}

#[test]
fn angle_axis_near_zero_rotate_point_gives_same_answer_as_rotation_matrix() {
    let mut rng = seeded_rng();
    let mut angle_axis = [0.0; 3];
    let mut p = [0.0; 3];

    for i in 0..10000 {
        let mut norm2 = 0.0;
        for k in 0..3 {
            angle_axis[k] = 2.0 * rand_double(&mut rng) - 1.0;
            p[k] = 2.0 * rand_double(&mut rng) - 1.0;
            // NOTE: `=` (not `+=`) as in the C++ test.
            norm2 = angle_axis[k] * angle_axis[k];
        }

        let theta = (2.0 * f64::from(i) * 0.0001 - 1.0) * 1e-16;
        let inv_norm = theta / f64::sqrt(norm2);
        for v in angle_axis.iter_mut() {
            *v *= inv_norm;
        }

        check_angle_axis_rotate_point_matches_matrix(&angle_axis, &p);
    }
}

#[test]
fn matrix_adapter_row_major_3x3_return_type_and_access_is_correct() {
    let mut array: [f64; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected_array = array;
    let const_array: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a: MatrixAdapter<'_, f64, 3, 1> = row_major_adapter_3x3(&mut array);
    let b = const_row_major_adapter_3x3(&const_array);

    for i in 0..3 {
        for j in 0..3 {
            // The values are integers from 1 to 9, so equality tests are
            // appropriate even for float and double values.
            assert_eq!(a.get(i, j), expected_array[3 * i + j]);
            assert_eq!(b.get(i, j), const_array[3 * i + j]);
        }
    }
}

#[test]
fn matrix_adapter_column_major_3x3_return_type_and_access_is_correct() {
    let mut array: [f64; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected_array = array;
    let const_array: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a: MatrixAdapter<'_, f64, 1, 3> = column_major_adapter_3x3(&mut array);
    let b = const_column_major_adapter_3x3(&const_array);

    for i in 0..3 {
        for j in 0..3 {
            // The values are integers from 1 to 9, so equality tests are
            // appropriate even for float and double values.
            assert_eq!(a.get(i, j), expected_array[3 * j + i]);
            assert_eq!(b.get(i, j), const_array[3 * j + i]);
        }
    }
}

#[test]
fn matrix_adapter_row_major_2x4_is_correct() {
    let expected: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut array = [0_i32; 8];
    {
        let mut m: MatrixAdapter<'_, i32, 4, 1> = MatrixAdapter::new(&mut array);
        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(0, 2, 3);
        m.set(0, 3, 4);
        m.set(1, 0, 5);
        m.set(1, 1, 6);
        m.set(1, 2, 7);
        m.set(1, 3, 8);
    }
    for k in 0..8 {
        assert_eq!(array[k], expected[k]);
    }
}

#[test]
fn matrix_adapter_column_major_2x4_is_correct() {
    let expected: [i32; 8] = [1, 5, 2, 6, 3, 7, 4, 8];
    let mut array = [0_i32; 8];
    {
        let mut m: MatrixAdapter<'_, i32, 1, 2> = MatrixAdapter::new(&mut array);
        *m.get_mut(0, 0) = 1;
        *m.get_mut(0, 1) = 2;
        *m.get_mut(0, 2) = 3;
        *m.get_mut(0, 3) = 4;
        *m.get_mut(1, 0) = 5;
        *m.get_mut(1, 1) = 6;
        *m.get_mut(1, 2) = 7;
        *m.get_mut(1, 3) = 8;
    }
    for k in 0..8 {
        assert_eq!(array[k], expected[k]);
    }
}

#[test]
fn rotation_matrix_to_angle_axis_near_pi_example_one_from_tobias_strauss() {
    // Example from Tobias Strauss
    #[rustfmt::skip]
    let rotation_matrix: [f64; 9] = [
        -0.999_807_135_425_239,   -0.012_815_439_119_447_0, -0.014_881_413_674_579_9,
        -0.012_815_439_119_447_0, -0.148_441_438_622_958,    0.988_838_158_557_669,
        -0.014_881_413_674_579_9,  0.988_838_158_557_669,    0.148_248_574_048_196,
    ];

    let mut angle_axis = [0.0; 3];
    rotation_matrix_to_angle_axis_adapter(
        &const_row_major_adapter_3x3(&rotation_matrix),
        &mut angle_axis,
    );
    let mut round_trip = [0.0; 9];
    angle_axis_to_rotation_matrix_adapter(&angle_axis, &mut row_major_adapter_3x3(&mut round_trip));
    expect_that(is_near_3x3_matrix(&rotation_matrix, &round_trip));
}

fn check_rotation_matrix_to_angle_axis_round_trip(theta: f64, phi: f64, angle: f64) {
    let angle_axis = [
        angle * phi.sin() * theta.cos(),
        angle * phi.sin() * theta.sin(),
        angle * phi.cos(),
    ];

    let mut rotation_matrix = [0.0; 9];
    angle_axis_to_rotation_matrix(&angle_axis, &mut rotation_matrix);

    let mut angle_axis_round_trip = [0.0; 3];
    rotation_matrix_to_angle_axis(&rotation_matrix, &mut angle_axis_round_trip);
    expect_that(is_near_angle_axis(&angle_axis_round_trip, &angle_axis));
}

#[test]
fn rotation_matrix_to_angle_axis_exhaustive_round_trip() {
    let mut rng = seeded_rng();
    const K_MAX_SMALL_ANGLE: f64 = 1e-8;
    const K_NUM_STEPS: i32 = 1000;
    for i in 0..K_NUM_STEPS {
        let theta = f64::from(i) / f64::from(K_NUM_STEPS) * 2.0 * K_PI;
        for j in 0..K_NUM_STEPS {
            let phi = f64::from(j) / f64::from(K_NUM_STEPS) * K_PI;
            // Rotations of angle Pi.
            check_rotation_matrix_to_angle_axis_round_trip(theta, phi, K_PI);
            // Rotation of angle approximately Pi.
            check_rotation_matrix_to_angle_axis_round_trip(
                theta,
                phi,
                K_PI - K_MAX_SMALL_ANGLE * rand_double(&mut rng),
            );
            // Rotations of angle approximately zero.
            check_rotation_matrix_to_angle_axis_round_trip(
                theta,
                phi,
                K_MAX_SMALL_ANGLE * 2.0 * rand_double(&mut rng) - 1.0,
            );
        }
    }
}
