// Ported from Ceres Solver 2.0.0: include/ceres/rotation.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/rotation.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: keir@google.com (Keir Mierle), sameeragarwal@google.com
// (Sameer Agarwal)

//! Templated functions for manipulating rotations. The templated functions
//! are useful when implementing functors for automatic differentiation.
//!
//! In the following, the Quaternions are laid out as 4-vectors, thus:
//!
//! ```text
//!   q[0]  scalar part.
//!   q[1]  coefficient of i.
//!   q[2]  coefficient of j.
//!   q[3]  coefficient of k.
//! ```
//!
//! where: `i*i = j*j = k*k = -1` and `i*j = k, j*k = i, k*i = j`.
//!
//! All functions are generic over [`Scalar`] (`f64` or [`Jet`](crate::Jet)).
//! RUST PORT NOTE: C++ raw pointers (`const T* q`) become slices; the
//! functions index them like the C++ code (and panic if they are too short).
//! Unless stated otherwise 3x3 matrices passed as plain slices are
//! column-major, as in C++.

use crate::scalar::Scalar;

/// Trivial wrapper to index linear arrays as matrices, given a fixed column
/// and row stride. `M.get(i, j)` is `array[i * ROW_STRIDE + j * COL_STRIDE]`.
///
/// Conversion functions to and from rotation matrices accept matrix adapters
/// to permit using row-major and column-major layouts, and rotation matrices
/// embedded in larger matrices (such as a 3x4 projection matrix).
pub struct MatrixAdapter<'a, T, const ROW_STRIDE: usize, const COL_STRIDE: usize> {
    pointer: &'a mut [T],
}

impl<'a, T: Copy, const RS: usize, const CS: usize> MatrixAdapter<'a, T, RS, CS> {
    /// Wraps `pointer`.
    pub fn new(pointer: &'a mut [T]) -> Self {
        Self { pointer }
    }
    /// `M(r, c)`.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.pointer[r * RS + c * CS]
    }
    /// `M(r, c) = value`.
    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, value: T) {
        self.pointer[r * RS + c * CS] = value;
    }
    /// Mutable access to `M(r, c)`.
    #[inline(always)]
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        &mut self.pointer[r * RS + c * CS]
    }
}

/// Read-only variant of [`MatrixAdapter`] (C++ `MatrixAdapter<const T, ...>`).
pub struct ConstMatrixAdapter<'a, T, const ROW_STRIDE: usize, const COL_STRIDE: usize> {
    pointer: &'a [T],
}

impl<'a, T: Copy, const RS: usize, const CS: usize> ConstMatrixAdapter<'a, T, RS, CS> {
    /// Wraps `pointer`.
    pub fn new(pointer: &'a [T]) -> Self {
        Self { pointer }
    }
    /// `M(r, c)`.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> T {
        self.pointer[r * RS + c * CS]
    }
}

/// Treats `pointer` as a 3x3 (contiguous) column-major matrix.
pub fn column_major_adapter_3x3<T: Copy>(pointer: &mut [T]) -> MatrixAdapter<'_, T, 1, 3> {
    MatrixAdapter::new(pointer)
}

/// Treats `pointer` as a 3x3 (contiguous) row-major matrix.
pub fn row_major_adapter_3x3<T: Copy>(pointer: &mut [T]) -> MatrixAdapter<'_, T, 3, 1> {
    MatrixAdapter::new(pointer)
}

/// Read-only [`column_major_adapter_3x3`].
pub fn const_column_major_adapter_3x3<T: Copy>(pointer: &[T]) -> ConstMatrixAdapter<'_, T, 1, 3> {
    ConstMatrixAdapter::new(pointer)
}

/// Read-only [`row_major_adapter_3x3`].
pub fn const_row_major_adapter_3x3<T: Copy>(pointer: &[T]) -> ConstMatrixAdapter<'_, T, 3, 1> {
    ConstMatrixAdapter::new(pointer)
}

/// Convert a value in combined axis-angle representation to a quaternion.
/// The value `angle_axis` is a triple whose norm is an angle in radians, and
/// whose direction is aligned with the axis of rotation, and `quaternion` is
/// a 4-tuple that will contain the resulting quaternion. The implementation
/// may be used with auto-differentiation up to the first derivative, higher
/// derivatives may have unexpected results near the origin.
pub fn angle_axis_to_quaternion<T: Scalar>(angle_axis: &[T], quaternion: &mut [T]) {
    let a0 = angle_axis[0];
    let a1 = angle_axis[1];
    let a2 = angle_axis[2];
    let theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    // For points not at the origin, the full conversion is numerically stable.
    if theta_squared > T::constant(0.0) {
        let theta = theta_squared.sqrt();
        let half_theta = theta * T::constant(0.5);
        let k = half_theta.sin() / theta;
        quaternion[0] = half_theta.cos();
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else {
        // At the origin, sqrt() will produce NaN in the derivative since the
        // argument is zero. By approximating with a Taylor series, and
        // truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used.
        let k = T::constant(0.5);
        quaternion[0] = T::constant(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

/// Convert a quaternion to the equivalent combined axis-angle representation.
/// The value `quaternion` must be a unit quaternion - it is not normalized
/// first, and `angle_axis` will be filled with a value whose norm is the
/// angle of rotation in radians, and whose direction is the axis of rotation.
/// The implementation may be used with auto-differentiation up to the first
/// derivative, higher derivatives may have unexpected results near the
/// origin.
pub fn quaternion_to_angle_axis<T: Scalar>(quaternion: &[T], angle_axis: &mut [T]) {
    let q1 = quaternion[1];
    let q2 = quaternion[2];
    let q3 = quaternion[3];
    let sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    // For quaternions representing non-zero rotation, the conversion is
    // numerically stable.
    if sin_squared_theta > T::constant(0.0) {
        let sin_theta = sin_squared_theta.sqrt();
        let cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which means
        // that angle for the angle_axis vector which is 2 * theta would be
        // greater than pi.
        //
        // While this will result in the correct rotation, it does not result
        // in a normalized angle-axis vector.
        //
        // In that case we observe that 2 * theta ~ 2 * theta - 2 * pi, which
        // is equivalent saying
        //
        //   theta - pi = atan(sin(theta - pi), cos(theta - pi))
        //              = atan(-sin(theta), -cos(theta))
        let two_theta = T::constant(2.0)
            * if cos_theta < T::constant(0.0) {
                (-sin_theta).atan2(-cos_theta)
            } else {
                sin_theta.atan2(cos_theta)
            };
        let k = two_theta / sin_theta;
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in the derivative since
        // the argument is zero. By approximating with a Taylor series, and
        // truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used.
        let k = T::constant(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}

/// Conversion of a 3x3 rotation matrix (column-major) to a quaternion.
pub fn rotation_matrix_to_quaternion<T: Scalar>(r: &[T], quaternion: &mut [T]) {
    rotation_matrix_to_quaternion_adapter(&const_column_major_adapter_3x3(r), quaternion);
}

/// Conversion of a 3x3 rotation matrix to a quaternion.
///
/// This algorithm comes from "Quaternion Calculus and Fast Animation", Ken
/// Shoemake, 1987 SIGGRAPH course notes.
pub fn rotation_matrix_to_quaternion_adapter<T: Scalar, const RS: usize, const CS: usize>(
    r: &ConstMatrixAdapter<'_, T, RS, CS>,
    quaternion: &mut [T],
) {
    let trace = r.get(0, 0) + r.get(1, 1) + r.get(2, 2);
    if trace >= T::constant(0.0) {
        let mut t = (trace + T::constant(1.0)).sqrt();
        quaternion[0] = T::constant(0.5) * t;
        t = T::constant(0.5) / t;
        quaternion[1] = (r.get(2, 1) - r.get(1, 2)) * t;
        quaternion[2] = (r.get(0, 2) - r.get(2, 0)) * t;
        quaternion[3] = (r.get(1, 0) - r.get(0, 1)) * t;
    } else {
        let mut i = 0;
        if r.get(1, 1) > r.get(0, 0) {
            i = 1;
        }
        if r.get(2, 2) > r.get(i, i) {
            i = 2;
        }
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        let mut t = (r.get(i, i) - r.get(j, j) - r.get(k, k) + T::constant(1.0)).sqrt();
        quaternion[i + 1] = T::constant(0.5) * t;
        t = T::constant(0.5) / t;
        quaternion[0] = (r.get(k, j) - r.get(j, k)) * t;
        quaternion[j + 1] = (r.get(j, i) + r.get(i, j)) * t;
        quaternion[k + 1] = (r.get(k, i) + r.get(i, k)) * t;
    }
}

/// Conversion of a 3x3 rotation matrix (column-major) to angle-axis.
///
/// The conversion of a rotation matrix to the angle-axis form is numerically
/// problematic when then rotation angle is close to zero or to Pi. The
/// implementation detects when these two cases occurs and deals with them by
/// taking code paths that are guaranteed to not perform division by a small
/// number.
pub fn rotation_matrix_to_angle_axis<T: Scalar>(r: &[T], angle_axis: &mut [T]) {
    rotation_matrix_to_angle_axis_adapter(&const_column_major_adapter_3x3(r), angle_axis);
}

/// Adapter form of [`rotation_matrix_to_angle_axis`].
pub fn rotation_matrix_to_angle_axis_adapter<T: Scalar, const RS: usize, const CS: usize>(
    r: &ConstMatrixAdapter<'_, T, RS, CS>,
    angle_axis: &mut [T],
) {
    let mut quaternion = [T::constant(0.0); 4];
    rotation_matrix_to_quaternion_adapter(r, &mut quaternion);
    quaternion_to_angle_axis(&quaternion, angle_axis);
}

/// Conversion of angle-axis to a 3x3 rotation matrix (column-major).
pub fn angle_axis_to_rotation_matrix<T: Scalar>(angle_axis: &[T], r: &mut [T]) {
    angle_axis_to_rotation_matrix_adapter(angle_axis, &mut column_major_adapter_3x3(r));
}

/// Adapter form of [`angle_axis_to_rotation_matrix`].
pub fn angle_axis_to_rotation_matrix_adapter<T: Scalar, const RS: usize, const CS: usize>(
    angle_axis: &[T],
    r: &mut MatrixAdapter<'_, T, RS, CS>,
) {
    let k_one = T::constant(1.0);
    let theta2 = dot_product(angle_axis, angle_axis);
    if theta2 > T::constant(f64::EPSILON) {
        // We want to be careful to only evaluate the square root if the norm
        // of the angle_axis vector is greater than zero. Otherwise we get a
        // division by zero.
        let theta = theta2.sqrt();
        let wx = angle_axis[0] / theta;
        let wy = angle_axis[1] / theta;
        let wz = angle_axis[2] / theta;

        let costheta = theta.cos();
        let sintheta = theta.sin();

        r.set(0, 0, costheta + wx * wx * (k_one - costheta));
        r.set(1, 0, wz * sintheta + wx * wy * (k_one - costheta));
        r.set(2, 0, -wy * sintheta + wx * wz * (k_one - costheta));
        r.set(0, 1, wx * wy * (k_one - costheta) - wz * sintheta);
        r.set(1, 1, costheta + wy * wy * (k_one - costheta));
        r.set(2, 1, wx * sintheta + wy * wz * (k_one - costheta));
        r.set(0, 2, wy * sintheta + wx * wz * (k_one - costheta));
        r.set(1, 2, -wx * sintheta + wy * wz * (k_one - costheta));
        r.set(2, 2, costheta + wz * wz * (k_one - costheta));
    } else {
        // Near zero, we switch to using the first order Taylor expansion.
        r.set(0, 0, k_one);
        r.set(1, 0, angle_axis[2]);
        r.set(2, 0, -angle_axis[1]);
        r.set(0, 1, -angle_axis[2]);
        r.set(1, 1, k_one);
        r.set(2, 1, angle_axis[0]);
        r.set(0, 2, angle_axis[1]);
        r.set(1, 2, -angle_axis[0]);
        r.set(2, 2, k_one);
    }
}

/// Conversion of Euler angles (in degrees) to a 3x3 rotation matrix in ROW
/// MAJOR order.
///
/// The {pitch,roll,yaw} Euler angles are rotations around the {x,y,z} axes,
/// respectively. They are applied in that same order, so the total rotation
/// R is Rz * Ry * Rx.
///
/// The `row_stride` argument is ignored (as in C++ 2.0.0) and only kept for
/// API compatibility.
pub fn euler_angles_to_rotation_matrix<T: Scalar>(euler: &[T], _row_stride: i32, r: &mut [T]) {
    euler_angles_to_rotation_matrix_adapter(euler, &mut row_major_adapter_3x3(r));
}

/// Adapter form of [`euler_angles_to_rotation_matrix`].
pub fn euler_angles_to_rotation_matrix_adapter<T: Scalar, const RS: usize, const CS: usize>(
    euler: &[T],
    r: &mut MatrixAdapter<'_, T, RS, CS>,
) {
    let k_pi = std::f64::consts::PI; // C++: 3.14159265358979323846
    let degrees_to_radians = T::constant(k_pi / 180.0);

    let pitch = euler[0] * degrees_to_radians;
    let roll = euler[1] * degrees_to_radians;
    let yaw = euler[2] * degrees_to_radians;

    let c1 = yaw.cos();
    let s1 = yaw.sin();
    let c2 = roll.cos();
    let s2 = roll.sin();
    let c3 = pitch.cos();
    let s3 = pitch.sin();

    r.set(0, 0, c1 * c2);
    r.set(0, 1, -s1 * c3 + c1 * s2 * s3);
    r.set(0, 2, s1 * s3 + c1 * s2 * c3);

    r.set(1, 0, s1 * c2);
    r.set(1, 1, c1 * c3 + s1 * s2 * s3);
    r.set(1, 2, -c1 * s3 + s1 * s2 * c3);

    r.set(2, 0, -s2);
    r.set(2, 1, c2 * s3);
    r.set(2, 2, c2 * c3);
}

/// Convert a 4-vector to a 3x3 scaled rotation matrix.
///
/// The choice of rotation is such that the quaternion [1 0 0 0] goes to an
/// identity matrix and for small a, b, c the quaternion [1 a b c] goes to the
/// matrix
///
/// ```text
///         [  0 -c  b ]
///   I + 2 [  c  0 -a ] + higher order terms
///         [ -b  a  0 ]
/// ```
///
/// which corresponds to a Rodrigues approximation, the last matrix being the
/// cross-product matrix of [a b c]. Together with the property that
/// R(q1 * q2) = R(q1) * R(q2) this uniquely defines the mapping from q to R.
///
/// No normalization of the quaternion is performed, i.e. R = ||q||^2 * Q,
/// where Q is an orthonormal matrix such that det(Q) = 1 and Q*Q' = I.
///
/// WARNING: The rotation matrix is ROW MAJOR.
pub fn quaternion_to_scaled_rotation<T: Scalar>(q: &[T], r: &mut [T]) {
    quaternion_to_scaled_rotation_adapter(q, &mut row_major_adapter_3x3(r));
}

/// Adapter form of [`quaternion_to_scaled_rotation`].
pub fn quaternion_to_scaled_rotation_adapter<T: Scalar, const RS: usize, const CS: usize>(
    q: &[T],
    r: &mut MatrixAdapter<'_, T, RS, CS>,
) {
    // Make convenient names for elements of q.
    let a = q[0];
    let b = q[1];
    let c = q[2];
    let d = q[3];
    // This is not to eliminate common sub-expression, but to make the lines
    // shorter so that they fit in 80 columns!
    let aa = a * a;
    let ab = a * b;
    let ac = a * c;
    let ad = a * d;
    let bb = b * b;
    let bc = b * c;
    let bd = b * d;
    let cc = c * c;
    let cd = c * d;
    let dd = d * d;
    let two = T::constant(2.0);

    r.set(0, 0, aa + bb - cc - dd);
    r.set(0, 1, two * (bc - ad));
    r.set(0, 2, two * (ac + bd));
    r.set(1, 0, two * (ad + bc));
    r.set(1, 1, aa - bb + cc - dd);
    r.set(1, 2, two * (cd - ab));
    r.set(2, 0, two * (bd - ac));
    r.set(2, 1, two * (ab + cd));
    r.set(2, 2, aa - bb - cc + dd);
}

/// Same as [`quaternion_to_scaled_rotation`] except that the rotation matrix
/// is normalized by the Frobenius norm, so that R * R' = I (and det(R) = 1).
///
/// WARNING: The rotation matrix is ROW MAJOR.
pub fn quaternion_to_rotation<T: Scalar>(q: &[T], r: &mut [T]) {
    quaternion_to_rotation_adapter(q, &mut row_major_adapter_3x3(r));
}

/// Adapter form of [`quaternion_to_rotation`].
pub fn quaternion_to_rotation_adapter<T: Scalar, const RS: usize, const CS: usize>(
    q: &[T],
    r: &mut MatrixAdapter<'_, T, RS, CS>,
) {
    quaternion_to_scaled_rotation_adapter(q, r);

    let mut normalizer = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    normalizer = T::constant(1.0) / normalizer;

    for i in 0..3 {
        for j in 0..3 {
            *r.get_mut(i, j) *= normalizer;
        }
    }
}

/// Rotates a point pt by a quaternion q:
///
/// ```text
///   result = R(q) * pt
/// ```
///
/// Assumes the quaternion is unit norm. This assumption allows us to write
/// the transform as (something)*pt + pt, as is clear from the formula below.
/// If you pass in a quaternion with |q|^2 = 2 then you WILL NOT get back 2
/// times the result you get for a unit quaternion.
pub fn unit_quaternion_rotate_point<T: Scalar>(q: &[T], pt: &[T], result: &mut [T]) {
    let t2 = q[0] * q[1];
    let t3 = q[0] * q[2];
    let t4 = q[0] * q[3];
    let t5 = -q[1] * q[1];
    let t6 = q[1] * q[2];
    let t7 = q[1] * q[3];
    let t8 = -q[2] * q[2];
    let t9 = q[2] * q[3];
    let t1 = -q[3] * q[3];
    let two = T::constant(2.0);
    result[0] = two * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0];
    result[1] = two * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1];
    result[2] = two * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2];
}

/// With this function you do not need to assume that q has unit norm. It
/// does assume that the norm is non-zero.
pub fn quaternion_rotate_point<T: Scalar>(q: &[T], pt: &[T], result: &mut [T]) {
    // 'scale' is 1 / norm(q).
    let scale = T::constant(1.0) / (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();

    // Make unit-norm version of q.
    let unit = [scale * q[0], scale * q[1], scale * q[2], scale * q[3]];

    unit_quaternion_rotate_point(&unit, pt, result);
}

/// `zw = z * w`, where `*` is the Quaternion product between 4 vectors.
pub fn quaternion_product<T: Scalar>(z: &[T], w: &[T], zw: &mut [T]) {
    zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
    zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
    zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
    zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
}

/// `x_cross_y = x cross y`.
pub fn cross_product<T: Scalar>(x: &[T], y: &[T], x_cross_y: &mut [T]) {
    x_cross_y[0] = x[1] * y[2] - x[2] * y[1];
    x_cross_y[1] = x[2] * y[0] - x[0] * y[2];
    x_cross_y[2] = x[0] * y[1] - x[1] * y[0];
}

/// Dot product of two 3-vectors.
pub fn dot_product<T: Scalar>(x: &[T], y: &[T]) -> T {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
}

/// `y = R(angle_axis) * x`.
pub fn angle_axis_rotate_point<T: Scalar>(angle_axis: &[T], pt: &[T], result: &mut [T]) {
    let theta2 = dot_product(angle_axis, angle_axis);
    if theta2 > T::constant(f64::EPSILON) {
        // Away from zero, use the rodriguez formula
        //
        //   result = pt costheta +
        //            (w x pt) * sintheta +
        //            w (w . pt) (1 - costheta)
        //
        // We want to be careful to only evaluate the square root if the norm
        // of the angle_axis vector is greater than zero. Otherwise we get a
        // division by zero.
        let theta = theta2.sqrt();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let theta_inverse = T::constant(1.0) / theta;

        let w = [
            angle_axis[0] * theta_inverse,
            angle_axis[1] * theta_inverse,
            angle_axis[2] * theta_inverse,
        ];

        // Explicitly inlined evaluation of the cross product for performance
        // reasons.
        let w_cross_pt = [
            w[1] * pt[2] - w[2] * pt[1],
            w[2] * pt[0] - w[0] * pt[2],
            w[0] * pt[1] - w[1] * pt[0],
        ];
        let tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T::constant(1.0) - costheta);

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        // Near zero, the first order Taylor approximation of the rotation
        // matrix R corresponding to a vector w and angle w is
        //
        //   R = I + hat(w) * sin(theta)
        //
        // But sintheta ~ theta and theta * w = angle_axis, which gives us
        //
        //  R = I + hat(w)
        //
        // and actually performing multiplication with the point pt, gives us
        // R * pt = pt + w x pt.
        //
        // Switching to the Taylor expansion near zero provides meaningful
        // derivatives when evaluated using Jets.
        //
        // Explicitly inlined evaluation of the cross product for performance
        // reasons.
        let w_cross_pt = [
            angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
            angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
            angle_axis[0] * pt[1] - angle_axis[1] * pt[0],
        ];

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}
