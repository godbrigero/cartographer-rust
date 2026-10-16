//! Port of `cartographer/mapping/internal/3d/rotation_parameterization` (.h).
//!
//! Plus functors for quaternion parameter blocks `(w, x, y, z)`, used with
//! `ceres::AutoDiffLocalParameterization` (C++ namespace
//! `cartographer::mapping`):
//!
//! ```
//! use cartographer::mapping::internal::three_d::rotation_parameterization::YawOnlyQuaternionPlus;
//! use ceres::{AutoDiffLocalParameterization, LocalParameterization};
//!
//! let parameterization = AutoDiffLocalParameterization::<YawOnlyQuaternionPlus, 4, 1>::default();
//! assert_eq!(parameterization.local_size(), 1);
//! ```

use ceres::rotation::quaternion_product;
use ceres::{PlusFunctor, Scalar};

use crate::common::math::{clamp, pow2};

/// Applies a rotation about the z-axis (in the parent frame) to a quaternion:
/// `x_plus_delta = q_delta * x` with `q_delta` a yaw rotation with
/// `sin(yaw / 2) = clamp(delta, -0.5, 0.5)`. Local size 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct YawOnlyQuaternionPlus;

impl PlusFunctor for YawOnlyQuaternionPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        let clamped_delta = clamp(delta[0], T::constant(-0.5), T::constant(0.5));
        let q_delta = [
            (T::constant(1.) - clamped_delta * clamped_delta).sqrt(),
            T::constant(0.),
            T::constant(0.),
            clamped_delta,
        ];
        quaternion_product(&q_delta, x, x_plus_delta);
        true
    }
}

/// Applies a rotation about an axis in the xy-plane of the submap frame to a
/// quaternion: `x_plus_delta = x * q_delta`. Local size 2.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstantYawQuaternionPlus;

impl PlusFunctor for ConstantYawQuaternionPlus {
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
        let delta_norm = (pow2(delta[0]) + pow2(delta[1])).sqrt();
        let small = delta_norm < T::constant(1e-6);
        let sin_delta_over_delta = if small {
            T::constant(1.)
        } else {
            delta_norm.sin() / delta_norm
        };
        let q_delta = [
            if small {
                T::constant(1.)
            } else {
                delta_norm.cos()
            },
            sin_delta_over_delta * delta[0],
            sin_delta_over_delta * delta[1],
            T::constant(0.),
        ];
        // We apply the 'delta' which is interpreted as an angle-axis rotation
        // vector in the xy-plane of the submap frame. This way we can align to
        // gravity because rotations around the z-axis in the submap frame do
        // not change gravity alignment, while disallowing random rotations of
        // the map that have nothing to do with gravity alignment (i.e. we
        // disallow steps just changing "yaw" of the complete map).
        quaternion_product(x, &q_delta, x_plus_delta);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ceres::{AutoDiffLocalParameterization, LocalParameterization};

    const X: [f64; 4] = [0.5, 0.5, -0.5, 0.5];

    #[test]
    fn yaw_only_plus() {
        let p = AutoDiffLocalParameterization::<YawOnlyQuaternionPlus, 4, 1>::default();
        let mut out = [0.; 4];
        assert!(p.plus(&X, &[0.], &mut out));
        assert_eq!(out, X);
        // A delta of sin(yaw / 2) = 0.25 rotates about z: q_delta * x.
        assert!(p.plus(&X, &[0.25], &mut out));
        let c = (1. - 0.0625f64).sqrt();
        let mut expected = [0.; 4];
        quaternion_product(&[c, 0., 0., 0.25], &X, &mut expected);
        assert_eq!(out, expected);
        // Clamped to 0.5.
        let mut clamped = [0.; 4];
        assert!(p.plus(&X, &[0.5], &mut out));
        assert!(p.plus(&X, &[3.], &mut clamped));
        assert_eq!(out, clamped);
        // Jacobian at delta = 0: d(q_delta * x)/d delta = (0, 0, 0, 1) * x.
        let mut jacobian = [0.; 4];
        assert!(p.compute_jacobian(&X, &mut jacobian));
        let mut expected_jacobian = [0.; 4];
        quaternion_product(&[0., 0., 0., 1.], &X, &mut expected_jacobian);
        assert_eq!(jacobian, expected_jacobian);
    }

    #[test]
    fn constant_yaw_plus() {
        let p = AutoDiffLocalParameterization::<ConstantYawQuaternionPlus, 4, 2>::default();
        let mut out = [0.; 4];
        assert!(p.plus(&X, &[0., 0.], &mut out));
        assert_eq!(out, X);
        let delta = [0.3, -0.4];
        assert!(p.plus(&X, &delta, &mut out));
        let norm = 0.5f64;
        let s = norm.sin() / norm;
        let mut expected = [0.; 4];
        quaternion_product(&X, &[norm.cos(), s * 0.3, s * -0.4, 0.], &mut expected);
        assert_eq!(out, expected);
        // Jacobian at delta = 0: columns x * (0, 1, 0, 0) and x * (0, 0, 1, 0).
        let mut jacobian = [0.; 8];
        assert!(p.compute_jacobian(&X, &mut jacobian));
        let mut column_0 = [0.; 4];
        let mut column_1 = [0.; 4];
        quaternion_product(&X, &[0., 1., 0., 0.], &mut column_0);
        quaternion_product(&X, &[0., 0., 1., 0.], &mut column_1);
        for row in 0..4 {
            assert_eq!(jacobian[2 * row], column_0[row]);
            assert_eq!(jacobian[2 * row + 1], column_1[row]);
        }
    }
}
