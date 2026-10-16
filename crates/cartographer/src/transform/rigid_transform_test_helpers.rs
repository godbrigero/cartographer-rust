//! Port of `cartographer/transform/rigid_transform_test_helpers` (.h).
//!
//! The gmock matcher `IsNearly(rigid, epsilon)` becomes the [`IsNearly`]
//! trait and the [`assert_is_nearly`] assertion helper, usable from tests of
//! other modules.

use std::fmt::Display;

use nalgebra::{Matrix3, Matrix4, RealField};

use super::rigid_transform::{Rigid2, Rigid3, quaternion_to_rotation_matrix};

/// `Eigen::Translation * Eigen::Rotation2D` as a homogeneous 3x3 matrix
/// (C++ `ToEigen(const Rigid2<T>&)`).
pub fn to_eigen_rigid2<T: RealField + Copy>(rigid2: &Rigid2<T>) -> Matrix3<T> {
    let r = rigid2.rotation().to_rotation_matrix();
    let t = rigid2.translation();
    Matrix3::new(
        r[(0, 0)],
        r[(0, 1)],
        t.x,
        r[(1, 0)],
        r[(1, 1)],
        t.y,
        T::zero(),
        T::zero(),
        T::one(),
    )
}

/// `Eigen::Translation * Eigen::Quaternion` as a homogeneous 4x4 matrix
/// (C++ `ToEigen(const Rigid3<T>&)`).
pub fn to_eigen_rigid3<T: RealField + Copy>(rigid3: &Rigid3<T>) -> Matrix4<T> {
    let r = quaternion_to_rotation_matrix(rigid3.rotation());
    let t = rigid3.translation();
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    m[(0, 3)] = t.x;
    m[(1, 3)] = t.y;
    m[(2, 3)] = t.z;
    m
}

/// Eigen's `isApprox`: `|a - b|_F^2 <= prec^2 * min(|a|_F^2, |b|_F^2)`.
fn is_approx<T: RealField + Copy, const N: usize>(
    a: &nalgebra::SMatrix<T, N, N>,
    b: &nalgebra::SMatrix<T, N, N>,
    prec: T,
) -> bool {
    let squared_norm =
        |m: &nalgebra::SMatrix<T, N, N>| m.iter().fold(T::zero(), |acc, &x| acc + x * x);
    let diff = squared_norm(&(a - b));
    let na = squared_norm(a);
    let nb = squared_norm(b);
    diff <= prec * prec * if na < nb { na } else { nb }
}

/// Approximate equality of rigid transforms with Eigen's `isApprox`
/// semantics on the homogeneous matrices (relative Frobenius norm).
pub trait IsNearly {
    /// True if `self` is nearly `expected` (C++ matcher `IsNearly`).
    fn is_nearly(&self, expected: &Self, epsilon: f64) -> bool;
}

impl<T: RealField + Copy> IsNearly for Rigid2<T> {
    fn is_nearly(&self, expected: &Self, epsilon: f64) -> bool {
        is_approx(
            &to_eigen_rigid2(self),
            &to_eigen_rigid2(expected),
            nalgebra::convert(epsilon),
        )
    }
}

impl<T: RealField + Copy> IsNearly for Rigid3<T> {
    fn is_nearly(&self, expected: &Self, epsilon: f64) -> bool {
        is_approx(
            &to_eigen_rigid3(self),
            &to_eigen_rigid3(expected),
            nalgebra::convert(epsilon),
        )
    }
}

/// Free-function form of [`IsNearly::is_nearly`].
pub fn is_nearly<R: IsNearly>(actual: &R, expected: &R, epsilon: f64) -> bool {
    actual.is_nearly(expected, epsilon)
}

/// Panics with a gmock-like message unless `actual` is nearly `expected`
/// (C++ `EXPECT_THAT(actual, IsNearly(expected, epsilon))`).
#[track_caller]
pub fn assert_is_nearly<R: IsNearly + Display>(actual: &R, expected: &R, epsilon: f64) {
    assert!(
        actual.is_nearly(expected, epsilon),
        "Value of: actual\nExpected: is nearly {expected}\n  Actual: {actual}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::{Rigid2d, Rigid3d};
    use nalgebra::{Vector2, Vector3};

    #[test]
    fn is_approx_is_relative() {
        let a = Rigid2d::new(Vector2::new(1000., 0.), 0.);
        let b = Rigid2d::new(Vector2::new(1000.001, 0.), 0.);
        // |diff|^2 = 1e-6, min |m|^2 = 1e6 + 3.
        assert!(a.is_nearly(&b, 1e-6));
        assert!(!a.is_nearly(&b, 1e-7));
        let c = Rigid3d::from_translation(Vector3::new(0., 0., 1e-9));
        assert!(c.is_nearly(&Rigid3d::identity(), 1e-9));
        assert!(!c.is_nearly(&Rigid3d::identity(), 1e-10));
        assert!(is_nearly(&c, &c, 0.));
    }

    #[test]
    #[should_panic(expected = "Expected: is nearly { t: [1, 0], r: [0] }")]
    fn assert_is_nearly_message() {
        assert_is_nearly(
            &Rigid2d::identity(),
            &Rigid2d::from_translation(Vector2::new(1., 0.)),
            1e-6,
        );
    }

    #[test]
    fn homogeneous_matrices() {
        let r = Rigid2d::new(Vector2::new(1., 2.), std::f64::consts::FRAC_PI_2);
        let m = to_eigen_rigid2(&r);
        assert!((m[(0, 1)] + 1.).abs() < 1e-15);
        assert_eq!(m[(0, 2)], 1.);
        assert_eq!(m[(2, 2)], 1.);
        let m = to_eigen_rigid3(&Rigid3d::from_translation(Vector3::new(1., 2., 3.)));
        assert_eq!(m[(2, 3)], 3.);
        assert_eq!(m[(3, 3)], 1.);
        assert_eq!(m[(0, 0)], 1.);
    }
}
