// RUST PORT NOTE: no direct Ceres counterpart. In C++ templated functors use
// `template <typename T>` and rely on `T(0.5)` / implicit conversions; in Rust
// the equivalent bound is `T: Scalar`, which is implemented for `f64`, `f32`
// and every `Jet<N>`.

//! The [`Scalar`] trait: the bound for code that must run on both `f64` and
//! [`Jet`](crate::Jet)s (cost functors, local parameterization functors and
//! generic Cartographer math).

use nalgebra::RealField;

/// Convenience bound for generic numeric code that should work with `f64`
/// and with [`Jet`](crate::Jet)s (forward-mode autodiff), like a C++
/// `template <typename T>` used with `ceres::Jet`.
///
/// Blanket-implemented for every `nalgebra::RealField + Copy` type. All of
/// nalgebra (vectors, matrices, `UnitQuaternion`, `Rotation2`, ...) and the
/// `ComplexField`/`RealField` math methods (`sqrt`, `sin`, `atan2`, ...) are
/// available on `T`.
///
/// ```
/// use ceres::{Jet, Scalar};
/// use nalgebra::Vector2;
///
/// fn norm<T: Scalar>(x: T, y: T) -> T {
///     Vector2::new(x, y).norm() + T::constant(0.0)
/// }
/// assert_eq!(norm(3.0, 4.0), 5.0);
/// let j = norm(Jet::<2>::variable(3.0, 0), Jet::variable(4.0, 1));
/// assert!((j.v[0] - 0.6).abs() < 1e-15 && (j.v[1] - 0.8).abs() < 1e-15);
/// ```
pub trait Scalar: RealField + Copy {
    /// Converts an `f64` constant into `Self` (C++ `T(x)`); for jets the
    /// infinitesimal part is zero.
    #[inline(always)]
    fn constant(x: f64) -> Self {
        nalgebra::convert::<f64, Self>(x)
    }

    /// The scalar (value) part as `f64` (C++ `x.a` for jets, `x` for
    /// doubles).
    #[inline(always)]
    fn scalar_part(&self) -> f64 {
        <Self as simba::scalar::SupersetOf<f64>>::to_subset_unchecked(self)
    }
}

impl<T: RealField + Copy> Scalar for T {}
