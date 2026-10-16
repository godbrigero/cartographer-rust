//! Port of `cartographer/transform/rigid_transform` (.h, .cc).
//!
//! Also contains Rust equivalents of the Eigen geometry types and operations
//! the C++ code relies on (`Eigen::Rotation2D`, `Eigen::AngleAxis` and the
//! `Eigen::Quaternion` products/rotations/slerp), implemented with Eigen's
//! formulas and operation order on top of raw `nalgebra::Quaternion`s (which,
//! like Eigen's, are not necessarily normalized).

// Composing transforms adds angles and translations inside `Mul` impls.
#![allow(clippy::suspicious_arithmetic_impl)]

use std::fmt;
use std::ops::Mul;

use nalgebra::{Matrix2, Matrix3, Quaternion, RealField, Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::math::normalize_angle_difference;
use crate::common::printf::format_absl;

#[inline]
fn c<T: RealField + Copy>(x: f64) -> T {
    nalgebra::convert(x)
}

/// Converts a scalar to `f64` for printing (the real part for dual numbers).
fn to_f64<T: RealField + Copy>(x: T) -> f64 {
    x.to_subset().unwrap_or(f64::NAN)
}

// ----- Eigen quaternion helpers -----

/// `Eigen::Quaternion::Identity()`.
pub fn quaternion_identity<T: RealField + Copy>() -> Quaternion<T> {
    Quaternion::new(T::one(), T::zero(), T::zero(), T::zero())
}

/// Hamilton product `a * b` with Eigen's operation order (not normalized).
pub fn quaternion_mul<T: RealField + Copy>(a: &Quaternion<T>, b: &Quaternion<T>) -> Quaternion<T> {
    let (aw, ax, ay, az) = (a.w, a.i, a.j, a.k);
    let (bw, bx, by, bz) = (b.w, b.i, b.j, b.k);
    Quaternion::new(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
    )
}

fn cross<T: RealField + Copy>(a: &Vector3<T>, b: &Vector3<T>) -> Vector3<T> {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// `q * v` for a quaternion and a 3D vector (Eigen `_transformVector`,
/// only meaningful for unit quaternions):
/// `uv = 2 (q.vec x v); v + q.w uv + q.vec x uv`.
pub fn quaternion_rotate<T: RealField + Copy>(q: &Quaternion<T>, v: &Vector3<T>) -> Vector3<T> {
    let qv = Vector3::new(q.i, q.j, q.k);
    let mut uv = cross(&qv, v);
    uv += uv;
    v + uv * q.w + cross(&qv, &uv)
}

/// `q.conjugate()`.
pub fn quaternion_conjugate<T: RealField + Copy>(q: &Quaternion<T>) -> Quaternion<T> {
    Quaternion::new(q.w, -q.i, -q.j, -q.k)
}

/// `q.squaredNorm()` (coefficients in storage order x, y, z, w).
pub fn quaternion_squared_norm<T: RealField + Copy>(q: &Quaternion<T>) -> T {
    q.i * q.i + q.j * q.j + q.k * q.k + q.w * q.w
}

/// `q.norm()`.
pub fn quaternion_norm<T: RealField + Copy>(q: &Quaternion<T>) -> T {
    quaternion_squared_norm(q).sqrt()
}

/// `q.normalized()`: divides by the norm; returns `q` unchanged if the norm
/// is zero (Eigen >= 3.3).
pub fn quaternion_normalized<T: RealField + Copy>(q: &Quaternion<T>) -> Quaternion<T> {
    let z = quaternion_squared_norm(q);
    if z > T::zero() {
        let n = z.sqrt();
        Quaternion::new(q.w / n, q.i / n, q.j / n, q.k / n)
    } else {
        *q
    }
}

/// `a.dot(b)` (coefficient-wise, storage order x, y, z, w).
pub fn quaternion_dot<T: RealField + Copy>(a: &Quaternion<T>, b: &Quaternion<T>) -> T {
    a.i * b.i + a.j * b.j + a.k * b.k + a.w * b.w
}

/// `q.toRotationMatrix()` (Eigen's formula, valid for unit quaternions).
pub fn quaternion_to_rotation_matrix<T: RealField + Copy>(q: &Quaternion<T>) -> Matrix3<T> {
    let one = T::one();
    let tx = q.i + q.i;
    let ty = q.j + q.j;
    let tz = q.k + q.k;
    let twx = tx * q.w;
    let twy = ty * q.w;
    let twz = tz * q.w;
    let txx = tx * q.i;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        one - (tyy + tzz),
        txy - twz,
        txz + twy,
        txy + twz,
        one - (txx + tzz),
        tyz - twx,
        txz - twy,
        tyz + twx,
        one - (txx + tyy),
    )
}

/// `a.slerp(t, b)` with Eigen's semantics: shortest path, linear
/// interpolation when the quaternions are (anti-)parallel up to machine
/// epsilon, no final normalization.
pub fn quaternion_slerp<T: RealField + Copy>(
    a: &Quaternion<T>,
    t: T,
    b: &Quaternion<T>,
) -> Quaternion<T> {
    let one = T::one() - T::default_epsilon();
    let d = quaternion_dot(a, b);
    let abs_d = d.abs();
    let scale0;
    let mut scale1;
    if abs_d >= one {
        scale0 = T::one() - t;
        scale1 = t;
    } else {
        // theta is the angle between the 2 quaternions
        let theta = abs_d.acos();
        let sin_theta = theta.sin();
        scale0 = ((T::one() - t) * theta).sin() / sin_theta;
        scale1 = (t * theta).sin() / sin_theta;
    }
    if d < T::zero() {
        scale1 = -scale1;
    }
    Quaternion::new(
        scale0 * a.w + scale1 * b.w,
        scale0 * a.i + scale1 * b.i,
        scale0 * a.j + scale1 * b.j,
        scale0 * a.k + scale1 * b.k,
    )
}

/// Casts the scalar type of a quaternion (`q.cast<O>()`, via `f64`).
pub fn quaternion_cast<T: RealField + Copy, O: RealField + Copy>(
    q: &Quaternion<T>,
) -> Quaternion<O> {
    Quaternion::new(cast(q.w), cast(q.i), cast(q.j), cast(q.k))
}

/// Casts a scalar between float types via `f64` (exact for f32 -> f64,
/// correctly rounded for f64 -> f32, like a C++ `static_cast`).
fn cast<T: RealField + Copy, O: RealField + Copy>(x: T) -> O {
    nalgebra::convert(to_f64(x))
}

// ----- Eigen::Rotation2D -----

/// A 2D rotation stored as a raw (not normalized) angle, like
/// `Eigen::Rotation2D`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotation2D<T> {
    angle: T,
}

impl<T: RealField + Copy> Rotation2D<T> {
    /// Rotation by `angle` radians.
    pub fn new(angle: T) -> Self {
        Rotation2D { angle }
    }

    /// The identity rotation.
    pub fn identity() -> Self {
        Rotation2D { angle: T::zero() }
    }

    /// The (raw) angle in radians.
    pub fn angle(&self) -> T {
        self.angle
    }

    /// The inverse rotation (negated angle).
    pub fn inverse(&self) -> Self {
        Rotation2D { angle: -self.angle }
    }

    /// `[[cos, -sin], [sin, cos]]`.
    pub fn to_rotation_matrix(&self) -> Matrix2<T> {
        let sin_a = self.angle.sin();
        let cos_a = self.angle.cos();
        Matrix2::new(cos_a, -sin_a, sin_a, cos_a)
    }

    /// The angle in [-pi, pi] (`smallestAngle()`).
    pub fn smallest_angle(&self) -> T {
        let two_pi = T::two_pi();
        let tmp = self.angle % two_pi;
        if tmp > T::pi() {
            tmp - two_pi
        } else if tmp < -T::pi() {
            tmp + two_pi
        } else {
            tmp
        }
    }

    /// The angle in [0, 2 pi] (`smallestPositiveAngle()`).
    pub fn smallest_positive_angle(&self) -> T {
        let two_pi = T::two_pi();
        let tmp = self.angle % two_pi;
        if tmp < T::zero() { tmp + two_pi } else { tmp }
    }

    /// Casts the scalar type.
    pub fn cast<O: RealField + Copy>(&self) -> Rotation2D<O> {
        Rotation2D {
            angle: cast(self.angle),
        }
    }
}

impl<T: RealField + Copy> Mul for Rotation2D<T> {
    type Output = Rotation2D<T>;
    fn mul(self, rhs: Rotation2D<T>) -> Rotation2D<T> {
        Rotation2D {
            angle: self.angle + rhs.angle,
        }
    }
}

impl<T: RealField + Copy> Mul<Vector2<T>> for Rotation2D<T> {
    type Output = Vector2<T>;
    fn mul(self, v: Vector2<T>) -> Vector2<T> {
        let m = self.to_rotation_matrix();
        Vector2::new(
            m[(0, 0)] * v.x + m[(0, 1)] * v.y,
            m[(1, 0)] * v.x + m[(1, 1)] * v.y,
        )
    }
}

// ----- Eigen::AngleAxis -----

/// A rotation of `angle` radians around the unit vector `axis`, like
/// `Eigen::AngleAxis`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngleAxis<T> {
    /// Rotation angle in radians.
    pub angle: T,
    /// Rotation axis (assumed to be normalized).
    pub axis: Vector3<T>,
}

impl<T: RealField + Copy> AngleAxis<T> {
    /// Rotation of `angle` around `axis` (which must be normalized).
    pub fn new(angle: T, axis: Vector3<T>) -> Self {
        AngleAxis { angle, axis }
    }

    /// The equivalent quaternion: `(cos(angle / 2), sin(angle / 2) axis)`.
    pub fn to_quaternion(&self) -> Quaternion<T> {
        let ha = c::<T>(0.5) * self.angle;
        let s = ha.sin();
        Quaternion::new(ha.cos(), s * self.axis.x, s * self.axis.y, s * self.axis.z)
    }

    /// Eigen's `AngleAxis(const QuaternionBase&)` (angle in [0, pi]).
    pub fn from_quaternion(q: &Quaternion<T>) -> Self {
        let n = (q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
        if n != T::zero() {
            let angle = c::<T>(2.) * n.atan2(q.w.abs());
            let axis = if q.w < T::zero() {
                Vector3::new(-q.i / n, -q.j / n, -q.k / n)
            } else {
                Vector3::new(q.i / n, q.j / n, q.k / n)
            };
            AngleAxis { angle, axis }
        } else {
            AngleAxis {
                angle: T::zero(),
                axis: Vector3::x(),
            }
        }
    }
}

impl<T: RealField + Copy> From<AngleAxis<T>> for Quaternion<T> {
    fn from(angle_axis: AngleAxis<T>) -> Self {
        angle_axis.to_quaternion()
    }
}

impl<T: RealField + Copy> Mul for AngleAxis<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: AngleAxis<T>) -> Quaternion<T> {
        quaternion_mul(&self.to_quaternion(), &rhs.to_quaternion())
    }
}

impl<T: RealField + Copy> Mul<AngleAxis<T>> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: AngleAxis<T>) -> Quaternion<T> {
        quaternion_mul(&self, &rhs.to_quaternion())
    }
}

// ----- Rigid2 -----

/// A 2D rigid transformation: rotation by an (unnormalized) angle followed
/// by a translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid2<T> {
    translation: Vector2<T>,
    rotation: Rotation2D<T>,
}

/// `Rigid2<f64>`.
pub type Rigid2d = Rigid2<f64>;
/// `Rigid2<f32>`.
pub type Rigid2f = Rigid2<f32>;

impl<T: RealField + Copy> Default for Rigid2<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealField + Copy> Rigid2<T> {
    /// Creates a transform from a translation and a rotation angle (C++
    /// `Rigid2(const Vector&, double)`).
    pub fn new(translation: Vector2<T>, rotation: T) -> Self {
        Rigid2 {
            translation,
            rotation: Rotation2D::new(rotation),
        }
    }

    /// Creates a transform from a translation and a rotation (C++
    /// `Rigid2(const Vector&, const Rotation2D&)`).
    pub fn from_parts(translation: Vector2<T>, rotation: Rotation2D<T>) -> Self {
        Rigid2 {
            translation,
            rotation,
        }
    }

    /// A pure rotation (C++ `Rigid2::Rotation(double)`).
    pub fn from_rotation(rotation: T) -> Self {
        Self::new(Vector2::zeros(), rotation)
    }

    /// A pure rotation (C++ `Rigid2::Rotation(const Rotation2D&)`).
    pub fn from_rotation2d(rotation: Rotation2D<T>) -> Self {
        Self::from_parts(Vector2::zeros(), rotation)
    }

    /// A pure translation (C++ `Rigid2::Translation`).
    pub fn from_translation(vector: Vector2<T>) -> Self {
        Self::from_parts(vector, Rotation2D::identity())
    }

    /// The identity transform.
    pub fn identity() -> Self {
        Self::from_parts(Vector2::zeros(), Rotation2D::identity())
    }

    /// Casts the scalar type.
    pub fn cast<O: RealField + Copy>(&self) -> Rigid2<O> {
        Rigid2 {
            translation: self.translation.map(cast),
            rotation: self.rotation.cast(),
        }
    }

    /// The translation.
    pub fn translation(&self) -> &Vector2<T> {
        &self.translation
    }

    /// The rotation.
    pub fn rotation(&self) -> Rotation2D<T> {
        self.rotation
    }

    /// The rotation angle normalized to [-pi, pi].
    ///
    /// RUST PORT NOTE: returns `T` (C++ returns `double`).
    pub fn normalized_angle(&self) -> T {
        normalize_angle_difference(self.rotation.angle())
    }

    /// The inverse transform.
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        let translation = -(rotation * self.translation);
        Self::from_parts(translation, rotation)
    }

    /// `"{ t: [x, y], r: [angle] }"`.
    pub fn debug_string(&self) -> String {
        format!(
            "{{ t: [{}, {}], r: [{}] }}",
            format_absl(to_f64(self.translation.x)),
            format_absl(to_f64(self.translation.y)),
            format_absl(to_f64(self.rotation.angle()))
        )
    }
}

impl<T: RealField + Copy> fmt::Display for Rigid2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.debug_string())
    }
}

impl<T: RealField + Copy> Mul for Rigid2<T> {
    type Output = Rigid2<T>;
    fn mul(self, rhs: Rigid2<T>) -> Rigid2<T> {
        Rigid2::from_parts(
            self.rotation() * rhs.translation + self.translation,
            self.rotation() * rhs.rotation(),
        )
    }
}

impl<T: RealField + Copy> Mul<&Rigid2<T>> for &Rigid2<T> {
    type Output = Rigid2<T>;
    fn mul(self, rhs: &Rigid2<T>) -> Rigid2<T> {
        *self * *rhs
    }
}

impl<T: RealField + Copy> Mul<Vector2<T>> for Rigid2<T> {
    type Output = Vector2<T>;
    fn mul(self, point: Vector2<T>) -> Vector2<T> {
        self.rotation() * point + self.translation
    }
}

impl<T: RealField + Copy> Mul<&Vector2<T>> for &Rigid2<T> {
    type Output = Vector2<T>;
    fn mul(self, point: &Vector2<T>) -> Vector2<T> {
        *self * *point
    }
}

// ----- Rigid3 -----

/// A 3D rigid transformation: rotation by a quaternion followed by a
/// translation. The quaternion is not normalized on construction (like in
/// C++); composition renormalizes.
#[derive(Clone, Copy, Debug)]
pub struct Rigid3<T> {
    translation: Vector3<T>,
    rotation: Quaternion<T>,
}

impl<T: RealField + Copy> PartialEq for Rigid3<T> {
    fn eq(&self, other: &Self) -> bool {
        self.translation == other.translation && self.rotation.coords == other.rotation.coords
    }
}

/// `Rigid3<f64>`.
pub type Rigid3d = Rigid3<f64>;
/// `Rigid3<f32>`.
pub type Rigid3f = Rigid3<f32>;

impl<T: RealField + Copy> Default for Rigid3<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: RealField + Copy> Rigid3<T> {
    /// Creates a transform from a translation and a (not normalized)
    /// quaternion.
    pub fn new(translation: Vector3<T>, rotation: Quaternion<T>) -> Self {
        Rigid3 {
            translation,
            rotation,
        }
    }

    /// Creates a transform from a translation and an angle-axis rotation.
    pub fn with_angle_axis(translation: Vector3<T>, rotation: AngleAxis<T>) -> Self {
        Self::new(translation, rotation.to_quaternion())
    }

    /// A pure rotation (C++ `Rigid3::Rotation(const AngleAxis&)`).
    pub fn from_angle_axis(angle_axis: AngleAxis<T>) -> Self {
        Self::new(Vector3::zeros(), angle_axis.to_quaternion())
    }

    /// A pure rotation (C++ `Rigid3::Rotation(const Quaternion&)`).
    pub fn from_rotation(rotation: Quaternion<T>) -> Self {
        Self::new(Vector3::zeros(), rotation)
    }

    /// A pure translation (C++ `Rigid3::Translation`).
    pub fn from_translation(vector: Vector3<T>) -> Self {
        Self::new(vector, quaternion_identity())
    }

    /// Creates a transform from a `[w, x, y, z]` rotation and an
    /// `[x, y, z]` translation.
    pub fn from_arrays(rotation: &[T; 4], translation: &[T; 3]) -> Self {
        Self::new(
            Vector3::new(translation[0], translation[1], translation[2]),
            Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]),
        )
    }

    /// The identity transform.
    pub fn identity() -> Self {
        Self::new(Vector3::zeros(), quaternion_identity())
    }

    /// Casts the scalar type.
    pub fn cast<O: RealField + Copy>(&self) -> Rigid3<O> {
        Rigid3 {
            translation: self.translation.map(cast),
            rotation: quaternion_cast(&self.rotation),
        }
    }

    /// The translation.
    pub fn translation(&self) -> &Vector3<T> {
        &self.translation
    }

    /// The rotation.
    pub fn rotation(&self) -> &Quaternion<T> {
        &self.rotation
    }

    /// The inverse transform (assumes a unit quaternion).
    pub fn inverse(&self) -> Self {
        let rotation = quaternion_conjugate(&self.rotation);
        let translation = -quaternion_rotate(&rotation, &self.translation);
        Self::new(translation, rotation)
    }

    /// `"{ t: [x, y, z], q: [w, x, y, z] }"`.
    pub fn debug_string(&self) -> String {
        format!(
            "{{ t: [{}, {}, {}], q: [{}, {}, {}, {}] }}",
            format_absl(to_f64(self.translation.x)),
            format_absl(to_f64(self.translation.y)),
            format_absl(to_f64(self.translation.z)),
            format_absl(to_f64(self.rotation.w)),
            format_absl(to_f64(self.rotation.i)),
            format_absl(to_f64(self.rotation.j)),
            format_absl(to_f64(self.rotation.k))
        )
    }

    /// True if the translation has no NaNs and the rotation is a unit
    /// quaternion (within 1e-3).
    pub fn is_valid(&self) -> bool {
        let is_nan = |x: T| x.partial_cmp(&x).is_none();
        !is_nan(self.translation.x)
            && !is_nan(self.translation.y)
            && !is_nan(self.translation.z)
            && (T::one() - quaternion_norm(&self.rotation)).abs() < c(1e-3)
    }
}

impl<T: RealField + Copy> fmt::Display for Rigid3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.debug_string())
    }
}

impl<T: RealField + Copy> Mul for Rigid3<T> {
    type Output = Rigid3<T>;
    fn mul(self, rhs: Rigid3<T>) -> Rigid3<T> {
        Rigid3::new(
            quaternion_rotate(&self.rotation, &rhs.translation) + self.translation,
            quaternion_normalized(&quaternion_mul(&self.rotation, &rhs.rotation)),
        )
    }
}

impl<T: RealField + Copy> Mul<&Rigid3<T>> for &Rigid3<T> {
    type Output = Rigid3<T>;
    fn mul(self, rhs: &Rigid3<T>) -> Rigid3<T> {
        *self * *rhs
    }
}

impl<T: RealField + Copy> Mul<Vector3<T>> for Rigid3<T> {
    type Output = Vector3<T>;
    fn mul(self, point: Vector3<T>) -> Vector3<T> {
        quaternion_rotate(&self.rotation, &point) + self.translation
    }
}

impl<T: RealField + Copy> Mul<&Vector3<T>> for &Rigid3<T> {
    type Output = Vector3<T>;
    fn mul(self, point: &Vector3<T>) -> Vector3<T> {
        *self * *point
    }
}

// ----- Free functions -----

/// Converts (roll, pitch, yaw) to a unit length quaternion. Based on the
/// URDF specification <http://wiki.ros.org/urdf/XML/joint>.
pub fn roll_pitch_yaw(roll: f64, pitch: f64, yaw: f64) -> Quaternion<f64> {
    let roll_angle = AngleAxis::new(roll, Vector3::x());
    let pitch_angle = AngleAxis::new(pitch, Vector3::y());
    let yaw_angle = AngleAxis::new(yaw, Vector3::z());
    (yaw_angle * pitch_angle) * roll_angle
}

fn translation_from_dictionary(dictionary: &LuaParameterDictionary) -> Vector3<f64> {
    let translation = dictionary.get_array_values_as_doubles();
    assert_eq!(
        3,
        translation.len(),
        "Check failed: 3 == translation.size() (3 vs. {}) Need (x, y, z) for translation.",
        translation.len()
    );
    Vector3::new(translation[0], translation[1], translation[2])
}

/// Returns a `Rigid3d` given a `dictionary` containing `translation`
/// (x, y, z) and `rotation` which can either be an array of (roll, pitch,
/// yaw) or a dictionary with (w, x, y, z) values as a quaternion.
pub fn from_dictionary(dictionary: &LuaParameterDictionary) -> Rigid3d {
    let translation = translation_from_dictionary(&dictionary.get_dictionary("translation"));

    let rotation_dictionary = dictionary.get_dictionary("rotation");
    if rotation_dictionary.has_key("w") {
        let rotation = Quaternion::new(
            rotation_dictionary.get_double("w"),
            rotation_dictionary.get_double("x"),
            rotation_dictionary.get_double("y"),
            rotation_dictionary.get_double("z"),
        );
        let norm = quaternion_norm(&rotation);
        assert!(
            (norm - 1.).abs() <= 1e-9,
            "Check failed: std::abs(rotation.norm() - 1.) <= 1e-9 ({norm} vs. 1)"
        );
        Rigid3d::new(translation, rotation)
    } else {
        let rotation = rotation_dictionary.get_array_values_as_doubles();
        assert_eq!(
            3,
            rotation.len(),
            "Check failed: 3 == rotation.size() (3 vs. {}) Need (roll, pitch, yaw) for rotation.",
            rotation.len()
        );
        Rigid3d::new(
            translation,
            roll_pitch_yaw(rotation[0], rotation[1], rotation[2]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::transform::rigid_transform_test_helpers::{IsNearly, assert_is_nearly};
    use crate::transform::transform::angle_axis_vector_to_rotation_quaternion;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Fixture of the C++ typed test `RigidTransformTest`.
    struct RigidTransformTest {
        prng: StdRng,
    }

    impl RigidTransformTest {
        fn new() -> Self {
            RigidTransformTest {
                prng: StdRng::seed_from_u64(42),
            }
        }

        fn distribution<T: RealField + Copy>(&mut self) -> T {
            c(self.prng.random_range(-1.0..1.0))
        }

        fn get_random_rigid2<T: RealField + Copy>(&mut self) -> Rigid2<T> {
            let x = c::<T>(0.7) * self.distribution::<T>();
            let y = c::<T>(0.7) * self.distribution::<T>();
            let theta = c::<T>(0.2) * self.distribution::<T>();
            Rigid2::new(Vector2::new(x, y), theta)
        }

        fn get_random_rigid3<T: RealField + Copy>(&mut self) -> Rigid3<T> {
            let x = c::<T>(0.7) * self.distribution::<T>();
            let y = c::<T>(0.7) * self.distribution::<T>();
            let z = c::<T>(0.7) * self.distribution::<T>();
            let ax = c::<T>(0.7) * self.distribution::<T>();
            let ay = c::<T>(0.7) * self.distribution::<T>();
            let az = c::<T>(0.7) * self.distribution::<T>();
            Rigid3::new(
                Vector3::new(x, y, z),
                angle_axis_vector_to_rotation_quaternion(&Vector3::new(ax, ay, az)),
            )
        }
    }

    fn eps<T: RealField + Copy>() -> f64 {
        to_f64(T::default_epsilon())
    }

    fn identity_2d_test<T: RealField + Copy>() {
        let mut fixture = RigidTransformTest::new();
        let pose = fixture.get_random_rigid2::<T>();
        assert_is_nearly(&(pose * Rigid2::<T>::default()), &pose, eps::<T>());
        assert_is_nearly(&(Rigid2::<T>::default() * pose), &pose, eps::<T>());
        assert_is_nearly(&(pose * Rigid2::<T>::identity()), &pose, eps::<T>());
        assert_is_nearly(&(Rigid2::<T>::identity() * pose), &pose, eps::<T>());
    }

    fn inverse_2d_test<T: RealField + Copy>() {
        let mut fixture = RigidTransformTest::new();
        let pose = fixture.get_random_rigid2::<T>();
        assert_is_nearly(
            &(pose.inverse() * pose),
            &Rigid2::<T>::identity(),
            eps::<T>(),
        );
        assert_is_nearly(
            &(pose * pose.inverse()),
            &Rigid2::<T>::identity(),
            eps::<T>(),
        );
    }

    fn identity_3d_test<T: RealField + Copy>() {
        let mut fixture = RigidTransformTest::new();
        let pose = fixture.get_random_rigid3::<T>();
        assert_is_nearly(&(pose * Rigid3::<T>::default()), &pose, eps::<T>());
        assert_is_nearly(&(Rigid3::<T>::default() * pose), &pose, eps::<T>());
        assert_is_nearly(&(pose * Rigid3::<T>::identity()), &pose, eps::<T>());
        assert_is_nearly(&(Rigid3::<T>::identity() * pose), &pose, eps::<T>());
    }

    fn inverse_3d_test<T: RealField + Copy>() {
        let mut fixture = RigidTransformTest::new();
        let pose = fixture.get_random_rigid3::<T>();
        assert_is_nearly(
            &(pose.inverse() * pose),
            &Rigid3::<T>::identity(),
            eps::<T>(),
        );
        assert_is_nearly(
            &(pose * pose.inverse()),
            &Rigid3::<T>::identity(),
            eps::<T>(),
        );
    }

    // TYPED_TEST over float and double. Like in C++, each test checks one
    // random pose (the random stream differs from C++'s std::mt19937).
    mod rigid_transform_test {
        use super::*;

        #[test]
        fn identity_2d_test_f32() {
            identity_2d_test::<f32>();
        }
        #[test]
        fn identity_2d_test_f64() {
            identity_2d_test::<f64>();
        }
        #[test]
        fn inverse_2d_test_f32() {
            inverse_2d_test::<f32>();
        }
        #[test]
        fn inverse_2d_test_f64() {
            inverse_2d_test::<f64>();
        }
        #[test]
        fn identity_3d_test_f32() {
            identity_3d_test::<f32>();
        }
        #[test]
        fn identity_3d_test_f64() {
            identity_3d_test::<f64>();
        }
        #[test]
        fn inverse_3d_test_f32() {
            inverse_3d_test::<f32>();
        }
        #[test]
        fn inverse_3d_test_f64() {
            inverse_3d_test::<f64>();
        }
    }

    #[test]
    fn debug_strings() {
        let r2 = Rigid2d::new(Vector2::new(1.5, -2.), 0.25);
        assert_eq!(r2.to_string(), "{ t: [1.5, -2], r: [0.25] }");
        let r3 = Rigid3d::new(Vector3::new(1., 2., 3.), Quaternion::new(1., 0., 0., 0.));
        assert_eq!(r3.to_string(), "{ t: [1, 2, 3], q: [1, 0, 0, 0] }");
        let r3 = Rigid3f::from_translation(Vector3::new(0.1, 1. / 3., 1e-7));
        assert_eq!(
            r3.debug_string(),
            "{ t: [0.1, 0.333333, 1e-07], q: [1, 0, 0, 0] }"
        );
    }

    #[test]
    #[allow(clippy::op_ref)] // Tests the reference impls.
    fn rigid2_operations() {
        let a = Rigid2d::new(Vector2::new(1., 0.), std::f64::consts::FRAC_PI_2);
        let p = a * Vector2::new(1., 0.);
        assert!((p - Vector2::new(1., 1.)).norm() < 1e-15);
        // Angles add up without wrapping.
        let b = Rigid2d::from_rotation(3.) * Rigid2d::from_rotation(3.);
        assert_eq!(b.rotation().angle(), 6.);
        assert!((b.normalized_angle() - (6. - 2. * std::f64::consts::PI)).abs() < 1e-15);
        assert_eq!(
            Rigid2d::from_translation(Vector2::new(1., 2.)).translation(),
            &Vector2::new(1., 2.)
        );
        let f: Rigid2f = a.cast();
        assert_eq!(f.translation().x, 1f32);
        assert_eq!((&a * &a).rotation().angle(), std::f64::consts::PI);
        assert!(
            (Rotation2D::new(7.).smallest_angle() - (7. - 2. * std::f64::consts::PI)).abs() < 1e-15
        );
        assert!(
            (Rotation2D::new(-1.).smallest_positive_angle() - (2. * std::f64::consts::PI - 1.))
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn rigid3_operations() {
        let q = AngleAxis::new(std::f64::consts::FRAC_PI_2, Vector3::z()).to_quaternion();
        let a = Rigid3d::new(Vector3::new(1., 0., 0.), q);
        let p = a * Vector3::new(1., 0., 0.);
        assert!((p - Vector3::new(1., 1., 0.)).norm() < 1e-15);
        assert!(a.is_valid());
        assert!(!Rigid3d::new(Vector3::new(f64::NAN, 0., 0.), q).is_valid());
        assert!(!Rigid3d::new(Vector3::zeros(), Quaternion::new(2., 0., 0., 0.)).is_valid());
        // Composition renormalizes.
        let unnormalized = Rigid3d::from_rotation(Quaternion::new(2., 0., 0., 0.));
        assert_eq!(
            (unnormalized * unnormalized).rotation(),
            &Quaternion::new(1., 0., 0., 0.)
        );
        let from_arrays = Rigid3d::from_arrays(&[1., 0., 0., 0.], &[1., 2., 3.]);
        assert_eq!(
            from_arrays,
            Rigid3d::from_translation(Vector3::new(1., 2., 3.))
        );
        let f: Rigid3f = a.cast();
        assert!(f.is_nearly(
            &Rigid3f::new(Vector3::new(1., 0., 0.), quaternion_cast(&q)),
            1e-6
        ));
        let aa = AngleAxis::from_quaternion(&quaternion_mul(&q, &q));
        assert!((aa.angle - std::f64::consts::PI).abs() < 1e-12);
        assert!((aa.axis - Vector3::z()).norm() < 1e-12);
        let aa = AngleAxis::from_quaternion(&Quaternion::new(-1., 0., 0., 0.));
        assert_eq!(aa.angle, 0.);
    }

    #[test]
    fn roll_pitch_yaw_matches_rotation_order() {
        let q = roll_pitch_yaw(0.1, 0.2, 0.3);
        let expected = quaternion_mul(
            &quaternion_mul(
                &AngleAxis::new(0.3, Vector3::z()).to_quaternion(),
                &AngleAxis::new(0.2, Vector3::y()).to_quaternion(),
            ),
            &AngleAxis::new(0.1, Vector3::x()).to_quaternion(),
        );
        assert_eq!(q, expected);
        let nalgebra_q = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        assert!((q.coords - nalgebra_q.coords).norm() < 1e-12);
    }

    #[test]
    fn from_dictionary_with_rpy_and_quaternion() {
        let dict = make_dictionary(
            "return { translation = { 1., 2., 3. }, rotation = { 0., 0., math.pi / 2 } }",
        );
        let rigid = from_dictionary(&dict);
        assert!(rigid.is_nearly(
            &Rigid3d::with_angle_axis(
                Vector3::new(1., 2., 3.),
                AngleAxis::new(std::f64::consts::FRAC_PI_2, Vector3::z())
            ),
            1e-12
        ));
        let dict = make_dictionary(
            "return { translation = { 0, 0, 0 }, rotation = { w = 0, x = 1, y = 0, z = 0 } }",
        );
        assert_eq!(
            from_dictionary(&dict),
            Rigid3d::from_rotation(Quaternion::new(0., 1., 0., 0.))
        );
    }

    #[test]
    #[should_panic(expected = "Need (x, y, z) for translation.")]
    fn from_dictionary_bad_translation() {
        let dict =
            make_dictionary("return { translation = { 1., 2. }, rotation = { 0., 0., 0. } }");
        from_dictionary(&dict);
    }

    #[test]
    #[should_panic(expected = "rotation.norm()")]
    fn from_dictionary_unnormalized_quaternion() {
        let dict = make_dictionary(
            "return { translation = { 0, 0, 0 }, rotation = { w = 1, x = 1, y = 0, z = 0 } }",
        );
        from_dictionary(&dict);
    }

    #[test]
    fn slerp_matches_eigen() {
        let a = quaternion_identity::<f64>();
        let b = AngleAxis::new(2., Vector3::z()).to_quaternion();
        let half = quaternion_slerp(&a, 0.5, &b);
        assert!(
            (half.coords - AngleAxis::new(1., Vector3::z()).to_quaternion().coords).norm() < 1e-15
        );
        // Shortest path: -b is the same rotation.
        let minus_b = Quaternion::new(-b.w, -b.i, -b.j, -b.k);
        let half = quaternion_slerp(&a, 0.5, &minus_b);
        assert!(
            (half.coords - AngleAxis::new(1., Vector3::z()).to_quaternion().coords).norm() < 1e-15
        );
        // Linear interpolation for identical quaternions.
        assert_eq!(
            quaternion_slerp(&a, 0.3, &a),
            Quaternion::new(1., 0., 0., 0.)
        );
        assert_eq!(quaternion_slerp(&a, 0., &b), a);
    }

    #[test]
    fn rotation_matrix_matches_nalgebra() {
        let q = quaternion_normalized(&Quaternion::new(1., 2., 3., 4.));
        let m = quaternion_to_rotation_matrix(&q);
        let expected = nalgebra::UnitQuaternion::new_unchecked(q).to_rotation_matrix();
        assert!((m - expected.matrix()).norm() < 1e-15);
        let v = Vector3::new(0.3, -0.2, 0.9);
        assert!((quaternion_rotate(&q, &v) - m * v).norm() < 1e-15);
    }

    fn arb_rigid2() -> impl Strategy<Value = Rigid2d> {
        (-10.0..10.0f64, -10.0..10.0f64, -10.0..10.0f64)
            .prop_map(|(x, y, a)| Rigid2d::new(Vector2::new(x, y), a))
    }

    fn arb_rigid3() -> impl Strategy<Value = Rigid3d> {
        (
            prop::array::uniform3(-10.0..10.0f64),
            prop::array::uniform3(-3.0..3.0f64),
        )
            .prop_map(|(t, aa)| {
                Rigid3d::new(
                    Vector3::from(t),
                    angle_axis_vector_to_rotation_quaternion(&Vector3::from(aa)),
                )
            })
    }

    proptest! {
        #[test]
        fn rigid2_inverse(a in arb_rigid2()) {
            prop_assert!((a * a.inverse()).is_nearly(&Rigid2d::identity(), 1e-12));
            prop_assert!((a.inverse() * a).is_nearly(&Rigid2d::identity(), 1e-12));
            prop_assert!(a.inverse().inverse().is_nearly(&a, 1e-12));
        }

        #[test]
        fn rigid2_associativity(a in arb_rigid2(), b in arb_rigid2(), c in arb_rigid2()) {
            prop_assert!(((a * b) * c).is_nearly(&(a * (b * c)), 1e-12));
            let p = Vector2::new(0.5, -1.5);
            prop_assert!(((a * b) * p - a * (b * p)).norm() < 1e-11);
        }

        #[test]
        fn rigid3_inverse(a in arb_rigid3()) {
            prop_assert!((a * a.inverse()).is_nearly(&Rigid3d::identity(), 1e-12));
            prop_assert!((a.inverse() * a).is_nearly(&Rigid3d::identity(), 1e-12));
            prop_assert!(a.inverse().inverse().is_nearly(&a, 1e-12));
        }

        #[test]
        fn rigid3_associativity(a in arb_rigid3(), b in arb_rigid3(), c in arb_rigid3()) {
            prop_assert!(((a * b) * c).is_nearly(&(a * (b * c)), 1e-12));
            let p = Vector3::new(0.5, -1.5, 2.);
            prop_assert!(((a * b) * p - a * (b * p)).norm() < 1e-11);
            prop_assert!((a * b).is_valid());
        }
    }
}
