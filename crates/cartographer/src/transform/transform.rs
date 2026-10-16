//! Port of `cartographer/transform/transform` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ `ToProto`/`ToEigen` overload sets become the
//! [`ToProto`] and [`ToEigen`] traits with the free functions [`to_proto`]
//! and [`to_eigen`] dispatching on the argument type. `GetYaw` is split into
//! [`get_yaw`] (for `Rigid3`) and [`get_yaw_quaternion`].

use nalgebra::{Quaternion, RealField, Vector2, Vector3, Vector4};

use super::proto;
use super::rigid_transform::{
    AngleAxis, Rigid2, Rigid2d, Rigid2f, Rigid3, Rigid3d, Rigid3f, quaternion_normalized,
    quaternion_rotate,
};

#[inline]
fn c<T: RealField + Copy>(x: f64) -> T {
    nalgebra::convert(x)
}

/// Returns the non-negative rotation angle in radians of the 3D
/// transformation `transform`.
pub fn get_angle<T: RealField + Copy>(transform: &Rigid3<T>) -> T {
    let q = transform.rotation();
    let vec_norm = (q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
    c::<T>(2.) * vec_norm.atan2(q.w.abs())
}

/// Returns the yaw component in radians of the given 3D `rotation`.
/// Assuming `rotation` is composed of three rotations around X, then Y, then
/// Z, returns the angle of the Z rotation.
pub fn get_yaw_quaternion<T: RealField + Copy>(rotation: &Quaternion<T>) -> T {
    let direction = quaternion_rotate(rotation, &Vector3::x());
    direction.y.atan2(direction.x)
}

/// Returns the yaw component in radians of the given 3D transformation
/// `transform`.
pub fn get_yaw<T: RealField + Copy>(transform: &Rigid3<T>) -> T {
    get_yaw_quaternion(transform.rotation())
}

/// Returns an angle-axis vector (a vector with the length of the rotation
/// angle pointing to the direction of the rotation axis) representing the
/// same rotation as the given `quaternion`.
pub fn rotation_quaternion_to_angle_axis_vector<T: RealField + Copy>(
    quaternion: &Quaternion<T>,
) -> Vector3<T> {
    let mut normalized_quaternion = quaternion_normalized(quaternion);
    // We choose the quaternion with positive 'w', i.e., the one with a
    // smaller angle that represents this orientation.
    if normalized_quaternion.w < T::zero() {
        // Multiply by -1. http://eigen.tuxfamily.org/bz/show_bug.cgi?id=560
        let minus_one = c::<T>(-1.);
        normalized_quaternion = Quaternion::new(
            minus_one * normalized_quaternion.w,
            minus_one * normalized_quaternion.i,
            minus_one * normalized_quaternion.j,
            minus_one * normalized_quaternion.k,
        );
    }
    // We convert the normalized_quaternion into a vector along the rotation
    // axis with length of the rotation angle.
    let q = &normalized_quaternion;
    let vec_norm = (q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
    let angle = c::<T>(2.) * vec_norm.atan2(q.w);
    const CUTOFF_ANGLE: f64 = 1e-7; // We linearize below this angle.
    let scale = if angle < c(CUTOFF_ANGLE) {
        c::<T>(2.)
    } else {
        angle / (angle / c(2.)).sin()
    };
    Vector3::new(scale * q.i, scale * q.j, scale * q.k)
}

/// Returns a quaternion representing the same rotation as the given
/// `angle_axis` vector.
pub fn angle_axis_vector_to_rotation_quaternion<T: RealField + Copy>(
    angle_axis: &Vector3<T>,
) -> Quaternion<T> {
    let mut scale = c::<T>(0.5);
    let mut w = c::<T>(1.);
    const CUTOFF_ANGLE: f64 = 1e-8; // We linearize below this angle.
    let squared_norm =
        angle_axis.x * angle_axis.x + angle_axis.y * angle_axis.y + angle_axis.z * angle_axis.z;
    if squared_norm > c(CUTOFF_ANGLE) {
        let norm = squared_norm.sqrt();
        scale = (norm / c(2.)).sin() / norm;
        w = (norm / c(2.)).cos();
    }
    let quaternion_xyz = angle_axis * scale;
    Quaternion::new(w, quaternion_xyz.x, quaternion_xyz.y, quaternion_xyz.z)
}

/// Projects `transform` onto the XY plane.
pub fn project_2d<T: RealField + Copy>(transform: &Rigid3<T>) -> Rigid2<T> {
    let t = transform.translation();
    Rigid2::new(Vector2::new(t.x, t.y), get_yaw(transform))
}

/// Embeds `transform` into 3D space in the XY plane.
pub fn embed_3d<T: RealField + Copy>(transform: &Rigid2<T>) -> Rigid3<T> {
    let t = transform.translation();
    Rigid3::with_angle_axis(
        Vector3::new(t.x, t.y, T::zero()),
        AngleAxis::new(transform.rotation().angle(), Vector3::z()),
    )
}

/// Conversion from a proto message to the corresponding nalgebra type
/// (C++ `ToEigen` overloads).
pub trait ToEigen {
    /// The nalgebra type.
    type Output;
    /// Converts the proto (no normalization).
    fn to_eigen(&self) -> Self::Output;
}

/// Converts a proto to the corresponding nalgebra type (C++ `ToEigen`).
pub fn to_eigen<P: ToEigen>(proto: &P) -> P::Output {
    proto.to_eigen()
}

impl ToEigen for proto::Vector2d {
    type Output = Vector2<f64>;
    fn to_eigen(&self) -> Vector2<f64> {
        Vector2::new(self.x, self.y)
    }
}

impl ToEigen for proto::Vector3f {
    type Output = Vector3<f32>;
    fn to_eigen(&self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.z)
    }
}

impl ToEigen for proto::Vector4f {
    type Output = Vector4<f32>;
    fn to_eigen(&self) -> Vector4<f32> {
        Vector4::new(self.x, self.y, self.z, self.t)
    }
}

impl ToEigen for proto::Vector3d {
    type Output = Vector3<f64>;
    fn to_eigen(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

impl ToEigen for proto::Quaterniond {
    type Output = Quaternion<f64>;
    fn to_eigen(&self) -> Quaternion<f64> {
        Quaternion::new(self.w, self.x, self.y, self.z)
    }
}

/// Conversion to the corresponding proto message (C++ `ToProto` overloads).
pub trait ToProto {
    /// The proto message type.
    type Proto;
    /// Converts to the proto message.
    fn to_proto(&self) -> Self::Proto;
}

/// Converts to the corresponding proto message (C++ `ToProto`).
pub fn to_proto<T: ToProto + ?Sized>(value: &T) -> T::Proto {
    value.to_proto()
}

/// Converts a proto to a `Rigid2d`.
pub fn to_rigid2(transform: &proto::Rigid2d) -> Rigid2d {
    let translation = transform.translation.unwrap_or_default();
    Rigid2d::new(
        Vector2::new(translation.x, translation.y),
        transform.rotation,
    )
}

/// Converts a proto to a `Rigid3d` (no normalization; a missing rotation
/// is the all-zero quaternion, like the proto3 default in C++).
pub fn to_rigid3(rigid: &proto::Rigid3d) -> Rigid3d {
    Rigid3d::new(
        rigid.translation.unwrap_or_default().to_eigen(),
        rigid.rotation.unwrap_or_default().to_eigen(),
    )
}

impl ToProto for Rigid2d {
    type Proto = proto::Rigid2d;
    fn to_proto(&self) -> proto::Rigid2d {
        proto::Rigid2d {
            translation: Some(proto::Vector2d {
                x: self.translation().x,
                y: self.translation().y,
            }),
            rotation: self.rotation().angle(),
        }
    }
}

impl ToProto for Rigid2f {
    type Proto = proto::Rigid2f;
    fn to_proto(&self) -> proto::Rigid2f {
        proto::Rigid2f {
            translation: Some(proto::Vector2f {
                x: self.translation().x,
                y: self.translation().y,
            }),
            rotation: self.rotation().angle(),
        }
    }
}

impl ToProto for Rigid3d {
    type Proto = proto::Rigid3d;
    fn to_proto(&self) -> proto::Rigid3d {
        proto::Rigid3d {
            translation: Some(self.translation().to_proto()),
            rotation: Some(self.rotation().to_proto()),
        }
    }
}

impl ToProto for Rigid3f {
    type Proto = proto::Rigid3f;
    fn to_proto(&self) -> proto::Rigid3f {
        proto::Rigid3f {
            translation: Some(self.translation().to_proto()),
            rotation: Some(self.rotation().to_proto()),
        }
    }
}

impl ToProto for Vector2<f64> {
    type Proto = proto::Vector2d;
    fn to_proto(&self) -> proto::Vector2d {
        proto::Vector2d {
            x: self.x,
            y: self.y,
        }
    }
}

impl ToProto for Vector3<f32> {
    type Proto = proto::Vector3f;
    fn to_proto(&self) -> proto::Vector3f {
        proto::Vector3f {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl ToProto for Vector4<f32> {
    type Proto = proto::Vector4f;
    fn to_proto(&self) -> proto::Vector4f {
        proto::Vector4f {
            x: self.x,
            y: self.y,
            z: self.z,
            t: self.w,
        }
    }
}

impl ToProto for Vector3<f64> {
    type Proto = proto::Vector3d;
    fn to_proto(&self) -> proto::Vector3d {
        proto::Vector3d {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl ToProto for Quaternion<f32> {
    type Proto = proto::Quaternionf;
    fn to_proto(&self) -> proto::Quaternionf {
        proto::Quaternionf {
            x: self.i,
            y: self.j,
            z: self.k,
            w: self.w,
        }
    }
}

impl ToProto for Quaternion<f64> {
    type Proto = proto::Quaterniond;
    fn to_proto(&self) -> proto::Quaterniond {
        proto::Quaterniond {
            x: self.i,
            y: self.j,
            z: self.k,
            w: self.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::{quaternion_mul, quaternion_normalized};
    use crate::transform::rigid_transform_test_helpers::{IsNearly, assert_is_nearly};
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    mod transform_test {
        use super::*;

        #[test]
        fn get_angle() {
            let mut rng = StdRng::seed_from_u64(42);
            for _ in 0..100 {
                let angle: f32 = rng.random_range(0f32..std::f32::consts::PI);
                let x: f32 = rng.random_range(-1f32..1f32);
                let y: f32 = rng.random_range(-1f32..1f32);
                let z: f32 = rng.random_range(-1f32..1f32);
                let axis = Vector3::new(x, y, z).normalize();
                let actual = super::super::get_angle(&Rigid3f::from_rotation(
                    angle_axis_vector_to_rotation_quaternion(&(axis * angle)),
                ));
                assert!((angle - actual).abs() <= 1e-6f32, "{angle} vs {actual}");
            }
        }

        #[test]
        fn get_yaw() {
            let rotation = Rigid3d::from_angle_axis(AngleAxis::new(1.2345, Vector3::z()));
            assert!((1.2345 - super::super::get_yaw(&rotation)).abs() <= 1e-9);
            assert!((-1.2345 - super::super::get_yaw(&rotation.inverse())).abs() <= 1e-9);
        }

        #[test]
        fn get_yaw_axis_ordering() {
            let rotation = Rigid3d::from_rotation(
                (AngleAxis::new(1.2345, Vector3::z()) * AngleAxis::new(0.4321, Vector3::y()))
                    * AngleAxis::new(0.6789, Vector3::x()),
            );
            assert!((1.2345 - super::super::get_yaw(&rotation)).abs() <= 1e-9);
        }

        #[test]
        fn embed_3d() {
            let rigid2d = Rigid2d::new(Vector2::new(1., 2.), 0.);
            let rigid3d = Rigid3d::from_translation(Vector3::new(1., 2., 3.))
                * Rigid3d::from_rotation(quaternion_normalized(&Quaternion::new(1., 2., 3., 4.)));
            let expected = rigid3d * Rigid3d::from_translation(Vector3::new(1., 2., 0.));
            assert_is_nearly(
                &expected,
                &(rigid3d * super::super::embed_3d(&rigid2d)),
                1e-9,
            );
        }
    }

    #[test]
    fn project_2d_and_embed_3d() {
        let rigid2 = Rigid2d::new(Vector2::new(1., -2.), 0.75);
        assert_is_nearly(&project_2d(&embed_3d(&rigid2)), &rigid2, 1e-12);
    }

    #[test]
    fn angle_axis_linearization() {
        let q = angle_axis_vector_to_rotation_quaternion(&Vector3::new(1e-5, 0., 0.));
        assert_eq!(q, Quaternion::new(1., 0.5e-5, 0., 0.));
        let v = rotation_quaternion_to_angle_axis_vector(&Quaternion::new(1f64, 1e-9, 0., 0.));
        assert!((v.x - 2e-9).abs() < 1e-20);
        // The quaternion with negative w is flipped.
        let q = angle_axis_vector_to_rotation_quaternion(&Vector3::new(0., 0., 1.));
        let minus_q = Quaternion::new(-q.w, -q.i, -q.j, -q.k);
        assert!(
            (rotation_quaternion_to_angle_axis_vector(&minus_q) - Vector3::new(0., 0., 1.)).norm()
                < 1e-15
        );
    }

    #[test]
    fn proto_conversions() {
        let v2 = Vector2::new(1., 2.);
        assert_eq!(to_eigen(&to_proto(&v2)), v2);
        let v3f = Vector3::new(1f32, 2., 3.);
        assert_eq!(to_eigen(&to_proto(&v3f)), v3f);
        let v4f = Vector4::new(1f32, 2., 3., 4.);
        assert_eq!(to_proto(&v4f).t, 4.);
        assert_eq!(to_eigen(&to_proto(&v4f)), v4f);
        let v3d = Vector3::new(1., 2., 3.);
        assert_eq!(to_eigen(&to_proto(&v3d)), v3d);
        let qd = Quaternion::new(1f64, 2., 3., 4.);
        let proto = to_proto(&qd);
        assert_eq!((proto.x, proto.y, proto.z, proto.w), (2., 3., 4., 1.));
        assert_eq!(to_eigen(&proto), qd);
        let qf = Quaternion::new(1f32, 2., 3., 4.);
        assert_eq!(to_proto(&qf).w, 1.);
        let r2 = Rigid2d::new(Vector2::new(1., 2.), 7.);
        assert_eq!(to_rigid2(&to_proto(&r2)), r2);
        let r2f = Rigid2f::new(Vector2::new(1., 2.), 7.);
        assert_eq!(to_proto(&r2f).rotation, 7.);
        let r3f = Rigid3f::from_translation(Vector3::new(1., 2., 3.));
        assert_eq!(to_proto(&r3f).rotation.unwrap().w, 1.);
        // Default proto: zero quaternion (not identity), like C++.
        let default_rigid3 = to_rigid3(&proto::Rigid3d::default());
        assert_eq!(default_rigid3.rotation(), &Quaternion::new(0., 0., 0., 0.));
        assert_eq!(default_rigid3.translation(), &Vector3::zeros());
    }

    fn arb_quaternion() -> impl Strategy<Value = Quaternion<f64>> {
        prop::array::uniform3(-3.0..3.0f64)
            .prop_map(|aa| angle_axis_vector_to_rotation_quaternion(&Vector3::from(aa)))
    }

    proptest! {
        #[test]
        fn rigid3_proto_round_trip(t in prop::array::uniform3(-100.0..100.0f64), q in arb_quaternion()) {
            let rigid = Rigid3d::new(Vector3::from(t), q);
            prop_assert_eq!(to_rigid3(&to_proto(&rigid)), rigid);
        }

        #[test]
        fn rigid2_proto_round_trip(x in -100.0..100.0f64, y in -100.0..100.0f64, a in -10.0..10.0f64) {
            let rigid = Rigid2d::new(Vector2::new(x, y), a);
            prop_assert_eq!(to_rigid2(&to_proto(&rigid)), rigid);
        }

        #[test]
        fn angle_axis_round_trip(aa in prop::array::uniform3(-1.8..1.8f64)) {
            // |aa| < pi, so the angle-axis vector is recovered.
            let aa = Vector3::from(aa);
            let q = angle_axis_vector_to_rotation_quaternion(&aa);
            prop_assert!((rotation_quaternion_to_angle_axis_vector(&q) - aa).norm() < 1e-9);
            prop_assert!((get_angle(&Rigid3d::from_rotation(q)) - aa.norm()).abs() < 1e-9);
        }

        #[test]
        fn yaw_of_composed_rotation(yaw in -3.1..3.1f64, pitch in -1.5..1.5f64, roll in -3.1..3.1f64) {
            let q = quaternion_mul(
                &quaternion_mul(
                    &AngleAxis::new(yaw, Vector3::z()).to_quaternion(),
                    &AngleAxis::new(pitch, Vector3::y()).to_quaternion(),
                ),
                &AngleAxis::new(roll, Vector3::x()).to_quaternion(),
            );
            prop_assert!((get_yaw_quaternion(&q) - yaw).abs() < 1e-9);
            let projected = project_2d(&Rigid3d::from_rotation(q));
            prop_assert!(projected.is_nearly(&Rigid2d::from_rotation(yaw), 1e-9));
        }
    }
}
