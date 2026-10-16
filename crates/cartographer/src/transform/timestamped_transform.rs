//! Port of `cartographer/transform/timestamped_transform` (.h, .cc).

use super::proto;
use super::rigid_transform::{Rigid3d, quaternion_slerp};
use super::transform::{ToProto, to_rigid3};
use crate::common::time::{Time, from_universal, to_seconds, to_universal};

/// A rigid transform at a point in time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedTransform {
    /// The time.
    pub time: Time,
    /// The transform.
    pub transform: Rigid3d,
}

/// Converts from proto.
pub fn from_proto(proto: &proto::TimestampedTransform) -> TimestampedTransform {
    TimestampedTransform {
        time: from_universal(proto.time),
        transform: to_rigid3(&proto.transform.unwrap_or_default()),
    }
}

/// Converts to proto.
pub fn to_proto(transform: &TimestampedTransform) -> proto::TimestampedTransform {
    proto::TimestampedTransform {
        time: to_universal(transform.time),
        transform: Some(transform.transform.to_proto()),
    }
}

impl ToProto for TimestampedTransform {
    type Proto = proto::TimestampedTransform;
    fn to_proto(&self) -> proto::TimestampedTransform {
        to_proto(self)
    }
}

/// Interpolates between `start` and `end` at `time` (linearly for the
/// translation, Eigen-style slerp for the rotation). Panics unless
/// `start.time <= time <= end.time`.
pub fn interpolate(
    start: &TimestampedTransform,
    end: &TimestampedTransform,
    time: Time,
) -> TimestampedTransform {
    assert!(
        start.time <= time,
        "Check failed: start.time <= time ({} vs. {time})",
        start.time
    );
    assert!(
        end.time >= time,
        "Check failed: end.time >= time ({} vs. {time})",
        end.time
    );

    let duration = to_seconds(end.time - start.time);
    let factor = to_seconds(time - start.time) / duration;
    let origin = start.transform.translation()
        + (end.transform.translation() - start.transform.translation()) * factor;
    let rotation = quaternion_slerp(start.transform.rotation(), factor, end.transform.rotation());
    TimestampedTransform {
        time,
        transform: Rigid3d::new(origin, rotation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::{AngleAxis, quaternion_normalized};
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use nalgebra::{Quaternion, Vector3};

    mod timestamped_transform_test {
        use super::*;

        #[test]
        fn to_proto_and_back() {
            let expected = TimestampedTransform {
                time: from_universal(12345678),
                transform: Rigid3d::new(
                    Vector3::new(1., 2., 3.),
                    quaternion_normalized(&Quaternion::new(1., 2., 3., 4.)),
                ),
            };
            let actual = from_proto(&to_proto(&expected));
            assert_eq!(expected.time, actual.time);
            assert_is_nearly(&actual.transform, &expected.transform, 1e-6);
        }
    }

    #[test]
    fn interpolate_halfway() {
        let start = TimestampedTransform {
            time: from_universal(0),
            transform: Rigid3d::identity(),
        };
        let end = TimestampedTransform {
            time: from_universal(100),
            transform: Rigid3d::with_angle_axis(
                Vector3::new(2., 4., 6.),
                AngleAxis::new(1., Vector3::z()),
            ),
        };
        let result = interpolate(&start, &end, from_universal(50));
        assert_eq!(result.time, from_universal(50));
        assert_is_nearly(
            &result.transform,
            &Rigid3d::with_angle_axis(Vector3::new(1., 2., 3.), AngleAxis::new(0.5, Vector3::z())),
            1e-12,
        );
        assert_eq!(
            interpolate(&start, &end, from_universal(0)).transform,
            start.transform
        );
    }

    #[test]
    #[should_panic(expected = "start.time <= time")]
    fn interpolate_before_start_panics() {
        let a = TimestampedTransform {
            time: from_universal(10),
            transform: Rigid3d::identity(),
        };
        interpolate(&a, &a, from_universal(5));
    }
}
