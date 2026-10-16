//! Port of `cartographer/sensor/odometry_data` (.h, .cc).

use crate::common::time::{Time, from_universal, to_universal};
use crate::sensor::data::HasTime;
use crate::sensor::proto;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{ToProto, to_rigid3};

/// An odometry pose measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OdometryData {
    /// Measurement time.
    pub time: Time,
    /// Pose of the tracking frame in the odometry frame.
    pub pose: Rigid3d,
}

impl HasTime for OdometryData {
    fn time(&self) -> Time {
        self.time
    }
}

/// Converts `odometry_data` to a `proto::OdometryData`.
pub fn to_proto(odometry_data: &OdometryData) -> proto::OdometryData {
    proto::OdometryData {
        timestamp: to_universal(odometry_data.time),
        pose: Some(odometry_data.pose.to_proto()),
    }
}

/// Converts `proto` to `OdometryData` (the quaternion is not normalized; a
/// missing pose is the C++ default instance: zero translation, all-zero
/// quaternion).
pub fn from_proto(proto: &proto::OdometryData) -> OdometryData {
    OdometryData {
        time: from_universal(proto.timestamp),
        pose: to_rigid3(&proto.pose.unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Quaternion, Vector3};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proto_round_trip(
            t in any::<i64>(),
            tr in prop::array::uniform3(-1e3f64..1e3),
            q in prop::array::uniform4(-1f64..1.),
        ) {
            let odometry_data = OdometryData {
                time: from_universal(t),
                pose: Rigid3d::new(Vector3::from(tr), Quaternion::new(q[0], q[1], q[2], q[3])),
            };
            prop_assert_eq!(from_proto(&to_proto(&odometry_data)), odometry_data);
        }
    }
}
