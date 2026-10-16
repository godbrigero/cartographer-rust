//! Port of `cartographer/sensor/fixed_frame_pose_data` (.h, .cc).

use crate::common::time::{Time, from_universal, to_universal};
use crate::sensor::data::HasTime;
use crate::sensor::proto;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{ToProto, to_rigid3};

/// The fixed frame pose data (like GPS, pose, etc.) will be used in the
/// optimization.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FixedFramePoseData {
    /// Measurement time.
    pub time: Time,
    /// The pose, if available.
    pub pose: Option<Rigid3d>,
}

impl HasTime for FixedFramePoseData {
    fn time(&self) -> Time {
        self.time
    }
}

/// Converts `pose_data` to a `proto::FixedFramePoseData` (the pose field is
/// only set if present).
pub fn to_proto(pose_data: &FixedFramePoseData) -> proto::FixedFramePoseData {
    proto::FixedFramePoseData {
        timestamp: to_universal(pose_data.time),
        pose: pose_data.pose.as_ref().map(ToProto::to_proto),
    }
}

/// Converts `proto` to `FixedFramePoseData`.
pub fn from_proto(proto: &proto::FixedFramePoseData) -> FixedFramePoseData {
    FixedFramePoseData {
        time: from_universal(proto.timestamp),
        pose: proto.pose.as_ref().map(to_rigid3),
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
            has_pose in any::<bool>(),
            tr in prop::array::uniform3(-1e3f64..1e3),
            q in prop::array::uniform4(-1f64..1.),
        ) {
            let data = FixedFramePoseData {
                time: from_universal(t),
                pose: has_pose.then(|| {
                    Rigid3d::new(Vector3::from(tr), Quaternion::new(q[0], q[1], q[2], q[3]))
                }),
            };
            let proto = to_proto(&data);
            prop_assert_eq!(proto.pose.is_some(), has_pose);
            prop_assert_eq!(from_proto(&proto), data);
        }
    }
}
