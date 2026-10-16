//! Port of `cartographer/sensor/imu_data` (.h, .cc).

use nalgebra::Vector3;

use crate::common::time::{Time, from_universal, to_universal};
use crate::sensor::data::HasTime;
use crate::sensor::proto;
use crate::transform::transform::{ToEigen, ToProto};

/// A single IMU measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImuData {
    /// Measurement time.
    pub time: Time,
    /// Linear acceleration (including gravity) in m/s^2.
    pub linear_acceleration: Vector3<f64>,
    /// Angular velocity in rad/s.
    pub angular_velocity: Vector3<f64>,
}

impl HasTime for ImuData {
    fn time(&self) -> Time {
        self.time
    }
}

/// Converts `imu_data` to a `proto::ImuData`.
pub fn to_proto(imu_data: &ImuData) -> proto::ImuData {
    proto::ImuData {
        timestamp: to_universal(imu_data.time),
        linear_acceleration: Some(imu_data.linear_acceleration.to_proto()),
        angular_velocity: Some(imu_data.angular_velocity.to_proto()),
    }
}

/// Converts `proto` to `ImuData`.
pub fn from_proto(proto: &proto::ImuData) -> ImuData {
    ImuData {
        time: from_universal(proto.timestamp),
        linear_acceleration: proto.linear_acceleration.unwrap_or_default().to_eigen(),
        angular_velocity: proto.angular_velocity.unwrap_or_default().to_eigen(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proto_round_trip(
            t in any::<i64>(),
            a in prop::array::uniform3(-1e3f64..1e3),
            w in prop::array::uniform3(-1e3f64..1e3),
        ) {
            let imu_data = ImuData {
                time: from_universal(t),
                linear_acceleration: Vector3::from(a),
                angular_velocity: Vector3::from(w),
            };
            prop_assert_eq!(from_proto(&to_proto(&imu_data)), imu_data);
        }
    }
}
