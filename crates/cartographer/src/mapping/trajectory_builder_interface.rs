//! Port of `cartographer/mapping/trajectory_builder_interface` (.h, .cc).
//!
//! NOTE: extended by the map builder port. Only the parts needed by
//! `sensor::Dispatchable` are ported so far: the `AddSensorData` overloads
//! and `SensorId` with its proto conversions. Pending: `InsertionResult`,
//! `LocalSlamResultCallback`, `AddLocalSlamResultData` and
//! `CreateTrajectoryBuilderOptions`.

use crate::mapping::proto;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;

/// Sensor type of a [`SensorId`] (C++ `SensorId::SensorType`). The
/// declaration order equals the C++ enum values and defines the ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SensorType {
    /// C++ `RANGE`.
    Range = 0,
    /// C++ `IMU`.
    Imu,
    /// C++ `ODOMETRY`.
    Odometry,
    /// C++ `FIXED_FRAME_POSE`.
    FixedFramePose,
    /// C++ `LANDMARK`.
    Landmark,
    /// C++ `LOCAL_SLAM_RESULT`.
    LocalSlamResult,
}

/// Identifies a sensor of a trajectory (C++
/// `TrajectoryBuilderInterface::SensorId`). Ordered by `(type, id)` like in
/// C++.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId {
    /// The sensor type.
    pub r#type: SensorType,
    /// The sensor id (topic name).
    pub id: String,
}

/// This interface is used for both 2D and 3D SLAM. Implementations wire up a
/// global SLAM stack, i.e. local SLAM for initial pose estimates, scan matching
/// to detect loop closure, and a sparse pose graph optimization to compute
/// optimized pose estimates.
///
/// RUST PORT NOTE: Rust has no overloading, so the C++ `AddSensorData`
/// overloads are disambiguated by a type suffix:
///
/// | C++ | Rust |
/// |---|---|
/// | `AddSensorData(id, const TimedPointCloudData&)` | [`add_timed_point_cloud_data`](Self::add_timed_point_cloud_data) |
/// | `AddSensorData(id, const ImuData&)` | [`add_imu_data`](Self::add_imu_data) |
/// | `AddSensorData(id, const OdometryData&)` | [`add_odometry_data`](Self::add_odometry_data) |
/// | `AddSensorData(id, const FixedFramePoseData&)` | [`add_fixed_frame_pose_data`](Self::add_fixed_frame_pose_data) |
/// | `AddSensorData(id, const LandmarkData&)` | [`add_landmark_data`](Self::add_landmark_data) |
// NOTE: extended by the map builder port.
pub trait TrajectoryBuilderInterface {
    /// C++ `AddSensorData(sensor_id, const sensor::TimedPointCloudData&)`.
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    );
    /// C++ `AddSensorData(sensor_id, const sensor::ImuData&)`.
    fn add_imu_data(&mut self, sensor_id: &str, imu_data: &ImuData);
    /// C++ `AddSensorData(sensor_id, const sensor::OdometryData&)`.
    fn add_odometry_data(&mut self, sensor_id: &str, odometry_data: &OdometryData);
    /// C++ `AddSensorData(sensor_id, const sensor::FixedFramePoseData&)`.
    fn add_fixed_frame_pose_data(&mut self, sensor_id: &str, fixed_frame_pose: &FixedFramePoseData);
    /// C++ `AddSensorData(sensor_id, const sensor::LandmarkData&)`.
    fn add_landmark_data(&mut self, sensor_id: &str, landmark_data: &LandmarkData);
}

/// Converts `sensor_id` to a `proto::SensorId`.
pub fn to_proto(sensor_id: &SensorId) -> proto::SensorId {
    use proto::sensor_id::SensorType as P;
    let r#type = match sensor_id.r#type {
        SensorType::Range => P::Range,
        SensorType::Imu => P::Imu,
        SensorType::Odometry => P::Odometry,
        SensorType::FixedFramePose => P::FixedFramePose,
        SensorType::Landmark => P::Landmark,
        SensorType::LocalSlamResult => P::LocalSlamResult,
    };
    proto::SensorId {
        r#type: r#type as i32,
        id: sensor_id.id.clone(),
    }
}

/// Converts `sensor_id_proto` to a `SensorId`. Panics on an unknown sensor
/// type (C++ `LOG(FATAL)`).
pub fn from_proto(sensor_id_proto: &proto::SensorId) -> SensorId {
    use proto::sensor_id::SensorType as P;
    let r#type = match P::try_from(sensor_id_proto.r#type) {
        Ok(P::Range) => SensorType::Range,
        Ok(P::Imu) => SensorType::Imu,
        Ok(P::Odometry) => SensorType::Odometry,
        Ok(P::FixedFramePose) => SensorType::FixedFramePose,
        Ok(P::Landmark) => SensorType::Landmark,
        Ok(P::LocalSlamResult) => SensorType::LocalSlamResult,
        Err(_) => panic!("Unsupported sensor type."),
    };
    SensorId {
        r#type,
        id: sensor_id_proto.id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_id_proto_round_trip() {
        for r#type in [
            SensorType::Range,
            SensorType::Imu,
            SensorType::Odometry,
            SensorType::FixedFramePose,
            SensorType::Landmark,
            SensorType::LocalSlamResult,
        ] {
            let sensor_id = SensorId {
                r#type,
                id: "some_sensor".to_string(),
            };
            let proto = to_proto(&sensor_id);
            assert_eq!(proto.r#type, r#type as i32);
            assert_eq!(from_proto(&proto), sensor_id);
        }
    }

    #[test]
    fn sensor_id_ordering_is_type_then_id() {
        let a = SensorId {
            r#type: SensorType::Range,
            id: "z".to_string(),
        };
        let b = SensorId {
            r#type: SensorType::Imu,
            id: "a".to_string(),
        };
        assert!(a < b);
    }

    #[test]
    #[should_panic(expected = "Unsupported sensor type.")]
    fn unknown_sensor_type_panics() {
        let _ = from_proto(&proto::SensorId {
            r#type: 42,
            id: String::new(),
        });
    }
}
