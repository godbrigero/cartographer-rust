//! Port of `cartographer/cloud/internal/testing/test_helpers` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ `BuildDataPredicateEquals<Request>` downcasts a
//! `sensor::Data*` to `Dispatchable<T>`. `sensor::Data` has no downcast
//! support in Rust, so [`dispatch`] feeds the packet into a recording
//! trajectory builder instead and [`DispatchedData`] holds what arrived;
//! [`DataPredicateEquals`] compares it with a request. The C++
//! `BuildProtoPredicateEquals` (`MessageDifferencer::Equals`) is plain `==`
//! on prost messages.

use cartographer::mapping::internal::local_slam_result_data::LocalSlamResultData;
use cartographer::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use cartographer::sensor::data::Data as SensorData;
use cartographer::sensor::fixed_frame_pose_data::{self, FixedFramePoseData};
use cartographer::sensor::imu_data::{self, ImuData};
use cartographer::sensor::landmark_data::{self, LandmarkData};
use cartographer::sensor::odometry_data::{self, OdometryData};
use cartographer::sensor::timed_point_cloud_data::{self, TimedPointCloudData};

use crate::proto;

/// The payload of a dispatched sensor data packet, with its sensor id.
#[derive(Clone, Debug, PartialEq)]
pub enum DispatchedData {
    /// `Dispatchable<TimedPointCloudData>`.
    TimedPointCloud(String, TimedPointCloudData),
    /// `Dispatchable<ImuData>`.
    Imu(String, ImuData),
    /// `Dispatchable<OdometryData>`.
    Odometry(String, OdometryData),
    /// `Dispatchable<FixedFramePoseData>`.
    FixedFramePose(String, FixedFramePoseData),
    /// `Dispatchable<LandmarkData>`.
    Landmark(String, LandmarkData),
    /// A `LocalSlamResultData` packet (sensor id only).
    LocalSlamResult(String),
}

#[derive(Default)]
struct Recorder(Option<DispatchedData>);

impl TrajectoryBuilderInterface for Recorder {
    fn add_timed_point_cloud_data(&mut self, sensor_id: &str, data: &TimedPointCloudData) {
        self.0 = Some(DispatchedData::TimedPointCloud(
            sensor_id.to_string(),
            data.clone(),
        ));
    }
    fn add_imu_data(&mut self, sensor_id: &str, data: &ImuData) {
        self.0 = Some(DispatchedData::Imu(sensor_id.to_string(), *data));
    }
    fn add_odometry_data(&mut self, sensor_id: &str, data: &OdometryData) {
        self.0 = Some(DispatchedData::Odometry(sensor_id.to_string(), *data));
    }
    fn add_fixed_frame_pose_data(&mut self, sensor_id: &str, data: &FixedFramePoseData) {
        self.0 = Some(DispatchedData::FixedFramePose(sensor_id.to_string(), *data));
    }
    fn add_landmark_data(&mut self, sensor_id: &str, data: &LandmarkData) {
        self.0 = Some(DispatchedData::Landmark(
            sensor_id.to_string(),
            data.clone(),
        ));
    }
    fn add_local_slam_result_data(&mut self, data: Box<dyn LocalSlamResultData>) {
        self.0 = Some(DispatchedData::LocalSlamResult(
            data.get_sensor_id().to_string(),
        ));
    }
}

/// Dispatches `data` into a recording trajectory builder and returns what
/// arrived.
pub fn dispatch(data: Box<dyn SensorData>) -> DispatchedData {
    let mut recorder = Recorder::default();
    data.add_to_trajectory_builder(&mut recorder);
    recorder
        .0
        .expect("the packet did not call any trajectory builder method")
}

/// C++ `BuildDataPredicateEquals<Request>`: whether a dispatched packet holds
/// exactly the request's payload and sensor id.
pub trait DataPredicateEquals {
    /// Returns true if `data` matches `self`.
    fn data_equals(&self, data: &DispatchedData) -> bool;
}

fn sensor_id_of(metadata: &Option<proto::SensorMetadata>) -> String {
    metadata.clone().unwrap_or_default().sensor_id
}

impl DataPredicateEquals for proto::AddImuDataRequest {
    fn data_equals(&self, data: &DispatchedData) -> bool {
        matches!(data, DispatchedData::Imu(sensor_id, imu)
            if imu_data::to_proto(imu) == self.imu_data.unwrap_or_default()
                && *sensor_id == sensor_id_of(&self.sensor_metadata))
    }
}

impl DataPredicateEquals for proto::AddFixedFramePoseDataRequest {
    fn data_equals(&self, data: &DispatchedData) -> bool {
        matches!(data, DispatchedData::FixedFramePose(sensor_id, pose)
            if fixed_frame_pose_data::to_proto(pose)
                == self.fixed_frame_pose_data.unwrap_or_default()
                && *sensor_id == sensor_id_of(&self.sensor_metadata))
    }
}

impl DataPredicateEquals for proto::AddOdometryDataRequest {
    fn data_equals(&self, data: &DispatchedData) -> bool {
        matches!(data, DispatchedData::Odometry(sensor_id, odometry)
            if odometry_data::to_proto(odometry) == self.odometry_data.unwrap_or_default()
                && *sensor_id == sensor_id_of(&self.sensor_metadata))
    }
}

impl DataPredicateEquals for proto::AddLandmarkDataRequest {
    fn data_equals(&self, data: &DispatchedData) -> bool {
        matches!(data, DispatchedData::Landmark(sensor_id, landmark)
            if landmark_data::to_proto(landmark)
                == self.landmark_data.clone().unwrap_or_default()
                && *sensor_id == sensor_id_of(&self.sensor_metadata))
    }
}

impl DataPredicateEquals for proto::AddRangefinderDataRequest {
    fn data_equals(&self, data: &DispatchedData) -> bool {
        matches!(data, DispatchedData::TimedPointCloud(sensor_id, cloud)
            if timed_point_cloud_data::to_proto(cloud)
                == self.timed_point_cloud_data.clone().unwrap_or_default()
                && *sensor_id == sensor_id_of(&self.sensor_metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartographer::common::time::from_universal;
    use cartographer::sensor::internal::dispatchable::make_dispatchable;

    #[test]
    fn dispatch_records_payload_and_predicate_matches() {
        let request = proto::AddImuDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "imu".to_string(),
                client_id: String::new(),
            }),
            imu_data: Some(imu_data::to_proto(&ImuData {
                time: from_universal(2),
                ..Default::default()
            })),
        };
        let data = dispatch(make_dispatchable(
            "imu",
            imu_data::from_proto(&request.imu_data.unwrap()),
        ));
        assert!(request.data_equals(&data));
        let other = dispatch(make_dispatchable(
            "other",
            imu_data::from_proto(&request.imu_data.unwrap()),
        ));
        assert!(!request.data_equals(&other));
    }
}
