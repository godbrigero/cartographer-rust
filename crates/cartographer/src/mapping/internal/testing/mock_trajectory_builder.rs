//! Port of `cartographer/mapping/internal/testing/mock_trajectory_builder` (.h).
//!
//! RUST PORT NOTE: the gmock mock becomes a hand-written fake that records
//! every call; tests inspect the shared [`MockTrajectoryBuilder::calls`] log.

use std::sync::{Arc, Mutex, PoisonError};

use crate::common::time::Time;
use crate::mapping::internal::local_slam_result_data::LocalSlamResultData;
use crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;

/// The kind of a recorded call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallKind {
    /// `AddSensorData(TimedPointCloudData)`.
    TimedPointCloud,
    /// `AddSensorData(ImuData)`.
    Imu,
    /// `AddSensorData(OdometryData)`.
    Odometry,
    /// `AddSensorData(FixedFramePoseData)`.
    FixedFramePose,
    /// `AddSensorData(LandmarkData)`.
    Landmark,
    /// `AddLocalSlamResultData` (C++ `DoAddLocalSlamResultData`).
    LocalSlamResult,
}

/// A recorded call: `(kind, sensor_id, time)`.
pub type Call = (CallKind, String, Time);

/// Fake `TrajectoryBuilderInterface` recording all calls.
#[derive(Debug, Default, Clone)]
pub struct MockTrajectoryBuilder {
    /// The shared call log.
    pub calls: Arc<Mutex<Vec<Call>>>,
}

impl MockTrajectoryBuilder {
    /// Creates a fake with an empty call log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of the calls recorded so far.
    pub fn recorded_calls(&self) -> Vec<Call> {
        self.calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn record(&self, kind: CallKind, sensor_id: &str, time: Time) {
        self.calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((kind, sensor_id.to_string(), time));
    }
}

impl TrajectoryBuilderInterface for MockTrajectoryBuilder {
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) {
        self.record(
            CallKind::TimedPointCloud,
            sensor_id,
            timed_point_cloud_data.time,
        );
    }

    fn add_imu_data(&mut self, sensor_id: &str, imu_data: &ImuData) {
        self.record(CallKind::Imu, sensor_id, imu_data.time);
    }

    fn add_odometry_data(&mut self, sensor_id: &str, odometry_data: &OdometryData) {
        self.record(CallKind::Odometry, sensor_id, odometry_data.time);
    }

    fn add_fixed_frame_pose_data(
        &mut self,
        sensor_id: &str,
        fixed_frame_pose: &FixedFramePoseData,
    ) {
        self.record(CallKind::FixedFramePose, sensor_id, fixed_frame_pose.time);
    }

    fn add_landmark_data(&mut self, sensor_id: &str, landmark_data: &LandmarkData) {
        self.record(CallKind::Landmark, sensor_id, landmark_data.time);
    }

    fn add_local_slam_result_data(&mut self, local_slam_result_data: Box<dyn LocalSlamResultData>) {
        self.record(
            CallKind::LocalSlamResult,
            local_slam_result_data.get_sensor_id(),
            local_slam_result_data.get_time(),
        );
    }
}
