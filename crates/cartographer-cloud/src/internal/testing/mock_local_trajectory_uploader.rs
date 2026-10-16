//! Port of `cartographer/cloud/internal/testing/mock_local_trajectory_uploader` (.h).
//!
//! RUST PORT NOTE: the gmock mock becomes a hand-written fake recording all
//! calls.

use std::collections::BTreeSet;
use std::sync::Mutex;

use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::{SensorId, SensorType};
use tonic::Status;

use crate::internal::local_trajectory_uploader::LocalTrajectoryUploaderInterface;
use crate::internal::map_builder_server::lock;
use crate::proto;

/// A recorded `AddTrajectory` call.
#[derive(Clone, Debug, PartialEq)]
pub struct AddTrajectoryCall {
    /// The client id.
    pub client_id: String,
    /// The local trajectory id.
    pub local_trajectory_id: i32,
    /// The expected sensors.
    pub expected_sensor_ids: BTreeSet<SensorId>,
    /// The options sent to the uplink.
    pub trajectory_options: mapping_proto::TrajectoryBuilderOptions,
}

/// Fake `LocalTrajectoryUploaderInterface`.
#[derive(Debug, Default)]
pub struct MockLocalTrajectoryUploader {
    /// `EnqueueSensorData` arguments (C++ `DoEnqueueSensorData`).
    pub enqueued_sensor_data: Mutex<Vec<proto::SensorData>>,
    /// `AddTrajectory` arguments.
    pub add_trajectory_calls: Mutex<Vec<AddTrajectoryCall>>,
    /// `FinishTrajectory` arguments.
    pub finish_trajectory_calls: Mutex<Vec<(String, i32)>>,
    /// Number of `Start`/`Shutdown` calls.
    pub start_and_shutdown_calls: Mutex<(usize, usize)>,
}

impl MockLocalTrajectoryUploader {
    /// Creates the fake.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy of the enqueued sensor data.
    pub fn enqueued_sensor_data(&self) -> Vec<proto::SensorData> {
        lock(&self.enqueued_sensor_data).clone()
    }

    /// Copy of the recorded `AddTrajectory` calls.
    pub fn add_trajectory_calls(&self) -> Vec<AddTrajectoryCall> {
        lock(&self.add_trajectory_calls).clone()
    }

    /// Copy of the recorded `FinishTrajectory` calls.
    pub fn finish_trajectory_calls(&self) -> Vec<(String, i32)> {
        lock(&self.finish_trajectory_calls).clone()
    }
}

impl LocalTrajectoryUploaderInterface for MockLocalTrajectoryUploader {
    fn start(&self) {
        lock(&self.start_and_shutdown_calls).0 += 1;
    }

    fn shutdown(&self) {
        lock(&self.start_and_shutdown_calls).1 += 1;
    }

    fn enqueue_sensor_data(&self, sensor_data: Box<proto::SensorData>) {
        lock(&self.enqueued_sensor_data).push(*sensor_data);
    }

    fn add_trajectory(
        &self,
        client_id: &str,
        local_trajectory_id: i32,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &mapping_proto::TrajectoryBuilderOptions,
    ) -> Result<(), Status> {
        lock(&self.add_trajectory_calls).push(AddTrajectoryCall {
            client_id: client_id.to_string(),
            local_trajectory_id,
            expected_sensor_ids: expected_sensor_ids.clone(),
            trajectory_options: trajectory_options.clone(),
        });
        Ok(())
    }

    fn finish_trajectory(&self, client_id: &str, local_trajectory_id: i32) -> Result<(), Status> {
        lock(&self.finish_trajectory_calls).push((client_id.to_string(), local_trajectory_id));
        Ok(())
    }

    fn get_local_slam_result_sensor_id(&self, local_trajectory_id: i32) -> SensorId {
        SensorId {
            r#type: SensorType::LocalSlamResult,
            id: format!("local_slam_result_{local_trajectory_id}"),
        }
    }
}
