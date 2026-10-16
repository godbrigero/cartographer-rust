//! Port of `cartographer/cloud/internal/handlers/add_sensor_data_batch_handler` (.h, .cc).

use std::collections::BTreeMap;
use std::sync::Mutex;

use cartographer::metrics::counter::Counter;
use cartographer::metrics::family_factory::{Family, FamilyFactory, Labels, NullMetric};
use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::{
    fixed_frame_pose_data, imu_data, landmark_data, odometry_data, timed_point_cloud_data,
};
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::map_builder_server::lock;
use crate::proto;
use crate::proto::sensor_data::SensorData as SensorDataCase;

static COUNTER_METRICS_FAMILY: Mutex<Option<&'static dyn Family<dyn Counter>>> = Mutex::new(None);

fn counter_metrics_family() -> &'static dyn Family<dyn Counter> {
    lock(&COUNTER_METRICS_FAMILY).unwrap_or_else(<dyn Counter as NullMetric>::null_family)
}

/// Per client and trajectory sensor data counters
/// (`cartographer_sensor_data_total{client_id, trajectory_id, sensor}`).
struct ClientMetrics {
    odometry_sensor_counter: &'static dyn Counter,
    imu_sensor_counter: &'static dyn Counter,
    timed_point_cloud_counter: &'static dyn Counter,
    fixed_frame_pose_counter: &'static dyn Counter,
    landmark_counter: &'static dyn Counter,
    local_slam_result_counter: &'static dyn Counter,
}

fn create_client_metrics(client_id: &str, trajectory_id: i32) -> ClientMetrics {
    let family = counter_metrics_family();
    let counter = |sensor: &str| {
        let labels: Labels = [
            ("client_id".to_string(), client_id.to_string()),
            ("trajectory_id".to_string(), trajectory_id.to_string()),
            ("sensor".to_string(), sensor.to_string()),
        ]
        .into_iter()
        .collect();
        family.add(&labels)
    };
    ClientMetrics {
        odometry_sensor_counter: counter("odometry"),
        imu_sensor_counter: counter("imu"),
        fixed_frame_pose_counter: counter("fixed_frame_pose"),
        landmark_counter: counter("landmark"),
        local_slam_result_counter: counter("local_slam_result"),
        timed_point_cloud_counter: counter("timed_point_cloud"),
    }
}

/// Registers the `cartographer_sensor_data_total` counter family (C++
/// `AddSensorDataBatchHandler::RegisterMetrics`).
pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
    *lock(&COUNTER_METRICS_FAMILY) = Some(
        family_factory.new_counter_family("cartographer_sensor_data_total", "Sensor data received"),
    );
}

/// Handles `AddSensorDataBatch`: enqueues every sensor data message (local
/// SLAM results are reassembled into submaps on the SLAM thread). Stops with
/// `NOT_FOUND` at the first message of an unknown trajectory.
///
/// RUST PORT NOTE: a message without payload is answered with
/// `INVALID_ARGUMENT` instead of the C++ `LOG(FATAL)`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::AddSensorDataBatchRequest,
) -> Result<(), Status> {
    // The C++ keeps the metrics per handler instance, i.e. per RPC.
    let mut client_metric_map: BTreeMap<String, ClientMetrics> = BTreeMap::new();
    for sensor_data in &request.sensor_data {
        let metadata = sensor_data.sensor_metadata.clone().unwrap_or_default();
        if !context.check_client_id_for_trajectory(&metadata.client_id, metadata.trajectory_id) {
            log::error!(
                "Unknown trajectory with ID {} and client_id {}",
                metadata.trajectory_id,
                metadata.client_id
            );
            return Err(Status::not_found("Unknown trajectory"));
        }
        let metrics = client_metric_map
            .entry(format!("{}{}", metadata.client_id, metadata.trajectory_id))
            .or_insert_with(|| create_client_metrics(&metadata.client_id, metadata.trajectory_id));
        let trajectory_id = metadata.trajectory_id;
        let sensor_id = &metadata.sensor_id;
        match &sensor_data.sensor_data {
            Some(SensorDataCase::OdometryData(data)) => {
                context.enqueue_sensor_data(
                    trajectory_id,
                    make_dispatchable(sensor_id, odometry_data::from_proto(data)),
                );
                metrics.odometry_sensor_counter.increment();
            }
            Some(SensorDataCase::ImuData(data)) => {
                context.enqueue_sensor_data(
                    trajectory_id,
                    make_dispatchable(sensor_id, imu_data::from_proto(data)),
                );
                metrics.imu_sensor_counter.increment();
            }
            Some(SensorDataCase::TimedPointCloudData(data)) => {
                context.enqueue_sensor_data(
                    trajectory_id,
                    make_dispatchable(sensor_id, timed_point_cloud_data::from_proto(data)),
                );
                metrics.timed_point_cloud_counter.increment();
            }
            Some(SensorDataCase::FixedFramePoseData(data)) => {
                context.enqueue_sensor_data(
                    trajectory_id,
                    make_dispatchable(sensor_id, fixed_frame_pose_data::from_proto(data)),
                );
                metrics.fixed_frame_pose_counter.increment();
            }
            Some(SensorDataCase::LandmarkData(data)) => {
                context.enqueue_sensor_data(
                    trajectory_id,
                    make_dispatchable(sensor_id, landmark_data::from_proto(data)),
                );
                metrics.landmark_counter.increment();
            }
            Some(SensorDataCase::LocalSlamResultData(data)) => {
                context.enqueue_local_slam_result_data(trajectory_id, sensor_id, data);
                metrics.local_slam_result_counter.increment();
            }
            None => {
                log::error!("Unknown sensor data type: 0");
                return Err(Status::invalid_argument("Unknown sensor data type"));
            }
        }
    }
    Ok(())
}
