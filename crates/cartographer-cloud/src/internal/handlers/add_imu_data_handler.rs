//! Port of `cartographer/cloud/internal/handlers/add_imu_data_handler` (.h, .cc).

use cartographer::sensor::imu_data;
use cartographer::sensor::internal::dispatchable::make_dispatchable;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddImuData` RPC.
#[derive(Debug)]
pub struct AddImuDataHandler;

impl AddSensorDataHandler for AddImuDataHandler {
    type Request = proto::AddImuDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddImuDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.imu_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(&sensor_metadata.sensor_id, imu_data::from_proto(&payload)),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::ImuData(payload)),
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use cartographer::sensor::proto as sensor_proto;
    use cartographer::transform::proto as transform_proto;

    use crate::internal::testing::handler_test::HandlerTest;
    use crate::proto;

    /// C++ `kMessage`.
    fn message() -> proto::AddImuDataRequest {
        proto::AddImuDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "sensor_id".to_string(),
                client_id: String::new(),
            }),
            imu_data: Some(sensor_proto::ImuData {
                timestamp: 2,
                linear_acceleration: Some(transform_proto::Vector3d {
                    x: 3.,
                    y: 4.,
                    z: 5.,
                }),
                angular_velocity: Some(transform_proto::Vector3d {
                    x: 6.,
                    y: 7.,
                    z: 8.,
                }),
            }),
        }
    }

    mod add_imu_data_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_no_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client.add_imu_data(tokio_stream::iter(requests)).await
                },
            );
            assert!(
                test.mock_local_trajectory_uploader
                    .enqueued_sensor_data()
                    .is_empty()
            );
        }

        #[test]
        fn with_mock_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_mock_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client.add_imu_data(tokio_stream::iter(requests)).await
                },
            );
            let sensor_data = proto::SensorData {
                sensor_metadata: request.sensor_metadata,
                sensor_data: Some(proto::sensor_data::SensorData::ImuData(
                    request.imu_data.unwrap(),
                )),
            };
            assert_eq!(
                test.mock_local_trajectory_uploader.enqueued_sensor_data(),
                vec![sensor_data]
            );
        }
    }
}
