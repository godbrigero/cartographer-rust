//! Port of `cartographer/cloud/internal/handlers/add_odometry_data_handler` (.h, .cc).

use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::odometry_data;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddOdometryData` RPC.
#[derive(Debug)]
pub struct AddOdometryDataHandler;

impl AddSensorDataHandler for AddOdometryDataHandler {
    type Request = proto::AddOdometryDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddOdometryDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.odometry_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                odometry_data::from_proto(&payload),
            ),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::OdometryData(payload)),
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
    fn message() -> proto::AddOdometryDataRequest {
        proto::AddOdometryDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "sensor_id".to_string(),
                client_id: String::new(),
            }),
            odometry_data: Some(sensor_proto::OdometryData {
                timestamp: 2,
                pose: Some(transform_proto::Rigid3d {
                    translation: Some(transform_proto::Vector3d {
                        x: 3.,
                        y: 4.,
                        z: 5.,
                    }),
                    rotation: Some(transform_proto::Quaterniond {
                        w: 6.,
                        x: 7.,
                        y: 8.,
                        z: 9.,
                    }),
                }),
            }),
        }
    }

    mod add_odometry_data_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_no_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client.add_odometry_data(tokio_stream::iter(requests)).await
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
                    client.add_odometry_data(tokio_stream::iter(requests)).await
                },
            );
            let sensor_data = proto::SensorData {
                sensor_metadata: request.sensor_metadata.clone(),
                sensor_data: Some(proto::sensor_data::SensorData::OdometryData(
                    request.odometry_data.unwrap(),
                )),
            };
            assert_eq!(
                test.mock_local_trajectory_uploader.enqueued_sensor_data(),
                vec![sensor_data]
            );
        }
    }
}
