//! Port of `cartographer/cloud/internal/handlers/add_landmark_data_handler` (.h, .cc).

use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::landmark_data;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddLandmarkData` RPC.
#[derive(Debug)]
pub struct AddLandmarkDataHandler;

impl AddSensorDataHandler for AddLandmarkDataHandler {
    type Request = proto::AddLandmarkDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddLandmarkDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.landmark_data.clone().unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                landmark_data::from_proto(&payload),
            ),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::LandmarkData(payload)),
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
    fn message() -> proto::AddLandmarkDataRequest {
        proto::AddLandmarkDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "sensor_id".to_string(),
                client_id: String::new(),
            }),
            landmark_data: Some(sensor_proto::LandmarkData {
                timestamp: 2,
                landmark_observations: vec![sensor_proto::landmark_data::LandmarkObservation {
                    id: b"3".to_vec(),
                    landmark_to_tracking_transform: Some(transform_proto::Rigid3d {
                        translation: Some(transform_proto::Vector3d {
                            x: 4.,
                            y: 5.,
                            z: 6.,
                        }),
                        rotation: Some(transform_proto::Quaterniond {
                            w: 7.,
                            x: 8.,
                            y: 9.,
                            z: 10.,
                        }),
                    }),
                    translation_weight: 11.,
                    rotation_weight: 12.,
                }],
            }),
        }
    }

    mod add_landmark_data_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_no_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client.add_landmark_data(tokio_stream::iter(requests)).await
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
                    client.add_landmark_data(tokio_stream::iter(requests)).await
                },
            );
            let sensor_data = proto::SensorData {
                sensor_metadata: request.sensor_metadata.clone(),
                sensor_data: Some(proto::sensor_data::SensorData::LandmarkData(
                    request.landmark_data.clone().unwrap(),
                )),
            };
            assert_eq!(
                test.mock_local_trajectory_uploader.enqueued_sensor_data(),
                vec![sensor_data]
            );
        }
    }
}
