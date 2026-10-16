//! Port of `cartographer/cloud/internal/handlers/add_fixed_frame_pose_data_handler` (.h, .cc).

use cartographer::sensor::fixed_frame_pose_data;
use cartographer::sensor::internal::dispatchable::make_dispatchable;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddFixedFramePoseData` RPC.
#[derive(Debug)]
pub struct AddFixedFramePoseDataHandler;

impl AddSensorDataHandler for AddFixedFramePoseDataHandler {
    type Request = proto::AddFixedFramePoseDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddFixedFramePoseDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.fixed_frame_pose_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                fixed_frame_pose_data::from_proto(&payload),
            ),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::FixedFramePoseData(payload)),
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
    fn message() -> proto::AddFixedFramePoseDataRequest {
        proto::AddFixedFramePoseDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "sensor_id".to_string(),
                client_id: String::new(),
            }),
            fixed_frame_pose_data: Some(sensor_proto::FixedFramePoseData {
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

    mod add_fixed_frame_pose_data_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_no_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client
                        .add_fixed_frame_pose_data(tokio_stream::iter(requests))
                        .await
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
                    client
                        .add_fixed_frame_pose_data(tokio_stream::iter(requests))
                        .await
                },
            );
            let sensor_data = proto::SensorData {
                sensor_metadata: request.sensor_metadata.clone(),
                sensor_data: Some(proto::sensor_data::SensorData::FixedFramePoseData(
                    request.fixed_frame_pose_data.unwrap(),
                )),
            };
            assert_eq!(
                test.mock_local_trajectory_uploader.enqueued_sensor_data(),
                vec![sensor_data]
            );
        }
    }
}
