//! Port of `cartographer/cloud/internal/handlers/add_rangefinder_data_handler` (.h, .cc).

use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::timed_point_cloud_data;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddRangefinderData` RPC.
#[derive(Debug)]
pub struct AddRangefinderDataHandler;

impl AddSensorDataHandler for AddRangefinderDataHandler {
    type Request = proto::AddRangefinderDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddRangefinderDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.timed_point_cloud_data.clone().unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                timed_point_cloud_data::from_proto(&payload),
            ),
        );
        // Range data is never forwarded to the uploader: it is uplinked as
        // part of the local SLAM results instead.
    }
}

#[cfg(test)]
mod tests {
    use cartographer::sensor::proto as sensor_proto;
    use cartographer::transform::proto as transform_proto;

    use crate::internal::testing::handler_test::HandlerTest;
    use crate::internal::testing::mock_map_builder_context::ContextCall;
    use crate::proto;

    /// C++ `kMessage`.
    fn message() -> proto::AddRangefinderDataRequest {
        proto::AddRangefinderDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 1,
                sensor_id: "sensor_id".to_string(),
                client_id: String::new(),
            }),
            timed_point_cloud_data: Some(sensor_proto::TimedPointCloudData {
                timestamp: 2,
                origin: Some(transform_proto::Vector3f {
                    x: 3.,
                    y: 4.,
                    z: 5.,
                }),
                point_data: vec![sensor_proto::TimedRangefinderPoint {
                    position: Some(transform_proto::Vector3f {
                        x: 6.,
                        y: 7.,
                        z: 8.,
                    }),
                    time: 9.,
                }],
                ..Default::default()
            }),
        }
    }

    mod add_rangefinder_data_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client
                        .add_rangefinder_data(tokio_stream::iter(requests))
                        .await
                },
            );
        }

        /// Range data is never forwarded to the uploader.
        #[test]
        fn never_forwarded_to_uploader() {
            let test = HandlerTest::new();
            let request = message();
            test.set_mock_local_trajectory_uploader();
            test.send_sensor_data_and_expect_enqueued(
                &request,
                |mut client, requests| async move {
                    client
                        .add_rangefinder_data(tokio_stream::iter(requests))
                        .await
                },
            );
            assert!(
                test.mock_local_trajectory_uploader
                    .enqueued_sensor_data()
                    .is_empty()
            );
            assert!(
                !test
                    .mock_map_builder_context
                    .calls()
                    .contains(&ContextCall::LocalTrajectoryUploader)
            );
        }
    }
}
