//! Port of `cartographer/cloud/internal/handlers/add_trajectory_handler` (.h, .cc).

use std::collections::BTreeSet;

use cartographer::mapping::trajectory_builder_interface::SensorId;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::sensor::serialization::from_proto;
use crate::proto;

/// Handles `AddTrajectory`: creates the trajectory builder, registers the
/// client id and, if an uploader is configured, creates the trajectory in the
/// uplink as well (errors there are only logged).
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::AddTrajectoryRequest,
) -> Result<proto::AddTrajectoryResponse, Status> {
    let local_slam_result_callback = context.get_local_slam_result_callback_for_subscriptions();
    let expected_sensor_ids: BTreeSet<SensorId> =
        request.expected_sensor_ids.iter().map(from_proto).collect();
    let trajectory_builder_options = request
        .trajectory_builder_options
        .clone()
        .unwrap_or_default();
    let trajectory_id = context.map_builder().add_trajectory_builder(
        &expected_sensor_ids,
        &trajectory_builder_options,
        local_slam_result_callback,
    );
    context.register_client_id_for_trajectory(&request.client_id, trajectory_id);
    if let Some(uploader) = context.local_trajectory_uploader() {
        let mut trajectory_builder_options = trajectory_builder_options;
        // Clear the trajectory builder options to convey to the cloud
        // Cartographer instance that does not need to instantiate a
        // 'LocalTrajectoryBuilder'.
        trajectory_builder_options.trajectory_builder_2d_options = None;
        trajectory_builder_options.trajectory_builder_3d_options = None;
        // Don't instantiate the 'PureLocalizationTrimmer' on the server and
        // don't freeze the trajectory on the server.
        trajectory_builder_options.pure_localization_trimmer = None;
        // Ignore initial poses in trajectory_builder_options.
        trajectory_builder_options.initial_trajectory_pose = None;
        if let Err(status) = uploader.add_trajectory(
            &request.client_id,
            trajectory_id,
            &expected_sensor_ids,
            &trajectory_builder_options,
        ) {
            log::error!(
                "Failed to create trajectory_id {trajectory_id} in uplink. Error: {}",
                status.message()
            );
        }
    }
    Ok(proto::AddTrajectoryResponse { trajectory_id })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use cartographer::mapping::proto as mapping_proto;
    use cartographer::mapping::trajectory_builder_interface::SensorId;
    use cartographer::transform::proto as transform_proto;

    use crate::internal::map_builder_server::lock;
    use crate::internal::sensor::serialization::from_proto;
    use crate::internal::testing::fake_map_builder::MapBuilderCall;
    use crate::internal::testing::handler_test::HandlerTest;
    use crate::internal::testing::mock_local_trajectory_uploader::AddTrajectoryCall;
    use crate::internal::testing::mock_map_builder_context::ContextCall;
    use crate::proto;

    /// C++ `kMessage`.
    fn message() -> proto::AddTrajectoryRequest {
        proto::AddTrajectoryRequest {
            client_id: "CLIENT_ID".to_string(),
            expected_sensor_ids: vec![
                proto::SensorId {
                    id: "range_sensor".to_string(),
                    r#type: proto::SensorType::Range as i32,
                },
                proto::SensorId {
                    id: "imu_sensor".to_string(),
                    r#type: proto::SensorType::Imu as i32,
                },
            ],
            trajectory_builder_options: Some(mapping_proto::TrajectoryBuilderOptions {
                trajectory_builder_2d_options: Some(
                    mapping_proto::LocalTrajectoryBuilderOptions2D {
                        min_range: 20.,
                        max_range: 30.,
                        ..Default::default()
                    },
                ),
                pure_localization_trimmer: Some(
                    mapping_proto::trajectory_builder_options::PureLocalizationTrimmerOptions {
                        max_submaps_to_keep: 3,
                    },
                ),
                initial_trajectory_pose: Some(mapping_proto::InitialTrajectoryPose {
                    relative_pose: Some(transform_proto::Rigid3d {
                        translation: Some(transform_proto::Vector3d {
                            x: 1.,
                            y: 2.,
                            z: 3.,
                        }),
                        rotation: Some(transform_proto::Quaterniond {
                            w: 4.,
                            x: 5.,
                            y: 6.,
                            z: 7.,
                        }),
                    }),
                    to_trajectory_id: 8,
                    timestamp: 9,
                }),
                ..Default::default()
            }),
        }
    }

    fn parse_sensor_ids(request: &proto::AddTrajectoryRequest) -> BTreeSet<SensorId> {
        request.expected_sensor_ids.iter().map(from_proto).collect()
    }

    /// C++ fixture `SetUp`: the mock map builder returns trajectory id 13.
    fn set_up() -> HandlerTest {
        let test = HandlerTest::new();
        lock(&test.mock_map_builder).next_trajectory_id = 13;
        test
    }

    fn send(test: &HandlerTest, request: &proto::AddTrajectoryRequest) -> i32 {
        let request = request.clone();
        test.call(|mut client| async move { client.add_trajectory(request).await })
            .unwrap()
            .into_inner()
            .trajectory_id
    }

    fn expect_add_trajectory_builder(test: &HandlerTest, request: &proto::AddTrajectoryRequest) {
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::AddTrajectoryBuilder {
                expected_sensor_ids: parse_sensor_ids(request),
                trajectory_options: Box::new(request.trajectory_builder_options.clone().unwrap()),
                has_local_slam_result_callback: false,
            }]
        );
        let calls = test.mock_map_builder_context.calls();
        assert!(calls.contains(&ContextCall::GetLocalSlamResultCallbackForSubscriptions));
        assert!(calls.contains(&ContextCall::RegisterClientIdForTrajectory(
            "CLIENT_ID".to_string(),
            13
        )));
    }

    mod add_trajectory_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = set_up();
            test.set_no_local_trajectory_uploader();
            let request = message();
            assert_eq!(send(&test, &request), 13);
            expect_add_trajectory_builder(&test, &request);
            assert!(
                test.mock_local_trajectory_uploader
                    .add_trajectory_calls()
                    .is_empty()
            );
        }

        #[test]
        fn with_local_slam_uploader() {
            let test = set_up();
            test.set_mock_local_trajectory_uploader();
            let request = message();
            assert_eq!(send(&test, &request), 13);
            expect_add_trajectory_builder(&test, &request);
            // Reproduce the changes to the request as done by the handler.
            let mut upstream_trajectory_builder_options =
                request.trajectory_builder_options.clone().unwrap();
            upstream_trajectory_builder_options.trajectory_builder_2d_options = None;
            upstream_trajectory_builder_options.trajectory_builder_3d_options = None;
            upstream_trajectory_builder_options.pure_localization_trimmer = None;
            upstream_trajectory_builder_options.initial_trajectory_pose = None;
            assert_eq!(
                test.mock_local_trajectory_uploader.add_trajectory_calls(),
                vec![AddTrajectoryCall {
                    client_id: "CLIENT_ID".to_string(),
                    local_trajectory_id: 13,
                    expected_sensor_ids: parse_sensor_ids(&request),
                    trajectory_options: upstream_trajectory_builder_options,
                }]
            );
        }
    }
}
