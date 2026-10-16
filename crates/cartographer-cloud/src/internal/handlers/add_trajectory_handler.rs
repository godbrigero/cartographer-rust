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
