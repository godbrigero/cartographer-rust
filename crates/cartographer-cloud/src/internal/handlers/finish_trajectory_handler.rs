//! Port of `cartographer/cloud/internal/handlers/finish_trajectory_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `FinishTrajectory`: finishes the trajectory, ends its local SLAM
/// subscriptions and finishes it in the uplink (errors there are logged).
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::FinishTrajectoryRequest,
) -> Result<(), Status> {
    if !context.check_client_id_for_trajectory(&request.client_id, request.trajectory_id) {
        log::error!(
            "Unknown trajectory with ID {} and client_id {}",
            request.trajectory_id,
            request.client_id
        );
        return Err(Status::not_found("Unknown trajectory"));
    }
    context
        .map_builder()
        .finish_trajectory(request.trajectory_id);
    context.notify_finish_trajectory(request.trajectory_id);
    if let Some(uploader) = context.local_trajectory_uploader()
        && let Err(status) = uploader.finish_trajectory(&request.client_id, request.trajectory_id)
    {
        log::error!(
            "Failed to finish trajectory in uplink: {}",
            status.message()
        );
    }
    Ok(())
}
