//! Port of `cartographer/cloud/internal/handlers/delete_trajectory_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `DeleteTrajectory` (asynchronous deletion in the pose graph).
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::DeleteTrajectoryRequest,
) -> Result<(), Status> {
    if !context.check_client_id_for_trajectory(&request.client_id, request.trajectory_id) {
        log::error!(
            "Unknown trajectory with ID {} and client_id {}",
            request.trajectory_id,
            request.client_id
        );
        return Err(Status::not_found("Unknown trajectory"));
    }
    let pose_graph = context.map_builder().pose_graph();
    pose_graph.delete_trajectory(request.trajectory_id);
    // TODO(gaschler): Think if LocalSlamUploader needs to be notified.
    Ok(())
}
