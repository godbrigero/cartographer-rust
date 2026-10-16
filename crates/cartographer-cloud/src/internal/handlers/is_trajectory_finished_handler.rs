//! Port of `cartographer/cloud/internal/handlers/is_trajectory_finished_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `IsTrajectoryFinished`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::IsTrajectoryFinishedRequest,
) -> Result<proto::IsTrajectoryFinishedResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    Ok(proto::IsTrajectoryFinishedResponse {
        is_finished: pose_graph.is_trajectory_finished(request.trajectory_id),
    })
}
