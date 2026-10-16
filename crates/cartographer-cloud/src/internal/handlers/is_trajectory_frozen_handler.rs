//! Port of `cartographer/cloud/internal/handlers/is_trajectory_frozen_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `IsTrajectoryFrozen`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::IsTrajectoryFrozenRequest,
) -> Result<proto::IsTrajectoryFrozenResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    Ok(proto::IsTrajectoryFrozenResponse {
        is_frozen: pose_graph.is_trajectory_frozen(request.trajectory_id),
    })
}
