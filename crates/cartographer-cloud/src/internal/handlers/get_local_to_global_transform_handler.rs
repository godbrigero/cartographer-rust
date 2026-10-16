//! Port of `cartographer/cloud/internal/handlers/get_local_to_global_transform_handler` (.h, .cc).

use cartographer::transform::transform::to_proto;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetLocalToGlobalTransform`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::GetLocalToGlobalTransformRequest,
) -> Result<proto::GetLocalToGlobalTransformResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let local_to_global = pose_graph.get_local_to_global_transform(request.trajectory_id);
    Ok(proto::GetLocalToGlobalTransformResponse {
        local_to_global: Some(to_proto(&local_to_global)),
    })
}
