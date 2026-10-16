//! Port of `cartographer/cloud/internal/handlers/get_constraints_handler` (.h, .cc).

use cartographer::mapping::pose_graph;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetConstraints`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetConstraintsResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let constraints = pose_graph.constraints();
    Ok(proto::GetConstraintsResponse {
        constraints: constraints.iter().map(pose_graph::to_proto).collect(),
    })
}
