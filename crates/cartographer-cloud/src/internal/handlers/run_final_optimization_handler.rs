//! Port of `cartographer/cloud/internal/handlers/run_final_optimization_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;

/// Handles `RunFinalOptimization`.
pub fn on_request(context: &dyn MapBuilderContextInterface) -> Result<(), Status> {
    let pose_graph = context.map_builder().pose_graph();
    pose_graph.run_final_optimization();
    Ok(())
}
