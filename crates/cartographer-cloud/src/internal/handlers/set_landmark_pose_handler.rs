//! Port of `cartographer/cloud/internal/handlers/set_landmark_pose_handler` (.h, .cc).

use cartographer::transform::transform::to_rigid3;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `SetLandmarkPose` (`frozen` keeps its default `false`).
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::SetLandmarkPoseRequest,
) -> Result<(), Status> {
    let landmark_pose = request.landmark_pose.clone().unwrap_or_default();
    let pose_graph = context.map_builder().pose_graph();
    pose_graph.set_landmark_pose(
        &landmark_pose.landmark_id,
        &to_rigid3(&landmark_pose.global_pose.unwrap_or_default()),
        false,
    );
    Ok(())
}
