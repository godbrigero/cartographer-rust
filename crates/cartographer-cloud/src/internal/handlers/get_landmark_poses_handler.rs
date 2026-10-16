//! Port of `cartographer/cloud/internal/handlers/get_landmark_poses_handler` (.h, .cc).

use cartographer::mapping::proto::pose_graph::LandmarkPose;
use cartographer::transform::transform::to_proto;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetLandmarkPoses`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetLandmarkPosesResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let landmark_poses = pose_graph.get_landmark_poses();
    Ok(proto::GetLandmarkPosesResponse {
        landmark_poses: landmark_poses
            .iter()
            .map(|(landmark_id, global_pose)| LandmarkPose {
                landmark_id: landmark_id.clone(),
                global_pose: Some(to_proto(global_pose)),
            })
            .collect(),
    })
}
