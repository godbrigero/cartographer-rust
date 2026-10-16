//! Port of `cartographer/cloud/internal/handlers/get_all_submap_poses` (.h, .cc)
//! (`GetAllSubmapPosesHandler`).

use cartographer::transform::transform::to_proto;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetAllSubmapPoses`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetAllSubmapPosesResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let submap_poses = pose_graph.get_all_submap_poses();
    Ok(proto::GetAllSubmapPosesResponse {
        submap_poses: submap_poses
            .iter()
            .map(|submap_id_pose| proto::SubmapPose {
                submap_id: Some(submap_id_pose.id.to_proto()),
                submap_version: submap_id_pose.data.version,
                global_pose: Some(to_proto(&submap_id_pose.data.pose)),
            })
            .collect(),
    })
}
