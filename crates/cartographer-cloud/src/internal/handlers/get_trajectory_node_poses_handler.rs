//! Port of `cartographer/cloud/internal/handlers/get_trajectory_node_poses_handler` (.h, .cc).

use cartographer::common::time::to_universal;
use cartographer::transform::transform::to_proto;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;
use crate::proto::trajectory_node_pose::ConstantPoseData;

/// Handles `GetTrajectoryNodePoses`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetTrajectoryNodePosesResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let node_poses = pose_graph.get_trajectory_node_poses();
    Ok(proto::GetTrajectoryNodePosesResponse {
        node_poses: node_poses
            .iter()
            .map(|node_id_pose| proto::TrajectoryNodePose {
                node_id: Some(node_id_pose.id.to_proto()),
                global_pose: Some(to_proto(&node_id_pose.data.global_pose)),
                constant_pose_data: node_id_pose.data.constant_pose_data.as_ref().map(
                    |constant_pose_data| ConstantPoseData {
                        timestamp: to_universal(constant_pose_data.time),
                        local_pose: Some(to_proto(&constant_pose_data.local_pose)),
                    },
                ),
            })
            .collect(),
    })
}
