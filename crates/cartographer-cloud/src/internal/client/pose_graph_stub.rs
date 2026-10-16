//! Port of `cartographer/cloud/internal/client/pose_graph_stub` (.h, .cc).

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use cartographer::common::time::from_universal;
use cartographer::mapping::id::{MapById, NodeId, SubmapId};
use cartographer::mapping::pose_graph;
use cartographer::mapping::pose_graph_interface::{
    Constraint, GlobalSlamOptimizationCallback, PoseGraphInterface, SubmapData, SubmapPose,
    TrajectoryData, TrajectoryState,
};
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::proto::pose_graph::LandmarkPose;
use cartographer::mapping::trajectory_node::{
    ConstantPoseData, TrajectoryNode, TrajectoryNodePose,
};
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer::transform::transform::{to_proto, to_rigid3};
use tonic::transport::Channel;
use tonic::{Response, Status};

use crate::internal::async_grpc::RuntimeHolder;
use crate::internal::mapping::serialization::trajectory_state_from_proto;
use crate::proto;
use crate::proto::map_builder_service_client::MapBuilderServiceClient;

/// A `PoseGraphInterface` forwarding to a remote map builder server.
///
/// Methods panic when an RPC fails (C++ `CHECK(client.Write(request))`), and
/// the methods the C++ stub does not implement panic with "Not implemented".
/// Like all stub methods they block and must not be called from within an
/// async context.
pub struct PoseGraphStub {
    runtime: Arc<RuntimeHolder>,
    client: MapBuilderServiceClient<Channel>,
    client_id: String,
}

impl std::fmt::Debug for PoseGraphStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseGraphStub")
            .field("client_id", &self.client_id)
            .finish_non_exhaustive()
    }
}

impl PoseGraphStub {
    /// Creates the stub on an existing channel.
    pub fn new(
        runtime: Arc<RuntimeHolder>,
        client: MapBuilderServiceClient<Channel>,
        client_id: &str,
    ) -> Self {
        Self {
            runtime,
            client,
            client_id: client_id.to_string(),
        }
    }

    /// Runs one RPC on a clone of the client and returns its result.
    fn call<T, F, Fut>(&self, rpc: F) -> Result<T, Status>
    where
        F: FnOnce(MapBuilderServiceClient<Channel>) -> Fut,
        Fut: Future<Output = Result<Response<T>, Status>>,
    {
        self.runtime
            .block_on(rpc(self.client.clone()))
            .map(Response::into_inner)
    }

    /// Like [`Self::call`], panicking on failure (C++ `CHECK`).
    fn call_or_die<T, F, Fut>(&self, rpc_name: &str, rpc: F) -> T
    where
        F: FnOnce(MapBuilderServiceClient<Channel>) -> Fut,
        Fut: Future<Output = Result<Response<T>, Status>>,
    {
        self.call(rpc)
            .unwrap_or_else(|status| panic!("Check failed: {rpc_name} RPC: {status}"))
    }
}

impl PoseGraphInterface for PoseGraphStub {
    fn run_final_optimization(&self) {
        self.call_or_die("RunFinalOptimization", |mut client| async move {
            client.run_final_optimization(()).await
        });
    }

    fn get_all_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        panic!("Not implemented");
    }

    fn get_submap_data(&self, _submap_id: &SubmapId) -> SubmapData {
        panic!("Not implemented");
    }

    fn get_all_submap_poses(&self) -> MapById<SubmapId, SubmapPose> {
        let response = self.call_or_die("GetAllSubmapPoses", |mut client| async move {
            client.get_all_submap_poses(()).await
        });
        let mut submap_poses = MapById::new();
        for submap_pose in &response.submap_poses {
            let submap_id = submap_pose.submap_id.unwrap_or_default();
            submap_poses.insert(
                &SubmapId::new(submap_id.trajectory_id, submap_id.submap_index),
                SubmapPose {
                    version: submap_pose.submap_version,
                    pose: to_rigid3(&submap_pose.global_pose.unwrap_or_default()),
                },
            );
        }
        submap_poses
    }

    fn get_local_to_global_transform(&self, trajectory_id: i32) -> Rigid3d {
        let response = self.call_or_die("GetLocalToGlobalTransform", |mut client| async move {
            client
                .get_local_to_global_transform(proto::GetLocalToGlobalTransformRequest {
                    trajectory_id,
                })
                .await
        });
        to_rigid3(&response.local_to_global.unwrap_or_default())
    }

    fn get_trajectory_nodes(&self) -> MapById<NodeId, TrajectoryNode> {
        panic!("Not implemented");
    }

    fn get_trajectory_node_poses(&self) -> MapById<NodeId, TrajectoryNodePose> {
        let response = self.call_or_die("GetTrajectoryNodePoses", |mut client| async move {
            client.get_trajectory_node_poses(()).await
        });
        let mut node_poses = MapById::new();
        for node_pose in &response.node_poses {
            let constant_pose_data =
                node_pose
                    .constant_pose_data
                    .as_ref()
                    .map(|constant_pose_data| ConstantPoseData {
                        time: from_universal(constant_pose_data.timestamp),
                        local_pose: to_rigid3(&constant_pose_data.local_pose.unwrap_or_default()),
                    });
            let node_id = node_pose.node_id.unwrap_or_default();
            node_poses.insert(
                &NodeId::new(node_id.trajectory_id, node_id.node_index),
                TrajectoryNodePose::new(
                    to_rigid3(&node_pose.global_pose.unwrap_or_default()),
                    constant_pose_data,
                ),
            );
        }
        node_poses
    }

    fn get_trajectory_states(&self) -> BTreeMap<i32, TrajectoryState> {
        let response = self.call_or_die("GetTrajectoryStates", |mut client| async move {
            client.get_trajectory_states(()).await
        });
        response
            .trajectories_state
            .iter()
            .map(|(trajectory_id, state)| (*trajectory_id, trajectory_state_from_proto(*state)))
            .collect()
    }

    fn get_landmark_poses(&self) -> BTreeMap<String, Rigid3d> {
        let response = self.call_or_die("GetLandmarkPoses", |mut client| async move {
            client.get_landmark_poses(()).await
        });
        response
            .landmark_poses
            .iter()
            .map(|landmark_pose| {
                (
                    landmark_pose.landmark_id.clone(),
                    to_rigid3(&landmark_pose.global_pose.unwrap_or_default()),
                )
            })
            .collect()
    }

    /// `frozen` is not transmitted (like in C++).
    fn set_landmark_pose(&self, landmark_id: &str, global_pose: &Rigid3d, _frozen: bool) {
        let request = proto::SetLandmarkPoseRequest {
            landmark_pose: Some(LandmarkPose {
                landmark_id: landmark_id.to_string(),
                global_pose: Some(to_proto(global_pose)),
            }),
        };
        self.call_or_die("SetLandmarkPose", |mut client| async move {
            client.set_landmark_pose(request).await
        });
    }

    fn delete_trajectory(&self, trajectory_id: i32) {
        let request = proto::DeleteTrajectoryRequest {
            trajectory_id,
            client_id: self.client_id.clone(),
        };
        if let Err(status) =
            self.call(|mut client| async move { client.delete_trajectory(request).await })
        {
            log::error!(
                "Failed to delete trajectory {trajectory_id} for client_id {}: {}",
                self.client_id,
                status.message()
            );
        }
    }

    fn is_trajectory_finished(&self, trajectory_id: i32) -> bool {
        self.call(|mut client| async move {
            client
                .is_trajectory_finished(proto::IsTrajectoryFinishedRequest { trajectory_id })
                .await
        })
        .unwrap_or_else(|status| {
            panic!(
                "Failed to check if trajectory {trajectory_id} is finished: {}",
                status.message()
            )
        })
        .is_finished
    }

    fn is_trajectory_frozen(&self, trajectory_id: i32) -> bool {
        self.call(|mut client| async move {
            client
                .is_trajectory_frozen(proto::IsTrajectoryFrozenRequest { trajectory_id })
                .await
        })
        .unwrap_or_else(|status| {
            panic!(
                "Failed to check if trajectory {trajectory_id} is frozen: {}",
                status.message()
            )
        })
        .is_frozen
    }

    fn get_trajectory_data(&self) -> BTreeMap<i32, TrajectoryData> {
        panic!("Not implemented");
    }

    fn constraints(&self) -> Vec<Constraint> {
        let response = self
            .call(|mut client| async move { client.get_constraints(()).await })
            .unwrap_or_else(|status| panic!("Failed to get constraints: {}", status.message()));
        pose_graph::from_proto(&response.constraints)
    }

    fn to_proto(&self, _include_unfinished_submaps: bool) -> mapping_proto::PoseGraph {
        panic!("Not implemented");
    }

    fn set_global_slam_optimization_callback(&self, _callback: GlobalSlamOptimizationCallback) {
        panic!("Not implemented");
    }
}
