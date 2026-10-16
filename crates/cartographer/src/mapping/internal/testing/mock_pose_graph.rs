//! Port of `cartographer/mapping/internal/testing/mock_pose_graph` (.h).
//!
//! RUST PORT NOTE: the gmock mock of `PoseGraphInterface` becomes a
//! hand-written fake. Getters return empty/default values; every call is
//! recorded in [`MockPoseGraph::calls`]. Beyond the C++ mock, the fake also
//! implements the write API (`PoseGraph`) and [`AddNodePoseGraph`] (for
//! `Submap2D`) so that it can back a `GlobalTrajectoryBuilder` in tests; it
//! assigns consecutive node indices per trajectory.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::common::time::Time;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::internal::global_trajectory_builder::AddNodePoseGraph;
use crate::mapping::pose_graph::PoseGraph;
use crate::mapping::pose_graph_interface::{
    Constraint, GlobalSlamOptimizationCallback, LandmarkNode, PoseGraphInterface, SubmapData,
    SubmapPose, TrajectoryData, TrajectoryState,
};
use crate::mapping::pose_graph_trimmer::PoseGraphTrimmer;
use crate::mapping::proto;
use crate::mapping::submaps::SharedSubmap;
use crate::mapping::trajectory_node::{TrajectoryNode, TrajectoryNodeData, TrajectoryNodePose};
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::Rigid3d;

/// A recorded call of a data-carrying method.
#[derive(Clone, Debug, PartialEq)]
pub enum PoseGraphCall {
    /// `AddNode(constant_data, trajectory_id, insertion_submaps)`.
    AddNode {
        /// The trajectory.
        trajectory_id: i32,
        /// `constant_data->time`.
        time: Time,
        /// `insertion_submaps.size()`.
        num_insertion_submaps: usize,
    },
    /// `AddImuData`.
    AddImuData(i32, Time),
    /// `AddOdometryData`.
    AddOdometryData(i32, Time),
    /// `AddFixedFramePoseData`.
    AddFixedFramePoseData(i32, Time),
    /// `AddLandmarkData`.
    AddLandmarkData(i32, Time),
    /// Any other method, by C++ name.
    Other(&'static str),
}

/// Fake pose graph recording calls.
#[derive(Default)]
pub struct MockPoseGraph {
    /// The call log.
    pub calls: Mutex<Vec<PoseGraphCall>>,
    next_node_index: Mutex<BTreeMap<i32, i32>>,
}

impl std::fmt::Debug for MockPoseGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockPoseGraph")
            .field("calls", &self.recorded_calls())
            .finish_non_exhaustive()
    }
}

impl MockPoseGraph {
    /// Creates a fake with an empty call log, shared for use as `Arc<P>`.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns a copy of the calls recorded so far.
    pub fn recorded_calls(&self) -> Vec<PoseGraphCall> {
        self.log().clone()
    }

    fn log(&self) -> MutexGuard<'_, Vec<PoseGraphCall>> {
        self.calls.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn record(&self, call: PoseGraphCall) {
        self.log().push(call);
    }
}

impl PoseGraphInterface for MockPoseGraph {
    fn run_final_optimization(&self) {
        self.record(PoseGraphCall::Other("RunFinalOptimization"));
    }
    fn get_all_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        MapById::new()
    }
    fn get_submap_data(&self, _submap_id: &SubmapId) -> SubmapData {
        SubmapData::default()
    }
    fn get_all_submap_poses(&self) -> MapById<SubmapId, SubmapPose> {
        MapById::new()
    }
    fn get_local_to_global_transform(&self, _trajectory_id: i32) -> Rigid3d {
        Rigid3d::identity()
    }
    fn get_trajectory_nodes(&self) -> MapById<NodeId, TrajectoryNode> {
        MapById::new()
    }
    fn get_trajectory_node_poses(&self) -> MapById<NodeId, TrajectoryNodePose> {
        MapById::new()
    }
    fn get_trajectory_states(&self) -> BTreeMap<i32, TrajectoryState> {
        BTreeMap::new()
    }
    fn get_landmark_poses(&self) -> BTreeMap<String, Rigid3d> {
        BTreeMap::new()
    }
    fn set_landmark_pose(&self, _landmark_id: &str, _global_pose: &Rigid3d, _frozen: bool) {
        self.record(PoseGraphCall::Other("SetLandmarkPose"));
    }
    fn delete_trajectory(&self, _trajectory_id: i32) {
        self.record(PoseGraphCall::Other("DeleteTrajectory"));
    }
    fn is_trajectory_finished(&self, _trajectory_id: i32) -> bool {
        false
    }
    fn is_trajectory_frozen(&self, _trajectory_id: i32) -> bool {
        false
    }
    fn get_trajectory_data(&self) -> BTreeMap<i32, TrajectoryData> {
        BTreeMap::new()
    }
    fn constraints(&self) -> Vec<Constraint> {
        Vec::new()
    }
    fn to_proto(&self, _include_unfinished_submaps: bool) -> proto::PoseGraph {
        proto::PoseGraph::default()
    }
    fn set_global_slam_optimization_callback(&self, _callback: GlobalSlamOptimizationCallback) {
        self.record(PoseGraphCall::Other("SetGlobalSlamOptimizationCallback"));
    }
}

impl PoseGraph for MockPoseGraph {
    fn add_imu_data(&self, trajectory_id: i32, imu_data: &ImuData) {
        self.record(PoseGraphCall::AddImuData(trajectory_id, imu_data.time));
    }
    fn add_odometry_data(&self, trajectory_id: i32, odometry_data: &OdometryData) {
        self.record(PoseGraphCall::AddOdometryData(
            trajectory_id,
            odometry_data.time,
        ));
    }
    fn add_fixed_frame_pose_data(
        &self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    ) {
        self.record(PoseGraphCall::AddFixedFramePoseData(
            trajectory_id,
            fixed_frame_pose_data.time,
        ));
    }
    fn add_landmark_data(&self, trajectory_id: i32, landmark_data: &LandmarkData) {
        self.record(PoseGraphCall::AddLandmarkData(
            trajectory_id,
            landmark_data.time,
        ));
    }
    fn finish_trajectory(&self, _trajectory_id: i32) {
        self.record(PoseGraphCall::Other("FinishTrajectory"));
    }
    fn freeze_trajectory(&self, _trajectory_id: i32) {
        self.record(PoseGraphCall::Other("FreezeTrajectory"));
    }
    fn add_submap_from_proto(&self, _global_pose: &Rigid3d, _submap: &proto::Submap) {
        self.record(PoseGraphCall::Other("AddSubmapFromProto"));
    }
    fn add_node_from_proto(&self, _global_pose: &Rigid3d, _node: &proto::Node) {
        self.record(PoseGraphCall::Other("AddNodeFromProto"));
    }
    fn set_trajectory_data_from_proto(&self, _data: &proto::TrajectoryData) {
        self.record(PoseGraphCall::Other("SetTrajectoryDataFromProto"));
    }
    fn add_node_to_submap(&self, _node_id: &NodeId, _submap_id: &SubmapId) {
        self.record(PoseGraphCall::Other("AddNodeToSubmap"));
    }
    fn add_serialized_constraints(&self, _constraints: &[Constraint]) {
        self.record(PoseGraphCall::Other("AddSerializedConstraints"));
    }
    fn add_trimmer(&self, _trimmer: Box<dyn PoseGraphTrimmer>) {
        self.record(PoseGraphCall::Other("AddTrimmer"));
    }
    fn get_connected_trajectories(&self) -> Vec<Vec<i32>> {
        Vec::new()
    }
    fn get_imu_data(&self) -> MapByTime<ImuData> {
        MapByTime::default()
    }
    fn get_odometry_data(&self) -> MapByTime<OdometryData> {
        MapByTime::default()
    }
    fn get_fixed_frame_pose_data(&self) -> MapByTime<FixedFramePoseData> {
        MapByTime::default()
    }
    fn get_landmark_nodes(&self) -> BTreeMap<String, LandmarkNode> {
        BTreeMap::new()
    }
    fn set_initial_trajectory_pose(
        &self,
        _from_trajectory_id: i32,
        _to_trajectory_id: i32,
        _pose: &Rigid3d,
        _time: Time,
    ) {
        self.record(PoseGraphCall::Other("SetInitialTrajectoryPose"));
    }
}

impl AddNodePoseGraph for MockPoseGraph {
    type Submap = Submap2D;

    fn add_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Submap2D>],
    ) -> NodeId {
        self.record(PoseGraphCall::AddNode {
            trajectory_id,
            time: constant_data.time,
            num_insertion_submaps: insertion_submaps.len(),
        });
        let mut next_node_index = self
            .next_node_index
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let index = next_node_index.entry(trajectory_id).or_insert(0);
        let node_id = NodeId::new(trajectory_id, *index);
        *index += 1;
        node_id
    }
}
