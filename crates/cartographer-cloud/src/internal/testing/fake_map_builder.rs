//! Scriptable fakes of `MapBuilderInterface` and `PoseGraphInterface` for the
//! handler and client/server tests (the C++ tests use gmock's
//! `mapping::testing::MockMapBuilder` / `MockPoseGraph` with `EXPECT_CALL`s
//! and canned return values).
//!
//! RUST PORT NOTE: new file. The core crate's `MockMapBuilder` /
//! `MockPoseGraph` cannot return canned values (e.g. trajectory id 13 or
//! landmark poses) and their logs are not reachable once the map builder is
//! moved into a server, so these fakes keep all state in an
//! `Arc<Mutex<..>>` shared with the test. Beyond the C++ mocks they keep
//! simple trajectory states (added = ACTIVE, finished = FINISHED, deleted =
//! DELETED) and store the local SLAM / global SLAM callbacks so tests can
//! trigger them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use cartographer::io::proto_stream_interface::{
    ProtoStreamReaderInterface, ProtoStreamWriterInterface,
};
use cartographer::mapping::id::{MapById, NodeId, SubmapId};
use cartographer::mapping::internal::testing::mock_trajectory_builder::MockTrajectoryBuilder;
use cartographer::mapping::map_builder_interface::{
    LocalSlamResultCallback, MapBuilderInterface, SensorId,
};
use cartographer::mapping::pose_graph_interface::{
    Constraint, GlobalSlamOptimizationCallback, PoseGraphInterface, SubmapData, SubmapPose,
    TrajectoryData, TrajectoryState,
};
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use cartographer::mapping::trajectory_node::{TrajectoryNode, TrajectoryNodePose};
use cartographer::transform::rigid_transform::Rigid3d;

use crate::internal::map_builder_server::lock;

/// A recorded map builder call.
#[derive(Clone, Debug, PartialEq)]
pub enum MapBuilderCall {
    /// `AddTrajectoryBuilder(expected_sensor_ids, options, callback)`.
    AddTrajectoryBuilder {
        /// The expected sensors.
        expected_sensor_ids: BTreeSet<SensorId>,
        /// The options.
        trajectory_options: Box<mapping_proto::TrajectoryBuilderOptions>,
        /// Whether a local SLAM result callback was given.
        has_local_slam_result_callback: bool,
    },
    /// `GetTrajectoryBuilder(trajectory_id)`.
    GetTrajectoryBuilder(i32),
    /// `FinishTrajectory(trajectory_id)`.
    FinishTrajectory(i32),
    /// `SubmapToProto(submap_id)`.
    SubmapToProto(SubmapId),
    /// `SerializeState(include_unfinished_submaps)`.
    SerializeState(bool),
    /// `SerializeStateToFile(include_unfinished_submaps, filename)`.
    SerializeStateToFile(bool, String),
    /// `LoadState(load_frozen_state)` with the number of messages read.
    LoadState(bool, usize),
    /// `LoadStateFromFile(filename, load_frozen_state)`.
    LoadStateFromFile(String, bool),
}

/// Canned results and recorded calls of a [`FakeMapBuilder`].
#[derive(Default)]
pub struct MapBuilderScript {
    /// Returned by the next `AddTrajectoryBuilder` (then incremented).
    pub next_trajectory_id: i32,
    /// Recorded calls.
    pub calls: Vec<MapBuilderCall>,
    /// The local SLAM result callbacks passed to `AddTrajectoryBuilder`.
    pub local_slam_result_callbacks: BTreeMap<i32, LocalSlamResultCallback>,
    /// Returned by `SubmapToProto`.
    pub submap_to_proto_error: String,
    /// Written by `SerializeState`.
    pub serialized_state: Vec<mapping_proto::SerializedData>,
    /// Returned by `LoadState`/`LoadStateFromFile`.
    pub trajectory_remapping: BTreeMap<i32, i32>,
}

/// Fake map builder; see the module docs.
pub struct FakeMapBuilder {
    script: Arc<Mutex<MapBuilderScript>>,
    pose_graph: Arc<FakePoseGraph>,
    trajectory_builder: Option<MockTrajectoryBuilder>,
}

impl FakeMapBuilder {
    /// Creates the fake. `trajectory_builder` is returned by
    /// `GetTrajectoryBuilder` for every trajectory (C++ `WillRepeatedly(
    /// Return(mock_trajectory_builder_.get()))`); `None` returns null.
    pub fn new(trajectory_builder: Option<MockTrajectoryBuilder>) -> Self {
        Self {
            script: Arc::new(Mutex::new(MapBuilderScript::default())),
            pose_graph: Arc::new(FakePoseGraph::default()),
            trajectory_builder,
        }
    }

    /// The shared script.
    pub fn script(&self) -> Arc<Mutex<MapBuilderScript>> {
        Arc::clone(&self.script)
    }

    /// The shared fake pose graph.
    pub fn fake_pose_graph(&self) -> Arc<FakePoseGraph> {
        Arc::clone(&self.pose_graph)
    }

    fn lock(&self) -> MutexGuard<'_, MapBuilderScript> {
        lock(&self.script)
    }
}

impl MapBuilderInterface for FakeMapBuilder {
    fn add_trajectory_builder(
        &mut self,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &mapping_proto::TrajectoryBuilderOptions,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> i32 {
        let mut script = self.lock();
        let trajectory_id = script.next_trajectory_id;
        script.next_trajectory_id += 1;
        script.calls.push(MapBuilderCall::AddTrajectoryBuilder {
            expected_sensor_ids: expected_sensor_ids.clone(),
            trajectory_options: Box::new(trajectory_options.clone()),
            has_local_slam_result_callback: local_slam_result_callback.is_some(),
        });
        if let Some(callback) = local_slam_result_callback {
            script
                .local_slam_result_callbacks
                .insert(trajectory_id, callback);
        }
        self.pose_graph
            .lock()
            .trajectory_states
            .insert(trajectory_id, TrajectoryState::Active);
        trajectory_id
    }

    fn add_trajectory_for_deserialization(
        &mut self,
        _options_with_sensor_ids_proto: &mapping_proto::TrajectoryBuilderOptionsWithSensorIds,
    ) -> i32 {
        let mut script = self.lock();
        script.next_trajectory_id += 1;
        script.next_trajectory_id - 1
    }

    fn get_trajectory_builder(
        &mut self,
        trajectory_id: i32,
    ) -> Option<&mut dyn TrajectoryBuilderInterface> {
        self.lock()
            .calls
            .push(MapBuilderCall::GetTrajectoryBuilder(trajectory_id));
        self.trajectory_builder
            .as_mut()
            .map(|builder| builder as &mut dyn TrajectoryBuilderInterface)
    }

    fn finish_trajectory(&mut self, trajectory_id: i32) {
        self.lock()
            .calls
            .push(MapBuilderCall::FinishTrajectory(trajectory_id));
        self.pose_graph
            .lock()
            .trajectory_states
            .insert(trajectory_id, TrajectoryState::Finished);
    }

    fn submap_to_proto(
        &mut self,
        submap_id: &SubmapId,
        _response: &mut mapping_proto::submap_query::Response,
    ) -> String {
        let mut script = self.lock();
        script.calls.push(MapBuilderCall::SubmapToProto(*submap_id));
        script.submap_to_proto_error.clone()
    }

    fn serialize_state(
        &mut self,
        include_unfinished_submaps: bool,
        writer: &mut dyn ProtoStreamWriterInterface,
    ) {
        let serialized_state = {
            let mut script = self.lock();
            script
                .calls
                .push(MapBuilderCall::SerializeState(include_unfinished_submaps));
            script.serialized_state.clone()
        };
        writer.write_proto(&mapping_proto::SerializationHeader { format_version: 2 });
        for serialized_data in &serialized_state {
            writer.write_proto(serialized_data);
        }
    }

    fn serialize_state_to_file(
        &mut self,
        include_unfinished_submaps: bool,
        filename: &str,
    ) -> bool {
        self.lock().calls.push(MapBuilderCall::SerializeStateToFile(
            include_unfinished_submaps,
            filename.to_string(),
        ));
        true
    }

    fn load_state(
        &mut self,
        reader: &mut dyn ProtoStreamReaderInterface,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        let mut num_messages = 0;
        let mut header = mapping_proto::SerializationHeader::default();
        if reader.read_proto(&mut header) {
            num_messages += 1;
            let mut serialized_data = mapping_proto::SerializedData::default();
            while reader.read_proto(&mut serialized_data) {
                num_messages += 1;
            }
        }
        let mut script = self.lock();
        script
            .calls
            .push(MapBuilderCall::LoadState(load_frozen_state, num_messages));
        script.trajectory_remapping.clone()
    }

    fn load_state_from_file(
        &mut self,
        filename: &str,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        let mut script = self.lock();
        script.calls.push(MapBuilderCall::LoadStateFromFile(
            filename.to_string(),
            load_frozen_state,
        ));
        script.trajectory_remapping.clone()
    }

    fn num_trajectory_builders(&self) -> i32 {
        self.lock().next_trajectory_id
    }

    fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
        Arc::clone(&self.pose_graph) as Arc<dyn PoseGraphInterface>
    }

    fn get_all_trajectory_builder_options(
        &self,
    ) -> &[mapping_proto::TrajectoryBuilderOptionsWithSensorIds] {
        &[]
    }
}

/// A recorded pose graph call.
#[derive(Clone, Debug, PartialEq)]
pub enum PoseGraphCall {
    /// `RunFinalOptimization()`.
    RunFinalOptimization,
    /// `SetLandmarkPose(landmark_id, global_pose, frozen)`.
    SetLandmarkPose(String, Rigid3d, bool),
    /// `DeleteTrajectory(trajectory_id)`.
    DeleteTrajectory(i32),
    /// `SetGlobalSlamOptimizationCallback(_)`.
    SetGlobalSlamOptimizationCallback,
}

/// Canned results and recorded calls of a [`FakePoseGraph`].
#[derive(Default)]
pub struct PoseGraphScript {
    /// Recorded calls.
    pub calls: Vec<PoseGraphCall>,
    /// Returned by `GetLandmarkPoses`.
    pub landmark_poses: BTreeMap<String, Rigid3d>,
    /// Returned by `GetTrajectoryStates`; also drives
    /// `IsTrajectoryFinished`/`IsTrajectoryFrozen`.
    pub trajectory_states: BTreeMap<i32, TrajectoryState>,
    /// Returned by `GetAllSubmapPoses`.
    pub submap_poses: Vec<(SubmapId, SubmapPose)>,
    /// Returned by `GetTrajectoryNodePoses`.
    pub node_poses: Vec<(NodeId, TrajectoryNodePose)>,
    /// Returned by `constraints()`.
    pub constraints: Vec<Constraint>,
    /// Returned by `GetLocalToGlobalTransform` (identity if unset).
    pub local_to_global: BTreeMap<i32, Rigid3d>,
    /// The callback passed to `SetGlobalSlamOptimizationCallback`.
    pub global_slam_optimization_callback: Option<GlobalSlamOptimizationCallback>,
}

/// Fake pose graph; see the module docs.
#[derive(Default)]
pub struct FakePoseGraph {
    script: Mutex<PoseGraphScript>,
}

impl FakePoseGraph {
    /// Locks the script.
    pub fn lock(&self) -> MutexGuard<'_, PoseGraphScript> {
        lock(&self.script)
    }

    /// Copy of the recorded calls.
    pub fn calls(&self) -> Vec<PoseGraphCall> {
        self.lock().calls.clone()
    }

    /// Invokes the registered global SLAM optimization callback.
    pub fn run_global_slam_optimization_callback(
        &self,
        last_optimized_submap_ids: &BTreeMap<i32, SubmapId>,
        last_optimized_node_ids: &BTreeMap<i32, NodeId>,
    ) {
        let script = self.lock();
        if let Some(callback) = &script.global_slam_optimization_callback {
            callback(last_optimized_submap_ids, last_optimized_node_ids);
        }
    }
}

impl PoseGraphInterface for FakePoseGraph {
    fn run_final_optimization(&self) {
        self.lock().calls.push(PoseGraphCall::RunFinalOptimization);
    }
    fn get_all_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        MapById::new()
    }
    fn get_submap_data(&self, _submap_id: &SubmapId) -> SubmapData {
        SubmapData::default()
    }
    fn get_all_submap_poses(&self) -> MapById<SubmapId, SubmapPose> {
        let mut poses = MapById::new();
        for (id, pose) in &self.lock().submap_poses {
            poses.insert(id, *pose);
        }
        poses
    }
    fn get_local_to_global_transform(&self, trajectory_id: i32) -> Rigid3d {
        self.lock()
            .local_to_global
            .get(&trajectory_id)
            .copied()
            .unwrap_or_else(Rigid3d::identity)
    }
    fn get_trajectory_nodes(&self) -> MapById<NodeId, TrajectoryNode> {
        MapById::new()
    }
    fn get_trajectory_node_poses(&self) -> MapById<NodeId, TrajectoryNodePose> {
        let mut poses = MapById::new();
        for (id, pose) in &self.lock().node_poses {
            poses.insert(id, *pose);
        }
        poses
    }
    fn get_trajectory_states(&self) -> BTreeMap<i32, TrajectoryState> {
        self.lock().trajectory_states.clone()
    }
    fn get_landmark_poses(&self) -> BTreeMap<String, Rigid3d> {
        self.lock().landmark_poses.clone()
    }
    fn set_landmark_pose(&self, landmark_id: &str, global_pose: &Rigid3d, frozen: bool) {
        self.lock().calls.push(PoseGraphCall::SetLandmarkPose(
            landmark_id.to_string(),
            *global_pose,
            frozen,
        ));
    }
    fn delete_trajectory(&self, trajectory_id: i32) {
        let mut script = self.lock();
        script
            .calls
            .push(PoseGraphCall::DeleteTrajectory(trajectory_id));
        if let Some(state) = script.trajectory_states.get_mut(&trajectory_id) {
            *state = TrajectoryState::Deleted;
        }
    }
    fn is_trajectory_finished(&self, trajectory_id: i32) -> bool {
        self.lock().trajectory_states.get(&trajectory_id) == Some(&TrajectoryState::Finished)
    }
    fn is_trajectory_frozen(&self, trajectory_id: i32) -> bool {
        self.lock().trajectory_states.get(&trajectory_id) == Some(&TrajectoryState::Frozen)
    }
    fn get_trajectory_data(&self) -> BTreeMap<i32, TrajectoryData> {
        BTreeMap::new()
    }
    fn constraints(&self) -> Vec<Constraint> {
        self.lock().constraints.clone()
    }
    fn to_proto(&self, _include_unfinished_submaps: bool) -> mapping_proto::PoseGraph {
        mapping_proto::PoseGraph::default()
    }
    fn set_global_slam_optimization_callback(&self, callback: GlobalSlamOptimizationCallback) {
        let mut script = self.lock();
        script
            .calls
            .push(PoseGraphCall::SetGlobalSlamOptimizationCallback);
        script.global_slam_optimization_callback = Some(callback);
    }
}
