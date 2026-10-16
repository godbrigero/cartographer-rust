//! Port of `cartographer/mapping/internal/testing/fake_trimmable` (.h).

use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::pose_graph_interface::{Constraint, SubmapData, TrajectoryState};
use crate::mapping::pose_graph_trimmer::Trimmable;
use crate::mapping::trajectory_node::TrajectoryNode;

/// A [`Trimmable`] for tests that records the trimmed submaps.
#[derive(Debug, Default)]
pub struct FakeTrimmable {
    trimmed_submaps: Vec<SubmapId>,

    constraints: Vec<Constraint>,
    trajectory_nodes: MapById<NodeId, TrajectoryNode>,
    submap_data: MapById<SubmapId, SubmapData>,
}

impl FakeTrimmable {
    /// Creates an empty fake pose graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Populates dummy SubmapIDs (C++ `FakeTrimmable(trajectory_id,
    /// num_submaps)`).
    pub fn with_submaps(trajectory_id: i32, num_submaps: i32) -> Self {
        let mut fake = Self::default();
        for index in 0..num_submaps {
            fake.submap_data
                .insert(&SubmapId::new(trajectory_id, index), SubmapData::default());
        }
        fake
    }

    /// Replaces the submap data.
    pub fn set_submap_data(&mut self, submap_data: MapById<SubmapId, SubmapData>) {
        self.submap_data = submap_data;
    }

    /// Mutable access to the submap data.
    pub fn mutable_submap_data(&mut self) -> &mut MapById<SubmapId, SubmapData> {
        &mut self.submap_data
    }

    /// Replaces the trajectory nodes.
    pub fn set_trajectory_nodes(&mut self, trajectory_nodes: MapById<NodeId, TrajectoryNode>) {
        self.trajectory_nodes = trajectory_nodes;
    }

    /// Mutable access to the trajectory nodes.
    pub fn mutable_trajectory_nodes(&mut self) -> &mut MapById<NodeId, TrajectoryNode> {
        &mut self.trajectory_nodes
    }

    /// Replaces the constraints.
    pub fn set_constraints(&mut self, constraints: Vec<Constraint>) {
        self.constraints = constraints;
    }

    /// Mutable access to the constraints.
    pub fn mutable_constraints(&mut self) -> &mut Vec<Constraint> {
        &mut self.constraints
    }

    /// The submaps trimmed so far, in call order.
    pub fn trimmed_submaps(&self) -> Vec<SubmapId> {
        self.trimmed_submaps.clone()
    }
}

impl Trimmable for FakeTrimmable {
    fn num_submaps(&self, _trajectory_id: i32) -> i32 {
        (self.submap_data.size() - self.trimmed_submaps.len()) as i32
    }

    fn get_submap_ids(&self, _trajectory_id: i32) -> Vec<SubmapId> {
        self.submap_data.iter().map(|submap| submap.id).collect()
    }

    fn get_optimized_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        self.submap_data.clone()
    }

    fn get_trajectory_nodes(&self) -> &MapById<NodeId, TrajectoryNode> {
        &self.trajectory_nodes
    }

    fn get_constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    fn trim_submap(&mut self, submap_id: &SubmapId) {
        self.trimmed_submaps.push(*submap_id);
    }

    fn is_finished(&self, _trajectory_id: i32) -> bool {
        false
    }

    fn set_trajectory_state(&mut self, _trajectory_id: i32, _state: TrajectoryState) {}
}
