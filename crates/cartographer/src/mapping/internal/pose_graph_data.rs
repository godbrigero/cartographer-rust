//! Port of `cartographer/mapping/internal/pose_graph_data` (.h).
//!
//! RUST PORT NOTE: C++ stores `std::shared_ptr<const Submap>` in
//! [`InternalSubmapData`] and `static_cast`s it to `Submap2D`/`Submap3D` in
//! the concrete pose graphs. Here the data is generic over the concrete
//! submap type `S` (`Submap2D` or `Submap3D`) and holds a
//! [`SharedSubmap<S>`], so no downcasts are needed; [`InternalSubmapData::
//! submap_as_dyn`] gives the polymorphic handle used by the public API.
//! Everything else is dimension-agnostic and shared by the 2D and 3D pose
//! graphs (both `global_submap_poses_2d` and `global_submap_poses_3d` exist,
//! like in C++; each pose graph uses one of them).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};

use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::optimization_problem_2d::SubmapSpec2D;
use crate::mapping::internal::optimization::optimization_problem_3d::SubmapSpec3D;
use crate::mapping::internal::trajectory_connectivity_state::TrajectoryConnectivityState;
use crate::mapping::pose_graph::InitialTrajectoryPose;
use crate::mapping::pose_graph_interface::{Constraint, LandmarkNode, TrajectoryState};
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::trajectory_node::TrajectoryNode;

/// The current state of the submap in the background threads. After this
/// transitions to `Finished`, all nodes are tried to match against this
/// submap. Likewise, all new nodes are matched against submaps in that
/// state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SubmapState {
    /// C++ `kNoConstraintSearch`.
    #[default]
    NoConstraintSearch,
    /// C++ `kFinished`.
    Finished,
}

/// C++ `InternalTrajectoryState::DeletionState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DeletionState {
    /// C++ `NORMAL`.
    #[default]
    Normal,
    /// C++ `SCHEDULED_FOR_DELETION`.
    ScheduledForDeletion,
    /// C++ `WAIT_FOR_DELETION`.
    WaitForDeletion,
}

/// The state of a trajectory including its deletion state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InternalTrajectoryState {
    /// The public trajectory state.
    pub state: TrajectoryState,
    /// The deletion state.
    pub deletion_state: DeletionState,
}

impl Default for InternalTrajectoryState {
    fn default() -> Self {
        Self {
            state: TrajectoryState::Active,
            deletion_state: DeletionState::Normal,
        }
    }
}

/// A submap known to the pose graph.
#[derive(Debug)]
pub struct InternalSubmapData<S> {
    /// The submap (shared with the local trajectory builder).
    pub submap: SharedSubmap<S>,
    /// The background state of the submap.
    pub state: SubmapState,
    /// IDs of the nodes that were inserted into this map together with
    /// constraints for them. They are not to be matched again when this
    /// submap becomes `Finished`.
    pub node_ids: BTreeSet<NodeId>,
}

impl<S> Clone for InternalSubmapData<S> {
    fn clone(&self) -> Self {
        Self {
            submap: Arc::clone(&self.submap),
            state: self.state,
            node_ids: self.node_ids.clone(),
        }
    }
}

impl<S> InternalSubmapData<S> {
    /// Creates the data for a newly seen submap (state
    /// `NoConstraintSearch`, no nodes).
    pub fn new(submap: SharedSubmap<S>) -> Self {
        Self {
            submap,
            state: SubmapState::NoConstraintSearch,
            node_ids: BTreeSet::new(),
        }
    }
}

impl<S: Submap> InternalSubmapData<S> {
    /// The submap as polymorphic handle (C++ `std::shared_ptr<const Submap>`).
    pub fn submap_as_dyn(&self) -> Arc<RwLock<dyn Submap>> {
        Arc::clone(&self.submap) as Arc<RwLock<dyn Submap>>
    }
}

/// The data of the pose graph guarded by its main mutex.
#[derive(Debug)]
pub struct PoseGraphData<S> {
    /// Submaps get assigned an ID and state as soon as they are seen, even
    /// before they take part in the background computations.
    pub submap_data: MapById<SubmapId, InternalSubmapData<S>>,

    /// Global submap poses currently used for displaying data (2D).
    pub global_submap_poses_2d: MapById<SubmapId, SubmapSpec2D>,
    /// Global submap poses currently used for displaying data (3D).
    pub global_submap_poses_3d: MapById<SubmapId, SubmapSpec3D>,

    /// Data that are currently being shown.
    pub trajectory_nodes: MapById<NodeId, TrajectoryNode>,

    /// Global landmark poses with all observations.
    pub landmark_nodes: BTreeMap<String /* landmark ID */, LandmarkNode>,

    /// How our various trajectories are related.
    pub trajectory_connectivity_state: TrajectoryConnectivityState,
    /// Number of nodes added through `AddNode` (live SLAM).
    pub num_trajectory_nodes: i32,
    /// The states of all known trajectories.
    pub trajectories_state: BTreeMap<i32, InternalTrajectoryState>,

    /// Set of all initial trajectory poses.
    pub initial_trajectory_poses: BTreeMap<i32, InitialTrajectoryPose>,

    /// All constraints (2D: in the gravity-aligned frame of the nodes).
    pub constraints: Vec<Constraint>,
}

impl<S> Default for PoseGraphData<S> {
    fn default() -> Self {
        Self {
            submap_data: MapById::new(),
            global_submap_poses_2d: MapById::new(),
            global_submap_poses_3d: MapById::new(),
            trajectory_nodes: MapById::new(),
            landmark_nodes: BTreeMap::new(),
            trajectory_connectivity_state: TrajectoryConnectivityState::new(),
            num_trajectory_nodes: 0,
            trajectories_state: BTreeMap::new(),
            initial_trajectory_poses: BTreeMap::new(),
            constraints: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::two_d::submap_2d::Submap2D;

    #[test]
    fn defaults() {
        let data = PoseGraphData::<Submap2D>::default();
        assert!(data.submap_data.empty());
        assert!(data.trajectory_nodes.empty());
        assert_eq!(data.num_trajectory_nodes, 0);
        let state = InternalTrajectoryState::default();
        assert_eq!(state.state, TrajectoryState::Active);
        assert_eq!(state.deletion_state, DeletionState::Normal);
        assert_eq!(SubmapState::default(), SubmapState::NoConstraintSearch);
    }
}
