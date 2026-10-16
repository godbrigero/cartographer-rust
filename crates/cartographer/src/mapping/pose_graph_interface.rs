//! Port of `cartographer/mapping/pose_graph_interface` (.h).
//!
//! RUST PORT NOTE: the types nested in the C++ `PoseGraphInterface` class
//! (`PoseGraphInterface::Constraint`, `::LandmarkNode`, `::TrajectoryData`,
//! ...) are free types of this module. Nested-nested types get flattened
//! names: `Constraint::Pose` -> [`Pose`], `Constraint::Tag` -> [`Tag`],
//! `LandmarkNode::LandmarkObservation` -> [`LandmarkObservation`].
//! [`SubmapData`] holds the submap as `Arc<RwLock<dyn Submap>>` (see
//! `mapping::submaps`).
//!
//! RUST PORT NOTE: the C++ abstract class becomes the object-safe trait
//! [`PoseGraphInterface`]. All methods take `&self`: implementations are
//! internally synchronized (C++ `absl::Mutex` members) and shared between
//! threads, hence the `Send + Sync` supertraits.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use crate::common::time::Time;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::proto;
use crate::mapping::submaps::Submap;
use crate::mapping::trajectory_node::{TrajectoryNode, TrajectoryNodePose};
use crate::transform::rigid_transform::Rigid3d;

/// `PoseGraphInterface::Constraint::Pose`: the relative pose of a constraint
/// and its weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Pose of the node 'j' relative to submap 'i'.
    pub zbar_ij: Rigid3d,
    /// Weight of the translational part.
    pub translation_weight: f64,
    /// Weight of the rotational part.
    pub rotation_weight: f64,
}

/// `PoseGraphInterface::Constraint::Tag`: differentiates between
/// intra-submap (where node 'j' was inserted into submap 'i') and
/// inter-submap constraints (where node 'j' was not inserted into submap
/// 'i').
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tag {
    /// C++ `INTRA_SUBMAP`.
    IntraSubmap,
    /// C++ `INTER_SUBMAP`.
    InterSubmap,
}

/// A "constraint" as in the paper by Konolige, Kurt, et al. "Efficient sparse
/// pose adjustment for 2d mapping." Intelligent Robots and Systems (IROS),
/// 2010 IEEE/RSJ International Conference on (pp. 22--29). IEEE, 2010.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraint {
    /// 'i' in the paper.
    pub submap_id: SubmapId,
    /// 'j' in the paper.
    pub node_id: NodeId,
    /// Pose of the node 'j' relative to submap 'i'.
    pub pose: Pose,
    /// Intra- or inter-submap constraint.
    pub tag: Tag,
}

/// `PoseGraphInterface::LandmarkNode::LandmarkObservation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkObservation {
    /// Trajectory of the observing node.
    pub trajectory_id: i32,
    /// Observation time.
    pub time: Time,
    /// Pose of the landmark in the tracking frame.
    pub landmark_to_tracking_transform: Rigid3d,
    /// Weight of the translational part.
    pub translation_weight: f64,
    /// Weight of the rotational part.
    pub rotation_weight: f64,
}

/// `PoseGraphInterface::LandmarkNode`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkNode {
    /// All observations of this landmark.
    pub landmark_observations: Vec<LandmarkObservation>,
    /// The optimized global pose, if known.
    pub global_landmark_pose: Option<Rigid3d>,
    /// Whether the landmark pose is fixed during optimization.
    pub frozen: bool,
}

/// `PoseGraphInterface::SubmapPose`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapPose {
    /// Submap version (number of insertions).
    pub version: i32,
    /// Global pose of the submap.
    pub pose: Rigid3d,
}

/// `PoseGraphInterface::SubmapData`: a submap and its global pose.
///
/// RUST PORT NOTE: C++ holds a `std::shared_ptr<const Submap>` (possibly
/// null); here an optional shared handle. A `SharedSubmap<Submap2D>` /
/// `SharedSubmap<Submap3D>` coerces to `Arc<RwLock<dyn Submap>>`; concrete
/// types can be recovered by upcasting `&dyn Submap` to `&dyn Any` (C++
/// `static_pointer_cast`).
#[derive(Clone, Default)]
pub struct SubmapData {
    /// The submap (`None` = C++ null `shared_ptr`).
    pub submap: Option<Arc<RwLock<dyn Submap>>>,
    /// Global pose of the submap.
    pub pose: Rigid3d,
}

impl fmt::Debug for SubmapData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmapData")
            .field("has_submap", &self.submap.is_some())
            .field("pose", &self.pose)
            .finish()
    }
}

/// `PoseGraphInterface::TrajectoryData`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryData {
    /// Gravity constant (default 9.8).
    pub gravity_constant: f64,
    /// IMU calibration quaternion `(w, x, y, z)` (default identity).
    pub imu_calibration: [f64; 4],
    /// Origin of the fixed frame in the map frame, if known.
    pub fixed_frame_origin_in_map: Option<Rigid3d>,
}

impl Default for TrajectoryData {
    fn default() -> Self {
        TrajectoryData {
            gravity_constant: 9.8,
            imu_calibration: [1., 0., 0., 0.],
            fixed_frame_origin_in_map: None,
        }
    }
}

/// `PoseGraphInterface::TrajectoryState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrajectoryState {
    /// C++ `ACTIVE`.
    Active,
    /// C++ `FINISHED`.
    Finished,
    /// C++ `FROZEN`.
    Frozen,
    /// C++ `DELETED`.
    Deleted,
}

/// Callback invoked whenever the global optimization problem is solved, with
/// the last optimized submap and node ID per trajectory.
pub type GlobalSlamOptimizationCallback = Box<
    dyn Fn(&BTreeMap<i32 /* trajectory_id */, SubmapId>, &BTreeMap<i32 /* trajectory_id */, NodeId>)
        + Send
        + Sync,
>;

/// The read API of the pose graph (C++ `PoseGraphInterface`).
pub trait PoseGraphInterface: Send + Sync {
    /// Waits for all computations to finish and computes optimized poses.
    fn run_final_optimization(&self);

    /// Returns data for all submaps.
    fn get_all_submap_data(&self) -> MapById<SubmapId, SubmapData>;

    /// Returns the current optimized transform and submap itself for the
    /// given `submap_id`. Returns `None` for the `submap` member if the submap
    /// does not exist (anymore).
    fn get_submap_data(&self, submap_id: &SubmapId) -> SubmapData;

    /// Returns the global poses for all submaps.
    fn get_all_submap_poses(&self) -> MapById<SubmapId, SubmapPose>;

    /// Returns the transform converting data in the local map frame (i.e. the
    /// continuous, non-loop-closed frame) into the global map frame (i.e. the
    /// discontinuous, loop-closed frame).
    fn get_local_to_global_transform(&self, trajectory_id: i32) -> Rigid3d;

    /// Returns the current optimized trajectories.
    fn get_trajectory_nodes(&self) -> MapById<NodeId, TrajectoryNode>;

    /// Returns the current optimized trajectory poses.
    fn get_trajectory_node_poses(&self) -> MapById<NodeId, TrajectoryNodePose>;

    /// Returns the states of trajectories.
    fn get_trajectory_states(&self) -> BTreeMap<i32, TrajectoryState>;

    /// Returns the current optimized landmark poses.
    fn get_landmark_poses(&self) -> BTreeMap<String, Rigid3d>;

    /// Sets global pose of landmark `landmark_id` to given `global_pose`
    /// (C++ default argument `frozen = false`).
    fn set_landmark_pose(&self, landmark_id: &str, global_pose: &Rigid3d, frozen: bool);

    /// Deletes a trajectory asynchronously.
    fn delete_trajectory(&self, trajectory_id: i32);

    /// Checks if the given trajectory is finished.
    fn is_trajectory_finished(&self, trajectory_id: i32) -> bool;

    /// Checks if the given trajectory is frozen.
    fn is_trajectory_frozen(&self, trajectory_id: i32) -> bool;

    /// Returns the trajectory data.
    fn get_trajectory_data(&self) -> BTreeMap<i32, TrajectoryData>;

    /// Returns the collection of constraints.
    fn constraints(&self) -> Vec<Constraint>;

    /// Serializes the constraints and trajectories. If
    /// `include_unfinished_submaps` is set to `true`, unfinished submaps, i.e.
    /// submaps that have not yet received all rangefinder data insertions,
    /// will be included, otherwise not.
    ///
    /// RUST PORT NOTE: C++ implements this once in the `PoseGraph` base
    /// class; implementations delegate to
    /// [`crate::mapping::pose_graph::pose_graph_to_proto`].
    fn to_proto(&self, include_unfinished_submaps: bool) -> proto::PoseGraph;

    /// Sets the callback function that is invoked whenever the global
    /// optimization problem is solved.
    fn set_global_slam_optimization_callback(&self, callback: GlobalSlamOptimizationCallback);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_data_defaults() {
        let data = TrajectoryData::default();
        assert_eq!(data.gravity_constant, 9.8);
        assert_eq!(data.imu_calibration, [1., 0., 0., 0.]);
        assert!(data.fixed_frame_origin_in_map.is_none());
        let node = LandmarkNode::default();
        assert!(!node.frozen);
        assert!(node.global_landmark_pose.is_none());
    }
}
