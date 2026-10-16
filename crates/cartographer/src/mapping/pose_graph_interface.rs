//! Port of `cartographer/mapping/pose_graph_interface` (.h).
//!
//! RUST PORT NOTE: the types nested in the C++ `PoseGraphInterface` class
//! (`PoseGraphInterface::Constraint`, `::LandmarkNode`, `::TrajectoryData`,
//! ...) are free types of this module. Nested-nested types get flattened
//! names: `Constraint::Pose` -> [`Pose`], `Constraint::Tag` -> [`Tag`],
//! `LandmarkNode::LandmarkObservation` -> [`LandmarkObservation`].
//! [`SubmapData`] holds the submap as `Arc<RwLock<dyn Submap>>` (see
//! `mapping::submaps`).

// NOTE: trait PoseGraphInterface added by the pose graph port

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};

use crate::common::time::Time;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::submaps::Submap;
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
