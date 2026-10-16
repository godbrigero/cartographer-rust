//! Port of `cartographer/mapping/internal/optimization/optimization_problem_interface` (.h).
//!
//! RUST PORT NOTE: the C++ class template
//! `OptimizationProblemInterface<NodeDataType, SubmapDataType,
//! RigidTransformType>` becomes a trait with the associated types
//! [`OptimizationProblemInterface::NodeData`],
//! [`OptimizationProblemInterface::SubmapData`] and
//! [`OptimizationProblemInterface::RigidTransform`] (each implementation
//! instantiates the template exactly once). The C++ nested aliases
//! `Constraint`/`LandmarkNode` are the types of
//! [`crate::mapping::pose_graph_interface`]. C++ `std::map` arguments and
//! results become `BTreeMap` (same ordering).

use std::collections::BTreeMap;

use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::pose_graph_interface::{Constraint, LandmarkNode, TrajectoryState};
use crate::sensor::imu_data::ImuData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::Rigid3d;

/// Implements the SPA loop closure method.
pub trait OptimizationProblemInterface {
    /// Per-node data (C++ `NodeDataType`).
    type NodeData;
    /// Per-submap data (C++ `SubmapDataType`).
    type SubmapData;
    /// Type of global submap poses (C++ `RigidTransformType`).
    type RigidTransform;

    /// Adds IMU data of `trajectory_id`.
    fn add_imu_data(&mut self, trajectory_id: i32, imu_data: &ImuData);
    /// Adds odometry data of `trajectory_id`.
    fn add_odometry_data(&mut self, trajectory_id: i32, odometry_data: &OdometryData);
    /// Appends a node to `trajectory_id`.
    fn add_trajectory_node(&mut self, trajectory_id: i32, node_data: &Self::NodeData);
    /// Inserts a node with a given ID.
    fn insert_trajectory_node(&mut self, node_id: &NodeId, node_data: &Self::NodeData);
    /// Removes a node (and the sensor data only needed for it).
    fn trim_trajectory_node(&mut self, node_id: &NodeId);
    /// Appends a submap to `trajectory_id`.
    fn add_submap(&mut self, trajectory_id: i32, global_submap_pose: &Self::RigidTransform);
    /// Inserts a submap with a given ID.
    fn insert_submap(&mut self, submap_id: &SubmapId, global_submap_pose: &Self::RigidTransform);
    /// Removes a submap.
    fn trim_submap(&mut self, submap_id: &SubmapId);
    /// Sets the maximum number of solver iterations used by `solve`.
    fn set_max_num_iterations(&mut self, max_num_iterations: i32);

    /// Optimizes the global poses.
    fn solve(
        &mut self,
        constraints: &[Constraint],
        trajectories_state: &BTreeMap<i32, TrajectoryState>,
        landmark_nodes: &BTreeMap<String, LandmarkNode>,
    );

    /// The node data (with optimized global poses after `solve`).
    fn node_data(&self) -> &MapById<NodeId, Self::NodeData>;
    /// The submap data (with optimized global poses after `solve`).
    fn submap_data(&self) -> &MapById<SubmapId, Self::SubmapData>;
    /// The optimized landmark poses.
    fn landmark_data(&self) -> &BTreeMap<String, Rigid3d>;
    /// The IMU data.
    fn imu_data(&self) -> &MapByTime<ImuData>;
    /// The odometry data.
    fn odometry_data(&self) -> &MapByTime<OdometryData>;
}
