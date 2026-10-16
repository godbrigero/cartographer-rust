//! Port of `cartographer/mapping/internal/optimization/optimization_problem_2d` (.h, .cc).
//!
//! NOTE: only `NodeSpec2D` and `SubmapSpec2D` are ported so far (added with
//! the cost functions, which need `NodeSpec2D`); `OptimizationProblem2D` is
//! added by the optimization problem port.

use nalgebra::Quaternion;

use crate::common::time::Time;
use crate::transform::rigid_transform::Rigid2d;

/// Per-node data of the 2D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec2D {
    /// Node time.
    pub time: Time,
    /// Pose in the local (non-loop-closed) frame.
    pub local_pose_2d: Rigid2d,
    /// Pose in the global (loop-closed) frame.
    pub global_pose_2d: Rigid2d,
    /// Rotation aligning the tracking frame with gravity.
    pub gravity_alignment: Quaternion<f64>,
}

/// Per-submap data of the 2D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapSpec2D {
    /// Global pose of the submap.
    pub global_pose: Rigid2d,
}
