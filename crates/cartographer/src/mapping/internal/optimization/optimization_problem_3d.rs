//! Port of `cartographer/mapping/internal/optimization/optimization_problem_3d` (.h, .cc).
//!
//! NOTE: only `NodeSpec3D` and `SubmapSpec3D` are ported so far (added with
//! the cost functions, which need `NodeSpec3D`); `OptimizationProblem3D` is
//! added by the optimization problem port.

use crate::common::time::Time;
use crate::transform::rigid_transform::Rigid3d;

/// Per-node data of the 3D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec3D {
    /// Node time.
    pub time: Time,
    /// Pose in the local (non-loop-closed) frame.
    pub local_pose: Rigid3d,
    /// Pose in the global (loop-closed) frame.
    pub global_pose: Rigid3d,
}

/// Per-submap data of the 3D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapSpec3D {
    /// Global pose of the submap.
    pub global_pose: Rigid3d,
}
