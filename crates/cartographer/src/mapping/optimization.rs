//! The C++ `cartographer::mapping::optimization` namespace.
//!
//! Implementations live under `mapping::internal` (mirroring the C++ source
//! directories); this module re-exports them under the C++ namespace path and
//! hosts the `cartographer.mapping.optimization.proto` package.

/// Generated protobuf types of package `cartographer.mapping.optimization.proto`.
pub mod proto {
    pub use crate::protos::cartographer::mapping::optimization::proto::*;
}

pub use crate::mapping::internal::optimization::ceres_pose::{CeresPose, Data, from_pose};
pub use crate::mapping::internal::optimization::cost_functions::cost_helpers::{
    compute_unscaled_error_2d, compute_unscaled_error_3d, interpolate_nodes_2d,
    interpolate_nodes_3d, scale_error_2d, scale_error_3d, slerp_quaternions,
};
pub use crate::mapping::internal::optimization::cost_functions::landmark_cost_function_2d::LandmarkCostFunction2D;
pub use crate::mapping::internal::optimization::cost_functions::landmark_cost_function_3d::LandmarkCostFunction3D;
pub use crate::mapping::internal::optimization::cost_functions::spa_cost_function_2d::{
    create_analytical_spa_cost_function, create_auto_diff_spa_cost_function,
};
pub use crate::mapping::internal::optimization::cost_functions::spa_cost_function_3d::SpaCostFunction3D;
pub use crate::mapping::internal::optimization::optimization_problem_2d::{
    NodeSpec2D, SubmapSpec2D,
};
pub use crate::mapping::internal::optimization::optimization_problem_3d::{
    NodeSpec3D, SubmapSpec3D,
};
