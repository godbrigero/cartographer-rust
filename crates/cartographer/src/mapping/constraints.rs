//! The C++ `cartographer::mapping::constraints` namespace.
//!
//! Implementations live under `mapping::internal` (mirroring the C++ source
//! directories); this module re-exports them under the C++ namespace path and
//! hosts the `cartographer.mapping.constraints.proto` package.

/// Generated protobuf types of package `cartographer.mapping.constraints.proto`.
pub mod proto {
    pub use crate::protos::cartographer::mapping::constraints::proto::*;
}

pub use crate::mapping::internal::constraints::constraint_builder::create_constraint_builder_options;
pub use crate::mapping::internal::constraints::constraint_builder_2d::{
    ConstraintBuilder2D, compute_submap_pose,
};
pub use crate::mapping::internal::constraints::constraint_builder_3d::ConstraintBuilder3D;
