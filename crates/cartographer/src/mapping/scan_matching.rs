//! The C++ `cartographer::mapping::scan_matching` namespace.
//!
//! Implementations live under `mapping::internal` (mirroring the C++ source
//! directories); this module re-exports them under the C++ namespace path and
//! hosts the `cartographer.mapping.scan_matching.proto` package.

/// Generated protobuf types of package `cartographer.mapping.scan_matching.proto`.
pub mod proto {
    pub use crate::protos::cartographer::mapping::scan_matching::proto::*;
}

pub use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
