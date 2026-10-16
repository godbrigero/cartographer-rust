//! The C++ `cartographer::mapping::scan_matching` namespace.
//!
//! Implementations live under `mapping::internal` (mirroring the C++ source
//! directories); this module re-exports them under the C++ namespace path and
//! hosts the `cartographer.mapping.scan_matching.proto` package.

/// Generated protobuf types of package `cartographer.mapping.scan_matching.proto`.
pub mod proto {
    pub use crate::protos::cartographer::mapping::scan_matching::proto::*;
}

pub use crate::mapping::internal::scan_matching::real_time_correlative_scan_matcher::create_real_time_correlative_scan_matcher_options;
pub use crate::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::{
    CeresScanMatcher3D, PointCloudAndHybridGridsPointers, create_ceres_scan_matcher_options_3d,
};
pub use crate::mapping::internal::three_d::scan_matching::fast_correlative_scan_matcher_3d::{
    Candidate3D, DiscreteScan3D, FastCorrelativeScanMatcher3D, FastCorrelativeScanMatcher3DResult,
    PrecomputationGridStack3D, create_fast_correlative_scan_matcher_options_3d,
};
pub use crate::mapping::internal::three_d::scan_matching::intensity_cost_function_3d::IntensityCostFunction3D;
pub use crate::mapping::internal::three_d::scan_matching::interpolated_grid::{
    InterpolatedGrid, InterpolatedIntensityGrid, InterpolatedProbabilityGrid, InterpolationSource,
};
pub use crate::mapping::internal::three_d::scan_matching::low_resolution_matcher::{
    MatchingFunction, create_low_resolution_matcher,
};
pub use crate::mapping::internal::three_d::scan_matching::occupied_space_cost_function_3d::OccupiedSpaceCostFunction3D;
pub use crate::mapping::internal::three_d::scan_matching::precomputation_grid_3d::{
    PrecomputationGrid3D, convert_to_precomputation_grid, precompute_grid,
};
pub use crate::mapping::internal::three_d::scan_matching::real_time_correlative_scan_matcher_3d::RealTimeCorrelativeScanMatcher3D;
pub use crate::mapping::internal::three_d::scan_matching::rotation_delta_cost_functor_3d::RotationDeltaCostFunctor3D;
pub use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
pub use crate::mapping::internal::three_d::scan_matching::translation_delta_cost_functor_3d::TranslationDeltaCostFunctor3D;
pub use crate::mapping::internal::two_d::scan_matching::ceres_scan_matcher_2d::{
    CeresScanMatcher2D, create_ceres_scan_matcher_options_2d,
};
pub use crate::mapping::internal::two_d::scan_matching::correlative_scan_matcher_2d::{
    Candidate2D, DiscreteScan2D, LinearBounds, SearchParameters, discretize_scans,
    generate_rotated_scans,
};
pub use crate::mapping::internal::two_d::scan_matching::fast_correlative_scan_matcher_2d::{
    FastCorrelativeScanMatcher2D, PrecomputationGrid2D, PrecomputationGridStack2D,
    create_fast_correlative_scan_matcher_options_2d,
};
pub use crate::mapping::internal::two_d::scan_matching::interpolated_tsdf_2d::InterpolatedTSDF2D;
pub use crate::mapping::internal::two_d::scan_matching::occupied_space_cost_function_2d::create_occupied_space_cost_function_2d;
pub use crate::mapping::internal::two_d::scan_matching::real_time_correlative_scan_matcher_2d::RealTimeCorrelativeScanMatcher2D;
pub use crate::mapping::internal::two_d::scan_matching::rotation_delta_cost_functor_2d::RotationDeltaCostFunctor2D;
pub use crate::mapping::internal::two_d::scan_matching::translation_delta_cost_functor_2d::TranslationDeltaCostFunctor2D;
pub use crate::mapping::internal::two_d::scan_matching::tsdf_match_cost_function_2d::create_tsdf_match_cost_function_2d;
