//! Port of `cartographer/mapping/`.

pub mod detect_floors;
pub mod grid_interface;
pub mod id;
pub mod imu_tracker;
pub mod internal;
pub mod map_builder;
pub mod map_builder_interface;
pub mod pose_extrapolator;
pub mod pose_extrapolator_interface;
pub mod pose_graph;
pub mod pose_graph_interface;
pub mod pose_graph_trimmer;
pub mod probability_values;
pub mod range_data_inserter_interface;
pub mod submaps;
pub mod three_d;
pub mod trajectory_builder_interface;
pub mod trajectory_node;
pub mod two_d;
pub mod value_conversion_tables;

/// Generated protobuf types of package `cartographer.mapping.proto`.
pub mod proto {
    pub use crate::protos::cartographer::mapping::proto::*;
}
pub mod constraints;
pub mod optimization;
pub mod scan_matching;

// Re-exports mirroring the C++ `cartographer::mapping` namespace, which spans
// several source directories.
pub use detect_floors::{Floor, Timespan, detect_floors};
pub use id::{MapById, NodeId, SubmapId};
pub use imu_tracker::ImuTracker;
pub use internal::collated_trajectory_builder::CollatedTrajectoryBuilder;
pub use internal::connected_components::ConnectedComponents;
pub use internal::eigen_quaterniond_from_two_vectors::from_two_vectors;
pub use internal::global_trajectory_builder::{
    create_global_trajectory_builder_2d, create_global_trajectory_builder_3d,
    global_trajectory_builder_register_metrics,
};
pub use internal::imu_based_pose_extrapolator::ImuBasedPoseExtrapolator;
pub use internal::local_slam_result_data::LocalSlamResultData;
pub use internal::motion_filter::{MotionFilter, create_motion_filter_options};
pub use internal::optimization::cost_functions::acceleration_cost_function_3d::AccelerationCostFunction3D;
pub use internal::optimization::cost_functions::rotation_cost_function_3d::RotationCostFunction3D;
pub use internal::range_data_collator::RangeDataCollator;
pub use internal::submap_controller::SubmapController;
pub use internal::three_d::imu_integration::{
    ExtrapolatePoseResult, ImuDataRange, IntegrateImuResult, extrapolate_pose_with_imu,
    extrapolate_pose_with_imu_from_poses, integrate_imu, integrate_imu_with_calibration,
};
pub use internal::three_d::local_slam_result_3d::LocalSlamResult3D;
pub use internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
pub use internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
pub use internal::three_d::pose_graph_3d::PoseGraph3D;
pub use internal::three_d::rotation_parameterization::{
    ConstantYawQuaternionPlus, YawOnlyQuaternionPlus,
};
pub use internal::trajectory_connectivity_state::TrajectoryConnectivityState;
pub use internal::two_d::local_slam_result_2d::LocalSlamResult2D;
pub use internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
pub use internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
pub use internal::two_d::normal_estimation_2d::{
    create_normal_estimation_options_2d, estimate_normals,
};
pub use internal::two_d::overlapping_submaps_trimmer_2d::OverlappingSubmapsTrimmer2D;
pub use internal::two_d::pose_graph_2d::PoseGraph2D;
pub use internal::two_d::ray_to_pixel_mask::ray_to_pixel_mask;
pub use internal::two_d::tsd_value_converter::TSDValueConverter;
pub use internal::two_d::tsdf_2d::TSDF2D;
pub use internal::two_d::tsdf_range_data_inserter_2d::{
    TSDFRangeDataInserter2D, create_tsdf_range_data_inserter_options_2d,
};
pub use map_builder::{MapBuilder, MapBuilderPoseGraph, create_map_builder};
pub use map_builder_interface::{MapBuilderInterface, create_map_builder_options};
pub use pose_extrapolator::PoseExtrapolator;
pub use pose_extrapolator_interface::{
    ExtrapolationResult, PoseExtrapolatorInterface, create_pose_extrapolator_options,
};
pub use pose_graph::{
    InitialTrajectoryPose, PoseGraph, create_pose_graph_options, pose_graph_to_proto,
};
pub use pose_graph_interface::PoseGraphInterface;
pub use pose_graph_interface::SubmapData;
pub use pose_graph_trimmer::{PoseGraphTrimmer, PureLocalizationTrimmer, Trimmable};
pub use range_data_inserter_interface::{
    RangeDataInserterInterface, create_range_data_inserter_options,
};
pub use submaps::{SharedSubmap, Submap, SubmapBase};
pub use three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
pub use three_d::range_data_inserter_3d::{
    RangeDataInserter3D, create_range_data_inserter_options_3d,
};
pub use three_d::submap_3d::{ActiveSubmaps3D, Submap3D, create_submaps_options_3d};
pub use trajectory_builder_interface::{
    InsertionResult, LocalSlamResultCallback, TrajectoryBuilderInterface,
    create_trajectory_builder_options,
};
pub use trajectory_builder_interface::{SensorId, SensorType};
pub use trajectory_node::{TrajectoryNode, TrajectoryNodeData, TrajectoryNodePose};
pub use two_d::grid_2d::{AlignedBox2i, Grid2D, Grid2DTrait, GridType, create_grid_options_2d};
pub use two_d::map_limits::MapLimits;
pub use two_d::probability_grid::ProbabilityGrid;
pub use two_d::probability_grid_range_data_inserter_2d::{
    ProbabilityGridRangeDataInserter2D, create_probability_grid_range_data_inserter_options_2d,
};
pub use two_d::submap_2d::{ActiveSubmaps2D, Submap2D, create_submaps_options_2d};
pub use two_d::xy_index::{Array2i, CellLimits, XYIndexRangeIterator};
pub use value_conversion_tables::ValueConversionTables;
