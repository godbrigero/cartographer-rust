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
pub use internal::range_data_collator::RangeDataCollator;
pub use internal::two_d::ray_to_pixel_mask::ray_to_pixel_mask;
pub use internal::two_d::tsd_value_converter::TSDValueConverter;
pub use internal::two_d::tsdf_2d::TSDF2D;
pub use three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
pub use trajectory_builder_interface::{SensorId, SensorType};
pub use two_d::grid_2d::{AlignedBox2i, Grid2D, Grid2DTrait, GridType, create_grid_options_2d};
pub use two_d::map_limits::MapLimits;
pub use two_d::probability_grid::ProbabilityGrid;
pub use two_d::xy_index::{Array2i, CellLimits, XYIndexRangeIterator};
pub use value_conversion_tables::ValueConversionTables;
