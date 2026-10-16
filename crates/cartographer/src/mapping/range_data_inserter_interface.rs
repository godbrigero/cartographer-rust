//! Port of `cartographer/mapping/range_data_inserter_interface` (.h, .cc).
//!
//! `create_range_data_inserter_options` is added by the 2D submaps port
//! (it needs the 2D inserter option factories).

use crate::mapping::grid_interface::GridInterface;
use crate::sensor::range_data::RangeData;

/// Inserts range data into a grid (C++ `RangeDataInserterInterface`).
pub trait RangeDataInserterInterface: Send + Sync {
    /// Inserts 'range_data' into 'grid'.
    fn insert(&self, range_data: &RangeData, grid: &mut dyn GridInterface);
}
