//! Port of `cartographer/io/vertical_range_filtering_points_processor` (.h, .cc).

use std::collections::HashSet;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::{PointsBatch, remove_points};
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Filters all points whose z-distance from their `origin` is not in
/// `[min_z, max_z]`.
pub struct VerticalRangeFilteringPointsProcessor {
    min_z: f64,
    max_z: f64,
    next: Box<dyn PointsProcessor>,
}

impl VerticalRangeFilteringPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "vertical_range_filter";

    /// Creates the processor.
    pub fn new(min_z: f64, max_z: f64, next: Box<dyn PointsProcessor>) -> Self {
        Self { min_z, max_z, next }
    }

    /// Creates the processor from `min_z` and `max_z`.
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        Box::new(Self::new(
            dictionary.get_double("min_z"),
            dictionary.get_double("max_z"),
            next,
        ))
    }
}

impl PointsProcessor for VerticalRangeFilteringPointsProcessor {
    fn process(&mut self, mut batch: Box<PointsBatch>) {
        let mut to_remove = HashSet::new();
        for (i, point) in batch.points.iter().enumerate() {
            let distance_z = f64::from(point.position.z - batch.origin.z);
            if !(self.min_z <= distance_z && distance_z <= self.max_z) {
                to_remove.insert(i);
            }
        }
        remove_points(&to_remove, &mut batch);
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        self.next.flush()
    }
}
