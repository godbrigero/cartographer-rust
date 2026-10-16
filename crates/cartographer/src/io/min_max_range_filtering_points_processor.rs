//! Port of `cartographer/io/min_max_range_filtering_points_processor` (.h, .cc).

use std::collections::HashSet;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::{PointsBatch, remove_points};
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Filters all points that are farther away from their `origin` than
/// `max_range` or closer than `min_range` (bounds inclusive).
pub struct MinMaxRangeFilteringPointsProcessor {
    min_range_squared: f64,
    max_range_squared: f64,
    next: Box<dyn PointsProcessor>,
}

impl MinMaxRangeFilteringPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "min_max_range_filter";

    /// Creates the processor.
    pub fn new(min_range: f64, max_range: f64, next: Box<dyn PointsProcessor>) -> Self {
        Self {
            min_range_squared: min_range * min_range,
            max_range_squared: max_range * max_range,
            next,
        }
    }

    /// Creates the processor from `min_range` and `max_range`.
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        Box::new(Self::new(
            dictionary.get_double("min_range"),
            dictionary.get_double("max_range"),
            next,
        ))
    }
}

impl PointsProcessor for MinMaxRangeFilteringPointsProcessor {
    fn process(&mut self, mut batch: Box<PointsBatch>) {
        let mut to_remove = HashSet::new();
        for (i, point) in batch.points.iter().enumerate() {
            let range_squared = f64::from((point.position - batch.origin).norm_squared());
            if !(self.min_range_squared <= range_squared && range_squared <= self.max_range_squared)
            {
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
