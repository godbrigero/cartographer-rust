//! Port of `cartographer/io/counting_points_processor` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Passes through points, but keeps track of how many points it saw and
/// output that on Flush.
pub struct CountingPointsProcessor {
    num_points: i64,
    next: Box<dyn PointsProcessor>,
}

impl CountingPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "dump_num_points";

    /// Creates the processor.
    pub fn new(next: Box<dyn PointsProcessor>) -> Self {
        Self {
            num_points: 0,
            next,
        }
    }

    /// Creates the processor (no options).
    pub fn from_dictionary(
        _dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        Box::new(Self::new(next))
    }

    /// Number of points seen in the current pass.
    pub fn num_points(&self) -> i64 {
        self.num_points
    }
}

impl PointsProcessor for CountingPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        self.num_points += batch.points.len() as i64;
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        match self.next.flush() {
            FlushResult::Finished => {
                log::info!("Processed {} and finishing.", self.num_points);
                FlushResult::Finished
            }
            FlushResult::RestartStream => {
                log::info!("Processed {} and restarting stream.", self.num_points);
                self.num_points = 0;
                FlushResult::RestartStream
            }
        }
    }
}
