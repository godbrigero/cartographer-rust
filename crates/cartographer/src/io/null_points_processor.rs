//! Port of `cartographer/io/null_points_processor` (.h).

use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// A points processor that just drops all points. The end of a pipeline
/// usually.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullPointsProcessor;

impl NullPointsProcessor {
    /// Creates the processor.
    pub fn new() -> Self {
        Self
    }
}

impl PointsProcessor for NullPointsProcessor {
    fn process(&mut self, _points_batch: Box<PointsBatch>) {}

    fn flush(&mut self) -> FlushResult {
        FlushResult::Finished
    }
}
