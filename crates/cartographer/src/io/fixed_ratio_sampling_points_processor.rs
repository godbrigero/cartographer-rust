//! Port of `cartographer/io/fixed_ratio_sampling_points_processor` (.h, .cc).

use std::collections::HashSet;

use crate::common::fixed_ratio_sampler::FixedRatioSampler;
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::{PointsBatch, remove_points};
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Only let a fixed `sampling_ratio` of points through. A `sampling_ratio`
/// of 1. makes this filter a no-op.
pub struct FixedRatioSamplingPointsProcessor {
    sampling_ratio: f64,
    next: Box<dyn PointsProcessor>,
    sampler: FixedRatioSampler,
}

impl FixedRatioSamplingPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "fixed_ratio_sampler";

    /// Creates the processor.
    pub fn new(sampling_ratio: f64, next: Box<dyn PointsProcessor>) -> Self {
        Self {
            sampling_ratio,
            next,
            sampler: FixedRatioSampler::new(sampling_ratio),
        }
    }

    /// Creates the processor from `sampling_ratio`, which must be in (0, 1).
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let sampling_ratio = dictionary.get_double("sampling_ratio");
        assert!(
            0. < sampling_ratio,
            "Check failed: 0. < sampling_ratio Sampling ratio <= 0 makes no sense."
        );
        assert!(
            sampling_ratio < 1.,
            "Check failed: sampling_ratio < 1. Sampling ratio >= 1 makes no sense."
        );
        Box::new(Self::new(sampling_ratio, next))
    }
}

impl PointsProcessor for FixedRatioSamplingPointsProcessor {
    fn process(&mut self, mut batch: Box<PointsBatch>) {
        let mut to_remove = HashSet::new();
        for i in 0..batch.points.len() {
            if !self.sampler.pulse() {
                to_remove.insert(i);
            }
        }
        remove_points(&to_remove, &mut batch);
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        match self.next.flush() {
            FlushResult::Finished => FlushResult::Finished,
            FlushResult::RestartStream => {
                self.sampler = FixedRatioSampler::new(self.sampling_ratio);
                FlushResult::RestartStream
            }
        }
    }
}
