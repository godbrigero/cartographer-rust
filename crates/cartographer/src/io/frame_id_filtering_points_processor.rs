//! Port of `cartographer/io/frame_id_filtering_points_processor` (.h, .cc).

use std::collections::HashSet;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Filters all points with blacklisted `frame_id` or a non-whitelisted
/// `frame_id`. Batches that do not pass are dropped entirely.
pub struct FrameIdFilteringPointsProcessor {
    keep_frame_ids: HashSet<String>,
    drop_frame_ids: HashSet<String>,
    next: Box<dyn PointsProcessor>,
}

impl FrameIdFilteringPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "frame_id_filter";

    /// Creates the processor. Exactly one of the sets must be non-empty.
    pub fn new(
        keep_frame_ids: HashSet<String>,
        drop_frame_ids: HashSet<String>,
        next: Box<dyn PointsProcessor>,
    ) -> Self {
        assert_ne!(
            keep_frame_ids.is_empty(),
            drop_frame_ids.is_empty(),
            "You have to specify exactly one of the `keep_frames` property or the \
             `drop_frames` property, but not both at the same time."
        );
        Self {
            keep_frame_ids,
            drop_frame_ids,
            next,
        }
    }

    /// Creates the processor from the optional `keep_frames` / `drop_frames`
    /// string arrays.
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let mut keep_frames = Vec::new();
        let mut drop_frames = Vec::new();
        if dictionary.has_key("keep_frames") {
            keep_frames = dictionary
                .get_dictionary("keep_frames")
                .get_array_values_as_strings();
        }
        if dictionary.has_key("drop_frames") {
            drop_frames = dictionary
                .get_dictionary("drop_frames")
                .get_array_values_as_strings();
        }
        Box::new(Self::new(
            keep_frames.into_iter().collect(),
            drop_frames.into_iter().collect(),
            next,
        ))
    }
}

impl PointsProcessor for FrameIdFilteringPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        if (!self.keep_frame_ids.is_empty() && self.keep_frame_ids.contains(&batch.frame_id))
            || (!self.drop_frame_ids.is_empty() && !self.drop_frame_ids.contains(&batch.frame_id))
        {
            self.next.process(batch);
        }
    }

    fn flush(&mut self) -> FlushResult {
        self.next.flush()
    }
}
