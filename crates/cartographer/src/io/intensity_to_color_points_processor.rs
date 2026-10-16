//! Port of `cartographer/io/intensity_to_color_points_processor` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::math::clamp;
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Applies `color = (intensity - min) / (max - min)` (clamped to `[0, 1]`,
/// gray) to all points with `frame_id` (or all points if `frame_id` is empty)
/// that have intensities.
pub struct IntensityToColorPointsProcessor {
    min_intensity: f32,
    max_intensity: f32,
    frame_id: String,
    next: Box<dyn PointsProcessor>,
}

impl IntensityToColorPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "intensity_to_color";

    /// Creates the processor. An empty `frame_id` applies to all batches.
    pub fn new(
        min_intensity: f32,
        max_intensity: f32,
        frame_id: &str,
        next: Box<dyn PointsProcessor>,
    ) -> Self {
        Self {
            min_intensity,
            max_intensity,
            frame_id: frame_id.to_string(),
            next,
        }
    }

    /// Creates the processor from `min_intensity`, `max_intensity` and the
    /// optional `frame_id`.
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let frame_id = if dictionary.has_key("frame_id") {
            dictionary.get_string("frame_id")
        } else {
            String::new()
        };
        let min_intensity = dictionary.get_double("min_intensity") as f32;
        let max_intensity = dictionary.get_double("max_intensity") as f32;
        Box::new(Self::new(min_intensity, max_intensity, &frame_id, next))
    }
}

impl PointsProcessor for IntensityToColorPointsProcessor {
    fn process(&mut self, mut batch: Box<PointsBatch>) {
        if !batch.intensities.is_empty()
            && (self.frame_id.is_empty() || batch.frame_id == self.frame_id)
        {
            let colors = batch
                .intensities
                .iter()
                .map(|&intensity| {
                    let gray = clamp(
                        (intensity - self.min_intensity)
                            / (self.max_intensity - self.min_intensity),
                        0.0,
                        1.0,
                    );
                    [gray, gray, gray]
                })
                .collect();
            batch.colors = colors;
        }
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        self.next.flush()
    }
}
