//! Port of `cartographer/io/coloring_points_processor` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::color::{FloatColor, Uint8Color, to_float_color};
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Colors points with a fixed color by frame_id.
pub struct ColoringPointsProcessor {
    color: FloatColor,
    frame_id: String,
    next: Box<dyn PointsProcessor>,
}

impl ColoringPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "color_points";

    /// Creates the processor.
    pub fn new(color: FloatColor, frame_id: &str, next: Box<dyn PointsProcessor>) -> Self {
        Self {
            color,
            frame_id: frame_id.to_string(),
            next,
        }
    }

    /// Creates the processor from `frame_id` and `color` (array of three
    /// numbers in `[0, 255]`, truncated to integers).
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let frame_id = dictionary.get_string("frame_id");
        let color_values = dictionary
            .get_dictionary("color")
            .get_array_values_as_doubles();
        // `static_cast<uint8>(double)`: truncation (Rust `as` additionally
        // saturates out-of-range values, which is UB in C++).
        let color: Uint8Color = [
            color_values[0] as u8,
            color_values[1] as u8,
            color_values[2] as u8,
        ];
        Box::new(Self::new(to_float_color(&color), &frame_id, next))
    }
}

impl PointsProcessor for ColoringPointsProcessor {
    fn process(&mut self, mut batch: Box<PointsBatch>) {
        if batch.frame_id == self.frame_id {
            batch.colors.clear();
            batch.colors.resize(batch.points.len(), self.color);
        }
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        self.next.flush()
    }
}
