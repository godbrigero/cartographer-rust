//! Port of `cartographer/io/xyz_writing_points_processor` (.h, .cc).

use nalgebra::Vector3;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::printf::format_g;
use crate::io::file_writer::{FileWriter, FileWriterFactory};
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};

/// Formats a point like `stream << std::setprecision(6) << x << " " << y <<
/// " " << z << "\n"`, i.e. C `%.6g` of the float promoted to double.
fn format_xyz_point(point: &Vector3<f32>) -> String {
    format!(
        "{} {} {}\n",
        format_g(f64::from(point.x), 6),
        format_g(f64::from(point.y), 6),
        format_g(f64::from(point.z), 6)
    )
}

fn write_xyz_point(point: &Vector3<f32>, file_writer: &mut dyn FileWriter) {
    let out = format_xyz_point(point);
    assert!(
        file_writer.write(out.as_bytes()),
        "Check failed: file_writer->Write(out.data(), out.size())"
    );
}

/// Writes ASCII xyz points.
pub struct XyzWriterPointsProcessor {
    next: Box<dyn PointsProcessor>,
    file_writer: Box<dyn FileWriter>,
}

impl XyzWriterPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "write_xyz";

    /// Creates the processor writing into `file_writer`.
    pub fn new(file_writer: Box<dyn FileWriter>, next: Box<dyn PointsProcessor>) -> Self {
        Self { next, file_writer }
    }

    /// Creates the processor writing to `filename`.
    pub fn from_dictionary(
        file_writer_factory: &FileWriterFactory,
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        Box::new(Self::new(
            file_writer_factory(&dictionary.get_string("filename")),
            next,
        ))
    }
}

impl PointsProcessor for XyzWriterPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        for point in &batch.points {
            write_xyz_point(&point.position, self.file_writer.as_mut());
        }
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        assert!(self.file_writer.close(), "Closing XYZ file failed.");
        match self.next.flush() {
            FlushResult::Finished => FlushResult::Finished,
            FlushResult::RestartStream => panic!(
                "XYZ generation must be configured to occur after any stages that require \
                 multiple passes."
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::io::fake_file_writer::FakeFileWriter;
    use crate::io::null_points_processor::NullPointsProcessor;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    #[test]
    fn number_formatting_matches_iostream_defaults() {
        let cases: [([f32; 3], &str); 6] = [
            ([0.0, 1.0, -2.5], "0 1 -2.5\n"),
            ([1e-7, 123456.7, 1234567.0], "1e-07 123457 1.23457e+06\n"),
            ([0.1, 0.0001, 0.00001], "0.1 0.0001 1e-05\n"),
            ([2.718_281_7, -0.333333, 100.0], "2.71828 -0.333333 100\n"),
            ([-0.0, 999999.4, 999999.6], "-0 999999 1e+06\n"),
            ([1.5e10, -2.25e-20, 42.0], "1.5e+10 -2.25e-20 42\n"),
        ];
        for (p, expected) in cases {
            assert_eq!(format_xyz_point(&Vector3::new(p[0], p[1], p[2])), expected);
        }
    }

    #[test]
    fn golden_output() {
        let content = Arc::new(Mutex::new(Vec::new()));
        let mut processor = XyzWriterPointsProcessor::new(
            Box::new(FakeFileWriter::new("test.xyz", Arc::clone(&content))),
            Box::new(NullPointsProcessor::new()),
        );
        processor.process(Box::new(PointsBatch {
            points: vec![
                RangefinderPoint {
                    position: Vector3::new(1.0, 2.0, 3.0),
                },
                RangefinderPoint {
                    position: Vector3::new(0.123_456_79, -1e-7, 1e7),
                },
            ],
            ..Default::default()
        }));
        assert_eq!(processor.flush(), FlushResult::Finished);
        assert_eq!(
            String::from_utf8(content.lock().unwrap().clone()).unwrap(),
            "1 2 3\n0.123457 -1e-07 1e+07\n"
        );
    }
}
