//! Port of `cartographer/io/probability_grid_points_processor` (.h, .cc).

use std::sync::Arc;

use nalgebra::Vector2;
use prost::Message;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::port::round_to_int;
use crate::io::cairo::Surface;
use crate::io::color::get_color;
use crate::io::draw_trajectories::draw_trajectory;
use crate::io::file_writer::{FileWriter, FileWriterFactory};
use crate::io::image::Image;
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};
use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY};
use crate::mapping::proto;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::probability_grid::ProbabilityGrid;
use crate::mapping::two_d::probability_grid_range_data_inserter_2d::{
    ProbabilityGridRangeDataInserter2D, create_probability_grid_range_data_inserter_options_2d,
};
use crate::mapping::two_d::xy_index::{Array2i, CellLimits, XYIndexRangeIterator};
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::range_data::RangeData;
use crate::transform::rigid_transform::Rigid3d;

/// Whether to draw the trajectories into the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawTrajectories {
    /// Do not draw trajectories.
    No,
    /// Draw trajectories.
    Yes,
}

/// Output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputType {
    /// A PNG image.
    Png,
    /// A serialized `mapping::proto::Grid2D`.
    Pb,
}

fn draw_trajectories_into_image(
    probability_grid: &ProbabilityGrid,
    offset: &Array2i,
    trajectories: &[proto::Trajectory],
    cairo_surface: &mut Surface,
) {
    for (i, trajectory) in trajectories.iter().enumerate() {
        draw_trajectory(
            trajectory,
            &get_color(i as i32),
            &|pose: &Rigid3d| -> Vector2<i32> {
                let translation = pose.cast::<f32>().translation().xy();
                probability_grid.limits().get_cell_index(&translation) - offset
            },
            cairo_surface,
        );
    }
}

fn probability_to_color(probability_from_grid: f32) -> u8 {
    let probability = 1.0 - probability_from_grid;
    round_to_int(255.0 * ((probability - MIN_PROBABILITY) / (MAX_PROBABILITY - MIN_PROBABILITY)))
        as u8
}

fn file_extension_from_output_type(output_type: OutputType) -> &'static str {
    match output_type {
        OutputType::Png => ".png",
        OutputType::Pb => ".pb",
    }
}

fn output_type_from_string(output_type: &str) -> OutputType {
    match output_type {
        "png" => OutputType::Png,
        "pb" => OutputType::Pb,
        _ => panic!("OutputType {output_type} does not exist!"),
    }
}

/// Creates a probability grid with the specified `resolution`. As all points
/// are projected into the x-y plane the z component of the data is ignored.
/// `range_data_inserter` options are used to configure the range data ray
/// tracing through the probability grid.
pub struct ProbabilityGridPointsProcessor {
    draw_trajectories: DrawTrajectories,
    output_type: OutputType,
    trajectories: Arc<Vec<proto::Trajectory>>,
    file_writer: Box<dyn FileWriter>,
    next: Box<dyn PointsProcessor>,
    range_data_inserter: ProbabilityGridRangeDataInserter2D,
    // Kept alive for the grid (C++ member `conversion_tables_`).
    _conversion_tables: Arc<ValueConversionTables>,
    probability_grid: ProbabilityGrid,
}

impl ProbabilityGridPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "write_probability_grid";

    /// Creates the processor.
    pub fn new(
        resolution: f64,
        probability_grid_range_data_inserter_options: &proto::ProbabilityGridRangeDataInserterOptions2D,
        draw_trajectories: DrawTrajectories,
        output_type: OutputType,
        file_writer: Box<dyn FileWriter>,
        trajectories: Arc<Vec<proto::Trajectory>>,
        next: Box<dyn PointsProcessor>,
    ) -> Self {
        let conversion_tables = Arc::new(ValueConversionTables::new());
        let probability_grid = create_probability_grid(resolution, &conversion_tables);
        if output_type == OutputType::Pb && draw_trajectories == DrawTrajectories::Yes {
            log::warn!(
                "Drawing the trajectories is not supported when writing the probability grid \
                 as protobuf."
            );
        }
        Self {
            draw_trajectories,
            output_type,
            trajectories,
            file_writer,
            next,
            range_data_inserter: ProbabilityGridRangeDataInserter2D::new(
                probability_grid_range_data_inserter_options,
            ),
            _conversion_tables: conversion_tables,
            probability_grid,
        }
    }

    /// Creates the processor from `resolution`, `range_data_inserter`,
    /// `filename` and the optional `draw_trajectories` (default true) and
    /// `output_type` (`"png"` (default) or `"pb"`).
    ///
    /// RUST PORT NOTE: the C++ registration captures `trajectories` by
    /// reference; here they are shared through an `Arc`.
    pub fn from_dictionary(
        trajectories: Arc<Vec<proto::Trajectory>>,
        file_writer_factory: &FileWriterFactory,
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let draw_trajectories = if !dictionary.has_key("draw_trajectories")
            || dictionary.get_bool("draw_trajectories")
        {
            DrawTrajectories::Yes
        } else {
            DrawTrajectories::No
        };
        let output_type = if dictionary.has_key("output_type") {
            output_type_from_string(&dictionary.get_string("output_type"))
        } else {
            OutputType::Png
        };
        Box::new(Self::new(
            dictionary.get_double("resolution"),
            &create_probability_grid_range_data_inserter_options_2d(
                &dictionary.get_dictionary("range_data_inserter"),
            ),
            draw_trajectories,
            output_type,
            file_writer_factory(&format!(
                "{}{}",
                dictionary.get_string("filename"),
                file_extension_from_output_type(output_type)
            )),
            trajectories,
            next,
        ))
    }
}

impl PointsProcessor for ProbabilityGridPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        self.range_data_inserter.insert_into_probability_grid(
            &RangeData {
                origin: batch.origin,
                returns: PointCloud::from_points(batch.points.clone()),
                misses: PointCloud::new(),
            },
            &mut self.probability_grid,
        );
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        match self.output_type {
            OutputType::Png => {
                if let Some((mut image, offset)) = draw_probability_grid(&self.probability_grid) {
                    if self.draw_trajectories == DrawTrajectories::Yes {
                        draw_trajectories_into_image(
                            &self.probability_grid,
                            &offset,
                            &self.trajectories,
                            image.get_cairo_surface(),
                        );
                    }
                    image.write_png(self.file_writer.as_mut());
                    assert!(
                        self.file_writer.close(),
                        "Check failed: file_writer_->Close()"
                    );
                }
            }
            OutputType::Pb => {
                let probability_grid_proto = self.probability_grid.to_proto();
                let serialized = probability_grid_proto.encode_to_vec();
                self.file_writer.write(&serialized);
                assert!(
                    self.file_writer.close(),
                    "Check failed: file_writer_->Close()"
                );
            }
        }

        match self.next.flush() {
            FlushResult::RestartStream => panic!(
                "ProbabilityGrid generation must be configured to occur after any stages that \
                 require multiple passes."
            ),
            FlushResult::Finished => FlushResult::Finished,
        }
    }
}

/// Draws `probability_grid` into an image and returns it together with the
/// offset of the cropped map limits. Returns `None` if the cropped limits are
/// empty.
///
/// Note: as in C++, an empty grid has 1x1 cropped limits, so this returns a
/// 1x1 image with the unknown value rather than `None`.
pub fn draw_probability_grid(probability_grid: &ProbabilityGrid) -> Option<(Image, Array2i)> {
    let (offset, cell_limits) = probability_grid.compute_cropped_limits();
    if cell_limits.num_x_cells == 0 || cell_limits.num_y_cells == 0 {
        log::warn!("Not writing output: empty probability grid");
        return None;
    }
    let mut image = Image::new(cell_limits.num_x_cells, cell_limits.num_y_cells);
    for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
        let index = xy_index + offset;
        const UNKNOWN_VALUE: u8 = 128;
        let value = if probability_grid.is_known(&index) {
            probability_to_color(probability_grid.get_probability(&index))
        } else {
            UNKNOWN_VALUE
        };
        image.set_pixel(xy_index.x, xy_index.y, &[value, value, value]);
    }
    Some((image, offset))
}

/// Create an initially empty probability grid with `resolution` and a small
/// size, suitable for a PointsBatchProcessor.
pub fn create_probability_grid(
    resolution: f64,
    conversion_tables: &Arc<ValueConversionTables>,
) -> ProbabilityGrid {
    const INITIAL_PROBABILITY_GRID_SIZE: i32 = 100;
    let max: Vector2<f64> =
        0.5 * f64::from(INITIAL_PROBABILITY_GRID_SIZE) * resolution * Vector2::new(1., 1.);
    ProbabilityGrid::new(
        MapLimits::new(
            resolution,
            max,
            CellLimits::new(INITIAL_PROBABILITY_GRID_SIZE, INITIAL_PROBABILITY_GRID_SIZE),
        ),
        conversion_tables,
    )
}

#[cfg(test)]
mod tests {
    mod probability_grid_points_processor_test {
        use std::sync::{Arc, Mutex};

        use nalgebra::Vector3;
        use prost::Message;

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::DummyFileResolver;
        use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
        use crate::io::fake_file_writer::FakeFileWriter;
        use crate::io::file_writer::{FileWriter, FileWriterFactory};
        use crate::io::points_batch::PointsBatch;
        use crate::io::points_processor::{FlushResult, PointsProcessor};
        use crate::io::points_processor_pipeline_builder::PointsProcessorPipelineBuilder;
        use crate::io::probability_grid_points_processor::*;
        use crate::mapping::proto;
        use crate::mapping::two_d::grid_2d::Grid2DTrait;
        use crate::mapping::two_d::probability_grid_range_data_inserter_2d::{
            ProbabilityGridRangeDataInserter2D,
            create_probability_grid_range_data_inserter_options_2d,
        };
        use crate::mapping::value_conversion_tables::ValueConversionTables;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        fn create_points_batch() -> Box<PointsBatch> {
            let mut points_batch = Box::new(PointsBatch::default());
            points_batch.origin = Vector3::new(0., 0., 0.);
            for p in [
                [0.0f32, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 4.0],
                [0.0, 3.0, 5.0],
                [3.0, 0.0, 6.0],
            ] {
                points_batch.points.push(RangefinderPoint {
                    position: Vector3::new(p[0], p[1], p[2]),
                });
            }
            points_batch
        }

        fn create_fake_file_writer_factory(
            expected_filename: &str,
            fake_file_writer_output: Arc<Mutex<Vec<u8>>>,
        ) -> FileWriterFactory {
            let expected_filename = expected_filename.to_string();
            Arc::new(move |full_filename: &str| {
                assert_eq!(expected_filename, full_filename);
                Box::new(FakeFileWriter::new(
                    full_filename,
                    Arc::clone(&fake_file_writer_output),
                )) as Box<dyn FileWriter>
            })
        }

        fn create_pipeline_from_dictionary(
            pipeline_dictionary: &LuaParameterDictionary,
            trajectories: Arc<Vec<proto::Trajectory>>,
            file_writer_factory: FileWriterFactory,
        ) -> Box<dyn PointsProcessor> {
            let mut builder = PointsProcessorPipelineBuilder::new();
            builder.register(
                ProbabilityGridPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
                Box::new(move |dictionary, next| {
                    ProbabilityGridPointsProcessor::from_dictionary(
                        Arc::clone(&trajectories),
                        &file_writer_factory,
                        dictionary,
                        next,
                    )
                }),
            );
            builder.create_pipeline(pipeline_dictionary)
        }

        fn create_expected_probability_grid(
            points_batch: &PointsBatch,
            probability_grid_options: &LuaParameterDictionary,
        ) -> Vec<u8> {
            let range_data_inserter = ProbabilityGridRangeDataInserter2D::new(
                &create_probability_grid_range_data_inserter_options_2d(
                    &probability_grid_options.get_dictionary("range_data_inserter"),
                ),
            );
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut probability_grid = create_probability_grid(
                probability_grid_options.get_double("resolution"),
                &conversion_tables,
            );
            range_data_inserter.insert_into_probability_grid(
                &RangeData {
                    origin: points_batch.origin,
                    returns: PointCloud::from_points(points_batch.points.clone()),
                    misses: PointCloud::new(),
                },
                &mut probability_grid,
            );
            probability_grid.to_proto().encode_to_vec()
        }

        fn create_parameter_dictionary() -> LuaParameterDictionary {
            LuaParameterDictionary::non_reference_counted(
                r#"
          pipeline = {
            {
              action = "write_probability_grid",
              resolution = 0.05,
              range_data_inserter = {
                insert_free_space = true,
                hit_probability = 0.55,
                miss_probability = 0.49,
              },
              draw_trajectories = false,
              output_type = "pb",
              filename = "map"
            }
          }
          return pipeline
    "#,
                Box::new(DummyFileResolver),
            )
        }

        struct ProbabilityGridPointsProcessorTest {
            fake_file_writer_output: Arc<Mutex<Vec<u8>>>,
            pipeline_dictionary: LuaParameterDictionary,
            dummy_trajectories: Arc<Vec<proto::Trajectory>>,
        }

        impl ProbabilityGridPointsProcessorTest {
            fn new() -> Self {
                Self {
                    fake_file_writer_output: Arc::new(Mutex::new(Vec::new())),
                    pipeline_dictionary: create_parameter_dictionary(),
                    dummy_trajectories: Arc::new(Vec::new()),
                }
            }

            fn run(&self, expected_filename: &str) {
                let mut pipeline = create_pipeline_from_dictionary(
                    &self.pipeline_dictionary,
                    Arc::clone(&self.dummy_trajectories),
                    create_fake_file_writer_factory(
                        expected_filename,
                        Arc::clone(&self.fake_file_writer_output),
                    ),
                );
                loop {
                    pipeline.process(create_points_batch());
                    if pipeline.flush() != FlushResult::RestartStream {
                        break;
                    }
                }
            }
        }

        #[test]
        fn write_proto() {
            let test = ProbabilityGridPointsProcessorTest::new();
            let expected_prob_grid_proto = create_expected_probability_grid(
                &create_points_batch(),
                &test
                    .pipeline_dictionary
                    .get_array_values_as_dictionaries()
                    .remove(0),
            );
            test.run("map.pb");
            assert_eq!(
                *test.fake_file_writer_output.lock().unwrap(),
                expected_prob_grid_proto
            );
            // Sanity check: the output is a non-trivial Grid2D.
            let grid = proto::Grid2D::decode(expected_prob_grid_proto.as_slice()).unwrap();
            assert!(grid.cells.len() >= 100 * 100);
        }

        #[test]
        fn write_png() {
            let output = Arc::new(Mutex::new(Vec::new()));
            let dictionary = LuaParameterDictionary::non_reference_counted(
                r#"return {
                  {
                    action = "write_probability_grid",
                    resolution = 0.05,
                    range_data_inserter = {
                      insert_free_space = true,
                      hit_probability = 0.55,
                      miss_probability = 0.49,
                    },
                    filename = "map",
                  }
                }"#,
                Box::new(DummyFileResolver),
            );
            let trajectory = proto::Trajectory {
                node: vec![proto::trajectory::Node {
                    pose: Some(crate::transform::transform::to_proto(
                        &crate::transform::rigid_transform::Rigid3d::identity(),
                    )),
                    ..Default::default()
                }],
                ..Default::default()
            };
            let mut pipeline = create_pipeline_from_dictionary(
                &dictionary,
                Arc::new(vec![trajectory]),
                create_fake_file_writer_factory("map.png", Arc::clone(&output)),
            );
            pipeline.process(create_points_batch());
            assert_eq!(pipeline.flush(), FlushResult::Finished);
            let bytes = output.lock().unwrap().clone();
            let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            // Points span 3 m in x and y at 5 cm resolution.
            assert!(info.width >= 60 && info.height >= 60);
        }

        #[test]
        fn draw_probability_grid_of_empty_grid_is_1x1_unknown() {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let grid = create_probability_grid(0.05, &conversion_tables);
            let (image, offset) = draw_probability_grid(&grid).unwrap();
            assert_eq!((image.width(), image.height()), (1, 1));
            assert_eq!(offset, nalgebra::Vector2::new(0, 0));
            assert_eq!(image.get_pixel(0, 0), [128, 128, 128]);
        }
    }
}
