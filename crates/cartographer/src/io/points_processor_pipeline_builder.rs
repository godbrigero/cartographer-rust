//! Port of `cartographer/io/points_processor_pipeline_builder` (.h, .cc).

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::coloring_points_processor::ColoringPointsProcessor;
use crate::io::counting_points_processor::CountingPointsProcessor;
use crate::io::file_writer::FileWriterFactory;
use crate::io::fixed_ratio_sampling_points_processor::FixedRatioSamplingPointsProcessor;
use crate::io::frame_id_filtering_points_processor::FrameIdFilteringPointsProcessor;
use crate::io::hybrid_grid_points_processor::HybridGridPointsProcessor;
use crate::io::intensity_to_color_points_processor::IntensityToColorPointsProcessor;
use crate::io::min_max_range_filtering_points_processor::MinMaxRangeFilteringPointsProcessor;
use crate::io::null_points_processor::NullPointsProcessor;
use crate::io::outlier_removing_points_processor::OutlierRemovingPointsProcessor;
use crate::io::pcd_writing_points_processor::PcdWritingPointsProcessor;
use crate::io::ply_writing_points_processor::PlyWritingPointsProcessor;
use crate::io::points_processor::PointsProcessor;
use crate::io::probability_grid_points_processor::ProbabilityGridPointsProcessor;
use crate::io::vertical_range_filtering_points_processor::VerticalRangeFilteringPointsProcessor;
use crate::io::xray_points_processor::XRayPointsProcessor;
use crate::io::xyz_writing_points_processor::XyzWriterPointsProcessor;
use crate::mapping::proto;

/// Creates a processor from its Lua configuration and the processor it
/// should hand its batches to.
pub type FactoryFunction =
    Box<dyn Fn(&LuaParameterDictionary, Box<dyn PointsProcessor>) -> Box<dyn PointsProcessor>>;

/// Builder to create a points processor pipeline out of a Lua configuration.
/// You can register all built-in PointsProcessors using
/// [`register_built_in_points_processors`]. Non-built-in PointsProcessors
/// must define a name and a factory method for building itself from a
/// LuaParameterDictionary. See the various built-in PointsProcessors for
/// examples.
#[derive(Default)]
pub struct PointsProcessorPipelineBuilder {
    factories: HashMap<String, FactoryFunction>,
}

impl PointsProcessorPipelineBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new PointsProcessor type uniquely identified by `name`
    /// which will be created using `factory`.
    pub fn register(&mut self, name: &str, factory: FactoryFunction) {
        assert!(
            !self.factories.contains_key(name),
            "A points processor named '{name}' has already been registered."
        );
        self.factories.insert(name.to_string(), factory);
    }

    /// Whether a processor named `name` has been registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Creates the pipeline described by `dictionary`, an array of tables
    /// each with an `action` naming a registered processor.
    ///
    /// RUST PORT NOTE: C++ returns the vector of all processors (index 0
    /// being the final `NullPointsProcessor`); callers feed
    /// `pipeline.back()`. Since processors own their successor here, only
    /// that head (the processor of the *first* configuration entry) is
    /// returned.
    pub fn create_pipeline(&self, dictionary: &LuaParameterDictionary) -> Box<dyn PointsProcessor> {
        // The last consumer in the pipeline must exist, so that the one
        // created after it (and being before it in the pipeline) has a valid
        // 'next' to point to. The last consumer will just drop all points.
        let mut pipeline: Box<dyn PointsProcessor> = Box::new(NullPointsProcessor::new());

        let configurations = dictionary.get_array_values_as_dictionaries();

        // We construct the pipeline starting at the back.
        for configuration in configurations.iter().rev() {
            let action = configuration.get_string("action");
            let factory = self.factories.get(&action).unwrap_or_else(|| {
                panic!(
                    "Unknown action '{action}'. Did you register the correspoinding \
                     PointsProcessor?"
                )
            });
            pipeline = factory(configuration, pipeline);
        }
        pipeline
    }
}

fn register_plain_points_processor(
    builder: &mut PointsProcessorPipelineBuilder,
    name: &str,
    from_dictionary: fn(
        &LuaParameterDictionary,
        Box<dyn PointsProcessor>,
    ) -> Box<dyn PointsProcessor>,
) {
    builder.register(name, Box::new(from_dictionary));
}

/// `FromDictionary` of a file writing processor.
type FileWritingFromDictionary = fn(
    &FileWriterFactory,
    &LuaParameterDictionary,
    Box<dyn PointsProcessor>,
) -> Box<dyn PointsProcessor>;

fn register_file_writing_points_processor(
    file_writer_factory: &FileWriterFactory,
    builder: &mut PointsProcessorPipelineBuilder,
    name: &str,
    from_dictionary: FileWritingFromDictionary,
) {
    let file_writer_factory = Arc::clone(file_writer_factory);
    builder.register(
        name,
        Box::new(move |dictionary, next| from_dictionary(&file_writer_factory, dictionary, next)),
    );
}

/// Register all `PointsProcessor`s that ship with Cartographer with this
/// `builder`.
///
/// RUST PORT NOTE: C++ captures `trajectories` by reference (the caller must
/// keep the vector alive); here they are shared through an `Arc`.
pub fn register_built_in_points_processors(
    trajectories: Arc<Vec<proto::Trajectory>>,
    file_writer_factory: &FileWriterFactory,
    builder: &mut PointsProcessorPipelineBuilder,
) {
    register_plain_points_processor(
        builder,
        CountingPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| CountingPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        FixedRatioSamplingPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| FixedRatioSamplingPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        FrameIdFilteringPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| FrameIdFilteringPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        MinMaxRangeFilteringPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| MinMaxRangeFilteringPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        VerticalRangeFilteringPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| VerticalRangeFilteringPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        OutlierRemovingPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| OutlierRemovingPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        ColoringPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| ColoringPointsProcessor::from_dictionary(d, n),
    );
    register_plain_points_processor(
        builder,
        IntensityToColorPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |d, n| IntensityToColorPointsProcessor::from_dictionary(d, n),
    );
    register_file_writing_points_processor(
        file_writer_factory,
        builder,
        PcdWritingPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |f, d, n| PcdWritingPointsProcessor::from_dictionary(f, d, n),
    );
    register_file_writing_points_processor(
        file_writer_factory,
        builder,
        PlyWritingPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |f, d, n| PlyWritingPointsProcessor::from_dictionary(f, d, n),
    );
    register_file_writing_points_processor(
        file_writer_factory,
        builder,
        XyzWriterPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |f, d, n| XyzWriterPointsProcessor::from_dictionary(f, d, n),
    );
    register_file_writing_points_processor(
        file_writer_factory,
        builder,
        HybridGridPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
        |f, d, n| HybridGridPointsProcessor::from_dictionary(f, d, n),
    );
    {
        let trajectories = Arc::clone(&trajectories);
        let file_writer_factory = Arc::clone(file_writer_factory);
        builder.register(
            XRayPointsProcessor::CONFIGURATION_FILE_ACTION_NAME,
            Box::new(move |dictionary, next| {
                XRayPointsProcessor::from_dictionary(
                    Arc::clone(&trajectories),
                    Arc::clone(&file_writer_factory),
                    dictionary,
                    next,
                )
            }),
        );
    }
    {
        let file_writer_factory = Arc::clone(file_writer_factory);
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
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use nalgebra::Vector3;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::io::fake_file_writer::FakeFileWriter;
    use crate::io::file_writer::FileWriter;
    use crate::io::points_batch::PointsBatch;
    use crate::io::points_processor::FlushResult;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    mod points_processor_pipeline_builder_test {
        use super::*;

        #[test]
        fn register_built_in_points_processors() {
            let mut builder = PointsProcessorPipelineBuilder::new();
            // RUST PORT NOTE: the C++ dummy factory returns nullptr; a
            // FileWriter cannot be null here, so return a fake one.
            let dummy_factory: FileWriterFactory = Arc::new(|filename: &str| {
                Box::new(FakeFileWriter::new(
                    filename,
                    Arc::new(Mutex::new(Vec::new())),
                )) as Box<dyn FileWriter>
            });
            crate::io::points_processor_pipeline_builder::register_built_in_points_processors(
                Arc::new(Vec::new()),
                &dummy_factory,
                &mut builder,
            );
        }
    }

    #[test]
    fn all_built_in_names_are_registered() {
        let mut builder = PointsProcessorPipelineBuilder::new();
        register_built_in_points_processors(
            Arc::new(Vec::new()),
            &FakeFileWriter::factory(Arc::new(Mutex::new(Vec::new()))),
            &mut builder,
        );
        for name in [
            "dump_num_points",
            "fixed_ratio_sampler",
            "frame_id_filter",
            "min_max_range_filter",
            "vertical_range_filter",
            "voxel_filter_and_remove_moving_objects",
            "color_points",
            "intensity_to_color",
            "write_pcd",
            "write_ply",
            "write_xyz",
            "write_hybrid_grid",
            "write_xray_image",
            "write_probability_grid",
        ] {
            assert!(builder.is_registered(name), "{name}");
        }
    }

    #[test]
    #[should_panic(expected = "has already been registered")]
    fn duplicate_registration_panics() {
        let mut builder = PointsProcessorPipelineBuilder::new();
        let factory = || -> FactoryFunction { Box::new(|_, next| next) };
        builder.register("a", factory());
        builder.register("a", factory());
    }

    #[test]
    #[should_panic(expected = "Unknown action 'nope'")]
    fn unknown_action_panics() {
        let builder = PointsProcessorPipelineBuilder::new();
        let dictionary = make_dictionary(r#"return { { action = "nope" } }"#);
        builder.create_pipeline(&dictionary);
    }

    #[test]
    fn pipeline_runs_stages_in_configuration_order() {
        let content = Arc::new(Mutex::new(Vec::new()));
        let mut builder = PointsProcessorPipelineBuilder::new();
        register_built_in_points_processors(
            Arc::new(Vec::new()),
            &FakeFileWriter::factory(Arc::clone(&content)),
            &mut builder,
        );
        // Color first, then filter far points, then write xyz: if the stages
        // ran in reverse order the far point would be written.
        let dictionary = make_dictionary(
            r#"return {
              { action = "dump_num_points" },
              { action = "min_max_range_filter", min_range = 0.5, max_range = 5. },
              { action = "color_points", frame_id = "laser", color = { 255, 0, 0 } },
              { action = "write_xyz", filename = "out.xyz" },
            }"#,
        );
        let mut pipeline = builder.create_pipeline(&dictionary);
        let mut passes = 0;
        loop {
            passes += 1;
            pipeline.process(Box::new(PointsBatch {
                frame_id: "laser".to_string(),
                points: vec![
                    RangefinderPoint {
                        position: Vector3::new(1., 2., 0.5),
                    },
                    RangefinderPoint {
                        position: Vector3::new(10., 0., 0.),
                    },
                ],
                ..Default::default()
            }));
            if pipeline.flush() == FlushResult::Finished {
                break;
            }
        }
        assert_eq!(passes, 1);
        assert_eq!(
            String::from_utf8(content.lock().unwrap().clone()).unwrap(),
            "1 2 0.5\n"
        );
    }
}
