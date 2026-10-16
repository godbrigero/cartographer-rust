//! Port of `cartographer/io/hybrid_grid_points_processor` (.h, .cc).

use prost::Message;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::file_writer::{FileWriter, FileWriterFactory};
use crate::io::points_batch::PointsBatch;
use crate::io::points_processor::{FlushResult, PointsProcessor};
use crate::mapping::proto;
use crate::mapping::three_d::hybrid_grid::HybridGrid;
use crate::mapping::three_d::range_data_inserter_3d::{
    RangeDataInserter3D, create_range_data_inserter_options_3d,
};
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::range_data::RangeData;

/// Creates a hybrid grid of the points with voxels being `voxel_size` big.
/// `range_data_inserter` options are used to configure the range data ray
/// tracing through the hybrid grid. The grid is written as a raw serialized
/// `mapping::proto::HybridGrid`.
pub struct HybridGridPointsProcessor {
    next: Box<dyn PointsProcessor>,
    range_data_inserter: RangeDataInserter3D,
    hybrid_grid: HybridGrid,
    file_writer: Box<dyn FileWriter>,
}

impl HybridGridPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str = "write_hybrid_grid";

    /// Creates the processor.
    pub fn new(
        voxel_size: f64,
        range_data_inserter_options: &proto::RangeDataInserterOptions3D,
        file_writer: Box<dyn FileWriter>,
        next: Box<dyn PointsProcessor>,
    ) -> Self {
        Self {
            next,
            range_data_inserter: RangeDataInserter3D::new(range_data_inserter_options),
            hybrid_grid: HybridGrid::new(voxel_size as f32),
            file_writer,
        }
    }

    /// Creates the processor from `voxel_size`, `range_data_inserter` and
    /// `filename`.
    pub fn from_dictionary(
        file_writer_factory: &FileWriterFactory,
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        Box::new(Self::new(
            dictionary.get_double("voxel_size"),
            &create_range_data_inserter_options_3d(
                &dictionary.get_dictionary("range_data_inserter"),
            ),
            file_writer_factory(&dictionary.get_string("filename")),
            next,
        ))
    }
}

impl PointsProcessor for HybridGridPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        self.range_data_inserter.insert(
            &RangeData {
                origin: batch.origin,
                returns: PointCloud::from_points(batch.points.clone()),
                misses: PointCloud::new(),
            },
            &mut self.hybrid_grid,
            /*intensity_hybrid_grid=*/ None,
        );
        self.next.process(batch);
    }

    fn flush(&mut self) -> FlushResult {
        let hybrid_grid_proto = self.hybrid_grid.to_proto();
        let serialized = hybrid_grid_proto.encode_to_vec();
        self.file_writer.write(&serialized);
        assert!(
            self.file_writer.close(),
            "Check failed: file_writer_->Close()"
        );

        match self.next.flush() {
            FlushResult::RestartStream => panic!(
                "Hybrid grid generation must be configured to occur after any stages that \
                 require multiple passes."
            ),
            FlushResult::Finished => FlushResult::Finished,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use nalgebra::Vector3;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::io::fake_file_writer::FakeFileWriter;
    use crate::io::null_points_processor::NullPointsProcessor;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    #[test]
    fn writes_serialized_hybrid_grid() {
        let content = Arc::new(Mutex::new(Vec::new()));
        let dictionary = make_dictionary(
            r#"return {
              voxel_size = 0.1,
              range_data_inserter = {
                hit_probability = 0.55,
                miss_probability = 0.49,
                num_free_space_voxels = 2,
                intensity_threshold = 1000,
              },
              filename = "grid.pb",
            }"#,
        );
        let mut processor = HybridGridPointsProcessor::from_dictionary(
            &FakeFileWriter::factory(Arc::clone(&content)),
            &dictionary,
            Box::new(NullPointsProcessor::new()),
        );
        processor.process(Box::new(PointsBatch {
            points: vec![RangefinderPoint {
                position: Vector3::new(1.0, 0.0, 0.0),
            }],
            ..Default::default()
        }));
        assert_eq!(processor.flush(), FlushResult::Finished);
        let grid = proto::HybridGrid::decode(content.lock().unwrap().as_slice()).unwrap();
        assert!((grid.resolution - 0.1).abs() < 1e-7);
        let restored = HybridGrid::from_proto(&grid);
        assert!(restored.get_probability(&Vector3::new(10, 0, 0)) > 0.5);
        assert!(restored.get_probability(&Vector3::new(9, 0, 0)) < 0.5);
    }
}
