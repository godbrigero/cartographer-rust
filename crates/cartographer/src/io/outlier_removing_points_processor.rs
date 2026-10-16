//! Port of `cartographer/io/outlier_removing_points_processor` (.h, .cc).

use std::collections::HashSet;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::points_batch::{PointsBatch, remove_points};
use crate::io::points_processor::{FlushResult, PointsProcessor};
use crate::mapping::three_d::hybrid_grid::{GridValue, HybridGridBase};

/// Per-voxel counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VoxelData {
    hits: i32,
    rays: i32,
}

impl GridValue for VoxelData {
    fn is_default_value(&self) -> bool {
        *self == VoxelData::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Phase1,
    Phase2,
    Phase3,
}

/// Voxel filters the data and only passes on points that we believe are on
/// non-moving objects.
pub struct OutlierRemovingPointsProcessor {
    voxel_size: f64,
    miss_per_hit_limit: f64,
    next: Box<dyn PointsProcessor>,
    state: State,
    // To reduce memory consumption by not having to keep all rays in memory,
    // we filter outliers in three phases each going over all data: First we
    // compute all voxels containing any hits, then we compute the rays passing
    // through each of these voxels, and finally we output all hits in voxels
    // that are considered obstructed.
    voxels: HybridGridBase<VoxelData>,
}

impl OutlierRemovingPointsProcessor {
    /// Name of this processor in the Lua pipeline configuration.
    pub const CONFIGURATION_FILE_ACTION_NAME: &'static str =
        "voxel_filter_and_remove_moving_objects";

    /// Creates the processor.
    pub fn new(voxel_size: f64, miss_per_hit_limit: f64, next: Box<dyn PointsProcessor>) -> Self {
        log::info!("Marking hits...");
        Self {
            voxel_size,
            miss_per_hit_limit,
            next,
            state: State::Phase1,
            voxels: HybridGridBase::new(voxel_size as f32),
        }
    }

    /// Creates the processor from `voxel_size` and the optional
    /// `miss_per_hit_limit` (default 3).
    pub fn from_dictionary(
        dictionary: &LuaParameterDictionary,
        next: Box<dyn PointsProcessor>,
    ) -> Box<Self> {
        let miss_per_hit_limit = if !dictionary.has_key("miss_per_hit_limit") {
            log::info!("Using default value of 3 for miss_per_hit_limit.");
            3.
        } else {
            dictionary.get_double("miss_per_hit_limit")
        };
        Box::new(Self::new(
            dictionary.get_double("voxel_size"),
            miss_per_hit_limit,
            next,
        ))
    }

    // First phase counts the number of hits per voxel.
    fn process_in_phase_one(&mut self, batch: &PointsBatch) {
        for point in &batch.points {
            let index = self.voxels.get_cell_index(&point.position);
            self.voxels.mutable_value(&index).hits += 1;
        }
    }

    // Second phase counts how many rays pass through each voxel. This is only
    // done for voxels that contain hits. This is to reduce memory consumption
    // by not adding data to free voxels.
    fn process_in_phase_two(&mut self, batch: &PointsBatch) {
        // TODO(whess): This samples every 'voxel_size' distance and could be
        // improved by better ray casting, and also by marking the hits of the
        // current range data to be excluded.
        for point in &batch.points {
            let delta = point.position - batch.origin;
            let length = delta.norm();
            // C++: `for (float x = 0; x < length; x += voxel_size_)` with a
            // double `voxel_size_`, i.e. the sum is computed in double and
            // narrowed back to float on every step.
            let mut x = 0.0_f32;
            while x < length {
                let index = self
                    .voxels
                    .get_cell_index(&(batch.origin + (x / length) * delta));
                if self.voxels.value(&index).hits > 0 {
                    self.voxels.mutable_value(&index).rays += 1;
                }
                x = (f64::from(x) + self.voxel_size) as f32;
            }
        }
    }

    // Third phase produces the output containing all inliers. We consider
    // each hit an inlier if it is inside a voxel that has a sufficiently high
    // hit-to-ray ratio.
    fn process_in_phase_three(&mut self, mut batch: Box<PointsBatch>) {
        let mut to_remove = HashSet::new();
        for (i, point) in batch.points.iter().enumerate() {
            let voxel = self
                .voxels
                .value(&self.voxels.get_cell_index(&point.position));
            // `!(rays < limit * hits)` (true for NaN, like the C++ code).
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let remove = !(f64::from(voxel.rays) < self.miss_per_hit_limit * f64::from(voxel.hits));
            if remove {
                to_remove.insert(i);
            }
        }
        remove_points(&to_remove, &mut batch);
        self.next.process(batch);
    }
}

impl PointsProcessor for OutlierRemovingPointsProcessor {
    fn process(&mut self, batch: Box<PointsBatch>) {
        match self.state {
            State::Phase1 => self.process_in_phase_one(&batch),
            State::Phase2 => self.process_in_phase_two(&batch),
            State::Phase3 => self.process_in_phase_three(batch),
        }
    }

    fn flush(&mut self) -> FlushResult {
        match self.state {
            State::Phase1 => {
                log::info!("Counting rays...");
                self.state = State::Phase2;
                FlushResult::RestartStream
            }
            State::Phase2 => {
                log::info!("Filtering outliers...");
                self.state = State::Phase3;
                FlushResult::RestartStream
            }
            State::Phase3 => {
                assert!(
                    self.next.flush() == FlushResult::Finished,
                    "Voxel filtering and outlier removal must be configured to occur after any \
                     stages that require multiple passes."
                );
                FlushResult::Finished
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use nalgebra::Vector3;

    use super::*;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    /// Records all processed batches.
    struct Recorder(Arc<Mutex<Vec<PointsBatch>>>);

    impl PointsProcessor for Recorder {
        fn process(&mut self, batch: Box<PointsBatch>) {
            self.0.lock().unwrap().push(*batch);
        }
        fn flush(&mut self) -> FlushResult {
            FlushResult::Finished
        }
    }

    fn point(x: f32, y: f32, z: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, y, z),
        }
    }

    #[test]
    fn removes_points_seen_through() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let mut processor =
            OutlierRemovingPointsProcessor::new(0.1, 3., Box::new(Recorder(Arc::clone(&output))));
        // Batch 0 sees a "moving object" at x = 1. Batches 1..=5 see a wall at
        // x = 2 through it (passing through the voxel at x = 1).
        let batches = || {
            let mut batches = vec![PointsBatch {
                points: vec![point(1.0, 0.0, 0.0)],
                ..Default::default()
            }];
            for _ in 0..5 {
                batches.push(PointsBatch {
                    points: vec![point(2.0, 0.0, 0.0)],
                    ..Default::default()
                });
            }
            batches
        };
        let mut passes = 0;
        loop {
            passes += 1;
            for batch in batches() {
                processor.process(Box::new(batch));
            }
            if processor.flush() == FlushResult::Finished {
                break;
            }
        }
        assert_eq!(passes, 3);
        let output = output.lock().unwrap();
        assert_eq!(output.len(), 6);
        assert!(output[0].points.is_empty());
        for batch in &output[1..] {
            assert_eq!(batch.points, vec![point(2.0, 0.0, 0.0)]);
        }
    }
}
