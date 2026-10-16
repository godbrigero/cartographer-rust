//! Port of `cartographer/mapping/3d/submap_3d` (.h, .cc).

use std::sync::{Arc, RwLock};

use nalgebra::{DVector, Quaternion, Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::port::{fast_gzip_string, round_to_int};
use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
use crate::mapping::probability_values::{clamp_probability, value_to_probability};
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap, SubmapBase, probability_to_log_odds_integer};
use crate::mapping::three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
use crate::mapping::three_d::range_data_inserter_3d::{
    RangeDataInserter3D, create_range_data_inserter_options_3d,
};
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::range_data::{RangeData, transform_range_data};
use crate::transform::rigid_transform::{Rigid3d, Rigid3f, quaternion_mul};
use crate::transform::transform::{get_yaw_quaternion, to_proto, to_rigid3};

#[derive(Debug, Clone, Copy)]
struct PixelData {
    min_z: i32,
    max_z: i32,
    count: i32,
    probability_sum: f32,
    max_probability: f32,
}

impl Default for PixelData {
    fn default() -> Self {
        Self {
            min_z: i32::MAX,
            max_z: i32::MIN,
            count: 0,
            probability_sum: 0.0,
            max_probability: 0.5,
        }
    }
}

/// Filters `range_data`, retaining only the returns that have no more than
/// `max_range` distance from the origin. Removes misses.
fn filter_range_data_by_max_range(range_data: &RangeData, max_range: f32) -> RangeData {
    RangeData {
        origin: range_data.origin,
        returns: range_data
            .returns
            .copy_if(|point| (point.position - range_data.origin).norm() <= max_range),
        misses: PointCloud::new(),
    }
}

fn accumulate_pixel_data(
    width: i32,
    height: i32,
    min_index: &Vector2<i32>,
    max_index: &Vector2<i32>,
    voxel_indices_and_probabilities: &[[i32; 4]],
) -> Vec<PixelData> {
    let num_pixels = usize::try_from(width.wrapping_mul(height)).unwrap_or(0);
    let mut accumulated_pixel_data = vec![PixelData::default(); num_pixels];
    for voxel_index_and_probability in voxel_indices_and_probabilities {
        let pixel_index = Vector2::new(
            voxel_index_and_probability[0],
            voxel_index_and_probability[1],
        );
        if pixel_index.x < min_index.x
            || pixel_index.y < min_index.y
            || pixel_index.x > max_index.x
            || pixel_index.y > max_index.y
        {
            // Out of bounds. This could happen because of floating point
            // inaccuracy.
            continue;
        }
        let x = max_index.x - pixel_index.x;
        let y = max_index.y - pixel_index.y;
        let pixel = &mut accumulated_pixel_data[(x * width + y) as usize];
        pixel.count += 1;
        pixel.min_z = pixel.min_z.min(voxel_index_and_probability[2]);
        pixel.max_z = pixel.max_z.max(voxel_index_and_probability[2]);
        let probability = value_to_probability(voxel_index_and_probability[3] as u16);
        pixel.probability_sum += probability;
        pixel.max_probability = pixel.max_probability.max(probability);
    }
    accumulated_pixel_data
}

/// The first three entries of each returned value are a cell_index and the
/// last is the corresponding probability value. We batch them together like
/// this to only have one vector and have better cache locality.
fn extract_voxel_data(
    hybrid_grid: &HybridGrid,
    transform: &Rigid3f,
    min_index: &mut Vector2<i32>,
    max_index: &mut Vector2<i32>,
) -> Vec<[i32; 4]> {
    let mut voxel_indices_and_probabilities = Vec::new();
    let resolution_inverse = 1.0f32 / hybrid_grid.resolution();

    const XRAY_OBSTRUCTED_CELL_PROBABILITY_LIMIT: f32 = 0.501;
    for (index, &probability_value) in hybrid_grid {
        let probability = value_to_probability(probability_value);
        if probability < XRAY_OBSTRUCTED_CELL_PROBABILITY_LIMIT {
            // We ignore non-obstructed cells.
            continue;
        }

        let cell_center_submap = hybrid_grid.get_center_of_cell(&index);
        let cell_center_global = transform * &cell_center_submap;
        let voxel_index_and_probability = [
            round_to_int(cell_center_global.x * resolution_inverse),
            round_to_int(cell_center_global.y * resolution_inverse),
            round_to_int(cell_center_global.z * resolution_inverse),
            i32::from(probability_value),
        ];

        voxel_indices_and_probabilities.push(voxel_index_and_probability);
        min_index.x = min_index.x.min(voxel_index_and_probability[0]);
        min_index.y = min_index.y.min(voxel_index_and_probability[1]);
        max_index.x = max_index.x.max(voxel_index_and_probability[0]);
        max_index.y = max_index.y.max(voxel_index_and_probability[1]);
    }
    voxel_indices_and_probabilities
}

/// Builds texture data containing interleaved value and alpha for the
/// visualization from `accumulated_pixel_data`.
fn compute_pixel_values(accumulated_pixel_data: &[PixelData]) -> Vec<u8> {
    let mut cell_data = Vec::with_capacity(2 * accumulated_pixel_data.len());
    const MIN_Z_DIFFERENCE: f32 = 3.0;
    const FREE_SPACE_WEIGHT: f32 = 0.15;
    for pixel in accumulated_pixel_data {
        // TODO(whess): Take into account submap rotation.
        // TODO(whess): Document the approach and make it more independent from
        // the chosen resolution.
        let z_difference = if pixel.count > 0 {
            (pixel.max_z - pixel.min_z) as f32
        } else {
            0.0
        };
        if z_difference < MIN_Z_DIFFERENCE {
            cell_data.push(0); // value
            cell_data.push(0); // alpha
            continue;
        }
        let free_space = (z_difference - pixel.count as f32).max(0.0);
        let free_space_weight = FREE_SPACE_WEIGHT * free_space;
        let total_weight = pixel.count as f32 + free_space_weight;
        let free_space_probability = 1.0 - pixel.max_probability;
        let average_probability = clamp_probability(
            (pixel.probability_sum + free_space_probability * free_space_weight) / total_weight,
        );
        let delta = 128 - i32::from(probability_to_log_odds_integer(average_probability));
        let alpha: u8 = if delta > 0 { 0 } else { (-delta) as u8 };
        let value: u8 = if delta > 0 { delta as u8 } else { 0 };
        cell_data.push(value); // value
        cell_data.push(if value != 0 || alpha != 0 { alpha } else { 1 }); // alpha
    }
    cell_data
}

fn add_to_texture_proto(
    hybrid_grid: &HybridGrid,
    global_submap_pose: &Rigid3d,
) -> proto::submap_query::response::SubmapTexture {
    let mut texture = proto::submap_query::response::SubmapTexture::default();
    // Generate an X-ray view through the 'hybrid_grid', aligned to the
    // xy-plane in the global map frame.
    let resolution = hybrid_grid.resolution();
    texture.resolution = f64::from(resolution);

    // Compute a bounding box for the texture.
    let mut min_index = Vector2::new(i32::MAX, i32::MAX);
    let mut max_index = Vector2::new(i32::MIN, i32::MIN);
    let voxel_indices_and_probabilities = extract_voxel_data(
        hybrid_grid,
        &global_submap_pose.cast::<f32>(),
        &mut min_index,
        &mut max_index,
    );

    // RUST PORT NOTE: with no obstructed voxels the C++ computes
    // `INT_MIN - INT_MAX + 1`, which wraps to 2 on all supported platforms
    // (formally UB). We replicate the two's complement wrap-around.
    let width = max_index.y.wrapping_sub(min_index.y).wrapping_add(1);
    let height = max_index.x.wrapping_sub(min_index.x).wrapping_add(1);
    texture.width = width;
    texture.height = height;

    let accumulated_pixel_data = accumulate_pixel_data(
        width,
        height,
        &min_index,
        &max_index,
        &voxel_indices_and_probabilities,
    );
    let cell_data = compute_pixel_values(&accumulated_pixel_data);

    fast_gzip_string(&cell_data, &mut texture.cells);
    // C++: `Eigen::Vector3d(max_index.x() * resolution, ...)` (int * float).
    texture.slice_pose = Some(to_proto(
        &(global_submap_pose.inverse()
            * Rigid3d::from_translation(Vector3::new(
                f64::from(max_index.x as f32 * resolution),
                f64::from(max_index.y as f32 * resolution),
                global_submap_pose.translation().z,
            ))),
    ));
    texture
}

/// C++ `CreateSubmapsOptions3D`.
pub fn create_submaps_options_3d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::SubmapsOptions3D {
    let options = proto::SubmapsOptions3D {
        high_resolution: parameter_dictionary.get_double("high_resolution"),
        high_resolution_max_range: parameter_dictionary.get_double("high_resolution_max_range"),
        low_resolution: parameter_dictionary.get_double("low_resolution"),
        num_range_data: parameter_dictionary.get_non_negative_int("num_range_data"),
        range_data_inserter_options: Some(create_range_data_inserter_options_3d(
            &parameter_dictionary.get_dictionary("range_data_inserter"),
        )),
    };
    assert!(
        options.num_range_data > 0,
        "Check failed: options.num_range_data() > 0 ({} vs. 0)",
        options.num_range_data
    );
    options
}

/// A 3D submap: a high resolution and a low resolution [`HybridGrid`], a high
/// resolution [`IntensityHybridGrid`] and the accumulated rotational scan
/// matcher histogram.
///
/// RUST PORT NOTE: the C++ `std::unique_ptr` grids may be null (a submap
/// constructed from a proto without grid data, or after
/// `ForgetIntensityHybridGrid`); they are `Option`s here and the accessors
/// panic where the C++ would dereference null / CHECK-fail.
#[derive(Clone)]
pub struct Submap3D {
    base: SubmapBase,
    high_resolution_hybrid_grid: Option<HybridGrid>,
    low_resolution_hybrid_grid: Option<HybridGrid>,
    high_resolution_intensity_hybrid_grid: Option<IntensityHybridGrid>,
    rotational_scan_matcher_histogram: DVector<f32>,
}

impl Submap3D {
    /// Creates an empty submap at `local_submap_pose`.
    pub fn new(
        high_resolution: f32,
        low_resolution: f32,
        local_submap_pose: &Rigid3d,
        rotational_scan_matcher_histogram: &DVector<f32>,
    ) -> Self {
        Self {
            base: SubmapBase::new(*local_submap_pose),
            high_resolution_hybrid_grid: Some(HybridGrid::new(high_resolution)),
            low_resolution_hybrid_grid: Some(HybridGrid::new(low_resolution)),
            high_resolution_intensity_hybrid_grid: Some(IntensityHybridGrid::new(high_resolution)),
            rotational_scan_matcher_histogram: rotational_scan_matcher_histogram.clone(),
        }
    }

    /// C++ `Submap3D(const proto::Submap3D&)`. The intensity grid is not
    /// serialized and therefore absent.
    pub fn from_proto(proto: &proto::Submap3D) -> Self {
        let mut submap = Self {
            base: SubmapBase::new(to_rigid3(&proto.local_pose.unwrap_or_default())),
            high_resolution_hybrid_grid: None,
            low_resolution_hybrid_grid: None,
            high_resolution_intensity_hybrid_grid: None,
            rotational_scan_matcher_histogram: DVector::zeros(0),
        };
        submap.update_from_proto_3d(proto);
        submap
    }

    fn update_from_proto_3d(&mut self, submap_3d: &proto::Submap3D) {
        self.set_num_range_data(submap_3d.num_range_data);
        self.set_insertion_finished(submap_3d.finished);
        if let Some(grid) = &submap_3d.high_resolution_hybrid_grid {
            self.high_resolution_hybrid_grid = Some(HybridGrid::from_proto(grid));
        }
        if let Some(grid) = &submap_3d.low_resolution_hybrid_grid {
            self.low_resolution_hybrid_grid = Some(HybridGrid::from_proto(grid));
        }
        self.rotational_scan_matcher_histogram =
            DVector::from_column_slice(&submap_3d.rotational_scan_matcher_histogram);
    }

    /// The high resolution grid.
    pub fn high_resolution_hybrid_grid(&self) -> &HybridGrid {
        self.high_resolution_hybrid_grid
            .as_ref()
            .expect("high_resolution_hybrid_grid is not set")
    }

    /// The low resolution grid.
    pub fn low_resolution_hybrid_grid(&self) -> &HybridGrid {
        self.low_resolution_hybrid_grid
            .as_ref()
            .expect("low_resolution_hybrid_grid is not set")
    }

    /// The high resolution intensity grid.
    pub fn high_resolution_intensity_hybrid_grid(&self) -> &IntensityHybridGrid {
        self.high_resolution_intensity_hybrid_grid
            .as_ref()
            .expect("Check failed: high_resolution_intensity_hybrid_grid_ != nullptr")
    }

    /// Frees the intensity grid to save memory once the submap is no longer
    /// used for local scan matching.
    pub fn forget_intensity_hybrid_grid(&mut self) {
        self.high_resolution_intensity_hybrid_grid = None;
    }

    /// The accumulated rotational scan matcher histogram (in the submap
    /// frame).
    pub fn rotational_scan_matcher_histogram(&self) -> &DVector<f32> {
        &self.rotational_scan_matcher_histogram
    }

    /// Insert `range_data_in_local` into this submap using
    /// `range_data_inserter`. The submap must not be finished yet.
    pub fn insert_data(
        &mut self,
        range_data_in_local: &RangeData,
        range_data_inserter: &RangeDataInserter3D,
        high_resolution_max_range: f32,
        local_from_gravity_aligned: &Quaternion<f64>,
        scan_histogram_in_gravity: &DVector<f32>,
    ) {
        assert!(
            !self.insertion_finished(),
            "Check failed: !insertion_finished()"
        );
        // Transform range data into submap frame.
        let local_pose_inverse = self.local_pose().inverse();
        let transformed_range_data =
            transform_range_data(range_data_in_local, &local_pose_inverse.cast::<f32>());
        range_data_inserter.insert(
            &filter_range_data_by_max_range(&transformed_range_data, high_resolution_max_range),
            self.high_resolution_hybrid_grid
                .as_mut()
                .expect("Check failed: hybrid_grid != nullptr"),
            self.high_resolution_intensity_hybrid_grid.as_mut(),
        );
        range_data_inserter.insert(
            &transformed_range_data,
            self.low_resolution_hybrid_grid
                .as_mut()
                .expect("Check failed: hybrid_grid != nullptr"),
            None,
        );
        self.set_num_range_data(self.num_range_data() + 1);
        let yaw_in_submap_from_gravity = get_yaw_quaternion(&quaternion_mul(
            local_pose_inverse.rotation(),
            local_from_gravity_aligned,
        )) as f32;
        self.rotational_scan_matcher_histogram += RotationalScanMatcher::rotate_histogram(
            scan_histogram_in_gravity,
            yaw_in_submap_from_gravity,
        );
    }

    /// Marks the submap as finished. Must not be called twice.
    pub fn finish(&mut self) {
        assert!(
            !self.insertion_finished(),
            "Check failed: !insertion_finished()"
        );
        self.set_insertion_finished(true);
    }
}

impl Submap for Submap3D {
    fn base(&self) -> &SubmapBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut SubmapBase {
        &mut self.base
    }

    fn to_proto(&self, include_probability_grid_data: bool) -> proto::Submap {
        let mut submap_3d = proto::Submap3D {
            local_pose: Some(to_proto(&self.local_pose())),
            num_range_data: self.num_range_data(),
            finished: self.insertion_finished(),
            ..Default::default()
        };
        if include_probability_grid_data {
            submap_3d.high_resolution_hybrid_grid =
                Some(self.high_resolution_hybrid_grid().to_proto());
            submap_3d.low_resolution_hybrid_grid =
                Some(self.low_resolution_hybrid_grid().to_proto());
        }
        submap_3d.rotational_scan_matcher_histogram = self
            .rotational_scan_matcher_histogram
            .iter()
            .copied()
            .collect();
        proto::Submap {
            submap_3d: Some(submap_3d),
            ..Default::default()
        }
    }

    fn update_from_proto(&mut self, proto: &proto::Submap) {
        let submap_3d = proto
            .submap_3d
            .as_ref()
            .expect("Check failed: proto.has_submap_3d()");
        self.update_from_proto_3d(submap_3d);
    }

    fn to_response_proto(
        &self,
        global_submap_pose: &Rigid3d,
        response: &mut proto::submap_query::Response,
    ) {
        response.submap_version = self.num_range_data();

        response.textures.push(add_to_texture_proto(
            self.high_resolution_hybrid_grid(),
            global_submap_pose,
        ));
        response.textures.push(add_to_texture_proto(
            self.low_resolution_hybrid_grid(),
            global_submap_pose,
        ));
    }
}

/// The first active submap will be created on the insertion of the first
/// range data. Except during this initialization when no or only one single
/// submap exists, there are always two submaps into which range data is
/// inserted: an old submap that is used for matching, and a new one, which
/// will be used for matching next, that is being initialized.
///
/// Once a certain number of range data have been inserted, the new submap is
/// considered initialized: the old submap is no longer changed, the "new"
/// submap is now the "old" submap and is used for scan-to-map matching.
/// Moreover, a "new" submap gets created. The "old" submap is forgotten by
/// this object.
pub struct ActiveSubmaps3D {
    options: proto::SubmapsOptions3D,
    submaps: Vec<SharedSubmap<Submap3D>>,
    range_data_inserter: RangeDataInserter3D,
}

impl ActiveSubmaps3D {
    /// Creates an empty set of active submaps.
    pub fn new(options: &proto::SubmapsOptions3D) -> Self {
        Self {
            options: *options,
            submaps: Vec::new(),
            range_data_inserter: RangeDataInserter3D::new(
                &options.range_data_inserter_options.unwrap_or_default(),
            ),
        }
    }

    /// Inserts `range_data` into the submaps. Returns the active submaps
    /// (front: the older one used for matching).
    pub fn insert_data(
        &mut self,
        range_data: &RangeData,
        local_from_gravity_aligned: &Quaternion<f64>,
        rotational_scan_matcher_histogram_in_gravity: &DVector<f32>,
    ) -> Vec<SharedSubmap<Submap3D>> {
        if self.submaps.is_empty()
            || read(self.submaps.last().unwrap()).num_range_data() == self.options.num_range_data
        {
            self.add_submap(
                &Rigid3d::new(range_data.origin.cast::<f64>(), *local_from_gravity_aligned),
                rotational_scan_matcher_histogram_in_gravity.len(),
            );
        }
        let high_resolution_max_range = self.options.high_resolution_max_range as f32;
        for submap in &self.submaps {
            write(submap).insert_data(
                range_data,
                &self.range_data_inserter,
                high_resolution_max_range,
                local_from_gravity_aligned,
                rotational_scan_matcher_histogram_in_gravity,
            );
        }
        {
            let mut front = write(&self.submaps[0]);
            if front.num_range_data() == 2 * self.options.num_range_data {
                front.finish();
            }
        }
        self.submaps()
    }

    /// The active submaps (at most two; front is the older one).
    pub fn submaps(&self) -> Vec<SharedSubmap<Submap3D>> {
        self.submaps.clone()
    }

    fn add_submap(
        &mut self,
        local_submap_pose: &Rigid3d,
        rotational_scan_matcher_histogram_size: usize,
    ) {
        if self.submaps.len() >= 2 {
            // This will crop the finished Submap before inserting a new Submap
            // to reduce peak memory usage a bit.
            let front = self.submaps.remove(0);
            let mut front = write(&front);
            assert!(
                front.insertion_finished(),
                "Check failed: submaps_.front()->insertion_finished()"
            );
            // We use `forget_intensity_hybrid_grid` to reduce memory usage.
            // Since we use active submaps and their associated intensity
            // hybrid grids for scan matching, we call it once we remove the
            // submap from active submaps and no longer need the intensity
            // hybrid grid.
            front.forget_intensity_hybrid_grid();
        }
        let initial_rotational_scan_matcher_histogram =
            DVector::zeros(rotational_scan_matcher_histogram_size);
        self.submaps.push(Arc::new(RwLock::new(Submap3D::new(
            self.options.high_resolution as f32,
            self.options.low_resolution as f32,
            local_submap_pose,
            &initial_rotational_scan_matcher_histogram,
        ))));
    }
}

fn read(submap: &SharedSubmap<Submap3D>) -> std::sync::RwLockReadGuard<'_, Submap3D> {
    submap.read().expect("submap lock poisoned")
}

fn write(submap: &SharedSubmap<Submap3D>) -> std::sync::RwLockWriteGuard<'_, Submap3D> {
    submap.write().expect("submap lock poisoned")
}

#[cfg(test)]
mod tests {
    mod submaps_test {
        use nalgebra::{DVector, Quaternion, Vector3};

        use crate::mapping::submaps::Submap;
        use crate::mapping::three_d::submap_3d::Submap3D;
        use crate::transform::rigid_transform::Rigid3d;

        #[test]
        fn to_from_proto() {
            let histogram = DVector::from_vec(vec![1.0f32, 2.0]);
            // Eigen::Quaterniond(w = 0, x = 0, y = 0, z = 1).
            let expected = Submap3D::new(
                0.05,
                0.25,
                &Rigid3d::new(
                    Vector3::new(1.0, 2.0, 0.0),
                    Quaternion::new(0.0, 0.0, 0.0, 1.0),
                ),
                &histogram,
            );
            let proto = expected.to_proto(true /* include_probability_grid_data */);
            assert!(proto.submap_2d.is_none());
            assert!(proto.submap_3d.is_some());
            let actual = Submap3D::from_proto(proto.submap_3d.as_ref().unwrap());
            // Eigen `isApprox(other, prec)`: |a - b| <= prec * min(|a|, |b|).
            let is_approx = |a: &[f64], b: &[f64], prec: f64| {
                let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                let na: f64 = a.iter().map(|x| x * x).sum();
                let nb: f64 = b.iter().map(|x| x * x).sum();
                diff.sqrt() <= prec * na.sqrt().min(nb.sqrt())
            };
            let (et, at) = (expected.local_pose(), actual.local_pose());
            assert!(is_approx(
                et.translation().as_slice(),
                at.translation().as_slice(),
                1e-6
            ));
            let (er, ar) = (et.rotation(), at.rotation());
            assert!(is_approx(
                &[er.i, er.j, er.k, er.w],
                &[ar.i, ar.j, ar.k, ar.w],
                1e-6
            ));
            assert_eq!(expected.num_range_data(), actual.num_range_data());
            assert_eq!(expected.insertion_finished(), actual.insertion_finished());
            assert!(
                (f64::from(expected.high_resolution_hybrid_grid().resolution()) - 0.05).abs()
                    <= 1e-6
            );
            assert!(
                (f64::from(expected.low_resolution_hybrid_grid().resolution()) - 0.25).abs()
                    <= 1e-6
            );
            let eh: Vec<f64> = expected
                .rotational_scan_matcher_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let ah: Vec<f64> = actual
                .rotational_scan_matcher_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            assert!(is_approx(&eh, &ah, 1e-6));
        }
    }

    /// Additional tests (no C++ counterpart).
    mod extra {
        use std::collections::BTreeMap;

        use nalgebra::{DVector, Quaternion, Vector3};

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::common::port::fast_gunzip_string;
        use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
        use crate::mapping::proto;
        use crate::mapping::submaps::Submap;
        use crate::mapping::three_d::hybrid_grid::HybridGrid;
        use crate::mapping::three_d::range_data_inserter_3d::RangeDataInserter3D;
        use crate::mapping::three_d::submap_3d::*;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;
        use crate::transform::rigid_transform::{AngleAxis, Rigid3d};

        fn identity_rotation() -> Quaternion<f64> {
            Quaternion::new(1.0, 0.0, 0.0, 0.0)
        }

        fn inserter_options() -> proto::RangeDataInserterOptions3D {
            proto::RangeDataInserterOptions3D {
                hit_probability: 0.7,
                miss_probability: 0.4,
                num_free_space_voxels: 1000,
                intensity_threshold: 100.0,
            }
        }

        /// Returns from a 6 x 6 x 4 m box room around `origin`, sampled every
        /// 0.25 m, with intensity 42.
        fn box_room(origin: Vector3<f32>) -> RangeData {
            let mut points = Vec::new();
            let mut add = |p: Vector3<f32>| {
                points.push(RangefinderPoint {
                    position: origin + p,
                })
            };
            for a in -12..=12 {
                for b in -8..=8 {
                    let (u, v) = (a as f32 * 0.25, b as f32 * 0.25);
                    add(Vector3::new(3.0, u, v));
                    add(Vector3::new(-3.0, u, v));
                    add(Vector3::new(u, 3.0, v));
                    add(Vector3::new(u, -3.0, v));
                }
                for b in -12..=12 {
                    let (u, v) = (a as f32 * 0.25, b as f32 * 0.25);
                    add(Vector3::new(u, v, 2.0));
                    add(Vector3::new(u, v, -2.0));
                }
            }
            let intensities = vec![42.0; points.len()];
            RangeData {
                origin,
                returns: PointCloud::with_intensities(points, intensities),
                misses: PointCloud::new(),
            }
        }

        fn grid_cells(grid: &HybridGrid) -> BTreeMap<(i32, i32, i32), u16> {
            grid.into_iter()
                .map(|(index, &value)| ((index.x, index.y, index.z), value))
                .collect()
        }

        fn filled_submap() -> Submap3D {
            let inserter = RangeDataInserter3D::new(&inserter_options());
            let pose = Rigid3d::with_angle_axis(
                Vector3::new(0.5, -0.25, 0.0),
                AngleAxis::new(0.3, Vector3::z()),
            );
            let mut submap = Submap3D::new(0.1, 0.4, &pose, &DVector::zeros(8));
            let scan_histogram = DVector::from_vec(vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
            for _ in 0..3 {
                submap.insert_data(
                    &box_room(Vector3::new(0.5, -0.25, 0.0)),
                    &inserter,
                    3.5,
                    pose.rotation(),
                    &scan_histogram,
                );
            }
            submap
        }

        #[test]
        fn create_submaps_options_3d_reads_all_keys() {
            let dictionary = make_dictionary(
                "return { \
                 high_resolution = 0.1, \
                 high_resolution_max_range = 20., \
                 low_resolution = 0.45, \
                 num_range_data = 160, \
                 range_data_inserter = { \
                   hit_probability = 0.55, \
                   miss_probability = 0.49, \
                   num_free_space_voxels = 2, \
                   intensity_threshold = 4095, \
                 }, \
                 }",
            );
            let options = create_submaps_options_3d(&dictionary);
            assert_eq!(options.high_resolution, 0.1);
            assert_eq!(options.high_resolution_max_range, 20.0);
            assert_eq!(options.low_resolution, 0.45);
            assert_eq!(options.num_range_data, 160);
            let inserter = options.range_data_inserter_options.unwrap();
            assert_eq!(inserter.num_free_space_voxels, 2);
            assert_eq!(inserter.intensity_threshold, 4095.0);
        }

        #[test]
        fn box_room_has_occupied_walls() {
            let submap = filled_submap();
            assert_eq!(submap.num_range_data(), 3);
            let high = submap.high_resolution_hybrid_grid();
            let low = submap.low_resolution_hybrid_grid();
            let intensity = submap.high_resolution_intensity_hybrid_grid();
            // The submap frame equals the sensor frame rotated by 0.3 rad, so
            // query in the sensor frame: p_submap = R^-1 * p_sensor.
            let to_submap = |p: Vector3<f32>| {
                let r = Rigid3d::from_angle_axis(AngleAxis::new(-0.3, Vector3::z())).cast::<f32>();
                r * p
            };
            for &wall in &[
                Vector3::new(3.0f32, 0.0, 0.0),
                Vector3::new(-3.0, 1.0, 0.5),
                Vector3::new(0.5, 3.0, -1.0),
                Vector3::new(0.0, 0.0, 2.0),
            ] {
                let p = to_submap(wall);
                let low_p = low.get_probability(&low.get_cell_index(&p));
                assert!(low_p > 0.75, "low-res wall {wall:?}: {low_p}");
                let high_p = high.get_probability(&high.get_cell_index(&p));
                assert!(high_p > 0.75, "high-res wall {wall:?}: {high_p}");
                let i = intensity.get_intensity(&intensity.get_cell_index(&p));
                assert!((i - 42.0).abs() < 1e-4, "intensity at {wall:?}: {i}");
            }
            // A corner is beyond the 3.5 m high resolution max range: only in
            // the low resolution grid.
            let corner = to_submap(Vector3::new(3.0, 3.0, 2.0));
            assert!(low.get_probability(&low.get_cell_index(&corner)) > 0.75);
            assert!(!high.is_known(&high.get_cell_index(&corner)));
            // The center is free.
            let center = to_submap(Vector3::zeros());
            assert!(low.get_probability(&low.get_cell_index(&center)) < 0.5);
            assert!(high.get_probability(&high.get_cell_index(&center)) < 0.5);
        }

        #[test]
        fn histogram_is_rotated_into_submap_frame() {
            let inserter = RangeDataInserter3D::new(&inserter_options());
            let submap_pose = Rigid3d::from_angle_axis(AngleAxis::new(0.2, Vector3::z()));
            let mut submap = Submap3D::new(0.1, 0.4, &submap_pose, &DVector::zeros(10));
            let scan_histogram = DVector::from_vec((0..10).map(|i| i as f32).collect());
            let gravity_alignment = AngleAxis::new(0.7, Vector3::z()).to_quaternion();
            submap.insert_data(
                &RangeData::default(),
                &inserter,
                10.0,
                &gravity_alignment,
                &scan_histogram,
            );
            submap.insert_data(
                &RangeData::default(),
                &inserter,
                10.0,
                &gravity_alignment,
                &scan_histogram,
            );
            let expected = RotationalScanMatcher::rotate_histogram(&scan_histogram, 0.5) * 2.0;
            assert!((submap.rotational_scan_matcher_histogram() - expected).norm() < 1e-4);
        }

        #[test]
        fn proto_round_trip_with_data() {
            let mut submap = filled_submap();
            submap.finish();
            let proto = submap.to_proto(true);
            let actual = Submap3D::from_proto(proto.submap_3d.as_ref().unwrap());
            assert_eq!(actual.num_range_data(), 3);
            assert!(actual.insertion_finished());
            assert_eq!(actual.local_pose(), submap.local_pose());
            assert_eq!(
                grid_cells(actual.high_resolution_hybrid_grid()),
                grid_cells(submap.high_resolution_hybrid_grid())
            );
            assert_eq!(
                grid_cells(actual.low_resolution_hybrid_grid()),
                grid_cells(submap.low_resolution_hybrid_grid())
            );
            assert_eq!(
                actual.rotational_scan_matcher_histogram(),
                submap.rotational_scan_matcher_histogram()
            );
            assert_eq!(actual.to_proto(true), proto);
            assert!(
                std::panic::catch_unwind(|| {
                    actual.high_resolution_intensity_hybrid_grid();
                })
                .is_err()
            );

            // Without grid data, the grids are absent / kept.
            let proto_without_grids = submap.to_proto(false);
            let submap_3d = proto_without_grids.submap_3d.as_ref().unwrap();
            assert!(submap_3d.high_resolution_hybrid_grid.is_none());
            assert!(submap_3d.low_resolution_hybrid_grid.is_none());
            assert_eq!(submap_3d.rotational_scan_matcher_histogram.len(), 8);
            let mut updated = Submap3D::from_proto(proto.submap_3d.as_ref().unwrap());
            let mut changed = proto_without_grids.clone();
            changed.submap_3d.as_mut().unwrap().num_range_data = 7;
            updated.update_from_proto(&changed);
            assert_eq!(updated.num_range_data(), 7);
            assert_eq!(
                grid_cells(updated.high_resolution_hybrid_grid()),
                grid_cells(submap.high_resolution_hybrid_grid())
            );
        }

        #[test]
        #[should_panic(expected = "has_submap_3d")]
        fn update_from_2d_proto_panics() {
            let mut submap = Submap3D::new(0.1, 0.4, &Rigid3d::identity(), &DVector::zeros(0));
            submap.update_from_proto(&proto::Submap::default());
        }

        #[test]
        fn response_proto_has_two_textures() {
            let submap = filled_submap();
            let mut response = proto::submap_query::Response::default();
            submap.to_response_proto(&Rigid3d::identity(), &mut response);
            assert_eq!(response.submap_version, 3);
            assert_eq!(response.textures.len(), 2);
            for (texture, resolution) in response.textures.iter().zip([0.1f32, 0.4]) {
                assert_eq!(texture.resolution, f64::from(resolution));
                assert!(texture.width > 0 && texture.height > 0);
                let mut cells = Vec::new();
                fast_gunzip_string(&texture.cells, &mut cells);
                assert_eq!(cells.len(), (2 * texture.width * texture.height) as usize);
                // Walls are vertical, so some pixels are drawn as occupied.
                assert!(cells.chunks(2).any(|c| c[1] > 0));
                assert!(texture.slice_pose.is_some());
            }
        }

        #[test]
        fn response_proto_of_empty_submap() {
            // Replicates the C++ integer wrap-around: 2 x 2 empty texture.
            let submap = Submap3D::new(0.1, 0.4, &Rigid3d::identity(), &DVector::zeros(0));
            let mut response = proto::submap_query::Response::default();
            submap.to_response_proto(&Rigid3d::identity(), &mut response);
            assert_eq!(response.textures.len(), 2);
            for texture in &response.textures {
                assert_eq!((texture.width, texture.height), (2, 2));
                let mut cells = Vec::new();
                fast_gunzip_string(&texture.cells, &mut cells);
                assert_eq!(cells, vec![0; 8]);
            }
        }

        #[test]
        fn compute_pixel_values_matches_formula() {
            // count 2, z from 0 to 4 -> z_difference 4, free_space 2,
            // weight 0.3; probabilities 0.9 + 0.9.
            let pixel = PixelData {
                min_z: 0,
                max_z: 4,
                count: 2,
                probability_sum: 1.8,
                max_probability: 0.9,
            };
            let values = compute_pixel_values(&[pixel, PixelData::default()]);
            let average: f32 = (1.8 + 0.1 * 0.3) / 2.3;
            let delta = 128
                - i32::from(crate::mapping::submaps::probability_to_log_odds_integer(
                    average,
                ));
            assert!(delta < 0);
            assert_eq!(values, vec![0, (-delta) as u8, 0, 0]);
        }

        #[test]
        fn active_submaps_lifecycle() {
            let options = proto::SubmapsOptions3D {
                high_resolution: 0.2,
                high_resolution_max_range: 10.0,
                low_resolution: 0.5,
                num_range_data: 2,
                range_data_inserter_options: Some(inserter_options()),
            };
            let mut active = ActiveSubmaps3D::new(&options);
            assert!(active.submaps().is_empty());
            let histogram = DVector::from_element(4, 1.0f32);
            let insert = |active: &mut ActiveSubmaps3D, x: f32| {
                active.insert_data(
                    &box_room(Vector3::new(x, 0.0, 0.0)),
                    &identity_rotation(),
                    &histogram,
                )
            };
            let s = insert(&mut active, 0.0);
            assert_eq!(s.len(), 1);
            let first = s[0].clone();
            assert_eq!(
                *first.read().unwrap().local_pose().translation(),
                Vector3::new(0.0, 0.0, 0.0)
            );
            let s = insert(&mut active, 0.1);
            assert_eq!(s.len(), 1);
            let s = insert(&mut active, 0.2);
            assert_eq!(s.len(), 2);
            assert!(Arc::ptr_eq(&s[0], &first));
            let second = s[1].clone();
            assert!((second.read().unwrap().local_pose().translation().x - 0.2).abs() < 1e-6);
            assert_eq!(
                second
                    .read()
                    .unwrap()
                    .rotational_scan_matcher_histogram()
                    .len(),
                4
            );
            let s = insert(&mut active, 0.3);
            assert_eq!(s.len(), 2);
            assert_eq!(first.read().unwrap().num_range_data(), 4);
            assert!(first.read().unwrap().insertion_finished());
            assert!(!second.read().unwrap().insertion_finished());
            // The front is dropped; its intensity grid is forgotten.
            let s = insert(&mut active, 0.4);
            assert_eq!(s.len(), 2);
            assert!(Arc::ptr_eq(&s[0], &second));
            assert_eq!(second.read().unwrap().num_range_data(), 3);
            assert_eq!(s[1].read().unwrap().num_range_data(), 1);
            let first = first.read().unwrap();
            assert!(first.high_resolution_intensity_hybrid_grid.is_none());
            // The grids are still available for the pose graph.
            assert!(first.high_resolution_hybrid_grid().into_iter().count() > 0);
            assert_eq!(active.submaps().len(), 2);
        }
    }
}
