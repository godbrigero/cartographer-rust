//! Port of `cartographer/mapping/internal/2d/tsdf_range_data_inserter_2d` (.h, .cc).

use std::cmp::Ordering;

use nalgebra::{SVector, Vector2};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::math::{atan2, clamp, normalize_angle_difference};
use crate::mapping::grid_interface::GridInterface;
use crate::mapping::internal::two_d::normal_estimation_2d::{
    create_normal_estimation_options_2d, estimate_normals_for_points,
};
use crate::mapping::internal::two_d::ray_to_pixel_mask::ray_to_pixel_mask;
use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::mapping::proto;
use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::probability_grid_range_data_inserter_2d::{
    BoundingBox2f, SUBPIXEL_SCALE, superscaled_limits,
};
use crate::mapping::two_d::xy_index::Array2i;
use crate::sensor::range_data::RangeData;
use crate::sensor::rangefinder_point::RangefinderPoint;

/// Minimum distance between range observation and origin. Otherwise, range
/// observations are discarded.
const MIN_RANGE_METERS: f32 = 1e-6;

/// C++ `const float kSqrtTwoPi = std::sqrt(2.0 * M_PI);`.
#[inline]
fn sqrt_two_pi() -> f32 {
    (2.0 * std::f64::consts::PI).sqrt() as f32
}

/// Eigen's `normalized()`: returns the vector unchanged if its norm is zero
/// (nalgebra's `normalize` would produce NaNs).
#[inline]
fn eigen_normalized<const D: usize>(v: &SVector<f32, D>) -> SVector<f32, D> {
    let squared_norm = v.norm_squared();
    if squared_norm > 0.0 {
        v / squared_norm.sqrt()
    } else {
        *v
    }
}

fn grow_as_needed(range_data: &RangeData, truncation_distance: f32, tsdf: &mut TSDF2D) {
    let mut bounding_box = BoundingBox2f::new(range_data.origin.xy());
    for hit in range_data.returns.iter() {
        let direction = eigen_normalized(&(hit.position - range_data.origin));
        let end_position = hit.position + truncation_distance * direction;
        bounding_box.extend(&end_position.xy());
    }
    // Padding around bounding box to avoid numerical issues at cell
    // boundaries (applied by `grow`).
    bounding_box.grow(tsdf);
}

/// Gaussian kernel; computed in double precision like C++
/// (`1.0 / (kSqrtTwoPi * sigma) * std::exp(-0.5 * x * x / (sigma * sigma))`).
#[inline]
fn gaussian_kernel(x: f32, sigma: f32) -> f32 {
    let x = f64::from(x);
    (1.0 / f64::from(sqrt_two_pi() * sigma) * (-0.5 * x * x / f64::from(sigma * sigma)).exp())
        as f32
}

/// Returns `(superscaled_begin, superscaled_end)` cell indices.
fn superscale_ray(begin: &Vector2<f32>, end: &Vector2<f32>, tsdf: &TSDF2D) -> (Array2i, Array2i) {
    let superscaled_limits = superscaled_limits(tsdf.limits());
    (
        superscaled_limits.get_cell_index(begin),
        superscaled_limits.get_cell_index(end),
    )
}

/// Strict weak ordering of returns by bearing around `origin` (C++
/// `RangeDataSorter`): points with negative `y` direction first, ordered by
/// ascending `x` direction, then the others by descending `x` direction.
#[inline]
fn range_data_sorter_less(delta_lhs: &Vector2<f32>, delta_rhs: &Vector2<f32>) -> bool {
    if (delta_lhs[1] < 0.0) != (delta_rhs[1] < 0.0) {
        delta_lhs[1] < 0.0
    } else if delta_lhs[1] < 0.0 {
        delta_lhs[0] < delta_rhs[0]
    } else {
        delta_lhs[0] > delta_rhs[0]
    }
}

/// `std::pow(float, int)` promotes to double in C++11; the result is narrowed
/// on assignment to the float weight.
#[inline]
fn compute_range_weight_factor(range: f32, exponent: i32) -> f32 {
    let mut weight = 0.0_f32;
    if range.abs() > MIN_RANGE_METERS {
        weight = (1.0 / f64::from(range).powf(f64::from(exponent))) as f32;
    }
    weight
}

/// Reads `TSDFRangeDataInserterOptions2D` from `parameter_dictionary`.
pub fn create_tsdf_range_data_inserter_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::TsdfRangeDataInserterOptions2D {
    proto::TsdfRangeDataInserterOptions2D {
        truncation_distance: parameter_dictionary.get_double("truncation_distance"),
        maximum_weight: parameter_dictionary.get_double("maximum_weight"),
        update_free_space: parameter_dictionary.get_bool("update_free_space"),
        normal_estimation_options: Some(create_normal_estimation_options_2d(
            &parameter_dictionary.get_dictionary("normal_estimation_options"),
        )),
        project_sdf_distance_to_scan_normal: parameter_dictionary
            .get_bool("project_sdf_distance_to_scan_normal"),
        update_weight_range_exponent: parameter_dictionary.get_int("update_weight_range_exponent"),
        update_weight_angle_scan_normal_to_ray_kernel_bandwidth: parameter_dictionary
            .get_double("update_weight_angle_scan_normal_to_ray_kernel_bandwidth"),
        update_weight_distance_cell_to_hit_kernel_bandwidth: parameter_dictionary
            .get_double("update_weight_distance_cell_to_hit_kernel_bandwidth"),
    }
}

/// Inserts range data into a [`TSDF2D`].
#[derive(Debug, Clone)]
pub struct TSDFRangeDataInserter2D {
    options: proto::TsdfRangeDataInserterOptions2D,
}

impl TSDFRangeDataInserter2D {
    /// Creates the inserter.
    pub fn new(options: &proto::TsdfRangeDataInserterOptions2D) -> Self {
        Self { options: *options }
    }

    /// The options this inserter was created with.
    pub fn options(&self) -> &proto::TsdfRangeDataInserterOptions2D {
        &self.options
    }

    /// Casts a ray from origin towards hit for each hit in range data.
    /// If `options.update_free_space` is `true`, all cells along the ray
    /// until `truncation_distance` behind hit are updated. Otherwise, only
    /// the cells within `truncation_distance` around hit are updated.
    /// (Concrete-typed version of [`RangeDataInserterInterface::insert`].)
    pub fn insert_into_tsdf(&self, range_data: &RangeData, tsdf: &mut TSDF2D) {
        let truncation_distance = self.options.truncation_distance as f32;
        grow_as_needed(range_data, truncation_distance, tsdf);

        // Compute normals if needed.
        let scale_update_weight_angle_scan_normal_to_ray = self
            .options
            .update_weight_angle_scan_normal_to_ray_kernel_bandwidth
            != 0.0;
        // RUST PORT NOTE: C++ always copies the range data; here the returns
        // are only copied (and sorted) when normals are needed.
        let sorted_returns: Vec<RangefinderPoint>;
        let mut normals: Vec<f32> = Vec::new();
        let returns: &[RangefinderPoint] = if self.options.project_sdf_distance_to_scan_normal
            || scale_update_weight_angle_scan_normal_to_ray
        {
            let origin = range_data.origin.xy();
            // The sort keys (normalized bearings) are computed once per point
            // instead of once per comparison; the comparator is the same.
            // RUST PORT NOTE: `std::sort` is unstable like `sort_unstable_by`;
            // the relative order of returns with identical bearings may
            // differ from C++.
            let mut keyed: Vec<(Vector2<f32>, RangefinderPoint)> = range_data
                .returns
                .iter()
                .map(|p| (eigen_normalized(&(p.position.xy() - origin)), *p))
                .collect();
            keyed.sort_unstable_by(|(lhs, _), (rhs, _)| {
                if range_data_sorter_less(lhs, rhs) {
                    Ordering::Less
                } else if range_data_sorter_less(rhs, lhs) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            sorted_returns = keyed.into_iter().map(|(_, p)| p).collect();
            normals = estimate_normals_for_points(
                &sorted_returns,
                &range_data.origin,
                &self.options.normal_estimation_options.unwrap_or_default(),
            );
            &sorted_returns
        } else {
            range_data.returns.points()
        };

        let origin = range_data.origin.xy();
        for (hit_index, hit) in returns.iter().enumerate() {
            let hit = hit.position.xy();
            let normal = if normals.is_empty() {
                f32::NAN
            } else {
                normals[hit_index]
            };
            self.insert_hit(&hit, &origin, normal, tsdf);
        }
        tsdf.finish_update();
    }

    fn insert_hit(
        &self,
        hit: &Vector2<f32>,
        origin: &Vector2<f32>,
        normal: f32,
        tsdf: &mut TSDF2D,
    ) {
        let ray = hit - origin;
        let range = ray.norm();
        let truncation_distance = self.options.truncation_distance as f32;
        if range < truncation_distance {
            return;
        }
        let truncation_ratio = truncation_distance / range;
        let ray_begin = if self.options.update_free_space {
            *origin
        } else {
            origin + (1.0 - truncation_ratio) * ray
        };
        let ray_end = origin + (1.0 + truncation_ratio) * ray;
        let (superscaled_begin, superscaled_end) = superscale_ray(&ray_begin, &ray_end, tsdf);
        // RUST PORT NOTE: `ray_to_pixel_mask` returns a fresh `Vec` per ray,
        // like the C++ `RayToPixelMask` returns a `std::vector`.
        let ray_mask = ray_to_pixel_mask(&superscaled_begin, &superscaled_end, SUBPIXEL_SCALE);

        // Precompute weight factors.
        let mut weight_factor_angle_ray_normal = 1.0_f32;
        if self
            .options
            .update_weight_angle_scan_normal_to_ray_kernel_bandwidth
            != 0.0
        {
            let negative_ray = -ray;
            let angle_ray_normal = normalize_angle_difference(normal - atan2(&negative_ray));
            weight_factor_angle_ray_normal = gaussian_kernel(
                angle_ray_normal,
                self.options
                    .update_weight_angle_scan_normal_to_ray_kernel_bandwidth as f32,
            );
        }
        let mut weight_factor_range = 1.0_f32;
        if self.options.update_weight_range_exponent != 0 {
            weight_factor_range =
                compute_range_weight_factor(range, self.options.update_weight_range_exponent);
        }

        // Update cells.
        let distance_bandwidth = self
            .options
            .update_weight_distance_cell_to_hit_kernel_bandwidth;
        for cell_index in &ray_mask {
            if tsdf.cell_is_updated(cell_index) {
                continue;
            }
            let cell_center = tsdf.limits().get_cell_center(cell_index);
            let distance_cell_to_origin = (cell_center - origin).norm();
            let mut update_tsd = range - distance_cell_to_origin;
            if self.options.project_sdf_distance_to_scan_normal {
                let normal_orientation = normal;
                update_tsd = (cell_center - hit).dot(&Vector2::new(
                    normal_orientation.cos(),
                    normal_orientation.sin(),
                ));
            }
            update_tsd = clamp(update_tsd, -truncation_distance, truncation_distance);
            let mut update_weight = weight_factor_range * weight_factor_angle_ray_normal;
            if distance_bandwidth != 0.0 {
                update_weight *= gaussian_kernel(update_tsd, distance_bandwidth as f32);
            }
            self.update_cell(cell_index, update_tsd, update_weight, tsdf);
        }
    }

    fn update_cell(&self, cell: &Array2i, update_sdf: f32, update_weight: f32, tsdf: &mut TSDF2D) {
        if update_weight == 0.0 {
            return;
        }
        let (tsd, weight) = tsdf.get_tsd_and_weight(cell);
        let mut updated_weight = weight + update_weight;
        let updated_sdf = (tsd * weight + update_sdf * update_weight) / updated_weight;
        updated_weight = updated_weight.min(self.options.maximum_weight as f32);
        tsdf.set_cell(cell, updated_sdf, updated_weight);
    }
}

impl RangeDataInserterInterface for TSDFRangeDataInserter2D {
    /// Inserts `range_data` into `grid`, which must be a [`TSDF2D`] (panics
    /// otherwise; C++ uses an unchecked `static_cast`).
    fn insert(&self, range_data: &RangeData, grid: &mut dyn GridInterface) {
        let tsdf = grid
            .downcast_mut::<TSDF2D>()
            .expect("TSDFRangeDataInserter2D requires a TSDF2D");
        self.insert_into_tsdf(range_data, tsdf);
    }
}

#[cfg(test)]
mod tests {
    mod range_data_inserter_test_2d_tsdf {
        use std::f64::consts::PI;
        use std::sync::Arc;

        use nalgebra::{Vector2, Vector3};

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
        use crate::mapping::internal::two_d::tsdf_range_data_inserter_2d::*;
        use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
        use crate::mapping::two_d::grid_2d::Grid2DTrait;
        use crate::mapping::two_d::map_limits::MapLimits;
        use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
        use crate::mapping::value_conversion_tables::ValueConversionTables;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        struct Fixture {
            options: proto::TsdfRangeDataInserterOptions2D,
            tsdf: TSDF2D,
            range_data_inserter: TSDFRangeDataInserter2D,
        }

        impl Fixture {
            fn new() -> Self {
                let conversion_tables = Arc::new(ValueConversionTables::new());
                let tsdf = TSDF2D::new(
                    MapLimits::new(1.0, Vector2::new(1.0, 7.0), CellLimits::new(8, 1)),
                    2.0,
                    10.0,
                    &conversion_tables,
                );
                let parameter_dictionary = make_dictionary(
                    "return { \
                     truncation_distance = 2.0,\
                     maximum_weight = 10.,\
                     update_free_space = false,\
                     normal_estimation_options = {\
                     num_normal_samples = 2,\
                     sample_radius = 10.,\
                     },\
                     project_sdf_distance_to_scan_normal = false,\
                     update_weight_range_exponent = 0,\
                     update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0,\
                     update_weight_distance_cell_to_hit_kernel_bandwidth = 0,\
                     }",
                );
                let options = create_tsdf_range_data_inserter_options_2d(&parameter_dictionary);
                let range_data_inserter = TSDFRangeDataInserter2D::new(&options);
                Self {
                    options,
                    tsdf,
                    range_data_inserter,
                }
            }

            fn recreate_inserter(&mut self) {
                self.range_data_inserter = TSDFRangeDataInserter2D::new(&self.options);
            }

            fn insert_point(&mut self) {
                let mut range_data = RangeData::default();
                range_data.returns.push_back(point(-0.5, 3.5));
                range_data.origin.x = -0.5;
                range_data.origin.y = -0.5;
                self.range_data_inserter.insert(&range_data, &mut self.tsdf);
                self.tsdf.finish_update();
            }

            fn cell_index(&self, x: f32, y: f32) -> Array2i {
                self.tsdf.limits().get_cell_index(&Vector2::new(x, y))
            }

            /// C++ `EXPECT_THAT(MockCellProperties(cell_index, tsdf_),
            /// EqualCellProperties(known, tsd, weight))`.
            fn expect_cell_properties(
                &self,
                cell_index: &Array2i,
                expected_is_known: bool,
                expected_tsd: f32,
                expected_weight: f32,
            ) {
                let is_known = self.tsdf.is_known(cell_index);
                let tsd = self.tsdf.get_tsd(cell_index);
                let weight = self.tsdf.get_weight(cell_index);
                assert!(
                    expected_is_known == is_known
                        && (expected_tsd - tsd).abs() < 1e-4
                        && (expected_weight - weight).abs() < 1e-2,
                    "cell {cell_index:?}: expected {expected_is_known}\t{expected_tsd}\t\
                     {expected_weight}, actual {is_known}\t{tsd}\t{weight}"
                );
            }
        }

        fn point(x: f32, y: f32) -> RangefinderPoint {
            RangefinderPoint {
                position: Vector3::new(x, y, 0.0),
            }
        }

        /// Iterates `for (float y = start; y < 6.; ++y)`.
        fn ys(start: f32) -> impl Iterator<Item = f32> {
            std::iter::successors(Some(start), |y| Some(y + 1.0)).take_while(|y| *y < 6.0)
        }

        #[test]
        fn insert_point() {
            let mut f = Fixture::new();
            f.insert_point();
            let truncation_distance = f.options.truncation_distance as f32;
            let maximum_weight = f.options.maximum_weight as f32;

            for y in ys(1.5) {
                // Cell intersects with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                f.expect_cell_properties(&cell_index, true, expected_tsdf, 1.0);
                // Cells don't intersect with ray.
                let cell_index = f.cell_index(0.5, y);
                f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
                let cell_index = f.cell_index(1.5, y);
                f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            }

            // Cells don't intersect with ray.
            let cell_index = f.cell_index(0.5, 6.5);
            f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            let cell_index = f.cell_index(-0.5, -1.5);
            f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            for _ in 0..1000 {
                f.insert_point();
            }
            for y in ys(1.5) {
                // Cell intersects with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                f.expect_cell_properties(&cell_index, true, expected_tsdf, maximum_weight);
            }
        }

        #[test]
        fn insert_point_with_free_space_update() {
            let mut f = Fixture::new();
            f.options.update_free_space = true;
            f.recreate_inserter();
            f.insert_point();
            let truncation_distance = f.options.truncation_distance as f32;
            let maximum_weight = f.options.maximum_weight as f32;

            for y in ys(-0.5) {
                // Cells intersect with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                f.expect_cell_properties(&cell_index, true, expected_tsdf, 1.0);
                // Cells don't intersect with ray.
                let cell_index = f.cell_index(0.5, y);
                f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
                let cell_index = f.cell_index(1.5, y);
                f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            }

            // Cells don't intersect with the ray.
            let cell_index = f.cell_index(-0.5, 6.5);
            f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            let cell_index = f.cell_index(-0.5, -1.5);
            f.expect_cell_properties(&cell_index, false, -truncation_distance, 0.0);
            for _ in 0..1000 {
                f.insert_point();
            }
            for y in ys(-0.5) {
                // Cell intersects with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                f.expect_cell_properties(&cell_index, true, expected_tsdf, maximum_weight);
            }
        }

        #[test]
        fn insert_point_linear_weight() {
            let mut f = Fixture::new();
            f.options.update_weight_range_exponent = 1;
            f.recreate_inserter();
            f.insert_point();
            let truncation_distance = f.options.truncation_distance as f32;
            for y in ys(1.5) {
                // Cell intersects with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                let expected_weight = 1.0 / 4.0;
                f.expect_cell_properties(&cell_index, true, expected_tsdf, expected_weight);
            }
        }

        #[test]
        fn insert_point_quadratic_weight() {
            let mut f = Fixture::new();
            f.options.update_weight_range_exponent = 2;
            f.recreate_inserter();
            f.insert_point();
            let truncation_distance = f.options.truncation_distance as f32;
            for y in ys(1.5) {
                // Cell intersects with ray.
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                let expected_weight = 1.0 / 4.0_f32.powi(2);
                f.expect_cell_properties(&cell_index, true, expected_tsdf, expected_weight);
            }
        }

        #[test]
        fn insert_small_angle_point_without_normal_projection() {
            let mut f = Fixture::new();
            let mut range_data = RangeData::default();
            range_data.returns.push_back(point(-0.5, 3.5));
            range_data.returns.push_back(point(5.5, 3.5));
            range_data.returns.push_back(point(10.5, 3.5));
            range_data.origin.x = -0.5;
            range_data.origin.y = -0.5;
            f.range_data_inserter.insert(&range_data, &mut f.tsdf);
            f.tsdf.finish_update();
            let (x, y) = (4.5_f32, 2.5_f32);
            let cell_index = f.cell_index(x, y);
            let ray = Vector2::new(-0.5_f32, -0.5) - Vector2::new(5.5, 3.5);
            let ray_length = ray.norm();
            let origin_to_cell = Vector2::new(x, y) - Vector2::new(-0.5, -0.5);
            let distance_origin_to_cell = origin_to_cell.norm();
            let expected_tsdf = ray_length - distance_origin_to_cell;
            f.expect_cell_properties(&cell_index, true, expected_tsdf, 1.0);
        }

        #[test]
        fn insert_small_angle_point_wit_normal_projection() {
            let mut f = Fixture::new();
            f.options.project_sdf_distance_to_scan_normal = true;
            f.recreate_inserter();
            let mut range_data = RangeData::default();
            range_data.returns.push_back(point(-0.5, 3.5));
            range_data.returns.push_back(point(5.5, 3.5));
            range_data.origin.x = -0.5;
            range_data.origin.y = -0.5;
            f.range_data_inserter.insert(&range_data, &mut f.tsdf);
            f.tsdf.finish_update();
            let cell_index = f.cell_index(4.5, 2.5);
            f.expect_cell_properties(&cell_index, true, 1.0, 1.0);
            let cell_index = f.cell_index(6.5, 4.5);
            f.expect_cell_properties(&cell_index, true, -1.0, 1.0);
        }

        #[test]
        fn insert_points_with_angle_scan_normal_to_ray_weight() {
            let mut f = Fixture::new();
            let bandwidth = 10.0_f32;
            f.options
                .update_weight_angle_scan_normal_to_ray_kernel_bandwidth = f64::from(bandwidth);
            f.recreate_inserter();
            let mut range_data = RangeData::default();
            range_data.returns.push_back(point(-0.5, 3.5));
            range_data.returns.push_back(point(5.5, 3.5));
            range_data.origin.x = -0.5;
            range_data.origin.y = -0.5;
            f.range_data_inserter.insert(&range_data, &mut f.tsdf);
            f.tsdf.finish_update();
            // Ray is perpendicular to surface.
            let cell_index = f.cell_index(-0.5, 3.5);
            let mut expected_weight = (1.0 / ((2.0 * PI).sqrt() * f64::from(bandwidth))) as f32;
            assert!((expected_weight - f.tsdf.get_weight(&cell_index)).abs() <= 1e-3);
            let cell_index = f.cell_index(6.5, 4.5);
            assert!((expected_weight - f.tsdf.get_weight(&cell_index)).abs() <= 1e-3);
            // Ray is inclined relative to surface.
            // (The C++ test uses a positive exponent; it passes only thanks to
            // the 1e-3 tolerance. Kept as in C++.)
            let angle = (7.0_f32 / 5.0).atan();
            expected_weight = (1.0 / ((2.0 * PI).sqrt() * f64::from(bandwidth))
                * (f64::from(angle * angle) / (2.0 * f64::from(bandwidth).powi(2))).exp())
                as f32;
            assert!((expected_weight - f.tsdf.get_weight(&cell_index)).abs() <= 1e-3);
            let cell_index = f.cell_index(6.5, 4.5);
            assert!((expected_weight - f.tsdf.get_weight(&cell_index)).abs() <= 1e-3);
        }

        #[test]
        fn insert_points_with_distance_cell_to_hit() {
            let mut f = Fixture::new();
            let bandwidth = 10.0_f32;
            f.options
                .update_weight_distance_cell_to_hit_kernel_bandwidth = f64::from(bandwidth);
            f.recreate_inserter();
            f.insert_point();
            let truncation_distance = f.options.truncation_distance as f32;
            for y in ys(1.5) {
                let cell_index = f.cell_index(-0.5, y);
                let expected_tsdf = (3.5 - y).min(truncation_distance).max(-truncation_distance);
                // Same positive-exponent quirk as in C++ (tolerance 1e-2).
                let expected_weight = (1.0 / ((2.0 * PI).sqrt() * f64::from(bandwidth))
                    * (f64::from(expected_tsdf).powi(2) / (2.0 * f64::from(bandwidth).powi(2)))
                        .exp()) as f32;
                f.expect_cell_properties(&cell_index, true, expected_tsdf, expected_weight);
            }
        }

        // Additional tests (not in C++).

        #[test]
        fn gaussian_kernel_and_range_weight() {
            let value = gaussian_kernel(0.0, 1.0);
            assert!((f64::from(value) - 1.0 / (2.0 * PI).sqrt()).abs() < 1e-6);
            assert!((compute_range_weight_factor(4.0, 2) - 1.0 / 16.0).abs() < 1e-7);
            assert_eq!(compute_range_weight_factor(0.0, 1), 0.0);
        }

        #[test]
        fn sorter_orders_by_bearing() {
            let mut deltas: Vec<Vector2<f32>> = (0..16)
                .map(|i| {
                    let angle = -std::f32::consts::PI + 0.1 + i as f32 * 0.39;
                    Vector2::new(angle.cos(), angle.sin())
                })
                .rev()
                .collect();
            deltas.sort_unstable_by(|a, b| {
                if range_data_sorter_less(a, b) {
                    Ordering::Less
                } else if range_data_sorter_less(b, a) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            let angles: Vec<f32> = deltas.iter().map(|d| d.y.atan2(d.x)).collect();
            for w in angles.windows(2) {
                assert!(w[0] < w[1], "{angles:?}");
            }
        }

        #[test]
        fn grow_ignores_zero_length_rays() {
            let mut f = Fixture::new();
            let range_data = RangeData {
                origin: Vector3::new(-0.5, -0.5, 0.0),
                returns: PointCloud::from_points(vec![point(-0.5, -0.5)]),
                misses: PointCloud::new(),
            };
            f.range_data_inserter.insert(&range_data, &mut f.tsdf);
            // The zero-length ray does not produce NaNs; the hit is skipped.
            assert!(f.tsdf.limits().max().iter().all(|v| v.is_finite()));
            let cell_index = f.cell_index(-0.5, -0.5);
            assert!(f.tsdf.limits().contains(&cell_index));
            assert!(!f.tsdf.is_known(&cell_index));
        }
    }
}
