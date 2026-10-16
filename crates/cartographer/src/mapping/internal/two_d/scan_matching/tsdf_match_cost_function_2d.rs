//! Port of `cartographer/mapping/internal/2d/scan_matching/tsdf_match_cost_function_2d` (.h, .cc).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};

use crate::mapping::internal::two_d::scan_matching::interpolated_tsdf_2d::InterpolatedTSDF2D;
use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::sensor::point_cloud::PointCloud;

/// Computes a cost for matching the `point_cloud` in the `grid` at a
/// `pose`. The cost increases with the signed distance of the matched point
/// location in the `grid`.
struct TSDFMatchCostFunction2D<'a> {
    residual_scaling_factor: f64,
    point_cloud: &'a PointCloud,
    interpolated_grid: InterpolatedTSDF2D<'a>,
}

impl CostFunctor for TSDFMatchCostFunction2D<'_> {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let pose = parameters[0];
        let (sin, cos) = (pose[2].sin(), pose[2].cos());
        let num_points = T::constant(self.point_cloud.size() as f64);
        let scaling_factor = T::constant(self.residual_scaling_factor);

        let mut summed_weight = T::zero();
        for (point, r) in self.point_cloud.iter().zip(residual.iter_mut()) {
            // Note that this is a 2D point. The third (homogeneous)
            // component is 1.
            let x = T::constant(f64::from(point.position.x));
            let y = T::constant(f64::from(point.position.y));
            let world_x = cos * x + -sin * y + pose[0];
            let world_y = sin * x + cos * y + pose[1];
            let point_weight = self.interpolated_grid.get_weight(world_x, world_y);
            summed_weight += point_weight;
            *r = num_points
                * scaling_factor
                * self
                    .interpolated_grid
                    .get_correspondence_cost(world_x, world_y)
                * point_weight;
        }
        if summed_weight == T::zero() {
            return false;
        }
        for r in residual.iter_mut().take(self.point_cloud.size()) {
            *r /= summed_weight;
        }
        true
    }
}

/// Creates a cost function for matching the `point_cloud` to the `tsdf`
/// with a pose `[x, y, theta]`; one residual per point. Evaluation fails if
/// all points fall into cells without weight. The cost function borrows
/// `point_cloud` and `tsdf`.
pub fn create_tsdf_match_cost_function_2d<'a>(
    scaling_factor: f64,
    point_cloud: &'a PointCloud,
    tsdf: &'a TSDF2D,
) -> Box<dyn CostFunction + 'a> {
    Box::new(AutoDiffCostFunction::<_, 3>::new(
        TSDFMatchCostFunction2D {
            residual_scaling_factor: scaling_factor,
            point_cloud,
            interpolated_grid: InterpolatedTSDF2D::new(tsdf),
        },
        point_cloud.size(), // residuals
        &[3],               // pose variables
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::{Vector2, Vector3};

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::mapping::internal::two_d::tsdf_range_data_inserter_2d::{
        TSDFRangeDataInserter2D, create_tsdf_range_data_inserter_options_2d,
    };
    use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
    use crate::mapping::two_d::grid_2d::Grid2DTrait;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::range_data::RangeData;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    /// C++ fixture `TSDFSpaceCostFunction2DTest`.
    struct Fixture {
        tsdf: TSDF2D,
        range_data_inserter: TSDFRangeDataInserter2D,
    }

    impl Fixture {
        fn new() -> Self {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let tsdf = TSDF2D::new(
                MapLimits::new(0.1, Vector2::new(2.05, 2.05), CellLimits::new(40, 40)),
                0.3,
                1.0,
                &conversion_tables,
            );
            let parameter_dictionary = make_dictionary(
                "return { \
                 truncation_distance = 0.3,\
                 maximum_weight = 1.0,\
                 update_free_space = false,\
                 normal_estimation_options = {\
                 num_normal_samples = 2,\
                 sample_radius = 10.,\
                 },\
                 project_sdf_distance_to_scan_normal = true,\
                 update_weight_range_exponent = 0,\
                 update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0,\
                 update_weight_distance_cell_to_hit_kernel_bandwidth = 0,\
                 }",
            );
            let options = create_tsdf_range_data_inserter_options_2d(&parameter_dictionary);
            Self {
                tsdf,
                range_data_inserter: TSDFRangeDataInserter2D::new(&options),
            }
        }

        fn insert_pointcloud(&mut self) {
            let mut range_data = RangeData::default();
            let mut x = -0.5_f32;
            while x < 0.5 {
                range_data.returns.push_back(RangefinderPoint {
                    position: Vector3::new(x, 1.0, 0.0),
                });
                x += 0.1;
            }
            range_data.origin.x = -0.5;
            range_data.origin.y = -0.5;
            self.range_data_inserter.insert(&range_data, &mut self.tsdf);
            self.tsdf.finish_update();
        }
    }

    fn matching_cloud(x: f32, y: f32) -> PointCloud {
        PointCloud::from_points(vec![RangefinderPoint {
            position: Vector3::new(x, y, 0.0),
        }])
    }

    /// Evaluates the cost function with Jacobian at `pose`.
    fn evaluate(cost_function: &dyn CostFunction, pose: [f64; 3]) -> (bool, [f64; 1], [f64; 3]) {
        let mut residuals = [0.0; 1];
        let mut jacobians = [0.0; 3];
        let valid = cost_function.evaluate(
            &[&pose],
            &mut residuals,
            Some(&mut [Some(&mut jacobians[..])]),
        );
        (valid, residuals, jacobians)
    }

    fn assert_near(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tolerance, "{actual:?} vs. {expected:?}");
        }
    }

    mod tsdf_space_cost_function_2d_test {
        use super::*;

        #[test]
        fn match_empty_tsdf() {
            let fixture = Fixture::new();
            let matching_cloud = matching_cloud(0.0, 0.0);
            let cost_function =
                create_tsdf_match_cost_function_2d(1.0, &matching_cloud, &fixture.tsdf);
            let (valid_result, _, _) = evaluate(cost_function.as_ref(), [0.0, 0.0, 0.0]);
            assert!(!valid_result);
        }

        #[test]
        fn exact_initial_pose() {
            let mut fixture = Fixture::new();
            fixture.insert_pointcloud();
            let matching_cloud = matching_cloud(0.0, 1.0);
            let cost_function =
                create_tsdf_match_cost_function_2d(1.0, &matching_cloud, &fixture.tsdf);
            let (valid_result, residuals, jacobians) =
                evaluate(cost_function.as_ref(), [0.0, 0.0, 0.0]);
            assert!(valid_result);
            assert_near(&residuals, &[0.0], 1e-3);
            assert_near(&jacobians, &[0.0, -1.0, 0.0], 1e-3);
        }

        #[test]
        fn pertubated_initial_pose() {
            let mut fixture = Fixture::new();
            fixture.insert_pointcloud();
            let matching_cloud = matching_cloud(0.0, 1.0);
            let cost_function =
                create_tsdf_match_cost_function_2d(1.0, &matching_cloud, &fixture.tsdf);
            let (valid_result, residuals, jacobians) =
                evaluate(cost_function.as_ref(), [0.0, 0.1, 0.0]);
            assert!(valid_result);
            assert_near(&residuals, &[-0.1], 1e-3);
            assert_near(&jacobians, &[0.0, -1.0, 0.0], 1e-3);

            let (valid_result, residuals, jacobians) =
                evaluate(cost_function.as_ref(), [0.0, -0.1, 0.0]);
            assert!(valid_result);
            assert_near(&residuals, &[0.1], 1e-3);
            assert_near(&jacobians, &[0.0, -1.0, 0.0], 1e-3);
        }

        #[test]
        fn invalid_initial_pose() {
            let mut fixture = Fixture::new();
            fixture.insert_pointcloud();
            let matching_cloud = matching_cloud(0.0, 1.0);
            let cost_function =
                create_tsdf_match_cost_function_2d(1.0, &matching_cloud, &fixture.tsdf);
            let (valid_result, _, _) = evaluate(cost_function.as_ref(), [0.0, 0.4, 0.0]);
            assert!(!valid_result);
            let (valid_result, _, _) = evaluate(cost_function.as_ref(), [0.0, -0.4, 0.0]);
            assert!(!valid_result);
        }
    }
}
