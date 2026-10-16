//! Port of `cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d` (.h, .cc).

use ceres::{LinearSolverType, Problem, SolverOptions, SolverSummary, solve};
use nalgebra::Vector2;

use crate::common::internal::ceres_solver_options::{
    create_ceres_solver_options, create_ceres_solver_options_proto,
};
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::internal::two_d::scan_matching::occupied_space_cost_function_2d::create_occupied_space_cost_function_2d;
use crate::mapping::internal::two_d::scan_matching::rotation_delta_cost_functor_2d::RotationDeltaCostFunctor2D;
use crate::mapping::internal::two_d::scan_matching::translation_delta_cost_functor_2d::TranslationDeltaCostFunctor2D;
use crate::mapping::internal::two_d::scan_matching::tsdf_match_cost_function_2d::create_tsdf_match_cost_function_2d;
use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::mapping::scan_matching::proto;
use crate::mapping::two_d::grid_2d::{Grid2DTrait, GridType};
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid2d;

/// Reads `CeresScanMatcherOptions2D` from `parameter_dictionary`.
pub fn create_ceres_scan_matcher_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::CeresScanMatcherOptions2D {
    proto::CeresScanMatcherOptions2D {
        occupied_space_weight: parameter_dictionary.get_double("occupied_space_weight"),
        translation_weight: parameter_dictionary.get_double("translation_weight"),
        rotation_weight: parameter_dictionary.get_double("rotation_weight"),
        ceres_solver_options: Some(create_ceres_solver_options_proto(
            &parameter_dictionary.get_dictionary("ceres_solver_options"),
        )),
    }
}

/// Align scans with an existing map using Ceres.
#[derive(Debug)]
pub struct CeresScanMatcher2D {
    options: proto::CeresScanMatcherOptions2D,
    ceres_solver_options: SolverOptions,
}

impl CeresScanMatcher2D {
    /// Creates a matcher; the solver uses `DENSE_QR`.
    pub fn new(options: &proto::CeresScanMatcherOptions2D) -> Self {
        let mut ceres_solver_options =
            create_ceres_solver_options(&options.ceres_solver_options.unwrap_or_default());
        ceres_solver_options.linear_solver_type = LinearSolverType::DenseQr;
        Self {
            options: *options,
            ceres_solver_options,
        }
    }

    /// Aligns `point_cloud` within the `grid` given an
    /// `initial_pose_estimate` and returns the new pose estimate. The
    /// translation prior pulls towards `target_translation`, the rotation
    /// prior towards the angle of `initial_pose_estimate`. `summary` is
    /// filled with the solver summary.
    pub fn match_(
        &self,
        target_translation: &Vector2<f64>,
        initial_pose_estimate: &Rigid2d,
        point_cloud: &PointCloud,
        grid: &dyn Grid2DTrait,
        summary: &mut SolverSummary,
    ) -> Rigid2d {
        let ceres_pose_estimate = [
            initial_pose_estimate.translation().x,
            initial_pose_estimate.translation().y,
            initial_pose_estimate.rotation().angle(),
        ];
        let mut problem = Problem::new();
        let pose = problem.add_parameter_block(&ceres_pose_estimate);
        assert!(
            self.options.occupied_space_weight > 0.0,
            "Check failed: options_.occupied_space_weight() > 0. ({} vs. 0)",
            self.options.occupied_space_weight
        );
        let scaling_factor =
            self.options.occupied_space_weight / (point_cloud.size() as f64).sqrt();
        match grid.get_grid_type() {
            GridType::ProbabilityGrid => {
                problem.add_residual_block(
                    create_occupied_space_cost_function_2d(scaling_factor, point_cloud, grid),
                    None, // loss function
                    &[pose],
                );
            }
            GridType::Tsdf => {
                let tsdf = grid
                    .downcast_ref::<TSDF2D>()
                    .expect("TSDF must be a TSDF2D");
                problem.add_residual_block(
                    create_tsdf_match_cost_function_2d(scaling_factor, point_cloud, tsdf),
                    None, // loss function
                    &[pose],
                );
            }
        }
        assert!(
            self.options.translation_weight > 0.0,
            "Check failed: options_.translation_weight() > 0. ({} vs. 0)",
            self.options.translation_weight
        );
        problem.add_residual_block(
            TranslationDeltaCostFunctor2D::create_auto_diff_cost_function(
                self.options.translation_weight,
                target_translation,
            ),
            None, // loss function
            &[pose],
        );
        assert!(
            self.options.rotation_weight > 0.0,
            "Check failed: options_.rotation_weight() > 0. ({} vs. 0)",
            self.options.rotation_weight
        );
        problem.add_residual_block(
            RotationDeltaCostFunctor2D::create_auto_diff_cost_function(
                self.options.rotation_weight,
                ceres_pose_estimate[2],
            ),
            None, // loss function
            &[pose],
        );

        solve(&self.ceres_solver_options, &mut problem, summary);

        let result = problem.parameter_block(pose);
        Rigid2d::new(Vector2::new(result[0], result[1]), result[2])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::mapping::probability_values::MAX_PROBABILITY;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;

    /// C++ fixture `CeresScanMatcherTest`.
    struct Fixture {
        probability_grid: ProbabilityGrid,
        point_cloud: PointCloud,
        ceres_scan_matcher: CeresScanMatcher2D,
    }

    impl Fixture {
        fn new() -> Self {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(10.0, 10.0), CellLimits::new(20, 20)),
                &conversion_tables,
            );
            let cell = probability_grid
                .limits()
                .get_cell_index(&Vector2::new(-3.5, 2.5));
            probability_grid.set_probability(&cell, MAX_PROBABILITY);

            let point_cloud = PointCloud::from_points(vec![RangefinderPoint {
                position: nalgebra::Vector3::new(-3.0, 2.0, 0.0),
            }]);

            let parameter_dictionary = make_dictionary(
                r"
        return {
          occupied_space_weight = 1.,
          translation_weight = 0.1,
          rotation_weight = 1.5,
          ceres_solver_options = {
            use_nonmonotonic_steps = true,
            max_num_iterations = 50,
            num_threads = 1,
          },
        }",
            );
            let options = create_ceres_scan_matcher_options_2d(&parameter_dictionary);
            Self {
                probability_grid,
                point_cloud,
                ceres_scan_matcher: CeresScanMatcher2D::new(&options),
            }
        }

        fn test_from_initial_pose(&self, initial_pose: &Rigid2d) {
            let expected_pose = Rigid2d::from_translation(Vector2::new(-0.5, 0.5));
            let mut summary = SolverSummary::default();
            let pose = self.ceres_scan_matcher.match_(
                initial_pose.translation(),
                initial_pose,
                &self.point_cloud,
                &self.probability_grid,
                &mut summary,
            );
            assert!(
                summary.final_cost.abs() <= 1e-2,
                "{}",
                summary.full_report()
            );
            assert_is_nearly(&pose, &expected_pose, 1e-2);
        }
    }

    mod ceres_scan_matcher_test {
        use super::*;

        #[test]
        fn test_perfect_estimate() {
            Fixture::new()
                .test_from_initial_pose(&Rigid2d::from_translation(Vector2::new(-0.5, 0.5)));
        }

        #[test]
        fn test_optimize_along_x() {
            Fixture::new()
                .test_from_initial_pose(&Rigid2d::from_translation(Vector2::new(-0.3, 0.5)));
        }

        #[test]
        fn test_optimize_along_y() {
            Fixture::new()
                .test_from_initial_pose(&Rigid2d::from_translation(Vector2::new(-0.45, 0.3)));
        }

        #[test]
        fn test_optimize_along_xy() {
            Fixture::new()
                .test_from_initial_pose(&Rigid2d::from_translation(Vector2::new(-0.3, 0.3)));
        }
    }
}
