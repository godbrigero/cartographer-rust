//! Port of `cartographer/mapping/internal/3d/scan_matching/ceres_scan_matcher_3d` (.h, .cc).

use ceres::{
    AutoDiffLocalParameterization, HuberLoss, LinearSolverType, LocalParameterization, Problem,
    QuaternionParameterization, SolverOptions, SolverSummary,
};
use nalgebra::Vector3;

use crate::common::internal::ceres_solver_options::{
    create_ceres_solver_options, create_ceres_solver_options_proto,
};
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::internal::optimization::ceres_pose::CeresPose;
use crate::mapping::internal::three_d::rotation_parameterization::YawOnlyQuaternionPlus;
use crate::mapping::internal::three_d::scan_matching::intensity_cost_function_3d::IntensityCostFunction3D;
use crate::mapping::internal::three_d::scan_matching::occupied_space_cost_function_3d::OccupiedSpaceCostFunction3D;
use crate::mapping::internal::three_d::scan_matching::rotation_delta_cost_functor_3d::RotationDeltaCostFunctor3D;
use crate::mapping::internal::three_d::scan_matching::translation_delta_cost_functor_3d::TranslationDeltaCostFunctor3D;
use crate::mapping::scan_matching::proto;
use crate::mapping::three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid3d;

/// Reads `CeresScanMatcherOptions3D` from `parameter_dictionary`.
pub fn create_ceres_scan_matcher_options_3d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::CeresScanMatcherOptions3D {
    let mut options = proto::CeresScanMatcherOptions3D::default();
    for i in 0.. {
        let lua_identifier = format!("occupied_space_weight_{i}");
        if !parameter_dictionary.has_key(&lua_identifier) {
            break;
        }
        options
            .occupied_space_weight
            .push(parameter_dictionary.get_double(&lua_identifier));
    }
    for i in 0.. {
        let lua_identifier = format!("intensity_cost_function_options_{i}");
        if !parameter_dictionary.has_key(&lua_identifier) {
            break;
        }
        let intensity_cost_function_options_dictionary =
            parameter_dictionary.get_dictionary(&lua_identifier);
        options
            .intensity_cost_function_options
            .push(proto::IntensityCostFunctionOptions {
                weight: intensity_cost_function_options_dictionary.get_double("weight"),
                huber_scale: intensity_cost_function_options_dictionary.get_double("huber_scale"),
                intensity_threshold: intensity_cost_function_options_dictionary
                    .get_double("intensity_threshold") as f32,
            });
    }
    options.translation_weight = parameter_dictionary.get_double("translation_weight");
    options.rotation_weight = parameter_dictionary.get_double("rotation_weight");
    options.only_optimize_yaw = parameter_dictionary.get_bool("only_optimize_yaw");
    options.ceres_solver_options = Some(create_ceres_solver_options_proto(
        &parameter_dictionary.get_dictionary("ceres_solver_options"),
    ));
    options
}

/// A point cloud with the grids to match it against (C++
/// `PointCloudAndHybridGridsPointers`).
#[derive(Clone, Copy)]
pub struct PointCloudAndHybridGridsPointers<'a> {
    /// The point cloud (in the tracking frame).
    pub point_cloud: &'a PointCloud,
    /// The occupancy grid.
    pub hybrid_grid: &'a HybridGrid,
    /// Optional intensity grid (C++ nullable pointer).
    pub intensity_hybrid_grid: Option<&'a IntensityHybridGrid>,
}

/// This scan matcher uses Ceres to align scans with an existing map.
pub struct CeresScanMatcher3D {
    options: proto::CeresScanMatcherOptions3D,
    ceres_solver_options: SolverOptions,
}

impl CeresScanMatcher3D {
    /// Creates the matcher; the solver uses `DENSE_QR`.
    pub fn new(options: &proto::CeresScanMatcherOptions3D) -> Self {
        let mut ceres_solver_options = create_ceres_solver_options(
            options
                .ceres_solver_options
                .as_ref()
                .expect("CeresScanMatcherOptions3D.ceres_solver_options is required"),
        );
        ceres_solver_options.linear_solver_type = LinearSolverType::DenseQr;
        Self {
            options: options.clone(),
            ceres_solver_options,
        }
    }

    /// Aligns `point_clouds_and_hybrid_grids` within the grids given an
    /// `initial_pose_estimate` and returns a `pose_estimate` and the solver
    /// `summary`. The translation prior targets `target_translation`, the
    /// rotation prior the rotation of `initial_pose_estimate`.
    pub fn match_(
        &self,
        target_translation: &Vector3<f64>,
        initial_pose_estimate: &Rigid3d,
        point_clouds_and_hybrid_grids: &[PointCloudAndHybridGridsPointers<'_>],
        pose_estimate: &mut Rigid3d,
        summary: &mut SolverSummary,
    ) {
        let mut problem = Problem::new();
        let rotation_parameterization: Box<dyn LocalParameterization> =
            if self.options.only_optimize_yaw {
                Box::new(AutoDiffLocalParameterization::<YawOnlyQuaternionPlus, 4, 1>::default())
            } else {
                Box::new(QuaternionParameterization)
            };
        let ceres_pose = CeresPose::new(
            initial_pose_estimate,
            None, /* translation_parameterization */
            Some(rotation_parameterization),
            &mut problem,
        );

        assert_eq!(
            self.options.occupied_space_weight.len(),
            point_clouds_and_hybrid_grids.len(),
            "Check failed: options_.occupied_space_weight_size() == point_clouds_and_hybrid_grids.size()"
        );
        for (i, input) in point_clouds_and_hybrid_grids.iter().enumerate() {
            let occupied_space_weight = self.options.occupied_space_weight[i];
            assert!(
                occupied_space_weight > 0.,
                "Check failed: options_.occupied_space_weight(i) > 0."
            );
            let point_cloud = input.point_cloud;
            let num_points_sqrt = (point_cloud.size() as f64).sqrt();
            problem.add_residual_block(
                Box::new(OccupiedSpaceCostFunction3D::create_auto_diff_cost_function(
                    occupied_space_weight / num_points_sqrt,
                    point_cloud,
                    input.hybrid_grid,
                )),
                None, /* loss function */
                &[ceres_pose.translation(), ceres_pose.rotation()],
            );
            if let Some(intensity_hybrid_grid) = input.intensity_hybrid_grid {
                let intensity_options = &self.options.intensity_cost_function_options[i];
                assert!(
                    intensity_options.huber_scale > 0.,
                    "Check failed: intensity_cost_function_options(i).huber_scale() > 0."
                );
                assert!(
                    intensity_options.weight > 0.,
                    "Check failed: intensity_cost_function_options(i).weight() > 0."
                );
                assert!(
                    intensity_options.intensity_threshold > 0.,
                    "Check failed: intensity_cost_function_options(i).intensity_threshold() > 0"
                );
                problem.add_residual_block(
                    Box::new(IntensityCostFunction3D::create_auto_diff_cost_function(
                        intensity_options.weight / num_points_sqrt,
                        intensity_options.intensity_threshold,
                        point_cloud,
                        intensity_hybrid_grid,
                    )),
                    Some(Box::new(HuberLoss::new(intensity_options.huber_scale))),
                    &[ceres_pose.translation(), ceres_pose.rotation()],
                );
            }
        }

        assert!(
            self.options.translation_weight > 0.,
            "Check failed: options_.translation_weight() > 0."
        );
        problem.add_residual_block(
            Box::new(
                TranslationDeltaCostFunctor3D::create_auto_diff_cost_function(
                    self.options.translation_weight,
                    target_translation,
                ),
            ),
            None, /* loss function */
            &[ceres_pose.translation()],
        );
        assert!(
            self.options.rotation_weight > 0.,
            "Check failed: options_.rotation_weight() > 0."
        );
        problem.add_residual_block(
            Box::new(RotationDeltaCostFunctor3D::create_auto_diff_cost_function(
                self.options.rotation_weight,
                initial_pose_estimate.rotation(),
            )),
            None, /* loss function */
            &[ceres_pose.rotation()],
        );

        ceres::solve(&self.ceres_solver_options, &mut problem, summary);

        *pose_estimate = ceres_pose.to_rigid(&problem);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::sensor::point_cloud::transform_point_cloud;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::AngleAxis;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use nalgebra::Vector3;

    mod ceres_scan_matcher_3d_test {
        use super::*;

        struct Fixture {
            hybrid_grid: HybridGrid,
            intensity_hybrid_grid: IntensityHybridGrid,
            expected_pose: Rigid3d,
            point_cloud: PointCloud,
            options: proto::CeresScanMatcherOptions3D,
            ceres_scan_matcher: CeresScanMatcher3D,
        }

        impl Fixture {
            fn new() -> Self {
                let mut hybrid_grid = HybridGrid::new(1.);
                let mut intensity_hybrid_grid = IntensityHybridGrid::new(1.);
                let expected_pose = Rigid3d::from_translation(Vector3::new(-1., 0., 0.));
                let mut points = Vec::new();
                let mut intensities = Vec::new();
                for point in [
                    Vector3::new(-3.0f32, 2., 0.),
                    Vector3::new(-4., 2., 0.),
                    Vector3::new(-5., 2., 0.),
                    Vector3::new(-6., 2., 0.),
                    Vector3::new(-6., 3., 1.),
                    Vector3::new(-6., 4., 2.),
                    Vector3::new(-7., 3., 1.),
                ] {
                    points.push(RangefinderPoint { position: point });
                    intensities.push(50.);
                    let world = expected_pose.cast::<f32>() * point;
                    let index = hybrid_grid.get_cell_index(&world);
                    hybrid_grid.set_probability(&index, 1.);
                    let index = intensity_hybrid_grid.get_cell_index(&world);
                    intensity_hybrid_grid.add_intensity(&index, 50.);
                }
                let point_cloud = PointCloud::with_intensities(points, intensities);

                let parameter_dictionary = make_dictionary(
                    r#"
        return {
          occupied_space_weight_0 = 1.,
          intensity_cost_function_options_0 = {
            weight = 0.5,
            huber_scale = 55,
            intensity_threshold = 100,
          },
          translation_weight = 0.01,
          rotation_weight = 0.1,
          only_optimize_yaw = false,
          ceres_solver_options = {
            use_nonmonotonic_steps = true,
            max_num_iterations = 10,
            num_threads = 1,
          },
        }"#,
                );
                let options = create_ceres_scan_matcher_options_3d(&parameter_dictionary);
                let ceres_scan_matcher = CeresScanMatcher3D::new(&options);
                Self {
                    hybrid_grid,
                    intensity_hybrid_grid,
                    expected_pose,
                    point_cloud,
                    options,
                    ceres_scan_matcher,
                }
            }

            fn test_from_initial_pose(&self, initial_pose: &Rigid3d) {
                let mut pose = Rigid3d::identity();
                let mut summary = SolverSummary::default();
                let intensity_hybrid_grid = if self.point_cloud.intensities().is_empty() {
                    None
                } else {
                    Some(&self.intensity_hybrid_grid)
                };
                self.ceres_scan_matcher.match_(
                    initial_pose.translation(),
                    initial_pose,
                    &[PointCloudAndHybridGridsPointers {
                        point_cloud: &self.point_cloud,
                        hybrid_grid: &self.hybrid_grid,
                        intensity_hybrid_grid,
                    }],
                    &mut pose,
                    &mut summary,
                );
                assert!(
                    summary.final_cost.abs() <= 1e-2,
                    "{}",
                    summary.full_report()
                );
                assert_is_nearly(&pose, &self.expected_pose, 3e-2);
            }
        }

        #[test]
        fn perfect_estimate() {
            let fixture = Fixture::new();
            fixture.test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-1., 0., 0.)));
        }

        #[test]
        fn along_x() {
            let mut fixture = Fixture::new();
            fixture.ceres_scan_matcher = CeresScanMatcher3D::new(&fixture.options);
            fixture.test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-0.8, 0., 0.)));
        }

        #[test]
        fn along_z() {
            let fixture = Fixture::new();
            fixture.test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-1., 0., -0.2)));
        }

        #[test]
        fn along_xyz() {
            let fixture = Fixture::new();
            fixture
                .test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-0.9, -0.2, 0.2)));
        }

        #[test]
        fn full_pose_correction() {
            let mut fixture = Fixture::new();
            // We try to find the rotation around z...
            let additional_transform =
                Rigid3d::from_angle_axis(AngleAxis::new(0.05, Vector3::new(0., 0., 1.)));
            fixture.point_cloud =
                transform_point_cloud(&fixture.point_cloud, &additional_transform.cast::<f32>());
            fixture.expected_pose = fixture.expected_pose * additional_transform.inverse();
            // ...starting initially with rotation around x.
            fixture.test_from_initial_pose(&Rigid3d::with_angle_axis(
                Vector3::new(-0.95, -0.05, 0.05),
                AngleAxis::new(0.05, Vector3::new(1., 0., 0.)),
            ));
        }
    }

    mod additional {
        use super::*;
        use crate::mapping::internal::three_d::scan_matching::interpolated_grid::InterpolatedProbabilityGrid;
        use crate::mapping::three_d::hybrid_grid::Array3i;
        use crate::transform::transform::get_angle;

        /// Samples the six walls of an axis-aligned box room with corners
        /// `min` and `max` on a regular grid with spacing `step`, plus a few
        /// pillars to break the symmetry.
        fn box_room_points(min: Vector3<f32>, max: Vector3<f32>, step: f32) -> Vec<Vector3<f32>> {
            let mut points = Vec::new();
            let steps = |a: f32, b: f32| {
                let n = ((b - a) / step).round() as i32;
                (0..=n).map(move |i| a + i as f32 * step)
            };
            for u in steps(min.x, max.x) {
                for v in steps(min.y, max.y) {
                    points.push(Vector3::new(u, v, min.z));
                    points.push(Vector3::new(u, v, max.z));
                }
                for v in steps(min.z, max.z) {
                    points.push(Vector3::new(u, min.y, v));
                    points.push(Vector3::new(u, max.y, v));
                }
            }
            for u in steps(min.y, max.y) {
                for v in steps(min.z, max.z) {
                    points.push(Vector3::new(min.x, u, v));
                    points.push(Vector3::new(max.x, u, v));
                }
            }
            // Pillars.
            for (cx, cy) in [(1.5f32, 0.8f32), (-2.0, -1.2)] {
                for v in steps(min.z, max.z) {
                    for k in 0..12 {
                        let a = k as f32 * std::f32::consts::PI / 6.;
                        points.push(Vector3::new(cx + 0.3 * a.cos(), cy + 0.3 * a.sin(), v));
                    }
                }
            }
            points
        }

        /// Builds a smooth probability map (occupied voxels and a decaying
        /// neighborhood, similar to what range data insertion produces).
        fn build_grid(points: &[Vector3<f32>], resolution: f32) -> HybridGrid {
            let mut grid = HybridGrid::new(resolution);
            let mut max_probability = std::collections::HashMap::new();
            for point in points {
                let center = grid.get_cell_index(point);
                for dz in -2..=2 {
                    for dy in -2..=2 {
                        for dx in -2..=2 {
                            let index = center + Array3i::new(dx, dy, dz);
                            let distance = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                            let probability = 0.9 - 0.25 * distance;
                            if probability <= 0.1 {
                                continue;
                            }
                            let entry = max_probability
                                .entry((index.x, index.y, index.z))
                                .or_insert(0.1f32);
                            *entry = entry.max(probability);
                        }
                    }
                }
            }
            for ((x, y, z), probability) in max_probability {
                grid.set_probability(&Array3i::new(x, y, z), probability);
            }
            grid
        }

        fn options(only_optimize_yaw: bool) -> proto::CeresScanMatcherOptions3D {
            let parameter_dictionary = make_dictionary(&format!(
                r#"
        return {{
          occupied_space_weight_0 = 1.,
          translation_weight = 0.01,
          rotation_weight = 0.01,
          only_optimize_yaw = {only_optimize_yaw},
          ceres_solver_options = {{
            use_nonmonotonic_steps = false,
            max_num_iterations = 50,
            num_threads = 1,
          }},
        }}"#
            ));
            create_ceres_scan_matcher_options_3d(&parameter_dictionary)
        }

        #[test]
        fn options_without_intensity() {
            let options = options(true);
            assert_eq!(options.occupied_space_weight, vec![1.]);
            assert!(options.intensity_cost_function_options.is_empty());
            assert!(options.only_optimize_yaw);
            assert_eq!(
                options
                    .ceres_solver_options
                    .as_ref()
                    .unwrap()
                    .max_num_iterations,
                50
            );
        }

        #[test]
        fn recovers_6dof_offset_in_box_room() {
            let resolution = 0.1;
            let world_points =
                box_room_points(Vector3::new(-4., -3., -1.5), Vector3::new(5., 3.5, 2.), 0.1);
            let grid = build_grid(&world_points, resolution);

            // The scan is taken at `true_pose`, i.e. points in the tracking
            // frame are `true_pose^-1 * world`.
            let true_pose = Rigid3d::with_angle_axis(
                Vector3::new(0.3, -0.2, 0.1),
                AngleAxis::new(0.12, Vector3::new(0.2, -0.3, 1.).normalize()),
            );
            let true_pose_inverse = true_pose.inverse().cast::<f32>();
            // Subsample the scan.
            let scan = PointCloud::from_points(
                world_points
                    .iter()
                    .step_by(7)
                    .map(|p| RangefinderPoint {
                        position: true_pose_inverse * *p,
                    })
                    .collect(),
            );
            let initial_pose = Rigid3d::with_angle_axis(
                Vector3::new(0.15, -0.05, 0.),
                AngleAxis::new(0.07, Vector3::new(0., 0., 1.)),
            );

            let matcher = CeresScanMatcher3D::new(&options(false));
            let mut pose = Rigid3d::identity();
            let mut summary = SolverSummary::default();
            matcher.match_(
                initial_pose.translation(),
                &initial_pose,
                &[PointCloudAndHybridGridsPointers {
                    point_cloud: &scan,
                    hybrid_grid: &grid,
                    intensity_hybrid_grid: None,
                }],
                &mut pose,
                &mut summary,
            );
            assert!(summary.is_solution_usable(), "{}", summary.full_report());
            assert!(summary.final_cost < summary.initial_cost);
            let error = pose.inverse() * true_pose;
            assert!(
                error.translation().norm() < 0.02,
                "pose {pose} vs {true_pose}\n{}",
                summary.full_report()
            );
            assert!(
                get_angle(&error) < 0.01,
                "pose {pose} vs {true_pose}\n{}",
                summary.full_report()
            );

            // Sanity check: the final pose scores better than the initial one.
            let interpolated = InterpolatedProbabilityGrid::new(&grid);
            let mean_probability = |pose: &Rigid3d| {
                scan.points()
                    .iter()
                    .map(|p| {
                        let w = *pose * p.position.cast::<f64>();
                        interpolated.get_interpolated_value(w.x, w.y, w.z)
                    })
                    .sum::<f64>()
                    / scan.size() as f64
            };
            assert!(mean_probability(&pose) > mean_probability(&initial_pose));
        }

        #[test]
        fn only_optimize_yaw_keeps_roll_and_pitch() {
            let resolution = 0.1;
            let world_points =
                box_room_points(Vector3::new(-4., -3., -1.5), Vector3::new(5., 3.5, 2.), 0.1);
            let grid = build_grid(&world_points, resolution);
            let true_pose = Rigid3d::with_angle_axis(
                Vector3::new(0.2, 0.1, 0.),
                AngleAxis::new(0.08, Vector3::new(0., 0., 1.)),
            );
            let true_pose_inverse = true_pose.inverse().cast::<f32>();
            let scan = PointCloud::from_points(
                world_points
                    .iter()
                    .step_by(7)
                    .map(|p| RangefinderPoint {
                        position: true_pose_inverse * *p,
                    })
                    .collect(),
            );
            // Start within the basin of attraction of the smoothed map (about
            // 0.2 m at the walls).
            let initial_pose = Rigid3d::with_angle_axis(
                Vector3::new(0.1, 0.05, 0.),
                AngleAxis::new(0.04, Vector3::new(0., 0., 1.)),
            );
            let matcher = CeresScanMatcher3D::new(&options(true));
            let mut pose = Rigid3d::identity();
            let mut summary = SolverSummary::default();
            matcher.match_(
                initial_pose.translation(),
                &initial_pose,
                &[PointCloudAndHybridGridsPointers {
                    point_cloud: &scan,
                    hybrid_grid: &grid,
                    intensity_hybrid_grid: None,
                }],
                &mut pose,
                &mut summary,
            );
            // The rotation stays a pure yaw rotation.
            let q = pose.rotation();
            assert!(q.i.abs() < 1e-12 && q.j.abs() < 1e-12, "{pose}");
            let error = pose.inverse() * true_pose;
            assert!(
                error.translation().norm() < 0.02,
                "{pose} vs {true_pose} {}",
                summary.full_report()
            );
            assert!(get_angle(&error) < 0.01, "{pose} vs {true_pose}");
        }
    }
}
