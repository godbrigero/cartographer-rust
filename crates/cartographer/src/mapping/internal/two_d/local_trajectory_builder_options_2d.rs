//! Port of `cartographer/mapping/internal/2d/local_trajectory_builder_options_2d` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::internal::motion_filter::create_motion_filter_options;
use crate::mapping::internal::scan_matching::real_time_correlative_scan_matcher::create_real_time_correlative_scan_matcher_options;
use crate::mapping::internal::two_d::scan_matching::ceres_scan_matcher_2d::create_ceres_scan_matcher_options_2d;
use crate::mapping::pose_extrapolator_interface::create_pose_extrapolator_options;
use crate::mapping::proto;
use crate::mapping::two_d::submap_2d::create_submaps_options_2d;
use crate::sensor::internal::voxel_filter::create_adaptive_voxel_filter_options;

/// Reads `LocalTrajectoryBuilderOptions2D` from `parameter_dictionary`
/// (the `TRAJECTORY_BUILDER_2D` table of `trajectory_builder_2d.lua`).
pub fn create_local_trajectory_builder_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::LocalTrajectoryBuilderOptions2D {
    // The proto fields are floats; C++ narrows the doubles the same way.
    proto::LocalTrajectoryBuilderOptions2D {
        min_range: parameter_dictionary.get_double("min_range") as f32,
        max_range: parameter_dictionary.get_double("max_range") as f32,
        min_z: parameter_dictionary.get_double("min_z") as f32,
        max_z: parameter_dictionary.get_double("max_z") as f32,
        missing_data_ray_length: parameter_dictionary.get_double("missing_data_ray_length") as f32,
        num_accumulated_range_data: parameter_dictionary.get_int("num_accumulated_range_data"),
        voxel_filter_size: parameter_dictionary.get_double("voxel_filter_size") as f32,
        use_online_correlative_scan_matching: parameter_dictionary
            .get_bool("use_online_correlative_scan_matching"),
        adaptive_voxel_filter_options: Some(create_adaptive_voxel_filter_options(
            &parameter_dictionary.get_dictionary("adaptive_voxel_filter"),
        )),
        loop_closure_adaptive_voxel_filter_options: Some(create_adaptive_voxel_filter_options(
            &parameter_dictionary.get_dictionary("loop_closure_adaptive_voxel_filter"),
        )),
        real_time_correlative_scan_matcher_options: Some(
            create_real_time_correlative_scan_matcher_options(
                &parameter_dictionary.get_dictionary("real_time_correlative_scan_matcher"),
            ),
        ),
        ceres_scan_matcher_options: Some(create_ceres_scan_matcher_options_2d(
            &parameter_dictionary.get_dictionary("ceres_scan_matcher"),
        )),
        motion_filter_options: Some(create_motion_filter_options(
            &parameter_dictionary.get_dictionary("motion_filter"),
        )),
        pose_extrapolator_options: Some(create_pose_extrapolator_options(
            &parameter_dictionary.get_dictionary("pose_extrapolator"),
        )),
        imu_gravity_time_constant: parameter_dictionary.get_double("imu_gravity_time_constant"),
        submaps_options: Some(create_submaps_options_2d(
            &parameter_dictionary.get_dictionary("submaps"),
        )),
        use_imu_data: parameter_dictionary.get_bool("use_imu_data"),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use crate::common::configuration_file_resolver::ConfigurationFileResolver;
    use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
    use crate::mapping::proto;

    use super::create_local_trajectory_builder_options_2d;

    /// Loads the shipped `trajectory_builder_2d.lua` and applies the Lua
    /// `overrides` (statements on `TRAJECTORY_BUILDER_2D`, like a
    /// cartographer_ros configuration would).
    pub(crate) fn load_options_2d(overrides: &str) -> proto::LocalTrajectoryBuilderOptions2D {
        let code = format!(
            "include \"trajectory_builder_2d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_2D"
        );
        let dictionary = LuaParameterDictionary::new(
            &code,
            Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
        );
        create_local_trajectory_builder_options_2d(&dictionary)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::load_options_2d;
    use crate::mapping::proto::grid_options2_d::GridType;

    // Not in C++ (no test there): the shipped defaults are read correctly.
    #[test]
    fn reads_shipped_defaults() {
        let options = load_options_2d("TRAJECTORY_BUILDER_2D.use_imu_data = false");
        assert!(!options.use_imu_data);
        assert_eq!(options.min_range, 0.0);
        assert_eq!(options.max_range, 30.0);
        assert_eq!(options.min_z, -0.8_f32);
        assert_eq!(options.max_z, 2.0);
        assert_eq!(options.missing_data_ray_length, 5.0);
        assert_eq!(options.num_accumulated_range_data, 1);
        assert_eq!(options.voxel_filter_size, 0.025_f32);
        assert!(!options.use_online_correlative_scan_matching);
        let adaptive = options.adaptive_voxel_filter_options.unwrap();
        assert_eq!(adaptive.max_length, 0.5);
        assert_eq!(adaptive.min_num_points, 200.0);
        assert_eq!(adaptive.max_range, 50.0);
        let loop_closure = options.loop_closure_adaptive_voxel_filter_options.unwrap();
        assert_eq!(loop_closure.max_length, 0.9_f32);
        assert_eq!(loop_closure.min_num_points, 100.0);
        let rt = options.real_time_correlative_scan_matcher_options.unwrap();
        assert_eq!(rt.linear_search_window, 0.1);
        assert!((rt.angular_search_window - 20.0_f64.to_radians()).abs() < 1e-12);
        let ceres = options.ceres_scan_matcher_options.unwrap();
        assert_eq!(ceres.occupied_space_weight, 1.0);
        assert_eq!(ceres.translation_weight, 10.0);
        assert_eq!(ceres.rotation_weight, 40.0);
        assert_eq!(ceres.ceres_solver_options.unwrap().max_num_iterations, 20);
        let motion_filter = options.motion_filter_options.unwrap();
        assert_eq!(motion_filter.max_time_seconds, 5.0);
        assert_eq!(motion_filter.max_distance_meters, 0.2);
        assert_eq!(options.imu_gravity_time_constant, 10.0);
        let extrapolator = options.pose_extrapolator_options.unwrap();
        assert!(!extrapolator.use_imu_based);
        assert_eq!(
            extrapolator.constant_velocity.unwrap().pose_queue_duration,
            0.001
        );
        let submaps = options.submaps_options.unwrap();
        assert_eq!(submaps.num_range_data, 90);
        let grid_options = submaps.grid_options_2d.unwrap();
        assert_eq!(grid_options.grid_type(), GridType::ProbabilityGrid);
        assert_eq!(grid_options.resolution, 0.05_f32);
    }
}
