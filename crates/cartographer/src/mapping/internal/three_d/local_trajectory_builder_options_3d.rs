//! Port of `cartographer/mapping/internal/3d/local_trajectory_builder_options_3d` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::internal::motion_filter::create_motion_filter_options;
use crate::mapping::internal::scan_matching::real_time_correlative_scan_matcher::create_real_time_correlative_scan_matcher_options;
use crate::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::create_ceres_scan_matcher_options_3d;
use crate::mapping::pose_extrapolator_interface::create_pose_extrapolator_options;
use crate::mapping::proto;
use crate::mapping::three_d::submap_3d::create_submaps_options_3d;
use crate::sensor::internal::voxel_filter::create_adaptive_voxel_filter_options;

/// Reads `LocalTrajectoryBuilderOptions3D` from `parameter_dictionary`
/// (C++ `CreateLocalTrajectoryBuilderOptions3D`). `initial_poses` and
/// `initial_imu_data` are not read from Lua (left empty), like in C++.
pub fn create_local_trajectory_builder_options_3d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::LocalTrajectoryBuilderOptions3D {
    proto::LocalTrajectoryBuilderOptions3D {
        min_range: parameter_dictionary.get_double("min_range") as f32,
        max_range: parameter_dictionary.get_double("max_range") as f32,
        num_accumulated_range_data: parameter_dictionary.get_int("num_accumulated_range_data"),
        voxel_filter_size: parameter_dictionary.get_double("voxel_filter_size") as f32,
        high_resolution_adaptive_voxel_filter_options: Some(create_adaptive_voxel_filter_options(
            &parameter_dictionary.get_dictionary("high_resolution_adaptive_voxel_filter"),
        )),
        low_resolution_adaptive_voxel_filter_options: Some(create_adaptive_voxel_filter_options(
            &parameter_dictionary.get_dictionary("low_resolution_adaptive_voxel_filter"),
        )),
        use_online_correlative_scan_matching: parameter_dictionary
            .get_bool("use_online_correlative_scan_matching"),
        real_time_correlative_scan_matcher_options: Some(
            create_real_time_correlative_scan_matcher_options(
                &parameter_dictionary.get_dictionary("real_time_correlative_scan_matcher"),
            ),
        ),
        ceres_scan_matcher_options: Some(create_ceres_scan_matcher_options_3d(
            &parameter_dictionary.get_dictionary("ceres_scan_matcher"),
        )),
        motion_filter_options: Some(create_motion_filter_options(
            &parameter_dictionary.get_dictionary("motion_filter"),
        )),
        pose_extrapolator_options: Some(create_pose_extrapolator_options(
            &parameter_dictionary.get_dictionary("pose_extrapolator"),
        )),
        imu_gravity_time_constant: parameter_dictionary.get_double("imu_gravity_time_constant"),
        rotational_histogram_size: parameter_dictionary.get_int("rotational_histogram_size"),
        submaps_options: Some(create_submaps_options_3d(
            &parameter_dictionary.get_dictionary("submaps"),
        )),
        use_intensities: parameter_dictionary.get_bool("use_intensities"),
        initial_poses: Vec::new(),
        initial_imu_data: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::SOURCE_DIRECTORY;
    use crate::common::configuration_file_resolver::ConfigurationFileResolver;

    /// Loads the shipped `trajectory_builder_3d.lua` and checks every field
    /// against the Lua source.
    #[test]
    fn default_configuration_file() {
        let dictionary = LuaParameterDictionary::non_reference_counted(
            "include 'trajectory_builder_3d.lua' return TRAJECTORY_BUILDER_3D",
            Box::new(ConfigurationFileResolver::new([format!(
                "{SOURCE_DIRECTORY}/configuration_files"
            )])),
        );
        let options = create_local_trajectory_builder_options_3d(&dictionary);
        assert_eq!(options.min_range, 1.);
        assert_eq!(options.max_range, 60.);
        assert_eq!(options.num_accumulated_range_data, 1);
        assert_eq!(options.voxel_filter_size, 0.15);
        let high = options
            .high_resolution_adaptive_voxel_filter_options
            .unwrap();
        assert_eq!(
            (high.max_length, high.min_num_points, high.max_range),
            (2., 150., 15.)
        );
        let low = options
            .low_resolution_adaptive_voxel_filter_options
            .unwrap();
        assert_eq!(
            (low.max_length, low.min_num_points, low.max_range),
            (4., 200., 60.)
        );
        assert!(!options.use_online_correlative_scan_matching);
        let rtcsm = options.real_time_correlative_scan_matcher_options.unwrap();
        assert_eq!(rtcsm.linear_search_window, 0.15);
        assert_eq!(
            rtcsm.angular_search_window,
            1f64 * (std::f64::consts::PI / 180.)
        );
        assert_eq!(rtcsm.translation_delta_cost_weight, 1e-1);
        assert_eq!(rtcsm.rotation_delta_cost_weight, 1e-1);
        let ceres = options.ceres_scan_matcher_options.as_ref().unwrap();
        assert_eq!(ceres.occupied_space_weight, vec![1., 6.]);
        assert_eq!(ceres.intensity_cost_function_options.len(), 1);
        assert_eq!(ceres.intensity_cost_function_options[0].weight, 0.5);
        assert_eq!(ceres.intensity_cost_function_options[0].huber_scale, 0.3);
        assert_eq!(
            ceres.intensity_cost_function_options[0].intensity_threshold,
            40.
        );
        assert_eq!(ceres.translation_weight, 5.);
        assert_eq!(ceres.rotation_weight, 4e2);
        assert!(!ceres.only_optimize_yaw);
        assert_eq!(ceres.ceres_solver_options.unwrap().max_num_iterations, 12);
        let motion_filter = options.motion_filter_options.unwrap();
        assert_eq!(motion_filter.max_time_seconds, 0.5);
        assert_eq!(motion_filter.max_distance_meters, 0.1);
        assert_eq!(motion_filter.max_angle_radians, 0.004);
        assert_eq!(options.rotational_histogram_size, 120);
        assert_eq!(options.imu_gravity_time_constant, 10.);
        let extrapolator = options.pose_extrapolator_options.unwrap();
        assert!(!extrapolator.use_imu_based);
        assert_eq!(
            extrapolator.constant_velocity.unwrap().pose_queue_duration,
            0.001
        );
        let submaps = options.submaps_options.unwrap();
        assert_eq!(submaps.high_resolution, 0.10);
        assert_eq!(submaps.high_resolution_max_range, 20.);
        assert_eq!(submaps.low_resolution, 0.45);
        assert_eq!(submaps.num_range_data, 160);
        let inserter = submaps.range_data_inserter_options.unwrap();
        assert_eq!(inserter.hit_probability, 0.55);
        assert_eq!(inserter.miss_probability, 0.49);
        assert_eq!(inserter.num_free_space_voxels, 2);
        assert_eq!(inserter.intensity_threshold, 40.);
        assert!(!options.use_intensities);
        assert!(options.initial_poses.is_empty());
        assert!(options.initial_imu_data.is_empty());
    }
}
