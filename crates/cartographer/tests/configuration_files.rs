//! Port of `cartographer/common/configuration_files_test.cc`, plus checks
//! that the option protos created from the shipped
//! `trajectory_builder_2d.lua`, `trajectory_builder_3d.lua` and
//! `pose_graph.lua` hold the values written in those files.

use std::f64::consts::PI;

use cartographer::common::config::SOURCE_DIRECTORY;
use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::mapping::map_builder_interface::create_map_builder_options;
use cartographer::mapping::proto;
use cartographer::mapping::trajectory_builder_interface::create_trajectory_builder_options;
use cartographer::mapping::{
    create_local_trajectory_builder_options_2d, create_local_trajectory_builder_options_3d,
    create_pose_graph_options,
};

/// A reference counted dictionary (like the C++ test) resolving `include`s
/// against the shipped `configuration_files` directory.
fn load(code: &str) -> LuaParameterDictionary {
    let file_resolver =
        ConfigurationFileResolver::new([format!("{SOURCE_DIRECTORY}/configuration_files")]);
    LuaParameterDictionary::new(code, Box::new(file_resolver))
}

/// `math.rad` of Lua 5.2: `x * (pi / 180)`.
fn lua_rad(x: f64) -> f64 {
    x * (PI / 180.)
}

mod configuration_files_test {
    use super::*;

    #[test]
    fn validate_map_builder_options() {
        const CODE: &str = r#"
      include "map_builder.lua"
      MAP_BUILDER.use_trajectory_builder_2d = true
      return MAP_BUILDER"#;
        let lua_parameter_dictionary = load(CODE);
        create_map_builder_options(&lua_parameter_dictionary);
    }

    #[test]
    fn validate_trajectory_builder_options() {
        const CODE: &str = r#"
      include "trajectory_builder.lua"
      TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false
      return TRAJECTORY_BUILDER"#;
        let lua_parameter_dictionary = load(CODE);
        create_trajectory_builder_options(&lua_parameter_dictionary);
    }
}

#[test]
fn map_builder_options_match_lua_defaults() {
    let options = create_map_builder_options(&load(
        r#"
      include "map_builder.lua"
      MAP_BUILDER.use_trajectory_builder_2d = true
      return MAP_BUILDER"#,
    ));
    assert!(options.use_trajectory_builder_2d);
    assert!(!options.use_trajectory_builder_3d);
    assert_eq!(options.num_background_threads, 4);
    assert!(!options.collate_by_trajectory);
    assert_eq!(
        options.pose_graph_options.unwrap().optimize_every_n_nodes,
        90
    );
}

#[test]
fn trajectory_builder_options_match_lua_defaults() {
    let options = create_trajectory_builder_options(&load(
        r#"include "trajectory_builder.lua" return TRAJECTORY_BUILDER"#,
    ));
    assert!(options.collate_fixed_frame);
    assert!(!options.collate_landmarks);
    // Optional and programmatic fields stay unset.
    assert!(options.pure_localization_trimmer.is_none());
    assert!(options.pose_graph_odometry_motion_filter.is_none());
    assert!(options.initial_trajectory_pose.is_none());
    assert!(options.trajectory_builder_2d_options.unwrap().use_imu_data);
    assert_eq!(
        options.trajectory_builder_3d_options.unwrap().max_range,
        60.
    );
}

#[test]
fn trajectory_builder_options_read_optional_dictionaries() {
    let options = create_trajectory_builder_options(&load(
        r#"
      include "trajectory_builder.lua"
      TRAJECTORY_BUILDER.pure_localization_trimmer = { max_submaps_to_keep = 3 }
      TRAJECTORY_BUILDER.pose_graph_odometry_motion_filter = {
        max_time_seconds = 0.5,
        max_distance_meters = 0.1,
        max_angle_radians = 0.004,
      }
      return TRAJECTORY_BUILDER"#,
    ));
    assert_eq!(
        options
            .pure_localization_trimmer
            .unwrap()
            .max_submaps_to_keep,
        3
    );
    let filter = options.pose_graph_odometry_motion_filter.unwrap();
    assert_eq!(filter.max_time_seconds, 0.5);
    assert_eq!(filter.max_distance_meters, 0.1);
    assert_eq!(filter.max_angle_radians, 0.004);
}

#[test]
fn trajectory_builder_2d_options_match_lua_defaults() {
    let o = create_local_trajectory_builder_options_2d(&load(
        "include 'trajectory_builder_2d.lua' return TRAJECTORY_BUILDER_2D",
    ));
    // float fields: the Lua doubles narrowed to f32.
    let float_fields: &[(&str, f32, f32)] = &[
        ("min_range", o.min_range, 0.),
        ("max_range", o.max_range, 30.),
        ("min_z", o.min_z, -0.8),
        ("max_z", o.max_z, 2.),
        ("missing_data_ray_length", o.missing_data_ray_length, 5.),
        ("voxel_filter_size", o.voxel_filter_size, 0.025),
    ];
    for &(name, actual, expected) in float_fields {
        assert_eq!(actual, expected, "{name}");
    }
    assert!(o.use_imu_data);
    assert_eq!(o.num_accumulated_range_data, 1);
    assert!(!o.use_online_correlative_scan_matching);
    assert_eq!(o.imu_gravity_time_constant, 10.);

    let adaptive = o.adaptive_voxel_filter_options.unwrap();
    assert_eq!(adaptive.max_length, 0.5);
    assert_eq!(adaptive.min_num_points, 200.);
    assert_eq!(adaptive.max_range, 50.);
    let loop_closure = o.loop_closure_adaptive_voxel_filter_options.unwrap();
    assert_eq!(loop_closure.max_length, 0.9);
    assert_eq!(loop_closure.min_num_points, 100.);
    assert_eq!(loop_closure.max_range, 50.);

    let rtcsm = o.real_time_correlative_scan_matcher_options.unwrap();
    assert_eq!(rtcsm.linear_search_window, 0.1);
    assert_eq!(rtcsm.angular_search_window, lua_rad(20.));
    assert_eq!(rtcsm.translation_delta_cost_weight, 1e-1);
    assert_eq!(rtcsm.rotation_delta_cost_weight, 1e-1);

    let ceres = o.ceres_scan_matcher_options.unwrap();
    assert_eq!(ceres.occupied_space_weight, 1.);
    assert_eq!(ceres.translation_weight, 10.);
    assert_eq!(ceres.rotation_weight, 40.);
    let solver = ceres.ceres_solver_options.unwrap();
    assert!(!solver.use_nonmonotonic_steps);
    assert_eq!(solver.max_num_iterations, 20);
    assert_eq!(solver.num_threads, 1);

    let motion_filter = o.motion_filter_options.unwrap();
    assert_eq!(motion_filter.max_time_seconds, 5.);
    assert_eq!(motion_filter.max_distance_meters, 0.2);
    assert_eq!(motion_filter.max_angle_radians, lua_rad(1.));

    let extrapolator = o.pose_extrapolator_options.unwrap();
    assert!(!extrapolator.use_imu_based);
    let constant_velocity = extrapolator.constant_velocity.unwrap();
    assert_eq!(constant_velocity.imu_gravity_time_constant, 10.);
    assert_eq!(constant_velocity.pose_queue_duration, 0.001);
    let imu_based = extrapolator.imu_based.unwrap();
    assert_eq!(imu_based.pose_queue_duration, 5.);
    assert_eq!(imu_based.gravity_constant, 9.806);
    assert_eq!(imu_based.pose_translation_weight, 1.);
    assert_eq!(imu_based.odometry_rotation_weight, 1.);
    let imu_solver = imu_based.solver_options.unwrap();
    assert!(!imu_solver.use_nonmonotonic_steps);
    assert_eq!(imu_solver.max_num_iterations, 10);
    assert_eq!(imu_solver.num_threads, 1);

    let submaps = o.submaps_options.unwrap();
    assert_eq!(submaps.num_range_data, 90);
    let grid = submaps.grid_options_2d.unwrap();
    assert_eq!(
        grid.grid_type,
        proto::grid_options2_d::GridType::ProbabilityGrid as i32
    );
    assert_eq!(grid.resolution, 0.05);
    let inserter = submaps.range_data_inserter_options.unwrap();
    assert_eq!(
        inserter.range_data_inserter_type,
        proto::range_data_inserter_options::RangeDataInserterType::ProbabilityGridInserter2d as i32
    );
    let probability = inserter
        .probability_grid_range_data_inserter_options_2d
        .unwrap();
    assert!(probability.insert_free_space);
    assert_eq!(probability.hit_probability, 0.55);
    assert_eq!(probability.miss_probability, 0.49);
    let tsdf = inserter.tsdf_range_data_inserter_options_2d.unwrap();
    assert_eq!(tsdf.truncation_distance, 0.3);
    assert_eq!(tsdf.maximum_weight, 10.);
    assert!(!tsdf.update_free_space);
    let normal_estimation = tsdf.normal_estimation_options.unwrap();
    assert_eq!(normal_estimation.num_normal_samples, 4);
    assert_eq!(normal_estimation.sample_radius, 0.5);
    assert!(tsdf.project_sdf_distance_to_scan_normal);
    assert_eq!(tsdf.update_weight_range_exponent, 0);
    assert_eq!(
        tsdf.update_weight_angle_scan_normal_to_ray_kernel_bandwidth,
        0.5
    );
    assert_eq!(
        tsdf.update_weight_distance_cell_to_hit_kernel_bandwidth,
        0.5
    );
}

#[test]
fn trajectory_builder_3d_options_match_lua_defaults() {
    let o = create_local_trajectory_builder_options_3d(&load(
        "include 'trajectory_builder_3d.lua' return TRAJECTORY_BUILDER_3D",
    ));
    assert_eq!(o.min_range, 1.);
    assert_eq!(o.max_range, 60.);
    assert_eq!(o.num_accumulated_range_data, 1);
    assert_eq!(o.voxel_filter_size, 0.15);
    assert!(!o.use_online_correlative_scan_matching);
    assert_eq!(o.rotational_histogram_size, 120);
    assert_eq!(o.imu_gravity_time_constant, 10.);
    assert!(!o.use_intensities);

    let high = o.high_resolution_adaptive_voxel_filter_options.unwrap();
    assert_eq!(high.max_length, 2.);
    assert_eq!(high.min_num_points, 150.);
    assert_eq!(high.max_range, 15.);
    let low = o.low_resolution_adaptive_voxel_filter_options.unwrap();
    assert_eq!(low.max_length, 4.);
    assert_eq!(low.min_num_points, 200.);
    assert_eq!(low.max_range, 60.);

    let rtcsm = o.real_time_correlative_scan_matcher_options.unwrap();
    assert_eq!(rtcsm.linear_search_window, 0.15);
    assert_eq!(rtcsm.angular_search_window, lua_rad(1.));

    let ceres = o.ceres_scan_matcher_options.unwrap();
    assert_eq!(ceres.occupied_space_weight, vec![1., 6.]);
    assert_eq!(ceres.intensity_cost_function_options.len(), 1);
    let intensity = &ceres.intensity_cost_function_options[0];
    assert_eq!(intensity.weight, 0.5);
    assert_eq!(intensity.huber_scale, 0.3);
    assert_eq!(intensity.intensity_threshold, 40.);
    assert_eq!(ceres.translation_weight, 5.);
    assert_eq!(ceres.rotation_weight, 4e2);
    assert!(!ceres.only_optimize_yaw);
    let solver = ceres.ceres_solver_options.unwrap();
    assert!(!solver.use_nonmonotonic_steps);
    assert_eq!(solver.max_num_iterations, 12);
    assert_eq!(solver.num_threads, 1);

    let motion_filter = o.motion_filter_options.unwrap();
    assert_eq!(motion_filter.max_time_seconds, 0.5);
    assert_eq!(motion_filter.max_distance_meters, 0.1);
    assert_eq!(motion_filter.max_angle_radians, 0.004);

    let extrapolator = o.pose_extrapolator_options.unwrap();
    assert!(!extrapolator.use_imu_based);
    assert_eq!(
        extrapolator.constant_velocity.unwrap().pose_queue_duration,
        0.001
    );
    assert_eq!(extrapolator.imu_based.unwrap().gravity_constant, 9.806);

    let submaps = o.submaps_options.unwrap();
    assert_eq!(submaps.high_resolution, 0.10);
    assert_eq!(submaps.high_resolution_max_range, 20.);
    assert_eq!(submaps.low_resolution, 0.45);
    assert_eq!(submaps.num_range_data, 160);
    let inserter = submaps.range_data_inserter_options.unwrap();
    assert_eq!(inserter.hit_probability, 0.55);
    assert_eq!(inserter.miss_probability, 0.49);
    assert_eq!(inserter.num_free_space_voxels, 2);
    assert_eq!(inserter.intensity_threshold, 40.);
}

#[test]
fn pose_graph_options_match_lua_defaults() {
    let o = create_pose_graph_options(&load("include 'pose_graph.lua' return POSE_GRAPH"));
    assert_eq!(o.optimize_every_n_nodes, 90);
    assert_eq!(o.matcher_translation_weight, 5e2);
    assert_eq!(o.matcher_rotation_weight, 1.6e3);
    assert_eq!(o.max_num_final_iterations, 200);
    assert_eq!(o.global_sampling_ratio, 0.003);
    assert!(o.log_residual_histograms);
    assert_eq!(o.global_constraint_search_after_n_seconds, 10.);
    // Commented out in pose_graph.lua.
    assert!(o.overlapping_submaps_trimmer_2d.is_none());

    let cb = o.constraint_builder_options.unwrap();
    assert_eq!(cb.sampling_ratio, 0.3);
    assert_eq!(cb.max_constraint_distance, 15.);
    assert_eq!(cb.min_score, 0.55);
    assert_eq!(cb.global_localization_min_score, 0.6);
    assert_eq!(cb.loop_closure_translation_weight, 1.1e4);
    assert_eq!(cb.loop_closure_rotation_weight, 1e5);
    assert!(cb.log_matches);

    let fcsm = cb.fast_correlative_scan_matcher_options.unwrap();
    assert_eq!(fcsm.linear_search_window, 7.);
    assert_eq!(fcsm.angular_search_window, lua_rad(30.));
    assert_eq!(fcsm.branch_and_bound_depth, 7);

    let ceres = cb.ceres_scan_matcher_options.unwrap();
    assert_eq!(ceres.occupied_space_weight, 20.);
    assert_eq!(ceres.translation_weight, 10.);
    assert_eq!(ceres.rotation_weight, 1.);
    let solver = ceres.ceres_solver_options.unwrap();
    assert!(solver.use_nonmonotonic_steps);
    assert_eq!(solver.max_num_iterations, 10);
    assert_eq!(solver.num_threads, 1);

    let fcsm_3d = cb.fast_correlative_scan_matcher_options_3d.unwrap();
    assert_eq!(fcsm_3d.branch_and_bound_depth, 8);
    assert_eq!(fcsm_3d.full_resolution_depth, 3);
    assert_eq!(fcsm_3d.min_rotational_score, 0.77);
    assert_eq!(fcsm_3d.min_low_resolution_score, 0.55);
    assert_eq!(fcsm_3d.linear_xy_search_window, 5.);
    assert_eq!(fcsm_3d.linear_z_search_window, 1.);
    assert_eq!(fcsm_3d.angular_search_window, lua_rad(15.));

    let ceres_3d = cb.ceres_scan_matcher_options_3d.unwrap();
    assert_eq!(ceres_3d.occupied_space_weight, vec![5., 30.]);
    assert!(ceres_3d.intensity_cost_function_options.is_empty());
    assert_eq!(ceres_3d.translation_weight, 10.);
    assert_eq!(ceres_3d.rotation_weight, 1.);
    assert!(!ceres_3d.only_optimize_yaw);
    let solver_3d = ceres_3d.ceres_solver_options.unwrap();
    assert!(!solver_3d.use_nonmonotonic_steps);
    assert_eq!(solver_3d.max_num_iterations, 10);
    assert_eq!(solver_3d.num_threads, 1);

    let op = o.optimization_problem_options.unwrap();
    let double_fields: &[(&str, f64, f64)] = &[
        ("huber_scale", op.huber_scale, 1e1),
        ("acceleration_weight", op.acceleration_weight, 1.1e2),
        ("rotation_weight", op.rotation_weight, 1.6e4),
        (
            "local_slam_pose_translation_weight",
            op.local_slam_pose_translation_weight,
            1e5,
        ),
        (
            "local_slam_pose_rotation_weight",
            op.local_slam_pose_rotation_weight,
            1e5,
        ),
        (
            "odometry_translation_weight",
            op.odometry_translation_weight,
            1e5,
        ),
        ("odometry_rotation_weight", op.odometry_rotation_weight, 1e5),
        (
            "fixed_frame_pose_translation_weight",
            op.fixed_frame_pose_translation_weight,
            1e1,
        ),
        (
            "fixed_frame_pose_rotation_weight",
            op.fixed_frame_pose_rotation_weight,
            1e2,
        ),
        (
            "fixed_frame_pose_tolerant_loss_param_a",
            op.fixed_frame_pose_tolerant_loss_param_a,
            1.,
        ),
        (
            "fixed_frame_pose_tolerant_loss_param_b",
            op.fixed_frame_pose_tolerant_loss_param_b,
            1.,
        ),
    ];
    for &(name, actual, expected) in double_fields {
        assert_eq!(actual, expected, "{name}");
    }
    assert!(!op.fixed_frame_pose_use_tolerant_loss);
    assert!(!op.log_solver_summary);
    assert!(op.use_online_imu_extrinsics_in_3d);
    assert!(!op.fix_z_in_3d);
    let op_solver = op.ceres_solver_options.unwrap();
    assert!(!op_solver.use_nonmonotonic_steps);
    assert_eq!(op_solver.max_num_iterations, 50);
    assert_eq!(op_solver.num_threads, 7);
}
