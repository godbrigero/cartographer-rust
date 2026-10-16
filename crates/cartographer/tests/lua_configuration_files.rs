//! Loads the shipped `configuration_files/*.lua` through
//! `ConfigurationFileResolver` and checks representative values against the
//! Lua sources (complements the C++ `configuration_files_test.cc`, which is
//! ported together with the `Create*Options` checks in
//! `tests/configuration_files.rs`).

use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::Command;

use cartographer::common::config::{CONFIGURATION_FILES_DIRECTORY, SOURCE_DIRECTORY};
use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::transform::rigid_transform::{Rigid3d, from_dictionary, roll_pitch_yaw};
use cartographer::transform::rigid_transform_test_helpers::assert_is_nearly;
use nalgebra::Vector3;

fn resolver() -> Box<ConfigurationFileResolver> {
    Box::new(ConfigurationFileResolver::new([format!(
        "{SOURCE_DIRECTORY}/configuration_files"
    )]))
}

fn load(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::non_reference_counted(code, resolver())
}

/// `math.rad` of Lua 5.2: `x * (pi / 180)`.
fn lua_rad(x: f64) -> f64 {
    x * (PI / 180.)
}

/// Gets a nested sub-dictionary by a dotted path.
fn sub(dict: &LuaParameterDictionary, path: &str) -> LuaParameterDictionary {
    let mut keys = path.split('.');
    let mut d = dict.get_dictionary(keys.next().unwrap());
    for key in keys {
        d = d.get_dictionary(key);
    }
    d
}

#[test]
fn all_configuration_files_are_copied() {
    let mut names: Vec<String> = std::fs::read_dir(CONFIGURATION_FILES_DIRECTORY)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "map_builder.lua",
            "map_builder_server.lua",
            "pose_graph.lua",
            "trajectory_builder.lua",
            "trajectory_builder_2d.lua",
            "trajectory_builder_3d.lua",
        ]
    );
}

#[test]
fn pose_graph() {
    let dict = load("include 'pose_graph.lua' return POSE_GRAPH");
    assert_eq!(dict.get_int("optimize_every_n_nodes"), 90);
    assert_eq!(dict.get_double("matcher_translation_weight"), 5e2);
    assert_eq!(dict.get_double("matcher_rotation_weight"), 1.6e3);
    assert_eq!(dict.get_int("max_num_final_iterations"), 200);
    assert_eq!(dict.get_double("global_sampling_ratio"), 0.003);
    assert!(dict.get_bool("log_residual_histograms"));
    assert_eq!(
        dict.get_double("global_constraint_search_after_n_seconds"),
        10.
    );
    // The commented-out block is not present.
    assert!(!dict.has_key("overlapping_submaps_trimmer_2d"));

    let constraint_builder = dict.get_dictionary("constraint_builder");
    assert_eq!(constraint_builder.get_double("sampling_ratio"), 0.3);
    assert_eq!(
        constraint_builder.get_double("loop_closure_translation_weight"),
        1.1e4
    );
    assert_eq!(
        constraint_builder.get_double("loop_closure_rotation_weight"),
        1e5
    );
    assert_eq!(
        sub(&constraint_builder, "fast_correlative_scan_matcher")
            .get_double("angular_search_window"),
        lua_rad(30.)
    );
    assert_eq!(
        sub(&constraint_builder, "fast_correlative_scan_matcher_3d")
            .get_double("angular_search_window"),
        lua_rad(15.)
    );
    let solver = sub(
        &constraint_builder,
        "ceres_scan_matcher.ceres_solver_options",
    );
    assert!(solver.get_bool("use_nonmonotonic_steps"));
    assert_eq!(solver.get_non_negative_int("max_num_iterations"), 10);
    assert_eq!(solver.get_non_negative_int("num_threads"), 1);

    let optimization_problem = dict.get_dictionary("optimization_problem");
    assert_eq!(optimization_problem.get_double("huber_scale"), 1e1);
    assert_eq!(
        optimization_problem.get_double("acceleration_weight"),
        1.1e2
    );
    assert_eq!(
        optimization_problem.get_int("fixed_frame_pose_tolerant_loss_param_a"),
        1
    );
    assert_eq!(
        sub(&optimization_problem, "ceres_solver_options").get_int("num_threads"),
        7
    );
}

#[test]
fn map_builder_references_pose_graph() {
    // Like ConfigurationFilesTest.ValidateMapBuilderOptions.
    let dict = load(
        r#"
      include "map_builder.lua"
      MAP_BUILDER.use_trajectory_builder_2d = true
      return MAP_BUILDER"#,
    );
    assert!(dict.get_bool("use_trajectory_builder_2d"));
    assert!(!dict.get_bool("use_trajectory_builder_3d"));
    assert_eq!(dict.get_int("num_background_threads"), 4);
    assert!(!dict.get_bool("collate_by_trajectory"));
    assert_eq!(
        sub(&dict, "pose_graph").get_int("optimize_every_n_nodes"),
        90
    );
}

#[test]
fn trajectory_builder_2d() {
    // Like ConfigurationFilesTest.ValidateTrajectoryBuilderOptions.
    let dict = load(
        r#"
      include "trajectory_builder.lua"
      TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false
      return TRAJECTORY_BUILDER"#,
    );
    assert!(dict.get_bool("collate_fixed_frame"));
    assert!(!dict.get_bool("collate_landmarks"));
    assert!(!dict.has_key("pure_localization_trimmer"));
    let tb2d = dict.get_dictionary("trajectory_builder_2d");
    assert!(!tb2d.get_bool("use_imu_data"));
    assert_eq!(tb2d.get_double("min_range"), 0.);
    assert_eq!(tb2d.get_double("max_range"), 30.);
    assert_eq!(tb2d.get_double("min_z"), -0.8);
    assert_eq!(tb2d.get_double("voxel_filter_size"), 0.025);
    assert_eq!(tb2d.get_int("num_accumulated_range_data"), 1);
    let ceres = tb2d.get_dictionary("ceres_scan_matcher");
    assert_eq!(ceres.get_double("occupied_space_weight"), 1.);
    assert_eq!(ceres.get_double("translation_weight"), 10.);
    assert_eq!(ceres.get_double("rotation_weight"), 40.);
    assert_eq!(
        sub(&ceres, "ceres_solver_options").get_int("max_num_iterations"),
        20
    );
    assert_eq!(
        sub(&tb2d, "real_time_correlative_scan_matcher").get_double("angular_search_window"),
        lua_rad(20.)
    );
    assert_eq!(
        sub(&tb2d, "motion_filter").get_double("max_angle_radians"),
        lua_rad(1.)
    );
    // Fields separated by ';'.
    let solver = sub(&tb2d, "pose_extrapolator.imu_based.solver_options");
    assert!(!solver.get_bool("use_nonmonotonic_steps"));
    assert_eq!(solver.get_int("max_num_iterations"), 10);
    assert_eq!(solver.get_int("num_threads"), 1);
    let submaps = tb2d.get_dictionary("submaps");
    assert_eq!(submaps.get_int("num_range_data"), 90);
    assert_eq!(
        sub(&submaps, "grid_options_2d").get_string("grid_type"),
        "PROBABILITY_GRID"
    );
    assert_eq!(
        sub(&submaps, "grid_options_2d").get_double("resolution"),
        0.05
    );
    let inserter = submaps.get_dictionary("range_data_inserter");
    assert_eq!(
        inserter.get_string("range_data_inserter_type"),
        "PROBABILITY_GRID_INSERTER_2D"
    );
    assert_eq!(
        sub(
            &inserter,
            "tsdf_range_data_inserter.normal_estimation_options"
        )
        .get_int("num_normal_samples"),
        4
    );
    assert_eq!(
        sub(&inserter, "tsdf_range_data_inserter").get_int("update_weight_range_exponent"),
        0
    );
}

#[test]
fn trajectory_builder_3d_uses_globals() {
    let dict = load("include 'trajectory_builder.lua' return TRAJECTORY_BUILDER");
    let tb3d = dict.get_dictionary("trajectory_builder_3d");
    assert_eq!(tb3d.get_double("max_range"), 60.);
    assert_eq!(
        sub(&tb3d, "low_resolution_adaptive_voxel_filter").get_double("max_range"),
        60.
    );
    assert_eq!(
        sub(
            &tb3d,
            "ceres_scan_matcher.intensity_cost_function_options_0"
        )
        .get_double("intensity_threshold"),
        40.
    );
    assert_eq!(
        sub(&tb3d, "ceres_scan_matcher").get_double("rotation_weight"),
        4e2
    );
    assert_eq!(sub(&tb3d, "submaps").get_double("high_resolution"), 0.10);
    assert_eq!(
        sub(&tb3d, "submaps.range_data_inserter").get_int("intensity_threshold"),
        40
    );
    assert_eq!(tb3d.get_int("rotational_histogram_size"), 120);
    assert!(!tb3d.get_bool("use_intensities"));
}

#[test]
fn map_builder_server_mutates_referenced_table() {
    let dict = load("include 'map_builder_server.lua' return MAP_BUILDER_SERVER");
    assert_eq!(dict.get_string("server_address"), "0.0.0.0:50051");
    assert_eq!(dict.get_string("uplink_server_address"), "");
    assert_eq!(dict.get_int("upload_batch_size"), 100);
    assert!(!dict.get_bool("enable_ssl_encryption"));
    // `MAP_BUILDER.collate_by_trajectory = true` runs after MAP_BUILDER is
    // referenced by MAP_BUILDER_SERVER: tables are shared by reference.
    assert!(sub(&dict, "map_builder").get_bool("collate_by_trajectory"));
}

#[test]
fn reference_counted_dictionary_accepts_full_consumption() {
    let dict = LuaParameterDictionary::new(
        "include 'pose_graph.lua' return POSE_GRAPH.constraint_builder.ceres_scan_matcher",
        resolver(),
    );
    assert_eq!(dict.get_double("occupied_space_weight"), 20.);
    assert_eq!(dict.get_double("translation_weight"), 10.);
    assert_eq!(dict.get_double("rotation_weight"), 1.);
    let solver = dict.get_dictionary("ceres_solver_options");
    let proto =
        cartographer::common::internal::ceres_solver_options::create_ceres_solver_options_proto(
            &solver,
        );
    assert!(proto.use_nonmonotonic_steps);
    assert_eq!(proto.max_num_iterations, 10);
    assert_eq!(proto.num_threads, 1);
}

#[test]
#[should_panic(expected = "Key 'rotation_weight' was used the wrong number of times.")]
fn reference_counted_dictionary_detects_unused_keys() {
    let dict = LuaParameterDictionary::new(
        "include 'pose_graph.lua' return POSE_GRAPH.constraint_builder.ceres_scan_matcher",
        resolver(),
    );
    dict.get_double("occupied_space_weight");
    dict.get_double("translation_weight");
    let solver = dict.get_dictionary("ceres_solver_options");
    solver.get_bool("use_nonmonotonic_steps");
    solver.get_int("max_num_iterations");
    solver.get_int("num_threads");
}

#[test]
#[should_panic(expected = "twice. Already included files in order of inclusion: ")]
fn including_a_file_twice_indirectly_panics() {
    // map_builder.lua includes pose_graph.lua.
    load("include 'pose_graph.lua' include 'map_builder.lua' return {}");
}

#[test]
fn to_string_round_trips_full_configurations() {
    for code in [
        "include 'map_builder_server.lua' return MAP_BUILDER_SERVER",
        "include 'trajectory_builder.lua' return TRAJECTORY_BUILDER",
    ] {
        let dict = load(code);
        let printed = dict.to_string();
        let reparsed = load(&format!("return {printed}"));
        assert_eq!(reparsed.to_string(), printed);
    }
    let printed = load("include 'map_builder.lua' return MAP_BUILDER").to_string();
    assert!(
        printed.contains("\n  num_background_threads = 4.000000,\n"),
        "{printed}"
    );
    assert!(
        printed.contains("\n    optimize_every_n_nodes = 90.000000,\n"),
        "{printed}"
    );
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cartographer_lua_configuration_files_{}_{}",
            name,
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn write(&self, name: &str, content: &str) {
        std::fs::write(self.0.join(name), content).unwrap();
    }

    fn path(&self) -> String {
        self.0.to_str().unwrap().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const ROS_STYLE_CONFIG: &str = r#"
include "map_builder.lua"
include "trajectory_builder.lua"

VOXEL_SIZE = 5e-2
XY_TRANSFORM = {
  translation = { 0., 0., 0. },
  rotation = { 0., -math.pi / 2., 0. },
}

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  pose_publish_period_sec = 5e-3,
  pipeline = {
    { action = "min_max_range_filter", min_range = 1., max_range = 60. },
    { action = "write_xray_image", voxel_size = VOXEL_SIZE, filename = "xray_yz_all",
      transform = XY_TRANSFORM },
  },
  rates = { 1, 2.5, 1e3 },
  frames = { "base_link", "odom" },
}

MAP_BUILDER.use_trajectory_builder_2d = true
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 10
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)
POSE_GRAPH.optimize_every_n_nodes = 0
POSE_GRAPH.overlapping_submaps_trimmer_2d = {
  fresh_submaps_count = 1,
  min_covered_area = 2,
  min_added_submaps_count = 5,
}

return options
"#;

#[test]
fn ros_style_configuration_with_nested_arrays() {
    let dir = TempDir::new("ros");
    dir.write("robot.lua", ROS_STYLE_CONFIG);
    let resolver = ConfigurationFileResolver::new([dir.path()]);
    let code = std::fs::read_to_string(dir.0.join("robot.lua")).unwrap();
    let dict = LuaParameterDictionary::non_reference_counted(&code, Box::new(resolver));

    assert_eq!(dict.get_string("map_frame"), "map");
    assert_eq!(dict.get_double("pose_publish_period_sec"), 5e-3);
    // Mutations after building `options` are visible through references.
    assert!(sub(&dict, "map_builder").get_bool("use_trajectory_builder_2d"));
    assert_eq!(
        sub(&dict, "map_builder.pose_graph").get_int("optimize_every_n_nodes"),
        0
    );
    assert_eq!(
        sub(
            &dict,
            "map_builder.pose_graph.overlapping_submaps_trimmer_2d"
        )
        .get_int("min_added_submaps_count"),
        5
    );
    let tb2d = sub(&dict, "trajectory_builder.trajectory_builder_2d");
    assert_eq!(tb2d.get_int("num_accumulated_range_data"), 10);
    assert_eq!(
        sub(&tb2d, "motion_filter").get_double("max_angle_radians"),
        lua_rad(0.1)
    );

    // Arrays.
    assert_eq!(
        dict.get_dictionary("rates").get_array_values_as_doubles(),
        vec![1., 2.5, 1e3]
    );
    assert_eq!(
        dict.get_dictionary("frames").get_array_values_as_strings(),
        vec!["base_link", "odom"]
    );
    let pipeline = dict
        .get_dictionary("pipeline")
        .get_array_values_as_dictionaries();
    assert_eq!(pipeline.len(), 2);
    assert_eq!(pipeline[0].get_string("action"), "min_max_range_filter");
    assert_eq!(pipeline[0].get_double("max_range"), 60.);
    assert_eq!(pipeline[1].get_double("voxel_size"), 5e-2);
    let transform = from_dictionary(&pipeline[1].get_dictionary("transform"));
    assert_is_nearly(
        &transform,
        &Rigid3d::new(Vector3::zeros(), roll_pitch_yaw(0., -PI / 2., 0.)),
        1e-15,
    );
}

fn print_configuration(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_cartographer_print_configuration"))
        .args(args)
        .output()
        .expect("failed to run cartographer_print_configuration")
}

#[test]
fn print_configuration_binary() {
    let dir = TempDir::new("print");
    dir.write(
        "options.lua",
        "include 'map_builder.lua'\nreturn { map_builder = MAP_BUILDER, name = 'x' }",
    );
    let directories = format!(",{},", dir.path());

    let output = print_configuration(&[
        "--configuration_directories",
        &directories,
        "--configuration_basename",
        "options.lua",
    ]);
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.starts_with("return {\n  map_builder = {\n"),
        "{stdout}"
    );
    assert!(stdout.contains("\n  name = \"x\",\n}\n"), "{stdout}");
    let expected = format!(
        "return {}\n",
        load("include 'map_builder.lua' return { map_builder = MAP_BUILDER, name = 'x' }")
    );
    assert_eq!(stdout, expected);

    let output = print_configuration(&[
        &format!("--configuration_directories={}", dir.path()),
        "--configuration_basename=options.lua",
        "--subdictionary",
        "map_builder.pose_graph.constraint_builder.ceres_scan_matcher.ceres_solver_options",
    ]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "map_builder.pose_graph.constraint_builder.ceres_scan_matcher.ceres_solver_options = {\n  max_num_iterations = 10.000000,\n  num_threads = 1.000000,\n  use_nonmonotonic_steps = true,\n}\n"
    );

    // Missing flags print the usage and fail.
    let output = print_configuration(&["--configuration_basename", "options.lua"]);
    assert!(!output.status.success());
}
