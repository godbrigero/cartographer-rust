//! End-to-end 3D acceptance tests through the public API (no C++
//! counterpart): `MapBuilder` with options loaded from the shipped
//! `configuration_files` like a cartographer_ros configuration
//! ([`common::load_options`]), fed with `cartographer-sim` datasets (a
//! VLP-16-like 3D LiDAR and an IMU, handheld walking motion) through
//! `get_trajectory_builder(id).add_*`. Feeding waits for the global SLAM
//! backend to keep up like a live system would ([`common::RunOptions::paced`]).
//!
//! Scenarios:
//! * `corridor_3d`: two laps of `World3d::loop_corridor`.
//! * `room_3d`: a handheld tour of `World3d::room` (furniture at several
//!   heights).
//!
//! Besides trajectory metrics (after a rigid 3D alignment) the map is
//! checked for consistency: the occupied voxels of all high-resolution
//! submap hybrid grids, placed with the optimized global submap poses and
//! the trajectory alignment, must lie on the true `World3d` surfaces
//! (median / 95th percentile point-to-surface distance).
//!
//! Each scenario prints a metrics table and writes
//! `target/e2e/<scenario>.json`. `E2E_SEED=<n>` overrides the dataset seed.

mod common;

use cartographer::mapping::map_builder::MapBuilder;
use cartographer::mapping::map_builder_interface::MapBuilderInterface;
use cartographer_sim::metrics::{Alignment, TimedPose};
use cartographer_sim::{Dataset, DatasetBuilder, ImuConfig, Lidar3dConfig, Trajectory, World3d};
use common::{
    AlignWith, Json, Report, RunOptions, constraint_summary, global_node_poses, load_options,
    occupied_voxels_3d, pairs_with_ground_truth, poses_json, run_trajectory, surface_distance,
    trajectory_metrics, transform_poses,
};

fn seed(default: u64) -> u64 {
    std::env::var("E2E_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Library defaults (`trajectory_builder_3d.lua`, `pose_graph.lua`), as in
/// the 3D configurations of cartographer_ros, with the dimension switch
/// only: the simulated LiDAR delivers one full revolution per message, so
/// `num_accumulated_range_data = 1` (the default) is right.
const COMMON_3D: &str = "";

/// Voxels of the high-resolution (10 cm) submap grids with at least this
/// probability count as occupied for the map consistency check.
const OCCUPIED_PROBABILITY: f32 = 0.6;

/// Limits of one 3D scenario.
struct Limits {
    ate_rmse: f64,
    ate_max: f64,
    ate_rotation_max: f64,
    rpe_translation_rmse: f64,
    rpe_rotation_rmse: f64,
    end_drift: f64,
    /// Global ATE RMSE / local-SLAM-only ATE RMSE.
    ate_ratio_to_local: f64,
    /// Global end drift / local-SLAM-only end drift.
    end_drift_ratio_to_local: f64,
    min_loop_closure_constraints: f64,
    surface_median: f64,
    surface_p95: f64,
}

fn run_scenario(name: &str, dataset: &Dataset, world: &World3d, limits: &Limits) {
    let mut report = Report::new(name);
    let options = load_options(true, COMMON_3D);
    let mut map_builder = MapBuilder::new(&options.map_builder);
    let log = run_trajectory(
        &mut map_builder,
        dataset,
        &options.trajectory_builder,
        RunOptions::paced(&options.map_builder),
    );
    let pose_graph = map_builder.pose_graph();
    pose_graph.run_final_optimization();

    let global = global_node_poses(pose_graph.as_ref(), log.trajectory_id);
    let metrics = trajectory_metrics(
        &pairs_with_ground_truth(dataset, &global),
        AlignWith::Fit(Alignment::Rigid3d),
    );
    let local = trajectory_metrics(
        &pairs_with_ground_truth(dataset, &log.local_poses),
        AlignWith::Fit(Alignment::Rigid3d),
    );
    let constraints = constraint_summary(pose_graph.as_ref());
    let voxels = occupied_voxels_3d(
        pose_graph.as_ref(),
        &metrics.alignment,
        OCCUPIED_PROBABILITY,
    );
    let surface = surface_distance(world, &voxels);

    report.at_most("ate_rmse_m", metrics.ate_rmse, limits.ate_rmse);
    report.at_most("ate_max_m", metrics.ate_max, limits.ate_max);
    report.at_most(
        "ate_rotation_max_rad",
        metrics.ate_rotation_max,
        limits.ate_rotation_max,
    );
    report.at_most(
        "rpe_1m_translation_rmse_m_per_m",
        metrics.rpe_translation_rmse,
        limits.rpe_translation_rmse,
    );
    report.at_most(
        "rpe_1m_rotation_rmse_rad_per_m",
        metrics.rpe_rotation_rmse,
        limits.rpe_rotation_rmse,
    );
    report.at_most("end_drift_m", metrics.end_drift, limits.end_drift);
    report.at_most(
        "ate_rmse_ratio_to_local_slam",
        metrics.ate_rmse / local.ate_rmse,
        limits.ate_ratio_to_local,
    );
    report.at_most(
        "end_drift_ratio_to_local_slam",
        metrics.end_drift / local.end_drift,
        limits.end_drift_ratio_to_local,
    );
    report.at_least(
        "loop_closure_constraints",
        constraints.loop_closure as f64,
        limits.min_loop_closure_constraints,
    );
    report.at_most(
        "map_surface_distance_median_m",
        surface.median,
        limits.surface_median,
    );
    report.at_most(
        "map_surface_distance_p95_m",
        surface.p95,
        limits.surface_p95,
    );
    report.info("ate_horizontal_rmse_m", metrics.ate_horizontal_rmse);
    report.info("ate_vertical_rmse_m", metrics.ate_vertical_rmse);
    report.info("ate_rotation_rmse_rad", metrics.ate_rotation_rmse);
    report.info("local_slam_ate_rmse_m", local.ate_rmse);
    report.info("local_slam_end_drift_m", local.end_drift);
    report.info("num_nodes", metrics.num_poses as f64);
    report.info(
        "num_submaps",
        pose_graph.get_all_submap_data().size() as f64,
    );
    report.info("path_length_m", metrics.path_length);
    report.info("inter_submap_constraints", constraints.inter_submap as f64);
    report.info("map_occupied_voxels", surface.num_points as f64);
    report.info("map_surface_distance_mean_m", surface.mean);
    report.info("map_outliers_over_20cm", surface.outlier_fraction);
    report.info("slam_seconds", log.seconds);
    report.section("trajectory", metrics.to_json());
    report.section("local_slam_trajectory", local.to_json());
    report.section("constraints", constraints.to_json());
    report.section(
        "map",
        Json::object([
            (
                "occupied_probability",
                Json::from(f64::from(OCCUPIED_PROBABILITY)),
            ),
            ("occupied_voxels", surface.num_points.into()),
            ("surface_distance_median_m", surface.median.into()),
            ("surface_distance_p95_m", surface.p95.into()),
            ("surface_distance_mean_m", surface.mean.into()),
            ("outliers_over_20cm", surface.outlier_fraction.into()),
        ]),
    );
    report.section(
        "estimated_poses_gt_frame",
        poses_json(&transform_poses(&metrics.alignment, &global), 10),
    );
    report.section(
        "ground_truth_poses",
        poses_json(&TimedPose::from_states(&dataset.ground_truth), 50),
    );
    report.finish();
}

/// Two laps (2 x 108 m) of the 3D loop corridor (ceiling beams, pipes,
/// floor clutter) walking at 1.2 m/s with handheld motion (3 cm bobbing,
/// 2-3 deg roll/pitch sway), started from rest; VLP-16-like LiDAR (16 rings
/// over +-15 deg, 2 cm noise) and IMU (default noise).
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 1.5-2x the worst case): ATE RMSE < 20 cm (10.1-12.8 cm; mostly
/// horizontal: lap 1's local SLAM drift of 0.7-1.1 m is distributed over
/// the loop, and lap 2 matches lap 1's submaps; vertical 3.2-3.8 cm), max
/// < 60 cm (23-45 cm), rotation max < 30 mrad (9-15 mrad); RPE over 1 m:
/// translation < 7 cm/m (3.6-4.2 cm/m; handheld bobbing), rotation
/// < 4 mrad/m (1.5-1.8 mrad/m); end drift < 60 cm (2-38 cm) and < 60 % of
/// the local-SLAM-only drift (3-36 %); ATE < 70 % of local SLAM's (26-42 %);
/// more than 160 loop closure constraints (320-338); occupied map voxels on the true
/// surfaces: median distance < 6 cm (3.5-3.8 cm, 10 cm voxels), 95th
/// percentile < 20 cm (13.1-13.7 cm).
#[test]
fn corridor_3d() {
    corridor_3d_laps("corridor_3d", 2.0);
}

/// Long-duration variant: five laps (~550 m, ~8 min of data); same limits.
#[test]
#[ignore = "long-duration variant (~35 s)"]
fn corridor_3d_long() {
    corridor_3d_laps("corridor_3d_long", 5.0);
}

fn corridor_3d_laps(name: &str, laps: f64) {
    let world = World3d::loop_corridor();
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops_3d(laps, 1.2).with_start_from_rest(3.0),
        world.clone(),
    )
    .seed(seed(1))
    .lidar_3d("points", Lidar3dConfig::default())
    .imu("imu", ImuConfig::default())
    .build();
    run_scenario(
        name,
        &dataset,
        &world,
        &Limits {
            ate_rmse: 0.2,
            ate_max: 0.6,
            ate_rotation_max: 0.03,
            rpe_translation_rmse: 0.07,
            rpe_rotation_rmse: 0.004,
            end_drift: 0.6,
            ate_ratio_to_local: 0.7,
            end_drift_ratio_to_local: 0.6,
            min_loop_closure_constraints: 160.0,
            surface_median: 0.06,
            surface_p95: 0.2,
        },
    );
}

/// Two handheld tours (walking at 1 m/s) weaving between the pillars and
/// boxes of `World3d::room` (12 x 8 x 3 m with a table, a shelf, a hanging
/// lamp and a crate), started from rest.
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 1.5-2x the worst case): ATE RMSE < 7 cm (3.8-4.2 cm; vertical 3.2-3.7
/// cm: the +-15 deg LiDAR sees the floor only beyond 4.5 m and the ceiling
/// beyond 6.7 m, mostly hidden by the walls, so z is weakly observable;
/// local SLAM drifts 33-37 cm, mostly in z, even with ideal sensors), max
/// < 20 cm (8.6-12 cm), rotation max < 30 mrad (11-17 mrad); RPE over 1 m:
/// translation < 5 cm/m (2.8-3.3 cm/m), rotation < 6 mrad/m (2.6-3.1
/// mrad/m); end drift < 25 cm (8.5-14.6 cm) and < 60 % of local SLAM's
/// (24-45 %); ATE < 60 % of local SLAM's (35-37 %); > 20 loop closure
/// constraints (40); occupied map voxels on the true surfaces: median
/// distance < 5 cm (3.0-3.1 cm), 95th percentile < 20 cm (9.4-12.8 cm).
#[test]
fn room_3d() {
    let world = World3d::room();
    let dataset = DatasetBuilder::new(
        Trajectory::room_wander_3d(2.0, 1.0).with_start_from_rest(2.0),
        world.clone(),
    )
    .seed(seed(2))
    .lidar_3d("points", Lidar3dConfig::default())
    .imu("imu", ImuConfig::default())
    .build();
    run_scenario(
        "room_3d",
        &dataset,
        &world,
        &Limits {
            ate_rmse: 0.07,
            ate_max: 0.2,
            ate_rotation_max: 0.03,
            rpe_translation_rmse: 0.05,
            rpe_rotation_rmse: 0.006,
            end_drift: 0.25,
            ate_ratio_to_local: 0.6,
            end_drift_ratio_to_local: 0.6,
            min_loop_closure_constraints: 20.0,
            surface_median: 0.05,
            surface_p95: 0.2,
        },
    );
}
