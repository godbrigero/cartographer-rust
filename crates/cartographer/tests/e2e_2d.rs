//! End-to-end 2D acceptance tests through the public API (no C++
//! counterpart): `MapBuilder` with options loaded from the shipped
//! `configuration_files` like a cartographer_ros configuration
//! ([`common::load_options`]), fed with `cartographer-sim` datasets through
//! `get_trajectory_builder(id).add_*`. Feeding waits for the global SLAM
//! backend to keep up like a live system would ([`common::RunOptions::paced`]).
//!
//! Scenarios:
//! * `corridor_2d`: two laps of `World2d::loop_corridor`, LiDAR + IMU (with
//!   gyro/accelerometer bias) + miscalibrated wheel odometry, start from
//!   rest.
//! * `office_2d`: two laps of `World2d::office` with LiDAR + odometry only.
//! * `multi_trajectory_2d`: two trajectories in the office; the second one
//!   starts elsewhere and global localization must connect it to the first.
//! * `pure_localization_2d`: an office map is built, serialized to a
//!   pbstream, loaded frozen into a new `MapBuilder`, and a new trajectory
//!   with a pure localization trimmer localizes in it.
//!
//! Global localization scenarios use the office: `World2d::loop_corridor`
//! is nearly symmetric under a 180 deg rotation, and full-submap matches of
//! a scan against the "opposite" stretch of the ring score almost as high
//! as true matches (observed 0.82 vs 0.85), so the first global constraint
//! may legitimately be a false one there (perceptual aliasing, not a port
//! issue: `FastCorrelativeScanMatcher2D::MatchFullSubmap` returns the best
//! score of the submap, which simply does not contain the true pose).
//!
//! Each scenario prints a metrics table and writes
//! `target/e2e/<scenario>.json` and `target/e2e/<scenario>_map.png` (the
//! global map composed from all submaps, see
//! [`common::compose_global_grid_2d`]). `E2E_SEED=<n>` overrides the dataset
//! seeds (used to calibrate the thresholds over several seeds).

mod common;

use cartographer::mapping::map_builder::MapBuilder;
use cartographer::mapping::map_builder_interface::MapBuilderInterface;
use cartographer::mapping::pose_graph_interface::PoseGraphInterface;
use cartographer_sim::metrics::{
    Alignment, OccupancyMetrics, OccupancyMetricsConfig, PosePair, TimedPose, occupancy_metrics,
};
use cartographer_sim::{
    Dataset, DatasetBuilder, GroundTruthGrid, ImuConfig, Lidar2dConfig, OdometryConfig, Trajectory,
    World2d,
};
use common::{
    AlignWith, ConstraintSummary, Json, Report, RunOptions, TrajectoryMetrics,
    compose_global_grid_2d, constraint_summary, global_node_poses, isometry_json, load_options,
    num_submaps, observable_ground_truth, pairs_with_ground_truth, planar, poses_json,
    run_trajectory, trajectory_metrics, transform_poses, write_map_png,
};
use nalgebra::{Isometry3, Point2, Translation3, UnitQuaternion, Vector2, Vector3};

fn seed(default: u64) -> u64 {
    std::env::var("E2E_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A 2D LiDAR mounted 15 cm ahead of and 25 cm above the tracking frame
/// (`base_link`), as on a typical wheeled platform. 10 Hz, 720 beams,
/// 30 m, 1 cm range noise, rotating (motion-distorted) scans.
fn lidar() -> Lidar2dConfig {
    Lidar2dConfig {
        extrinsic: Isometry3::from_parts(
            Translation3::new(0.15, 0.0, 0.25),
            UnitQuaternion::identity(),
        ),
        ..Lidar2dConfig::default()
    }
}

/// An IMU at the tracking frame with white noise (default: 1e-3 rad/s,
/// 1e-2 m/s^2) plus constant consumer-grade MEMS biases: 0.1 deg/s gyro
/// bias on every axis and 0.02-0.05 m/s^2 accelerometer bias.
fn imu() -> ImuConfig {
    ImuConfig {
        gyro_bias: Vector3::repeat(0.1_f64.to_radians()),
        accel_bias: Vector3::new(0.03, -0.02, 0.05),
        ..ImuConfig::default()
    }
}

/// Wheel odometry with 2 % translation / rotation random-walk noise plus a
/// 1 % scale error and a small yaw bias (miscalibrated wheel radius and
/// track width).
fn odometry() -> OdometryConfig {
    OdometryConfig {
        translation_scale: 1.01,
        yaw_bias: 0.002,
        ..OdometryConfig::default()
    }
}

/// Settings shared by the scenarios, as in a cartographer_ros `.lua` file
/// on top of `map_builder.lua` / `trajectory_builder.lua`:
/// * `TRAJECTORY_BUILDER_2D.max_range = 30.`: the simulated LiDAR's range.
/// * `POSE_GRAPH.optimize_every_n_nodes = 35` (cartographer_ros'
///   `backpack_2d.lua`): optimize more often than the library default 90.
/// * `POSE_GRAPH.optimization_problem.odometry_{translation,rotation}_weight
///   = 1e4` (default 1e5, equal to the local SLAM weights): the usual
///   tuning when wheel odometry is less trustworthy than scan matching.
///   With the default weights the 1 % odometry scale error pulls the
///   optimized corridor trajectory to 6.7 cm ATE RMSE (1.1 cm with ideal
///   odometry scale; local SLAM alone 2.9 cm), with 1e4 it is 1.25 cm.
const COMMON_2D: &str = "\
TRAJECTORY_BUILDER_2D.max_range = 30.
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e4
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 1e4
";

/// A GT wall cell counts for the map recall if the LiDAR, simulated from the
/// ground-truth poses at 2 Hz, hit it at least `OBSERVATION_MIN_HITS` times
/// from within `OBSERVATION_RANGE` m (see [`observable_ground_truth`]):
/// at 10 Hz scans that is about 50 real hits (Cartographer needs about 4 net
/// hits per submap to reach the 0.65 occupied threshold).
const OBSERVATION_RANGE: f64 = 10.0;
const OBSERVATION_MIN_HITS: u32 = 5;

/// Occupancy metrics: 5 cm GT cells (the submap resolution), occupied
/// threshold 0.65 and a tolerance of one cell.
const OCCUPANCY: OccupancyMetricsConfig = OccupancyMetricsConfig {
    occupied_threshold: 0.65,
    tolerance: 0.05,
};

/// GT raster with 5 cm cells covering the walls plus 1 m, shifted by half a
/// cell so that the walls (all at multiples of 5 cm) run through cell
/// centres instead of along cell boundaries, where the GT cell would be an
/// arbitrary one of the two neighbours.
fn ground_truth_raster(world: &World2d) -> GroundTruthGrid {
    const RESOLUTION: f64 = 0.05;
    const PADDING: f64 = 1.0;
    let (min, max) = world.bounds();
    let origin = Point2::new(
        min.x - PADDING - 0.5 * RESOLUTION,
        min.y - PADDING - 0.5 * RESOLUTION,
    );
    let width = ((max.x - min.x + 2.0 * PADDING) / RESOLUTION).ceil() as usize + 1;
    let height = ((max.y - min.y + 2.0 * PADDING) / RESOLUTION).ceil() as usize + 1;
    GroundTruthGrid::from_world(world, origin, RESOLUTION, width, height)
}

fn xy(poses: &[TimedPose]) -> Vec<Vector2<f64>> {
    poses
        .iter()
        .map(|p| p.pose.translation.vector.xy())
        .collect()
}

/// Pairs the planar projection of estimated poses with the ground truth
/// (2D SLAM is evaluated in `(x, y, yaw)`).
fn pairs_2d(dataset: &Dataset, estimate: &[TimedPose]) -> Vec<PosePair> {
    pairs_with_ground_truth(dataset, &planar(estimate))
}

fn ground_truth_poses(dataset: &Dataset) -> Vec<TimedPose> {
    TimedPose::from_states(&dataset.ground_truth)
}

/// Composes the global map from all submaps, computes the occupancy
/// metrics against the GT raster restricted to what the LiDAR could see
/// along the runs, writes the PNG (GT paths green, estimates blue, GT walls
/// missing from the map red) and adds a `map` section to the report.
fn map_metrics(
    report: &mut Report,
    pose_graph: &dyn PoseGraphInterface,
    world: &World2d,
    world_from_map: &Isometry3<f64>,
    runs: &[(&Dataset, &[TimedPose])],
) -> OccupancyMetrics {
    let full_ground_truth = ground_truth_raster(world);
    let lidar = lidar();
    let sensor_poses: Vec<Isometry3<f64>> = runs
        .iter()
        .flat_map(|(dataset, _)| dataset.ground_truth.iter().step_by(50))
        .map(|state| state.pose * lidar.extrinsic)
        .collect();
    let ground_truth = observable_ground_truth(
        world,
        &full_ground_truth,
        &sensor_poses,
        OBSERVATION_RANGE,
        720,
        OBSERVATION_MIN_HITS,
    );
    let grid = compose_global_grid_2d(pose_graph, ground_truth.layout, world_from_map);
    // Precision against all walls, recall and free space against the
    // well-observed ones.
    let all_walls = occupancy_metrics(&full_ground_truth, |p| grid.lookup(p), &OCCUPANCY);
    let metrics = OccupancyMetrics {
        precision: all_walls.precision,
        ..occupancy_metrics(&ground_truth, |p| grid.lookup(p), &OCCUPANCY)
    };
    let metrics = OccupancyMetrics {
        f1: 2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall),
        ..metrics
    };
    let mut paths = Vec::new();
    for (dataset, estimate) in runs {
        paths.push((xy(&ground_truth_poses(dataset)), [40, 170, 40]));
        paths.push((
            xy(&transform_poses(world_from_map, estimate)),
            [40, 80, 230],
        ));
    }
    let png = report.map_png_path();
    write_map_png(
        &png,
        &grid,
        &ground_truth,
        OCCUPANCY.occupied_threshold,
        &paths,
    );
    report.section(
        "map",
        Json::object([
            ("resolution_m", Json::from(ground_truth.layout.resolution)),
            (
                "occupied_threshold",
                f64::from(OCCUPANCY.occupied_threshold).into(),
            ),
            ("tolerance_m", OCCUPANCY.tolerance.into()),
            ("precision", metrics.precision.into()),
            ("recall", metrics.recall.into()),
            ("f1", metrics.f1.into()),
            ("free_marked_occupied", metrics.free_marked_occupied.into()),
            ("free_coverage", metrics.free_coverage.into()),
            ("estimated_occupied", metrics.estimated_occupied.into()),
            ("gt_occupied_observable", metrics.gt_occupied.into()),
            (
                "gt_occupied_total",
                full_ground_truth
                    .count(cartographer_sim::CellState::Occupied)
                    .into(),
            ),
            ("gt_free_observable", metrics.gt_free.into()),
            ("png", Json::from(png.to_str().unwrap())),
        ]),
    );
    metrics
}

/// Limits of the trajectory checks.
struct TrajectoryLimits {
    ate_rmse: f64,
    ate_max: f64,
    rpe_translation_rmse: f64,
    rpe_rotation_rmse: f64,
    end_drift: f64,
}

fn check_trajectory(
    report: &mut Report,
    prefix: &str,
    metrics: &TrajectoryMetrics,
    limits: &TrajectoryLimits,
) {
    report.at_most(
        &format!("{prefix}ate_rmse_m"),
        metrics.ate_rmse,
        limits.ate_rmse,
    );
    report.at_most(
        &format!("{prefix}ate_max_m"),
        metrics.ate_max,
        limits.ate_max,
    );
    report.at_most(
        &format!("{prefix}rpe_1m_translation_rmse_m_per_m"),
        metrics.rpe_translation_rmse,
        limits.rpe_translation_rmse,
    );
    report.at_most(
        &format!("{prefix}rpe_1m_rotation_rmse_rad_per_m"),
        metrics.rpe_rotation_rmse,
        limits.rpe_rotation_rmse,
    );
    report.at_most(
        &format!("{prefix}end_drift_m"),
        metrics.end_drift,
        limits.end_drift,
    );
}

/// `[precision >, recall >, free_marked_occupied <]`.
fn check_map(report: &mut Report, metrics: &OccupancyMetrics, limits: [f64; 3]) {
    let [precision, recall, free_marked_occupied] = limits;
    report.at_least("map_precision", metrics.precision, precision);
    report.at_least("map_recall", metrics.recall, recall);
    report.at_most(
        "map_free_marked_occupied",
        metrics.free_marked_occupied,
        free_marked_occupied,
    );
    report.info("map_f1", metrics.f1);
    report.info("map_free_coverage", metrics.free_coverage);
}

/// Runs a single-trajectory SLAM scenario and fills the report sections.
fn single_trajectory(
    report: &mut Report,
    dataset: &Dataset,
    world: &World2d,
    overrides: &str,
) -> (TrajectoryMetrics, OccupancyMetrics, ConstraintSummary) {
    let options = load_options(false, &format!("{COMMON_2D}{overrides}"));
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
    let pairs = pairs_2d(dataset, &global);
    let metrics = trajectory_metrics(&pairs, AlignWith::Fit(Alignment::Rigid2d));
    let local_pairs = pairs_2d(dataset, &log.local_poses);
    let local = trajectory_metrics(&local_pairs, AlignWith::Fit(Alignment::Rigid2d));
    let constraints = constraint_summary(pose_graph.as_ref());
    let map = map_metrics(
        report,
        pose_graph.as_ref(),
        world,
        &metrics.alignment,
        &[(dataset, &global)],
    );
    report.info("num_nodes", metrics.num_poses as f64);
    report.info(
        "num_submaps",
        pose_graph.get_all_submap_data().size() as f64,
    );
    report.info("path_length_m", metrics.path_length);
    report.info("inter_submap_constraints", constraints.inter_submap as f64);
    report.info("local_slam_ate_rmse_m", local.ate_rmse);
    report.info("local_slam_end_drift_m", local.end_drift);
    report.info("slam_seconds", log.seconds);
    report.section("trajectory", metrics.to_json());
    report.section("local_slam_trajectory", local.to_json());
    report.section("constraints", constraints.to_json());
    report.section(
        "estimated_poses_gt_frame",
        poses_json(&transform_poses(&metrics.alignment, &global), 5),
    );
    report.section(
        "ground_truth_poses",
        poses_json(&ground_truth_poses(dataset), 50),
    );
    (metrics, map, constraints)
}

/// Two laps (2 x 108 m at 1.5 m/s, ~150 s, ~790 nodes, 9 submaps) of the
/// loop corridor with LiDAR, biased IMU and miscalibrated odometry; shipped
/// 2D defaults (`use_imu_data = true`) plus [`COMMON_2D`].
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 2x the worst case): ATE RMSE < 3 cm (1.0-1.6 cm), max < 8 cm
/// (2.4-4.3 cm); RPE over 1 m: translation < 1 cm/m (0.48-0.51 cm/m),
/// rotation < 1 mrad/m (0.39-0.45 mrad/m); end drift < 5 cm (1.1-2.4 cm);
/// more than 60 loop closure constraints (117-129: the whole second lap closes
/// against the first); map precision > 0.99 (0.9998-1), recall > 0.99
/// (0.9987-0.9996), GT free cells marked occupied < 0.1 % (0-0.002 %).
/// Local SLAM alone: ATE RMSE 2.0-4.8 cm, end drift 2.8-14.6 cm.
#[test]
fn corridor_2d() {
    corridor_2d_laps("corridor_2d", 2.0);
}

/// Long-duration variant: ten laps (~1.1 km, ~12 min of data); same
/// limits.
#[test]
#[ignore = "long-duration variant (~45 s)"]
fn corridor_2d_long() {
    corridor_2d_laps("corridor_2d_long", 10.0);
}

fn corridor_2d_laps(name: &str, laps: f64) {
    let mut report = Report::new(name);
    let world = World2d::loop_corridor();
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops(laps, 1.5).with_start_from_rest(3.0),
        world.clone(),
    )
    .seed(seed(1))
    .lidar_2d("scan", lidar())
    .imu("imu", imu())
    .odometry("odom", odometry())
    .build();
    let (metrics, map, constraints) = single_trajectory(&mut report, &dataset, &world, "");
    check_trajectory(
        &mut report,
        "",
        &metrics,
        &TrajectoryLimits {
            ate_rmse: 0.03,
            ate_max: 0.08,
            rpe_translation_rmse: 0.01,
            rpe_rotation_rmse: 0.001,
            end_drift: 0.05,
        },
    );
    report.at_least(
        "loop_closure_constraints",
        constraints.loop_closure as f64,
        60.0,
    );
    check_map(&mut report, &map, [0.99, 0.99, 0.001]);
    report.finish();
}

/// Two laps of the office tour (two rooms entered through 1 m doors) at
/// 1 m/s with LiDAR and odometry only, configured like cartographer_ros'
/// `revo_lds.lua`: `use_imu_data = false`,
/// `use_online_correlative_scan_matching = true`.
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 2x the worst case): ATE RMSE < 1.2 cm (0.53-0.57 cm), max < 4 cm
/// (1.9-2.2 cm); RPE over 1 m: translation < 1.2 cm/m (0.60-0.65 cm/m),
/// rotation < 6 mrad/m (3.3-3.5 mrad/m: no IMU); end drift < 3 cm
/// (0.7-1.4 cm); > 250 loop closure constraints (511-534); map precision
/// more than 0.99 (1.0), recall > 0.95 (0.978-0.982), GT free cells marked
/// occupied < 0.1 % (0-0.002 %).
#[test]
fn office_2d() {
    let mut report = Report::new("office_2d");
    let world = World2d::office();
    let dataset = DatasetBuilder::new(
        Trajectory::office_tour(2.0, 1.0).with_start_from_rest(2.0),
        world.clone(),
    )
    .seed(seed(2))
    .lidar_2d("scan", lidar())
    .odometry("odom", odometry())
    .build();
    let (metrics, map, constraints) = single_trajectory(
        &mut report,
        &dataset,
        &world,
        "TRAJECTORY_BUILDER_2D.use_imu_data = false\n\
         TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true\n",
    );
    check_trajectory(
        &mut report,
        "",
        &metrics,
        &TrajectoryLimits {
            ate_rmse: 0.012,
            ate_max: 0.04,
            rpe_translation_rmse: 0.012,
            rpe_rotation_rmse: 0.006,
            end_drift: 0.03,
        },
    );
    report.at_least(
        "loop_closure_constraints",
        constraints.loop_closure as f64,
        250.0,
    );
    check_map(&mut report, &map, [0.99, 0.95, 0.001]);
    report.finish();
}

/// The waypoints of `Trajectory::office_tour` traversed in the opposite
/// direction, starting at the waypoint closest to `start`.
fn office_tour_reversed_from(start: Point2<f64>, laps: f64, speed: f64) -> Trajectory {
    let mut waypoints: Vec<Point2<f64>> = [
        (-6.0, 0.3),
        (-5.0, 0.0),
        (-4.5, -0.6),
        (-4.5, -1.5),
        (-5.2, -2.6),
        (-7.0, -3.4),
        (-7.6, -5.0),
        (-6.0, -6.0),
        (-3.6, -5.6),
        (-3.0, -4.2),
        (-3.8, -2.6),
        (-4.5, -1.5),
        (-4.5, -0.6),
        (-3.8, 0.0),
        (-1.6, 0.0),
        (-0.6, 0.4),
        (-0.5, 1.5),
        (0.2, 3.0),
        (1.5, 4.6),
        (0.0, 5.4),
        (-2.4, 4.6),
        (-2.4, 3.0),
        (-1.1, 2.2),
        (-0.5, 1.5),
        (-0.5, 0.5),
        (-1.6, -0.3),
        (-4.5, -0.4),
        (-6.0, -0.45),
        (-7.8, -0.55),
        (-8.9, -0.3),
        (-8.9, 0.3),
        (-7.8, 0.55),
    ]
    .iter()
    .map(|&(x, y)| Point2::new(x, y))
    .collect();
    waypoints.reverse();
    let first = (0..waypoints.len())
        .min_by(|&a, &b| {
            (waypoints[a] - start)
                .norm()
                .total_cmp(&(waypoints[b] - start).norm())
        })
        .unwrap();
    waypoints.rotate_left(first);
    Trajectory::from_waypoints(&waypoints, speed, true).with_laps(laps)
}

/// Asserts that the ground-truth path keeps clear of the walls.
fn assert_clear_path(world: &World2d, dataset: &Dataset) {
    for state in &dataset.ground_truth {
        let p = Point2::from(state.pose.translation.vector.xy());
        assert!(
            world.is_free_with_margin(&p, 0.25),
            "path too close to a wall at {p:?}"
        );
    }
}

/// Builds an office dataset with LiDAR, IMU and odometry.
fn office_dataset(trajectory: Trajectory, seed: u64) -> Dataset {
    let world = World2d::office();
    let dataset = DatasetBuilder::new(trajectory, world.clone())
        .seed(seed)
        .lidar_2d("scan", lidar())
        .imu("imu", imu())
        .odometry("odom", odometry())
        .build();
    assert_clear_path(&world, &dataset);
    dataset
}

/// One rigid 2D alignment fitted on the pairs of all trajectories, and the
/// metrics of each trajectory under it.
fn joint_metrics(pairs: &[Vec<PosePair>]) -> (Isometry3<f64>, Vec<TrajectoryMetrics>) {
    let all: Vec<PosePair> = pairs.iter().flatten().copied().collect();
    let joint = trajectory_metrics(&all, AlignWith::Fit(Alignment::Rigid2d)).alignment;
    (
        joint,
        pairs
            .iter()
            .map(|p| trajectory_metrics(p, AlignWith::Fixed(joint)))
            .collect(),
    )
}

/// Two trajectories in one map builder: trajectory 0 is one office tour;
/// after it is finished, trajectory 1 starts at rest 10 s later in the
/// north-middle room and drives 1.5 tours in the opposite direction.
/// Trajectory 1 starts unconnected (its local frame is placed at the map
/// origin, about 8 m and 150 deg off); only the sampled global constraint
/// search (`global_sampling_ratio`, full-submap
/// `FastCorrelativeScanMatcher2D::MatchFullSubmap`,
/// `global_localization_min_score`) can connect it, after which the local
/// constraint search against trajectory 0 takes over. Both trajectories
/// must then align to the ground truth with a single rigid transform.
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 2x the worst case), per trajectory under the joint alignment: ATE RMSE
/// < 1.2 cm (0.49-0.57 cm), max < 3 cm (1.1-1.5 cm); RPE over 1 m:
/// translation < 1.2 cm/m (0.49-0.59 cm/m), rotation < 1.5 mrad/m
/// (0.52-0.68 mrad/m); end drift < 5 cm (0.8-2.7 cm); > 500
/// inter-trajectory constraints (1073-1079); first inter-trajectory
/// constraint < 30 s after the start of trajectory 1 (17.1-17.3 s; with
/// `global_sampling_ratio = 0.003` a global search is only tried every ~40
/// nodes per finished submap); map precision > 0.99 (1.0), recall > 0.95
/// (0.975-0.980), GT free cells marked occupied < 0.1 % (0).
#[test]
fn multi_trajectory_2d() {
    let mut report = Report::new("multi_trajectory_2d");
    let world = World2d::office();
    let dataset_0 = office_dataset(
        Trajectory::office_tour(1.0, 1.0).with_start_from_rest(2.0),
        seed(3),
    );
    let dataset_1 = office_dataset(
        office_tour_reversed_from(Point2::new(0.0, 5.4), 1.5, 1.0)
            .with_start_from_rest(2.0)
            .with_start_time(dataset_0.end_time + 10.0),
        seed(3) + 100,
    );

    let options = load_options(false, COMMON_2D);
    let mut map_builder = MapBuilder::new(&options.map_builder);
    let paced = RunOptions::paced(&options.map_builder);
    let log_0 = run_trajectory(
        &mut map_builder,
        &dataset_0,
        &options.trajectory_builder,
        paced,
    );
    let log_1 = run_trajectory(
        &mut map_builder,
        &dataset_1,
        &options.trajectory_builder,
        paced,
    );
    assert_eq!((log_0.trajectory_id, log_1.trajectory_id), (0, 1));
    let pose_graph = map_builder.pose_graph();
    pose_graph.run_final_optimization();

    let global_0 = global_node_poses(pose_graph.as_ref(), 0);
    let global_1 = global_node_poses(pose_graph.as_ref(), 1);
    let pairs = [
        pairs_2d(&dataset_0, &global_0),
        pairs_2d(&dataset_1, &global_1),
    ];
    let (joint, metrics) = joint_metrics(&pairs);
    let constraints = constraint_summary(pose_graph.as_ref());
    let map = map_metrics(
        &mut report,
        pose_graph.as_ref(),
        &world,
        &joint,
        &[(&dataset_0, &global_0), (&dataset_1, &global_1)],
    );
    for (i, m) in metrics.iter().enumerate() {
        check_trajectory(
            &mut report,
            &format!("trajectory_{i}_"),
            m,
            &TrajectoryLimits {
                ate_rmse: 0.012,
                ate_max: 0.03,
                rpe_translation_rmse: 0.012,
                rpe_rotation_rmse: 0.0015,
                end_drift: 0.05,
            },
        );
        report.section(&format!("trajectory_{i}"), m.to_json());
    }
    report.at_least(
        "inter_trajectory_constraints",
        constraints.inter_trajectory as f64,
        500.0,
    );
    report.at_most(
        "trajectory_1_connected_after_s",
        log_1
            .first_connected_time
            .map_or(f64::INFINITY, |t| t - dataset_1.start_time),
        30.0,
    );
    if let Some(t) = log_1.first_placed_time {
        report.info("trajectory_1_placed_after_s", t - dataset_1.start_time);
    }
    check_map(&mut report, &map, [0.99, 0.95, 0.001]);
    report.info("num_nodes_0", global_0.len() as f64);
    report.info("num_nodes_1", global_1.len() as f64);
    report.info("loop_closure_constraints", constraints.loop_closure as f64);
    report.section("constraints", constraints.to_json());
    report.section("joint_alignment", isometry_json(&joint));
    report.section(
        "estimated_poses_gt_frame",
        Json::Array(vec![
            poses_json(&transform_poses(&joint, &global_0), 5),
            poses_json(&transform_poses(&joint, &global_1), 5),
        ]),
    );
    report.section(
        "ground_truth_poses",
        Json::Array(vec![
            poses_json(&ground_truth_poses(&dataset_0), 50),
            poses_json(&ground_truth_poses(&dataset_1), 50),
        ]),
    );
    report.finish();
}

/// Pure localization: a map of 1.5 office tours is built, serialized with
/// `serialize_state_to_file` and loaded frozen into a new `MapBuilder`
/// configured like cartographer_ros' `demo_backpack_2d_localization.lua`
/// (`POSE_GRAPH.optimize_every_n_nodes = 20`,
/// `TRAJECTORY_BUILDER.pure_localization_trimmer = { max_submaps_to_keep =
/// 3 }`). A new trajectory starts at rest in the south-west room (no
/// initial pose: global localization) and drives 1.5 tours in the opposite
/// direction. Its
/// pose in the map frame is evaluated the way it is published online
/// (`local_to_global * local_pose` after every local SLAM result) and for
/// the nodes still in the pose graph before the trajectory is finished,
/// without any further alignment: the map frame is mapped to the GT frame
/// with the alignment of the frozen mapping trajectory. Once the
/// trajectory is finished, the trimmer removes all its submaps (C++
/// `PureLocalizationTrimmer::Trim` sets `num_submaps_to_keep_ = 0` for a
/// finished trajectory).
///
/// Thresholds (observed range over seeds 1-5 in parentheses; limits about
/// 2x the worst case): placed in the map < 15 s after the start (5.1-7.6 s,
/// incl. 2 s at rest); online pose after that: ATE RMSE < 1.3 cm
/// (0.64-0.66 cm), max < 3.5 cm (1.5-1.8 cm), rotation RMSE < 1.5 mrad
/// (0.68-0.77 mrad), RPE over 1 m < 1.2 cm/m (0.59-0.64 cm/m); nodes kept
/// before finishing: ATE RMSE < 1 cm (0.49-0.52 cm), max < 3 cm
/// (1.4-1.6 cm); > 250 inter-trajectory constraints (519-523); exactly the
/// trimmer's 3 submaps before finishing and none after; all 8 map submaps
/// kept.
#[test]
fn pure_localization_2d() {
    let mut report = Report::new("pure_localization_2d");
    let world = World2d::office();
    let mapping = office_dataset(
        Trajectory::office_tour(1.5, 1.0).with_start_from_rest(2.0),
        seed(4),
    );
    let filename = std::env::temp_dir()
        .join(format!(
            "cartographer_e2e_2d_{}.pbstream",
            std::process::id()
        ))
        .to_str()
        .unwrap()
        .to_string();
    let world_from_map;
    {
        let options = load_options(false, COMMON_2D);
        let mut map_builder = MapBuilder::new(&options.map_builder);
        run_trajectory(
            &mut map_builder,
            &mapping,
            &options.trajectory_builder,
            RunOptions::paced(&options.map_builder),
        );
        map_builder.pose_graph().run_final_optimization();
        assert!(map_builder.serialize_state_to_file(true, &filename));
        let global = global_node_poses(map_builder.pose_graph().as_ref(), 0);
        let metrics = trajectory_metrics(
            &pairs_2d(&mapping, &global),
            AlignWith::Fit(Alignment::Rigid2d),
        );
        report.info("mapping_ate_rmse_m", metrics.ate_rmse);
        world_from_map = metrics.alignment;
    }

    let dataset = office_dataset(
        office_tour_reversed_from(Point2::new(-6.0, -6.0), 1.5, 1.0)
            .with_start_from_rest(2.0)
            .with_start_time(mapping.end_time + 10.0),
        seed(4) + 100,
    );
    let options = load_options(
        false,
        &format!(
            "{COMMON_2D}\
             POSE_GRAPH.optimize_every_n_nodes = 20\n\
             TRAJECTORY_BUILDER.pure_localization_trimmer = {{ max_submaps_to_keep = 3 }}\n"
        ),
    );
    let mut map_builder = MapBuilder::new(&options.map_builder);
    let remapping = map_builder.load_state_from_file(&filename, true);
    std::fs::remove_file(&filename).unwrap();
    assert_eq!(remapping.len(), 1);
    let pose_graph = map_builder.pose_graph();
    assert!(pose_graph.is_trajectory_frozen(0));
    let num_map_submaps = num_submaps(pose_graph.as_ref(), 0);
    let log = run_trajectory(
        &mut map_builder,
        &dataset,
        &options.trajectory_builder,
        RunOptions::paced(&options.map_builder),
    );
    assert_eq!(log.trajectory_id, 1);
    pose_graph.run_final_optimization();

    // Online poses after the trajectory was first placed in the map.
    let localized_at = log.first_placed_time.unwrap_or(f64::INFINITY);
    report.at_most(
        "time_to_localize_s",
        localized_at - dataset.start_time,
        15.0,
    );
    if let Some(t) = log.first_connected_time {
        report.info("first_global_constraint_after_s", t - dataset.start_time);
    }
    let online: Vec<TimedPose> = log
        .online_poses
        .iter()
        .filter(|p| p.time >= localized_at)
        .copied()
        .collect();
    assert!(online.len() >= 2, "never localized");
    let fixed = AlignWith::Fixed(world_from_map);
    let online_metrics = trajectory_metrics(&pairs_2d(&dataset, &online), fixed);
    let final_metrics = trajectory_metrics(&pairs_2d(&dataset, &log.nodes_before_finish), fixed);
    // The trimmer removes the localization trajectory's submaps, nodes and
    // constraints once it is finished: summarize them before.
    let constraints = &log.constraints_before_finish;

    report.at_most("online_ate_rmse_m", online_metrics.ate_rmse, 0.013);
    report.at_most("online_ate_max_m", online_metrics.ate_max, 0.035);
    report.at_most(
        "online_ate_rotation_rmse_rad",
        online_metrics.ate_rotation_rmse,
        0.0015,
    );
    report.at_most(
        "online_rpe_1m_translation_rmse_m_per_m",
        online_metrics.rpe_translation_rmse,
        0.012,
    );
    report.at_most("kept_nodes_ate_rmse_m", final_metrics.ate_rmse, 0.01);
    report.at_most("kept_nodes_ate_max_m", final_metrics.ate_max, 0.03);
    report.at_least(
        "inter_trajectory_constraints",
        constraints.inter_trajectory as f64,
        250.0,
    );
    // The trimmer keeps at most `max_submaps_to_keep` submaps of the
    // localization trajectory while it runs, none once it is finished; the
    // frozen map is untouched.
    report.at_most(
        "localization_submaps_before_finish",
        log.submaps_before_finish as f64,
        3.5,
    );
    report.at_most(
        "localization_submaps_after_finish",
        num_submaps(pose_graph.as_ref(), 1) as f64,
        0.5,
    );
    report.at_least(
        "map_submaps_kept",
        num_submaps(pose_graph.as_ref(), 0) as f64,
        num_map_submaps as f64 - 0.5,
    );
    report.info("num_online_poses", online.len() as f64);
    report.info("num_kept_nodes", log.nodes_before_finish.len() as f64);
    report.info("slam_seconds", log.seconds);
    let map = map_metrics(
        &mut report,
        pose_graph.as_ref(),
        &world,
        &world_from_map,
        &[(&mapping, &[]), (&dataset, &online)],
    );
    report.info("map_precision", map.precision);
    report.info("map_recall", map.recall);
    report.section("online_trajectory", online_metrics.to_json());
    report.section("kept_nodes_trajectory", final_metrics.to_json());
    report.section("constraints", constraints.to_json());
    report.section(
        "estimated_poses_gt_frame",
        poses_json(&transform_poses(&world_from_map, &online), 5),
    );
    report.section(
        "ground_truth_poses",
        poses_json(&ground_truth_poses(&dataset), 50),
    );
    report.finish();
}
