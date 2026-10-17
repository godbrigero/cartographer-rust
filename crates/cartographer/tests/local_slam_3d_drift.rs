//! Diagnostics for the vertical drift of 3D local SLAM in simulation (no C++
//! counterpart). See `docs/notes/3d_local_slam_drift.md`.
//!
//! The experiments run `LocalTrajectoryBuilder3D` (shipped
//! `trajectory_builder_3d.lua`) on ideal (noise-free, instantaneous) sensor
//! data, and `CeresScanMatcher3D` against submaps built from ground-truth
//! poses. They print their measurements; run them with
//! `cargo test -p cartographer --test local_slam_3d_drift -- --ignored --nocapture`.

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
use cartographer::mapping::internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
use cartographer::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::{
    CeresScanMatcher3D, PointCloudAndHybridGridsPointers,
};
use cartographer::mapping::proto::LocalTrajectoryBuilderOptions3D;
use cartographer::mapping::three_d::submap_3d::ActiveSubmaps3D;
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::internal::voxel_filter::{adaptive_voxel_filter, voxel_filter};
use cartographer::sensor::point_cloud::{PointCloud, transform_point_cloud};
use cartographer::sensor::range_data::{RangeData, transform_range_data};
use cartographer::sensor::rangefinder_point::{RangefinderPoint, TimedRangefinderPoint};
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::{DatasetBuilder, ImuConfig, Lidar3dConfig, SensorData, Trajectory, World3d};
use nalgebra::{DVector, Isometry3, Point2, Quaternion, Translation3, UnitQuaternion, Vector3};

fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

fn to_seconds_since_base(time: Time) -> f64 {
    (time - to_time(0.0)).to_seconds()
}

fn to_isometry(pose: &Rigid3d) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(*pose.translation()),
        UnitQuaternion::new_normalize(*pose.rotation()),
    )
}

fn to_rigid(pose: &Isometry3<f64>) -> Rigid3d {
    Rigid3d::new(pose.translation.vector, *pose.rotation.quaternion())
}

fn options(overrides: &str) -> LocalTrajectoryBuilderOptions3D {
    create_local_trajectory_builder_options_3d(&LuaParameterDictionary::new(
        &format!(
            "include \"trajectory_builder_3d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_3D"
        ),
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    ))
}

/// A trajectory that sits (almost) still at `(x, y, z)` facing +x.
fn nearly_static(x: f64, y: f64, z: f64) -> Trajectory {
    Trajectory::from_waypoints(&[Point2::new(x, y), Point2::new(x + 1e-3, y)], 1e-6, false)
        .with_height(z)
}

#[derive(Clone, Copy)]
struct Sample {
    time: f64,
    reference: Isometry3<f64>,
    estimate: Isometry3<f64>,
    /// Whether this result started a new submap.
    new_submap: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum ImuMode {
    /// The simulated IMU (specific force including the kinematic
    /// acceleration).
    Measured,
    /// Gravity only: the accelerometer reports `R^T (0, 0, g)`, so the
    /// `ImuTracker` gravity estimate is exact even while accelerating.
    GravityOnly,
}

/// Runs local SLAM on ideal data and returns the local poses with ground
/// truth.
fn run_local_slam(
    trajectory: &Trajectory,
    world: World3d,
    lidar: Lidar3dConfig,
    imu_mode: ImuMode,
    t1: f64,
    options: &LocalTrajectoryBuilderOptions3D,
) -> Vec<Sample> {
    let dataset = DatasetBuilder::new(trajectory.clone(), world)
        .lidar_3d("points", lidar)
        .imu("imu", ImuConfig::ideal())
        .time_range(0., t1)
        .build();
    let mut builder = LocalTrajectoryBuilder3D::new(options, &["points"]);
    let mut samples = Vec::new();
    let mut newest_submap_origin = None;
    for message in &dataset.messages {
        match &message.data {
            SensorData::Imu(imu) => {
                let linear_acceleration = match imu_mode {
                    ImuMode::Measured => imu.linear_acceleration,
                    ImuMode::GravityOnly => {
                        dataset
                            .ground_truth_at(message.time)
                            .pose
                            .rotation
                            .inverse()
                            * Vector3::new(0., 0., 9.81)
                    }
                };
                builder.add_imu_data(&ImuData {
                    time: to_time(message.time),
                    linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                });
            }
            SensorData::PointCloud(cloud) => {
                let data = TimedPointCloudData {
                    time: to_time(message.time),
                    origin: cloud.origin,
                    ranges: cloud
                        .points
                        .iter()
                        .map(|p| TimedRangefinderPoint {
                            position: p.position,
                            time: p.time,
                        })
                        .collect(),
                    intensities: cloud.intensities.clone(),
                };
                if let Some(result) = builder.add_range_data("points", &data) {
                    let time = to_seconds_since_base(result.time);
                    let mut new_submap = false;
                    if let Some(insertion) = &result.insertion_result {
                        let newest = insertion.insertion_submaps.last().unwrap();
                        let origin = *cartographer::mapping::submaps::Submap::local_pose(
                            &*newest.read().unwrap(),
                        )
                        .translation();
                        new_submap = newest_submap_origin != Some(origin);
                        newest_submap_origin = Some(origin);
                    }
                    samples.push(Sample {
                        time,
                        reference: trajectory.state(time).pose,
                        estimate: to_isometry(&result.local_pose),
                        new_submap,
                    });
                }
            }
            _ => {}
        }
    }
    samples
}

struct Drift {
    path_length: f64,
    final_error: Vector3<f64>,
    max_vertical: f64,
    max_horizontal: f64,
    final_roll_pitch: f64,
    final_yaw: f64,
}

/// Errors in the ground-truth frame after anchoring the first estimate to
/// the first ground-truth pose.
fn drift(name: &str, trajectory: &Trajectory, samples: &[Sample], print_every: usize) -> Drift {
    let first = samples[0];
    let anchor = first.reference * first.estimate.inverse();
    let mut max_vertical: f64 = 0.;
    let mut max_horizontal: f64 = 0.;
    for (i, s) in samples.iter().enumerate() {
        let e = (anchor * s.estimate).translation.vector - s.reference.translation.vector;
        max_vertical = max_vertical.max(e.z.abs());
        max_horizontal = max_horizontal.max(e.xy().norm());
        if print_every > 0 && i % print_every == 0 {
            let r = s.reference.rotation.inverse() * (anchor * s.estimate).rotation;
            let (roll, pitch, yaw) = r.euler_angles();
            println!(
                "{name} t {:7.2} gt z {:.3} err x {:+.4} y {:+.4} z {:+.4} | rot err roll {:+.5} pitch {:+.5} yaw {:+.5}",
                s.time, s.reference.translation.z, e.x, e.y, e.z, roll, pitch, yaw
            );
        }
    }
    let submap_starts: Vec<String> = samples
        .iter()
        .filter(|s| s.new_submap)
        .map(|s| {
            let e = (anchor * s.estimate).translation.vector - s.reference.translation.vector;
            format!("{:.1} s ({:+.3})", s.time, e.z)
        })
        .collect();
    println!(
        "{name}: new submaps at (z error): {}",
        submap_starts.join(", ")
    );
    let last = samples.last().unwrap();
    let estimate = anchor * last.estimate;
    let final_error = estimate.translation.vector - last.reference.translation.vector;
    let r = last.reference.rotation.inverse() * estimate.rotation;
    let (roll, pitch, yaw) = r.euler_angles();
    let d = Drift {
        path_length: trajectory.path_length(first.time, last.time),
        final_error,
        max_vertical,
        max_horizontal,
        final_roll_pitch: roll.hypot(pitch),
        final_yaw: yaw,
    };
    println!(
        "{name}: {} results, path {:.1} m | final error x {:+.4} y {:+.4} z {:+.4} m | \
         max |z| {:.4} m, max xy {:.4} m | final roll/pitch {:.5} rad, yaw {:+.5} rad",
        samples.len(),
        d.path_length,
        d.final_error.x,
        d.final_error.y,
        d.final_error.z,
        d.max_vertical,
        d.max_horizontal,
        d.final_roll_pitch,
        d.final_yaw,
    );
    d
}

/// The room tour of the report: `Trajectory::room_wander` (two laps,
/// ~62 m) at 1 m/s and constant 1.2 m height, i.e. without handheld motion.
fn room_tour() -> Trajectory {
    Trajectory::room_wander(2.0, 1.0)
        .with_height(1.2)
        .with_start_from_rest(2.0)
}

/// (a) Static sensor in the room with ideal data for 30 s.
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_a_static_room() {
    let trajectory = nearly_static(2.5, 1.0, 1.2);
    let samples = run_local_slam(
        &trajectory,
        World3d::room(),
        Lidar3dConfig::ideal(),
        ImuMode::Measured,
        30.,
        &options(""),
    );
    drift("static room", &trajectory, &samples, 50);
}

/// (b) Pure translation along x at constant height along the south
/// corridor of the loop corridor.
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_b_straight_corridor() {
    let trajectory = straight_corridor();
    for (name, overrides) in [
        ("defaults", ""),
        (
            "translation_weight 0.5",
            "TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 0.5",
        ),
        (
            "low_resolution 0.15",
            "TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.15",
        ),
    ] {
        let samples = run_local_slam(
            &trajectory,
            World3d::loop_corridor(),
            Lidar3dConfig::ideal(),
            ImuMode::Measured,
            trajectory.end_time(),
            &options(overrides),
        );
        drift(
            &format!("straight corridor [{name}]"),
            &trajectory,
            &samples,
            0,
        );
    }
}

/// The room tour with ideal sensors (as in the report), with the simulated
/// IMU and with a gravity-only IMU (e).
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_room_tour() {
    let trajectory = room_tour();
    for (name, imu_mode) in [
        ("room tour", ImuMode::Measured),
        ("room tour, gravity-only IMU", ImuMode::GravityOnly),
    ] {
        let samples = run_local_slam(
            &trajectory,
            World3d::room(),
            Lidar3dConfig::ideal(),
            imu_mode,
            trajectory.end_time(),
            &options(""),
        );
        drift(name, &trajectory, &samples, 50);
    }
}

/// Room tour and static sensor with option variants that change the
/// low-resolution grid (summary lines only).
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_variants() {
    let variants = [
        ("defaults", ""),
        (
            "low_resolution 0.15",
            "TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.15",
        ),
        (
            "occupied_space_weight_1 = 1",
            "TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_1 = 1.",
        ),
        (
            "translation_weight 0.5, rotation_weight 40",
            "TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 0.5\n\
             TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 40.",
        ),
    ];
    for (name, overrides) in variants {
        let options = options(overrides);
        let trajectory = room_tour();
        let samples = run_local_slam(
            &trajectory,
            World3d::room(),
            Lidar3dConfig::ideal(),
            ImuMode::Measured,
            trajectory.end_time(),
            &options,
        );
        drift(&format!("room tour [{name}]"), &trajectory, &samples, 0);
        let trajectory = nearly_static(2.5, 1.0, 1.2);
        let samples = run_local_slam(
            &trajectory,
            World3d::room(),
            Lidar3dConfig::ideal(),
            ImuMode::Measured,
            30.,
            &options,
        );
        drift(&format!("static [{name}]"), &trajectory, &samples, 0);
    }
}

/// Static sensor at several heights: the sign of the vertical error follows
/// the offset of the floor / ceiling from the 0.45 m low-resolution voxel
/// centres of the first submap (whose origin is the sensor position).
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_static_heights() {
    for height in [1.2, 1.35, 1.4, 1.1] {
        for (name, overrides) in [
            ("defaults", ""),
            (
                "low_resolution 0.15",
                "TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.15",
            ),
        ] {
            let trajectory = nearly_static(2.5, 1.0, height);
            let samples = run_local_slam(
                &trajectory,
                World3d::room(),
                Lidar3dConfig::ideal(),
                ImuMode::Measured,
                10.,
                &options(overrides),
            );
            drift(
                &format!("static height {height} [{name}]"),
                &trajectory,
                &samples,
                0,
            );
        }
    }
}

/// The room tour at several heights: the direction of the vertical drift
/// follows the floor / ceiling offset from the low-resolution voxel centres
/// (each new submap is anchored at the current sensor position, so the
/// offset, and with it the pull, repeats in every submap).
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_room_tour_heights() {
    for height in [1.2, 1.35, 1.4, 1.1] {
        let trajectory = Trajectory::room_wander(2.0, 1.0)
            .with_height(height)
            .with_start_from_rest(2.0);
        let samples = run_local_slam(
            &trajectory,
            World3d::room(),
            Lidar3dConfig::ideal(),
            ImuMode::GravityOnly,
            trajectory.end_time(),
            &options(""),
        );
        drift(
            &format!("room tour height {height}, gravity-only IMU"),
            &trajectory,
            &samples,
            0,
        );
    }
}

/// Builds the two active submaps from ground-truth poses (perfect map) and
/// returns them with the per-scan point clouds in the tracking frame.
struct PerfectMap {
    submaps: ActiveSubmaps3D,
    scans: Vec<(Rigid3d, PointCloud)>,
}

fn ideal_scan(
    lidar: &Lidar3dConfig,
    trajectory: &Trajectory,
    world: &World3d,
    time: f64,
) -> PointCloud {
    let dataset = DatasetBuilder::new(trajectory.clone(), world.clone())
        .lidar_3d("points", lidar.clone())
        .time_range(time, time)
        .build();
    let cloud = dataset
        .messages
        .iter()
        .find_map(|m| match &m.data {
            SensorData::PointCloud(c) => Some(c.clone()),
            _ => None,
        })
        .expect("one scan");
    PointCloud::from_points(
        cloud
            .points
            .iter()
            .filter(|p| p.position.norm() >= 1.)
            .map(|p| RangefinderPoint {
                position: p.position,
            })
            .collect(),
    )
}

fn build_perfect_map(
    trajectory: &Trajectory,
    world: &World3d,
    times: &[f64],
    options: &LocalTrajectoryBuilderOptions3D,
) -> PerfectMap {
    let lidar = Lidar3dConfig::ideal();
    let mut submaps = ActiveSubmaps3D::new(&options.submaps_options.unwrap());
    let mut scans = Vec::new();
    for &t in times {
        let pose = to_rigid(&trajectory.state(t).pose);
        let cloud = voxel_filter(
            &ideal_scan(&lidar, trajectory, world, t),
            options.voxel_filter_size,
        );
        let range_data_in_local = transform_range_data(
            &RangeData {
                origin: Vector3::zeros(),
                returns: cloud.clone(),
                misses: PointCloud::new(),
            },
            &pose.cast::<f32>(),
        );
        // Level trajectory: the tracking frame's gravity alignment is its
        // roll/pitch, so local_from_gravity_aligned is the yaw.
        let local_from_gravity_aligned = *pose.rotation();
        submaps.insert_data(
            &range_data_in_local,
            &Quaternion::new(
                local_from_gravity_aligned.w,
                local_from_gravity_aligned.i,
                local_from_gravity_aligned.j,
                local_from_gravity_aligned.k,
            ),
            &DVector::zeros(options.rotational_histogram_size as usize),
        );
        scans.push((pose, cloud));
    }
    PerfectMap { submaps, scans }
}

/// (c) Scan matching against a perfect map from GT + offset initial
/// guesses, and (d) the cost profile along z at GT.
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_c_d_scan_matcher_on_perfect_map() {
    // 6 s (60 scans) of the room tour / the straight corridor build the
    // map; every 5th scan from the 10th on is matched.
    let build_times: Vec<f64> = (0..60).map(|k| 2. + 0.1 * f64::from(k)).collect();
    println!("room tour:");
    perfect_map_matching(&World3d::room(), &room_tour(), &build_times);
    println!("straight corridor:");
    perfect_map_matching(
        &World3d::loop_corridor(),
        &straight_corridor(),
        &build_times,
    );
}

fn straight_corridor() -> Trajectory {
    Trajectory::from_waypoints(
        &[Point2::new(-16., -8.75), Point2::new(16., -8.75)],
        1.2,
        false,
    )
    .with_height(1.2)
    .with_start_from_rest(2.0)
}

fn perfect_map_matching(world: &World3d, trajectory: &Trajectory, build_times: &[f64]) {
    let options = options("");
    let map = build_perfect_map(trajectory, world, build_times, &options);
    let matcher = CeresScanMatcher3D::new(&options.ceres_scan_matcher_options.clone().unwrap());
    let submaps = map.submaps.submaps();
    let submap = submaps[0].read().unwrap();
    let submap_pose = cartographer::mapping::submaps::Submap::local_pose(&*submap);
    let high = options
        .high_resolution_adaptive_voxel_filter_options
        .unwrap();
    let low = options
        .low_resolution_adaptive_voxel_filter_options
        .unwrap();
    for offset in [
        Vector3::new(0., 0., 0.),
        Vector3::new(0., 0., 0.05),
        Vector3::new(0., 0., -0.05),
        Vector3::new(0.05, 0., 0.),
        Vector3::new(-0.05, 0., 0.),
        Vector3::new(0.1, 0., 0.),
        Vector3::new(-0.1, 0., 0.),
    ] {
        let mut sum = Vector3::zeros();
        let mut max_z: f64 = 0.;
        let mut n = 0;
        for (pose, cloud) in map.scans.iter().skip(10).step_by(5) {
            let high_cloud = adaptive_voxel_filter(cloud, &high);
            let low_cloud = adaptive_voxel_filter(cloud, &low);
            let gt_in_submap = submap_pose.inverse() * *pose;
            let initial = Rigid3d::from_translation(offset) * gt_in_submap;
            let mut estimate = Rigid3d::identity();
            let mut summary = ceres::SolverSummary::default();
            matcher.match_(
                initial.translation(),
                &initial,
                &[
                    PointCloudAndHybridGridsPointers {
                        point_cloud: &high_cloud,
                        hybrid_grid: submap.high_resolution_hybrid_grid(),
                        intensity_hybrid_grid: None,
                    },
                    PointCloudAndHybridGridsPointers {
                        point_cloud: &low_cloud,
                        hybrid_grid: submap.low_resolution_hybrid_grid(),
                        intensity_hybrid_grid: None,
                    },
                ],
                &mut estimate,
                &mut summary,
            );
            let e = estimate.translation() - gt_in_submap.translation();
            sum += e;
            max_z = max_z.max(e.z.abs());
            n += 1;
        }
        let mean = sum / f64::from(n);
        println!(
            "(c) offset {:+.2?}: mean error after match x {:+.4} y {:+.4} z {:+.4} (max |z| {:.4}) over {n} scans",
            offset.as_slice(),
            mean.x,
            mean.y,
            mean.z,
            max_z
        );
    }

    // (d) Occupied-space cost of one scan along z around GT, for the high
    // and the low resolution grid separately.
    let (pose, cloud) = &map.scans[30];
    let gt_in_submap = submap_pose.inverse() * *pose;
    for (name, grid, filter) in [
        ("high", submap.high_resolution_hybrid_grid(), &high),
        ("low", submap.low_resolution_hybrid_grid(), &low),
    ] {
        let filtered = adaptive_voxel_filter(cloud, filter);
        let interpolated = cartographer::mapping::internal::three_d::scan_matching::interpolated_grid::InterpolatedProbabilityGrid::new(grid);
        let cost_at = |dz: f64| -> f64 {
            let p = Rigid3d::from_translation(Vector3::new(0., 0., dz)) * gt_in_submap;
            let cloud = transform_point_cloud(&filtered, &p.cast::<f32>());
            cloud
                .points()
                .iter()
                .map(|q| {
                    let v: f64 = interpolated.get_interpolated_value(
                        f64::from(q.position.x),
                        f64::from(q.position.y),
                        f64::from(q.position.z),
                    );
                    (1. - v) * (1. - v)
                })
                .sum::<f64>()
                / filtered.size() as f64
        };
        let mut best = (f64::MAX, 0.);
        let mut profile = String::new();
        for k in -20..=20 {
            let dz = 0.01 * f64::from(k);
            let c = cost_at(dz);
            if c < best.0 {
                best = (c, dz);
            }
            if k % 4 == 0 {
                profile += &format!(" {dz:+.2}:{c:.5}");
            }
        }
        println!(
            "(d) {name} resolution grid, {} points: cost minimum at dz {:+.2}; profile{profile}",
            filtered.size(),
            best.1
        );
    }
}

/// Total Ceres objective `0.5 * sum r^2` of the scan matcher problem.
fn scan_matcher_objective(
    options: &LocalTrajectoryBuilderOptions3D,
    clouds: &[(
        &PointCloud,
        &cartographer::mapping::three_d::hybrid_grid::HybridGrid,
    )],
    target: &Rigid3d,
    pose: &Rigid3d,
) -> f64 {
    use cartographer::mapping::internal::three_d::scan_matching::occupied_space_cost_function_3d::OccupiedSpaceCostFunction3D;
    use cartographer::mapping::internal::three_d::scan_matching::rotation_delta_cost_functor_3d::RotationDeltaCostFunctor3D;
    use cartographer::mapping::internal::three_d::scan_matching::translation_delta_cost_functor_3d::TranslationDeltaCostFunctor3D;
    use ceres::CostFunction;
    let o = options.ceres_scan_matcher_options.as_ref().unwrap();
    let t = pose.translation();
    let r = pose.rotation();
    let translation = [t.x, t.y, t.z];
    let rotation = [r.w, r.i, r.j, r.k];
    let mut total = 0.;
    let mut add = |cost: &dyn CostFunction, params: &[&[f64]]| {
        let mut residuals = vec![0.; cost.num_residuals()];
        assert!(cost.evaluate(params, &mut residuals, None));
        total += 0.5 * residuals.iter().map(|r| r * r).sum::<f64>();
    };
    for (i, (cloud, grid)) in clouds.iter().enumerate() {
        let cost = OccupiedSpaceCostFunction3D::create_auto_diff_cost_function(
            o.occupied_space_weight[i] / (cloud.size() as f64).sqrt(),
            cloud,
            grid,
        );
        add(&cost, &[&translation, &rotation]);
    }
    add(
        &TranslationDeltaCostFunctor3D::create_auto_diff_cost_function(
            o.translation_weight,
            target.translation(),
        ),
        &[&translation],
    );
    add(
        &RotationDeltaCostFunctor3D::create_auto_diff_cost_function(
            o.rotation_weight,
            target.rotation(),
        ),
        &[&rotation],
    );
    total
}

/// (f) Autodiff vs finite-difference Jacobians of the occupied space cost,
/// and whether the scan matcher result is a minimum of its objective.
#[test]
#[ignore = "diagnostic experiment"]
fn experiment_f_gradients_and_optimality() {
    use cartographer::mapping::internal::three_d::scan_matching::occupied_space_cost_function_3d::OccupiedSpaceCostFunction3D;
    use ceres::CostFunction;
    let options = options("");
    let world = World3d::room();
    let trajectory = room_tour();
    let build_times: Vec<f64> = (0..60).map(|k| 2. + 0.1 * f64::from(k)).collect();
    let map = build_perfect_map(&trajectory, &world, &build_times, &options);
    let submaps = map.submaps.submaps();
    let submap = submaps[0].read().unwrap();
    let submap_pose = cartographer::mapping::submaps::Submap::local_pose(&*submap);
    let high = options
        .high_resolution_adaptive_voxel_filter_options
        .unwrap();
    let low = options
        .low_resolution_adaptive_voxel_filter_options
        .unwrap();
    let (pose, cloud) = &map.scans[30];
    let high_cloud = adaptive_voxel_filter(cloud, &high);
    let low_cloud = adaptive_voxel_filter(cloud, &low);
    let gt = submap_pose.inverse() * *pose;
    let start = Rigid3d::from_translation(Vector3::new(0.013, -0.021, 0.017)) * gt;

    // Jacobian check.
    for (name, c, grid) in [
        ("high", &high_cloud, submap.high_resolution_hybrid_grid()),
        ("low", &low_cloud, submap.low_resolution_hybrid_grid()),
    ] {
        let cost = OccupiedSpaceCostFunction3D::create_auto_diff_cost_function(1., c, grid);
        let n = cost.num_residuals();
        let t = start.translation();
        let r = start.rotation();
        let translation = [t.x, t.y, t.z];
        let rotation = [r.w, r.i, r.j, r.k];
        let mut residuals = vec![0.; n];
        let mut jt = vec![0.; n * 3];
        let mut jr = vec![0.; n * 4];
        {
            let mut jacobians = [Some(&mut jt[..]), Some(&mut jr[..])];
            assert!(cost.evaluate(
                &[&translation, &rotation],
                &mut residuals,
                Some(&mut jacobians)
            ));
        }
        let h = 1e-6;
        let mut max_diff: f64 = 0.;
        let mut max_abs: f64 = 0.;
        for block in 0..2 {
            let size = if block == 0 { 3 } else { 4 };
            for k in 0..size {
                let (mut tp, mut rp) = (translation, rotation);
                let (mut tm, mut rm) = (translation, rotation);
                if block == 0 {
                    tp[k] += h;
                    tm[k] -= h;
                } else {
                    rp[k] += h;
                    rm[k] -= h;
                }
                let mut fp = vec![0.; n];
                let mut fm = vec![0.; n];
                assert!(cost.evaluate(&[&tp, &rp], &mut fp, None));
                assert!(cost.evaluate(&[&tm, &rm], &mut fm, None));
                for i in 0..n {
                    let fd = (fp[i] - fm[i]) / (2. * h);
                    let ad = if block == 0 {
                        jt[i * 3 + k]
                    } else {
                        jr[i * 4 + k]
                    };
                    max_diff = max_diff.max((fd - ad).abs());
                    max_abs = max_abs.max(ad.abs());
                }
            }
        }
        println!(
            "(f) {name}: max |autodiff - finite difference| {max_diff:.3e} (max |J| {max_abs:.3e})"
        );
    }

    // Optimality check with the shipped 12 iterations and with 100.
    for iterations in [12, 100] {
        let mut matcher_options = options.ceres_scan_matcher_options.clone().unwrap();
        matcher_options
            .ceres_solver_options
            .as_mut()
            .unwrap()
            .max_num_iterations = iterations;
        let matcher = CeresScanMatcher3D::new(&matcher_options);
        let mut result = Rigid3d::identity();
        let mut summary = ceres::SolverSummary::default();
        matcher.match_(
            start.translation(),
            &start,
            &[
                PointCloudAndHybridGridsPointers {
                    point_cloud: &high_cloud,
                    hybrid_grid: submap.high_resolution_hybrid_grid(),
                    intensity_hybrid_grid: None,
                },
                PointCloudAndHybridGridsPointers {
                    point_cloud: &low_cloud,
                    hybrid_grid: submap.low_resolution_hybrid_grid(),
                    intensity_hybrid_grid: None,
                },
            ],
            &mut result,
            &mut summary,
        );
        let clouds = [
            (&high_cloud, submap.high_resolution_hybrid_grid()),
            (&low_cloud, submap.low_resolution_hybrid_grid()),
        ];
        let f0 = scan_matcher_objective(&options, &clouds, &start, &result);
        println!(
            "(f) {iterations} iterations: {} iterations used, cost {:.6} -> {:.6} (recomputed {f0:.6}), \
             result - gt translation {:+.4?}",
            summary.iterations.len(),
            summary.initial_cost,
            summary.final_cost,
            (result.translation() - gt.translation()).as_slice()
        );
        let mut slopes = String::new();
        for k in 0..6 {
            let h = if k < 3 { 1e-4 } else { 1e-5 };
            let mut v = Vector3::zeros();
            v[k % 3] = h;
            let step = |s: f64| {
                if k < 3 {
                    Rigid3d::from_translation(v * s) * result
                } else {
                    let q = cartographer::transform::transform::angle_axis_vector_to_rotation_quaternion(&(v * s));
                    Rigid3d::new(
                        *result.translation(),
                        cartographer::transform::rigid_transform::quaternion_mul(
                            &q,
                            result.rotation(),
                        ),
                    )
                }
            };
            let fp = scan_matcher_objective(&options, &clouds, &start, &step(1.));
            let fm = scan_matcher_objective(&options, &clouds, &start, &step(-1.));
            slopes += &format!(" {:+.3e}", (fp - fm) / (2. * h));
        }
        println!("(f)   objective gradient at result (tx ty tz rx ry rz):{slopes}");
    }
}

/// Faithful behaviour (as in C++): the interpolated grid has its maxima at
/// voxel centres, so the occupied-space cost of a horizontal plane is
/// minimal when the plane is moved onto the nearest voxel-centre layer, not
/// at its true height. With the 0.45 m low-resolution grid a floor 1.2 m
/// below the grid origin (the sensor position) is pulled 0.15 m down; with
/// 0.1 m voxels it lies on a centre layer and is not pulled.
#[test]
fn occupied_space_cost_snaps_horizontal_plane_to_voxel_centres() {
    use cartographer::mapping::internal::three_d::scan_matching::interpolated_grid::InterpolatedProbabilityGrid;
    use cartographer::mapping::proto::RangeDataInserterOptions3D;
    use cartographer::mapping::three_d::hybrid_grid::HybridGrid;
    use cartographer::mapping::three_d::range_data_inserter_3d::RangeDataInserter3D;

    let floor_z = -1.2f32;
    let plane: Vec<RangefinderPoint> = (-50..=50)
        .flat_map(|i| {
            (-50..=50).map(move |j| RangefinderPoint {
                position: Vector3::new(0.1 * i as f32, 0.1 * j as f32, floor_z),
            })
        })
        .filter(|p| p.position.xy().norm() > 1.)
        .collect();
    let range_data = RangeData {
        origin: Vector3::zeros(),
        returns: PointCloud::from_points(plane.clone()),
        misses: PointCloud::new(),
    };
    // Shipped `submaps.range_data_inserter` options.
    let inserter = RangeDataInserter3D::new(&RangeDataInserterOptions3D {
        hit_probability: 0.55,
        miss_probability: 0.49,
        num_free_space_voxels: 2,
        intensity_threshold: 40.,
    });
    for (resolution, expected_dz) in [(0.45f32, -0.15), (0.10, 0.)] {
        let mut grid = HybridGrid::new(resolution);
        for _ in 0..10 {
            inserter.insert(&range_data, &mut grid, None);
        }
        let interpolated = InterpolatedProbabilityGrid::new(&grid);
        let cost = |dz: f64| -> f64 {
            plane
                .iter()
                .step_by(7)
                .map(|p| {
                    let probability: f64 = interpolated.get_interpolated_value(
                        f64::from(p.position.x),
                        f64::from(p.position.y),
                        f64::from(p.position.z) + dz,
                    );
                    (1. - probability).powi(2)
                })
                .sum()
        };
        let best_dz = (-60..=60)
            .map(|k| 0.005 * f64::from(k))
            .min_by(|a, b| cost(*a).total_cmp(&cost(*b)))
            .unwrap();
        assert!(
            (best_dz - expected_dz).abs() < 0.011,
            "resolution {resolution}: cost minimum at dz {best_dz}, expected {expected_dz}"
        );
    }
}

/// Faithful behaviour (as in C++), see `docs/notes/3d_local_slam_drift.md`:
/// a static sensor with ideal data in `World3d::room()` (floor at 0, ceiling
/// at 3 m) settles at a height biased towards placing floor and ceiling
/// onto the 0.45 m low-resolution voxel centres of the submap, whose origin
/// is the sensor position. At 1.2 m the floor sits 0.15 m above a centre
/// layer (bias down), at 1.35 m the ceiling sits 0.15 m below one (bias up).
/// With a 0.15 m low-resolution grid both surfaces are on centre layers at
/// 1.2 m and the bias vanishes.
#[test]
fn static_sensor_height_bias_follows_low_resolution_voxel_centres() {
    let final_z_error = |height: f64, overrides: &str| {
        let trajectory = nearly_static(2.5, 1.0, height);
        let samples = run_local_slam(
            &trajectory,
            World3d::room(),
            Lidar3dConfig::ideal(),
            ImuMode::Measured,
            5.,
            &options(overrides),
        );
        drift(
            &format!("static {height} m {overrides}"),
            &trajectory,
            &samples,
            0,
        )
        .final_error
        .z
    };
    // Observed: -0.033 m, +0.019 m, +0.003 m.
    assert!(final_z_error(1.2, "") < -0.015);
    assert!(final_z_error(1.35, "") > 0.01);
    assert!(
        final_z_error(1.2, "TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.15").abs() < 0.008
    );
}
