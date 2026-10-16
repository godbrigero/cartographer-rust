//! End-to-end tests of the high-level `cartographer::Slam` API on
//! `cartographer-sim` data (no C++ counterpart):
//!
//! * 2D and 3D corridor runs with loop closure, fed only through
//!   `Slam::push_*` with the simulated extrinsics; accuracy thresholds match
//!   the faithful-API tests `pose_graph_2d_sim` / `pose_graph_3d_sim`;
//! * the composed 2D occupancy grid against the simulator's ground-truth
//!   walls, and the 3D point map against the ground-truth geometry;
//! * map export files, `.pbstream` save / load round trip, and localization
//!   of a second run in the saved map.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cartographer::{Extrinsics, ImuSample, Mode, Slam, SlamBuilder, SlamStats, TimedPoint};
use cartographer_sim::metrics::{
    Alignment, Association, OccupancyMetricsConfig, SimilarityTransform, TimedPose, associate, ate,
    occupancy_metrics,
};
use cartographer_sim::{
    Dataset, DatasetBuilder, ImuConfig, Lidar2dConfig, Lidar3dConfig, OdometryConfig, RayCast,
    SensorData, SensorKind, Trajectory, World2d, World3d,
};
use nalgebra::{Isometry3, Point2, Point3, Unit, Vector3};

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "cartographer_slam_api_{}_{name}",
        std::process::id()
    ))
}

/// Declares the dataset's LiDAR, IMU and odometry sensors (with their
/// simulated extrinsics) on `builder`.
fn declare_sensors(mut builder: SlamBuilder, dataset: &Dataset) -> SlamBuilder {
    for sensor in &dataset.sensors {
        let extrinsics = Extrinsics::from_isometry(sensor.extrinsic);
        builder = match sensor.kind {
            SensorKind::Lidar2d | SensorKind::Lidar3d => builder.lidar(&sensor.id, extrinsics),
            SensorKind::Imu => builder.imu(&sensor.id, extrinsics),
            SensorKind::Odometry => builder.odometry_with_extrinsics(&sensor.id, extrinsics),
            _ => builder,
        };
    }
    builder
}

/// Feeds all LiDAR, IMU and odometry messages (sensor frames, simulator
/// time in seconds) to `slam`.
fn feed(slam: &mut Slam, dataset: &Dataset) {
    for message in &dataset.messages {
        match &message.data {
            SensorData::PointCloud(cloud) => {
                let points: Vec<TimedPoint> = cloud
                    .points
                    .iter()
                    .map(|p| TimedPoint {
                        position: p.position,
                        time: p.time,
                    })
                    .collect();
                slam.push_lidar(&message.sensor_id, message.time, &points)
                    .unwrap();
            }
            SensorData::Imu(imu) => slam
                .push_imu(
                    message.time,
                    ImuSample {
                        linear_acceleration: imu.linear_acceleration,
                        angular_velocity: imu.angular_velocity,
                    },
                )
                .unwrap(),
            SensorData::Odometry(odometry) => {
                slam.push_odometry(message.time, odometry.pose).unwrap()
            }
            _ => {}
        }
    }
}

fn timed_poses(poses: &[cartographer::StampedPose]) -> Vec<TimedPose> {
    poses
        .iter()
        .map(|p| TimedPose::new(p.time.as_secs_f64(), p.pose))
        .collect()
}

#[derive(Debug)]
#[allow(dead_code)] // Fields are printed.
struct Accuracy {
    rmse: f64,
    max: f64,
    vertical_rmse: f64,
    rotation_max: f64,
    /// Error of the last node relative to the first one (m).
    end_error: f64,
    /// The alignment applied to the estimates.
    world_from_estimate: SimilarityTransform,
}

fn accuracy(dataset: &Dataset, estimates: &[TimedPose], alignment: Alignment) -> Accuracy {
    let reference = TimedPose::from_states(&dataset.ground_truth);
    let pairs = associate(
        &reference,
        estimates,
        Association::Interpolate { max_gap: 0.05 },
    );
    assert_eq!(pairs.len(), estimates.len());
    let result = ate(&pairs, alignment);
    let vertical = pairs
        .iter()
        .map(|pair| {
            let error = result
                .alignment
                .transform_pose(&pair.estimate)
                .translation
                .z
                - pair.reference.translation.z;
            error * error
        })
        .sum::<f64>();
    let (first, last) = (&pairs[0], &pairs[pairs.len() - 1]);
    let estimated = first.estimate.inverse() * last.estimate;
    let truth = first.reference.inverse() * last.reference;
    Accuracy {
        rmse: result.translation.rmse,
        max: result.translation.max,
        vertical_rmse: (vertical / pairs.len() as f64).sqrt(),
        rotation_max: result.rotation.max,
        end_error: (estimated.translation.vector - truth.translation.vector).norm(),
        world_from_estimate: result.alignment,
    }
}

/// The map frame is the first tracking pose: `world_from_map`.
fn world_from_map(dataset: &Dataset) -> Isometry3<f64> {
    dataset.ground_truth_at(dataset.start_time).pose
}

#[derive(Debug)]
#[allow(dead_code)] // Fields are printed.
struct Run {
    stats: SlamStats,
    local: Accuracy,
    global: Accuracy,
    num_local_callbacks: usize,
    num_optimization_callbacks: usize,
}

fn run(slam: &mut Slam, dataset: &Dataset, alignment: Alignment) -> Run {
    let num_local_callbacks = Arc::new(AtomicUsize::new(0));
    let num_optimization_callbacks = Arc::new(AtomicUsize::new(0));
    {
        let counter = Arc::clone(&num_local_callbacks);
        slam.on_local_result(move |result| {
            assert!(!result.points_in_local.is_empty());
            counter.fetch_add(1, Ordering::Relaxed);
        });
        let counter = Arc::clone(&num_optimization_callbacks);
        slam.on_global_optimization(move |event| {
            assert!(event.last_optimized_nodes.contains_key(&0));
            counter.fetch_add(1, Ordering::Relaxed);
        });
    }
    feed(slam, dataset);
    slam.finish().unwrap();
    let local = accuracy(dataset, &timed_poses(&slam.local_trajectory()), alignment);
    let global = accuracy(dataset, &timed_poses(&slam.trajectory()), alignment);
    Run {
        stats: slam.stats(),
        local,
        global,
        num_local_callbacks: num_local_callbacks.load(Ordering::Relaxed),
        num_optimization_callbacks: num_optimization_callbacks.load(Ordering::Relaxed),
    }
}

fn dataset_2d(laps: f64, seed: u64) -> Dataset {
    DatasetBuilder::new(
        Trajectory::corridor_loops(laps, 1.5).with_start_from_rest(3.0),
        World2d::loop_corridor(),
    )
    .seed(seed)
    .lidar_2d("scan", Lidar2dConfig::default())
    .imu("imu", ImuConfig::default())
    .odometry("odom", OdometryConfig::default())
    .build()
}

fn builder_2d(dataset: &Dataset) -> SlamBuilder {
    declare_sensors(Slam::builder(Mode::TwoD), dataset).config(|c| c.num_background_threads = 1)
}

/// 2D, 1.25 laps of the loop corridor (seed 2, the configuration of
/// `pose_graph_2d_sim::corridor_loop_closure_single_thread`, here through
/// `MapBuilder` with sensor collation), followed by map export, save, load
/// and localization of a second run in the saved map.
#[test]
fn two_d_mapping_export_and_localization() {
    let dataset = dataset_2d(1.25, 2);
    let mut slam = builder_2d(&dataset).build().unwrap();
    let outcome = run(&mut slam, &dataset, Alignment::Rigid2d);
    println!("2D: {outcome:?}");
    // Same thresholds as the faithful-API test.
    assert!(
        outcome.stats.num_loop_closure_constraints >= 10,
        "{outcome:?}"
    );
    assert!(
        outcome.stats.num_inter_submap_constraints > outcome.stats.num_loop_closure_constraints
    );
    assert_eq!(outcome.stats.num_inter_trajectory_constraints, 0);
    assert!(outcome.global.rmse < 0.02, "{outcome:?}");
    assert!(outcome.global.max < 0.06, "{outcome:?}");
    assert!(
        outcome.global.rmse < 0.75 * outcome.local.rmse,
        "{outcome:?}"
    );
    assert!(
        outcome.global.end_error < outcome.local.end_error,
        "{outcome:?}"
    );
    assert_eq!(outcome.stats.num_nodes, slam.trajectory().len());
    assert_eq!(outcome.stats.num_submaps, slam.submaps().len());
    // Like in Cartographer, the (at most two) active submaps of a finished
    // trajectory stay unfinished.
    let num_finished = slam.submaps().iter().filter(|s| s.finished).count();
    assert!(
        num_finished + 2 >= outcome.stats.num_submaps,
        "{num_finished}"
    );
    assert!(outcome.num_local_callbacks >= outcome.stats.num_nodes);
    assert_eq!(outcome.num_local_callbacks, outcome.stats.num_local_results);
    assert!(outcome.num_optimization_callbacks > 0);
    assert_eq!(
        outcome.num_optimization_callbacks,
        outcome.stats.num_global_optimizations
    );

    // Global poses without alignment: the map frame is the first pose.
    let world_from_map = world_from_map(&dataset);
    let trajectory = slam.trajectory();
    let max_error = trajectory
        .iter()
        .map(|p| {
            let truth = dataset.ground_truth_at(p.time.as_secs_f64()).pose;
            ((world_from_map * p.pose).translation.vector - truth.translation.vector).norm()
        })
        .fold(0.0, f64::max);
    assert!(max_error < 0.1, "{max_error}");
    // pose_at interpolates between nodes.
    let (a, b) = (&trajectory[10], &trajectory[11]);
    let middle = a.time.add_secs_f64(b.time.seconds_since(a.time) / 2.0);
    let interpolated = slam.pose_at(middle).unwrap();
    let expected = (a.pose.translation.vector + b.pose.translation.vector) / 2.0;
    assert!((interpolated.translation.vector - expected).norm() < 1e-9);
    assert_eq!(slam.pose_at(a.time).unwrap(), a.pose);
    assert!(
        slam.pose_at(trajectory.last().unwrap().time.add_secs_f64(1.0))
            .is_none()
    );
    let current = slam.current_pose().unwrap();
    assert!(current.time >= trajectory.last().unwrap().time);

    // From here on, the map frame is realized by the ATE alignment of the
    // optimized trajectory (removes the few centimeters by which the first
    // node differs from the first ground-truth pose).
    let world_from_map = outcome.global.world_from_estimate.isometry;

    // The composed occupancy grid matches the ground-truth walls.
    let map = slam.map_2d().unwrap();
    assert!((map.resolution - 0.05).abs() < 1e-6);
    let map_from_world = world_from_map.inverse();
    let world = World2d::loop_corridor();
    let metrics = occupancy_metrics(
        &world.rasterize(0.05, 1.0),
        |p: &Point2<f64>| {
            let q = map_from_world * Point3::new(p.x, p.y, 0.0);
            map.probability_at(q.x, q.y)
        },
        &OccupancyMetricsConfig::default(),
    );
    println!("2D map: {metrics:?}");
    assert!(metrics.precision > 0.97, "{metrics:?}");
    assert!(metrics.recall > 0.95, "{metrics:?}");
    assert!(metrics.free_marked_occupied < 0.005, "{metrics:?}");

    // Export files.
    let png = temp_path("map.png");
    map.to_png(&png).unwrap();
    assert!(std::fs::metadata(&png).unwrap().len() > 100);
    let yaml = map.to_ros_map(temp_path("map")).unwrap();
    let yaml_text = std::fs::read_to_string(&yaml).unwrap();
    assert!(yaml_text.contains("resolution: 0.05"), "{yaml_text}");
    let pgm = std::fs::read(yaml.with_extension("pgm")).unwrap();
    assert!(pgm.len() >= map.width * map.height);
    for path in [png, yaml.clone(), yaml.with_extension("pgm")] {
        std::fs::remove_file(path).unwrap();
    }

    // Save, then load for localization.
    let pbstream = temp_path("map_2d.pbstream");
    slam.save(&pbstream).unwrap();
    let saved_trajectory = slam.trajectory();
    let saved_stats = slam.stats();
    drop(slam);

    let second_run = DatasetBuilder::new(
        Trajectory::corridor_loops(0.4, 1.5).with_start_from_rest(3.0),
        World2d::loop_corridor(),
    )
    .seed(5)
    .lidar_2d("scan", Lidar2dConfig::default())
    .imu("imu", ImuConfig::default())
    .odometry("odom", OdometryConfig::default())
    .build();
    let mut localization = Slam::load_for_localization(
        &pbstream,
        builder_2d(&second_run).config(|c| {
            c.global_localization.sampling_ratio = 0.1;
            c.num_background_threads = 4;
        }),
    )
    .unwrap();
    std::fs::remove_file(&pbstream).unwrap();
    assert_eq!(localization.trajectory_id(), 1);
    assert_eq!(
        localization
            .options()
            .1
            .pure_localization_trimmer
            .unwrap()
            .max_submaps_to_keep,
        3
    );
    // The loaded (frozen) map reproduces the saved poses and the map.
    let loaded = localization.all_trajectories();
    let loaded_trajectory = &loaded[&0];
    assert_eq!(loaded_trajectory.len(), saved_trajectory.len());
    for (a, b) in loaded_trajectory.iter().zip(&saved_trajectory) {
        assert_eq!(a.time, b.time);
        assert!((a.pose.translation.vector - b.pose.translation.vector).norm() < 1e-9);
    }
    assert_eq!(localization.stats().num_submaps, saved_stats.num_submaps);
    let loaded_map = localization.map_2d().unwrap();
    assert_eq!(
        (loaded_map.width, loaded_map.height),
        (map.width, map.height)
    );
    assert!((loaded_map.origin - map.origin).norm() < 1e-6);
    let max_cell_difference = loaded_map
        .cells
        .iter()
        .zip(&map.cells)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_cell_difference < 1e-3, "{max_cell_difference}");

    feed(&mut localization, &second_run);
    // Results of a localization trajectory are read before `finish`: like
    // in Cartographer, the pure localization trimmer removes the whole
    // trajectory once it is finished.
    localization.optimize();
    let stats = localization.stats();
    println!("2D localization: {stats:?}");
    assert!(stats.num_inter_trajectory_constraints > 0, "{stats:?}");
    // Pure localization trims the new trajectory's old submaps.
    let new_submaps = localization
        .submaps()
        .iter()
        .filter(|s| s.id.trajectory_id == 1)
        .count();
    assert!((1..=3).contains(&new_submaps), "{new_submaps}");
    // The second run is localized in the frame of the first map.
    let poses = localization.trajectory();
    assert!(!poses.is_empty());
    let current = localization.current_pose().unwrap();
    let truth = second_run.ground_truth_at(current.time.as_secs_f64()).pose;
    let current_error =
        ((world_from_map * current.pose).translation.vector - truth.translation.vector).norm();
    assert!(current_error < 0.06, "{current_error}");
    let errors: Vec<f64> = poses
        .iter()
        .map(|p| {
            let truth = second_run.ground_truth_at(p.time.as_secs_f64()).pose;
            ((world_from_map * p.pose).translation.vector - truth.translation.vector).norm()
        })
        .collect();
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let max = errors.iter().copied().fold(0.0, f64::max);
    println!(
        "2D localization error: {} nodes, rmse {rmse:.4} m, max {max:.4} m, current {current_error:.4} m",
        poses.len()
    );
    assert!(rmse < 0.04, "{rmse}");
    assert!(max < 0.08, "{max}");

    localization.finish().unwrap();
    assert!(localization.trajectory().is_empty());
    assert_eq!(localization.stats().num_submaps, saved_stats.num_submaps);
}

/// 3D, 1.25 laps of the 3D loop corridor (seed 2, the configuration of
/// `pose_graph_3d_sim::corridor_loop_closure_single_thread`), point map
/// export and pbstream round trip.
#[test]
fn three_d_mapping_and_point_map() {
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops_3d(1.25, 1.2).with_start_from_rest(3.0),
        World3d::loop_corridor(),
    )
    .seed(2)
    .lidar_3d("points", Lidar3dConfig::default())
    .imu("imu", ImuConfig::default())
    .build();
    let mut slam = declare_sensors(Slam::builder(Mode::ThreeD), &dataset)
        .config(|c| c.num_background_threads = 1)
        .build()
        .unwrap();
    let outcome = run(&mut slam, &dataset, Alignment::Rigid3d);
    println!("3D: {outcome:?}");
    // Same thresholds as the faithful-API test.
    assert!(
        outcome.stats.num_loop_closure_constraints >= 40,
        "{outcome:?}"
    );
    assert!(
        outcome.stats.num_inter_submap_constraints > outcome.stats.num_loop_closure_constraints
    );
    assert!(outcome.num_optimization_callbacks > 0);
    assert!(outcome.global.rmse < 0.2, "{outcome:?}");
    assert!(outcome.global.max < 0.5, "{outcome:?}");
    assert!(outcome.global.vertical_rmse < 0.06, "{outcome:?}");
    assert!(outcome.global.rotation_max < 0.03, "{outcome:?}");
    assert!(
        outcome.global.rmse < 0.8 * outcome.local.rmse,
        "{outcome:?}"
    );
    assert!(
        outcome.global.end_error < outcome.local.end_error,
        "{outcome:?}"
    );
    assert!(matches!(
        slam.map_2d(),
        Err(cartographer::SlamError::Unsupported(_))
    ));

    // Occupied voxels lie on the ground-truth surfaces.
    let map = slam.map_3d().unwrap();
    assert!((map.resolution - 0.10).abs() < 1e-6);
    assert!(map.points.len() > 1000, "{}", map.points.len());
    let world = World3d::loop_corridor();
    // The 3D map frame is gravity-aligned at the first pose; use the ATE
    // alignment of the optimized trajectory to go to the world frame.
    let world_from_map = outcome.global.world_from_estimate;
    let near_surface = map
        .points
        .iter()
        .filter(|point| {
            let p = world_from_map.transform_point(&point.position.cast::<f64>());
            let p = Point3::from(p);
            // Probe the six axis directions for a surface within 0.3 m.
            [
                Vector3::x(),
                -Vector3::x(),
                Vector3::y(),
                -Vector3::y(),
                Vector3::z(),
                -Vector3::z(),
            ]
            .iter()
            .any(|direction| {
                world
                    .raycast(
                        &(p - direction * 0.3),
                        &Unit::new_normalize(*direction),
                        0.6,
                    )
                    .is_some()
            })
        })
        .count();
    let fraction = near_surface as f64 / map.points.len() as f64;
    println!(
        "3D map: {} points, {fraction:.3} near a surface",
        map.points.len()
    );
    assert!(fraction > 0.9, "{fraction}");
    assert!(
        map.points
            .iter()
            .all(|p| p.probability >= cartographer::slam::DEFAULT_MIN_OCCUPIED_PROBABILITY_3D)
    );
    let ply = temp_path("map.ply");
    map.to_ply(&ply).unwrap();
    assert!(std::fs::metadata(&ply).unwrap().len() as usize > 16 * map.points.len());
    std::fs::remove_file(&ply).unwrap();

    // pbstream round trip: extend the saved map (frozen), no new data.
    let pbstream = temp_path("map_3d.pbstream");
    slam.save(&pbstream).unwrap();
    let saved = slam.trajectory();
    let points = map.points.len();
    drop(slam);
    let extended = declare_sensors(Slam::builder(Mode::ThreeD), &dataset)
        .extend_map(&pbstream)
        .build()
        .unwrap();
    // A 3D map cannot be loaded into a 2D system.
    let error = Slam::builder(Mode::TwoD)
        .lidar("scan", Extrinsics::identity())
        .localize_in(&pbstream)
        .build()
        .unwrap_err();
    assert!(
        matches!(error, cartographer::SlamError::InvalidMap(_)),
        "{error}"
    );
    std::fs::remove_file(&pbstream).unwrap();
    assert!(extended.options().1.pure_localization_trimmer.is_none());
    let loaded = &extended.all_trajectories()[&0];
    assert_eq!(loaded.len(), saved.len());
    for (a, b) in loaded.iter().zip(&saved) {
        assert!((a.pose.translation.vector - b.pose.translation.vector).norm() < 1e-9);
    }
    assert_eq!(extended.map_3d().unwrap().points.len(), points);
}

/// Loading a map of the wrong mode or a corrupt file is an error, not a
/// panic.
#[test]
fn invalid_maps_are_errors() {
    let path = temp_path("garbage.pbstream");
    std::fs::write(&path, b"definitely not a pbstream").unwrap();
    let dataset = dataset_2d(0.05, 1);
    let error = Slam::load_for_localization(&path, builder_2d(&dataset)).unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(
        matches!(error, cartographer::SlamError::InvalidMap(_)),
        "{error}"
    );
}

/// `Slam` can be moved to another thread, callbacks may capture shared
/// state, and data pushed through the high-level API reaches the map.
#[test]
fn slam_runs_on_a_worker_thread() {
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops(0.1, 1.5).with_start_from_rest(1.0),
        World2d::loop_corridor(),
    )
    .seed(3)
    .time_range(0.0, 6.0)
    .lidar_2d("scan", Lidar2dConfig::default())
    .odometry("odom", OdometryConfig::default())
    .build();
    let slam = declare_sensors(Slam::builder(Mode::TwoD), &dataset)
        .build()
        .unwrap();
    let times = Arc::new(Mutex::new(Vec::new()));
    {
        let times = Arc::clone(&times);
        slam.on_local_result(move |result| times.lock().unwrap().push(result.time));
    }
    let handle = std::thread::spawn(move || {
        let mut slam = slam;
        feed(&mut slam, &dataset);
        slam.finish().unwrap();
        slam
    });
    let slam = handle.join().unwrap();
    let times = times.lock().unwrap();
    assert!(times.len() > 10);
    assert!(times.windows(2).all(|w| w[0] < w[1]));
    assert!(!slam.trajectory().is_empty());
}
