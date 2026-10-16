//! End-to-end test of global SLAM in 3D on simulated data (no C++
//! counterpart): `LocalTrajectoryBuilder3D` feeds `PoseGraph3D::add_node`
//! while a handheld platform walks laps of `World3d::loop_corridor()` at
//! 1.2 m/s (`Trajectory::corridor_loops_3d(..).with_start_from_rest`), with
//! a VLP-16-like 3D LiDAR and an IMU (fed to both local SLAM and the pose
//! graph, whose 3D optimization problem needs IMU data).
//!
//! Both use the shipped Lua defaults (`trajectory_builder_3d.lua`,
//! `pose_graph.lua`). The pose graph runs on a real `ThreadPool`. Node poses
//! are compared with ground truth after a rigid 3D alignment, for the local
//! SLAM poses (no loop closure) and for the globally optimized poses.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::thread_pool::{ThreadPool, ThreadPoolInterface};
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::id::NodeId;
use cartographer::mapping::internal::optimization::optimization_problem_3d::OptimizationProblem3D;
use cartographer::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
use cartographer::mapping::internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
use cartographer::mapping::internal::three_d::pose_graph_3d::PoseGraph3D;
use cartographer::mapping::pose_graph::{PoseGraph, create_pose_graph_options};
use cartographer::mapping::pose_graph_interface::{PoseGraphInterface, Tag};
use cartographer::mapping::proto::{LocalTrajectoryBuilderOptions3D, PoseGraphOptions};
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::metrics::{Alignment, Association, TimedPose, associate, ate};
use cartographer_sim::{DatasetBuilder, ImuConfig, Lidar3dConfig, SensorData, Trajectory, World3d};
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion};

/// Offset of the simulated time axis (simulated times start at 0 s).
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

fn dictionary(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::new(
        code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    )
}

fn load_trajectory_builder_options(overrides: &str) -> LocalTrajectoryBuilderOptions3D {
    create_local_trajectory_builder_options_3d(&dictionary(&format!(
        "include \"trajectory_builder_3d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_3D"
    )))
}

fn load_pose_graph_options(overrides: &str) -> PoseGraphOptions {
    create_pose_graph_options(&dictionary(&format!(
        "include \"pose_graph.lua\"\n{overrides}\nreturn POSE_GRAPH"
    )))
}

/// ATE after a rigid 3D alignment (m / rad).
#[derive(Debug)]
#[allow(dead_code)] // Fields are printed.
struct Ate {
    rmse: f64,
    max: f64,
    /// RMSE of the horizontal (xy) and vertical (z) error components.
    horizontal_rmse: f64,
    vertical_rmse: f64,
    rotation_max: f64,
}

#[derive(Debug)]
#[allow(dead_code)] // Fields are printed.
struct Outcome {
    num_nodes: usize,
    num_submaps: usize,
    num_inter_submap_constraints: usize,
    /// Inter-submap constraints to submaps that were finished long before
    /// the node (more than one submap back), i.e. loop closures.
    num_loop_closure_constraints: usize,
    local_ate: Ate,
    global_ate: Ate,
    /// Error of the last node relative to the first one (m), i.e. the
    /// accumulated drift at loop closure.
    local_end_error: f64,
    global_end_error: f64,
    /// Number of global SLAM optimization callbacks.
    num_optimizations: usize,
    seconds: f64,
}

fn run(laps: f64, seed: u64, num_threads: i32) -> Outcome {
    let start = std::time::Instant::now();
    let trajectory = Trajectory::corridor_loops_3d(laps, 1.2).with_start_from_rest(3.0);
    let dataset = DatasetBuilder::new(trajectory, World3d::loop_corridor())
        .seed(seed)
        .lidar_3d("points", Lidar3dConfig::default())
        .imu("imu", ImuConfig::default())
        .build();

    let thread_pool = Arc::new(ThreadPool::new(num_threads));
    let pose_graph_options = load_pose_graph_options("");
    let pose_graph = PoseGraph3D::new(
        &pose_graph_options,
        Box::new(OptimizationProblem3D::new(
            &pose_graph_options
                .optimization_problem_options
                .unwrap_or_default(),
        )),
        Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
    );
    const TRAJECTORY_ID: i32 = 0;
    let num_optimizations = Arc::new(AtomicUsize::new(0));
    {
        let num_optimizations = Arc::clone(&num_optimizations);
        pose_graph.set_global_slam_optimization_callback(Box::new(move |submaps, nodes| {
            assert!(submaps.contains_key(&TRAJECTORY_ID));
            assert!(nodes.contains_key(&TRAJECTORY_ID));
            num_optimizations.fetch_add(1, Ordering::Relaxed);
        }));
    }

    let options = load_trajectory_builder_options("");
    let mut local_trajectory_builder = LocalTrajectoryBuilder3D::new(&options, &["points"]);
    // Local poses of the nodes, keyed by node ID.
    let mut local_poses: BTreeMap<NodeId, TimedPose> = BTreeMap::new();
    for message in &dataset.messages {
        let extrinsic = dataset
            .sensor(&message.sensor_id)
            .expect("known sensor")
            .extrinsic;
        let time = to_time(message.time);
        match &message.data {
            SensorData::PointCloud(cloud) => {
                let extrinsic_f32 = extrinsic.cast::<f32>();
                let transform = |p| (extrinsic_f32 * Point3::from(p)).coords;
                let data = TimedPointCloudData {
                    time,
                    origin: transform(cloud.origin),
                    ranges: cloud
                        .points
                        .iter()
                        .map(|p| TimedRangefinderPoint {
                            position: transform(p.position),
                            time: p.time,
                        })
                        .collect(),
                    intensities: cloud.intensities.clone(),
                };
                if let Some(result) = local_trajectory_builder.add_range_data("points", &data)
                    && let Some(insertion_result) = result.insertion_result
                {
                    let node_id = pose_graph.add_node(
                        insertion_result.constant_data,
                        TRAJECTORY_ID,
                        &insertion_result.insertion_submaps,
                    );
                    local_poses.insert(
                        node_id,
                        TimedPose::new(
                            to_seconds_since_base(result.time),
                            to_isometry(&result.local_pose),
                        ),
                    );
                }
            }
            SensorData::Imu(imu) => {
                assert_eq!(extrinsic, Isometry3::identity());
                let imu_data = ImuData {
                    time,
                    linear_acceleration: imu.linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                };
                local_trajectory_builder.add_imu_data(&imu_data);
                pose_graph.add_imu_data(TRAJECTORY_ID, &imu_data);
            }
            _ => {}
        }
    }
    pose_graph.finish_trajectory(TRAJECTORY_ID);
    pose_graph.run_final_optimization();
    assert!(pose_graph.is_trajectory_finished(TRAJECTORY_ID));

    let nodes = pose_graph.get_trajectory_nodes();
    assert_eq!(nodes.size(), local_poses.len());
    let global_poses: Vec<TimedPose> = nodes
        .iter()
        .map(|node| {
            TimedPose::new(
                to_seconds_since_base(node.data.time()),
                to_isometry(&node.data.global_pose),
            )
        })
        .collect();
    let local_poses: Vec<TimedPose> = local_poses.into_values().collect();

    // Loop closures: inter-submap constraints of a node against a submap
    // older than the ones the node was inserted into.
    let constraints = pose_graph.constraints();
    let mut first_insertion_submap: BTreeMap<NodeId, i32> = BTreeMap::new();
    for constraint in constraints.iter().filter(|c| c.tag == Tag::IntraSubmap) {
        let index = first_insertion_submap
            .entry(constraint.node_id)
            .or_insert(i32::MAX);
        *index = (*index).min(constraint.submap_id.submap_index);
    }
    let inter: Vec<_> = constraints
        .iter()
        .filter(|c| c.tag == Tag::InterSubmap)
        .collect();
    let num_loop_closure_constraints = inter
        .iter()
        .filter(|c| c.submap_id.submap_index + 1 < first_insertion_submap[&c.node_id])
        .count();

    let reference = TimedPose::from_states(&dataset.ground_truth);
    let evaluate = |estimates: &[TimedPose]| {
        let pairs = associate(
            &reference,
            estimates,
            Association::Interpolate { max_gap: 0.05 },
        );
        assert_eq!(pairs.len(), estimates.len());
        let result = ate(&pairs, Alignment::Rigid3d);
        let (mut horizontal, mut vertical) = (0., 0.);
        for pair in &pairs {
            let error = result
                .alignment
                .transform_pose(&pair.estimate)
                .translation
                .vector
                - pair.reference.translation.vector;
            horizontal += error.xy().norm_squared();
            vertical += error.z * error.z;
        }
        let n = pairs.len() as f64;
        // Drift: relative pose error between the first and the last node.
        let (first, last) = (&pairs[0], &pairs[pairs.len() - 1]);
        let estimated = first.estimate.inverse() * last.estimate;
        let truth = first.reference.inverse() * last.reference;
        let end_error = (estimated.translation.vector - truth.translation.vector).norm();
        (
            Ate {
                rmse: result.translation.rmse,
                max: result.translation.max,
                horizontal_rmse: (horizontal / n).sqrt(),
                vertical_rmse: (vertical / n).sqrt(),
                rotation_max: result.rotation.max,
            },
            end_error,
        )
    };
    let (local_ate, local_end_error) = evaluate(&local_poses);
    let (global_ate, global_end_error) = evaluate(&global_poses);
    Outcome {
        num_nodes: nodes.size(),
        num_submaps: pose_graph.get_all_submap_data().size(),
        num_inter_submap_constraints: inter.len(),
        num_loop_closure_constraints,
        local_ate,
        global_ate,
        local_end_error,
        global_end_error,
        num_optimizations: num_optimizations.load(Ordering::Relaxed),
        seconds: start.elapsed().as_secs_f64(),
    }
}

/// Thresholds, from runs with seeds 1-4 and 1, 2 or 4 threads (results
/// are deterministic per seed, independent of the number of threads;
/// observed values in parentheses). Local SLAM (VLP-16-like LiDAR, 2 cm
/// range noise, handheld walking motion, IMU) drifts: ATE RMSE after rigid
/// 3D alignment 0.17-0.30 m (mostly vertical: the +-15 deg field of view
/// sees floor and ceiling only far away), drift between first and last node
/// 0.47-0.85 m. Loop closure (97 loop closure constraints after 1.25 laps,
/// 327-332 after 2 laps) and the IMU terms of the 3D optimization pull the
/// map together: global ATE RMSE 0.11-0.15 m (vertical 0.03 m; the
/// remaining horizontal error is a slow distortion along the 110 m loop
/// that SPA with rigid submaps cannot remove; 15 s windows aligned on their
/// own have 3-10 cm), max 0.22-0.36 m, rotation max 0.013-0.018 rad, end
/// drift 0.07-0.29 m. Required: RMSE < 0.2 m, max < 0.5 m, vertical RMSE
/// < 0.06 m, rotation max < 0.03 rad, RMSE at most 80 % of the local-only
/// RMSE (observed 38-74 %), and less end drift than local SLAM.
fn check(name: &str, outcome: &Outcome, min_loop_closure_constraints: usize) {
    println!("{name}: {outcome:?}");
    assert!(
        outcome.num_loop_closure_constraints >= min_loop_closure_constraints,
        "{name}: {outcome:?}"
    );
    assert!(
        outcome.num_inter_submap_constraints > outcome.num_loop_closure_constraints,
        "{name}: {outcome:?}"
    );
    assert!(outcome.num_optimizations > 0, "{name}: {outcome:?}");
    let (local, global) = (&outcome.local_ate, &outcome.global_ate);
    assert!(global.rmse < 0.2, "{name}: {outcome:?}");
    assert!(global.max < 0.5, "{name}: {outcome:?}");
    assert!(global.vertical_rmse < 0.06, "{name}: {outcome:?}");
    assert!(global.rotation_max < 0.03, "{name}: {outcome:?}");
    assert!(global.rmse < 0.8 * local.rmse, "{name}: {outcome:?}");
    assert!(
        outcome.global_end_error < outcome.local_end_error,
        "{name}: {outcome:?}"
    );
}

/// Two laps (~185 s, ~1810 nodes, 12 submaps) on a 4-thread pool; the whole
/// second lap closes loops against the first (observed 327 loop closure
/// constraints for this seed). Takes about 17 s.
#[test]
fn corridor_two_laps_loop_closure() {
    let outcome = run(2.0, 1, 4);
    assert_eq!(outcome.num_submaps, 12, "{outcome:?}");
    check("corridor_two_laps_loop_closure", &outcome, 150);
}

/// 1.25 laps on a single-threaded pool (the configuration of the C++ unit
/// tests): the loop is closed once (observed 97 loop closure constraints).
/// Takes about 14 s.
#[test]
fn corridor_loop_closure_single_thread() {
    let outcome = run(1.25, 2, 1);
    assert_eq!(outcome.num_submaps, 8, "{outcome:?}");
    check("corridor_loop_closure_single_thread", &outcome, 40);
}
