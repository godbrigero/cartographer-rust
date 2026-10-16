//! End-to-end test of global SLAM in 2D on simulated data (no C++
//! counterpart): `LocalTrajectoryBuilder2D` feeds `PoseGraph2D::add_node`
//! while a platform drives laps of `World2d::loop_corridor()` at 1.5 m/s
//! (`Trajectory::with_start_from_rest`), with a 2D LiDAR, an IMU and wheel
//! odometry (fed to both local SLAM and the pose graph; without odometry the
//! scan matcher slips along the long corridors during the start-up).
//!
//! Both use the shipped Lua defaults (`trajectory_builder_2d.lua`,
//! `pose_graph.lua`), adjusted like a cartographer_ros configuration
//! (`use_imu_data`). The pose graph runs on a real `ThreadPool`. Node poses
//! are compared with ground truth after a rigid 2D alignment, for the local
//! SLAM poses (no loop closure) and for the globally optimized poses.

use std::collections::BTreeMap;
use std::sync::Arc;

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::thread_pool::{ThreadPool, ThreadPoolInterface};
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::id::NodeId;
use cartographer::mapping::internal::optimization::optimization_problem_2d::OptimizationProblem2D;
use cartographer::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use cartographer::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
use cartographer::mapping::internal::two_d::pose_graph_2d::PoseGraph2D;
use cartographer::mapping::pose_graph::{PoseGraph, create_pose_graph_options};
use cartographer::mapping::pose_graph_interface::{PoseGraphInterface, Tag};
use cartographer::mapping::proto::{LocalTrajectoryBuilderOptions2D, PoseGraphOptions};
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::odometry_data::OdometryData;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::metrics::{Alignment, Association, TimedPose, associate, ate};
use cartographer_sim::{
    DatasetBuilder, ImuConfig, Lidar2dConfig, OdometryConfig, SensorData, Trajectory, World2d,
};
use nalgebra::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion};

/// Offset of the simulated time axis (simulated times start at 0 s).
fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

fn to_seconds_since_base(time: Time) -> f64 {
    (time - to_time(0.0)).to_seconds()
}

fn to_rigid3d(pose: &Isometry3<f64>) -> Rigid3d {
    let q = pose.rotation.quaternion();
    Rigid3d::new(pose.translation.vector, Quaternion::new(q.w, q.i, q.j, q.k))
}

fn to_isometry(pose: &Rigid3d) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(*pose.translation()),
        UnitQuaternion::from_quaternion(*pose.rotation()),
    )
}

fn dictionary(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::new(
        code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    )
}

fn load_trajectory_builder_options(overrides: &str) -> LocalTrajectoryBuilderOptions2D {
    create_local_trajectory_builder_options_2d(&dictionary(&format!(
        "include \"trajectory_builder_2d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_2D"
    )))
}

fn load_pose_graph_options(overrides: &str) -> PoseGraphOptions {
    create_pose_graph_options(&dictionary(&format!(
        "include \"pose_graph.lua\"\n{overrides}\nreturn POSE_GRAPH"
    )))
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
    local_ate_rmse: f64,
    local_ate_max: f64,
    global_ate_rmse: f64,
    global_ate_max: f64,
    /// Error of the last node relative to the first one (m), i.e. the
    /// accumulated drift at loop closure.
    local_end_error: f64,
    global_end_error: f64,
}

fn run(laps: f64, seed: u64, num_threads: i32) -> Outcome {
    let trajectory = Trajectory::corridor_loops(laps, 1.5).with_start_from_rest(3.0);
    let dataset = DatasetBuilder::new(trajectory, World2d::loop_corridor())
        .seed(seed)
        .lidar_2d("scan", Lidar2dConfig::default())
        .imu("imu", ImuConfig::default())
        .odometry("odom", OdometryConfig::default())
        .build();

    let thread_pool = Arc::new(ThreadPool::new(num_threads));
    let pose_graph_options = load_pose_graph_options("");
    let pose_graph = PoseGraph2D::new(
        &pose_graph_options,
        Box::new(OptimizationProblem2D::new(
            &pose_graph_options
                .optimization_problem_options
                .unwrap_or_default(),
        )),
        Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
    );
    const TRAJECTORY_ID: i32 = 0;

    let options = load_trajectory_builder_options("TRAJECTORY_BUILDER_2D.use_imu_data = true");
    let mut local_trajectory_builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
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
                if let Some(result) = local_trajectory_builder.add_range_data("scan", data)
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
            SensorData::Odometry(odometry) => {
                let odometry_data = OdometryData {
                    time,
                    pose: to_rigid3d(&(odometry.pose * extrinsic.inverse())),
                };
                local_trajectory_builder.add_odometry_data(&odometry_data);
                pose_graph.add_odometry_data(TRAJECTORY_ID, &odometry_data);
            }
            _ => {}
        }
    }
    pose_graph.finish_trajectory(TRAJECTORY_ID);
    pose_graph.run_final_optimization();

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
        let result = ate(&pairs, Alignment::Rigid2d);
        // Drift: relative pose error between the first and the last node.
        let (first, last) = (&pairs[0], &pairs[pairs.len() - 1]);
        let estimated = first.estimate.inverse() * last.estimate;
        let truth = first.reference.inverse() * last.reference;
        let end_error = (estimated.translation.vector - truth.translation.vector).norm();
        (result.translation.rmse, result.translation.max, end_error)
    };
    let (local_ate_rmse, local_ate_max, local_end_error) = evaluate(&local_poses);
    let (global_ate_rmse, global_ate_max, global_end_error) = evaluate(&global_poses);
    Outcome {
        num_nodes: nodes.size(),
        num_submaps: pose_graph.get_all_submap_data().size(),
        num_inter_submap_constraints: inter.len(),
        num_loop_closure_constraints,
        local_ate_rmse,
        local_ate_max,
        global_ate_rmse,
        global_ate_max,
        local_end_error,
        global_end_error,
    }
}

/// Thresholds, from runs with seeds 1-3 and 1 or 4 threads (observed values
/// in parentheses). Local SLAM (5 cm cells, 1 cm range noise, odometry + IMU)
/// is already good: ATE RMSE 2.1-2.6 cm, max 7-10 cm (mostly the start-up
/// transient), drift between first and last node 7-12 cm. Loop closure
/// (inter-submap constraints at loop_closure weights 1.1e4/1e5) pulls the
/// map together: global ATE RMSE 0.8-1.2 cm, max 2.7-4.3 cm, end drift
/// 2.4-6.3 cm. Required: RMSE < 2 cm and max < 6 cm (about one cell),
/// RMSE at most 75 % of the local-only RMSE, and less end drift than local
/// SLAM.
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
    assert!(outcome.global_ate_rmse < 0.02, "{name}: {outcome:?}");
    assert!(outcome.global_ate_max < 0.06, "{name}: {outcome:?}");
    assert!(
        outcome.global_ate_rmse < 0.75 * outcome.local_ate_rmse,
        "{name}: {outcome:?}"
    );
    assert!(
        outcome.global_end_error < outcome.local_end_error,
        "{name}: {outcome:?}"
    );
}

/// Two laps (~145 s, ~790 nodes, 9 submaps) on a 4-thread pool; the whole
/// second lap closes loops against the first (observed 117-129 loop closure
/// constraints).
#[test]
fn corridor_two_laps_loop_closure() {
    let outcome = run(2.0, 1, 4);
    assert_eq!(outcome.num_submaps, 9, "{outcome:?}");
    check("corridor_two_laps_loop_closure", &outcome, 50);
}

/// 1.25 laps on a single-threaded pool (the configuration of the C++ unit
/// tests): the loop is closed once (observed 29-35 loop closure
/// constraints).
#[test]
fn corridor_loop_closure_single_thread() {
    let outcome = run(1.25, 2, 1);
    check("corridor_loop_closure_single_thread", &outcome, 10);
}
