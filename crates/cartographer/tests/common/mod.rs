//! Shared helpers of the end-to-end acceptance tests (`e2e_2d.rs`,
//! `e2e_3d.rs`): configuration loading the way cartographer_ros does it,
//! feeding `cartographer-sim` datasets into a `MapBuilder`, trajectory /
//! constraint / map metrics, and the report artifacts (metrics table on
//! stdout, `target/e2e/<scenario>.json`, `target/e2e/<scenario>_map.png`).

#![allow(dead_code)] // Each test binary uses a different subset.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::io::file_writer::{FileWriter, StreamFileWriter};
use cartographer::io::image::Image;
use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::map_builder::MapBuilder;
use cartographer::mapping::map_builder_interface::{
    MapBuilderInterface, create_map_builder_options,
};
use cartographer::mapping::pose_graph_interface::{PoseGraphInterface, Tag};
use cartographer::mapping::probability_values::value_to_probability;
use cartographer::mapping::proto;
use cartographer::mapping::three_d::submap_3d::Submap3D;
use cartographer::mapping::trajectory_builder_interface::{
    InsertionResult, SensorId, SensorType, create_trajectory_builder_options,
};
use cartographer::mapping::two_d::grid_2d::Grid2DTrait;
use cartographer::mapping::two_d::probability_grid::ProbabilityGrid as CartoProbabilityGrid;
use cartographer::mapping::two_d::submap_2d::Submap2D;
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::odometry_data::OdometryData;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::metrics::{
    Alignment, Association, PosePair, RpeDelta, SimilarityTransform, TimedPose, associate, ate, rpe,
};
use cartographer_sim::{
    CellState, Dataset, GridLayout, GroundTruthGrid, ProbabilityGrid, SensorData, SensorKind,
    World2d, World3d,
};
use nalgebra::{
    Isometry3, Point2, Point3, Quaternion, Translation3, UnitQuaternion, Vector2, Vector3,
};

// ------------------------------------------------------------------ basics

/// Offset of the simulated time axis (simulated times start at 0 s).
pub fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

/// Inverse of [`to_time`].
pub fn to_sim_seconds(time: Time) -> f64 {
    (time - to_time(0.0)).to_seconds()
}

pub fn to_rigid3d(pose: &Isometry3<f64>) -> Rigid3d {
    let q = pose.rotation.quaternion();
    Rigid3d::new(pose.translation.vector, Quaternion::new(q.w, q.i, q.j, q.k))
}

pub fn to_isometry(pose: &Rigid3d) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(*pose.translation()),
        UnitQuaternion::new_normalize(*pose.rotation()),
    )
}

// ---------------------------------------------------------- configuration

/// Map builder and trajectory builder options of one run.
pub struct Options {
    pub map_builder: proto::MapBuilderOptions,
    pub trajectory_builder: proto::TrajectoryBuilderOptions,
}

/// Loads the options like a cartographer_ros configuration file: the shipped
/// `map_builder.lua` and `trajectory_builder.lua` (resolved from the shipped
/// `configuration_files`), the dimension switch, then `overrides` (Lua
/// statements on `MAP_BUILDER`, `POSE_GRAPH`, `TRAJECTORY_BUILDER`,
/// `TRAJECTORY_BUILDER_2D` and `TRAJECTORY_BUILDER_3D`), and finally the
/// `options.map_builder` / `options.trajectory_builder` dictionaries are
/// parsed with `create_map_builder_options` and
/// `create_trajectory_builder_options`.
pub fn load_options(three_d: bool, overrides: &str) -> Options {
    let dimension = if three_d { "3d" } else { "2d" };
    let code = format!(
        "include \"map_builder.lua\"\n\
         include \"trajectory_builder.lua\"\n\
         options = {{\n\
           map_builder = MAP_BUILDER,\n\
           trajectory_builder = TRAJECTORY_BUILDER,\n\
         }}\n\
         MAP_BUILDER.use_trajectory_builder_{dimension} = true\n\
         {overrides}\n\
         return options"
    );
    let dictionary = LuaParameterDictionary::new(
        &code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    );
    Options {
        map_builder: create_map_builder_options(&dictionary.get_dictionary("map_builder")),
        trajectory_builder: create_trajectory_builder_options(
            &dictionary.get_dictionary("trajectory_builder"),
        ),
    }
}

/// The expected sensor ids of a dataset (C++ `SensorId`s the way
/// cartographer_ros derives them from the topics).
pub fn sensor_ids(dataset: &Dataset) -> BTreeSet<SensorId> {
    dataset
        .sensors
        .iter()
        .map(|sensor| SensorId {
            r#type: match sensor.kind {
                SensorKind::Lidar2d | SensorKind::Lidar3d => SensorType::Range,
                SensorKind::Imu => SensorType::Imu,
                SensorKind::Odometry => SensorType::Odometry,
                SensorKind::FixedFramePose => SensorType::FixedFramePose,
                SensorKind::Landmark => SensorType::Landmark,
            },
            id: sensor.id.clone(),
        })
        .collect()
}

// ------------------------------------------------------------------ driving

/// How [`run_trajectory`] feeds the data.
#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    /// Real-time pacing of the global SLAM backend: after each local SLAM
    /// result, wait until the pose graph has optimized up to at most this
    /// many nodes behind the newest node. Offline, local SLAM runs ~100x
    /// faster than real time and the pose graph work queue would otherwise
    /// fall thousands of items behind, so that constraint search windows,
    /// trimming and the published `local_to_global` transform would reflect
    /// the feeding speed instead of a live system keeping up with its
    /// sensors. `None`: feed as fast as possible.
    pub max_backend_lag_nodes: Option<i32>,
}

impl RunOptions {
    /// Paced with a lag of two optimization periods.
    pub fn paced(options: &proto::MapBuilderOptions) -> Self {
        let every_n = options
            .pose_graph_options
            .as_ref()
            .map_or(0, |o| o.optimize_every_n_nodes);
        RunOptions {
            max_backend_lag_nodes: (every_n > 0).then_some(2 * every_n),
        }
    }
}

/// What a trajectory run recorded through the local SLAM result callback.
pub struct RunLog {
    pub trajectory_id: i32,
    /// Local SLAM poses (`local_pose` of every local SLAM result).
    pub local_poses: Vec<TimedPose>,
    /// The pose published "online" (cartographer_ros' `map -> tracking`):
    /// `local_to_global * local_pose` right after each local SLAM result,
    /// with the then-current `GetLocalToGlobalTransform`.
    pub online_poses: Vec<TimedPose>,
    /// Time of the first local SLAM result at which an inter-trajectory
    /// constraint of this trajectory existed (only searched for if other
    /// trajectories exist).
    pub first_connected_time: Option<f64>,
    /// Time of the first local SLAM result after `first_connected_time` at
    /// which `local_to_global` had changed, i.e. global SLAM had placed the
    /// trajectory into the map using the connection.
    pub first_placed_time: Option<f64>,
    /// Global poses of the trajectory's nodes right before
    /// `finish_trajectory` (a pure localization trimmer removes all of them
    /// once the trajectory is finished).
    pub nodes_before_finish: Vec<TimedPose>,
    /// Number of the trajectory's submaps right before `finish_trajectory`.
    pub submaps_before_finish: usize,
    /// Constraints right before `finish_trajectory`.
    pub constraints_before_finish: ConstraintSummary,
    pub num_range_messages: usize,
    pub seconds: f64,
}

/// Adds a trajectory for `dataset` to `map_builder` (sensor ids from the
/// dataset, `options`), feeds all messages in time order through
/// `get_trajectory_builder(id).add_*` (sensor data transformed into the
/// tracking frame with the sensor extrinsics, like cartographer_ros' TF
/// lookups) and finishes the trajectory.
pub fn run_trajectory(
    map_builder: &mut MapBuilder,
    dataset: &Dataset,
    options: &proto::TrajectoryBuilderOptions,
    run_options: RunOptions,
) -> RunLog {
    let start = std::time::Instant::now();
    type LocalResult = (Time, Rigid3d, Option<i32>);
    let results: Arc<Mutex<Vec<LocalResult>>> = Arc::default();
    let callback = {
        let results = Arc::clone(&results);
        Arc::new(
            move |_, time, local_pose, _, insertion_result: Option<InsertionResult>| {
                results
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((
                        time,
                        local_pose,
                        insertion_result.map(|r| r.node_id.node_index),
                    ));
            },
        )
    };
    let pose_graph = map_builder.pose_graph();
    // Last optimized node index per trajectory.
    let optimized: Arc<Mutex<BTreeMap<i32, i32>>> = Arc::default();
    {
        let optimized = Arc::clone(&optimized);
        pose_graph.set_global_slam_optimization_callback(Box::new(move |_, nodes| {
            let mut optimized = optimized.lock().unwrap_or_else(PoisonError::into_inner);
            for (&trajectory_id, node_id) in nodes {
                optimized.insert(trajectory_id, node_id.node_index);
            }
        }));
    }
    let trajectory_id =
        map_builder.add_trajectory_builder(&sensor_ids(dataset), options, Some(callback));
    let mut local_poses = Vec::new();
    let mut online_poses = Vec::new();
    let mut first_connected_time = None;
    let mut first_placed_time = None;
    let mut local_to_global_at_connection: Option<Rigid3d> = None;
    let mut last_node_index = -1;
    let mut num_range_messages = 0;
    {
        let builder = map_builder
            .get_trajectory_builder(trajectory_id)
            .expect("live trajectory builder");
        for message in &dataset.messages {
            let extrinsic = dataset
                .sensor(&message.sensor_id)
                .expect("known sensor")
                .extrinsic;
            let time = to_time(message.time);
            match &message.data {
                SensorData::PointCloud(cloud) => {
                    num_range_messages += 1;
                    let extrinsic_f32 = extrinsic.cast::<f32>();
                    let transform = |p| (extrinsic_f32 * Point3::from(p)).coords;
                    builder.add_timed_point_cloud_data(
                        &message.sensor_id,
                        &TimedPointCloudData {
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
                        },
                    );
                }
                SensorData::Imu(imu) => {
                    // cartographer_ros requires the IMU to be colocated
                    // with the tracking frame and only rotates the data.
                    assert!(extrinsic.translation.vector.norm() < 1e-5);
                    builder.add_imu_data(
                        &message.sensor_id,
                        &ImuData {
                            time,
                            linear_acceleration: extrinsic.rotation * imu.linear_acceleration,
                            angular_velocity: extrinsic.rotation * imu.angular_velocity,
                        },
                    );
                }
                SensorData::Odometry(odometry) => builder.add_odometry_data(
                    &message.sensor_id,
                    &OdometryData {
                        time,
                        pose: to_rigid3d(&(odometry.pose * extrinsic.inverse())),
                    },
                ),
                _ => panic!("unexpected sensor data in an e2e dataset"),
            }
            let new_results: Vec<LocalResult> =
                std::mem::take(&mut *results.lock().unwrap_or_else(PoisonError::into_inner));
            if new_results.is_empty() {
                continue;
            }
            if trajectory_id > 0 && first_connected_time.is_none() {
                let connected = pose_graph.constraints().iter().any(|c| {
                    c.tag == Tag::InterSubmap
                        && c.node_id.trajectory_id == trajectory_id
                        && c.submap_id.trajectory_id != trajectory_id
                });
                if connected {
                    first_connected_time = Some(to_sim_seconds(new_results[0].0));
                    local_to_global_at_connection =
                        Some(pose_graph.get_local_to_global_transform(trajectory_id));
                }
            }
            let local_to_global = pose_graph.get_local_to_global_transform(trajectory_id);
            for (time, local_pose, node_index) in new_results {
                let seconds = to_sim_seconds(time);
                if first_placed_time.is_none()
                    && local_to_global_at_connection.is_some_and(|before| before != local_to_global)
                {
                    first_placed_time = Some(seconds);
                }
                local_poses.push(TimedPose::new(seconds, to_isometry(&local_pose)));
                online_poses.push(TimedPose::new(
                    seconds,
                    to_isometry(&(local_to_global * local_pose)),
                ));
                if let Some(node_index) = node_index {
                    last_node_index = node_index;
                }
            }
            if let Some(max_lag) = run_options.max_backend_lag_nodes {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
                loop {
                    let done = optimized
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .get(&trajectory_id)
                        .copied()
                        .unwrap_or(-1);
                    if last_node_index - done <= max_lag {
                        break;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "global SLAM stalled at node {done} (newest {last_node_index})"
                    );
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
    }
    let nodes_before_finish = global_node_poses(pose_graph.as_ref(), trajectory_id);
    let submaps_before_finish = num_submaps(pose_graph.as_ref(), trajectory_id);
    let constraints_before_finish = constraint_summary(pose_graph.as_ref());
    map_builder.finish_trajectory(trajectory_id);
    RunLog {
        trajectory_id,
        local_poses,
        online_poses,
        first_connected_time,
        first_placed_time,
        nodes_before_finish,
        submaps_before_finish,
        constraints_before_finish,
        num_range_messages,
        seconds: start.elapsed().as_secs_f64(),
    }
}

/// Optimized global poses of the (untrimmed) nodes of `trajectory_id`.
pub fn global_node_poses(
    pose_graph: &dyn PoseGraphInterface,
    trajectory_id: i32,
) -> Vec<TimedPose> {
    pose_graph
        .get_trajectory_nodes()
        .trajectory(trajectory_id)
        .map(|node| {
            TimedPose::new(
                to_sim_seconds(node.data.time()),
                to_isometry(&node.data.global_pose),
            )
        })
        .collect()
}

// ------------------------------------------------------- trajectory metrics

/// Planar projection `(x, y, yaw)` of poses (2D SLAM node poses contain the
/// IMU gravity alignment's roll and pitch, which 2D SLAM does not
/// estimate as part of the map).
pub fn planar(poses: &[TimedPose]) -> Vec<TimedPose> {
    poses
        .iter()
        .map(|p| TimedPose::from_2d(p.time, &cartographer_sim::trajectory::project_2d(&p.pose)))
        .collect()
}

/// Pairs estimated poses with the (interpolated) ground truth.
pub fn pairs_with_ground_truth(dataset: &Dataset, estimate: &[TimedPose]) -> Vec<PosePair> {
    let reference = TimedPose::from_states(&dataset.ground_truth);
    let pairs = associate(
        &reference,
        estimate,
        Association::Interpolate { max_gap: 0.05 },
    );
    assert_eq!(
        pairs.len(),
        estimate.len(),
        "estimates outside the GT range"
    );
    pairs
}

/// Trajectory accuracy of one set of pose pairs.
#[derive(Clone, Debug)]
pub struct TrajectoryMetrics {
    pub num_poses: usize,
    /// Path length of the ground truth (m).
    pub path_length: f64,
    /// ATE after the alignment (m / rad).
    pub ate_rmse: f64,
    pub ate_max: f64,
    pub ate_rotation_rmse: f64,
    pub ate_rotation_max: f64,
    /// Horizontal / vertical ATE RMSE (m).
    pub ate_horizontal_rmse: f64,
    pub ate_vertical_rmse: f64,
    /// RPE over 1 m of travel: translation error per metre (m/m), rotation
    /// error per metre (rad/m).
    pub rpe_translation_rmse: f64,
    pub rpe_translation_max: f64,
    pub rpe_rotation_rmse: f64,
    pub rpe_rotation_max: f64,
    /// Translation error of the last pose relative to the first one (m).
    pub end_drift: f64,
    /// `reference_from_estimate`.
    pub alignment: Isometry3<f64>,
    /// Per pose `[time, ATE translation (m), ATE rotation (rad), RPE
    /// translation over 1 m (m), RPE rotation over 1 m (rad)]` (RPE `NaN`
    /// within the last metre), for the report plots.
    pub errors: Vec<[f64; 5]>,
}

/// Computes ATE (after `alignment`, or after the given fixed
/// `reference_from_estimate` transform), RPE and end drift.
pub fn trajectory_metrics(pairs: &[PosePair], alignment: AlignWith) -> TrajectoryMetrics {
    assert!(pairs.len() >= 2);
    let transform = match alignment {
        AlignWith::Fit(alignment) => ate(pairs, alignment).alignment,
        AlignWith::Fixed(isometry) => SimilarityTransform {
            isometry,
            scale: 1.0,
        },
    };
    let mut translation = Vec::with_capacity(pairs.len());
    let mut rotation = Vec::with_capacity(pairs.len());
    let (mut horizontal, mut vertical) = (0.0, 0.0);
    for pair in pairs {
        let estimate = transform.transform_pose(&pair.estimate);
        let error = estimate.translation.vector - pair.reference.translation.vector;
        translation.push(error.norm());
        rotation.push(estimate.rotation.angle_to(&pair.reference.rotation));
        horizontal += error.xy().norm_squared();
        vertical += error.z * error.z;
    }
    let n = pairs.len() as f64;
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let relative = rpe(pairs, RpeDelta::Distance(1.0));
    let relative_series = rpe_series(pairs, 1.0);
    let errors = pairs
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let [t, r] = relative_series[i];
            [pair.time, translation[i], rotation[i], t, r]
        })
        .collect();
    let (first, last) = (&pairs[0], &pairs[pairs.len() - 1]);
    let estimated = first.estimate.inverse() * last.estimate;
    let truth = first.reference.inverse() * last.reference;
    let path_length = pairs
        .windows(2)
        .map(|w| (w[1].reference.translation.vector - w[0].reference.translation.vector).norm())
        .sum();
    TrajectoryMetrics {
        num_poses: pairs.len(),
        path_length,
        ate_rmse: rms(&translation),
        ate_max: max(&translation),
        ate_rotation_rmse: rms(&rotation),
        ate_rotation_max: max(&rotation),
        ate_horizontal_rmse: (horizontal / n).sqrt(),
        ate_vertical_rmse: (vertical / n).sqrt(),
        rpe_translation_rmse: relative.translation.rmse,
        rpe_translation_max: relative.translation.max,
        rpe_rotation_rmse: relative.rotation.rmse,
        rpe_rotation_max: relative.rotation.max,
        end_drift: (estimated.translation.vector - truth.translation.vector).norm(),
        alignment: transform.isometry,
        errors,
    }
}

/// Per pair `[translation, rotation]` relative pose error to the first pair
/// at least `distance` metres of ground-truth travel later (the samples of
/// `cartographer_sim::metrics::rpe` with `RpeDelta::Distance`), `NaN` where
/// there is no such pair.
fn rpe_series(pairs: &[PosePair], distance: f64) -> Vec<[f64; 2]> {
    let mut travelled = Vec::with_capacity(pairs.len());
    let mut sum = 0.0;
    for (k, pair) in pairs.iter().enumerate() {
        if k > 0 {
            sum += (pair.reference.translation.vector - pairs[k - 1].reference.translation.vector)
                .norm();
        }
        travelled.push(sum);
    }
    let mut j = 0;
    (0..pairs.len())
        .map(|i| {
            j = j.max(i + 1);
            while j < pairs.len() && travelled[j] - travelled[i] < distance - 1e-9 {
                j += 1;
            }
            if j >= pairs.len() {
                return [f64::NAN; 2];
            }
            let (a, b) = (&pairs[i], &pairs[j]);
            let error = (a.reference.inverse() * b.reference).inverse()
                * (a.estimate.inverse() * b.estimate);
            [error.translation.vector.norm(), error.rotation.angle()]
        })
        .collect()
}

/// Rounds to `decimals` decimal places (keeps the JSON reports small).
pub fn rounded(x: f64, decimals: i32) -> Json {
    let scale = 10_f64.powi(decimals);
    Json::from((x * scale).round() / scale)
}

/// How estimates are brought into the ground-truth frame.
#[derive(Clone, Copy, Debug)]
pub enum AlignWith {
    /// Least-squares fit of the given kind on these pairs.
    Fit(Alignment),
    /// A fixed `reference_from_estimate` transform.
    Fixed(Isometry3<f64>),
}

impl TrajectoryMetrics {
    pub fn to_json(&self) -> Json {
        Json::object([
            ("num_poses", Json::from(self.num_poses)),
            ("path_length_m", self.path_length.into()),
            ("ate_rmse_m", self.ate_rmse.into()),
            ("ate_max_m", self.ate_max.into()),
            ("ate_rotation_rmse_rad", self.ate_rotation_rmse.into()),
            ("ate_rotation_max_rad", self.ate_rotation_max.into()),
            ("ate_horizontal_rmse_m", self.ate_horizontal_rmse.into()),
            ("ate_vertical_rmse_m", self.ate_vertical_rmse.into()),
            (
                "rpe_1m_translation_rmse_m_per_m",
                self.rpe_translation_rmse.into(),
            ),
            (
                "rpe_1m_translation_max_m_per_m",
                self.rpe_translation_max.into(),
            ),
            (
                "rpe_1m_rotation_rmse_rad_per_m",
                self.rpe_rotation_rmse.into(),
            ),
            (
                "rpe_1m_rotation_max_rad_per_m",
                self.rpe_rotation_max.into(),
            ),
            ("end_drift_m", self.end_drift.into()),
            ("alignment", isometry_json(&self.alignment)),
            (
                // [t, ate_m, ate_rad, rpe_1m_m, rpe_1m_rad], at most ~500 rows.
                "errors_over_time",
                Json::Array(
                    self.errors
                        .iter()
                        .step_by(self.errors.len().div_ceil(500).max(1))
                        .map(|row| {
                            Json::Array(vec![
                                rounded(row[0], 3),
                                rounded(row[1], 5),
                                rounded(row[2], 6),
                                rounded(row[3], 5),
                                rounded(row[4], 6),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

/// Optimized global node and submap positions in the ground-truth frame
/// (`world_from_map` applied), for drawing the pose graph:
/// `{"nodes": [[trajectory, index, x, y, z], ...], "submaps": [...]}`.
pub fn pose_graph_json(
    pose_graph: &dyn PoseGraphInterface,
    world_from_map: &Isometry3<f64>,
) -> Json {
    let row = |trajectory: i32, index: i32, pose: &Rigid3d| {
        let p = world_from_map * Point3::from(to_isometry(pose).translation.vector);
        Json::Array(vec![
            i64::from(trajectory).into(),
            i64::from(index).into(),
            rounded(p.x, 3),
            rounded(p.y, 3),
            rounded(p.z, 3),
        ])
    };
    Json::object([
        (
            "nodes",
            Json::Array(
                pose_graph
                    .get_trajectory_nodes()
                    .iter()
                    .map(|node| {
                        row(
                            node.id.trajectory_id,
                            node.id.node_index,
                            &node.data.global_pose,
                        )
                    })
                    .collect(),
            ),
        ),
        (
            "submaps",
            Json::Array(
                pose_graph
                    .get_all_submap_data()
                    .iter()
                    .map(|submap| {
                        row(
                            submap.id.trajectory_id,
                            submap.id.submap_index,
                            &submap.data.pose,
                        )
                    })
                    .collect(),
            ),
        ),
    ])
}

/// At most `max_points` of `points` (uniform stride), `[[x, y, z], ...]`
/// rounded to centimetres.
pub fn points_json(points: &[Point3<f64>], max_points: usize) -> Json {
    Json::Array(
        points
            .iter()
            .step_by(points.len().div_ceil(max_points.max(1)).max(1))
            .map(|p| Json::Array(vec![rounded(p.x, 2), rounded(p.y, 2), rounded(p.z, 2)]))
            .collect(),
    )
}

// ------------------------------------------------------------- constraints

/// Constraint counts of a pose graph.
#[derive(Clone, Debug, Default)]
pub struct ConstraintSummary {
    pub total: usize,
    pub intra_submap: usize,
    pub inter_submap: usize,
    /// Inter-submap constraints between a node and a submap of the same
    /// trajectory that is older than the submaps the node was inserted into
    /// (more than one submap back), i.e. loop closures.
    pub loop_closure: usize,
    /// Inter-submap constraints between different trajectories.
    pub inter_trajectory: usize,
    /// Inter-submap constraints per `(node trajectory, submap trajectory)`.
    pub inter_by_trajectory_pair: BTreeMap<(i32, i32), usize>,
    /// `(submap, node, kind)` of every inter-submap constraint; kind 0:
    /// local (same trajectory, recent submap), 1: loop closure, 2:
    /// inter-trajectory.
    pub inter_list: Vec<(SubmapId, NodeId, u8)>,
}

pub fn constraint_summary(pose_graph: &dyn PoseGraphInterface) -> ConstraintSummary {
    let constraints = pose_graph.constraints();
    let mut first_insertion_submap: BTreeMap<NodeId, i32> = BTreeMap::new();
    for constraint in constraints.iter().filter(|c| c.tag == Tag::IntraSubmap) {
        let index = first_insertion_submap
            .entry(constraint.node_id)
            .or_insert(i32::MAX);
        *index = (*index).min(constraint.submap_id.submap_index);
    }
    let mut summary = ConstraintSummary {
        total: constraints.len(),
        ..Default::default()
    };
    for constraint in &constraints {
        match constraint.tag {
            Tag::IntraSubmap => summary.intra_submap += 1,
            Tag::InterSubmap => {
                summary.inter_submap += 1;
                let (node, submap) = (constraint.node_id, constraint.submap_id);
                *summary
                    .inter_by_trajectory_pair
                    .entry((node.trajectory_id, submap.trajectory_id))
                    .or_default() += 1;
                let kind = if node.trajectory_id != submap.trajectory_id {
                    summary.inter_trajectory += 1;
                    2
                } else if first_insertion_submap
                    .get(&node)
                    .is_some_and(|&first| submap.submap_index + 1 < first)
                {
                    summary.loop_closure += 1;
                    1
                } else {
                    0
                };
                summary.inter_list.push((submap, node, kind));
            }
        }
    }
    summary
}

impl ConstraintSummary {
    pub fn to_json(&self) -> Json {
        Json::object([
            ("total", Json::from(self.total)),
            ("intra_submap", self.intra_submap.into()),
            ("inter_submap", self.inter_submap.into()),
            ("loop_closure", self.loop_closure.into()),
            ("inter_trajectory", self.inter_trajectory.into()),
            (
                "inter_by_trajectory_pair",
                Json::Array(
                    self.inter_by_trajectory_pair
                        .iter()
                        .map(|(&(node, submap), &count)| {
                            Json::object([
                                ("node_trajectory", Json::from(node as i64)),
                                ("submap_trajectory", (submap as i64).into()),
                                ("count", count.into()),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                // [submap_trajectory, submap_index, node_trajectory, node_index,
                //  kind (0 local, 1 loop closure, 2 inter-trajectory)]
                "inter_submap_constraints",
                Json::Array(
                    self.inter_list
                        .iter()
                        .map(|(submap, node, kind)| {
                            Json::Array(vec![
                                (submap.trajectory_id as i64).into(),
                                (submap.submap_index as i64).into(),
                                (node.trajectory_id as i64).into(),
                                (node.node_index as i64).into(),
                                i64::from(*kind).into(),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

// ------------------------------------------------------------------ 2D map

/// Composes the 2D submap probability grids into one grid on `layout`
/// (ground-truth frame): every cell centre is mapped into the map frame
/// (`world_from_map^-1`), into each submap's local grid frame with the
/// optimized global submap pose (`local_pose * global_pose^-1`, as the
/// submap painter places submap textures) and looked up there. Finished
/// and unfinished submaps are used; where several submaps know a cell their
/// probabilities are combined by averaging log-odds (each range datum is
/// inserted into two overlapping submaps, so summing would double count).
pub fn compose_global_grid_2d(
    pose_graph: &dyn PoseGraphInterface,
    layout: GridLayout,
    world_from_map: &Isometry3<f64>,
) -> ProbabilityGrid {
    let mut log_odds_sum = vec![0.0_f64; layout.len()];
    let mut count = vec![0_u32; layout.len()];
    let map_from_world = world_from_map.inverse();
    for submap in pose_graph.get_all_submap_data().iter() {
        let Some(shared) = &submap.data.submap else {
            continue;
        };
        let guard = shared.read().unwrap_or_else(PoisonError::into_inner);
        let any: &dyn std::any::Any = &*guard;
        let submap_2d = any.downcast_ref::<Submap2D>().expect("2D submaps");
        let Some(grid) = submap_2d.try_grid() else {
            continue;
        };
        let grid = grid
            .downcast_ref::<CartoProbabilityGrid>()
            .expect("probability grid submaps");
        let local_from_world = to_isometry(&guard.local_pose())
            * to_isometry(&submap.data.pose).inverse()
            * map_from_world;
        let limits = grid.limits();
        for iy in 0..layout.height {
            for ix in 0..layout.width {
                let center = layout.cell_center(ix, iy);
                let local = local_from_world * Point3::new(center.x, center.y, 0.0);
                let index = limits.get_cell_index(&Vector2::new(local.x as f32, local.y as f32));
                if !limits.contains(&index) || !grid.is_known(&index) {
                    continue;
                }
                let p = f64::from(grid.get_probability(&index));
                let i = layout.index(ix, iy);
                log_odds_sum[i] += (p / (1.0 - p)).ln();
                count[i] += 1;
            }
        }
    }
    let cells = log_odds_sum
        .iter()
        .zip(&count)
        .map(|(&sum, &n)| {
            (n > 0).then(|| {
                let odds = (sum / f64::from(n)).exp();
                (odds / (1.0 + odds)) as f32
            })
        })
        .collect();
    ProbabilityGrid { layout, cells }
}

/// Restricts `ground_truth` to what a 2D LiDAR at `sensor_poses` (GT
/// frame) observes well: occupied cells are kept only if at least
/// `min_hits` rays hit their wall (within one cell) at a range below
/// `max_range` and an angle of incidence below 70 deg; free cells only if
/// such a ray passes through them; everything else becomes `Unknown`.
/// Recall and free-space coverage are then measured against the
/// well-observed part of the world (e.g. the office tour does not enter
/// every room, and walls only glimpsed through a door or grazed from far
/// away get too few hits to become occupied in any SLAM map).
pub fn observable_ground_truth(
    world: &World2d,
    ground_truth: &GroundTruthGrid,
    sensor_poses: &[Isometry3<f64>],
    max_range: f64,
    num_beams: usize,
    min_hits: u32,
) -> GroundTruthGrid {
    let layout = ground_truth.layout;
    let mut passed = vec![false; layout.len()];
    let mut hits = vec![0_u32; layout.len()];
    for pose in sensor_poses {
        let origin = Point2::from(pose.translation.vector.xy());
        for k in 0..num_beams {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / num_beams as f64;
            let direction = Vector2::new(angle.cos(), angle.sin());
            let hit = world.raycast(&origin, &direction, max_range);
            let range = hit.map_or(max_range, |hit| hit.range);
            let num_steps = (range / (0.5 * layout.resolution)).ceil().max(1.0) as usize;
            for step in 0..num_steps {
                let p = origin + direction * (range * step as f64 / num_steps as f64);
                if let Some((ix, iy)) = layout.cell_of(&p) {
                    passed[layout.index(ix, iy)] = true;
                }
            }
            let Some(hit) = hit else { continue };
            if hit.normal.dot(&direction).abs() < 70_f64.to_radians().cos() {
                continue;
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let p = hit.point + Vector2::new(dx as f64, dy as f64) * layout.resolution;
                    if let Some((ix, iy)) = layout.cell_of(&p) {
                        hits[layout.index(ix, iy)] += 1;
                    }
                }
            }
        }
    }
    GroundTruthGrid {
        layout,
        cells: ground_truth
            .cells
            .iter()
            .enumerate()
            .map(|(i, &cell)| match cell {
                CellState::Occupied if hits[i] >= min_hits => CellState::Occupied,
                CellState::Free if passed[i] => CellState::Free,
                _ => CellState::Unknown,
            })
            .collect(),
    }
}

/// Number of submaps of `trajectory_id` still in the pose graph.
pub fn num_submaps(pose_graph: &dyn PoseGraphInterface, trajectory_id: i32) -> usize {
    pose_graph
        .get_all_submap_data()
        .trajectory(trajectory_id)
        .count()
}

/// Writes the composed grid as PNG (north up): free white, occupied black,
/// unknown grey; ground-truth walls missing from the map in red; the
/// trajectories (`(poses in the GT frame, RGB)`) on top.
pub fn write_map_png(
    path: &Path,
    grid: &ProbabilityGrid,
    ground_truth: &GroundTruthGrid,
    occupied_threshold: f32,
    trajectories: &[(Vec<Vector2<f64>>, [u8; 3])],
) {
    let layout = grid.layout;
    let (width, height) = (layout.width as i32, layout.height as i32);
    let mut image = Image::new(width, height);
    for iy in 0..layout.height {
        for ix in 0..layout.width {
            let i = layout.index(ix, iy);
            let color = match grid.cells[i] {
                None => [128, 128, 128],
                Some(p) => {
                    let v = (255.0 * (1.0 - p)).round().clamp(0.0, 255.0) as u8;
                    [v, v, v]
                }
            };
            let missed = ground_truth.cells[i] == CellState::Occupied
                && !grid.cells[i].is_some_and(|p| p >= occupied_threshold);
            let color = if missed { [230, 60, 60] } else { color };
            image.set_pixel(ix as i32, height - 1 - iy as i32, &color);
        }
    }
    for (points, color) in trajectories {
        for point in points {
            if let Some((ix, iy)) = layout.cell_of(&Point2::from(*point)) {
                image.set_pixel(ix as i32, height - 1 - iy as i32, color);
            }
        }
    }
    let mut writer = StreamFileWriter::new(path.to_str().unwrap());
    image.write_png(&mut writer);
    assert!(writer.close());
}

// ------------------------------------------------------------------ 3D map

/// Centres of the high-resolution hybrid grid voxels with probability
/// `>= min_probability` of all submaps, in the ground-truth frame (via the
/// optimized global submap poses and `world_from_map`).
pub fn occupied_voxels_3d(
    pose_graph: &dyn PoseGraphInterface,
    world_from_map: &Isometry3<f64>,
    min_probability: f32,
) -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    for submap in pose_graph.get_all_submap_data().iter() {
        let Some(shared) = &submap.data.submap else {
            continue;
        };
        let guard = shared.read().unwrap_or_else(PoisonError::into_inner);
        let any: &dyn std::any::Any = &*guard;
        let submap_3d = any.downcast_ref::<Submap3D>().expect("3D submaps");
        // The hybrid grids are in the submap frame.
        let world_from_submap = world_from_map * to_isometry(&submap.data.pose);
        let grid = submap_3d.high_resolution_hybrid_grid();
        for (index, value) in grid {
            if value_to_probability(*value) >= min_probability {
                let center = grid.get_center_of_cell(&index).cast::<f64>();
                points.push(world_from_submap * Point3::from(center));
            }
        }
    }
    points
}

/// Distance from `p` to the closest surface of `world` (box faces and
/// planes).
pub fn distance_to_world_surface(world: &World3d, p: &Point3<f64>) -> f64 {
    let mut best = f64::INFINITY;
    for b in &world.boxes {
        let outside = Vector3::new(
            (b.min.x - p.x).max(p.x - b.max.x).max(0.0),
            (b.min.y - p.y).max(p.y - b.max.y).max(0.0),
            (b.min.z - p.z).max(p.z - b.max.z).max(0.0),
        );
        let d = if outside == Vector3::zeros() {
            (p.x - b.min.x)
                .min(b.max.x - p.x)
                .min(p.y - b.min.y)
                .min(b.max.y - p.y)
                .min(p.z - b.min.z)
                .min(b.max.z - p.z)
        } else {
            outside.norm()
        };
        best = best.min(d);
    }
    for plane in &world.planes {
        best = best.min((plane.normal.dot(&p.coords) - plane.offset).abs());
    }
    best
}

/// Map-to-surface distance statistics (m).
#[derive(Clone, Debug)]
pub struct SurfaceDistance {
    pub num_points: usize,
    pub median: f64,
    pub p95: f64,
    pub mean: f64,
    /// Fraction of points farther than 0.2 m from any surface.
    pub outlier_fraction: f64,
}

pub fn surface_distance(world: &World3d, points: &[Point3<f64>]) -> SurfaceDistance {
    assert!(!points.is_empty());
    let mut d: Vec<f64> = points
        .iter()
        .map(|p| distance_to_world_surface(world, p))
        .collect();
    d.sort_by(f64::total_cmp);
    let quantile = |q: f64| d[((d.len() - 1) as f64 * q).round() as usize];
    SurfaceDistance {
        num_points: d.len(),
        median: quantile(0.5),
        p95: quantile(0.95),
        mean: d.iter().sum::<f64>() / d.len() as f64,
        outlier_fraction: d.iter().filter(|&&x| x > 0.2).count() as f64 / d.len() as f64,
    }
}

// ------------------------------------------------------------------ report

/// A tiny JSON value (the workspace has no serde).
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn object<K: Into<String>>(entries: impl IntoIterator<Item = (K, Json)>) -> Self {
        Json::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(x) if x.is_finite() => write!(out, "{x}").unwrap(),
            Json::Number(_) => out.push_str("null"),
            Json::String(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Json::Array(values) => {
                out.push('[');
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    value.write(out);
                }
                out.push(']');
            }
            Json::Object(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    Json::String(key.clone()).write(out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }

    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }
}

impl From<f64> for Json {
    fn from(x: f64) -> Self {
        Json::Number(x)
    }
}
impl From<usize> for Json {
    fn from(x: usize) -> Self {
        Json::Number(x as f64)
    }
}
impl From<i64> for Json {
    fn from(x: i64) -> Self {
        Json::Number(x as f64)
    }
}
impl From<bool> for Json {
    fn from(x: bool) -> Self {
        Json::Bool(x)
    }
}
impl From<&str> for Json {
    fn from(x: &str) -> Self {
        Json::String(x.to_string())
    }
}

pub fn isometry_json(pose: &Isometry3<f64>) -> Json {
    let t = pose.translation.vector;
    let q = pose.rotation.quaternion();
    Json::object([
        (
            "translation",
            Json::Array(vec![t.x.into(), t.y.into(), t.z.into()]),
        ),
        (
            "rotation_wxyz",
            Json::Array(vec![q.w.into(), q.i.into(), q.j.into(), q.k.into()]),
        ),
    ])
}

/// `[[t, x, y, z, yaw], ...]` of every `step`-th pose (plus the last).
pub fn poses_json(poses: &[TimedPose], step: usize) -> Json {
    let mut indices: Vec<usize> = (0..poses.len()).step_by(step.max(1)).collect();
    if poses.len() > 1 && indices.last() != Some(&(poses.len() - 1)) {
        indices.push(poses.len() - 1);
    }
    Json::Array(
        indices
            .into_iter()
            .map(|i| {
                let pose = &poses[i].pose;
                let t = pose.translation.vector;
                let (_, _, yaw) = pose.rotation.euler_angles();
                Json::Array(vec![
                    poses[i].time.into(),
                    t.x.into(),
                    t.y.into(),
                    t.z.into(),
                    yaw.into(),
                ])
            })
            .collect(),
    )
}

/// Applies `world_from_estimate` to estimated poses.
pub fn transform_poses(
    world_from_estimate: &Isometry3<f64>,
    poses: &[TimedPose],
) -> Vec<TimedPose> {
    poses
        .iter()
        .map(|p| TimedPose::new(p.time, world_from_estimate * p.pose))
        .collect()
}

/// One pass/fail criterion.
pub struct Check {
    pub name: String,
    pub value: f64,
    /// `true`: `value < limit` required; `false`: `value > limit`.
    pub upper_bound: bool,
    pub limit: f64,
}

impl Check {
    pub fn passed(&self) -> bool {
        if self.upper_bound {
            self.value < self.limit
        } else {
            self.value > self.limit
        }
    }
}

/// Collects the metrics of one scenario, prints the table, writes the JSON
/// report and finally asserts every check.
pub struct Report {
    pub scenario: String,
    pub checks: Vec<Check>,
    /// Informational metrics (not checked).
    pub info: Vec<(String, f64)>,
    pub sections: Vec<(String, Json)>,
    start: std::time::Instant,
}

impl Report {
    pub fn new(scenario: &str) -> Self {
        Report {
            scenario: scenario.to_string(),
            checks: Vec::new(),
            info: Vec::new(),
            sections: Vec::new(),
            start: std::time::Instant::now(),
        }
    }

    /// Requires `value < limit`.
    pub fn at_most(&mut self, name: &str, value: f64, limit: f64) {
        self.checks.push(Check {
            name: name.to_string(),
            value,
            upper_bound: true,
            limit,
        });
    }

    /// Requires `value > limit`.
    pub fn at_least(&mut self, name: &str, value: f64, limit: f64) {
        self.checks.push(Check {
            name: name.to_string(),
            value,
            upper_bound: false,
            limit,
        });
    }

    pub fn info(&mut self, name: &str, value: f64) {
        self.info.push((name.to_string(), value));
    }

    pub fn section(&mut self, name: &str, value: Json) {
        self.sections.push((name.to_string(), value));
    }

    /// `target/e2e` (honours `CARGO_TARGET_DIR`).
    pub fn output_directory() -> PathBuf {
        let directory = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .parent()
            .expect("target directory")
            .join("e2e");
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    pub fn map_png_path(&self) -> PathBuf {
        Self::output_directory().join(format!("{}_map.png", self.scenario))
    }

    /// Prints the metrics table, writes `target/e2e/<scenario>.json` and
    /// asserts all checks.
    pub fn finish(self) {
        let seconds = self.start.elapsed().as_secs_f64();
        let mut table = format!(
            "\n=== e2e {} ({seconds:.1} s) ===\n{:<44} {:>12}  {:>14}  {}\n",
            self.scenario, "metric", "value", "threshold", "ok"
        );
        for check in &self.checks {
            writeln!(
                table,
                "{:<44} {:>12.5}  {:>2} {:>11.5}  {}",
                check.name,
                check.value,
                if check.upper_bound { "<" } else { ">" },
                check.limit,
                if check.passed() { "PASS" } else { "FAIL" }
            )
            .unwrap();
        }
        for (name, value) in &self.info {
            writeln!(table, "{name:<44} {value:>12.5}  {:>14}  -", "(info)").unwrap();
        }
        println!("{table}");

        let mut entries = vec![
            ("scenario".to_string(), Json::from(self.scenario.as_str())),
            ("seconds".to_string(), seconds.into()),
            (
                "passed".to_string(),
                self.checks.iter().all(Check::passed).into(),
            ),
            (
                "checks".to_string(),
                Json::Array(
                    self.checks
                        .iter()
                        .map(|check| {
                            Json::object([
                                ("name", Json::from(check.name.as_str())),
                                ("value", check.value.into()),
                                (
                                    "comparison",
                                    if check.upper_bound { "<" } else { ">" }.into(),
                                ),
                                ("limit", check.limit.into()),
                                ("passed", check.passed().into()),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "info".to_string(),
                Json::Object(
                    self.info
                        .iter()
                        .map(|(name, value)| (name.clone(), Json::from(*value)))
                        .collect(),
                ),
            ),
        ];
        entries.extend(self.sections);
        let path = Self::output_directory().join(format!("{}.json", self.scenario));
        std::fs::write(&path, Json::Object(entries).to_string_compact()).unwrap();
        println!("wrote {}", path.display());

        let failed: Vec<String> = self
            .checks
            .iter()
            .filter(|check| !check.passed())
            .map(|check| {
                format!(
                    "{} = {} (required {} {})",
                    check.name,
                    check.value,
                    if check.upper_bound { "<" } else { ">" },
                    check.limit
                )
            })
            .collect();
        assert!(
            failed.is_empty(),
            "{}: failed checks: {failed:#?}",
            self.scenario
        );
    }
}
