//! 3D SLAM on a ROS1 bag of a (simulated) Unitree 4D LiDAR L1, evaluated
//! against the ground truth recorded in the bag.
//!
//! ```text
//! cargo run --release -p cartographer-demos --example unitree_l1_bag -- \
//!     <BAG> [--out DIR] [--lua FILE]
//! ```
//!
//! The bag holds what the official `unitree_lidar_ros` driver publishes
//! (`/unilidar/cloud` PointCloud2 at 10 Hz, stamped with the time of the
//! first point and a per-point `time` field relative to it; `/unilidar/imu`
//! at 250 Hz), plus `/tf_static` (`unilidar_imu` → `unilidar_lidar`) and
//! `/ground_truth/pose` (pose of `unilidar_imu` in `map`). See
//! `crates/demos/README.md` for the data, the configuration choices and the
//! results. Outputs (default `target/unitree_l1/`): `map.ply`,
//! `map.pbstream`, `trajectory_tum.txt`, `local_trajectory_tum.txt`,
//! `ground_truth_tum.txt`, `metrics.json` and PNG plots.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cartographer::mapping::id::NodeId;
use cartographer::mapping::pose_graph_interface::Tag;
use cartographer::slam::points_relative_to_first;
use cartographer::{
    Extrinsics, ImuSample, MapPoint, Mode, Slam, SlamBuilder, SlamConfig, SlamStats, Timestamp,
};
use cartographer_demos::canvas::{Canvas, Color, View, colors};
use cartographer_rosbag::msgs::{Imu, PointCloud2, PoseStamped, TFMessage};
use cartographer_rosbag::{Bag, Time};
use cartographer_sim::metrics::{
    Alignment, Association, AteResult, PosePair, RpeDelta, RpeResult, TimedPose, associate, ate,
    rpe,
};
use clap::Parser;
use nalgebra::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};

const CLOUD_TOPIC: &str = "/unilidar/cloud";
const IMU_TOPIC: &str = "/unilidar/imu";
const TF_STATIC_TOPIC: &str = "/tf_static";
const GROUND_TRUTH_TOPIC: &str = "/ground_truth/pose";
/// Tracking frame: in 3D Cartographer wants the IMU frame.
const TRACKING_FRAME: &str = "unilidar_imu";

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser, Debug)]
#[command(about = "Cartographer 3D SLAM on a Unitree L1 ROS1 bag, evaluated against ground truth")]
struct Args {
    /// ROS1 bag with /unilidar/cloud, /unilidar/imu, /tf_static and
    /// (optionally) /ground_truth/pose.
    bag: PathBuf,
    /// Output directory.
    #[arg(long, default_value = "target/unitree_l1")]
    out: PathBuf,
    /// cartographer_ros-style Lua file (returns `{ map_builder = ...,
    /// trajectory_builder = ... }`) replacing the built-in L1 configuration.
    #[arg(long)]
    lua: Option<PathBuf>,
}

/// Configuration for a Unitree L1 on a small robot indoors: Cartographer's
/// shipped 3D defaults (`trajectory_builder_3d.lua`, `pose_graph.lua`) with
/// three overrides. The choices were validated on the noisy and noise-free
/// simulated bags with the voxel filter varied (0.10 / 0.15 / 0.20 m) as a
/// robustness check; see `crates/demos/README.md` for the experiments,
/// including the knobs that were tried and deliberately left at default.
fn unitree_l1_config() -> SlamConfig {
    let mut c = SlamConfig::new(Mode::ThreeD);
    // 1. Online correlative scan matching (the cartographer_ros tuning guide's
    //    advice when the motion prediction is poor). Without odometry the 3D
    //    extrapolator predicts with constant velocity (the IMU only supplies
    //    rotation), so it lags when the robot accelerates from rest. The
    //    L1's sparse (2,160 points/cloud), ceiling-dominated (~70 %) clouds
    //    give the Ceres matcher a small convergence basin: without this, local
    //    SLAM sometimes "sticks" while the robot starts moving (meters of
    //    error that loop closure cannot repair).
    c.use_online_correlative_scan_matching = true;
    c.advanced(|_, trajectory_builder| {
        let correlative = trajectory_builder
            .trajectory_builder_3d_options
            .as_mut()
            .and_then(|tb| tb.real_time_correlative_scan_matcher_options.as_mut())
            .expect("the shipped 3D options have a real-time correlative matcher");
        // 2. Search +-1 angular step (~0.5 deg here) instead of the default
        //    +-1 deg (2 steps). The L1's arcs precess between clouds, so a
        //    sparse map (the first submap, while the robot stands still) is
        //    a "spoke" pattern; the correlative matcher, which counts
        //    overlaps, then rotates each scan towards the previous spokes:
        //    with +-1 deg the yaw ran away by ~10 deg during the first 2 s;
        //    with +-0.5 deg it stays within ~1.5 deg. Gyro bias needs far
        //    less (~0.2 deg/s here).
        correlative.angular_search_window = 0.5_f64.to_radians();
    });
    // 3. Submaps of 120 scans (12 s, ~13 m at this speed) instead of 160.
    //    Loop closure only matches against finished submaps (2 x
    //    num_range_data insertions): with 120 the first one is finished
    //    after 24 s, before the end of the first ~30 s lap, and there are
    //    more, smaller submaps to close against. Modest gain (optimized ATE
    //    ~1.7-2.9 cm instead of ~2.6-3.6 cm averaged over the robustness
    //    runs); 160 also works.
    c.submap_num_range_data = 120;
    c
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out)?;

    // ------------------------------------------------------------------
    // Pass 1: static transforms, ground truth, cloud frame id.
    // ------------------------------------------------------------------
    let mut bag = Bag::open(&args.bag)?;
    let (bag_start, bag_end) = (
        bag.start_time().ok_or("empty bag")?,
        bag.end_time().ok_or("empty bag")?,
    );
    let bag_duration = bag_end.as_secs_f64() - bag_start.as_secs_f64();
    println!(
        "Bag {}: {} messages, {:.1} s",
        args.bag.display(),
        bag.message_count(),
        bag_duration
    );
    for topic in bag.topics() {
        println!(
            "  {:<22} {:>6} x {}",
            topic.topic, topic.message_count, topic.msg_type
        );
    }
    let has_topic = |bag: &Bag, name: &str| bag.topics().iter().any(|t| t.topic == name);
    let mut static_tf = Vec::new();
    let mut ground_truth = Vec::new();
    let mut prepass_topics = vec![TF_STATIC_TOPIC];
    if has_topic(&bag, GROUND_TRUTH_TOPIC) {
        prepass_topics.push(GROUND_TRUTH_TOPIC);
    }
    for message in bag.messages_for_topics(&prepass_topics)? {
        let message = message?;
        if message.topic() == TF_STATIC_TOPIC {
            let tf: TFMessage = message.decode()?;
            static_tf.extend(tf.transforms.into_iter().map(|t| {
                (
                    t.header.frame_id,
                    t.child_frame_id,
                    isometry(t.transform.translation.to_array(), &t.transform.rotation),
                )
            }));
        } else {
            let pose: PoseStamped = message.decode()?;
            ground_truth.push(TimedPose::new(
                pose.header.stamp.as_secs_f64(),
                isometry(pose.pose.position.to_array(), &pose.pose.orientation),
            ));
        }
    }
    let first_cloud: PointCloud2 = bag
        .messages_for_topics(&[CLOUD_TOPIC])?
        .next()
        .ok_or("no point cloud in the bag")??
        .decode()?;
    let lidar_frame = first_cloud.header.frame_id.clone();
    let first_imu: Imu = bag
        .messages_for_topics(&[IMU_TOPIC])?
        .next()
        .ok_or("no IMU message in the bag")??
        .decode()?;
    let imu_frame = first_imu.header.frame_id.clone();
    let tracking_from_lidar = lookup_static(&static_tf, TRACKING_FRAME, &lidar_frame)?;
    let tracking_from_imu = lookup_static(&static_tf, TRACKING_FRAME, &imu_frame)?;
    println!(
        "tracking_frame = {TRACKING_FRAME}; {lidar_frame} at t = {:?} m, rpy = {:?} rad",
        tracking_from_lidar.translation.vector.as_slice(),
        tracking_from_lidar.rotation.euler_angles()
    );
    println!("{} ground-truth poses", ground_truth.len());

    // ------------------------------------------------------------------
    // SLAM
    // ------------------------------------------------------------------
    let config = match &args.lua {
        Some(path) => {
            let code = std::fs::read_to_string(path)?;
            let dir = path
                .parent()
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default();
            SlamConfig::from_lua(&code, &[dir])?
        }
        None => unitree_l1_config(),
    };
    if config.mode() != Mode::ThreeD {
        return Err("the configuration must use the 3D trajectory builder".into());
    }
    print_config(&config);
    let mut slam = SlamBuilder::with_config(config.clone())
        .lidar(CLOUD_TOPIC, Extrinsics::from_isometry(tracking_from_lidar))
        .imu(IMU_TOPIC, Extrinsics::from_isometry(tracking_from_imu))
        .build()?;

    let wall_start = Instant::now();
    let mut next_report = bag_start.as_secs_f64() + 10.0;
    let (mut num_clouds, mut num_imu, mut num_points) = (0usize, 0usize, 0usize);
    for message in bag.messages_for_topics(&[IMU_TOPIC, CLOUD_TOPIC])? {
        let message = message?;
        if message.topic() == IMU_TOPIC {
            let imu: Imu = message.decode()?;
            slam.push_imu(
                stamp(imu.header.stamp),
                ImuSample {
                    linear_acceleration: Vector3::from(imu.linear_acceleration.to_array()),
                    angular_velocity: Vector3::from(imu.angular_velocity.to_array()),
                },
            )?;
            num_imu += 1;
        } else {
            let cloud: PointCloud2 = message.decode()?;
            let points = cloud.read_points()?;
            if num_clouds == 0 && points.first().is_some_and(|p| p.time.is_none()) {
                println!("warning: the clouds have no `time` field: no motion compensation");
            }
            // Driver convention: header.stamp = first point, `time` = seconds
            // since the stamp. Cartographer wants the stamp of the last point
            // and times <= 0 relative to it.
            let (time, points) = points_relative_to_first(
                stamp(cloud.header.stamp),
                points
                    .iter()
                    .filter(|p| p.position.iter().all(|v| v.is_finite()))
                    .map(|p| (Vector3::from(p.position), p.time.unwrap_or(0.0) as f32)),
            );
            num_points += points.len();
            slam.push_lidar(CLOUD_TOPIC, time, &points)?;
            num_clouds += 1;
        }
        let t = message.time.as_secs_f64();
        if t >= next_report {
            next_report += 10.0;
            let elapsed = wall_start.elapsed().as_secs_f64();
            let stats = slam.stats();
            println!(
                "  t = {:5.1} s: {} nodes, {} submaps, {} loop closures; {:.1} s wall, {:.1}x real time",
                t - bag_start.as_secs_f64(),
                stats.num_nodes,
                stats.num_submaps,
                stats.num_loop_closure_constraints,
                elapsed,
                (t - bag_start.as_secs_f64()) / elapsed
            );
        }
    }
    let feed_seconds = wall_start.elapsed().as_secs_f64();
    let finish_start = Instant::now();
    slam.finish()?;
    let finish_seconds = finish_start.elapsed().as_secs_f64();
    let total_seconds = wall_start.elapsed().as_secs_f64();
    println!(
        "Fed {num_clouds} clouds ({num_points} points) and {num_imu} IMU samples in {feed_seconds:.1} s; \
         finish() took {finish_seconds:.1} s; total {total_seconds:.1} s = {:.1}x real time",
        bag_duration / total_seconds
    );
    let stats = slam.stats();
    print_stats(&stats);

    // ------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------
    let trajectory = slam.trajectory();
    let local_trajectory = slam.local_trajectory();
    let to_timed = |poses: &[cartographer::StampedPose]| -> Vec<TimedPose> {
        poses
            .iter()
            .map(|p| TimedPose::new(p.time.as_secs_f64(), p.pose))
            .collect()
    };
    let estimate = to_timed(&trajectory);
    let local_estimate = to_timed(&local_trajectory);
    write_tum(&args.out.join("trajectory_tum.txt"), &estimate)?;
    write_tum(&args.out.join("local_trajectory_tum.txt"), &local_estimate)?;
    if !ground_truth.is_empty() {
        write_tum(&args.out.join("ground_truth_tum.txt"), &ground_truth)?;
    }
    let map = slam.map_3d()?;
    map.to_ply(args.out.join("map.ply"))?;
    slam.save(args.out.join("map.pbstream"))?;
    println!(
        "Map: {} occupied voxels at {:.2} m",
        map.points.len(),
        map.resolution
    );

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------
    let t0 = bag_start.as_secs_f64();
    let mut json = String::new();
    let evaluation = if ground_truth.is_empty() {
        println!("No ground truth in the bag: skipping the evaluation.");
        None
    } else {
        let optimized = evaluate(&ground_truth, &estimate)?;
        let local = evaluate(&ground_truth, &local_estimate)?;
        println!(
            "Ground-truth path length: {:.1} m",
            path_length(&ground_truth)
        );
        println!(
            "{:<16} {:>9} {:>9} {:>9} {:>11} {:>11} {:>12}",
            "trajectory",
            "ATE rmse",
            "ATE max",
            "rot rmse",
            "RPE1m rmse",
            "RPE1m rot",
            "final drift"
        );
        for (name, e) in [("local SLAM", &local), ("optimized", &optimized)] {
            println!(
                "{:<16} {:>7.3} m {:>7.3} m {:>7.2}° {:>9.3} m {:>9.2}° {:>7.3} m ({:.2} %)",
                name,
                e.ate.translation.rmse,
                e.ate.translation.max,
                e.ate.rotation.rmse.to_degrees(),
                e.rpe.translation.rmse,
                e.rpe.rotation.rmse.to_degrees(),
                e.final_drift,
                100.0 * e.final_drift / e.path_length.max(1e-9)
            );
        }
        Some((optimized, local))
    };

    write_json(
        &mut json,
        &args,
        bag_duration,
        (feed_seconds, finish_seconds, total_seconds),
        &stats,
        &config,
        evaluation.as_ref(),
    );
    std::fs::write(args.out.join("metrics.json"), &json)?;

    // ------------------------------------------------------------------
    // Plots, in the ground-truth frame when available.
    // ------------------------------------------------------------------
    let (world_from_map, world_from_local) = match &evaluation {
        Some((optimized, local)) => (
            optimized.ate.alignment.isometry,
            local.ate.alignment.isometry,
        ),
        None => (Isometry3::identity(), Isometry3::identity()),
    };
    let map_points: Vec<Vector3<f64>> = map
        .points
        .iter()
        .map(|p: &MapPoint| (world_from_map * Point3::from(p.position.cast::<f64>())).coords)
        .collect();
    let positions = |poses: &[TimedPose], t: &Isometry3<f64>| -> Vec<Vector3<f64>> {
        poses
            .iter()
            .map(|p| (t * p.pose).translation.vector)
            .collect()
    };
    let tracks = Tracks {
        ground_truth: positions(&ground_truth, &Isometry3::identity()),
        optimized: positions(&estimate, &world_from_map),
        local: positions(&local_estimate, &world_from_local),
    };
    let constraints = constraint_segments(&slam, &world_from_map);
    let plots = render_plots(
        &args.out,
        &map_points,
        map.resolution,
        &tracks,
        &constraints,
        evaluation.as_ref(),
        t0,
    )?;
    println!("Wrote to {}:", args.out.display());
    for name in [
        "map.ply",
        "map.pbstream",
        "trajectory_tum.txt",
        "local_trajectory_tum.txt",
        "ground_truth_tum.txt",
        "metrics.json",
    ]
    .into_iter()
    .chain(plots.iter().map(String::as_str))
    {
        println!("  {}", args.out.join(name).display());
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Bag helpers
// ----------------------------------------------------------------------

fn stamp(time: Time) -> Timestamp {
    Timestamp::from_nanos(time.as_nanos() as i64)
}

fn isometry(t: [f64; 3], q: &cartographer_rosbag::msgs::Quaternion) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z)),
    )
}

/// `target_from_source` from the static transforms (`parent_from_child`
/// each), following at most one edge in either direction.
fn lookup_static(
    transforms: &[(String, String, Isometry3<f64>)],
    target: &str,
    source: &str,
) -> Result<Isometry3<f64>> {
    if target == source {
        return Ok(Isometry3::identity());
    }
    for (parent, child, parent_from_child) in transforms {
        if parent == target && child == source {
            return Ok(*parent_from_child);
        }
        if parent == source && child == target {
            return Ok(parent_from_child.inverse());
        }
    }
    Err(format!("no static transform between {target} and {source} in {TF_STATIC_TOPIC}").into())
}

/// Writes poses in TUM format (`time x y z qx qy qz qw`).
fn write_tum(path: &Path, poses: &[TimedPose]) -> Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for p in poses {
        let t = p.pose.translation.vector;
        let q = p.pose.rotation.coords;
        writeln!(
            out,
            "{:.9} {:.6} {:.6} {:.6} {:.9} {:.9} {:.9} {:.9}",
            p.time, t.x, t.y, t.z, q.x, q.y, q.z, q.w
        )?;
    }
    out.flush()?;
    Ok(())
}

// ----------------------------------------------------------------------
// Evaluation
// ----------------------------------------------------------------------

struct Evaluation {
    pairs: Vec<PosePair>,
    ate: AteResult,
    rpe: RpeResult,
    /// Position error at the last pose with the trajectories aligned at
    /// their first pose (no global fit): the accumulated drift.
    final_drift: f64,
    /// Ground-truth path length over the evaluated interval (m).
    path_length: f64,
}

fn evaluate(ground_truth: &[TimedPose], estimate: &[TimedPose]) -> Result<Evaluation> {
    // Ground truth is at 100 Hz: interpolate it at each node time.
    let pairs = associate(
        ground_truth,
        estimate,
        Association::Interpolate { max_gap: 0.05 },
    );
    if pairs.len() < 3 {
        return Err("too few poses could be associated with the ground truth".into());
    }
    let ate = ate(&pairs, Alignment::Rigid3d);
    let rpe = rpe(&pairs, RpeDelta::Distance(1.0));
    let (first, last) = (&pairs[0], &pairs[pairs.len() - 1]);
    let reference_from_estimate = first.reference * first.estimate.inverse();
    let final_drift = ((reference_from_estimate * last.estimate).translation.vector
        - last.reference.translation.vector)
        .norm();
    let path_length = pairs
        .windows(2)
        .map(|w| (w[1].reference.translation.vector - w[0].reference.translation.vector).norm())
        .sum();
    Ok(Evaluation {
        pairs,
        ate,
        rpe,
        final_drift,
        path_length,
    })
}

fn path_length(poses: &[TimedPose]) -> f64 {
    poses
        .windows(2)
        .map(|w| (w[1].pose.translation.vector - w[0].pose.translation.vector).norm())
        .sum()
}

// ----------------------------------------------------------------------
// Reporting
// ----------------------------------------------------------------------

/// Angular search window (deg) of the online correlative scan matcher, if
/// enabled.
fn correlative_window_deg(c: &SlamConfig) -> Option<f64> {
    if !c.use_online_correlative_scan_matching {
        return None;
    }
    let (_, trajectory_builder) = c.to_protos().ok()?;
    let window = trajectory_builder
        .trajectory_builder_3d_options?
        .real_time_correlative_scan_matcher_options?
        .angular_search_window;
    Some(window.to_degrees())
}

fn print_config(c: &SlamConfig) {
    println!(
        "Config: range [{}, {}] m, voxel {} m, accumulate {}, submap {} scans, \
         grids {}/{} m, online correlative matching {}, scan matcher weights {}/{}, \
         gravity tc {} s, loop closure every {} nodes (min score {}, sampling {})",
        c.min_range,
        c.max_range,
        c.voxel_filter_size,
        c.num_accumulated_range_data,
        c.submap_num_range_data,
        c.resolution,
        c.low_resolution,
        correlative_window_deg(c).map_or("off".to_string(), |w| format!("+-{w:.2} deg")),
        c.scan_matcher.translation_weight,
        c.scan_matcher.rotation_weight,
        c.imu_gravity_time_constant,
        c.loop_closure.optimize_every_n_nodes,
        c.loop_closure.min_score,
        c.loop_closure.sampling_ratio
    );
}

fn print_stats(s: &SlamStats) {
    println!(
        "{} nodes, {} submaps, {} global optimizations; constraints: {} total = {} intra-submap \
         + {} inter-submap ({} close loops, {} inter-trajectory)",
        s.num_nodes,
        s.num_submaps,
        s.num_global_optimizations,
        s.num_constraints,
        s.num_intra_submap_constraints,
        s.num_inter_submap_constraints,
        s.num_loop_closure_constraints,
        s.num_inter_trajectory_constraints
    );
}

fn json_metrics(out: &mut String, name: &str, e: &Evaluation, last: bool) {
    let _ = write!(
        out,
        "    \"{name}\": {{\n      \"num_poses\": {},\n      \"ate_rmse_m\": {:.5},\n      \
         \"ate_mean_m\": {:.5},\n      \"ate_median_m\": {:.5},\n      \"ate_max_m\": {:.5},\n      \
         \"ate_rotation_rmse_deg\": {:.4},\n      \"rpe_1m_translation_rmse_m\": {:.5},\n      \
         \"rpe_1m_rotation_rmse_deg\": {:.4},\n      \"final_drift_m\": {:.5},\n      \
         \"final_drift_percent\": {:.4}\n    }}{}\n",
        e.pairs.len(),
        e.ate.translation.rmse,
        e.ate.translation.mean,
        e.ate.translation.median,
        e.ate.translation.max,
        e.ate.rotation.rmse.to_degrees(),
        e.rpe.translation.rmse,
        e.rpe.rotation.rmse.to_degrees(),
        e.final_drift,
        100.0 * e.final_drift / e.path_length.max(1e-9),
        if last { "" } else { "," }
    );
}

fn write_json(
    out: &mut String,
    args: &Args,
    bag_duration: f64,
    (feed, finish, total): (f64, f64, f64),
    s: &SlamStats,
    c: &SlamConfig,
    evaluation: Option<&(Evaluation, Evaluation)>,
) {
    let escape = |p: &Path| {
        p.display()
            .to_string()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    };
    let _ = write!(
        out,
        "{{\n  \"bag\": \"{}\",\n  \"lua\": {},\n  \"bag_duration_s\": {bag_duration:.3},\n  \
         \"timing\": {{ \"feed_s\": {feed:.3}, \"finish_s\": {finish:.3}, \"total_s\": {total:.3}, \
         \"realtime_factor\": {:.3} }},\n",
        escape(&args.bag),
        args.lua
            .as_ref()
            .map_or("null".to_string(), |p| format!("\"{}\"", escape(p))),
        bag_duration / total
    );
    let _ = writeln!(
        out,
        "  \"stats\": {{ \"nodes\": {}, \"submaps\": {}, \"global_optimizations\": {}, \
         \"constraints\": {}, \"intra_submap\": {}, \"inter_submap\": {}, \"loop_closures\": {}, \
         \"inter_trajectory\": {} }},",
        s.num_nodes,
        s.num_submaps,
        s.num_global_optimizations,
        s.num_constraints,
        s.num_intra_submap_constraints,
        s.num_inter_submap_constraints,
        s.num_loop_closure_constraints,
        s.num_inter_trajectory_constraints
    );
    let _ = write!(
        out,
        "  \"config\": {{ \"min_range\": {}, \"max_range\": {}, \"voxel_filter_size\": {}, \
         \"num_accumulated_range_data\": {}, \"submap_num_range_data\": {}, \
         \"high_resolution\": {}, \"low_resolution\": {}, \
         \"online_correlative_angular_window_deg\": {}, \"ceres_translation_weight\": {}, \
         \"ceres_rotation_weight\": {}, \"imu_gravity_time_constant\": {}, \
         \"optimize_every_n_nodes\": {}, \"loop_closure_min_score\": {}, \
         \"loop_closure_sampling_ratio\": {} }}",
        c.min_range,
        c.max_range,
        c.voxel_filter_size,
        c.num_accumulated_range_data,
        c.submap_num_range_data,
        c.resolution,
        c.low_resolution,
        correlative_window_deg(c).map_or("null".to_string(), |w| format!("{w:.3}")),
        c.scan_matcher.translation_weight,
        c.scan_matcher.rotation_weight,
        c.imu_gravity_time_constant,
        c.loop_closure.optimize_every_n_nodes,
        c.loop_closure.min_score,
        c.loop_closure.sampling_ratio
    );
    match evaluation {
        Some((optimized, local)) => {
            out.push_str(",\n  \"evaluation\": {\n");
            let _ = writeln!(
                out,
                "    \"association\": \"ground truth interpolated at node times\",\n    \
                 \"alignment\": \"SE(3) Umeyama on positions\",\n    \
                 \"path_length_m\": {:.3},",
                optimized.path_length
            );
            json_metrics(out, "optimized", optimized, false);
            json_metrics(out, "local", local, true);
            out.push_str("  }\n}\n");
        }
        None => out.push_str("\n}\n"),
    }
}

// ----------------------------------------------------------------------
// Plots
// ----------------------------------------------------------------------

struct Tracks {
    ground_truth: Vec<Vector3<f64>>,
    optimized: Vec<Vector3<f64>>,
    local: Vec<Vector3<f64>>,
}

/// An inter-submap constraint in the plot frame.
struct ConstraintSegment {
    node: Vector3<f64>,
    submap_origin: Vector3<f64>,
    submap_index: i32,
    closes_loop: bool,
}

/// The inter-submap constraints of the pose graph.
fn constraint_segments(slam: &Slam, world_from_map: &Isometry3<f64>) -> Vec<ConstraintSegment> {
    let pose_graph = slam.pose_graph();
    let constraints = pose_graph.constraints();
    let nodes = pose_graph.get_trajectory_node_poses();
    let submaps = pose_graph.get_all_submap_poses();
    // Same classification as `Slam::stats`: a constraint closes a loop if
    // its submap is older than the one preceding the node's first insertion
    // submap.
    let mut first_insertion: HashMap<NodeId, i32> = HashMap::new();
    for c in constraints.iter().filter(|c| c.tag == Tag::IntraSubmap) {
        let index = first_insertion.entry(c.node_id).or_insert(i32::MAX);
        *index = (*index).min(c.submap_id.submap_index);
    }
    let to_world = |v: &Vector3<f64>| (world_from_map * Point3::from(*v)).coords;
    constraints
        .iter()
        .filter(|c| c.tag == Tag::InterSubmap)
        .filter_map(|c| {
            let node = nodes.get(&c.node_id)?;
            let submap = submaps.get(&c.submap_id)?;
            Some(ConstraintSegment {
                node: to_world(node.global_pose.translation()),
                submap_origin: to_world(submap.pose.translation()),
                submap_index: c.submap_id.submap_index,
                closes_loop: first_insertion
                    .get(&c.node_id)
                    .is_some_and(|&first| c.submap_id.submap_index + 1 < first),
            })
        })
        .collect()
}

/// Height of the ceiling: the most populated 10 cm z slice of the map (the
/// L1 sees mostly ceiling; it never sees the floor when mounted upright).
fn ceiling_height(points: &[Vector3<f64>]) -> Option<f64> {
    let mut histogram: HashMap<i64, usize> = HashMap::new();
    for p in points {
        *histogram.entry((p.z / 0.1).floor() as i64).or_default() += 1;
    }
    histogram
        .into_iter()
        .max_by_key(|&(bin, count)| (count, bin))
        .map(|(bin, _)| (bin as f64 + 0.5) * 0.1)
}

/// Bounding box (min, max) of all points, padded.
fn bounds<'a>(
    points: impl Iterator<Item = &'a Vector3<f64>>,
    pad: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let mut min = Vector3::repeat(f64::INFINITY);
    let mut max = Vector3::repeat(f64::NEG_INFINITY);
    for p in points {
        min = min.inf(p);
        max = max.sup(p);
    }
    if !min.x.is_finite() {
        return (Vector3::repeat(-1.0), Vector3::repeat(1.0));
    }
    (min.add_scalar(-pad), max.add_scalar(pad))
}

/// Draws a density projection of `points` onto axes (`u`, `v`) (indices
/// into x, y, z) with cells of `resolution`.
fn draw_density(
    canvas: &mut Canvas,
    view: &View,
    points: &[Vector3<f64>],
    (u, v): (usize, usize),
    resolution: f64,
    saturation: f64,
) {
    let mut cells: HashMap<(i64, i64), usize> = HashMap::new();
    for p in points {
        let key = (
            (p[u] / resolution).floor() as i64,
            (p[v] / resolution).floor() as i64,
        );
        *cells.entry(key).or_default() += 1;
    }
    for ((i, j), count) in cells {
        let (x0, y1) = view.px(i as f64 * resolution, j as f64 * resolution);
        let (x1, y0) = view.px((i + 1) as f64 * resolution, (j + 1) as f64 * resolution);
        let level = (count as f64 / saturation).min(1.0).sqrt();
        let gray = (215.0 - 185.0 * level) as u8;
        canvas.fill_rect(
            x0.round() as i64,
            y0.round() as i64,
            x1.round() as i64,
            y1.round() as i64,
            [gray, gray, gray],
        );
    }
}

fn project(points: &[Vector3<f64>], view: &View, (u, v): (usize, usize)) -> Vec<(f64, f64)> {
    points.iter().map(|p| view.px(p[u], p[v])).collect()
}

/// A legend box at the top-left corner: (color, label) entries.
fn legend(canvas: &mut Canvas, x: i64, y: i64, entries: &[(Color, &str)]) {
    let width = entries
        .iter()
        .map(|(_, label)| Canvas::text_width(label, 2))
        .max()
        .unwrap_or(0)
        + 40;
    canvas.fill_rect(
        x,
        y,
        x + width,
        y + 18 * entries.len() as i64 + 6,
        colors::WHITE,
    );
    for (k, (color, label)) in entries.iter().enumerate() {
        let row = y + 6 + 18 * k as i64;
        canvas.fill_rect(x + 6, row + 3, x + 26, row + 7, *color);
        canvas.text(x + 32, row, label, 2, colors::BLACK);
    }
}

fn title(canvas: &mut Canvas, text: &str) {
    canvas.text(12, 10, text, 3, colors::BLACK);
}

fn render_plots(
    out: &Path,
    map_points: &[Vector3<f64>],
    resolution: f64,
    tracks: &Tracks,
    constraints: &[ConstraintSegment],
    evaluation: Option<&(Evaluation, Evaluation)>,
    t0: f64,
) -> Result<Vec<String>> {
    let mut written = Vec::new();
    const SCALE: f64 = 50.0; // pixels per meter
    const MARGIN: f64 = 70.0;
    let (min, max) = bounds(
        map_points
            .iter()
            .chain(&tracks.ground_truth)
            .chain(&tracks.optimized),
        0.5,
    );
    let ceiling = ceiling_height(map_points);
    // Remove the ceiling (and anything within 30 cm of it) from the top-down
    // view, otherwise it hides everything below.
    let cut = ceiling.map_or(f64::INFINITY, |z| z - 0.3);
    let below_ceiling: Vec<Vector3<f64>> =
        map_points.iter().filter(|p| p.z < cut).copied().collect();
    let trajectory_entries = [
        (colors::GREEN, "GROUND TRUTH"),
        (colors::RED, "SLAM (OPTIMIZED)"),
        (colors::BLUE, "LOCAL SLAM ONLY"),
    ];

    // (a) top-down
    let view = View::new(min.x, max.x, min.y, max.y, SCALE, MARGIN);
    let mut canvas = view.canvas(colors::WHITE);
    view.grid(&mut canvas, 1.0);
    draw_density(&mut canvas, &view, &below_ceiling, (0, 1), resolution, 12.0);
    canvas.polyline(&project(&tracks.local, &view, (0, 1)), 1, colors::BLUE);
    canvas.polyline(
        &project(&tracks.ground_truth, &view, (0, 1)),
        3,
        colors::GREEN,
    );
    canvas.polyline(&project(&tracks.optimized, &view, (0, 1)), 1, colors::RED);
    view.frame(&mut canvas);
    title(
        &mut canvas,
        &format!(
            "TOP-DOWN (X-Y, M): VOXEL DENSITY BELOW Z = {:.2} M (CEILING REMOVED)",
            cut.min(max.z)
        ),
    );
    legend(
        &mut canvas,
        MARGIN as i64 + 8,
        MARGIN as i64 + 8,
        &trajectory_entries,
    );
    canvas.save_png(out.join("top_down.png"))?;
    written.push("top_down.png".to_string());

    // (d) constraints on the top-down view
    let mut canvas = view.canvas(colors::WHITE);
    view.grid(&mut canvas, 1.0);
    draw_density(&mut canvas, &view, &below_ceiling, (0, 1), resolution, 40.0);
    // Only the loop-closing constraints are drawn (the others link a node to
    // a neighboring submap and would just cover the trajectory).
    const LIGHT_MAGENTA: Color = [235, 170, 235];
    let loops: Vec<&ConstraintSegment> = constraints.iter().filter(|c| c.closes_loop).collect();
    for c in &loops {
        canvas.line(
            view.px(c.node.x, c.node.y),
            view.px(c.submap_origin.x, c.submap_origin.y),
            LIGHT_MAGENTA,
        );
    }
    canvas.polyline(&project(&tracks.optimized, &view, (0, 1)), 2, colors::RED);
    for c in &loops {
        canvas.disc(view.px(c.node.x, c.node.y), 2.5, colors::MAGENTA);
    }
    let mut labeled = std::collections::BTreeSet::new();
    for c in constraints {
        if labeled.insert(c.submap_index) {
            let (x, y) = view.px(c.submap_origin.x, c.submap_origin.y);
            canvas.disc((x, y), 5.0, colors::BLACK);
            canvas.text(
                x as i64 + 8,
                y as i64 - 14,
                &format!("S{}", c.submap_index),
                2,
                colors::BLACK,
            );
        }
    }
    view.frame(&mut canvas);
    title(
        &mut canvas,
        &format!(
            "LOOP CLOSURES: {} OF {} INTER-SUBMAP CONSTRAINTS (NODE TO SUBMAP ORIGIN)",
            loops.len(),
            constraints.len()
        ),
    );
    legend(
        &mut canvas,
        MARGIN as i64 + 8,
        MARGIN as i64 + 8,
        &[
            (colors::RED, "SLAM (OPTIMIZED)"),
            (colors::MAGENTA, "NODE WITH LOOP CLOSURE"),
            (LIGHT_MAGENTA, "LOOP CLOSURE CONSTRAINT"),
            (colors::BLACK, "SUBMAP ORIGIN"),
        ],
    );
    canvas.save_png(out.join("loop_closures.png"))?;
    written.push("loop_closures.png".to_string());

    // (b) side views
    for (name, axes, label) in [
        ("side_xz.png", (0, 2), "SIDE VIEW X-Z (M)"),
        ("side_yz.png", (1, 2), "SIDE VIEW Y-Z (M)"),
    ] {
        let view = View::new(min[axes.0], max[axes.0], min.z, max.z, SCALE, MARGIN);
        let mut canvas = view.canvas(colors::WHITE);
        view.grid(&mut canvas, 1.0);
        draw_density(&mut canvas, &view, map_points, axes, resolution, 25.0);
        canvas.polyline(&project(&tracks.local, &view, axes), 1, colors::BLUE);
        canvas.polyline(
            &project(&tracks.ground_truth, &view, axes),
            3,
            colors::GREEN,
        );
        canvas.polyline(&project(&tracks.optimized, &view, axes), 1, colors::RED);
        view.frame(&mut canvas);
        title(
            &mut canvas,
            &format!("{label}: VOXEL DENSITY (ALL HEIGHTS)"),
        );
        legend(
            &mut canvas,
            MARGIN as i64 + 8,
            MARGIN as i64 + 8,
            &trajectory_entries,
        );
        canvas.save_png(out.join(name))?;
        written.push(name.to_string());
    }

    // (c) error vs time
    if let Some((optimized, local)) = evaluation {
        let series = [
            (&local.pairs, &local.ate.translation_errors, colors::BLUE),
            (
                &optimized.pairs,
                &optimized.ate.translation_errors,
                colors::RED,
            ),
        ];
        let t_max = series
            .iter()
            .flat_map(|(pairs, _, _)| pairs.last().map(|p| p.time - t0))
            .fold(1.0, f64::max);
        let e_max = series
            .iter()
            .flat_map(|(_, errors, _)| errors.iter().copied())
            .fold(0.01, f64::max);
        // A "nice" y range.
        let step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
            .into_iter()
            .find(|s| e_max / s <= 8.0)
            .unwrap_or(10.0);
        let y_max = (e_max / step).ceil() * step;
        let (width, height, margin) = (1200.0, 500.0, 80.0);
        let mut canvas = Canvas::new(
            (width + 2.0 * margin) as usize,
            (height + 2.0 * margin) as usize,
            colors::WHITE,
        );
        let px = |t: f64, e: f64| {
            (
                margin + t / t_max * width,
                margin + (1.0 - e / y_max) * height,
            )
        };
        let mut k = 0.0;
        while k <= y_max + 1e-9 {
            let (_, y) = px(0.0, k);
            canvas.line((margin, y), (margin + width, y), colors::GRID);
            let label = format!("{k:.2}");
            canvas.text(
                (margin - 8.0) as i64 - Canvas::text_width(&label, 2),
                y as i64 - 5,
                &label,
                2,
                colors::BLACK,
            );
            k += step;
        }
        let mut t = 0.0;
        while t <= t_max {
            let (x, _) = px(t, 0.0);
            canvas.line((x, margin), (x, margin + height), colors::GRID);
            let label = format!("{t:.0}");
            canvas.text(
                x as i64 - Canvas::text_width(&label, 2) / 2,
                (margin + height + 8.0) as i64,
                &label,
                2,
                colors::BLACK,
            );
            t += 10.0;
        }
        for (pairs, errors, color) in series {
            let points: Vec<(f64, f64)> = pairs
                .iter()
                .zip(errors.iter())
                .map(|(p, &e)| px(p.time - t0, e))
                .collect();
            canvas.polyline(&points, 2, color);
        }
        canvas.polyline(
            &[
                (margin, margin),
                (margin + width, margin),
                (margin + width, margin + height),
                (margin, margin + height),
                (margin, margin),
            ],
            1,
            colors::BLACK,
        );
        title(
            &mut canvas,
            &format!(
                "TRANSLATION ERROR (M) VS TIME (S) AFTER SE(3) ALIGNMENT. ATE RMSE: OPTIMIZED {:.3} M, LOCAL {:.3} M",
                optimized.ate.translation.rmse, local.ate.translation.rmse
            ),
        );
        legend(
            &mut canvas,
            margin as i64 + 8,
            margin as i64 + 8,
            &[
                (colors::RED, "SLAM (OPTIMIZED)"),
                (colors::BLUE, "LOCAL SLAM ONLY"),
            ],
        );
        canvas.save_png(out.join("error_vs_time.png"))?;
        written.push("error_vs_time.png".to_string());
    }
    Ok(written)
}
