//! 2D SLAM with the high-level API on a simulated corridor loop.
//!
//! ```text
//! cargo run --release -p cartographer --example slam_2d -- [OUTPUT_DIR] [LAPS]
//! ```
//!
//! A robot drives `LAPS` (default 1.25) laps around a 40 m x 20 m corridor
//! loop with a 2D LiDAR, an IMU and wheel odometry (simulated by
//! `cartographer-sim`). The example prints statistics and the trajectory
//! error against ground truth, and writes to `OUTPUT_DIR` (default
//! `slam_2d_output`):
//!
//! * `map.png` — the occupancy grid (north up);
//! * `map.pgm` + `map.yaml` — the same map for ROS `map_server`;
//! * `map.pbstream` — the full SLAM state (for localization or the
//!   cartographer_ros tools).

use std::path::PathBuf;
use std::time::Instant;

use cartographer::{Extrinsics, ImuSample, Mode, Slam, SlamError, TimedPoint};
use cartographer_sim::metrics::{Alignment, Association, TimedPose, associate, ate};
use cartographer_sim::{
    DatasetBuilder, ImuConfig, Lidar2dConfig, OdometryConfig, SensorData, Trajectory, World2d,
};

fn main() -> Result<(), SlamError> {
    let mut args = std::env::args().skip(1);
    let output_dir = PathBuf::from(args.next().unwrap_or_else(|| "slam_2d_output".to_string()));
    let laps: f64 = args
        .next()
        .map_or(1.25, |laps| laps.parse().expect("LAPS: a number"));
    std::fs::create_dir_all(&output_dir).map_err(|e| SlamError::Io {
        path: output_dir.clone(),
        source: e,
    })?;

    // Simulated sensor data: times in seconds, data in the sensor frames.
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops(laps, 1.5).with_start_from_rest(3.0),
        World2d::loop_corridor(),
    )
    .seed(1)
    .lidar_2d("scan", Lidar2dConfig::default())
    .imu("imu", ImuConfig::default())
    .odometry("odom", OdometryConfig::default())
    .build();
    let extrinsics = |id: &str| Extrinsics::from_isometry(dataset.sensor(id).unwrap().extrinsic);
    println!(
        "Simulated {:.0} s: {} messages.",
        dataset.end_time - dataset.start_time,
        dataset.messages.len()
    );

    // The SLAM system: sensors with their mounting poses, a few knobs.
    let mut slam = Slam::builder(Mode::TwoD)
        .lidar("scan", extrinsics("scan"))
        .imu("imu", extrinsics("imu"))
        .odometry_with_extrinsics("odom", extrinsics("odom"))
        .config(|c| {
            c.resolution = 0.05;
            c.max_range = 30.0;
            c.loop_closure.optimize_every_n_nodes = 90;
        })
        .build()?;
    slam.on_global_optimization(|event| {
        if let Some(node) = event.last_optimized_nodes.get(&0) {
            println!("  loop closure optimization up to node {}", node.node_index);
        }
    });

    let start = Instant::now();
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
                slam.push_lidar(&message.sensor_id, message.time, &points)?;
            }
            SensorData::Imu(imu) => slam.push_imu(
                message.time,
                ImuSample {
                    linear_acceleration: imu.linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                },
            )?,
            SensorData::Odometry(odometry) => slam.push_odometry(message.time, odometry.pose)?,
            _ => {}
        }
    }
    slam.finish()?;
    println!("SLAM took {:.1} s.", start.elapsed().as_secs_f64());

    let stats = slam.stats();
    println!(
        "{} nodes, {} submaps, {} constraints ({} intra-submap, {} inter-submap of which {} \
         close loops), {} optimizations.",
        stats.num_nodes,
        stats.num_submaps,
        stats.num_constraints,
        stats.num_intra_submap_constraints,
        stats.num_inter_submap_constraints,
        stats.num_loop_closure_constraints,
        stats.num_global_optimizations
    );

    // Accuracy against ground truth (after a rigid 2D alignment).
    let reference = TimedPose::from_states(&dataset.ground_truth);
    for (name, poses) in [
        ("local SLAM", slam.local_trajectory()),
        ("with loop closure", slam.trajectory()),
    ] {
        let estimates: Vec<TimedPose> = poses
            .iter()
            .map(|p| TimedPose::new(p.time.as_secs_f64(), p.pose))
            .collect();
        let pairs = associate(
            &reference,
            &estimates,
            Association::Interpolate { max_gap: 0.05 },
        );
        let result = ate(&pairs, Alignment::Rigid2d);
        println!(
            "ATE {name}: RMSE {:.3} m, max {:.3} m",
            result.translation.rmse, result.translation.max
        );
    }

    let map = slam.map_2d()?;
    map.to_png(output_dir.join("map.png"))?;
    let yaml = map.to_ros_map(output_dir.join("map"))?;
    slam.save(output_dir.join("map.pbstream"))?;
    println!(
        "Map {} x {} cells at {} m, origin ({:.2}, {:.2}). Wrote {}, {} (+ .pgm) and {}.",
        map.width,
        map.height,
        map.resolution,
        map.origin.x,
        map.origin.y,
        output_dir.join("map.png").display(),
        yaml.display(),
        output_dir.join("map.pbstream").display()
    );
    Ok(())
}
