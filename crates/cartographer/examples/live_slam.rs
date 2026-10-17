//! Live SLAM with the high-level API, in 2D or 3D: sensor data arrives over
//! time, and the current pose and the map are read while the map is being
//! built.
//!
//! ```text
//! cargo run --release -p cartographer-slam --example live_slam -- [2d|3d] [SPEEDUP]
//! ```
//!
//! A "driver" thread plays a simulated robot in real time, sped up `SPEEDUP`
//! times (default 10), and sends each message over a channel, like a sensor
//! driver callback would: a 2D LiDAR, IMU and wheel odometry in 2D (default),
//! a 16-beam rotating LiDAR and IMU carried by a walking person in 3D
//! (simulated by `cartographer-sim`). The main thread pushes every message
//! into [`Slam`] as it arrives. Local SLAM runs inside the `push_*` calls;
//! loop closure and global optimization run on Cartographer's background
//! threads.
//!
//! Every simulated second the main thread prints the live pose and a summary
//! of the current map. At the end it compares the live poses it read, and
//! the final optimized trajectory, with ground truth.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use cartographer::{Extrinsics, ImuSample, Mode, Slam, SlamError, TimedPoint};
use cartographer_sim::metrics::{Alignment, Association, TimedPose, associate, ate};
use cartographer_sim::{
    Dataset, DatasetBuilder, ImuConfig, Lidar2dConfig, Lidar3dConfig, OdometryConfig, SensorData,
    SensorMessage, Trajectory, World2d, World3d,
};

fn main() -> Result<(), SlamError> {
    let mut args = std::env::args().skip(1);
    let mode = match args.next().as_deref() {
        None | Some("2d") => Mode::TwoD,
        Some("3d") => Mode::ThreeD,
        Some(other) => panic!("unknown mode '{other}': use 2d or 3d"),
    };
    let speedup: f64 = args
        .next()
        .map_or(10.0, |s| s.parse().expect("SPEEDUP: a number"));

    let (dataset, mut slam) = match mode {
        Mode::TwoD => setup_2d()?,
        Mode::ThreeD => setup_3d()?,
    };
    slam.on_global_optimization(|_| println!("  -- loop closure optimization finished --"));

    // The sensor "driver": sends each message when it would arrive live.
    let (sender, receiver) = mpsc::channel::<SensorMessage>();
    let messages = dataset.messages.clone();
    let driver = thread::spawn(move || {
        let wall_start = Instant::now();
        let data_start = messages.first().map_or(0.0, |m| m.time);
        for message in messages {
            let due = Duration::from_secs_f64((message.time - data_start) / speedup);
            if let Some(wait) = due.checked_sub(wall_start.elapsed()) {
                thread::sleep(wait);
            }
            if sender.send(message).is_err() {
                return;
            }
        }
    });

    // The SLAM loop: push data as it arrives, read results whenever needed.
    let mut live_poses = Vec::new();
    let mut next_report = dataset.start_time + 1.0;
    for message in receiver {
        match &message.data {
            SensorData::PointCloud(cloud) => {
                // Cartographer's convention: `time` is the stamp of the last
                // point and each point's time is relative to it (<= 0).
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

        if message.time >= next_report {
            next_report += 1.0;
            if let Some(pose) = report(&slam, message.time)? {
                live_poses.push(TimedPose::new(pose.time.as_secs_f64(), pose.pose));
            }
        }
    }
    driver.join().expect("driver thread");
    slam.finish()?;

    let stats = slam.stats();
    println!(
        "Finished: {} nodes, {} submaps, {} loop closure constraints, {} optimizations.",
        stats.num_nodes,
        stats.num_submaps,
        stats.num_loop_closure_constraints,
        stats.num_global_optimizations
    );

    // Accuracy against ground truth (rigidly aligned, since the map frame is
    // Cartographer's own). The live poses are the online estimates: before a
    // loop closes they carry local SLAM drift, which the optimized
    // trajectory removes afterwards.
    let alignment = match mode {
        Mode::TwoD => Alignment::Rigid2d,
        Mode::ThreeD => Alignment::Rigid3d,
    };
    let reference = TimedPose::from_states(&dataset.ground_truth);
    let optimized: Vec<TimedPose> = slam
        .trajectory()
        .iter()
        .map(|p| TimedPose::new(p.time.as_secs_f64(), p.pose))
        .collect();
    for (name, poses) in [
        ("live poses read during the run", &live_poses),
        ("final optimized trajectory", &optimized),
    ] {
        let pairs = associate(
            &reference,
            poses,
            Association::Interpolate { max_gap: 0.05 },
        );
        let result = ate(&pairs, alignment);
        println!(
            "Error of the {name} ({} poses): RMSE {:.1} cm, max {:.1} cm.",
            pairs.len(),
            result.translation.rmse * 100.0,
            result.translation.max * 100.0
        );
    }
    Ok(())
}

/// Prints the live pose and a summary of the map built so far, and returns
/// the pose.
fn report(slam: &Slam, time: f64) -> Result<Option<cartographer::StampedPose>, SlamError> {
    // Live pose: the latest local SLAM pose with the current loop closure
    // correction applied.
    let Some(pose) = slam.current_pose() else {
        println!("t={time:6.1} s  waiting for the first scan match");
        return Ok(None);
    };
    let p = pose.pose.translation.vector;
    let yaw = pose.pose.rotation.euler_angles().2.to_degrees();

    // The current map: all submaps at their latest optimized poses.
    let map = match slam.mode() {
        Mode::TwoD => {
            let grid = slam.map_2d()?;
            let occupied = grid.cells.iter().filter(|&&c| c > 0.65).count();
            format!(
                "map {}x{} cells, {occupied} occupied",
                grid.width, grid.height
            )
        }
        Mode::ThreeD => format!("map {} occupied voxels", slam.map_3d()?.points.len()),
    };
    println!(
        "t={time:6.1} s  pose x={:6.2} y={:6.2} z={:5.2} yaw={yaw:7.1} deg  {map}, {} submaps",
        p.x,
        p.y,
        p.z,
        slam.submaps().len()
    );
    Ok(Some(pose))
}

fn setup_2d() -> Result<(Dataset, Slam), SlamError> {
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops(1.25, 1.5).with_start_from_rest(3.0),
        World2d::loop_corridor(),
    )
    .seed(1)
    .lidar_2d("scan", Lidar2dConfig::default())
    .imu("imu", ImuConfig::default())
    .odometry("odom", OdometryConfig::default())
    .build();
    let extrinsics = |id: &str| Extrinsics::from_isometry(dataset.sensor(id).unwrap().extrinsic);
    let slam = Slam::builder(Mode::TwoD)
        .lidar("scan", extrinsics("scan"))
        .imu("imu", extrinsics("imu"))
        .odometry_with_extrinsics("odom", extrinsics("odom"))
        .config(|c| {
            c.resolution = 0.05;
            c.max_range = 30.0;
            // Run a global optimization (loop closure) every 90 new nodes.
            c.loop_closure.optimize_every_n_nodes = 90;
        })
        .build()?;
    Ok((dataset, slam))
}

fn setup_3d() -> Result<(Dataset, Slam), SlamError> {
    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops_3d(1.25, 1.2).with_start_from_rest(3.0),
        World3d::loop_corridor(),
    )
    .seed(2)
    .lidar_3d("points", Lidar3dConfig::default())
    .imu("imu", ImuConfig::default())
    .build();
    let extrinsics = |id: &str| Extrinsics::from_isometry(dataset.sensor(id).unwrap().extrinsic);
    // In 3D the IMU is required and the tracking frame must be the IMU frame.
    let slam = Slam::builder(Mode::ThreeD)
        .lidar("points", extrinsics("points"))
        .imu("imu", extrinsics("imu"))
        .config(|c| {
            c.max_range = 60.0;
            c.submap_num_range_data = 160;
        })
        .build()?;
    Ok((dataset, slam))
}
