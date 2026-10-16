//! 3D SLAM with the high-level API on a simulated, handheld corridor walk.
//!
//! ```text
//! cargo run --release -p cartographer --example slam_3d -- [OUTPUT_DIR] [LAPS]
//! ```
//!
//! A person walks `LAPS` (default 1.25) laps around a 3 m high corridor
//! loop carrying a 16-beam rotating LiDAR and an IMU (simulated by
//! `cartographer-sim`). The example prints statistics and the trajectory
//! error against ground truth, and writes to `OUTPUT_DIR` (default
//! `slam_3d_output`):
//!
//! * `map.ply` — occupied voxels of the high-resolution submaps in the
//!   optimized map frame (open with CloudCompare, MeshLab or Open3D);
//! * `trajectory.txt` — optimized poses in TUM format
//!   (`time x y z qx qy qz qw`);
//! * `map.pbstream` — the full SLAM state.
//!
//! The LiDAR scans are converted from a "time relative to the first point"
//! convention, as delivered by Velodyne, Ouster or Unitree drivers, with
//! [`cartographer::slam::points_relative_to_first`].

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use cartographer::slam::points_relative_to_first;
use cartographer::{Extrinsics, ImuSample, Mode, Slam, SlamError, Timestamp};
use cartographer_sim::metrics::{Alignment, Association, TimedPose, associate, ate};
use cartographer_sim::{DatasetBuilder, ImuConfig, Lidar3dConfig, SensorData, Trajectory, World3d};

fn main() -> Result<(), SlamError> {
    let mut args = std::env::args().skip(1);
    let output_dir = PathBuf::from(args.next().unwrap_or_else(|| "slam_3d_output".to_string()));
    let laps: f64 = args
        .next()
        .map_or(1.25, |laps| laps.parse().expect("LAPS: a number"));
    let io_error = |path: &PathBuf| {
        let path = path.clone();
        move |source| SlamError::Io { path, source }
    };
    std::fs::create_dir_all(&output_dir).map_err(io_error(&output_dir))?;

    let dataset = DatasetBuilder::new(
        Trajectory::corridor_loops_3d(laps, 1.2).with_start_from_rest(3.0),
        World3d::loop_corridor(),
    )
    .seed(2)
    .lidar_3d("points", Lidar3dConfig::default())
    .imu("imu", ImuConfig::default())
    .build();
    let extrinsics = |id: &str| Extrinsics::from_isometry(dataset.sensor(id).unwrap().extrinsic);
    println!(
        "Simulated {:.0} s: {} messages.",
        dataset.end_time - dataset.start_time,
        dataset.messages.len()
    );

    // In 3D the tracking frame must be the IMU frame (an IMU lever arm is
    // rejected); other extrinsics are relative to it.
    let mut slam = Slam::builder(Mode::ThreeD)
        .lidar("points", extrinsics("points"))
        .imu("imu", extrinsics("imu"))
        .config(|c| {
            c.max_range = 60.0;
            c.submap_num_range_data = 160;
        })
        .build()?;

    let start = Instant::now();
    for message in &dataset.messages {
        match &message.data {
            SensorData::PointCloud(cloud) => {
                // What a typical driver delivers: the stamp of the first point
                // and per-point offsets from it.
                let first = cloud.points.first().map_or(0.0, |p| p.time);
                let stamp = Timestamp::from_secs_f64(message.time + f64::from(first));
                let driver_points = cloud.points.iter().map(|p| (p.position, p.time - first));
                // Cartographer's convention: stamp of the last point, times <= 0.
                let (time, points) = points_relative_to_first(stamp, driver_points);
                slam.push_lidar(&message.sensor_id, time, &points)?;
            }
            SensorData::Imu(imu) => slam.push_imu(
                message.time,
                ImuSample {
                    linear_acceleration: imu.linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                },
            )?,
            _ => {}
        }
    }
    slam.finish()?;
    println!("SLAM took {:.1} s.", start.elapsed().as_secs_f64());

    let stats = slam.stats();
    println!(
        "{} nodes, {} submaps, {} constraints ({} close loops), {} optimizations.",
        stats.num_nodes,
        stats.num_submaps,
        stats.num_constraints,
        stats.num_loop_closure_constraints,
        stats.num_global_optimizations
    );

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
        let result = ate(&pairs, Alignment::Rigid3d);
        println!(
            "ATE {name}: RMSE {:.3} m, max {:.3} m",
            result.translation.rmse, result.translation.max
        );
    }

    let map = slam.map_3d()?;
    let ply = output_dir.join("map.ply");
    map.to_ply(&ply)?;

    let tum = output_dir.join("trajectory.txt");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tum).map_err(io_error(&tum))?);
    for pose in slam.trajectory() {
        let t = pose.pose.translation.vector;
        let q = pose.pose.rotation.coords; // (x, y, z, w)
        writeln!(
            file,
            "{:.6} {} {} {} {} {} {} {}",
            pose.time.as_secs_f64(),
            t.x,
            t.y,
            t.z,
            q.x,
            q.y,
            q.z,
            q.w
        )
        .map_err(io_error(&tum))?;
    }
    file.flush().map_err(io_error(&tum))?;

    let pbstream = output_dir.join("map.pbstream");
    slam.save(&pbstream)?;
    println!(
        "{} occupied voxels at {} m. Wrote {}, {} and {}.",
        map.points.len(),
        map.resolution,
        ply.display(),
        tum.display(),
        pbstream.display()
    );
    Ok(())
}
