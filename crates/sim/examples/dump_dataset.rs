//! Prints a summary of a generated dataset.
//!
//! Usage: `cargo run -p cartographer-sim --example dump_dataset -- [2d|3d]
//! [laps] [seed]` (defaults: `2d 1 0`).

use cartographer_sim::metrics::{Alignment, Association, RpeDelta, TimedPose, associate, ate, rpe};
use cartographer_sim::{
    CellState, DatasetBuilder, FixedFramePoseConfig, LandmarkConfig, SensorData, SimWorld,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map_or("2d", String::as_str);
    let laps: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    let builder = match mode {
        "3d" => DatasetBuilder::corridor_3d(laps),
        _ => DatasetBuilder::corridor_2d(laps),
    };
    let dataset = builder
        .fixed_frame_pose("gps", FixedFramePoseConfig::default())
        .landmarks("landmarks", LandmarkConfig::default())
        .seed(seed)
        .build();

    println!("== cartographer-sim dataset ({mode}, {laps} laps, seed {seed}) ==");
    println!(
        "time range: [{:.3}, {:.3}] s, {} messages, {} ground-truth samples",
        dataset.start_time,
        dataset.end_time,
        dataset.messages.len(),
        dataset.ground_truth.len()
    );
    let path = dataset
        .trajectory
        .path_length(dataset.start_time, dataset.end_time);
    println!(
        "trajectory: {path:.1} m, mean speed {:.2} m/s, lap period {:?} s",
        path / (dataset.end_time - dataset.start_time),
        dataset.trajectory.period()
    );
    match &dataset.world {
        SimWorld::Planar(w) => {
            let g = w.rasterize(0.05, 1.0);
            println!(
                "world: 2D, {} wall segments; GT raster {}x{} @ 5 cm: {} occupied, {} free",
                w.segments.len(),
                g.layout.width,
                g.layout.height,
                g.count(CellState::Occupied),
                g.count(CellState::Free)
            );
        }
        SimWorld::Volumetric(w) => {
            println!(
                "world: 3D, {} boxes, {} planes",
                w.boxes.len(),
                w.planes.len()
            );
        }
    }

    println!("\nsensors:");
    for (info, (_, count)) in dataset.sensors.iter().zip(dataset.message_counts()) {
        let t = info.extrinsic.translation.vector;
        println!(
            "  {:<10} {:<15} {:>6} msgs   extrinsic t = ({:.2}, {:.2}, {:.2})",
            info.id,
            format!("{:?}", info.kind),
            count,
            t.x,
            t.y,
            t.z
        );
    }

    for info in &dataset.sensors {
        let clouds: Vec<_> = dataset.point_clouds(&info.id).collect();
        if clouds.is_empty() {
            continue;
        }
        let n_points: usize = clouds.iter().map(|(_, c)| c.points.len()).sum();
        let (mut rmin, mut rmax) = (f32::INFINITY, 0f32);
        let mut first_time = 0f32;
        for (_, c) in &clouds {
            for p in &c.points {
                let r = p.position.norm();
                rmin = rmin.min(r);
                rmax = rmax.max(r);
            }
            first_time = first_time.min(c.points.first().map_or(0.0, |p| p.time));
        }
        let (t, c) = clouds[clouds.len() / 2];
        println!(
            "\n{}: {} clouds, {:.0} points/cloud, ranges [{rmin:.2}, {rmax:.2}] m, \
             earliest relative point time {first_time:.4} s",
            info.id,
            clouds.len(),
            n_points as f64 / clouds.len() as f64,
        );
        println!(
            "  middle cloud @ {t:.4} s: {} points, first t = {:.4}, last t = {:.4}, {} intensities",
            c.points.len(),
            c.points.first().map_or(0.0, |p| p.time),
            c.points.last().map_or(0.0, |p| p.time),
            c.intensities.len()
        );
    }

    let imu: Vec<_> = dataset.imu("imu").collect();
    if !imu.is_empty() {
        let n = imu.len() as f64;
        let acc = imu.iter().fold(nalgebra::Vector3::zeros(), |a, (_, m)| {
            a + m.linear_acceleration
        }) / n;
        let gyro = imu.iter().fold(nalgebra::Vector3::zeros(), |a, (_, m)| {
            a + m.angular_velocity
        }) / n;
        println!(
            "\nimu: {} samples, mean accel ({:.3}, {:.3}, {:.3}), mean gyro ({:.4}, {:.4}, {:.4})",
            imu.len(),
            acc.x,
            acc.y,
            acc.z,
            gyro.x,
            gyro.y,
            gyro.z
        );
    }

    let landmark_obs: usize = dataset
        .messages_for("landmarks")
        .map(|m| match &m.data {
            SensorData::Landmark(l) => l.observations.len(),
            _ => 0,
        })
        .sum();
    println!("landmarks: {landmark_obs} observations");

    // Metrics demo: odometry vs ground truth.
    let reference = TimedPose::from_states(&dataset.ground_truth);
    let odom: Vec<TimedPose> = dataset
        .messages_for("odom")
        .filter_map(|m| match &m.data {
            SensorData::Odometry(o) => Some(TimedPose::new(m.time, o.pose)),
            _ => None,
        })
        .collect();
    if !odom.is_empty() {
        let pairs = associate(
            &reference,
            &odom,
            Association::Interpolate { max_gap: 0.05 },
        );
        let raw = ate(&pairs, Alignment::None);
        let aligned = ate(&pairs, Alignment::Rigid2d);
        let r = rpe(&pairs, RpeDelta::Distance(10.0));
        println!(
            "\nodometry vs GT: ATE rmse {:.3} m (unaligned) / {:.3} m (2D-aligned), \
             max {:.3} m; RPE@10m trans rmse {:.3} m, rot rmse {:.2} deg",
            raw.translation.rmse,
            aligned.translation.rmse,
            raw.translation.max,
            r.translation.rmse,
            r.rotation.rmse.to_degrees()
        );
    }
}
