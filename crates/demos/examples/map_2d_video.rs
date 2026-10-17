//! Minimal 2D map video: Cartographer's occupancy grid being built while a
//! simulated robot tours an office. No UI: every frame is the real
//! `Slam::map_2d()` (all submaps at their current optimized poses) plus the
//! optimized trajectory and the robot.
//!
//! ```sh
//! cargo run --release -p cartographer-demos --example map_2d_video -- target/media/cartographer_rust_2d_map.mp4
//! ```
//! Writes PNG frames to a temporary directory next to the output and encodes
//! them with `ffmpeg` (must be on PATH).

use std::path::PathBuf;
use std::process::Command;

use cartographer::{Extrinsics, ImuSample, Mode, OccupancyGrid, Slam, SlamError, TimedPoint};
use cartographer_demos::canvas::{Canvas, Color};
use cartographer_sim::{
    DatasetBuilder, ImuConfig, Lidar2dConfig, OdometryConfig, SensorData, Trajectory, World2d,
};
use nalgebra::{Isometry2, Point2, Vector2};

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const FPS: u32 = 30;
/// One frame every this many scans.
const SCANS_PER_FRAME: usize = 2;
const HOLD_SECONDS: u32 = 3;

const UNKNOWN: Color = [10, 12, 18];
const FREE: Color = [30, 38, 54];
const WALL: Color = [235, 245, 255];
const PATH: Color = [64, 182, 255];
const ROBOT: Color = [255, 255, 255];

/// Fixed map-frame view fitted to the office.
struct View {
    min: Vector2<f64>,
    scale: f64,
    offset: Vector2<f64>,
    /// Cartographer's map frame is the first tracking pose; the view shows
    /// the world frame (office axis-aligned).
    world_from_map: Isometry2<f64>,
}

impl View {
    fn new(min: Vector2<f64>, max: Vector2<f64>, world_from_map: Isometry2<f64>) -> Self {
        let size = max - min;
        let scale = (WIDTH as f64 * 0.94 / size.x).min(HEIGHT as f64 * 0.94 / size.y);
        let offset = Vector2::new(
            (WIDTH as f64 - size.x * scale) / 2.0,
            (HEIGHT as f64 - size.y * scale) / 2.0,
        );
        View {
            min,
            scale,
            offset,
            world_from_map,
        }
    }
    /// Pixel of a map-frame point.
    fn px(&self, p_map: Vector2<f64>) -> (f64, f64) {
        let p = (self.world_from_map * Point2::from(p_map)).coords;
        let q = (p - self.min) * self.scale + self.offset;
        (q.x, HEIGHT as f64 - q.y)
    }
    /// Map-frame point at sub-pixel position `(x, y)` (pixels, y down).
    fn map_point_sub(&self, x: f64, y: f64) -> Vector2<f64> {
        let q = Vector2::new(x, HEIGHT as f64 - y);
        let world = (q - self.offset) / self.scale + self.min;
        (self.world_from_map.inverse() * Point2::from(world)).coords
    }
}

fn lerp(a: Color, b: Color, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| (f64::from(a[i]) + (f64::from(b[i]) - f64::from(a[i])) * t).round() as u8)
}

fn render(
    view: &View,
    grid: &OccupancyGrid,
    path: &[(f64, f64)],
    robot: Option<(f64, f64)>,
) -> Canvas {
    let mut canvas = Canvas::new(WIDTH, HEIGHT, UNKNOWN);
    // 3x3 supersampling: the map grid is rotated against the pixel grid.
    const SS: usize = 3;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut sum = [0.0f64; 3];
            for sy in 0..SS {
                for sx in 0..SS {
                    let p = view.map_point_sub(
                        x as f64 + (sx as f64 + 0.5) / SS as f64,
                        y as f64 + (sy as f64 + 0.5) / SS as f64,
                    );
                    let color = match grid.probability_at(p.x, p.y) {
                        None => UNKNOWN,
                        Some(prob) if prob < 0.5 => FREE,
                        Some(prob) => lerp(FREE, WALL, (f64::from(prob) - 0.5) * 2.5),
                    };
                    for i in 0..3 {
                        sum[i] += f64::from(color[i]);
                    }
                }
            }
            let n = (SS * SS) as f64;
            canvas.set(x as i64, y as i64, sum.map(|v| (v / n).round() as u8));
        }
    }
    canvas.polyline(path, 3, PATH);
    if let Some(r) = robot {
        canvas.disc(r, 13.0, lerp(UNKNOWN, PATH, 0.45));
        canvas.disc(r, 8.0, ROBOT);
    }
    canvas
}

/// The simulator's office tour (twice), as an open path that starts with a
/// straight run due east. Cartographer's map frame (and its grid axes) is the
/// first tracking pose, so starting axis-aligned keeps the office walls
/// aligned with the 5 cm grid instead of drawn as staircases.
fn office_tour_heading_east() -> Trajectory {
    let lap = [
        (-5.0, 0.0),
        (-4.5, -0.6),
        (-4.5, -1.5),
        (-5.2, -2.6),
        (-7.0, -3.4),
        (-7.6, -5.0),
        (-6.0, -6.0),
        (-3.6, -5.6),
        (-3.0, -4.2),
        (-3.8, -2.6),
        (-4.5, -1.5),
        (-4.5, -0.6),
        (-3.8, 0.0),
        (-1.6, 0.0),
        (-0.6, 0.4),
        (-0.5, 1.5),
        (0.2, 3.0),
        (1.5, 4.6),
        (0.0, 5.4),
        (-2.4, 4.6),
        (-2.4, 3.0),
        (-1.1, 2.2),
        (-0.5, 1.5),
        (-0.5, 0.5),
        (-1.6, -0.3),
        (-4.5, -0.4),
        (-6.0, -0.45),
        (-7.8, -0.55),
        (-8.9, -0.3),
        (-8.9, 0.3),
        (-7.8, 0.55),
        (-6.0, 0.3),
    ];
    let mut waypoints = vec![
        Point2::new(-9.0, 0.0),
        Point2::new(-8.4, 0.0),
        Point2::new(-7.8, 0.0),
        Point2::new(-7.2, 0.0),
    ];
    for _ in 0..2 {
        waypoints.extend(lap.iter().map(|&(x, y)| Point2::new(x, y)));
    }
    Trajectory::from_waypoints(&waypoints, 1.0, false)
}

fn main() -> Result<(), SlamError> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "cartographer_rust_2d_map.mp4".into()),
    );
    let frames_dir = out.with_extension("frames");
    std::fs::create_dir_all(&frames_dir).expect("create frames dir");

    let trajectory = office_tour_heading_east().with_start_from_rest(2.0);
    println!(
        "start heading: {:.3} deg",
        trajectory
            .state(trajectory.start_time())
            .pose
            .rotation
            .euler_angles()
            .2
            .to_degrees()
    );
    let start = trajectory.state(trajectory.start_time()).pose;
    let dataset = DatasetBuilder::new(trajectory, World2d::office())
        .seed(3)
        .lidar_2d("scan", Lidar2dConfig::default())
        .imu("imu", ImuConfig::default())
        .odometry("odom", OdometryConfig::default())
        .build();
    let extrinsics = |id: &str| Extrinsics::from_isometry(dataset.sensor(id).unwrap().extrinsic);

    // Cartographer's map frame is the first tracking pose; show the office
    // (x in [-10, 10], y in [-7, 7]) in the world frame.
    let world_from_map = Isometry2::new(
        Vector2::new(start.translation.x, start.translation.y),
        start.rotation.euler_angles().2,
    );
    let view = View::new(
        Vector2::new(-10.5, -7.5),
        Vector2::new(10.5, 7.5),
        world_from_map,
    );

    let mut slam = Slam::builder(Mode::TwoD)
        .lidar("scan", extrinsics("scan"))
        .imu("imu", extrinsics("imu"))
        .odometry_with_extrinsics("odom", extrinsics("odom"))
        .config(|c| {
            c.resolution = 0.05;
            c.max_range = 30.0;
            c.loop_closure.optimize_every_n_nodes = 60;
        })
        .build()?;

    let mut scans = 0usize;
    let mut frame = 0usize;
    let mut save = |canvas: &Canvas| {
        canvas
            .save_png(frames_dir.join(format!("{frame:05}.png")))
            .expect("write frame");
        frame += 1;
    };
    let snapshot = |slam: &Slam| -> Result<Canvas, SlamError> {
        let grid = slam.map_2d()?;
        let path: Vec<(f64, f64)> = slam
            .trajectory()
            .iter()
            .map(|p| view.px(Vector2::new(p.pose.translation.x, p.pose.translation.y)))
            .collect();
        let robot = slam
            .current_pose()
            .map(|p| view.px(Vector2::new(p.pose.translation.x, p.pose.translation.y)));
        Ok(render(&view, &grid, &path, robot))
    };

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
                scans += 1;
                if scans.is_multiple_of(SCANS_PER_FRAME) && !slam.trajectory().is_empty() {
                    save(&snapshot(&slam)?);
                }
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
    let last = {
        let grid = slam.map_2d()?;
        let path: Vec<(f64, f64)> = slam
            .trajectory()
            .iter()
            .map(|p| view.px(Vector2::new(p.pose.translation.x, p.pose.translation.y)))
            .collect();
        render(&view, &grid, &path, None)
    };
    for _ in 0..HOLD_SECONDS * FPS {
        save(&last);
    }
    println!("{scans} scans, {frame} frames");

    let status = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            &FPS.to_string(),
            "-i",
        ])
        .arg(frames_dir.join("%05d.png"))
        .args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
        ])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    std::fs::remove_dir_all(&frames_dir).ok();
    println!("wrote {}", out.display());
    Ok(())
}
