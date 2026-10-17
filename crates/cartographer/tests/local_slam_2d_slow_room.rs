//! A robot creeping through a rectangular room without IMU or odometry (no
//! C++ counterpart). See `docs/notes/2d_local_slam_slow_motion_sparse_scans.md`.
//!
//! `LocalTrajectoryBuilder2D` with the shipped `trajectory_builder_2d.lua`
//! (`use_imu_data = false`, online correlative scan matching on) is fed
//! noise-free planar scans of the room `[-5, 5] x [-3, 3]` taken from
//! `(x0 + 0.2 t, 0)`, `t = 0 .. 9.9 s` at 10 Hz, i.e. 2 cm (0.4 cells) per
//! scan. The local frame starts at the first scan, so the true local
//! position at the end is `x = 1.98`.

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use cartographer::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use nalgebra::Vector3;

const TRUE_FINAL_LOCAL_X: f64 = 1.98;

/// A planar scan with `num_beams` evenly spaced beams (the first one along
/// +x) of the room `[-5, 5] x [-3, 3]`, taken from `(x, 0)`.
fn scan(time: Time, x: f64, num_beams: u32) -> TimedPointCloudData {
    let ranges = (0..num_beams)
        .map(|i| {
            let (s, c) =
                (2.0 * std::f64::consts::PI * f64::from(i) / f64::from(num_beams)).sin_cos();
            let to_x = if c > 0.0 {
                (5.0 - x) / c
            } else {
                (-5.0 - x) / c
            };
            let r = to_x.abs().min(3.0 / s.abs());
            TimedRangefinderPoint {
                position: Vector3::new((r * c) as f32, (r * s) as f32, 0.0),
                time: 0.0,
            }
        })
        .collect();
    TimedPointCloudData {
        time,
        origin: Vector3::zeros(),
        ranges,
        intensities: vec![],
    }
}

/// Runs local SLAM over the 100 scans and returns the x of the last local
/// pose.
fn final_local_x(x0: f64, num_beams: u32) -> f64 {
    let options = create_local_trajectory_builder_options_2d(&LuaParameterDictionary::new(
        r#"
          include "trajectory_builder_2d.lua"
          TRAJECTORY_BUILDER_2D.use_imu_data = false
          TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
          return TRAJECTORY_BUILDER_2D
        "#,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    ));
    let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
    let mut last_x = f64::NAN;
    for k in 0..100 {
        let t = 0.1 * f64::from(k);
        let time = Time::from_time_since_epoch(from_seconds(1000.0 + t));
        if let Some(result) = builder.add_range_data("scan", scan(time, x0 + 0.2 * t, num_beams)) {
            last_x = result.local_pose.translation().x;
        }
    }
    last_x
}

/// With 0.25 deg beams (typical of real 2D LiDARs) local SLAM tracks the
/// slow motion wherever the robot starts.
#[test]
fn dense_scans_track_slow_motion_for_all_start_positions() {
    for x0 in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let x = final_local_x(x0, 1440);
        assert!(
            (x - TRUE_FINAL_LOCAL_X).abs() < 0.1,
            "x0 = {x0}: final local x = {x}, expected {TRUE_FINAL_LOCAL_X}"
        );
    }
}

/// Pins the (faithful) Cartographer behaviour explained in the note: with
/// noise-free 1 deg beams, the side walls hit the first (and, because of the
/// motion filter, only) scan's grid in a sparse comb that is identical in
/// every later scan, which anchors the scan matcher at "no motion". Without
/// odometry nothing else starts the extrapolator moving, so from x0 = -1 the
/// local pose stays at the origin.
#[test]
fn sparse_noise_free_scans_stall_from_minus_one() {
    let x = final_local_x(-1.0, 360);
    assert!(x.abs() < 0.05, "final local x = {x}, expected ~0");
}
