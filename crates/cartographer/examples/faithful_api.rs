//! The faithful (C++-shaped) API, driven like a cartographer_ros node would:
//! Lua options -> `MapBuilder` -> trajectory builder -> sensor data ->
//! final optimization -> `.pbstream`.
//!
//! `cargo run --release -p cartographer-slam --example faithful_api -- [OUT.pbstream]`

use std::collections::BTreeSet;

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::{
    MapBuilder, MapBuilderInterface, SensorId, SensorType, create_map_builder_options,
    create_trajectory_builder_options,
};
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use nalgebra::Vector3;

/// Number of beams per scan, i.e. 0.25 deg angular resolution like typical
/// 2D LiDARs.
///
/// Do not lower this to 360 (1 deg): with noise-free 1 deg beams, the side
/// walls (parallel to the motion) leave a sparse comb of hit cells in the
/// submap that every later scan reproduces exactly, and this anchors the
/// scan matcher at "no motion". Without odometry/IMU, the robot creeping
/// 2 cm (0.4 cells) per scan then never gets the extrapolator moving, and
/// local SLAM stalls at the origin for some start positions (e.g. x = -1).
/// That is faithful Cartographer behaviour, see
/// `docs/notes/2d_local_slam_slow_motion_sparse_scans.md`.
const NUM_BEAMS: u32 = 1440;

/// A planar scan of a 10 m x 6 m room (walls at x = +-5, y = +-3), taken
/// from `(x, 0)`.
fn scan(time: Time, x: f64) -> TimedPointCloudData {
    let ranges = (0..NUM_BEAMS)
        .map(|i| {
            let (s, c) =
                (2.0 * std::f64::consts::PI * f64::from(i) / f64::from(NUM_BEAMS)).sin_cos();
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

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "faithful_api.pbstream".into());
    let code = r#"
      include "map_builder.lua"
      include "trajectory_builder.lua"
      MAP_BUILDER.use_trajectory_builder_2d = true
      TRAJECTORY_BUILDER_2D.use_imu_data = false  -- this example has no IMU
      TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
      return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }
    "#;
    let dict = LuaParameterDictionary::new(
        code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    );
    let mut map_builder = MapBuilder::new(&create_map_builder_options(
        &dict.get_dictionary("map_builder"),
    ));
    let sensors: BTreeSet<SensorId> = [SensorId {
        r#type: SensorType::Range,
        id: "scan".into(),
    }]
    .into();
    let tb_options = create_trajectory_builder_options(&dict.get_dictionary("trajectory_builder"));
    let trajectory_id = map_builder.add_trajectory_builder(&sensors, &tb_options, None);

    let builder = map_builder.get_trajectory_builder(trajectory_id).unwrap();
    for k in 0..100 {
        let t = 0.1 * f64::from(k);
        let time = Time::from_time_since_epoch(from_seconds(1000.0 + t));
        // The robot drives along +x at 0.2 m/s from (-1, 0).
        builder.add_timed_point_cloud_data("scan", &scan(time, -1.0 + 0.2 * t));
    }
    map_builder.finish_trajectory(trajectory_id);
    let pose_graph = map_builder.pose_graph();
    pose_graph.run_final_optimization();
    let nodes = pose_graph.get_trajectory_nodes();
    println!("{} nodes", nodes.size());
    if let Some(last) = nodes.iter().next_back() {
        println!("last node global pose: {}", last.data.global_pose);
    }
    assert!(map_builder.serialize_state_to_file(true, &out));
    println!("wrote {out}");
}
