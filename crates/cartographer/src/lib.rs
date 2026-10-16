//! Native Rust port of Google Cartographer.
//!
//! # Two layers
//!
//! * **High-level API** — [`slam`] (re-exported here: [`Slam`],
//!   [`SlamBuilder`], [`SlamConfig`], [`Mode`], [`Extrinsics`], ...): declare
//!   sensors, push data with times in seconds, get optimized poses,
//!   occupancy grids (PNG / ROS map), 3D point maps (PLY) and `.pbstream`
//!   files. Typed configuration with Cartographer's defaults; no Lua or ROS
//!   knowledge needed.
//! * **Faithful port** — [`mapping`], [`sensor`], [`transform`], [`io`],
//!   [`common`], ...: Cartographer's C++ API one-to-one (`MapBuilder`,
//!   option protos, Lua dictionaries). The high-level API is a thin wrapper
//!   around it and never changes its semantics.
//!
//! ```no_run
//! use cartographer::{Extrinsics, ImuSample, Mode, Slam, TimedPoint};
//! # fn sensor_data() -> Vec<(f64, Vec<TimedPoint>, ImuSample)> { Vec::new() }
//!
//! let mut slam = Slam::builder(Mode::TwoD)
//!     .lidar("scan", Extrinsics::from_translation(0.1, 0.0, 0.2))
//!     .imu("imu", Extrinsics::identity())
//!     .config(|c| c.max_range = 20.0)
//!     .build()?;
//! for (t, points, imu) in sensor_data() {
//!     slam.push_imu(t, imu)?;
//!     slam.push_lidar("scan", t, &points)?;
//! }
//! slam.finish()?;
//! slam.map_2d()?.to_png("map.png")?;
//! slam.save("map.pbstream")?;
//! # Ok::<(), cartographer::SlamError>(())
//! ```

/// Generated protobuf types, laid out by proto package
/// (`cartographer.mapping.proto` -> `protos::cartographer::mapping::proto`).
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod protos {
    include!(concat!(env!("OUT_DIR"), "/protos.rs"));
}

pub mod common;
pub mod ground_truth;
pub mod io;
pub mod mapping;
pub mod metrics;
pub mod sensor;
pub mod slam;
pub mod transform;

pub use slam::{
    Extrinsics, ImuSample, LocalResult, MapPoint, Mode, OccupancyGrid, OptimizationEvent, PointMap,
    SensorKind, Slam, SlamBuilder, SlamConfig, SlamError, SlamStats, StampedPose, SubmapInfo,
    TimedPoint, Timestamp,
};
