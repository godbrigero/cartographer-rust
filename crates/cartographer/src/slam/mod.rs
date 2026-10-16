//! High-level, ergonomic SLAM API on top of the faithful Cartographer port.
//!
//! The rest of this crate mirrors Cartographer's C++ API one-to-one
//! ([`MapBuilder`](crate::mapping::map_builder::MapBuilder), option protos,
//! Lua dictionaries, `common::Time`, ...). This module wraps it — without
//! changing its behavior — for users who simply want a map and a trajectory
//! from their sensors:
//!
//! * [`Slam::builder`] declares the sensors with their mounting poses
//!   ([`Extrinsics`]) and returns a [`SlamBuilder`];
//! * [`SlamConfig`] exposes the most used knobs as typed, documented fields;
//!   defaults are read from Cartographer's shipped Lua files, Lua is never
//!   required (but [`SlamBuilder::from_lua`] accepts cartographer_ros-style
//!   configuration files), and [`SlamConfig::advanced`] edits the raw option
//!   protos;
//! * `push_*` methods take plain `nalgebra` data with times in seconds
//!   ([`Timestamp`]), validate it ([`SlamError`]) and transform it into the
//!   tracking frame like cartographer_ros does;
//! * results: optimized ([`Slam::trajectory`]) and local
//!   ([`Slam::local_trajectory`]) poses, [`Slam::current_pose`],
//!   [`Slam::submaps`], [`Slam::stats`], 2D occupancy grids
//!   ([`Slam::map_2d`] → PNG / ROS `map_server` files) and 3D point maps
//!   ([`Slam::map_3d`] → PLY), and `.pbstream` files ([`Slam::save`],
//!   [`SlamBuilder::localize_in`]).
//!
//! # Example
//!
//! A robot drives at 0.2 m/s through a 10 m × 6 m room, scanned 10 times
//! per second by a 360-beam planar LiDAR (synthetic data):
//!
//! ```
//! use cartographer::{Extrinsics, Mode, Slam};
//! use nalgebra::Vector3;
//!
//! /// A 360-beam scan from (x, 0) of the walls x = ±5, y = ±3.
//! fn scan(x: f64) -> Vec<Vector3<f32>> {
//!     (0..360)
//!         .map(|i| {
//!             let (s, c) = f64::from(i).to_radians().sin_cos();
//!             let to_x = if c > 0.0 { (5.0 - x) / c } else { (-5.0 - x) / c };
//!             let to_y = 3.0 / s.abs();
//!             let r = to_x.abs().min(to_y);
//!             Vector3::new((r * c) as f32, (r * s) as f32, 0.0)
//!         })
//!         .collect()
//! }
//!
//! let mut slam = Slam::builder(Mode::TwoD)
//!     .lidar("scan", Extrinsics::identity())
//!     .config(|c| {
//!         // No odometry and no IMU: let a correlative search find the motion.
//!         c.use_online_correlative_scan_matching = true;
//!         c.num_background_threads = 1;
//!     })
//!     .build()?;
//! for k in 0..50 {
//!     let t = 0.1 * f64::from(k); // seconds
//!     slam.push_lidar_untimed("scan", t, &scan(0.02 * f64::from(k)))?;
//! }
//! slam.finish()?;
//!
//! let trajectory = slam.trajectory(); // one pose per node (motion-filtered scan)
//! let last = trajectory.last().unwrap();
//! let true_x = 0.2 * last.time.as_secs_f64();
//! assert!((last.pose.translation.x - true_x).abs() < 0.1);
//! let map = slam.map_2d()?;
//! // The north wall (y = 3) is occupied, the room center is free.
//! let wall = (0..9)
//!     .filter_map(|i| map.probability_at(0.5, 2.8 + 0.05 * f64::from(i)))
//!     .fold(0.0, f32::max);
//! assert!(wall > 0.6);
//! assert!(map.probability_at(0.5, 0.5).unwrap() < 0.5);
//! # let dir = std::env::temp_dir();
//! # let png = dir.join(format!("slam_doc_{}.png", std::process::id()));
//! map.to_png(&png)?;
//! # std::fs::remove_file(&png).unwrap();
//! # Ok::<(), cartographer::SlamError>(())
//! ```
//!
//! The runnable examples `slam_2d` and `slam_3d`
//! (`cargo run --release -p cartographer --example slam_2d -- out/`) map
//! simulated corridors and write PNG, ROS map, PLY and `.pbstream` files.
//!
//! # Frames and conventions
//!
//! * **Tracking frame**: the body frame whose trajectory is estimated
//!   (`tracking_frame` in cartographer_ros). In 3D it should be the IMU
//!   frame: [`SlamBuilder::build`] rejects an IMU with a translation offset,
//!   like cartographer_ros.
//! * **Extrinsics** are `tracking_from_sensor`. Point clouds (and their ray
//!   origins) are transformed with them; IMU samples are rotated; odometry
//!   poses are composed as `odom_from_sensor * sensor_from_tracking`.
//! * **Map frame**: the first tracking pose (gravity-aligned in 3D), unless
//!   a map was loaded.
//! * **Times**: any epoch, as long as all sensors share it (see
//!   [`Timestamp`]). Each sensor's timestamps must be strictly increasing.
//!   Point times follow Cartographer's convention (relative to the last
//!   point, see [`TimedPoint`]); [`points_relative_to_first`] converts the
//!   "relative to the first point" convention of Velodyne, Ouster, Unitree
//!   and Livox drivers.
//! * Data of different sensors may arrive slightly out of order:
//!   Cartographer sorts them, but waits until every declared sensor has
//!   delivered data, so feed all declared sensors.

mod builder;
mod config;
mod error;
mod instance;
mod map;
mod time;
mod types;

pub use builder::SlamBuilder;
pub use config::{
    GlobalLocalizationConfig, LoopClosureConfig, MotionFilterConfig, OptimizationConfig,
    PureLocalizationConfig, ScanMatcherConfig, SlamConfig,
};
pub use error::SlamError;
pub use instance::{DEFAULT_MIN_OCCUPIED_PROBABILITY_3D, Slam};
pub use map::{MapPoint, OccupancyGrid, PointMap};
pub use time::Timestamp;
pub use types::{
    Extrinsics, ImuSample, LocalResult, Mode, OptimizationEvent, SensorKind, SlamStats,
    StampedPose, SubmapInfo, TimedPoint, points_relative_to_first, points_with_absolute_times,
};

pub use crate::mapping::id::{NodeId, SubmapId};
