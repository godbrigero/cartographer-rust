//! # cartographer-sim
//!
//! Deterministic synthetic-data simulator for end-to-end tests, benchmarks
//! and the visual report of the Cartographer Rust port. It is deliberately
//! independent of the `cartographer` crate: everything is plain data
//! (`f64` seconds, `nalgebra` types) that a thin adapter converts into
//! Cartographer's sensor structs.
//!
//! ## Pieces
//!
//! * **Worlds** — [`World2d`] (line-segment walls + walkable polygons;
//!   presets [`World2d::rect_room`], [`World2d::loop_corridor`],
//!   [`World2d::office`]) and [`World3d`] (axis-aligned boxes + planes;
//!   presets [`World3d::room`], [`World3d::loop_corridor`], or
//!   [`World3d::extrude_2d`] of any 2D world). Both implement [`RayCast`].
//!   [`World2d::rasterize`] produces a [`GroundTruthGrid`] for map metrics;
//!   [`World2d::is_free`] tests walkability.
//! * **Trajectories** — [`Trajectory`]: C2 cubic spline through waypoints,
//!   constant speed, yaw tangent to the path, optional [`HandheldMotion`]
//!   (z bobbing, roll/pitch sway). [`Trajectory::state`] gives pose, linear
//!   velocity/acceleration and angular velocity analytically. Presets:
//!   [`Trajectory::corridor_loops`], [`Trajectory::room_wander`],
//!   [`Trajectory::office_tour`] and the `_3d` variants.
//! * **Sensors** ([`sensors`]) — [`Lidar2dConfig`], [`Lidar3dConfig`],
//!   [`ImuConfig`], [`OdometryConfig`], [`FixedFramePoseConfig`],
//!   [`LandmarkConfig`]; each has `rate_hz`, noise parameters and an
//!   `extrinsic`.
//! * **Datasets** — [`DatasetBuilder`] runs all sensors over `[t0, t1]` and
//!   returns a [`Dataset`] with time-sorted [`SensorMessage`]s and sampled
//!   ground truth. Presets: [`DatasetBuilder::corridor_2d`],
//!   [`DatasetBuilder::corridor_3d`].
//! * **Metrics** ([`metrics`]) — Umeyama alignment, ATE, RPE, timestamp
//!   association, occupancy precision/recall.
//!
//! ## Conventions
//!
//! * **Frames.** The world frame is the ground-truth map frame (z up).
//!   Trajectory poses are `world_from_tracking`; linear velocity and
//!   acceleration are in the world frame, angular velocity in the tracking
//!   (body) frame. Every sensor has `extrinsic = tracking_from_sensor`
//!   (default identity) and reports data **in its sensor frame**; the
//!   consumer applies the extrinsic, exactly as cartographer_ros does with TF.
//!   Note Cartographer's map frame is anchored at the first tracking pose,
//!   so compare estimates after alignment (or transform by the initial GT
//!   pose).
//! * **Point-cloud timing** (identical to Cartographer's
//!   `TimedPointCloudData`): the message time is the absolute time of the
//!   **last** point; every point's `time` is relative to it, so the last
//!   point has `0` and earlier points negative offsets, non-decreasing in
//!   scan order. Each point was measured from the sensor pose at
//!   `message.time + point.time` (motion distortion for
//!   [`ScanTiming::Rotating`]). With [`MissHandling::Return`] missing beams
//!   produce flagged points beyond `max_range` that Cartographer turns into
//!   `missing_data_ray_length` misses.
//! * **IMU.** `linear_acceleration` is the specific force (kinematic
//!   acceleration minus gravity) in the IMU frame — a level IMU at rest reads
//!   `(0, 0, +9.81)`; `angular_velocity` in the IMU frame.
//! * **Odometry.** Pose of the odometry sensor frame in an odometry frame
//!   that coincides with the world frame at the first sample; drift is a
//!   random walk on increments (plus optional scale / yaw-rate bias).
//! * **Times** are `f64` seconds; sensor sample `k` is at
//!   `t0 + k / rate_hz`.
//! * **Determinism.** Each sensor uses its own ChaCha8 stream seeded from
//!   the dataset seed and the sensor id, so identical inputs give
//!   bit-identical datasets and adding a sensor never perturbs another.
//!
//! ## Example
//!
//! ```
//! use cartographer_sim::{DatasetBuilder, SensorData};
//!
//! let dataset = DatasetBuilder::corridor_2d(1.0).seed(7).time_range(0.0, 1.0).build();
//! for msg in &dataset.messages {
//!     match &msg.data {
//!         SensorData::PointCloud(cloud) => {
//!             assert_eq!(cloud.points.last().map(|p| p.time), Some(0.0));
//!         }
//!         SensorData::Imu(imu) => assert!(imu.linear_acceleration.z > 9.0),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! See `examples/dump_dataset.rs` for a summary dump
//! (`cargo run -p cartographer-sim --example dump_dataset -- 3d`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod dataset;
pub mod grid;
pub mod metrics;
pub mod raycast;
pub mod sensors;
pub mod spline;
pub mod trajectory;
pub mod world2d;
pub mod world3d;

pub use dataset::{Dataset, DatasetBuilder, SensorInfo, SensorMessage, SimWorld};
pub use grid::{CellState, GridLayout, GroundTruthGrid, ProbabilityGrid};
pub use raycast::{Hit3, RayCast};
pub use sensors::{
    FixedFramePoseConfig, FixedFramePoseMessage, ImuConfig, ImuMessage, Landmark, LandmarkConfig,
    LandmarkMessage, LandmarkObservation, Lidar2dConfig, Lidar3dConfig, MissHandling,
    OdometryConfig, OdometryMessage, PointCloudMessage, ScanTiming, SensorConfig, SensorData,
    SensorKind, TimedPoint,
};
pub use trajectory::{HandheldMotion, Trajectory, TrajectoryState};
pub use world2d::{Hit2, Polygon2, Segment2, World2d};
pub use world3d::{Aabb, Plane, World3d};
