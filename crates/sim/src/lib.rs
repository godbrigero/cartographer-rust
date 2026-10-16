//! Deterministic synthetic-data simulator for the Cartographer Rust port.

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
