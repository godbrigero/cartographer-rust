//! Port of `cartographer/mapping/internal/local_slam_result_data` (.h).

use std::any::Any;

use crate::sensor::data::Data;

/// A local SLAM result (a node plus the submaps it was inserted into)
/// travelling through the sensor collators like other sensor data (C++
/// abstract class `LocalSlamResultData : public sensor::Data`).
///
/// Implementations store the sensor id and the time passed to the C++
/// constructor `LocalSlamResultData(sensor_id, time)` and return them from
/// [`Data::get_sensor_id`] / [`Data::get_time`]. Their
/// [`Data::add_to_trajectory_builder`] hands the packet to
/// `TrajectoryBuilderInterface::add_local_slam_result_data`.
pub trait LocalSlamResultData: Data {
    /// Adds the result to `pose_graph` for `trajectory_id`.
    ///
    /// RUST PORT NOTE: the C++ takes a `PoseGraph*` and `static_cast`s it to
    /// the concrete `PoseGraph2D`/`PoseGraph3D` (guarded by a `DCHECK` on a
    /// `dynamic_cast`). The Rust `PoseGraph` trait cannot be downcast, so the
    /// concrete pose graph is passed as `&dyn Any` and implementations
    /// downcast it, panicking on a mismatch.
    fn add_to_pose_graph(&self, trajectory_id: i32, pose_graph: &dyn Any);
}
