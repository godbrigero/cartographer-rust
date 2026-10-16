//! Port of `cartographer/sensor/`.

pub mod collator_interface;
pub mod compressed_point_cloud;
pub mod data;
pub mod fixed_frame_pose_data;
pub mod imu_data;
pub mod internal;
pub mod landmark_data;
pub mod map_by_time;
pub mod odometry_data;
pub mod point_cloud;
pub mod range_data;
pub mod rangefinder_point;
pub mod timed_point_cloud_data;

/// Generated protobuf types of package `cartographer.sensor.proto`.
pub mod proto {
    pub use crate::protos::cartographer::sensor::proto::*;
}
