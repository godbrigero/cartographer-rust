//! Port of `cartographer/sensor/internal/dispatchable` (.h).

use crate::common::time::Time;
use crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use crate::sensor::data::{Data, HasTime};
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;

/// Sensor data types that can be wrapped in a [`Dispatchable`].
///
/// RUST PORT NOTE: replaces the C++ overload resolution of
/// `trajectory_builder->AddSensorData(sensor_id_, data_)` with one trait impl
/// per data type.
pub trait DispatchableData: HasTime + Send + 'static {
    /// Calls the matching `TrajectoryBuilderInterface::add_*` method.
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    );
}

impl DispatchableData for TimedPointCloudData {
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_timed_point_cloud_data(sensor_id, self);
    }
}

impl DispatchableData for ImuData {
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_imu_data(sensor_id, self);
    }
}

impl DispatchableData for OdometryData {
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_odometry_data(sensor_id, self);
    }
}

impl DispatchableData for FixedFramePoseData {
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_fixed_frame_pose_data(sensor_id, self);
    }
}

impl DispatchableData for LandmarkData {
    fn add_to_trajectory_builder(
        &self,
        sensor_id: &str,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_landmark_data(sensor_id, self);
    }
}

/// A [`Data`] packet carrying a sensor measurement of type `D`.
#[derive(Clone, Debug)]
pub struct Dispatchable<D> {
    sensor_id: String,
    data: D,
}

impl<D: DispatchableData> Dispatchable<D> {
    /// Creates a packet for `sensor_id`.
    pub fn new(sensor_id: &str, data: D) -> Self {
        Dispatchable {
            sensor_id: sensor_id.to_string(),
            data,
        }
    }

    /// The payload.
    pub fn data(&self) -> &D {
        &self.data
    }
}

impl<D: DispatchableData> Data for Dispatchable<D> {
    fn get_time(&self) -> Time {
        self.data.time()
    }

    fn get_sensor_id(&self) -> &str {
        &self.sensor_id
    }

    fn add_to_trajectory_builder(
        self: Box<Self>,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        self.data
            .add_to_trajectory_builder(&self.sensor_id, trajectory_builder);
    }
}

/// Wraps `data` into a boxed [`Dispatchable`] (C++ `MakeDispatchable`).
///
/// RUST PORT NOTE: takes the payload by value (the C++ copies from a const
/// reference); callers clone if they need to keep it.
pub fn make_dispatchable<D: DispatchableData>(sensor_id: &str, data: D) -> Box<Dispatchable<D>> {
    Box::new(Dispatchable::new(sensor_id, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_universal;

    #[derive(Default)]
    struct RecordingBuilder {
        calls: Vec<(&'static str, String, Time)>,
    }

    impl TrajectoryBuilderInterface for RecordingBuilder {
        fn add_timed_point_cloud_data(&mut self, sensor_id: &str, d: &TimedPointCloudData) {
            self.calls.push(("range", sensor_id.to_string(), d.time));
        }
        fn add_imu_data(&mut self, sensor_id: &str, d: &ImuData) {
            self.calls.push(("imu", sensor_id.to_string(), d.time));
        }
        fn add_odometry_data(&mut self, sensor_id: &str, d: &OdometryData) {
            self.calls.push(("odometry", sensor_id.to_string(), d.time));
        }
        fn add_fixed_frame_pose_data(&mut self, sensor_id: &str, d: &FixedFramePoseData) {
            self.calls.push(("fixed", sensor_id.to_string(), d.time));
        }
        fn add_landmark_data(&mut self, sensor_id: &str, d: &LandmarkData) {
            self.calls.push(("landmark", sensor_id.to_string(), d.time));
        }
    }

    #[test]
    fn dispatches_to_matching_method() {
        let packets: Vec<Box<dyn Data>> = vec![
            make_dispatchable(
                "a",
                TimedPointCloudData {
                    time: from_universal(1),
                    ..Default::default()
                },
            ),
            make_dispatchable(
                "b",
                ImuData {
                    time: from_universal(2),
                    ..Default::default()
                },
            ),
            make_dispatchable(
                "c",
                OdometryData {
                    time: from_universal(3),
                    ..Default::default()
                },
            ),
            make_dispatchable(
                "d",
                FixedFramePoseData {
                    time: from_universal(4),
                    pose: None,
                },
            ),
            make_dispatchable(
                "e",
                LandmarkData {
                    time: from_universal(5),
                    landmark_observations: Vec::new(),
                },
            ),
        ];
        let mut builder = RecordingBuilder::default();
        for (i, packet) in packets.into_iter().enumerate() {
            assert_eq!(packet.get_time(), from_universal(i as i64 + 1));
            packet.add_to_trajectory_builder(&mut builder);
        }
        let kinds: Vec<_> = builder.calls.iter().map(|c| c.0).collect();
        assert_eq!(kinds, ["range", "imu", "odometry", "fixed", "landmark"]);
        assert_eq!(builder.calls[2].1, "c");
        assert_eq!(builder.calls[4].2, from_universal(5));
    }
}
