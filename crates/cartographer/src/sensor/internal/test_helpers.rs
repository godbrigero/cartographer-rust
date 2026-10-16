//! Port of `cartographer/sensor/internal/test_helpers` (.h).

use nalgebra::Vector3;

use crate::common::time::{Time, from_universal};
use crate::sensor::collator_interface::CollatorInterface;
use crate::sensor::data::Data;
use crate::sensor::imu_data::ImuData;
use crate::sensor::internal::dispatchable::make_dispatchable;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::Rigid3d;

/// C++ matcher `Near(point)`: Eigen `arg.isApprox(point, 0.001f)`.
pub fn near(arg: &Vector3<f32>, point: &Vector3<f32>) -> bool {
    (arg - point).norm() <= 0.001 * arg.norm().min(point.norm())
}

/// Helpers of C++ namespace `sensor::testing`.
pub mod testing {
    use super::*;

    /// `(trajectory_id, sensor_id, time)`.
    pub type CollatorOutput = (i32, String, Time);

    /// A sensor packet to feed into a collator together with the output it
    /// is expected to produce.
    pub struct CollatorInput {
        /// The trajectory id.
        pub trajectory_id: i32,
        /// The packet; `None` once moved into a collator.
        pub data: Option<Box<dyn Data>>,
        /// The expected collator output.
        pub expected_output: CollatorOutput,
    }

    impl CollatorInput {
        fn new(trajectory_id: i32, sensor_id: &str, time: i64, data: Box<dyn Data>) -> Self {
            CollatorInput {
                trajectory_id,
                data: Some(data),
                expected_output: (trajectory_id, sensor_id.to_string(), from_universal(time)),
            }
        }

        /// IMU data at `FromUniversal(time)`.
        pub fn create_imu_data(trajectory_id: i32, sensor_id: &str, time: i64) -> Self {
            let data = make_dispatchable(
                sensor_id,
                ImuData {
                    time: from_universal(time),
                    ..Default::default()
                },
            );
            Self::new(trajectory_id, sensor_id, time, data)
        }

        /// Empty point cloud data at `FromUniversal(time)`.
        pub fn create_timed_point_cloud_data(
            trajectory_id: i32,
            sensor_id: &str,
            time: i64,
        ) -> Self {
            let data = make_dispatchable(
                sensor_id,
                TimedPointCloudData {
                    time: from_universal(time),
                    origin: Vector3::zeros(),
                    ranges: Vec::new(),
                    intensities: Vec::new(),
                },
            );
            Self::new(trajectory_id, sensor_id, time, data)
        }

        /// Identity odometry data at `FromUniversal(time)`.
        pub fn create_odometry_data(trajectory_id: i32, sensor_id: &str, time: i64) -> Self {
            let data = make_dispatchable(
                sensor_id,
                OdometryData {
                    time: from_universal(time),
                    pose: Rigid3d::identity(),
                },
            );
            Self::new(trajectory_id, sensor_id, time, data)
        }

        /// Moves the packet into `collator` (C++ `MoveToCollator`).
        pub fn move_to_collator(&mut self, collator: &mut dyn CollatorInterface) {
            let data = self.data.take().expect("data was already moved");
            collator.add_sensor_data(self.trajectory_id, data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_is_relative() {
        assert!(near(
            &Vector3::new(1000., 0., 0.),
            &Vector3::new(1000.5, 0., 0.)
        ));
        assert!(!near(&Vector3::new(1., 0., 0.), &Vector3::new(1.01, 0., 0.)));
    }
}
