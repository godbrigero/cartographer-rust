//! Port of `cartographer/mapping/imu_tracker` (.h, .cc).

use nalgebra::{Quaternion, Vector3};

use crate::common::time::{Time, to_seconds};
use crate::mapping::internal::eigen_quaterniond_from_two_vectors::from_two_vectors;
use crate::transform::rigid_transform::{
    quaternion_conjugate, quaternion_identity, quaternion_mul, quaternion_normalized,
    quaternion_rotate,
};
use crate::transform::transform::angle_axis_vector_to_rotation_quaternion;

/// Keeps track of the orientation using angular velocities and linear
/// accelerations from an IMU. Because averaged linear acceleration (assuming
/// slow movement) is a direct measurement of gravity, roll/pitch does not
/// drift, though yaw does.
#[derive(Clone, Debug)]
pub struct ImuTracker {
    imu_gravity_time_constant: f64,
    time: Time,
    last_linear_acceleration_time: Time,
    orientation: Quaternion<f64>,
    gravity_vector: Vector3<f64>,
    imu_angular_velocity: Vector3<f64>,
}

impl ImuTracker {
    /// Creates a tracker at `time` with identity orientation.
    pub fn new(imu_gravity_time_constant: f64, time: Time) -> Self {
        ImuTracker {
            imu_gravity_time_constant,
            time,
            last_linear_acceleration_time: Time::min(),
            orientation: quaternion_identity(),
            gravity_vector: Vector3::z(),
            imu_angular_velocity: Vector3::zeros(),
        }
    }

    /// Advances to the given `time` and updates the orientation to reflect
    /// this.
    pub fn advance(&mut self, time: Time) {
        assert!(
            self.time <= time,
            "Check failed: time_ <= time ({} vs. {})",
            self.time,
            time
        );
        let delta_t = to_seconds(time - self.time);
        let rotation =
            angle_axis_vector_to_rotation_quaternion(&(self.imu_angular_velocity * delta_t));
        self.orientation = quaternion_normalized(&quaternion_mul(&self.orientation, &rotation));
        self.gravity_vector =
            quaternion_rotate(&quaternion_conjugate(&rotation), &self.gravity_vector);
        self.time = time;
    }

    /// Updates from an IMU linear acceleration reading (in the IMU frame).
    pub fn add_imu_linear_acceleration_observation(
        &mut self,
        imu_linear_acceleration: &Vector3<f64>,
    ) {
        // Update the 'gravity_vector_' with an exponential moving average using
        // the 'imu_gravity_time_constant'.
        let delta_t = if self.last_linear_acceleration_time > Time::min() {
            to_seconds(self.time - self.last_linear_acceleration_time)
        } else {
            f64::INFINITY
        };
        self.last_linear_acceleration_time = self.time;
        let alpha = 1. - (-delta_t / self.imu_gravity_time_constant).exp();
        self.gravity_vector = (1. - alpha) * self.gravity_vector + alpha * imu_linear_acceleration;
        // Change the 'orientation_' so that it agrees with the current
        // 'gravity_vector_'.
        let rotation = from_two_vectors(
            &self.gravity_vector,
            &quaternion_rotate(&quaternion_conjugate(&self.orientation), &Vector3::z()),
        );
        self.orientation = quaternion_normalized(&quaternion_mul(&self.orientation, &rotation));
        let rotated_gravity = quaternion_rotate(&self.orientation, &self.gravity_vector);
        assert!(
            rotated_gravity.z > 0.,
            "Check failed: (orientation_ * gravity_vector_).z() > 0. ({} vs. 0)",
            rotated_gravity.z
        );
        assert!(
            rotated_gravity.normalize().z > 0.99,
            "Check failed: (orientation_ * gravity_vector_).normalized().z() > 0.99 ({} vs. 0.99)",
            rotated_gravity.normalize().z
        );
    }

    /// Updates from an IMU angular velocity reading (in the IMU frame).
    pub fn add_imu_angular_velocity_observation(&mut self, imu_angular_velocity: &Vector3<f64>) {
        self.imu_angular_velocity = *imu_angular_velocity;
    }

    /// Query the current time.
    pub fn time(&self) -> Time {
        self.time
    }

    /// Query the current orientation estimate.
    pub fn orientation(&self) -> Quaternion<f64> {
        self.orientation
    }
}

/// Eigen's `QuaternionBase::angularDistance` (test helper shared with the
/// pose extrapolator tests).
#[cfg(test)]
pub(crate) fn angular_distance(a: &Quaternion<f64>, b: &Quaternion<f64>) -> f64 {
    let d = crate::transform::rigid_transform::quaternion_dot(a, b).abs();
    if d >= 1. { 0. } else { 2. * d.acos() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{Duration, from_seconds, from_universal};
    use crate::transform::rigid_transform::AngleAxis;
    use proptest::prelude::*;

    mod imu_tracker_test {
        use super::*;

        const DURATION: f64 = 3.0;
        const GRAVITY_TIME_CONSTANT: f64 = 0.1 * DURATION;
        const PRECISION: f64 = 1e-8;
        const STEPS: i32 = 10;

        /// The C++ test fixture `ImuTrackerTest`.
        struct Fixture {
            angular_velocity: Vector3<f64>,
            imu_tracker: ImuTracker,
            linear_acceleration: Vector3<f64>,
            time: Time,
        }

        impl Fixture {
            fn set_up() -> Self {
                let time = from_universal(12345678);
                let fixture = Fixture {
                    imu_tracker: ImuTracker::new(GRAVITY_TIME_CONSTANT, time),
                    angular_velocity: Vector3::new(0., 0., 0.),
                    linear_acceleration: Vector3::new(0., 0., 9.9),
                    time,
                };
                assert!(
                    angular_distance(&fixture.imu_tracker.orientation(), &quaternion_identity())
                        .abs()
                        <= PRECISION
                );
                fixture
            }

            fn advance_imu(&mut self) {
                for _ in 0..STEPS {
                    self.imu_tracker
                        .add_imu_linear_acceleration_observation(&self.linear_acceleration);
                    self.imu_tracker
                        .add_imu_angular_velocity_observation(&self.angular_velocity);
                    self.time += from_seconds(DURATION / f64::from(STEPS));
                    self.imu_tracker.advance(self.time);
                }
                assert_eq!(self.time, self.imu_tracker.time());
            }
        }

        fn expected_orientation(angular_velocity: &Vector3<f64>) -> Quaternion<f64> {
            AngleAxis::new(
                DURATION * angular_velocity.norm(),
                angular_velocity.normalize(),
            )
            .to_quaternion()
        }

        #[test]
        fn integrate_yaw_rotation() {
            let mut f = Fixture::set_up();
            f.angular_velocity = Vector3::new(0., 0., 0.3);
            f.advance_imu();
            let expected = expected_orientation(&f.angular_velocity);
            assert!(angular_distance(&f.imu_tracker.orientation(), &expected) <= PRECISION);
        }

        #[test]
        fn integrate_full_rotation() {
            let mut f = Fixture::set_up();
            f.angular_velocity = Vector3::new(0.1, 0.4, 0.1);
            // Using a huge gravity time constant effectively disables gravity
            // vector tracking.
            f.imu_tracker = ImuTracker::new(1e10 * DURATION, f.time);
            f.advance_imu();
            let expected = expected_orientation(&f.angular_velocity);
            assert!(angular_distance(&f.imu_tracker.orientation(), &expected) <= PRECISION);
        }

        #[test]
        fn learn_gravity_vector() {
            let mut f = Fixture::set_up();
            f.linear_acceleration = Vector3::new(0.5, 0.3, 9.5);
            f.advance_imu();
            let expected = from_two_vectors(&f.linear_acceleration, &Vector3::z());
            assert!(angular_distance(&f.imu_tracker.orientation(), &expected) <= PRECISION);
        }
    }

    #[test]
    #[should_panic(expected = "time_ <= time")]
    fn advance_into_the_past_panics() {
        let mut tracker = ImuTracker::new(1., from_universal(100));
        tracker.advance(from_universal(99));
    }

    proptest! {
        /// With constant (non-rotating) acceleration measurements, the
        /// gravity estimate converges so that the orientation maps the
        /// measured acceleration direction onto +Z.
        #[test]
        fn gravity_converges(
            x in -3.0..3.0f64,
            y in -3.0..3.0f64,
            z in 5.0..12.0f64,
            time_constant in 0.05..1.0f64,
        ) {
            let acceleration = Vector3::new(x, y, z);
            let mut time = from_universal(1_000_000);
            let mut tracker = ImuTracker::new(time_constant, time);
            for _ in 0..200 {
                tracker.add_imu_linear_acceleration_observation(&acceleration);
                tracker.add_imu_angular_velocity_observation(&Vector3::zeros());
                time += Duration::from_seconds(0.1);
                tracker.advance(time);
            }
            let up = quaternion_rotate(&tracker.orientation(), &acceleration.normalize());
            prop_assert!((up - Vector3::z()).norm() < 1e-6, "{:?}", up);
        }
    }
}
