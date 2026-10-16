//! Port of `cartographer/mapping/internal/3d/imu_integration` (.h).
//!
//! RUST PORT NOTE: the C++ templates take a range (`std::deque<ImuData>`,
//! `MapByTime::Range`) and an in/out iterator into it. Here the range is any
//! [`ImuDataRange`] (slices, `Vec`, `VecDeque`) and the iterator is an index
//! (`&mut usize`) that is advanced in place, so callers can reuse it across
//! consecutive intervals exactly like the C++ iterator.

use std::collections::VecDeque;

use nalgebra::{Affine3, Point3, Quaternion, RealField, Vector3};

use crate::common::time::{Time, to_seconds};
use crate::sensor::imu_data::ImuData;
use crate::transform::rigid_transform::{
    Rigid3, quaternion_conjugate, quaternion_identity, quaternion_mul, quaternion_rotate,
};
use crate::transform::transform::angle_axis_vector_to_rotation_quaternion;

/// Random-access, time-ordered sequence of IMU measurements.
pub trait ImuDataRange {
    /// Number of measurements.
    fn len(&self) -> usize;
    /// Whether there are no measurements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The measurement at `index` (`index < len()`).
    fn at(&self, index: usize) -> &ImuData;
}

impl ImuDataRange for [ImuData] {
    fn len(&self) -> usize {
        <[ImuData]>::len(self)
    }
    fn at(&self, index: usize) -> &ImuData {
        &self[index]
    }
}

impl ImuDataRange for Vec<ImuData> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn at(&self, index: usize) -> &ImuData {
        &self[index]
    }
}

impl ImuDataRange for VecDeque<ImuData> {
    fn len(&self) -> usize {
        VecDeque::len(self)
    }
    fn at(&self, index: usize) -> &ImuData {
        &self[index]
    }
}

/// Result of [`integrate_imu`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrateImuResult<T: RealField + Copy> {
    /// Integrated change in velocity (in the frame at `start_time`).
    pub delta_velocity: Vector3<T>,
    /// Integrated change in translation (in the frame at `start_time`).
    pub delta_translation: Vector3<T>,
    /// Integrated rotation (not normalized, like the Eigen product).
    pub delta_rotation: Quaternion<T>,
}

#[inline]
fn from_f64<T: RealField + Copy>(x: f64) -> T {
    nalgebra::convert(x)
}

#[inline]
fn cast_vector<T: RealField + Copy>(v: &Vector3<f64>) -> Vector3<T> {
    v.map(from_f64)
}

/// `Eigen::Transform<T, 3, Affine> * Vector3`: applies the linear part and
/// the translation.
#[inline]
fn apply_affine<T: RealField + Copy>(transform: &Affine3<T>, v: &Vector3<T>) -> Vector3<T> {
    transform.transform_point(&Point3::from(*v)).coords
}

/// Integrates IMU data between `start_time` and `end_time` (zero-order hold,
/// forward Euler), applying the given calibrations to the measurements.
///
/// `it` must point to the last measurement at or before `start_time` (with
/// the next one, if any, strictly after it); it is advanced past consumed
/// measurements.
pub fn integrate_imu_with_calibration<T: RealField + Copy, R: ImuDataRange + ?Sized>(
    imu_data: &R,
    linear_acceleration_calibration: &Affine3<T>,
    angular_velocity_calibration: &Affine3<T>,
    start_time: Time,
    end_time: Time,
    it: &mut usize,
) -> IntegrateImuResult<T> {
    assert!(
        start_time <= end_time,
        "Check failed: start_time <= end_time ({start_time:?} vs. {end_time:?})"
    );
    assert!(*it < imu_data.len(), "Check failed: *it != imu_data.end()");
    assert!(
        imu_data.at(*it).time <= start_time,
        "Check failed: (*it)->time <= start_time"
    );
    if *it + 1 < imu_data.len() {
        assert!(
            imu_data.at(*it + 1).time > start_time,
            "Check failed: std::next(*it)->time > start_time"
        );
    }

    let mut current_time = start_time;

    let mut result = IntegrateImuResult {
        delta_velocity: Vector3::zeros(),
        delta_translation: Vector3::zeros(),
        delta_rotation: quaternion_identity(),
    };
    while current_time < end_time {
        let next_imu_data = if *it + 1 < imu_data.len() {
            imu_data.at(*it + 1).time
        } else {
            Time::max()
        };
        let next_time = next_imu_data.min(end_time);
        let delta_t: T = from_f64(to_seconds(next_time - current_time));

        let imu = imu_data.at(*it);
        let delta_angle = apply_affine(
            angular_velocity_calibration,
            &cast_vector(&imu.angular_velocity),
        ) * delta_t;
        result.delta_rotation = quaternion_mul(
            &result.delta_rotation,
            &angle_axis_vector_to_rotation_quaternion(&delta_angle),
        );
        result.delta_velocity += quaternion_rotate(
            &result.delta_rotation,
            &(apply_affine(
                linear_acceleration_calibration,
                &cast_vector(&imu.linear_acceleration),
            ) * delta_t),
        );
        result.delta_translation += result.delta_velocity * delta_t;
        current_time = next_time;
        if current_time == next_imu_data {
            *it += 1;
        }
    }
    result
}

/// [`integrate_imu_with_calibration`] with identity calibrations, in `f64`.
pub fn integrate_imu<R: ImuDataRange + ?Sized>(
    imu_data: &R,
    start_time: Time,
    end_time: Time,
    it: &mut usize,
) -> IntegrateImuResult<f64> {
    integrate_imu_with_calibration(
        imu_data,
        &Affine3::identity(),
        &Affine3::identity(),
        start_time,
        end_time,
        it,
    )
}

/// Result of [`extrapolate_pose_with_imu`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtrapolatePoseResult<T: RealField + Copy> {
    /// Extrapolated pose.
    pub pose: Rigid3<T>,
    /// Linear velocity in the parent frame of the pose (note: the input
    /// velocity of [`extrapolate_pose_with_imu`] is in the tracking frame).
    pub velocity: Vector3<T>,
}

/// Returns pose and linear velocity at `time` which is equal to
/// `prev_from_tracking` extrapolated using IMU data.
#[allow(clippy::too_many_arguments)]
pub fn extrapolate_pose_with_imu<T: RealField + Copy, R: ImuDataRange + ?Sized>(
    prev_from_tracking: &Rigid3<T>,
    prev_velocity_in_tracking: &Vector3<T>,
    prev_time: Time,
    gravity: &Vector3<T>,
    time: Time,
    imu_data: &R,
    imu_it: &mut usize,
) -> ExtrapolatePoseResult<T> {
    let result = integrate_imu_with_calibration(
        imu_data,
        &Affine3::<T>::identity(),
        &Affine3::<T>::identity(),
        prev_time,
        time,
        imu_it,
    );

    let delta_t: T = from_f64(to_seconds(time - prev_time));
    let prev_rotation = prev_from_tracking.rotation();
    let translation = prev_from_tracking.translation()
        + quaternion_rotate(
            prev_rotation,
            &(prev_velocity_in_tracking * delta_t + result.delta_translation),
        )
        - gravity * (from_f64::<T>(0.5) * delta_t * delta_t);
    let rotation = quaternion_mul(prev_rotation, &result.delta_rotation);

    let velocity = quaternion_rotate(
        prev_rotation,
        &(prev_velocity_in_tracking + result.delta_velocity),
    ) - gravity * delta_t;
    ExtrapolatePoseResult {
        pose: Rigid3::new(translation, rotation),
        velocity,
    }
}

/// Same as [`extrapolate_pose_with_imu`] but given the last two poses (the
/// previous velocity is estimated from them).
#[allow(clippy::too_many_arguments)]
pub fn extrapolate_pose_with_imu_from_poses<T: RealField + Copy, R: ImuDataRange + ?Sized>(
    prev_from_tracking: &Rigid3<T>,
    prev_time: Time,
    prev_prev_from_tracking: &Rigid3<T>,
    prev_prev_time: Time,
    gravity: &Vector3<T>,
    time: Time,
    imu_data: &R,
    imu_it: &mut usize,
) -> ExtrapolatePoseResult<T> {
    // TODO(danielsievers): Really we should integrate velocities starting from
    // the midpoint in between two poses, since this is how we fit them to
    // poses in the optimization.
    let prev_delta_t: T = from_f64(to_seconds(prev_time - prev_prev_time));
    // C++: `prev_from_tracking.inverse().rotation() * (dp) / dt`.
    let prev_velocity_in_tracking = quaternion_rotate(
        &quaternion_conjugate(prev_from_tracking.rotation()),
        &(prev_from_tracking.translation() - prev_prev_from_tracking.translation()),
    ) / prev_delta_t;

    extrapolate_pose_with_imu(
        prev_from_tracking,
        &prev_velocity_in_tracking,
        prev_time,
        gravity,
        time,
        imu_data,
        imu_it,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use nalgebra::{Affine3, Matrix4, Vector3};

    use super::*;
    use crate::common::time::{Time, from_seconds, from_universal};
    use crate::transform::rigid_transform::{AngleAxis, Rigid3d};
    use crate::transform::transform::get_angle;

    fn t(seconds: f64) -> Time {
        from_universal(1_000_000_000) + from_seconds(seconds)
    }

    fn imu(
        time: f64,
        linear_acceleration: Vector3<f64>,
        angular_velocity: Vector3<f64>,
    ) -> ImuData {
        ImuData {
            time: t(time),
            linear_acceleration,
            angular_velocity,
        }
    }

    fn assert_vec_near(a: &Vector3<f64>, b: &Vector3<f64>, eps: f64) {
        assert!((a - b).norm() <= eps, "{a:?} vs {b:?}");
    }

    #[test]
    fn constant_angular_velocity_integrates_rotation() {
        let w = Vector3::new(0.1, -0.2, 0.3);
        let imu_data: Vec<ImuData> = (0..10)
            .map(|i| imu(0.1 * f64::from(i), Vector3::zeros(), w))
            .collect();
        let mut it = 0;
        let result = integrate_imu(&imu_data, t(0.0), t(0.95), &mut it);
        assert_vec_near(&result.delta_velocity, &Vector3::zeros(), 1e-12);
        assert_vec_near(&result.delta_translation, &Vector3::zeros(), 1e-12);
        let expected = angle_axis_vector_to_rotation_quaternion(&(w * 0.95));
        let actual = Rigid3d::from_rotation(result.delta_rotation);
        let expected = Rigid3d::from_rotation(expected);
        assert!(get_angle(&(actual.inverse() * expected)) < 1e-9);
        assert_eq!(it, 9);
    }

    #[test]
    fn forward_euler_translation() {
        // dv is updated before dp, so with n steps of dt:
        // dv = a n dt, dp = a dt^2 n (n + 1) / 2.
        let a = Vector3::new(1.0, 2.0, -3.0);
        let n = 5;
        let dt = 0.1;
        let imu_data: VecDeque<ImuData> = (0..n)
            .map(|i| imu(dt * f64::from(i), a, Vector3::zeros()))
            .collect();
        let mut it = 0;
        let result = integrate_imu(&imu_data, t(0.0), t(dt * f64::from(n)), &mut it);
        let nf = f64::from(n);
        assert_vec_near(&result.delta_velocity, &(a * (nf * dt)), 1e-9);
        assert_vec_near(
            &result.delta_translation,
            &(a * (dt * dt * nf * (nf + 1.0) / 2.0)),
            1e-9,
        );
        // Only the last sample is held beyond the data.
        assert_eq!(it, (n - 1) as usize);
    }

    #[test]
    fn iterator_is_advanced_and_reusable() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let imu_data: Vec<ImuData> = (0..4)
            .map(|i| imu(0.1 * f64::from(i), a, Vector3::zeros()))
            .collect();
        let mut it = 0;
        let first = integrate_imu(&imu_data, t(0.0), t(0.25), &mut it);
        assert_eq!(it, 2);
        let second = integrate_imu(&imu_data, t(0.25), t(0.5), &mut it);
        assert_eq!(it, 3);
        let mut it_all = 0;
        let all = integrate_imu(&imu_data, t(0.0), t(0.5), &mut it_all);
        assert_vec_near(
            &(first.delta_velocity + second.delta_velocity),
            &all.delta_velocity,
            1e-12,
        );
        // Zero-length interval: nothing integrated, iterator unchanged.
        let empty = integrate_imu(&imu_data, t(0.5), t(0.5), &mut it);
        assert_eq!(empty.delta_velocity, Vector3::zeros());
        assert_eq!(it, 3);
    }

    #[test]
    #[should_panic(expected = "(*it)->time <= start_time")]
    fn iterator_after_start_panics() {
        let imu_data = vec![imu(0.1, Vector3::zeros(), Vector3::zeros())];
        let mut it = 0;
        integrate_imu(&imu_data, t(0.0), t(0.5), &mut it);
    }

    #[test]
    #[should_panic(expected = "std::next(*it)->time > start_time")]
    fn stale_iterator_panics() {
        let imu_data = vec![
            imu(0.0, Vector3::zeros(), Vector3::zeros()),
            imu(0.1, Vector3::zeros(), Vector3::zeros()),
        ];
        let mut it = 0;
        integrate_imu(&imu_data, t(0.2), t(0.5), &mut it);
    }

    #[test]
    fn calibration_is_applied_as_affine_transform() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let imu_data = vec![imu(0.0, a, Vector3::zeros())];
        // Scale by 2 and add an offset of (0, 0, 1).
        let calibration = Affine3::from_matrix_unchecked(Matrix4::new(
            2.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 1.0, //
            0.0, 0.0, 0.0, 1.0,
        ));
        let mut it = 0;
        let result = integrate_imu_with_calibration(
            imu_data.as_slice(),
            &calibration,
            &Affine3::identity(),
            t(0.0),
            t(1.0),
            &mut it,
        );
        assert_vec_near(&result.delta_velocity, &Vector3::new(2.0, 0.0, 1.0), 1e-12);
    }

    #[test]
    fn extrapolate_stationary_with_gravity() {
        // A stationary IMU measures +g; extrapolating with gravity keeps the
        // velocity at zero, and the translation error is the forward Euler
        // bias 0.5 g T^2 / n.
        let g = Vector3::new(0.0, 0.0, 9.81);
        let n = 10;
        let dt = 0.01;
        let imu_data: Vec<ImuData> = (0..n)
            .map(|i| imu(dt * f64::from(i), g, Vector3::zeros()))
            .collect();
        let mut it = 0;
        let total = dt * f64::from(n);
        let result = extrapolate_pose_with_imu(
            &Rigid3d::identity(),
            &Vector3::zeros(),
            t(0.0),
            &g,
            t(total),
            &imu_data,
            &mut it,
        );
        assert_vec_near(&result.velocity, &Vector3::zeros(), 1e-9);
        assert_vec_near(
            result.pose.translation(),
            &(g * (0.5 * total * total / f64::from(n))),
            1e-9,
        );
    }

    #[test]
    fn extrapolate_velocity_is_in_parent_frame() {
        let g = Vector3::new(0.0, 0.0, 9.81);
        let imu_data = vec![imu(0.0, g, Vector3::zeros())];
        // Yaw of 90 degrees: tracking x is parent y.
        let prev = Rigid3d::with_angle_axis(
            Vector3::new(1.0, 2.0, 3.0),
            AngleAxis::new(std::f64::consts::FRAC_PI_2, Vector3::z()),
        );
        let v_tracking = Vector3::new(1.0, 0.0, 0.0);
        let mut it = 0;
        let result =
            extrapolate_pose_with_imu(&prev, &v_tracking, t(0.0), &g, t(0.5), &imu_data, &mut it);
        assert_vec_near(&result.velocity, &Vector3::new(0.0, 1.0, 0.0), 1e-9);
        // translation = p + R (v T + g T^2) - 0.5 g T^2 (one Euler step).
        assert_vec_near(
            result.pose.translation(),
            &(Vector3::new(1.0, 2.5, 3.0) + g * (0.25 - 0.125)),
            1e-9,
        );
        assert!(get_angle(&(prev.inverse() * result.pose)) < 1e-12);

        // Two-pose form: the previous velocity is estimated from the poses.
        let prev_prev = Rigid3d::with_angle_axis(
            Vector3::new(1.0, 1.0, 3.0),
            AngleAxis::new(std::f64::consts::FRAC_PI_2, Vector3::z()),
        );
        let mut it = 0;
        let from_poses = extrapolate_pose_with_imu_from_poses(
            &prev,
            t(0.0),
            &prev_prev,
            t(-1.0),
            &g,
            t(0.5),
            &imu_data,
            &mut it,
        );
        assert_vec_near(&from_poses.velocity, &result.velocity, 1e-9);
        assert_vec_near(
            from_poses.pose.translation(),
            result.pose.translation(),
            1e-9,
        );
    }
}
