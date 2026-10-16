//! Port of `cartographer/mapping/pose_extrapolator` (.h, .cc).

use std::collections::VecDeque;

use nalgebra::{Quaternion, Vector3};

use crate::common::time::{Duration, Time, to_seconds};
use crate::mapping::imu_tracker::ImuTracker;
use crate::mapping::pose_extrapolator_interface::{ExtrapolationResult, PoseExtrapolatorInterface};
use crate::sensor::imu_data::ImuData;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{
    Rigid3d, quaternion_conjugate, quaternion_mul, quaternion_rotate,
};
use crate::transform::transform::rotation_quaternion_to_angle_axis_vector;

/// A pose at a point in time.
#[derive(Clone, Copy, Debug)]
struct TimedPose {
    time: Time,
    pose: Rigid3d,
}

/// Keep poses for a certain duration to estimate linear and angular velocity.
/// Uses the velocities to extrapolate motion. Uses IMU and/or odometry data if
/// available to improve the extrapolation.
#[derive(Debug)]
pub struct PoseExtrapolator {
    pose_queue_duration: Duration,
    timed_pose_queue: VecDeque<TimedPose>,
    linear_velocity_from_poses: Vector3<f64>,
    angular_velocity_from_poses: Vector3<f64>,

    gravity_time_constant: f64,
    imu_data: VecDeque<ImuData>,
    imu_tracker: Option<ImuTracker>,
    odometry_imu_tracker: Option<ImuTracker>,
    extrapolation_imu_tracker: Option<ImuTracker>,
    cached_extrapolated_pose: TimedPose,

    odometry_data: VecDeque<OdometryData>,
    linear_velocity_from_odometry: Vector3<f64>,
    angular_velocity_from_odometry: Vector3<f64>,
}

impl PoseExtrapolator {
    /// Creates an extrapolator that estimates velocities from the poses of
    /// the last `pose_queue_duration`.
    pub fn new(pose_queue_duration: Duration, imu_gravity_time_constant: f64) -> Self {
        PoseExtrapolator {
            pose_queue_duration,
            timed_pose_queue: VecDeque::new(),
            linear_velocity_from_poses: Vector3::zeros(),
            angular_velocity_from_poses: Vector3::zeros(),
            gravity_time_constant: imu_gravity_time_constant,
            imu_data: VecDeque::new(),
            imu_tracker: None,
            odometry_imu_tracker: None,
            extrapolation_imu_tracker: None,
            cached_extrapolated_pose: TimedPose {
                time: Time::min(),
                pose: Rigid3d::identity(),
            },
            odometry_data: VecDeque::new(),
            linear_velocity_from_odometry: Vector3::zeros(),
            angular_velocity_from_odometry: Vector3::zeros(),
        }
    }

    /// Creates an extrapolator whose first pose is the gravity-aligned
    /// orientation measured by `imu_data`.
    pub fn initialize_with_imu(
        pose_queue_duration: Duration,
        imu_gravity_time_constant: f64,
        imu_data: &ImuData,
    ) -> Self {
        let mut extrapolator =
            PoseExtrapolator::new(pose_queue_duration, imu_gravity_time_constant);
        extrapolator.add_imu_data(imu_data);
        let mut imu_tracker = ImuTracker::new(imu_gravity_time_constant, imu_data.time);
        imu_tracker.add_imu_linear_acceleration_observation(&imu_data.linear_acceleration);
        imu_tracker.add_imu_angular_velocity_observation(&imu_data.angular_velocity);
        imu_tracker.advance(imu_data.time);
        let orientation = imu_tracker.orientation();
        extrapolator.imu_tracker = Some(imu_tracker);
        extrapolator.add_pose(imu_data.time, &Rigid3d::from_rotation(orientation));
        extrapolator
    }

    fn imu_tracker(&self) -> &ImuTracker {
        self.imu_tracker
            .as_ref()
            .expect("PoseExtrapolator: no pose was added yet (imu_tracker_ is null)")
    }

    fn newest_timed_pose(&self) -> &TimedPose {
        self.timed_pose_queue
            .back()
            .expect("PoseExtrapolator: no pose was added yet (timed_pose_queue_ is empty)")
    }

    fn update_velocities_from_poses(&mut self) {
        if self.timed_pose_queue.len() < 2 {
            // We need two poses to estimate velocities.
            return;
        }
        assert!(!self.timed_pose_queue.is_empty());
        let newest_timed_pose = *self.newest_timed_pose();
        let newest_time = newest_timed_pose.time;
        let oldest_timed_pose = self.timed_pose_queue[0];
        let oldest_time = oldest_timed_pose.time;
        let queue_delta = to_seconds(newest_time - oldest_time);
        if queue_delta < to_seconds(self.pose_queue_duration) {
            log::warn!("Queue too short for velocity estimation. Queue duration: {queue_delta} s");
            return;
        }
        let newest_pose = &newest_timed_pose.pose;
        let oldest_pose = &oldest_timed_pose.pose;
        self.linear_velocity_from_poses =
            (newest_pose.translation() - oldest_pose.translation()) / queue_delta;
        self.angular_velocity_from_poses =
            rotation_quaternion_to_angle_axis_vector(&quaternion_mul(
                &quaternion_conjugate(oldest_pose.rotation()),
                newest_pose.rotation(),
            )) / queue_delta;
    }

    fn trim_imu_data(&mut self) {
        while self.imu_data.len() > 1
            && self
                .timed_pose_queue
                .back()
                .is_some_and(|newest| self.imu_data[1].time <= newest.time)
        {
            self.imu_data.pop_front();
        }
    }

    fn trim_odometry_data(&mut self) {
        while self.odometry_data.len() > 2
            && self
                .timed_pose_queue
                .back()
                .is_some_and(|newest| self.odometry_data[1].time <= newest.time)
        {
            self.odometry_data.pop_front();
        }
    }

    fn advance_imu_tracker(&self, time: Time, imu_tracker: &mut ImuTracker) {
        assert!(
            time >= imu_tracker.time(),
            "Check failed: time >= imu_tracker->time() ({} vs. {})",
            time,
            imu_tracker.time()
        );
        if self.imu_data.is_empty() || time < self.imu_data[0].time {
            // There is no IMU data until 'time', so we advance the ImuTracker
            // and use the angular velocities from poses and fake gravity to
            // help 2D stability.
            imu_tracker.advance(time);
            imu_tracker.add_imu_linear_acceleration_observation(&Vector3::z());
            imu_tracker.add_imu_angular_velocity_observation(if self.odometry_data.len() < 2 {
                &self.angular_velocity_from_poses
            } else {
                &self.angular_velocity_from_odometry
            });
            return;
        }
        if imu_tracker.time() < self.imu_data[0].time {
            // Advance to the beginning of 'imu_data_'.
            imu_tracker.advance(self.imu_data[0].time);
        }
        let tracker_time = imu_tracker.time();
        let start = self
            .imu_data
            .partition_point(|imu_data| imu_data.time < tracker_time);
        for imu_data in self.imu_data.range(start..) {
            if imu_data.time >= time {
                break;
            }
            imu_tracker.advance(imu_data.time);
            imu_tracker.add_imu_linear_acceleration_observation(&imu_data.linear_acceleration);
            imu_tracker.add_imu_angular_velocity_observation(&imu_data.angular_velocity);
        }
        imu_tracker.advance(time);
    }

    fn extrapolate_rotation(&self, time: Time, imu_tracker: &mut ImuTracker) -> Quaternion<f64> {
        assert!(
            time >= imu_tracker.time(),
            "Check failed: time >= imu_tracker->time() ({} vs. {})",
            time,
            imu_tracker.time()
        );
        self.advance_imu_tracker(time, imu_tracker);
        let last_orientation = self.imu_tracker().orientation();
        // RUST PORT NOTE: Eigen's `inverse()` of a unit quaternion is its
        // conjugate divided by the squared norm (== 1 up to rounding).
        quaternion_mul(
            &quaternion_inverse(&last_orientation),
            &imu_tracker.orientation(),
        )
    }

    fn extrapolate_translation(&self, time: Time) -> Vector3<f64> {
        let newest_timed_pose = self.newest_timed_pose();
        let extrapolation_delta = to_seconds(time - newest_timed_pose.time);
        if self.odometry_data.len() < 2 {
            return extrapolation_delta * self.linear_velocity_from_poses;
        }
        extrapolation_delta * self.linear_velocity_from_odometry
    }

    /// Runs `f` on the tracker stored in `slot` while `self` is borrowed
    /// immutably.
    ///
    /// RUST PORT NOTE: the C++ passes raw `ImuTracker*` members of `this` into
    /// const member functions. Here the tracker is temporarily moved out of
    /// its slot to satisfy the borrow checker.
    fn with_tracker<R>(
        &mut self,
        slot: fn(&mut Self) -> &mut Option<ImuTracker>,
        f: impl FnOnce(&Self, &mut ImuTracker) -> R,
    ) -> R {
        let mut tracker = slot(self)
            .take()
            .expect("PoseExtrapolator: no pose was added yet (imu tracker is null)");
        let result = f(self, &mut tracker);
        *slot(self) = Some(tracker);
        result
    }
}

/// Eigen's `Quaternion::inverse()`: conjugate / squared norm (zero if the
/// norm is zero).
fn quaternion_inverse(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = crate::transform::rigid_transform::quaternion_squared_norm(q);
    if n2 > 0. {
        let c = quaternion_conjugate(q);
        Quaternion::new(c.w / n2, c.i / n2, c.j / n2, c.k / n2)
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

impl PoseExtrapolatorInterface for PoseExtrapolator {
    fn get_last_pose_time(&self) -> Time {
        match self.timed_pose_queue.back() {
            None => Time::min(),
            Some(newest) => newest.time,
        }
    }

    fn get_last_extrapolated_time(&self) -> Time {
        match &self.extrapolation_imu_tracker {
            None => Time::min(),
            Some(tracker) => tracker.time(),
        }
    }

    fn add_pose(&mut self, time: Time, pose: &Rigid3d) {
        if self.imu_tracker.is_none() {
            let mut tracker_start = time;
            if let Some(front) = self.imu_data.front() {
                tracker_start = tracker_start.min(front.time);
            }
            self.imu_tracker = Some(ImuTracker::new(self.gravity_time_constant, tracker_start));
        }
        self.timed_pose_queue
            .push_back(TimedPose { time, pose: *pose });
        while self.timed_pose_queue.len() > 2
            && self.timed_pose_queue[1].time <= time - self.pose_queue_duration
        {
            self.timed_pose_queue.pop_front();
        }
        self.update_velocities_from_poses();
        self.with_tracker(
            |s| &mut s.imu_tracker,
            |s, tracker| s.advance_imu_tracker(time, tracker),
        );
        self.trim_imu_data();
        self.trim_odometry_data();
        self.odometry_imu_tracker = self.imu_tracker.clone();
        self.extrapolation_imu_tracker = self.imu_tracker.clone();
    }

    fn add_imu_data(&mut self, imu_data: &ImuData) {
        assert!(
            self.timed_pose_queue
                .back()
                .is_none_or(|newest| imu_data.time >= newest.time),
            "Check failed: timed_pose_queue_.empty() || imu_data.time >= timed_pose_queue_.back().time"
        );
        self.imu_data.push_back(*imu_data);
        self.trim_imu_data();
    }

    fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        assert!(
            self.timed_pose_queue
                .back()
                .is_none_or(|newest| odometry_data.time >= newest.time),
            "Check failed: timed_pose_queue_.empty() || odometry_data.time >= timed_pose_queue_.back().time"
        );
        self.odometry_data.push_back(*odometry_data);
        self.trim_odometry_data();
        if self.odometry_data.len() < 2 {
            return;
        }
        // TODO(whess): Improve by using more than just the last two odometry
        // poses.
        // Compute extrapolation in the tracking frame.
        let odometry_data_oldest = self.odometry_data[0];
        let odometry_data_newest = *self.odometry_data.back().expect("len >= 2");
        let odometry_time_delta = to_seconds(odometry_data_oldest.time - odometry_data_newest.time);
        let odometry_pose_delta = odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
        self.angular_velocity_from_odometry =
            rotation_quaternion_to_angle_axis_vector(odometry_pose_delta.rotation())
                / odometry_time_delta;
        let Some(newest_timed_pose) = self.timed_pose_queue.back().copied() else {
            return;
        };
        let linear_velocity_in_tracking_frame_at_newest_odometry_time =
            odometry_pose_delta.translation() / odometry_time_delta;
        let extrapolated_rotation = self.with_tracker(
            |s| &mut s.odometry_imu_tracker,
            |s, tracker| s.extrapolate_rotation(odometry_data_newest.time, tracker),
        );
        let orientation_at_newest_odometry_time =
            quaternion_mul(newest_timed_pose.pose.rotation(), &extrapolated_rotation);
        self.linear_velocity_from_odometry = quaternion_rotate(
            &orientation_at_newest_odometry_time,
            &linear_velocity_in_tracking_frame_at_newest_odometry_time,
        );
    }

    fn extrapolate_pose(&mut self, time: Time) -> Rigid3d {
        let newest_timed_pose = *self.newest_timed_pose();
        assert!(
            time >= newest_timed_pose.time,
            "Check failed: time >= newest_timed_pose.time ({} vs. {})",
            time,
            newest_timed_pose.time
        );
        if self.cached_extrapolated_pose.time != time {
            let translation =
                self.extrapolate_translation(time) + newest_timed_pose.pose.translation();
            let extrapolated_rotation = self.with_tracker(
                |s| &mut s.extrapolation_imu_tracker,
                |s, tracker| s.extrapolate_rotation(time, tracker),
            );
            let rotation =
                quaternion_mul(newest_timed_pose.pose.rotation(), &extrapolated_rotation);
            self.cached_extrapolated_pose = TimedPose {
                time,
                pose: Rigid3d::new(translation, rotation),
            };
        }
        self.cached_extrapolated_pose.pose
    }

    fn extrapolate_poses_with_gravity(&mut self, times: &[Time]) -> ExtrapolationResult {
        let (&last_time, previous_times) = times
            .split_last()
            .expect("ExtrapolatePosesWithGravity requires at least one time");
        let previous_poses = previous_times
            .iter()
            .map(|&time| self.extrapolate_pose(time).cast::<f32>())
            .collect();
        let current_velocity = if self.odometry_data.len() < 2 {
            self.linear_velocity_from_poses
        } else {
            self.linear_velocity_from_odometry
        };
        ExtrapolationResult {
            previous_poses,
            current_pose: self.extrapolate_pose(last_time),
            current_velocity,
            gravity_from_tracking: self.estimate_gravity_orientation(last_time),
        }
    }

    fn estimate_gravity_orientation(&mut self, time: Time) -> Quaternion<f64> {
        let mut imu_tracker = self.imu_tracker().clone();
        self.advance_imu_tracker(time, &mut imu_tracker);
        imu_tracker.orientation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{from_seconds, from_universal, to_universal};
    use crate::mapping::imu_tracker::angular_distance;
    use crate::mapping::internal::eigen_quaterniond_from_two_vectors::from_two_vectors;
    use crate::transform::rigid_transform::AngleAxis;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use proptest::prelude::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    // The C++ constants are float literals stored in doubles.
    const POSE_QUEUE_DURATION: f64 = 0.5f32 as f64;
    const GRAVITY_TIME_CONSTANT: f64 = 0.1f32 as f64;
    const EXTRAPOLATE_DURATION: f64 = 0.1f32 as f64;
    const PRECISION: f64 = 1e-8;
    const EXTRAPOLATE_PRECISION: f64 = 1e-2;

    /// C++ `EXPECT_DEATH(statement, "")`. The C++ runs the statement in a
    /// forked child, so it cannot affect the parent's state; here callers
    /// pass a value that is consumed by the statement.
    #[track_caller]
    fn expect_death<T>(value: T, statement: impl FnOnce(T)) {
        let result = catch_unwind(AssertUnwindSafe(move || statement(value)));
        assert!(result.is_err(), "statement did not panic");
    }

    fn new_extrapolator() -> PoseExtrapolator {
        PoseExtrapolator::new(from_seconds(POSE_QUEUE_DURATION), GRAVITY_TIME_CONSTANT)
    }

    fn angle_axis_of(angular_velocity: &Vector3<f64>, scale: f64) -> AngleAxis<f64> {
        AngleAxis::new(
            scale * angular_velocity.norm(),
            angular_velocity.normalize(),
        )
    }

    mod pose_extrapolator_death_test {
        use super::*;

        #[test]
        fn incomplete_initialization() {
            let time = from_universal(123);
            expect_death(new_extrapolator(), |mut e| {
                e.estimate_gravity_orientation(time);
            });
            expect_death(new_extrapolator(), |mut e| {
                e.extrapolate_pose(time);
            });
            let with_imu = || {
                let mut extrapolator = new_extrapolator();
                let acceleration = Vector3::new(0., 0., 9.);
                let angular_velocity = Vector3::new(0., 0., 0.);
                extrapolator.add_imu_data(&ImuData {
                    time,
                    linear_acceleration: acceleration,
                    angular_velocity,
                });
                extrapolator
            };
            expect_death(with_imu(), |mut e| {
                e.estimate_gravity_orientation(time);
            });
            expect_death(with_imu(), |mut e| {
                e.extrapolate_pose(time);
            });
        }

        #[test]
        fn extrapolate_in_past() {
            let time_present = from_universal(123);
            let pose = Rigid3d::from_translation(Vector3::new(0.1, 0.2, 0.3));
            let set_up = || {
                let mut extrapolator = new_extrapolator();
                extrapolator.add_pose(time_present, &pose);
                extrapolator.estimate_gravity_orientation(time_present);
                extrapolator
            };
            let mut extrapolator = set_up();
            assert_is_nearly(
                &extrapolator.extrapolate_pose(time_present),
                &pose,
                PRECISION,
            );
            let time_in_past = time_present - from_seconds(10.);
            expect_death(set_up(), |mut e| {
                e.estimate_gravity_orientation(time_in_past);
            });
            expect_death(set_up(), |mut e| {
                e.extrapolate_pose(time_in_past);
            });
            let time_in_future = time_present + from_seconds(20.);
            extrapolator.extrapolate_pose(time_in_future);
            expect_death(extrapolator, |mut e| {
                e.extrapolate_pose(time_present);
            });
        }
    }

    mod pose_extrapolator_test {
        use super::*;

        #[test]
        fn estimate_gravity_orientation_with_imu() {
            let initial_gravity_acceleration = Vector3::new(1.6, 2.0, 8.0);
            let angular_velocity = Vector3::new(0., 0., 0.);
            let mut current_time = from_universal(123);
            let imu_data = ImuData {
                time: current_time,
                linear_acceleration: initial_gravity_acceleration,
                angular_velocity,
            };
            let mut extrapolator = PoseExtrapolator::initialize_with_imu(
                from_seconds(POSE_QUEUE_DURATION),
                GRAVITY_TIME_CONSTANT,
                &imu_data,
            );
            let mut expected_orientation =
                from_two_vectors(&initial_gravity_acceleration, &Vector3::z());
            assert!(
                angular_distance(
                    &extrapolator.estimate_gravity_orientation(current_time),
                    &expected_orientation
                ) <= PRECISION
            );
            let gravity_acceleration = Vector3::new(1.6, 2.0, 8.0);
            for _ in 0..10 {
                current_time += from_seconds(GRAVITY_TIME_CONSTANT);
                extrapolator.add_imu_data(&ImuData {
                    time: current_time,
                    linear_acceleration: gravity_acceleration,
                    angular_velocity,
                });
            }
            expected_orientation = from_two_vectors(&gravity_acceleration, &Vector3::z());
            assert!(
                angular_distance(
                    &extrapolator.estimate_gravity_orientation(current_time),
                    &expected_orientation
                ) <= PRECISION
            );
        }

        #[test]
        fn extrapolate_with_poses() {
            let mut extrapolator = new_extrapolator();
            let mut current_time = from_universal(123);
            let mut current_pose = Rigid3d::from_translation(Vector3::new(0.3, 0.7, 0.2));
            let velocity = Vector3::new(0., 0.1, 0.);
            let angular_velocity = Vector3::new(0.01, 0., 0.1);
            let motion_per_second =
                Rigid3d::with_angle_axis(velocity, angle_axis_of(&angular_velocity, 1.));
            extrapolator.add_pose(current_time, &current_pose);
            assert_eq!(
                to_universal(extrapolator.get_last_pose_time()),
                to_universal(current_time)
            );
            assert_is_nearly(
                &extrapolator.extrapolate_pose(current_time),
                &current_pose,
                PRECISION,
            );
            for _ in 0..5 {
                current_time += from_seconds(1.);
                current_pose = current_pose * motion_per_second;
                extrapolator.add_pose(current_time, &current_pose);
                assert_is_nearly(
                    &extrapolator.extrapolate_pose(current_time),
                    &current_pose,
                    PRECISION,
                );
                let expected_pose = current_pose
                    * Rigid3d::with_angle_axis(
                        EXTRAPOLATE_DURATION * velocity,
                        angle_axis_of(&angular_velocity, EXTRAPOLATE_DURATION),
                    );
                assert_is_nearly(
                    &extrapolator
                        .extrapolate_pose(current_time + from_seconds(EXTRAPOLATE_DURATION)),
                    &expected_pose,
                    EXTRAPOLATE_PRECISION,
                );
                assert_eq!(
                    to_universal(extrapolator.get_last_extrapolated_time()),
                    to_universal(current_time + from_seconds(EXTRAPOLATE_DURATION))
                );
            }
            let gravity_axis = AngleAxis::from_quaternion(
                &extrapolator.estimate_gravity_orientation(current_time),
            );
            assert!((gravity_axis.axis.normalize() - Vector3::z()).norm() <= EXTRAPOLATE_PRECISION);
        }

        #[test]
        fn extrapolate_with_imu() {
            let initial_gravity_acceleration = Vector3::new(0., 0., 9.8);
            let initial_angular_velocity = Vector3::new(0., 0., 0.);
            let mut current_time = from_universal(123);
            let imu_data = ImuData {
                time: current_time,
                linear_acceleration: initial_gravity_acceleration,
                angular_velocity: initial_angular_velocity,
            };
            let mut extrapolator = PoseExtrapolator::initialize_with_imu(
                from_seconds(POSE_QUEUE_DURATION),
                GRAVITY_TIME_CONSTANT,
                &imu_data,
            );
            let current_pose = Rigid3d::from_translation(Vector3::new(0.3, 0.7, 0.2));
            // Let velocity estimation come to rest
            // (C++: `for (int i = 0; i < kPoseQueueDuration + 2; ++i)`, i.e. 3
            // iterations).
            let mut i = 0;
            while f64::from(i) < POSE_QUEUE_DURATION + 2. {
                current_time += from_seconds(1.);
                extrapolator.add_imu_data(&ImuData {
                    time: current_time,
                    linear_acceleration: initial_gravity_acceleration,
                    angular_velocity: initial_angular_velocity,
                });
                extrapolator.add_pose(current_time, &current_pose);
                i += 1;
            }
            assert_eq!(i, 3);
            let angular_velocity_yaw = Vector3::new(0., 0., 0.1);
            let mut expected_pose = current_pose;
            for _ in 0..3 {
                for _ in 0..10 {
                    current_time += from_seconds(GRAVITY_TIME_CONSTANT);
                    extrapolator.add_imu_data(&ImuData {
                        time: current_time,
                        linear_acceleration: initial_gravity_acceleration,
                        angular_velocity: angular_velocity_yaw,
                    });
                    let expected_rotation =
                        angle_axis_of(&angular_velocity_yaw, GRAVITY_TIME_CONSTANT);
                    expected_pose = expected_pose * Rigid3d::from_angle_axis(expected_rotation);
                }
                assert_is_nearly(
                    &extrapolator.extrapolate_pose(current_time),
                    &expected_pose,
                    EXTRAPOLATE_PRECISION,
                );
                extrapolator.add_pose(current_time, &expected_pose);
            }
            let gravity_axis = AngleAxis::from_quaternion(
                &extrapolator.estimate_gravity_orientation(current_time),
            );
            assert!((gravity_axis.axis.normalize() - Vector3::z()).norm() <= PRECISION);
        }
    }

    #[test]
    #[should_panic]
    fn extrapolate_poses_with_gravity_requires_times() {
        let mut extrapolator = new_extrapolator();
        extrapolator.add_pose(from_universal(0), &Rigid3d::identity());
        extrapolator.extrapolate_poses_with_gravity(&[]);
    }

    #[test]
    fn extrapolate_poses_with_gravity_matches_extrapolate_pose() {
        let mut extrapolator = new_extrapolator();
        let velocity = Vector3::new(1., -0.5, 0.25);
        for k in 0..4 {
            let t = from_universal(0) + from_seconds(f64::from(k));
            extrapolator.add_pose(t, &Rigid3d::from_translation(velocity * f64::from(k)));
        }
        let base = from_universal(0) + from_seconds(3.);
        let times: Vec<Time> = (1..=3)
            .map(|k| base + from_seconds(0.1 * f64::from(k)))
            .collect();
        let result = extrapolator.extrapolate_poses_with_gravity(&times);
        assert_eq!(result.previous_poses.len(), 2);
        assert!((result.current_velocity - velocity).norm() < 1e-9);
        let expected = Rigid3d::from_translation(velocity * (3. + 0.3));
        assert_is_nearly(&result.current_pose, &expected, 1e-9);
        assert_is_nearly(
            &result.previous_poses[0],
            &Rigid3d::from_translation(velocity * 3.1).cast::<f32>(),
            1e-6,
        );
    }

    #[test]
    fn odometry_velocity_is_used_for_translation() {
        let mut extrapolator = new_extrapolator();
        let t0 = from_universal(1_000_000);
        extrapolator.add_pose(t0, &Rigid3d::identity());
        let odometry_velocity = Vector3::new(2., 0., 0.);
        for k in 0..3 {
            let dt = 0.1 * f64::from(k);
            extrapolator.add_odometry_data(&OdometryData {
                time: t0 + from_seconds(dt),
                pose: Rigid3d::from_translation(odometry_velocity * dt),
            });
        }
        let t = t0 + from_seconds(0.5);
        let pose = extrapolator.extrapolate_pose(t);
        assert!((pose.translation() - odometry_velocity * 0.5).norm() < 1e-6);
    }

    proptest! {
        /// Extrapolating a constant velocity (translation only) motion
        /// reproduces it exactly (up to rounding).
        #[test]
        fn constant_velocity_extrapolation_is_exact(
            vx in -5.0..5.0f64, vy in -5.0..5.0f64, vz in -1.0..1.0f64,
            num_poses in 2usize..8,
            extrapolate_seconds in 0.0..2.0f64,
        ) {
            let mut extrapolator = new_extrapolator();
            let velocity = Vector3::new(vx, vy, vz);
            let start = from_universal(1_000_000_000);
            for k in 0..num_poses {
                let t = start + from_seconds(k as f64);
                extrapolator.add_pose(t, &Rigid3d::from_translation(velocity * k as f64));
            }
            let last = start + from_seconds((num_poses - 1) as f64);
            let t = last + from_seconds(extrapolate_seconds);
            let dt = to_seconds(t - start);
            let pose = extrapolator.extrapolate_pose(t);
            prop_assert!((pose.translation() - velocity * dt).norm() < 1e-9);
            prop_assert!(angular_distance(pose.rotation(), &Quaternion::identity()) < 1e-9);
        }
    }
}
