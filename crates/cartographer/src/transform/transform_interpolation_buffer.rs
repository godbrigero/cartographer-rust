//! Port of `cartographer/transform/transform_interpolation_buffer` (.h, .cc).

use std::collections::VecDeque;

use super::rigid_transform::Rigid3d;
use super::timestamped_transform::{TimestampedTransform, interpolate};
use super::transform::to_rigid3;
use crate::common::time::{Time, from_universal};
use crate::protos::cartographer::mapping::proto::Trajectory;

/// Buffer size limit meaning "unlimited".
pub const UNLIMITED_BUFFER_SIZE: usize = usize::MAX;

/// A time-ordered buffer of transforms that supports interpolated lookups.
/// Unless explicitly set, the buffer size is unlimited.
#[derive(Clone, Debug)]
pub struct TransformInterpolationBuffer {
    timestamped_transforms: VecDeque<TimestampedTransform>,
    buffer_size_limit: usize,
}

impl Default for TransformInterpolationBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformInterpolationBuffer {
    /// Creates an empty buffer with unlimited size.
    pub fn new() -> Self {
        TransformInterpolationBuffer {
            timestamped_transforms: VecDeque::new(),
            buffer_size_limit: UNLIMITED_BUFFER_SIZE,
        }
    }

    /// Creates a buffer containing the poses of all nodes of `trajectory`.
    pub fn from_trajectory(trajectory: &Trajectory) -> Self {
        let mut buffer = Self::new();
        for node in &trajectory.node {
            buffer.push(
                from_universal(node.timestamp),
                &to_rigid3(&node.pose.unwrap_or_default()),
            );
        }
        buffer
    }

    /// Sets the transform buffer size limit and removes old transforms if it
    /// is exceeded.
    pub fn set_size_limit(&mut self, buffer_size_limit: usize) {
        self.buffer_size_limit = buffer_size_limit;
        self.remove_old_transforms_if_needed();
    }

    /// Adds a new transform to the buffer and removes the oldest transform
    /// if the buffer size limit is exceeded. Panics if `time` is older than
    /// the latest transform (equal times are allowed).
    pub fn push(&mut self, time: Time, transform: &Rigid3d) {
        if !self.timestamped_transforms.is_empty() {
            assert!(
                time >= self.latest_time(),
                "Check failed: time >= latest_time() ({time} vs. {}) New transform is older than latest.",
                self.latest_time()
            );
        }
        self.timestamped_transforms.push_back(TimestampedTransform {
            time,
            transform: *transform,
        });
        self.remove_old_transforms_if_needed();
    }

    /// Clears the transform buffer.
    pub fn clear(&mut self) {
        self.timestamped_transforms.clear();
    }

    /// Returns true if an interpolated transform can be computed at `time`.
    pub fn has(&self, time: Time) -> bool {
        if self.timestamped_transforms.is_empty() {
            return false;
        }
        self.earliest_time() <= time && time <= self.latest_time()
    }

    /// Returns an interpolated transform at `time`. Panics unless a
    /// transform at `time` is available.
    pub fn lookup(&self, time: Time) -> Rigid3d {
        assert!(self.has(time), "Missing transform for: {time}");
        let end = self
            .timestamped_transforms
            .partition_point(|timestamped_transform| timestamped_transform.time < time);
        let end_transform = &self.timestamped_transforms[end];
        if end_transform.time == time {
            return end_transform.transform;
        }
        let start_transform = &self.timestamped_transforms[end - 1];
        interpolate(start_transform, end_transform, time).transform
    }

    fn remove_old_transforms_if_needed(&mut self) {
        while self.timestamped_transforms.len() > self.buffer_size_limit {
            self.timestamped_transforms.pop_front();
        }
    }

    /// Returns the timestamp of the earliest transform in the buffer. Panics
    /// if the buffer is empty (the C++ header comment claims it returns 0).
    pub fn earliest_time(&self) -> Time {
        assert!(!self.empty(), "Empty buffer.");
        self.timestamped_transforms.front().expect("non-empty").time
    }

    /// Returns the timestamp of the latest transform in the buffer. Panics
    /// if the buffer is empty.
    pub fn latest_time(&self) -> Time {
        assert!(!self.empty(), "Empty buffer.");
        self.timestamped_transforms.back().expect("non-empty").time
    }

    /// Returns true if the buffer is empty.
    pub fn empty(&self) -> bool {
        self.timestamped_transforms.is_empty()
    }

    /// Returns the maximum allowed size of the transform buffer.
    pub fn size_limit(&self) -> usize {
        self.buffer_size_limit
    }

    /// Returns the current size of the transform buffer.
    pub fn size(&self) -> usize {
        self.timestamped_transforms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::cartographer::mapping::proto::trajectory::Node;
    use crate::transform::rigid_transform::AngleAxis;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use crate::transform::transform::to_proto;
    use nalgebra::Vector3;

    mod transform_interpolation_buffer_test {
        use super::*;

        #[test]
        fn test_has() {
            let mut buffer = TransformInterpolationBuffer::new();
            assert!(!buffer.has(from_universal(50)));
            buffer.push(from_universal(50), &Rigid3d::identity());
            assert!(!buffer.has(from_universal(25)));
            assert!(buffer.has(from_universal(50)));
            assert!(!buffer.has(from_universal(75)));
            buffer.push(from_universal(100), &Rigid3d::identity());
            assert!(!buffer.has(from_universal(25)));
            assert!(buffer.has(from_universal(50)));
            assert!(buffer.has(from_universal(75)));
            assert!(buffer.has(from_universal(100)));
            assert!(!buffer.has(from_universal(125)));
            assert_eq!(from_universal(50), buffer.earliest_time());
            assert_eq!(from_universal(100), buffer.latest_time());
        }

        #[test]
        fn test_lookup() {
            let mut buffer = TransformInterpolationBuffer::new();
            buffer.push(from_universal(50), &Rigid3d::identity());
            // The rotation needs to be relatively small in order for the
            // interpolation to remain a z-axis rotation.
            buffer.push(
                from_universal(100),
                &(Rigid3d::from_translation(Vector3::new(10., 10., 10.))
                    * Rigid3d::from_angle_axis(AngleAxis::new(2., Vector3::z()))),
            );
            let time = from_universal(75);
            let interpolated = buffer.lookup(time);
            assert_is_nearly(
                &interpolated,
                &(Rigid3d::from_translation(Vector3::new(5., 5., 5.))
                    * Rigid3d::from_angle_axis(AngleAxis::new(1., Vector3::z()))),
                1e-6,
            );
        }

        #[test]
        fn test_lookup_single_transform() {
            let mut buffer = TransformInterpolationBuffer::new();
            let time = from_universal(75);
            buffer.push(time, &Rigid3d::identity());
            let interpolated = buffer.lookup(time);
            assert_is_nearly(&interpolated, &Rigid3d::identity(), 1e-6);
        }

        #[test]
        fn test_set_size_limit() {
            let mut buffer = TransformInterpolationBuffer::new();
            assert_eq!(buffer.size_limit(), UNLIMITED_BUFFER_SIZE);
            buffer.push(from_universal(0), &Rigid3d::identity());
            buffer.push(from_universal(1), &Rigid3d::identity());
            buffer.push(from_universal(2), &Rigid3d::identity());
            assert_eq!(buffer.size(), 3);
            buffer.set_size_limit(2);
            assert_eq!(buffer.size_limit(), 2);
            assert_eq!(buffer.size(), 2);
            assert!(!buffer.has(from_universal(0)));
            buffer.push(from_universal(3), &Rigid3d::identity());
            assert_eq!(buffer.size(), 2);
            assert!(!buffer.has(from_universal(1)));
        }
    }

    #[test]
    fn duplicate_timestamps_return_the_first() {
        let mut buffer = TransformInterpolationBuffer::new();
        let a = Rigid3d::from_translation(Vector3::new(1., 0., 0.));
        let b = Rigid3d::from_translation(Vector3::new(2., 0., 0.));
        buffer.push(from_universal(0), &Rigid3d::identity());
        buffer.push(from_universal(10), &a);
        buffer.push(from_universal(10), &b);
        assert_eq!(buffer.lookup(from_universal(10)), a);
        buffer.clear();
        assert!(buffer.empty());
    }

    #[test]
    #[should_panic(expected = "New transform is older than latest.")]
    fn push_older_panics() {
        let mut buffer = TransformInterpolationBuffer::new();
        buffer.push(from_universal(10), &Rigid3d::identity());
        buffer.push(from_universal(5), &Rigid3d::identity());
    }

    #[test]
    #[should_panic(expected = "Missing transform for: 5")]
    fn lookup_missing_panics() {
        let mut buffer = TransformInterpolationBuffer::new();
        buffer.push(from_universal(10), &Rigid3d::identity());
        buffer.lookup(from_universal(5));
    }

    #[test]
    #[should_panic(expected = "Empty buffer.")]
    fn earliest_time_of_empty_buffer_panics() {
        TransformInterpolationBuffer::new().earliest_time();
    }

    #[test]
    fn from_trajectory() {
        let trajectory = Trajectory {
            node: vec![
                Node {
                    timestamp: 1,
                    pose: Some(to_proto(&Rigid3d::identity())),
                    ..Default::default()
                },
                Node {
                    timestamp: 3,
                    pose: Some(to_proto(&Rigid3d::from_translation(Vector3::new(
                        2., 0., 0.,
                    )))),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let buffer = TransformInterpolationBuffer::from_trajectory(&trajectory);
        assert_eq!(buffer.size(), 2);
        assert_is_nearly(
            &buffer.lookup(from_universal(2)),
            &Rigid3d::from_translation(Vector3::new(1., 0., 0.)),
            1e-12,
        );
    }
}
