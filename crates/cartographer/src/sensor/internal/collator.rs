//! Port of `cartographer/sensor/internal/collator` (.h, .cc).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use crate::sensor::collator_interface::{Callback, CollatorInterface};
use crate::sensor::data::Data;
use crate::sensor::internal::ordered_multi_queue::{OrderedMultiQueue, QueueKey};

/// Wraps a shared user callback into a per-sensor queue callback (C++ copies
/// the `std::function` into a lambda per sensor).
///
/// RUST PORT NOTE: the `FnMut` callback is shared between the per-sensor
/// queue callbacks via `Arc<Mutex<..>>`; the lock is never contended because
/// the queue invokes callbacks synchronously on the caller's thread.
pub(crate) fn per_sensor_callback(
    callback: &Arc<Mutex<Callback>>,
    sensor_id: &str,
) -> crate::sensor::internal::ordered_multi_queue::Callback {
    let callback = Arc::clone(callback);
    let sensor_id = sensor_id.to_string();
    Box::new(move |data: Box<dyn Data>| {
        let mut callback = callback.lock().unwrap_or_else(PoisonError::into_inner);
        (*callback)(&sensor_id, data);
    })
}

/// Returns the sensor ids in sorted order.
///
/// RUST PORT NOTE: C++ iterates the `absl::flat_hash_set` in unspecified
/// order; sorting makes the order of `queue_keys` (and thus of
/// `finish_trajectory`) deterministic.
pub(crate) fn sorted_sensor_ids(expected_sensor_ids: &HashSet<String>) -> Vec<&String> {
    let mut sensor_ids: Vec<&String> = expected_sensor_ids.iter().collect();
    sensor_ids.sort();
    sensor_ids
}

/// Collates the sensor data of all trajectories in one global merge-sorted
/// order (one `OrderedMultiQueue` for all trajectories).
#[derive(Debug, Default)]
pub struct Collator {
    queue: OrderedMultiQueue,
    // Map of trajectory ID to all associated QueueKeys.
    queue_keys: HashMap<i32, Vec<QueueKey>>,
}

impl Collator {
    /// Creates an empty collator.
    pub fn new() -> Self {
        Self::default()
    }
}

impl CollatorInterface for Collator {
    fn add_trajectory(
        &mut self,
        trajectory_id: i32,
        expected_sensor_ids: &HashSet<String>,
        callback: Callback,
    ) {
        let callback = Arc::new(Mutex::new(callback));
        for sensor_id in sorted_sensor_ids(expected_sensor_ids) {
            let queue_key = QueueKey {
                trajectory_id,
                sensor_id: sensor_id.clone(),
            };
            self.queue
                .add_queue(queue_key.clone(), per_sensor_callback(&callback, sensor_id));
            self.queue_keys
                .entry(trajectory_id)
                .or_default()
                .push(queue_key);
        }
    }

    fn finish_trajectory(&mut self, trajectory_id: i32) {
        if let Some(queue_keys) = self.queue_keys.get(&trajectory_id) {
            for queue_key in queue_keys {
                self.queue.mark_queue_as_finished(queue_key);
            }
        }
    }

    fn add_sensor_data(&mut self, trajectory_id: i32, data: Box<dyn Data>) {
        let queue_key = QueueKey {
            trajectory_id,
            sensor_id: data.get_sensor_id().to_string(),
        };
        self.queue.add(&queue_key, data);
    }

    fn flush(&mut self) {
        self.queue.flush();
    }

    fn get_blocking_trajectory_id(&self) -> Option<i32> {
        Some(self.queue.get_blocker().trajectory_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::internal::test_helpers::testing::{CollatorInput, CollatorOutput};

    type Received = Arc<Mutex<Vec<CollatorOutput>>>;

    fn recording_callback(received: &Received, trajectory_id: i32) -> Callback {
        let received = Arc::clone(received);
        Box::new(move |_sensor_id: &str, data: Box<dyn Data>| {
            received.lock().unwrap().push((
                trajectory_id,
                data.get_sensor_id().to_string(),
                data.get_time(),
            ));
        })
    }

    fn ids(sensor_ids: &[&str]) -> HashSet<String> {
        sensor_ids.iter().map(|s| s.to_string()).collect()
    }

    mod collator {
        use super::*;

        #[test]
        fn ordering() {
            const TRAJECTORY_ID: i32 = 0;
            let sensor_id = [
                "horizontal_rangefinder",
                "vertical_rangefinder",
                "imu",
                "odometry",
            ];

            let mut input_data = vec![
                // Send each sensor_id once to establish a common start time.
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[0], 0),
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[1], 0),
                CollatorInput::create_imu_data(TRAJECTORY_ID, sensor_id[2], 0),
                CollatorInput::create_odometry_data(TRAJECTORY_ID, sensor_id[3], 0),
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[0], 100),
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[1], 200),
                CollatorInput::create_imu_data(TRAJECTORY_ID, sensor_id[2], 300),
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[0], 400),
                CollatorInput::create_timed_point_cloud_data(TRAJECTORY_ID, sensor_id[1], 500),
                CollatorInput::create_odometry_data(TRAJECTORY_ID, sensor_id[3], 600),
            ];

            let received: Received = Arc::default();
            let mut collator = Collator::new();
            collator.add_trajectory(
                TRAJECTORY_ID,
                &ids(&sensor_id),
                recording_callback(&received, TRAJECTORY_ID),
            );

            for i in [0, 1, 2, 3, 4, 9, 7, 5, 8, 6] {
                input_data[i].move_to_collator(&mut collator);
            }
            assert_eq!(
                TRAJECTORY_ID,
                collator.get_blocking_trajectory_id().unwrap()
            );

            {
                let received = received.lock().unwrap();
                assert_eq!(7, received.len());
                assert_eq!(input_data[4].expected_output, received[4]);
                assert_eq!(input_data[5].expected_output, received[5]);
                assert_eq!(input_data[6].expected_output, received[6]);
            }

            collator.finish_trajectory(TRAJECTORY_ID);
            collator.flush();
            let received = received.lock().unwrap();
            assert_eq!(input_data.len(), received.len());
            for i in 4..input_data.len() {
                assert_eq!(input_data[i].expected_output, received[i]);
            }
        }

        #[test]
        fn ordering_multiple_trajectories() {
            let trajectory_id = [8, 5];
            let sensor_id = ["my_points", "some_imu"];

            let mut input_data = vec![
                // Send each sensor_id once to establish a common start time.
                CollatorInput::create_timed_point_cloud_data(trajectory_id[0], sensor_id[0], 0),
                CollatorInput::create_imu_data(trajectory_id[0], sensor_id[1], 0),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 0),
                CollatorInput::create_imu_data(trajectory_id[1], sensor_id[1], 0),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[0], sensor_id[0], 100),
                CollatorInput::create_imu_data(trajectory_id[1], sensor_id[1], 200),
                CollatorInput::create_imu_data(trajectory_id[0], sensor_id[1], 300),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 400),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 400),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 500),
                CollatorInput::create_imu_data(trajectory_id[1], sensor_id[1], 600),
            ];

            let received: Received = Arc::default();
            let mut collator = Collator::new();
            collator.add_trajectory(
                trajectory_id[0],
                &ids(&sensor_id),
                recording_callback(&received, trajectory_id[0]),
            );
            collator.add_trajectory(
                trajectory_id[1],
                &ids(&sensor_id),
                recording_callback(&received, trajectory_id[1]),
            );

            for input in input_data.iter_mut().take(4) {
                input.move_to_collator(&mut collator);
            }

            input_data[4].move_to_collator(&mut collator);
            input_data[6].move_to_collator(&mut collator);
            assert_eq!(
                trajectory_id[1],
                collator.get_blocking_trajectory_id().unwrap()
            );
            input_data[7].move_to_collator(&mut collator);
            input_data[8].move_to_collator(&mut collator);
            assert_eq!(
                trajectory_id[1],
                collator.get_blocking_trajectory_id().unwrap()
            );
            input_data[5].move_to_collator(&mut collator);
            assert_eq!(
                trajectory_id[0],
                collator.get_blocking_trajectory_id().unwrap()
            );
            input_data[10].move_to_collator(&mut collator);
            input_data[9].move_to_collator(&mut collator);
            assert_eq!(
                trajectory_id[0],
                collator.get_blocking_trajectory_id().unwrap()
            );

            {
                let received = received.lock().unwrap();
                assert_eq!(5, received.len());
                assert_eq!(input_data[4].expected_output, received[4]);
            }

            collator.finish_trajectory(trajectory_id[0]);
            collator.finish_trajectory(trajectory_id[1]);
            collator.flush();
            let received = received.lock().unwrap();
            assert_eq!(input_data.len(), received.len());
            for i in 4..input_data.len() {
                assert_eq!(input_data[i].expected_output, received[i]);
            }
        }
    }

    #[test]
    fn callback_receives_sensor_id() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let seen_clone = Arc::clone(&seen);
        let mut collator = Collator::new();
        collator.add_trajectory(
            1,
            &ids(&["a"]),
            Box::new(move |sensor_id: &str, _data: Box<dyn Data>| {
                seen_clone.lock().unwrap().push(sensor_id.to_string());
            }),
        );
        CollatorInput::create_imu_data(1, "a", 10).move_to_collator(&mut collator);
        collator.finish_trajectory(1);
        collator.flush();
        assert_eq!(*seen.lock().unwrap(), ["a"]);
    }
}
