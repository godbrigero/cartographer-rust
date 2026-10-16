//! Port of `cartographer/sensor/internal/trajectory_collator` (.h, .cc).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use crate::metrics::counter::Counter;
use crate::metrics::family_factory::{Family, FamilyFactory, Labels, NullMetric};
use crate::sensor::collator_interface::{Callback, CollatorInterface};
use crate::sensor::data::Data;
use crate::sensor::internal::collator::{per_sensor_callback, sorted_sensor_ids};
use crate::sensor::internal::ordered_multi_queue::{OrderedMultiQueue, QueueKey};

// RUST PORT NOTE: the C++ static member `collator_metrics_family_` (initialised
// to `Family<Counter>::Null()`); `None` means the null family.
static COLLATOR_METRICS_FAMILY: Mutex<Option<&'static dyn Family<dyn Counter>>> = Mutex::new(None);

fn collator_metrics_family() -> &'static dyn Family<dyn Counter> {
    COLLATOR_METRICS_FAMILY
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(<dyn Counter as NullMetric>::null_family)
}

/// Waits to see at least one data item for all sensor ids and dispatches data
/// in merge-sorted order. Contrary to `Collator`, it does not wait for other
/// trajectories.
/// Also contrary to `Collator`, whose output is deterministic, the sequence in
/// which data is dispatched is not sorted, so non-deterministic input
/// sequences will result in non-deterministic output.
#[derive(Default)]
pub struct TrajectoryCollator {
    // Holds individual counters for each trajectory/sensor pair.
    metrics_map: HashMap<String, &'static dyn Counter>,
    // RUST PORT NOTE: a `BTreeMap` (C++ `flat_hash_map`) so that `flush`
    // processes trajectories in a deterministic (ascending id) order.
    trajectory_to_queue: BTreeMap<i32, OrderedMultiQueue>,
    // Map of trajectory ID to all associated QueueKeys.
    trajectory_to_queue_keys: HashMap<i32, Vec<QueueKey>>,
}

impl std::fmt::Debug for TrajectoryCollator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrajectoryCollator")
            .field("trajectory_to_queue", &self.trajectory_to_queue)
            .finish_non_exhaustive()
    }
}

impl TrajectoryCollator {
    /// Creates an empty collator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the `collator_input_total` counter family.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        let family =
            family_factory.new_counter_family("collator_input_total", "Sensor data received");
        *COLLATOR_METRICS_FAMILY
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(family);
    }

    fn get_or_create_sensor_metric(
        &mut self,
        sensor_id: &str,
        trajectory_id: i32,
    ) -> &'static dyn Counter {
        let trajectory_id_str = trajectory_id.to_string();
        let map_key = format!("{sensor_id}/{trajectory_id_str}");
        if let Some(counter) = self.metrics_map.get(&map_key) {
            return *counter;
        }
        log::info!("Create metrics handler for key: {map_key}");
        let labels: Labels = [
            ("sensor_id".to_string(), sensor_id.to_string()),
            ("trajectory_id".to_string(), trajectory_id_str),
        ]
        .into_iter()
        .collect();
        let new_counter = collator_metrics_family().add(&labels);
        self.metrics_map.insert(map_key, new_counter);
        new_counter
    }
}

impl CollatorInterface for TrajectoryCollator {
    fn add_trajectory(
        &mut self,
        trajectory_id: i32,
        expected_sensor_ids: &HashSet<String>,
        callback: Callback,
    ) {
        assert!(
            !self.trajectory_to_queue.contains_key(&trajectory_id),
            "Check failed: trajectory_to_queue_.count(trajectory_id) == 0"
        );
        let callback = Arc::new(Mutex::new(callback));
        for sensor_id in sorted_sensor_ids(expected_sensor_ids) {
            let queue_key = QueueKey {
                trajectory_id,
                sensor_id: sensor_id.clone(),
            };
            self.trajectory_to_queue
                .entry(trajectory_id)
                .or_default()
                .add_queue(queue_key.clone(), per_sensor_callback(&callback, sensor_id));
            self.trajectory_to_queue_keys
                .entry(trajectory_id)
                .or_default()
                .push(queue_key);
        }
    }

    fn finish_trajectory(&mut self, trajectory_id: i32) {
        let Some(queue_keys) = self.trajectory_to_queue_keys.get(&trajectory_id) else {
            return;
        };
        for queue_key in queue_keys {
            self.trajectory_to_queue
                .get_mut(&trajectory_id)
                .unwrap_or_else(|| panic!("Unknown trajectory {trajectory_id}"))
                .mark_queue_as_finished(queue_key);
        }
    }

    fn add_sensor_data(&mut self, trajectory_id: i32, data: Box<dyn Data>) {
        let queue_key = QueueKey {
            trajectory_id,
            sensor_id: data.get_sensor_id().to_string(),
        };
        let metric = self.get_or_create_sensor_metric(&queue_key.sensor_id, trajectory_id);
        metric.increment();
        // Like C++ `.at()`: unknown trajectories are fatal.
        self.trajectory_to_queue
            .get_mut(&trajectory_id)
            .unwrap_or_else(|| panic!("Unknown trajectory {trajectory_id}"))
            .add(&queue_key, data);
    }

    fn flush(&mut self) {
        for queue in self.trajectory_to_queue.values_mut() {
            queue.flush();
        }
    }

    fn get_blocking_trajectory_id(&self) -> Option<i32> {
        None
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

    mod trajectory_collator {
        use super::*;

        #[test]
        fn ordering_multiple_trajectories() {
            let trajectory_id = [4, 7];
            let sensor_id = ["my_points", "some_imu"];

            let mut input_data = vec![
                CollatorInput::create_timed_point_cloud_data(trajectory_id[0], sensor_id[0], 0),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 0),
                CollatorInput::create_imu_data(trajectory_id[1], sensor_id[1], 0),
                CollatorInput::create_imu_data(trajectory_id[0], sensor_id[1], 0),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 100),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[0], sensor_id[0], 50),
                CollatorInput::create_imu_data(trajectory_id[0], sensor_id[1], 60),
                CollatorInput::create_timed_point_cloud_data(trajectory_id[1], sensor_id[0], 150),
                CollatorInput::create_imu_data(trajectory_id[1], sensor_id[1], 120),
            ];

            let received: Received = Arc::default();
            let len = || received.lock().unwrap().len();
            let at = |i: usize| received.lock().unwrap()[i].clone();
            let ids: HashSet<String> = sensor_id.iter().map(|s| s.to_string()).collect();
            let mut collator = TrajectoryCollator::new();
            collator.add_trajectory(
                trajectory_id[0],
                &ids,
                recording_callback(&received, trajectory_id[0]),
            );
            collator.add_trajectory(
                trajectory_id[1],
                &ids,
                recording_callback(&received, trajectory_id[1]),
            );

            // Send each sensor_id once to establish a common start time.
            input_data[0].move_to_collator(&mut collator);
            input_data[1].move_to_collator(&mut collator);
            assert_eq!(0, len());
            input_data[2].move_to_collator(&mut collator);
            input_data[3].move_to_collator(&mut collator);
            assert_eq!(2, len());
            assert_eq!(input_data[1].expected_output, at(0));
            assert_eq!(input_data[0].expected_output, at(1));

            // Does not wait for other trajectory.
            input_data[4].move_to_collator(&mut collator);
            assert_eq!(3, len());
            assert_eq!(input_data[2].expected_output, at(2));

            input_data[5].move_to_collator(&mut collator);
            assert_eq!(4, len());
            assert_eq!(input_data[3].expected_output, at(3));
            input_data[6].move_to_collator(&mut collator);
            assert_eq!(5, len());
            assert_eq!(input_data[5].expected_output, at(4));

            // Sorts different sensors.
            input_data[7].move_to_collator(&mut collator);
            assert_eq!(5, len());
            input_data[8].move_to_collator(&mut collator);
            assert_eq!(7, len());
            assert_eq!(input_data[4].expected_output, at(5));
            assert_eq!(input_data[8].expected_output, at(6));

            assert!(collator.get_blocking_trajectory_id().is_none());

            collator.finish_trajectory(trajectory_id[0]);
            collator.finish_trajectory(trajectory_id[1]);
            collator.flush();
            assert_eq!(input_data.len(), len());
        }
    }

    #[test]
    #[should_panic(expected = "Unknown trajectory")]
    fn unknown_trajectory_panics() {
        let mut collator = TrajectoryCollator::new();
        CollatorInput::create_imu_data(3, "imu", 0).move_to_collator(&mut collator);
    }
}
