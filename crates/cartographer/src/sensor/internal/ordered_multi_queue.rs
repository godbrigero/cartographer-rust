//! Port of `cartographer/sensor/internal/ordered_multi_queue` (.h, .cc).

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::common::time::Time;
use crate::sensor::data::Data;

// Number of items that can be queued up before we log which queues are waiting
// for data.
const MAX_QUEUE_SIZE: usize = 500;

/// Emulates glog's `LOG_EVERY_N`: true on the 1st, (n+1)th, ... call.
pub(crate) fn log_every_n(counter: &AtomicU64, n: u64) -> bool {
    counter.fetch_add(1, Ordering::Relaxed).is_multiple_of(n)
}

/// Key of a queue: `(trajectory_id, sensor_id)`, ordered lexicographically.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueKey {
    /// The trajectory id.
    pub trajectory_id: i32,
    /// The sensor id.
    pub sensor_id: String,
}

impl fmt::Display for QueueKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.trajectory_id, self.sensor_id)
    }
}

/// Callback receiving dispatched data (C++ `OrderedMultiQueue::Callback`).
pub type Callback = Box<dyn FnMut(Box<dyn Data>) + Send>;

struct Queue {
    // RUST PORT NOTE: C++ uses a `common::BlockingQueue`, but only its
    // non-blocking operations; a `VecDeque` is equivalent.
    queue: VecDeque<Box<dyn Data>>,
    callback: Callback,
    finished: bool,
}

/// Maintains multiple queues of sorted sensor data and dispatches it in merge
/// sorted order. It will wait to see at least one value for each unfinished
/// queue before dispatching the next time ordered value across all queues.
///
/// This class is thread-compatible.
pub struct OrderedMultiQueue {
    // Used to verify that values are dispatched in sorted order.
    last_dispatched_time: Time,
    common_start_time_per_trajectory: BTreeMap<i32, Time>,
    queues: BTreeMap<QueueKey, Queue>,
    // RUST PORT NOTE: `None` until the first `CannotMakeProgress` (the C++
    // member is default-constructed with an uninitialized trajectory id).
    blocker: Option<QueueKey>,
}

impl Default for OrderedMultiQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for OrderedMultiQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrderedMultiQueue")
            .field("last_dispatched_time", &self.last_dispatched_time)
            .field("queues", &self.queues.keys().collect::<Vec<_>>())
            .field("blocker", &self.blocker)
            .finish()
    }
}

impl OrderedMultiQueue {
    /// Creates an empty multi-queue.
    pub fn new() -> Self {
        OrderedMultiQueue {
            last_dispatched_time: Time::min(),
            common_start_time_per_trajectory: BTreeMap::new(),
            queues: BTreeMap::new(),
            blocker: None,
        }
    }

    /// Adds a new queue with key `queue_key` which must not already exist.
    /// `callback` will be called whenever data from this queue can be
    /// dispatched.
    pub fn add_queue(&mut self, queue_key: QueueKey, callback: Callback) {
        assert!(
            !self.queues.contains_key(&queue_key),
            "Check failed: queues_.count(queue_key) == 0 for '{queue_key}'"
        );
        self.queues.insert(
            queue_key,
            Queue {
                queue: VecDeque::new(),
                callback,
                finished: false,
            },
        );
    }

    /// Marks a queue as finished, i.e. no further data can be added. The queue
    /// will be removed once the last piece of data from it has been
    /// dispatched.
    pub fn mark_queue_as_finished(&mut self, queue_key: &QueueKey) {
        let queue = self
            .queues
            .get_mut(queue_key)
            .unwrap_or_else(|| panic!("Did not find '{queue_key}'."));
        assert!(!queue.finished, "Check failed: !queue.finished");
        queue.finished = true;
        self.dispatch();
    }

    /// Adds `data` to a queue with the given `queue_key`. Data must be added
    /// sorted per queue. Data for unknown queues is dropped with a warning.
    pub fn add(&mut self, queue_key: &QueueKey, data: Box<dyn Data>) {
        let Some(queue) = self.queues.get_mut(queue_key) else {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            if log_every_n(&COUNTER, 1000) {
                log::warn!("Ignored data for queue: '{queue_key}'");
            }
            return;
        };
        queue.queue.push_back(data);
        self.dispatch();
    }

    /// Dispatches all remaining values in sorted order and removes the
    /// underlying queues.
    pub fn flush(&mut self) {
        let unfinished_queues: Vec<QueueKey> = self
            .queues
            .iter()
            .filter(|(_, queue)| !queue.finished)
            .map(|(key, _)| key.clone())
            .collect();
        for unfinished_queue in &unfinished_queues {
            self.mark_queue_as_finished(unfinished_queue);
        }
    }

    /// Must only be called if at least one unfinished queue exists. Returns the
    /// key of a queue that needs more data before the `OrderedMultiQueue` can
    /// dispatch data.
    ///
    /// RUST PORT NOTE: if no queue has blocked yet (C++ returns a key with an
    /// uninitialized trajectory id) this returns `QueueKey::default()`.
    pub fn get_blocker(&self) -> QueueKey {
        assert!(!self.queues.is_empty(), "Check failed: !queues_.empty()");
        self.blocker.clone().unwrap_or_default()
    }

    fn dispatch(&mut self) {
        loop {
            let mut next: Option<(QueueKey, Time)> = None;
            let mut finished_and_empty = Vec::new();
            let mut blocked_on = None;
            for (key, queue) in &self.queues {
                let Some(data) = queue.queue.front() else {
                    if queue.finished {
                        finished_and_empty.push(key.clone());
                        continue;
                    }
                    blocked_on = Some(key.clone());
                    break;
                };
                let time = data.get_time();
                if next.as_ref().is_none_or(|(_, next_time)| time < *next_time) {
                    next = Some((key.clone(), time));
                }
                let next_time = next.as_ref().map(|(_, t)| *t).unwrap_or(time);
                assert!(
                    self.last_dispatched_time <= next_time,
                    "Non-sorted data added to queue: '{key}'"
                );
            }
            // RUST PORT NOTE: C++ erases these during the scan; erasing
            // afterwards is equivalent (only queues before the blocker are
            // erased in both cases).
            for key in &finished_and_empty {
                self.queues.remove(key);
            }
            if let Some(key) = blocked_on {
                self.cannot_make_progress(key);
                return;
            }
            let Some((next_queue_key, next_time)) = next else {
                assert!(self.queues.is_empty(), "Check failed: queues_.empty()");
                return;
            };

            // If we haven't dispatched any data for this trajectory yet, fast
            // forward all queues of this trajectory until a common start time
            // has been reached.
            let common_start_time = self.get_common_start_time(next_queue_key.trajectory_id);

            let next_queue = self
                .queues
                .get_mut(&next_queue_key)
                .expect("selected queue exists");
            if next_time >= common_start_time {
                // Happy case, we are beyond the 'common_start_time' already.
                self.last_dispatched_time = next_time;
                let data = next_queue.queue.pop_front().expect("non-empty");
                (next_queue.callback)(data);
            } else if next_queue.queue.len() < 2 {
                if !next_queue.finished {
                    // We cannot decide whether to drop or dispatch this yet.
                    self.cannot_make_progress(next_queue_key);
                    return;
                }
                self.last_dispatched_time = next_time;
                let data = next_queue.queue.pop_front().expect("non-empty");
                (next_queue.callback)(data);
            } else {
                // We take a peek at the time after next data. If it also is
                // not beyond 'common_start_time' we drop 'next_data',
                // otherwise we just found the first packet to dispatch from
                // this queue.
                let next_data_owner = next_queue.queue.pop_front().expect("non-empty");
                let following_time = next_queue.queue.front().expect("size >= 2").get_time();
                if following_time > common_start_time {
                    self.last_dispatched_time = next_time;
                    (next_queue.callback)(next_data_owner);
                }
            }
        }
    }

    fn cannot_make_progress(&mut self, queue_key: QueueKey) {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        if self
            .queues
            .values()
            .any(|queue| queue.queue.len() > MAX_QUEUE_SIZE)
            && log_every_n(&COUNTER, 60)
        {
            log::warn!("Queue waiting for data: {queue_key}");
        }
        self.blocker = Some(queue_key);
    }

    fn get_common_start_time(&mut self, trajectory_id: i32) -> Time {
        if let Some(time) = self.common_start_time_per_trajectory.get(&trajectory_id) {
            return *time;
        }
        let mut common_start_time = Time::min();
        for (key, queue) in &self.queues {
            if key.trajectory_id == trajectory_id {
                let front_time = queue
                    .queue
                    .front()
                    .expect("all queues are non-empty here")
                    .get_time();
                common_start_time = common_start_time.max(front_time);
            }
        }
        self.common_start_time_per_trajectory
            .insert(trajectory_id, common_start_time);
        log::info!(
            "All sensor data for trajectory {trajectory_id} is available starting at \
             '{common_start_time}'."
        );
        common_start_time
    }
}

impl Drop for OrderedMultiQueue {
    fn drop(&mut self) {
        // RUST PORT NOTE: skipped while unwinding to avoid a double panic.
        if std::thread::panicking() {
            return;
        }
        for (key, queue) in &self.queues {
            assert!(
                queue.finished,
                "Check failed: queue '{key}' is not finished"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{from_universal, to_universal};
    use crate::sensor::imu_data::ImuData;
    use crate::sensor::internal::dispatchable::make_dispatchable;
    use nalgebra::Vector3;
    use std::sync::{Arc, Mutex};

    // These are keys are chosen so that they sort first, second, third.
    fn first() -> QueueKey {
        QueueKey {
            trajectory_id: 1,
            sensor_id: "a".into(),
        }
    }
    fn second() -> QueueKey {
        QueueKey {
            trajectory_id: 1,
            sensor_id: "b".into(),
        }
    }
    fn third() -> QueueKey {
        QueueKey {
            trajectory_id: 2,
            sensor_id: "b".into(),
        }
    }

    struct OrderedMultiQueueTest {
        values: Arc<Mutex<Vec<Box<dyn Data>>>>,
        queue: OrderedMultiQueue,
    }

    impl OrderedMultiQueueTest {
        fn new() -> Self {
            let values: Arc<Mutex<Vec<Box<dyn Data>>>> = Arc::default();
            let mut queue = OrderedMultiQueue::new();
            for queue_key in [first(), second(), third()] {
                let values = Arc::clone(&values);
                queue.add_queue(
                    queue_key,
                    Box::new(move |data| {
                        let mut values = values.lock().unwrap();
                        if let Some(last) = values.last() {
                            assert!(data.get_time() >= last.get_time());
                        }
                        values.push(data);
                    }),
                );
            }
            OrderedMultiQueueTest { values, queue }
        }

        fn make_imu(ordinal: i64) -> Box<dyn Data> {
            make_dispatchable(
                "imu",
                ImuData {
                    time: from_universal(ordinal),
                    linear_acceleration: Vector3::zeros(),
                    angular_velocity: Vector3::zeros(),
                },
            )
        }

        fn add(&mut self, key: QueueKey, ordinal: i64) {
            self.queue.add(&key, Self::make_imu(ordinal));
        }

        fn len(&self) -> usize {
            self.values.lock().unwrap().len()
        }
    }

    mod ordered_multi_queue_test {
        use super::*;

        #[test]
        fn ordering() {
            let mut t = OrderedMultiQueueTest::new();
            t.add(first(), 0);
            t.add(first(), 4);
            t.add(first(), 5);
            t.add(first(), 6);
            assert_eq!(t.len(), 0);
            t.add(second(), 0);
            t.add(second(), 1);
            assert_eq!(t.len(), 0);
            t.add(third(), 0);
            t.add(third(), 2);
            assert_eq!(t.len(), 4);
            t.add(second(), 3);
            assert_eq!(t.len(), 5);
            t.add(second(), 7);
            t.add(third(), 8);
            t.queue.flush();

            let values = t.values.lock().unwrap();
            assert_eq!(11, values.len());
            for i in 0..values.len() - 1 {
                assert!(values[i].get_time() <= values[i + 1].get_time());
            }
        }

        #[test]
        fn mark_queue_as_finished() {
            let mut t = OrderedMultiQueueTest::new();
            t.add(first(), 1);
            t.add(first(), 2);
            t.add(first(), 3);
            assert_eq!(t.len(), 0);
            t.queue.mark_queue_as_finished(&first());
            assert_eq!(t.len(), 0);
            t.queue.mark_queue_as_finished(&second());
            assert_eq!(t.len(), 0);
            t.queue.mark_queue_as_finished(&third());

            let values = t.values.lock().unwrap();
            assert_eq!(3, values.len());
            for (i, value) in values.iter().enumerate() {
                assert_eq!(i as i64 + 1, to_universal(value.get_time()));
            }
        }

        #[test]
        fn common_start_time_per_trajectory() {
            let mut t = OrderedMultiQueueTest::new();
            t.add(first(), 0);
            t.add(first(), 1);
            t.add(first(), 2);
            t.add(first(), 3);
            t.add(second(), 2);
            assert_eq!(t.len(), 0);
            t.add(third(), 4);
            assert_eq!(t.len(), 2);
            t.queue.mark_queue_as_finished(&first());
            assert_eq!(t.len(), 2);
            t.queue.mark_queue_as_finished(&second());
            assert_eq!(t.len(), 4);
            t.queue.mark_queue_as_finished(&third());
            assert_eq!(t.len(), 4);
        }
    }

    #[test]
    fn blocker_and_unknown_queue() {
        let mut t = OrderedMultiQueueTest::new();
        // Data for an unknown queue is dropped.
        t.add(
            QueueKey {
                trajectory_id: 9,
                sensor_id: "x".into(),
            },
            0,
        );
        t.add(first(), 0);
        assert_eq!(t.queue.get_blocker(), second());
        t.add(second(), 0);
        assert_eq!(t.queue.get_blocker(), third());
        t.queue.flush();
        assert_eq!(t.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Non-sorted data added to queue")]
    fn non_sorted_data_panics() {
        let mut queue = OrderedMultiQueue::new();
        queue.add_queue(first(), Box::new(|_| {}));
        queue.add(&first(), OrderedMultiQueueTest::make_imu(5));
        queue.add(&first(), OrderedMultiQueueTest::make_imu(6));
        // 5 was dispatched (single queue), now 4 is out of order.
        queue.add(&first(), OrderedMultiQueueTest::make_imu(4));
    }
}
