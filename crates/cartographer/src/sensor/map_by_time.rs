//! Port of `cartographer/sensor/map_by_time` (.h).

use std::collections::BTreeMap;
use std::ops::Bound;

use crate::common::time::Time;
use crate::sensor::data::HasTime;

/// The node lookups that [`MapByTime::trim`] needs from the C++
/// `mapping::MapById<mapping::NodeId, NodeType>` argument.
///
/// RUST PORT NOTE: `mapping::id` (`MapById`, `NodeId`) is ported separately;
/// that port is expected to implement this trait for
/// `MapById<NodeId, N> where N: HasTime`.
pub trait NodeTimes {
    /// For the node `(trajectory_id, node_index)` returns `Some((previous,
    /// next))`: the times of the previous and next node of the same
    /// trajectory (`None` if it is the first/last node), or `None` if the
    /// node does not exist.
    fn neighbor_times(
        &self,
        trajectory_id: i32,
        node_index: i32,
    ) -> Option<(Option<Time>, Option<Time>)>;
}

/// Time-indexed data per trajectory. `D` must have a time (C++: a `time`
/// member).
#[derive(Clone, Debug)]
pub struct MapByTime<D> {
    data: BTreeMap<i32, BTreeMap<Time, D>>,
}

impl<D> Default for MapByTime<D> {
    fn default() -> Self {
        MapByTime {
            data: BTreeMap::new(),
        }
    }
}

impl<D: HasTime> MapByTime<D> {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends data to a `trajectory_id`, creating trajectories as needed.
    /// Times must be strictly increasing per trajectory.
    pub fn append(&mut self, trajectory_id: i32, data: D) {
        assert!(trajectory_id >= 0, "Check failed: trajectory_id >= 0");
        let trajectory = self.data.entry(trajectory_id).or_default();
        let time = data.time();
        if let Some((&last_time, _)) = trajectory.last_key_value() {
            assert!(
                time > last_time,
                "Check failed: data.time > last time ({time} vs. {last_time})"
            );
        }
        trajectory.insert(time, data);
    }

    /// Removes data no longer needed once node `(trajectory_id, node_index)`
    /// gets removed from `nodes`. Must be called before removing the node
    /// from `nodes` (C++ `Trim(nodes, node_id)`).
    pub fn trim<N: NodeTimes + ?Sized>(&mut self, nodes: &N, trajectory_id: i32, node_index: i32) {
        assert!(trajectory_id >= 0, "Check failed: trajectory_id >= 0");
        if !self.data.contains_key(&trajectory_id) {
            return;
        }
        // Data only important between 'gap_start' and 'gap_end' is no longer
        // needed. We retain the first and last data of the gap so that
        // interpolation with the adjacent data outside the gap is still
        // possible.
        let (previous, next) = nodes
            .neighbor_times(trajectory_id, node_index)
            .expect("Check failed: node_it != nodes.end()");
        let gap_start = previous.unwrap_or(Time::min());
        let gap_end = next.unwrap_or(Time::max());
        self.trim_gap(trajectory_id, gap_start, gap_end);
    }

    /// Core of [`MapByTime::trim`]: removes the data of `trajectory_id` with
    /// time in `[gap_start, gap_end]`, retaining the first (unless
    /// `gap_start == Time::min()`) and the last (unless
    /// `gap_end == Time::max()`) data inside the gap.
    pub fn trim_gap(&mut self, trajectory_id: i32, gap_start: Time, gap_end: Time) {
        assert!(gap_start < gap_end, "Check failed: gap_start < gap_end");
        let Some(trajectory) = self.data.get_mut(&trajectory_id) else {
            return;
        };
        let mut keys: Vec<Time> = trajectory
            .range((Bound::Included(gap_start), Bound::Included(gap_end)))
            .map(|(time, _)| *time)
            .collect();
        if keys.is_empty() {
            return;
        }
        if gap_end != Time::max() {
            // Retain the last data inside the gap.
            keys.pop();
            if keys.is_empty() {
                return;
            }
        }
        let start = usize::from(gap_start != Time::min());
        // Retain the first data inside the gap (if `start == 1`).
        for time in keys.iter().skip(start) {
            trajectory.remove(time);
        }
        if trajectory.is_empty() {
            self.data.remove(&trajectory_id);
        }
    }

    /// True if there is data for `trajectory_id`.
    pub fn has_trajectory(&self, trajectory_id: i32) -> bool {
        self.data.contains_key(&trajectory_id)
    }

    /// The trajectory ids with data, ascending.
    pub fn trajectory_ids(&self) -> impl DoubleEndedIterator<Item = i32> + '_ {
        self.data.keys().copied()
    }

    /// The data of `trajectory_id` in ascending time order. Panics if the
    /// trajectory does not exist (C++ `data_.at()`).
    pub fn trajectory(&self, trajectory_id: i32) -> impl DoubleEndedIterator<Item = &D> + '_ {
        self.trajectory_map(trajectory_id).values()
    }

    /// The underlying time-ordered map of `trajectory_id` (replaces the C++
    /// bidirectional `BeginOfTrajectory`/`EndOfTrajectory` iterators). Panics
    /// if the trajectory does not exist.
    pub fn trajectory_map(&self, trajectory_id: i32) -> &BTreeMap<Time, D> {
        self.data
            .get(&trajectory_id)
            .unwrap_or_else(|| panic!("No data for trajectory {trajectory_id}."))
    }

    /// The data of `trajectory_id` with time `>= time`, ascending (C++
    /// `lower_bound(trajectory_id, time)` .. `EndOfTrajectory`). Panics if
    /// the trajectory does not exist.
    pub fn lower_bound(
        &self,
        trajectory_id: i32,
        time: Time,
    ) -> impl DoubleEndedIterator<Item = &D> + '_ {
        self.trajectory_map(trajectory_id)
            .range(time..)
            .map(|(_, data)| data)
    }

    /// The data of `trajectory_id` with time `< time`, ascending
    /// (`BeginOfTrajectory` .. C++ `lower_bound(trajectory_id, time)`);
    /// `.next_back()` is C++ `std::prev(lower_bound(..))`. Panics if the
    /// trajectory does not exist.
    pub fn before(&self, trajectory_id: i32, time: Time) -> impl DoubleEndedIterator<Item = &D> + '_ {
        self.trajectory_map(trajectory_id)
            .range(..time)
            .map(|(_, data)| data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_milliseconds;

    fn create_time(milliseconds: i64) -> Time {
        Time::from_time_since_epoch(from_milliseconds(milliseconds))
    }

    #[derive(Clone, Debug)]
    struct Data {
        time: Time,
    }

    impl HasTime for Data {
        fn time(&self) -> Time {
            self.time
        }
    }

    /// Minimal stand-in for `mapping::MapById<NodeId, NodeData>`: node
    /// indices are assigned in append order and never reused.
    #[derive(Default)]
    struct FakeMapById {
        nodes: BTreeMap<i32, BTreeMap<i32, Time>>,
        next_index: BTreeMap<i32, i32>,
    }

    impl FakeMapById {
        fn append(&mut self, trajectory_id: i32, time: Time) {
            let index = self.next_index.entry(trajectory_id).or_insert(0);
            self.nodes
                .entry(trajectory_id)
                .or_default()
                .insert(*index, time);
            *index += 1;
        }
        fn trim(&mut self, trajectory_id: i32, node_index: i32) {
            let trajectory = self.nodes.get_mut(&trajectory_id).unwrap();
            assert!(trajectory.remove(&node_index).is_some());
            if trajectory.is_empty() {
                self.nodes.remove(&trajectory_id);
            }
        }
    }

    impl NodeTimes for FakeMapById {
        fn neighbor_times(
            &self,
            trajectory_id: i32,
            node_index: i32,
        ) -> Option<(Option<Time>, Option<Time>)> {
            let trajectory = self.nodes.get(&trajectory_id)?;
            trajectory.get(&node_index)?;
            let previous = trajectory.range(..node_index).next_back().map(|(_, t)| *t);
            let next = trajectory.range(node_index + 1..).next().map(|(_, t)| *t);
            Some((previous, next))
        }
    }

    fn times(map_by_time: &MapByTime<Data>, trajectory_id: i32) -> Vec<Time> {
        map_by_time.trajectory(trajectory_id).map(|d| d.time).collect()
    }

    mod map_by_time_test {
        use super::*;

        #[test]
        fn append_and_view_trajectory() {
            let mut map_by_time = MapByTime::new();
            map_by_time.append(0, Data { time: create_time(10) });
            map_by_time.append(42, Data { time: create_time(42) });
            map_by_time.append(42, Data { time: create_time(43) });
            assert_eq!(times(&map_by_time, 42), [create_time(42), create_time(43)]);
        }

        #[test]
        fn trimming() {
            let mut map_by_time = MapByTime::new();
            assert!(!map_by_time.has_trajectory(42));
            for ms in [1, 41, 42, 43, 47, 48, 49, 5000] {
                map_by_time.append(42, Data { time: create_time(ms) });
            }
            assert!(map_by_time.has_trajectory(42));
            // Trim one node.
            let mut map_by_id = FakeMapById::default();
            map_by_id.append(42, create_time(42));
            map_by_id.append(42, create_time(46));
            map_by_id.append(42, create_time(48));
            map_by_time.trim(&map_by_id, 42, 1);
            map_by_id.trim(42, 1);
            assert!(map_by_time.has_trajectory(42));
            let expected: Vec<Time> = [1, 41, 42, 48, 49, 5000]
                .into_iter()
                .map(create_time)
                .collect();
            assert_eq!(times(&map_by_time, 42), expected);
            // Trim everything.
            map_by_time.trim(&map_by_id, 42, 2);
            map_by_id.trim(42, 2);
            map_by_time.trim(&map_by_id, 42, 0);
            map_by_id.trim(42, 0);
            assert!(!map_by_time.has_trajectory(42));
        }

        #[test]
        fn trimming_does_not_create_trajectory() {
            let mut map_by_time: MapByTime<Data> = MapByTime::new();
            assert!(!map_by_time.has_trajectory(42));
            let mut map_by_id = FakeMapById::default();
            map_by_id.append(42, create_time(42));
            map_by_time.trim(&map_by_id, 42, 0);
            assert!(!map_by_time.has_trajectory(42));
        }
    }

    #[test]
    fn lower_bound_and_before() {
        let mut map_by_time = MapByTime::new();
        for ms in [10, 20, 30] {
            map_by_time.append(1, Data { time: create_time(ms) });
        }
        map_by_time.append(3, Data { time: create_time(1) });
        assert_eq!(map_by_time.trajectory_ids().collect::<Vec<_>>(), [1, 3]);
        let mut it = map_by_time.lower_bound(1, create_time(15));
        assert_eq!(it.next().unwrap().time, create_time(20));
        assert_eq!(
            map_by_time.before(1, create_time(20)).next_back().unwrap().time,
            create_time(10)
        );
        assert_eq!(map_by_time.lower_bound(1, create_time(31)).count(), 0);
    }

    #[test]
    #[should_panic]
    fn append_requires_increasing_time() {
        let mut map_by_time = MapByTime::new();
        map_by_time.append(0, Data { time: create_time(10) });
        map_by_time.append(0, Data { time: create_time(10) });
    }
}
