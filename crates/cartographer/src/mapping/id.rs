//! Port of `cartographer/mapping/id` (.h).
//!
//! [`NodeId`] and [`SubmapId`] identify trajectory nodes and submaps;
//! [`MapById`] is the `std::map`-like container indexed by them that the pose
//! graph uses everywhere.
//!
//! RUST PORT NOTE: the C++ bidirectional `ConstIterator` becomes the cursor
//! type [`ConstIterator`] (comparable with `==`, movable with
//! [`ConstIterator::increment`]/[`ConstIterator::decrement`], i.e. `++`/`--`,
//! and [`ConstIterator::to_next`]/[`ConstIterator::to_prev`], i.e.
//! `std::next`/`std::prev`). A cursor stores the id it points at, so it stays
//! valid when *other* elements are trimmed (C++ `std::map` iterators have the
//! same property). `Range<ConstIterator>` becomes [`Range`], a Rust
//! `Iterator` between two cursors. The C++ SFINAE `internal::GetTime`
//! (`t.time()` or `t.time`) becomes the [`HasTime`] trait required by
//! [`MapById::lower_bound`].

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound::{Excluded, Included, Unbounded};

use crate::common::time::Time;
use crate::mapping::proto;

/// Access to the timestamp of data stored in a [`MapById`] (C++
/// `internal::GetTime`, which accepts either a `time()` method or a `time`
/// member).
pub use crate::sensor::data::HasTime;

impl<T: HasTime + ?Sized> HasTime for &T {
    fn time(&self) -> Time {
        (**self).time()
    }
}

impl<T: HasTime + ?Sized> HasTime for std::sync::Arc<T> {
    fn time(&self) -> Time {
        (**self).time()
    }
}

/// Uniquely identifies a trajectory node using a combination of a unique
/// trajectory ID and a zero-based index of the node inside that trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId {
    /// The trajectory ID.
    pub trajectory_id: i32,
    /// Zero-based index of the node inside the trajectory.
    pub node_index: i32,
}

impl NodeId {
    /// Creates a node ID.
    pub const fn new(trajectory_id: i32, node_index: i32) -> Self {
        NodeId {
            trajectory_id,
            node_index,
        }
    }

    /// Converts to the proto message (C++ `ToProto(proto::NodeId*)`).
    pub fn to_proto(&self) -> proto::NodeId {
        proto::NodeId {
            trajectory_id: self.trajectory_id,
            node_index: self.node_index,
        }
    }

    /// Creates a node ID from its proto message.
    pub fn from_proto(proto: &proto::NodeId) -> Self {
        NodeId::new(proto.trajectory_id, proto.node_index)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.trajectory_id, self.node_index)
    }
}

/// Uniquely identifies a submap using a combination of a unique trajectory ID
/// and a zero-based index of the submap inside that trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmapId {
    /// The trajectory ID.
    pub trajectory_id: i32,
    /// Zero-based index of the submap inside the trajectory.
    pub submap_index: i32,
}

impl SubmapId {
    /// Creates a submap ID.
    pub const fn new(trajectory_id: i32, submap_index: i32) -> Self {
        SubmapId {
            trajectory_id,
            submap_index,
        }
    }

    /// Converts to the proto message (C++ `ToProto(proto::SubmapId*)`).
    pub fn to_proto(&self) -> proto::SubmapId {
        proto::SubmapId {
            trajectory_id: self.trajectory_id,
            submap_index: self.submap_index,
        }
    }

    /// Creates a submap ID from its proto message.
    pub fn from_proto(proto: &proto::SubmapId) -> Self {
        SubmapId::new(proto.trajectory_id, proto.submap_index)
    }
}

impl fmt::Display for SubmapId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.trajectory_id, self.submap_index)
    }
}

/// An ID type usable as the key of a [`MapById`]: a `(trajectory_id, index)`
/// pair (C++: `NodeId` or `SubmapId`, with the private `MapById::GetIndex`
/// overloads).
pub trait IdType: Copy + Ord + fmt::Debug + fmt::Display {
    /// Creates the ID `(trajectory_id, index)`.
    fn from_parts(trajectory_id: i32, index: i32) -> Self;
    /// The trajectory ID.
    fn trajectory_id(&self) -> i32;
    /// The index inside the trajectory (C++ `MapById::GetIndex`).
    fn index(&self) -> i32;
}

impl IdType for NodeId {
    fn from_parts(trajectory_id: i32, index: i32) -> Self {
        NodeId::new(trajectory_id, index)
    }
    fn trajectory_id(&self) -> i32 {
        self.trajectory_id
    }
    fn index(&self) -> i32 {
        self.node_index
    }
}

impl IdType for SubmapId {
    fn from_parts(trajectory_id: i32, index: i32) -> Self {
        SubmapId::new(trajectory_id, index)
    }
    fn trajectory_id(&self) -> i32 {
        self.trajectory_id
    }
    fn index(&self) -> i32 {
        self.submap_index
    }
}

/// An `(id, data)` pair yielded when iterating a [`MapById`].
#[derive(Debug)]
pub struct IdDataReference<'a, I, D> {
    /// The ID.
    pub id: I,
    /// The data stored for `id`.
    pub data: &'a D,
}

impl<I: Copy, D> Clone for IdDataReference<'_, I, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I: Copy, D> Copy for IdDataReference<'_, I, D> {}

#[derive(Clone, Debug)]
struct MapByIndex<D> {
    can_append: bool,
    data: BTreeMap<i32, D>,
}

impl<D> Default for MapByIndex<D> {
    fn default() -> Self {
        MapByIndex {
            can_append: true,
            data: BTreeMap::new(),
        }
    }
}

/// Reminiscent of `std::map`, but indexed by `I` which can be [`NodeId`] or
/// [`SubmapId`].
///
/// Note: This container will only ever contain non-empty trajectories.
/// Trimming the last remaining node of a trajectory drops the trajectory.
#[derive(Clone, Debug)]
pub struct MapById<I, D> {
    trajectories: BTreeMap<i32, MapByIndex<D>>,
    _id: PhantomData<I>,
}

impl<I, D> Default for MapById<I, D> {
    fn default() -> Self {
        MapById {
            trajectories: BTreeMap::new(),
            _id: PhantomData,
        }
    }
}

/// A position in a [`MapById`] (C++ `MapById::ConstIterator`): either an
/// element or the end position.
pub struct ConstIterator<'a, I, D> {
    map: &'a MapById<I, D>,
    /// `(trajectory_id, index)` of the current element; `None` is the end.
    current: Option<(i32, i32)>,
}

impl<I, D> Clone for ConstIterator<'_, I, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I, D> Copy for ConstIterator<'_, I, D> {}

impl<I, D> PartialEq for ConstIterator<'_, I, D> {
    fn eq(&self, other: &Self) -> bool {
        self.current == other.current
    }
}

impl<I, D> Eq for ConstIterator<'_, I, D> {}

impl<I, D> fmt::Debug for ConstIterator<'_, I, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.current {
            Some((t, i)) => write!(f, "ConstIterator({t}, {i})"),
            None => write!(f, "ConstIterator(end)"),
        }
    }
}

impl<'a, I: IdType, D> ConstIterator<'a, I, D> {
    /// Iterator to the first element of the first trajectory with ID
    /// `>= trajectory_id` (C++ `ConstIterator(map_by_id, trajectory_id)`).
    pub fn at_trajectory(map: &'a MapById<I, D>, trajectory_id: i32) -> Self {
        let current = map
            .trajectories
            .range(trajectory_id..)
            .find_map(|(&t, trajectory)| trajectory.data.keys().next().map(|&i| (t, i)));
        ConstIterator { map, current }
    }

    /// Iterator to the element `id`, or the end iterator if it does not exist
    /// (C++ `ConstIterator(map_by_id, id)`).
    pub fn at_id(map: &'a MapById<I, D>, id: &I) -> Self {
        let current = map
            .trajectories
            .get(&id.trajectory_id())
            .filter(|trajectory| trajectory.data.contains_key(&id.index()))
            .map(|_| (id.trajectory_id(), id.index()));
        ConstIterator { map, current }
    }

    /// Whether this is the end iterator.
    pub fn is_end(&self) -> bool {
        self.current.is_none()
    }

    /// The `(id, data)` pair at this position, or `None` at the end.
    pub fn get(&self) -> Option<IdDataReference<'a, I, D>> {
        let (t, i) = self.current?;
        let data = &self.map.trajectories[&t].data[&i];
        Some(IdDataReference {
            id: I::from_parts(t, i),
            data,
        })
    }

    /// The `(id, data)` pair at this position (C++ `operator*`). Panics at the
    /// end.
    pub fn item(&self) -> IdDataReference<'a, I, D> {
        self.get()
            .expect("Check failed: current_trajectory_ != end_trajectory_")
    }

    /// The ID at this position (C++ `it->id`). Panics at the end.
    pub fn id(&self) -> I {
        self.item().id
    }

    /// The data at this position (C++ `it->data`). Panics at the end.
    pub fn data(&self) -> &'a D {
        self.item().data
    }

    /// Moves to the next element (C++ `operator++`). Panics at the end.
    pub fn increment(&mut self) {
        let (t, i) = self
            .current
            .expect("Check failed: current_trajectory_ != end_trajectory_");
        let in_trajectory = self.map.trajectories[&t]
            .data
            .range((Excluded(i), Unbounded))
            .next()
            .map(|(&i, _)| (t, i));
        self.current = in_trajectory.or_else(|| {
            self.map
                .trajectories
                .range((Excluded(t), Unbounded))
                .find_map(|(&t, trajectory)| trajectory.data.keys().next().map(|&i| (t, i)))
        });
    }

    /// Moves to the previous element (C++ `operator--`). Panics if this is the
    /// first element.
    pub fn decrement(&mut self) {
        let previous = match self.current {
            None => self
                .map
                .trajectories
                .iter()
                .rev()
                .find_map(|(&t, trajectory)| trajectory.data.keys().next_back().map(|&i| (t, i))),
            Some((t, i)) => self.map.trajectories[&t]
                .data
                .range(..i)
                .next_back()
                .map(|(&i, _)| (t, i))
                .or_else(|| {
                    self.map
                        .trajectories
                        .range(..t)
                        .rev()
                        .find_map(|(&t, trajectory)| {
                            trajectory.data.keys().next_back().map(|&i| (t, i))
                        })
                }),
        };
        assert!(
            previous.is_some(),
            "Decremented the begin iterator of a MapById."
        );
        self.current = previous;
    }

    /// Returns the iterator to the next element (C++ `std::next(it)`).
    pub fn to_next(mut self) -> Self {
        self.increment();
        self
    }

    /// Returns the iterator to the previous element (C++ `std::prev(it)`).
    pub fn to_prev(mut self) -> Self {
        self.decrement();
        self
    }

    /// Range from this iterator to `end` (exclusive).
    pub fn range_to(self, end: Self) -> Range<'a, I, D> {
        Range::new(self, end)
    }
}

/// Iteration over the half-open range `[begin, end)` of a [`MapById`] (C++
/// `Range<ConstIterator>`), yielding [`IdDataReference`]s in
/// `(trajectory_id, index)` order.
#[derive(Debug)]
pub struct Range<'a, I, D> {
    begin: ConstIterator<'a, I, D>,
    end: ConstIterator<'a, I, D>,
}

impl<I, D> Clone for Range<'_, I, D> {
    fn clone(&self) -> Self {
        Range {
            begin: self.begin,
            end: self.end,
        }
    }
}

impl<'a, I: IdType, D> Range<'a, I, D> {
    /// Creates the range `[begin, end)`.
    pub fn new(begin: ConstIterator<'a, I, D>, end: ConstIterator<'a, I, D>) -> Self {
        Range { begin, end }
    }

    /// The (current) begin iterator.
    pub fn begin(&self) -> ConstIterator<'a, I, D> {
        self.begin
    }

    /// The end iterator.
    pub fn end(&self) -> ConstIterator<'a, I, D> {
        self.end
    }
}

impl<'a, I: IdType, D> Iterator for Range<'a, I, D> {
    type Item = IdDataReference<'a, I, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.begin == self.end {
            return None;
        }
        let item = self.begin.item();
        self.begin.increment();
        Some(item)
    }
}

impl<I: IdType, D> DoubleEndedIterator for Range<'_, I, D> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.begin == self.end {
            return None;
        }
        self.end.decrement();
        Some(self.end.item())
    }
}

impl<I: IdType, D> std::iter::FusedIterator for Range<'_, I, D> {}

impl<I: IdType, D> MapById<I, D> {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends data to a `trajectory_id`, creating trajectories as needed.
    pub fn append(&mut self, trajectory_id: i32, data: D) -> I {
        assert!(
            trajectory_id >= 0,
            "Check failed: trajectory_id >= 0 ({trajectory_id} vs. 0)"
        );
        let trajectory = self.trajectories.entry(trajectory_id).or_default();
        assert!(
            trajectory.can_append,
            "Check failed: trajectory.can_append_"
        );
        let index = trajectory
            .data
            .keys()
            .next_back()
            .map_or(0, |&last| last + 1);
        trajectory.data.insert(index, data);
        I::from_parts(trajectory_id, index)
    }

    /// Returns an iterator to the element at `id` or the end iterator if it
    /// does not exist.
    pub fn find(&self, id: &I) -> ConstIterator<'_, I, D> {
        ConstIterator::at_id(self, id)
    }

    /// Inserts data (which must not exist already) into a trajectory.
    pub fn insert(&mut self, id: &I, data: D) {
        assert!(
            id.trajectory_id() >= 0,
            "Check failed: id.trajectory_id >= 0 ({} vs. 0)",
            id.trajectory_id()
        );
        assert!(
            id.index() >= 0,
            "Check failed: GetIndex(id) >= 0 ({} vs. 0)",
            id.index()
        );
        let trajectory = self.trajectories.entry(id.trajectory_id()).or_default();
        trajectory.can_append = false;
        let previous = trajectory.data.insert(id.index(), data);
        assert!(
            previous.is_none(),
            "Check failed: data for {id} already exists"
        );
    }

    /// Removes the data for `id` which must exist.
    pub fn trim(&mut self, id: &I) {
        let trajectory = self
            .trajectories
            .get_mut(&id.trajectory_id())
            .unwrap_or_else(|| panic!("map::at: trajectory of {id} does not exist"));
        let last_index = trajectory.data.keys().next_back().copied();
        assert!(
            trajectory.data.contains_key(&id.index()),
            "Check failed: it != trajectory.data_.end() {id}"
        );
        if last_index == Some(id.index()) {
            // We are removing the data with the highest index from this
            // trajectory. We assume that we will never append to it anymore. If
            // we did, we would have to make sure that gaps in indices are
            // properly chosen to maintain correct connectivity.
            trajectory.can_append = false;
        }
        trajectory.data.remove(&id.index());
        if trajectory.data.is_empty() {
            self.trajectories.remove(&id.trajectory_id());
        }
    }

    /// Whether data for `id` exists.
    pub fn contains(&self, id: &I) -> bool {
        self.get(id).is_some()
    }

    /// The data for `id`, or `None` (Rust convenience).
    pub fn get(&self, id: &I) -> Option<&D> {
        self.trajectories
            .get(&id.trajectory_id())
            .and_then(|trajectory| trajectory.data.get(&id.index()))
    }

    /// The mutable data for `id`, or `None` (Rust convenience).
    pub fn get_mut(&mut self, id: &I) -> Option<&mut D> {
        self.trajectories
            .get_mut(&id.trajectory_id())
            .and_then(|trajectory| trajectory.data.get_mut(&id.index()))
    }

    /// The data for `id`, which must exist (C++ `at`, which throws).
    pub fn at(&self, id: &I) -> &D {
        self.get(id)
            .unwrap_or_else(|| panic!("MapById::at: {id} does not exist"))
    }

    /// The mutable data for `id`, which must exist (C++ non-const `at`).
    pub fn at_mut(&mut self, id: &I) -> &mut D {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("MapById::at: {id} does not exist"))
    }

    /// Iterator to the first element of trajectory `trajectory_id` (or of the
    /// next existing trajectory).
    pub fn begin_of_trajectory(&self, trajectory_id: i32) -> ConstIterator<'_, I, D> {
        ConstIterator::at_trajectory(self, trajectory_id)
    }

    /// Iterator past the last element of trajectory `trajectory_id`.
    pub fn end_of_trajectory(&self, trajectory_id: i32) -> ConstIterator<'_, I, D> {
        self.begin_of_trajectory(trajectory_id + 1)
    }

    /// Returns 0 if `trajectory_id` does not exist.
    pub fn size_of_trajectory_or_zero(&self, trajectory_id: i32) -> usize {
        self.trajectories
            .get(&trajectory_id)
            .map_or(0, |trajectory| trajectory.data.len())
    }

    /// Returns count of all elements.
    pub fn size(&self) -> usize {
        self.trajectories
            .values()
            .map(|trajectory| trajectory.data.len())
            .sum()
    }

    /// Iteration over the elements of a trajectory.
    pub fn trajectory(&self, trajectory_id: i32) -> Range<'_, I, D> {
        Range::new(
            self.begin_of_trajectory(trajectory_id),
            self.end_of_trajectory(trajectory_id),
        )
    }

    /// Iteration over the trajectory IDs, in increasing order.
    pub fn trajectory_ids(
        &self,
    ) -> impl DoubleEndedIterator<Item = i32> + ExactSizeIterator + Clone + '_ {
        self.trajectories.keys().copied()
    }

    /// The first element.
    pub fn begin(&self) -> ConstIterator<'_, I, D> {
        self.begin_of_trajectory(0)
    }

    /// The end iterator.
    pub fn end(&self) -> ConstIterator<'_, I, D> {
        self.begin_of_trajectory(i32::MAX)
    }

    /// Whether the map is empty.
    pub fn empty(&self) -> bool {
        self.begin() == self.end()
    }

    /// Iteration over all elements in `(trajectory_id, index)` order (C++
    /// range-based `for` over the map).
    pub fn iter(&self) -> Range<'_, I, D> {
        Range::new(self.begin(), self.end())
    }

    /// Mutable iteration over all elements in `(trajectory_id, index)` order.
    ///
    /// RUST PORT NOTE: C++ mutates through `at(id)` while iterating; Rust's
    /// borrow rules make this helper necessary.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (I, &mut D)> + '_ {
        self.trajectories.iter_mut().flat_map(|(&t, trajectory)| {
            trajectory
                .data
                .iter_mut()
                .map(move |(&i, data)| (I::from_parts(t, i), data))
        })
    }

    /// Mutable iteration over the elements of one trajectory (Rust
    /// convenience, see [`MapById::iter_mut`]).
    pub fn trajectory_mut(&mut self, trajectory_id: i32) -> impl Iterator<Item = (I, &mut D)> + '_ {
        self.trajectories
            .get_mut(&trajectory_id)
            .into_iter()
            .flat_map(move |trajectory| {
                trajectory
                    .data
                    .iter_mut()
                    .map(move |(&i, data)| (I::from_parts(trajectory_id, i), data))
            })
    }

    /// Returns an iterator to the first element in the container belonging to
    /// trajectory `trajectory_id` whose time is not considered to go before
    /// `time`, or `end_of_trajectory(trajectory_id)` if all keys are
    /// considered to go before `time`.
    pub fn lower_bound(&self, trajectory_id: i32, time: Time) -> ConstIterator<'_, I, D>
    where
        D: HasTime,
    {
        if self.size_of_trajectory_or_zero(trajectory_id) == 0 {
            return self.end_of_trajectory(trajectory_id);
        }

        let trajectory = &self.trajectories[&trajectory_id].data;

        let (&last_index, last) = trajectory.iter().next_back().unwrap();
        if last.time() < time {
            return self.end_of_trajectory(trajectory_id);
        }

        let mut left = *trajectory.keys().next().unwrap();
        let mut right = last_index;
        while left != right {
            // This is never 'right' which is important to guarantee progress.
            let middle = left + (right - left) / 2;
            // This could be 'right' in the presence of gaps, so we need to use
            // the previous element in this case.
            let (mut lower_bound_middle, mut lower_bound_middle_data) = trajectory
                .range((Included(middle), Unbounded))
                .next()
                .unwrap();
            if *lower_bound_middle > middle {
                assert!(
                    *lower_bound_middle != left,
                    "Check failed: lower_bound_middle != left"
                );
                (lower_bound_middle, lower_bound_middle_data) =
                    trajectory.range(..*lower_bound_middle).next_back().unwrap();
            }
            if lower_bound_middle_data.time() < time {
                left = *trajectory
                    .range((Excluded(*lower_bound_middle), Unbounded))
                    .next()
                    .unwrap()
                    .0;
            } else {
                right = *lower_bound_middle;
            }
        }

        ConstIterator::at_id(self, &I::from_parts(trajectory_id, left))
    }
}

impl<'a, I: IdType, D> IntoIterator for &'a MapById<I, D> {
    type Item = IdDataReference<'a, I, D>;
    type IntoIter = Range<'a, I, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<D: HasTime> crate::sensor::map_by_time::NodeTimes for MapById<NodeId, D> {
    fn neighbor_times(
        &self,
        trajectory_id: i32,
        node_index: i32,
    ) -> Option<(Option<Time>, Option<Time>)> {
        let data = &self.trajectories.get(&trajectory_id)?.data;
        data.get(&node_index)?;
        let previous = data.range(..node_index).next_back().map(|(_, d)| d.time());
        let next = data.range(node_index + 1..).next().map(|(_, d)| d.time());
        Some((previous, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_milliseconds;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::VecDeque;

    fn create_time(milliseconds: i64) -> Time {
        Time::from_time_since_epoch(from_milliseconds(milliseconds))
    }

    struct Data {
        time: Time,
    }

    impl Data {
        fn new(milliseconds: i64) -> Self {
            Data {
                time: create_time(milliseconds),
            }
        }
    }

    impl HasTime for Data {
        fn time(&self) -> Time {
            self.time
        }
    }

    #[test]
    fn node_times_neighbors() {
        use crate::sensor::map_by_time::NodeTimes;
        let mut map: MapById<NodeId, Data> = MapById::default();
        map.append(0, Data::new(10));
        map.append(0, Data::new(20));
        map.append(0, Data::new(30));
        assert_eq!(
            map.neighbor_times(0, 0),
            Some((None, Some(create_time(20))))
        );
        assert_eq!(
            map.neighbor_times(0, 1),
            Some((Some(create_time(10)), Some(create_time(30))))
        );
        assert_eq!(
            map.neighbor_times(0, 2),
            Some((Some(create_time(20)), None))
        );
        assert_eq!(map.neighbor_times(0, 3), None);
        assert_eq!(map.neighbor_times(1, 0), None);
    }

    fn create_test_map_by_id<I: IdType>() -> MapById<I, i32> {
        let mut map_by_id = MapById::new();
        map_by_id.append(7, 2);
        map_by_id.append(42, 3);
        map_by_id.append(0, 0);
        map_by_id.append(0, 1);
        map_by_id
    }

    /// `std::lower_bound` over `[begin_of_trajectory, end_of_trajectory)`.
    fn ground_truth_lower_bound<D: HasTime>(
        map_by_id: &MapById<SubmapId, D>,
        trajectory_id: i32,
        time: Time,
    ) -> ConstIterator<'_, SubmapId, D> {
        match map_by_id
            .trajectory(trajectory_id)
            .find(|a| a.data.time() >= time)
        {
            Some(entry) => map_by_id.find(&entry.id),
            None => map_by_id.end_of_trajectory(trajectory_id),
        }
    }

    mod id_test {
        use super::*;

        #[test]
        fn empty_map_by_id() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            assert!(map_by_id.empty());
            assert_eq!(map_by_id.trajectory_ids().next(), None);
            let unknown_trajectory_id = 3;
            assert_eq!(
                map_by_id.trajectory(unknown_trajectory_id).begin(),
                map_by_id.trajectory(unknown_trajectory_id).end()
            );
            let id = map_by_id.append(42, 42);
            assert!(!map_by_id.empty());
            map_by_id.trim(&id);
            assert!(map_by_id.empty());
            assert_eq!(0, map_by_id.size());
        }

        #[test]
        fn delete_trajectory() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            let trajectory_id = 3;
            let other_trajectory_id = 5;
            map_by_id.insert(&NodeId::new(trajectory_id, 4), 5);
            map_by_id.insert(&NodeId::new(trajectory_id, 5), 7);
            map_by_id.insert(&NodeId::new(other_trajectory_id, 1), 3);
            assert_eq!(map_by_id.size(), 3);
            assert_eq!(2, map_by_id.trajectory_ids().count());
            map_by_id.trim(&NodeId::new(trajectory_id, 4));
            map_by_id.trim(&NodeId::new(trajectory_id, 5));
            assert_eq!(0, map_by_id.trajectory(trajectory_id).count());
            let invalid_trajectory_id = 2;
            assert_eq!(
                map_by_id.trajectory(invalid_trajectory_id).begin(),
                map_by_id.trajectory(invalid_trajectory_id).end()
            );
            assert_eq!(map_by_id.size(), 1);
            assert_eq!(1, map_by_id.trajectory(other_trajectory_id).count());
            assert_eq!(1, map_by_id.trajectory_ids().count());
            assert!(!map_by_id.empty());
        }

        #[test]
        fn map_by_id_iterator() {
            let map_by_id = create_test_map_by_id::<NodeId>();
            assert_eq!(4, map_by_id.size());
            assert_eq!(2, *map_by_id.begin_of_trajectory(7).data());
            assert!(map_by_id.begin_of_trajectory(7).to_next() == map_by_id.end_of_trajectory(7));
            let mut expected_id_data: VecDeque<(NodeId, i32)> = VecDeque::from([
                (NodeId::new(0, 0), 0),
                (NodeId::new(0, 1), 1),
                (NodeId::new(7, 0), 2),
                (NodeId::new(42, 0), 3),
            ]);
            for id_data in &map_by_id {
                let front = expected_id_data.pop_front().expect("unexpected element");
                assert_eq!(front.0, id_data.id);
                assert_eq!(front.1, *id_data.data);
            }
            assert!(expected_id_data.is_empty());
        }

        #[test]
        fn map_by_id_trajectory_range() {
            let map_by_id = create_test_map_by_id::<NodeId>();
            let mut expected_data: VecDeque<(NodeId, i32)> =
                VecDeque::from([(NodeId::new(0, 0), 0), (NodeId::new(0, 1), 1)]);
            for entry in map_by_id.trajectory(0) {
                let front = expected_data.pop_front().expect("unexpected element");
                assert_eq!(front.0, entry.id);
                assert_eq!(front.1, *entry.data);
            }
            assert!(expected_data.is_empty());
        }

        #[test]
        fn map_by_id_trajectory_id_range() {
            let map_by_id = create_test_map_by_id::<NodeId>();
            let mut expected_data: VecDeque<i32> = VecDeque::from([0, 7, 42]);
            for trajectory_id in map_by_id.trajectory_ids() {
                let front = expected_data.pop_front().expect("unexpected element");
                assert_eq!(front, trajectory_id);
            }
            assert!(expected_data.is_empty());
        }

        #[test]
        fn map_by_id_iterate_by_trajectories() {
            let map_by_id = create_test_map_by_id::<NodeId>();
            let mut expected_id_data: VecDeque<(NodeId, i32)> = VecDeque::from([
                (NodeId::new(0, 0), 0),
                (NodeId::new(0, 1), 1),
                (NodeId::new(7, 0), 2),
                (NodeId::new(42, 0), 3),
            ]);
            for trajectory_id in map_by_id.trajectory_ids() {
                for entry in map_by_id.trajectory(trajectory_id) {
                    let front = expected_id_data.pop_front().expect("unexpected element");
                    assert_eq!(front.0, entry.id);
                    assert_eq!(front.1, *entry.data);
                }
            }
            assert!(expected_id_data.is_empty());
        }

        #[test]
        fn map_by_id_prev_iterator() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            map_by_id.append(42, 42);
            let mut it = map_by_id.end();
            assert!(it != map_by_id.begin());
            it.decrement();
            assert!(it == map_by_id.begin());
        }

        #[test]
        fn insert_into_map_by_id() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            assert_eq!(0, map_by_id.size_of_trajectory_or_zero(42));
            map_by_id.append(42, 42);
            map_by_id.insert(&NodeId::new(42, 5), 42);
            assert_eq!(2, map_by_id.size_of_trajectory_or_zero(42));
        }

        #[test]
        fn find_node_id() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            map_by_id.append(42, 42);
            map_by_id.append(42, 43);
            map_by_id.append(42, 44);
            assert_eq!(*map_by_id.find(&NodeId::new(42, 1)).data(), 43);
            assert!(map_by_id.find(&NodeId::new(41, 0)) == map_by_id.end());
            assert!(map_by_id.find(&NodeId::new(42, 3)) == map_by_id.end());
        }

        #[test]
        fn find_submap_id() {
            let mut map_by_id = MapById::<SubmapId, i32>::new();
            map_by_id.append(42, 42);
            map_by_id.append(42, 43);
            map_by_id.append(42, 44);
            assert_eq!(*map_by_id.find(&SubmapId::new(42, 1)).data(), 43);
            assert!(map_by_id.find(&SubmapId::new(42, 3)) == map_by_id.end());
        }

        #[test]
        fn lower_bound_edge_cases() {
            let mut map_by_id = MapById::<SubmapId, Data>::new();
            map_by_id.append(0, Data::new(1));
            map_by_id.append(2, Data::new(2));
            assert!(map_by_id.lower_bound(1, create_time(10)) == map_by_id.end_of_trajectory(1));
            assert!(map_by_id.lower_bound(2, create_time(3)) == map_by_id.end_of_trajectory(2));
            assert!(map_by_id.lower_bound(2, create_time(1)) == map_by_id.begin_of_trajectory(2));
        }

        #[test]
        fn lower_bound() {
            let mut map_by_id = MapById::<SubmapId, Data>::new();
            map_by_id.append(0, Data::new(1));
            map_by_id.append(0, Data::new(2));
            map_by_id.append(0, Data::new(4));
            map_by_id.append(0, Data::new(5));
            assert!(
                map_by_id.lower_bound(0, create_time(3))
                    == ConstIterator::at_id(&map_by_id, &SubmapId::new(0, 2))
            );
            assert!(
                map_by_id.lower_bound(0, create_time(2))
                    == ConstIterator::at_id(&map_by_id, &SubmapId::new(0, 1))
            );
            assert!(
                map_by_id.lower_bound(0, create_time(4))
                    == ConstIterator::at_id(&map_by_id, &SubmapId::new(0, 2))
            );
        }

        #[test]
        fn lower_bound_fuzz() {
            const MAX_TIME_INCREMENT: i64 = 20;
            const MAX_NUM_NODES: i64 = 20;
            const NUM_TESTS: i32 = 100;
            const TRAJECTORY_ID: i32 = 1;

            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..NUM_TESTS {
                let n = rng.random_range(1..=MAX_NUM_NODES);
                let mut t = 0;
                let mut map_by_id = MapById::<SubmapId, Data>::new();
                for _ in 0..n {
                    t += rng.random_range(1..=MAX_TIME_INCREMENT);
                    map_by_id.append(TRAJECTORY_ID, Data::new(t));
                }
                let t0 = rng.random_range(1..=n * MAX_TIME_INCREMENT + 1);
                let it = map_by_id.lower_bound(TRAJECTORY_ID, create_time(t0));
                let ground_truth =
                    ground_truth_lower_bound(&map_by_id, TRAJECTORY_ID, create_time(t0));
                assert!(ground_truth == it);
            }
        }

        #[test]
        fn lower_bound_trimmed_trajectory() {
            const TRAJECTORY_ID: i32 = 1;
            let mut rng = StdRng::seed_from_u64(0);

            const N: usize = 500;
            let mut t = 0;
            let mut map_by_id = MapById::<SubmapId, Data>::new();
            for _ in 0..N {
                t += rng.random_range(1..=20);
                map_by_id.append(TRAJECTORY_ID, Data::new(t));
            }

            // Choose random length of a trim segment.
            let trim_segment_length = rng.random_range(1..=N / 2);
            // Choose random start for a trim_segment.
            let trim_segment_start_index = rng.random_range(2..=N - trim_segment_length - 1);

            let ids: Vec<SubmapId> = map_by_id
                .iter()
                .skip(trim_segment_start_index)
                .take(trim_segment_length)
                .map(|entry| entry.id)
                .collect();
            for id in &ids {
                map_by_id.trim(id);
            }

            let it = map_by_id.lower_bound(TRAJECTORY_ID, create_time(0));
            let ground_truth = ground_truth_lower_bound(&map_by_id, TRAJECTORY_ID, create_time(0));
            assert_eq!(ground_truth, it);
        }

        struct DataStruct {
            time: Time,
        }

        impl HasTime for DataStruct {
            fn time(&self) -> Time {
                self.time
            }
        }

        #[test]
        fn lower_bound_fuzz_with_struct() {
            const MAX_TIME_INCREMENT: i64 = 20;
            const MAX_NUM_NODES: i64 = 20;
            const NUM_TESTS: i32 = 100;
            const TRAJECTORY_ID: i32 = 1;

            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..NUM_TESTS {
                let n = rng.random_range(1..=MAX_NUM_NODES);
                let mut t = 0;
                let mut map_by_id = MapById::<SubmapId, DataStruct>::new();
                for _ in 0..n {
                    t += rng.random_range(1..=MAX_TIME_INCREMENT);
                    map_by_id.append(
                        TRAJECTORY_ID,
                        DataStruct {
                            time: create_time(t),
                        },
                    );
                }
                let t0 = rng.random_range(1..=n * MAX_TIME_INCREMENT + 1);
                let it = map_by_id.lower_bound(TRAJECTORY_ID, create_time(t0));
                let ground_truth =
                    ground_truth_lower_bound(&map_by_id, TRAJECTORY_ID, create_time(t0));
                assert!(ground_truth == it);
            }
        }
    }

    /// Additional Rust-side checks (not in the C++ suite).
    mod rust_extras {
        use super::*;

        #[test]
        fn display_and_proto_round_trip() {
            let node_id = NodeId::new(3, 7);
            assert_eq!(node_id.to_string(), "(3, 7)");
            assert_eq!(NodeId::from_proto(&node_id.to_proto()), node_id);
            let submap_id = SubmapId::new(1, 2);
            assert_eq!(submap_id.to_string(), "(1, 2)");
            assert_eq!(SubmapId::from_proto(&submap_id.to_proto()), submap_id);
            assert!(NodeId::new(0, 5) < NodeId::new(1, 0));
            assert!(SubmapId::new(1, 0) < SubmapId::new(1, 1));
        }

        #[test]
        fn bidirectional_iteration_with_gaps() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            map_by_id.insert(&NodeId::new(1, 3), 13);
            map_by_id.insert(&NodeId::new(1, 9), 19);
            map_by_id.insert(&NodeId::new(4, 0), 40);
            let forward: Vec<i32> = map_by_id.iter().map(|e| *e.data).collect();
            assert_eq!(forward, vec![13, 19, 40]);
            let backward: Vec<i32> = map_by_id.iter().rev().map(|e| *e.data).collect();
            assert_eq!(backward, vec![40, 19, 13]);
            let mut it = map_by_id.end();
            it.decrement();
            assert_eq!(it.id(), NodeId::new(4, 0));
            it.decrement();
            assert_eq!(it.id(), NodeId::new(1, 9));
            assert_eq!(
                map_by_id.end_of_trajectory(1).to_prev().id(),
                NodeId::new(1, 9)
            );
            for (_, data) in map_by_id.iter_mut() {
                *data += 1;
            }
            assert_eq!(*map_by_id.at(&NodeId::new(1, 9)), 20);
            assert!(map_by_id.contains(&NodeId::new(1, 3)));
            assert!(!map_by_id.contains(&NodeId::new(1, 4)));
        }

        #[test]
        #[should_panic]
        fn append_after_insert_panics() {
            let mut map_by_id = MapById::<NodeId, i32>::new();
            map_by_id.insert(&NodeId::new(0, 3), 0);
            map_by_id.append(0, 1);
        }
    }
}
