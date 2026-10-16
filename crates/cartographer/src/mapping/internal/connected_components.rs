//! Port of `cartographer/mapping/internal/connected_components` (.h, .cc).

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::mapping::proto;

#[derive(Debug, Default)]
struct State {
    // Tracks transitive connectivity using a disjoint set forest, i.e. each
    // entry points towards the representative for the given trajectory.
    forest: BTreeMap<i32, i32>,
    // Tracks the number of direct connections between a pair of trajectories.
    connection_map: BTreeMap<(i32, i32), i32>,
}

impl State {
    // Find the representative and compresses the path to it.
    fn find_set(&mut self, trajectory_id: i32) -> i32 {
        let parent = *self
            .forest
            .get(&trajectory_id)
            .expect("Check failed: it != forest_.end()");
        if parent != trajectory_id {
            // Path compression for efficiency.
            let representative = self.find_set(parent);
            self.forest.insert(trajectory_id, representative);
            return representative;
        }
        parent
    }

    fn union(&mut self, trajectory_id_a: i32, trajectory_id_b: i32) {
        self.forest
            .entry(trajectory_id_a)
            .or_insert(trajectory_id_a);
        self.forest
            .entry(trajectory_id_b)
            .or_insert(trajectory_id_b);
        let representative_a = self.find_set(trajectory_id_a);
        let representative_b = self.find_set(trajectory_id_b);
        self.forest.insert(representative_a, representative_b);
    }
}

/// `std::minmax` of two trajectory IDs.
fn minmax(a: i32, b: i32) -> (i32, i32) {
    if b < a { (b, a) } else { (a, b) }
}

/// A class that tracks the connectivity structure between trajectories.
///
/// Connectivity includes both the count ("How many times have I _directly_
/// connected trajectories i and j?") and the transitive connectivity.
///
/// This class is thread-safe.
#[derive(Debug, Default)]
pub struct ConnectedComponents {
    lock: Mutex<State>,
}

impl ConnectedComponents {
    /// Creates an empty structure.
    pub fn new() -> Self {
        Self::default()
    }

    fn locked(&self) -> MutexGuard<'_, State> {
        self.lock.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Add a trajectory which is initially connected to only itself.
    pub fn add(&self, trajectory_id: i32) {
        self.locked()
            .forest
            .entry(trajectory_id)
            .or_insert(trajectory_id);
    }

    /// Connect two trajectories. If either trajectory is untracked, it will be
    /// tracked. This function is invariant to the order of its arguments.
    /// Repeated calls to `connect` increment the connectivity count.
    pub fn connect(&self, trajectory_id_a: i32, trajectory_id_b: i32) {
        let mut state = self.locked();
        state.union(trajectory_id_a, trajectory_id_b);
        *state
            .connection_map
            .entry(minmax(trajectory_id_a, trajectory_id_b))
            .or_insert(0) += 1;
    }

    /// Determines if two trajectories have been (transitively) connected. If
    /// either trajectory is not being tracked, returns false, except when it
    /// is the same trajectory, where it returns true. This function is
    /// invariant to the order of its arguments.
    pub fn transitively_connected(&self, trajectory_id_a: i32, trajectory_id_b: i32) -> bool {
        if trajectory_id_a == trajectory_id_b {
            return true;
        }

        let mut state = self.locked();
        if !state.forest.contains_key(&trajectory_id_a)
            || !state.forest.contains_key(&trajectory_id_b)
        {
            return false;
        }
        state.find_set(trajectory_id_a) == state.find_set(trajectory_id_b)
    }

    /// Return the number of _direct_ connections between `trajectory_id_a`
    /// and `trajectory_id_b`. If either trajectory is not being tracked,
    /// returns 0. This function is invariant to the order of its arguments.
    pub fn connection_count(&self, trajectory_id_a: i32, trajectory_id_b: i32) -> i32 {
        self.locked()
            .connection_map
            .get(&minmax(trajectory_id_a, trajectory_id_b))
            .copied()
            .unwrap_or(0)
    }

    /// The trajectory IDs, grouped by connectivity.
    ///
    // RUST PORT NOTE: C++ groups through an `absl::flat_hash_map`, so the
    // order of the components is unspecified. Here the components are ordered
    // by their representative (each component is sorted ascending, as in
    // C++), which makes the output deterministic.
    pub fn components(&self) -> Vec<Vec<i32>> {
        // Map from cluster exemplar -> growing cluster.
        let mut map: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let mut state = self.locked();
        let ids: Vec<i32> = state.forest.keys().copied().collect();
        for trajectory_id in ids {
            let representative = state.find_set(trajectory_id);
            map.entry(representative).or_default().push(trajectory_id);
        }
        map.into_values().collect()
    }

    /// The list of trajectory IDs that belong to the same connected component
    /// as `trajectory_id`. Panics if `trajectory_id` is untracked (C++
    /// `CHECK`).
    pub fn get_component(&self, trajectory_id: i32) -> Vec<i32> {
        let mut state = self.locked();
        let set_id = state.find_set(trajectory_id);
        let ids: Vec<i32> = state.forest.keys().copied().collect();
        ids.into_iter()
            .filter(|&id| state.find_set(id) == set_id)
            .collect()
    }
}

/// Returns a proto encoding connected components.
pub fn to_proto(mut connected_components: Vec<Vec<i32>>) -> proto::ConnectedComponents {
    for connected_component in &mut connected_components {
        connected_component.sort();
    }
    connected_components.sort();
    proto::ConnectedComponents {
        connected_component: connected_components
            .into_iter()
            .map(|trajectory_id| proto::connected_components::ConnectedComponent { trajectory_id })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    mod connected_components_test {
        use crate::mapping::internal::connected_components::*;

        const NUM_TRAJECTORIES: i32 = 10;

        #[test]
        fn transitively_connected() {
            let connected_components = ConnectedComponents::new();

            // Make sure nothing's connected until we connect some things.
            for trajectory_a in 0..NUM_TRAJECTORIES {
                for trajectory_b in 0..NUM_TRAJECTORIES {
                    assert_eq!(
                        trajectory_a == trajectory_b,
                        connected_components.transitively_connected(trajectory_a, trajectory_b)
                    );
                }
            }

            // Connect some stuff up.
            connected_components.connect(0, 1);
            assert!(connected_components.transitively_connected(0, 1));
            connected_components.connect(8, 9);
            assert!(connected_components.transitively_connected(8, 9));
            assert!(!connected_components.transitively_connected(0, 9));

            connected_components.connect(1, 8);
            for i in [0, 1] {
                for j in [8, 9] {
                    assert!(connected_components.transitively_connected(i, j));
                }
            }
        }

        #[test]
        fn empty_connected_components() {
            let connected_components = ConnectedComponents::new();
            let connections = connected_components.components();
            assert_eq!(0, connections.len());
        }

        #[test]
        fn connected_components() {
            let connected_components = ConnectedComponents::new();
            for i in 0..=4 {
                connected_components.connect(0, i);
            }
            for i in 5..=9 {
                connected_components.connect(5, i);
            }
            let connections = connected_components.components();
            assert_eq!(2, connections.len());
            // The clustering is arbitrary; we need to figure out which one is
            // which.
            let (zero_cluster, five_cluster) = if connections[0].contains(&0) {
                (&connections[0], &connections[1])
            } else {
                (&connections[1], &connections[0])
            };
            for i in 0..=9 {
                assert_eq!(i <= 4, zero_cluster.contains(&i));
                assert_eq!(i > 4, five_cluster.contains(&i));
            }
        }

        #[test]
        fn connection_count() {
            let connected_components = ConnectedComponents::new();
            for i in 0..NUM_TRAJECTORIES {
                connected_components.connect(0, 1);
                // Permute the arguments to check invariance.
                assert_eq!(i + 1, connected_components.connection_count(1, 0));
            }
            for i in 1..9 {
                assert_eq!(0, connected_components.connection_count(i, i + 1));
            }
        }

        #[test]
        fn reflexive_connectivity() {
            let connected_components = ConnectedComponents::new();
            assert!(connected_components.transitively_connected(0, 0));
            assert_eq!(0, connected_components.connection_count(0, 0));
            connected_components.add(0);
            assert!(connected_components.transitively_connected(0, 0));
            assert_eq!(0, connected_components.connection_count(0, 0));
        }
    }

    // Additional Rust tests (no C++ counterpart).
    mod extra {
        use crate::mapping::internal::connected_components::*;

        #[test]
        fn get_component_and_to_proto() {
            let connected_components = ConnectedComponents::new();
            connected_components.add(7);
            connected_components.connect(3, 1);
            connected_components.connect(1, 2);
            assert_eq!(connected_components.get_component(2), vec![1, 2, 3]);
            assert_eq!(connected_components.get_component(7), vec![7]);
            let proto = to_proto(connected_components.components());
            let components: Vec<Vec<i32>> = proto
                .connected_component
                .iter()
                .map(|c| c.trajectory_id.clone())
                .collect();
            assert_eq!(components, vec![vec![1, 2, 3], vec![7]]);
        }

        #[test]
        #[should_panic(expected = "forest_.end()")]
        fn get_component_of_untracked_trajectory_panics() {
            ConnectedComponents::new().get_component(0);
        }
    }
}
