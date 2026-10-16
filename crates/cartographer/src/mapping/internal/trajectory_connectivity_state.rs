//! Port of `cartographer/mapping/internal/trajectory_connectivity_state` (.h, .cc).

use std::collections::BTreeMap;

use crate::common::time::Time;
use crate::mapping::internal::connected_components::ConnectedComponents;

/// `std::minmax` of two trajectory IDs.
fn minmax(a: i32, b: i32) -> (i32, i32) {
    if b < a { (b, a) } else { (a, b) }
}

/// A class that tracks the connectivity state between trajectories. Compared
/// to [`ConnectedComponents`] it tracks additionally the last time that a
/// global constraint connected to trajectories.
///
/// This class is thread-compatible.
#[derive(Debug, Default)]
pub struct TrajectoryConnectivityState {
    // ConnectedComponents are thread safe.
    connected_components: ConnectedComponents,

    // Tracks the last time a direct connection between two trajectories has
    // been added. The exception is when a connection between two trajectories
    // connects two formerly unconnected connected components. In this case all
    // bipartite trajectories entries for these components are updated with the
    // new connection time.
    last_connection_time_map: BTreeMap<(i32, i32), Time>,
}

impl TrajectoryConnectivityState {
    /// Creates an empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trajectory which is initially connected to only itself.
    pub fn add(&mut self, trajectory_id: i32) {
        self.connected_components.add(trajectory_id);
    }

    /// Connect two trajectories. If either trajectory is untracked, it will
    /// be tracked. This function is invariant to the order of its arguments.
    /// Repeated calls to `connect` increment the connectivity count and update
    /// the last connected time.
    ///
    /// Like in C++, connecting two formerly unconnected trajectories of which
    /// one is untracked panics (`GetComponent` CHECK); callers `add` first.
    pub fn connect(&mut self, trajectory_id_a: i32, trajectory_id_b: i32, time: Time) {
        if self.transitively_connected(trajectory_id_a, trajectory_id_b) {
            // The trajectories are transitively connected, i.e. they belong to
            // the same connected component. In this case we only update the
            // last connection time of those two trajectories.
            let last = self
                .last_connection_time_map
                .entry(minmax(trajectory_id_a, trajectory_id_b))
                .or_default();
            if *last < time {
                *last = time;
            }
        } else {
            // The connection between these two trajectories is about to join
            // to connected components. Here we update all bipartite trajectory
            // pairs for the two connected components with the connection time.
            // This is to quickly change to a more efficient loop closure
            // search (by constraining the search window) when connected
            // components are joined.
            let component_a = self.connected_components.get_component(trajectory_id_a);
            let component_b = self.connected_components.get_component(trajectory_id_b);
            for &id_a in &component_a {
                for &id_b in &component_b {
                    self.last_connection_time_map
                        .insert(minmax(id_a, id_b), time);
                }
            }
        }
        self.connected_components
            .connect(trajectory_id_a, trajectory_id_b);
    }

    /// Determines if two trajectories have been (transitively) connected. If
    /// either trajectory is not being tracked, returns false, except when it
    /// is the same trajectory, where it returns true. This function is
    /// invariant to the order of its arguments.
    pub fn transitively_connected(&self, trajectory_id_a: i32, trajectory_id_b: i32) -> bool {
        self.connected_components
            .transitively_connected(trajectory_id_a, trajectory_id_b)
    }

    /// The trajectory IDs, grouped by connectivity.
    pub fn components(&self) -> Vec<Vec<i32>> {
        self.connected_components.components()
    }

    /// Return the last connection count between the two trajectories. If
    /// either of the trajectories is untracked or they have never been
    /// connected returns the beginning of time.
    ///
    /// Takes `&mut self` because, like the C++ `operator[]`, it inserts the
    /// default entry for a pair never seen before.
    pub fn last_connection_time(&mut self, trajectory_id_a: i32, trajectory_id_b: i32) -> Time {
        *self
            .last_connection_time_map
            .entry(minmax(trajectory_id_a, trajectory_id_b))
            .or_default()
    }
}

#[cfg(test)]
mod tests {
    mod trajectory_connectivity_state_test {
        use crate::common::time::{Time, from_universal};
        use crate::mapping::internal::trajectory_connectivity_state::*;

        #[test]
        fn unknown_trajectory() {
            let mut state = TrajectoryConnectivityState::new();
            state.add(0);

            // Nothing is transitively connected to an unknown trajectory.
            assert!(!state.transitively_connected(0, 1));
            assert_eq!(state.last_connection_time(0, 1), Time::default());
        }

        #[test]
        fn not_connected() {
            let mut state = TrajectoryConnectivityState::new();
            state.add(0);
            state.add(1);
            assert!(!state.transitively_connected(0, 1));
            assert_eq!(state.last_connection_time(0, 1), Time::default());
        }

        #[test]
        fn connected() {
            let mut state = TrajectoryConnectivityState::new();
            state.add(0);
            state.add(1);
            state.connect(0, 1, from_universal(123456));
            assert!(state.transitively_connected(0, 1));
            assert_eq!(state.last_connection_time(0, 1), from_universal(123456));
        }

        #[test]
        fn update_connection_time() {
            let mut state = TrajectoryConnectivityState::new();
            state.add(0);
            state.add(1);
            state.connect(0, 1, from_universal(123456));
            state.connect(0, 1, from_universal(234567));
            assert!(state.transitively_connected(0, 1));
            assert_eq!(state.last_connection_time(0, 1), from_universal(234567));

            // Connections with an earlier connection time do not update the
            // last connection time.
            state.connect(0, 1, from_universal(123456));
            assert_eq!(state.last_connection_time(0, 1), from_universal(234567));
        }

        #[test]
        fn join_two_components() {
            let mut state = TrajectoryConnectivityState::new();
            state.add(0);
            state.add(1);
            state.add(2);
            state.add(3);
            state.connect(0, 1, from_universal(123456));
            state.connect(2, 3, from_universal(123456));

            // Connect the two disjoint connected components.
            state.connect(0, 2, from_universal(234567));
            assert!(state.transitively_connected(0, 2));
            assert!(state.transitively_connected(1, 3));

            // All bipartite trajectory pairs between the two connected
            // components should have the updated connection time.
            assert_eq!(state.last_connection_time(0, 2), from_universal(234567));
            assert_eq!(state.last_connection_time(0, 3), from_universal(234567));
            assert_eq!(state.last_connection_time(1, 3), from_universal(234567));

            // A pair of trajectory IDs belonging to the same connected
            // component should be unaffected.
            assert_eq!(state.last_connection_time(0, 1), from_universal(123456));
        }
    }
}
