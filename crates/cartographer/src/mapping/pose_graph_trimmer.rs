//! Port of `cartographer/mapping/pose_graph_trimmer` (.h, .cc).

use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::pose_graph_interface::{Constraint, SubmapData, TrajectoryState};
use crate::mapping::trajectory_node::TrajectoryNode;

/// Implemented by the pose graph to provide thread-safe access to functions
/// for trimming the graph.
pub trait Trimmable {
    /// Number of (optimized) submaps of `trajectory_id`.
    fn num_submaps(&self, trajectory_id: i32) -> i32;

    /// IDs of the (optimized) submaps of `trajectory_id`.
    fn get_submap_ids(&self, trajectory_id: i32) -> Vec<SubmapId>;
    /// Returns finished submaps with optimized poses only.
    fn get_optimized_submap_data(&self) -> MapById<SubmapId, SubmapData>;
    /// All trajectory nodes.
    fn get_trajectory_nodes(&self) -> &MapById<NodeId, TrajectoryNode>;
    /// All constraints.
    fn get_constraints(&self) -> &[Constraint];

    /// Trim `submap_id` and corresponding intra-submap nodes. They will no
    /// longer take part in scan matching, loop closure, visualization. The
    /// numbering remains unchanged.
    fn trim_submap(&mut self, submap_id: &SubmapId);

    /// Checks if the given trajectory is finished or not.
    fn is_finished(&self, trajectory_id: i32) -> bool;

    /// Sets the state for a specific trajectory.
    fn set_trajectory_state(&mut self, trajectory_id: i32, state: TrajectoryState);
}

/// An interface to implement algorithms that choose how to trim the pose
/// graph.
///
// RUST PORT NOTE: `Send` because the pose graph owns its trimmers and runs
// them on thread-pool threads.
pub trait PoseGraphTrimmer: Send {
    /// Called once after each pose graph optimization.
    fn trim(&mut self, pose_graph: &mut dyn Trimmable);

    /// Checks if this trimmer is in a terminatable state.
    fn is_finished(&self) -> bool;
}

/// Keeps the last `num_submaps_to_keep` of the trajectory with
/// `trajectory_id` to implement localization without mapping.
#[derive(Debug)]
pub struct PureLocalizationTrimmer {
    trajectory_id: i32,
    num_submaps_to_keep: i32,
    finished: bool,
}

impl PureLocalizationTrimmer {
    /// Creates the trimmer. Panics if `num_submaps_to_keep < 2`.
    pub fn new(trajectory_id: i32, num_submaps_to_keep: i32) -> Self {
        assert!(
            num_submaps_to_keep >= 2,
            "Check failed: num_submaps_to_keep >= 2 ({num_submaps_to_keep} vs. 2) \
             Cannot trim with less than 2 submaps"
        );
        Self {
            trajectory_id,
            num_submaps_to_keep,
            finished: false,
        }
    }
}

impl PoseGraphTrimmer for PureLocalizationTrimmer {
    fn trim(&mut self, pose_graph: &mut dyn Trimmable) {
        if pose_graph.is_finished(self.trajectory_id) {
            self.num_submaps_to_keep = 0;
        }

        let submap_ids = pose_graph.get_submap_ids(self.trajectory_id);
        let num_submaps_to_keep = self.num_submaps_to_keep as usize;
        let mut i = 0usize;
        while i + num_submaps_to_keep < submap_ids.len() {
            pose_graph.trim_submap(&submap_ids[i]);
            i += 1;
        }

        if self.num_submaps_to_keep == 0 {
            self.finished = true;
            pose_graph.set_trajectory_state(self.trajectory_id, TrajectoryState::Deleted);
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    mod pure_localization_trimmer_test {
        use crate::mapping::id::SubmapId;
        use crate::mapping::internal::testing::fake_trimmable::FakeTrimmable;
        use crate::mapping::pose_graph_trimmer::*;

        #[test]
        fn marks_submaps_as_expected() {
            const TRAJECTORY_ID: i32 = 42;
            let mut trimmer = PureLocalizationTrimmer::new(TRAJECTORY_ID, 15);
            let mut fake_pose_graph = FakeTrimmable::with_submaps(TRAJECTORY_ID, 17);
            trimmer.trim(&mut fake_pose_graph);

            let trimmed_submaps = fake_pose_graph.trimmed_submaps();
            assert_eq!(2, trimmed_submaps.len());
            assert_eq!(SubmapId::new(TRAJECTORY_ID, 0), trimmed_submaps[0]);
            assert_eq!(SubmapId::new(TRAJECTORY_ID, 1), trimmed_submaps[1]);
        }
    }

    // Additional Rust tests (no C++ counterpart).
    mod extra {
        use crate::mapping::id::SubmapId;
        use crate::mapping::pose_graph_trimmer::*;

        #[derive(Default)]
        struct FinishedTrimmable {
            trimmed: Vec<SubmapId>,
            states: Vec<(i32, TrajectoryState)>,
            nodes: MapById<NodeId, TrajectoryNode>,
        }

        impl Trimmable for FinishedTrimmable {
            fn num_submaps(&self, _: i32) -> i32 {
                3
            }
            fn get_submap_ids(&self, trajectory_id: i32) -> Vec<SubmapId> {
                (0..3).map(|i| SubmapId::new(trajectory_id, i)).collect()
            }
            fn get_optimized_submap_data(&self) -> MapById<SubmapId, SubmapData> {
                MapById::new()
            }
            fn get_trajectory_nodes(&self) -> &MapById<NodeId, TrajectoryNode> {
                &self.nodes
            }
            fn get_constraints(&self) -> &[Constraint] {
                &[]
            }
            fn trim_submap(&mut self, submap_id: &SubmapId) {
                self.trimmed.push(*submap_id);
            }
            fn is_finished(&self, _: i32) -> bool {
                true
            }
            fn set_trajectory_state(&mut self, trajectory_id: i32, state: TrajectoryState) {
                self.states.push((trajectory_id, state));
            }
        }

        #[test]
        fn finished_trajectory_is_fully_trimmed_and_deleted() {
            let mut trimmer = PureLocalizationTrimmer::new(1, 2);
            assert!(!trimmer.is_finished());
            let mut pose_graph = FinishedTrimmable::default();
            trimmer.trim(&mut pose_graph);
            assert!(trimmer.is_finished());
            assert_eq!(pose_graph.trimmed.len(), 3);
            assert_eq!(pose_graph.states, vec![(1, TrajectoryState::Deleted)]);
        }

        #[test]
        #[should_panic(expected = "Cannot trim with less than 2 submaps")]
        fn keeping_less_than_two_submaps_panics() {
            PureLocalizationTrimmer::new(0, 1);
        }
    }
}
