//! Dimension-agnostic parts of `PoseGraph2D` and `PoseGraph3D`.
//!
//! RUST PORT NOTE: no C++ counterpart. `pose_graph_2d.cc` and
//! `pose_graph_3d.cc` duplicate these member functions (and the
//! `TrimmingHandle` class) almost verbatim; here they are written once,
//! generic over the submap type `S` (`Submap2D`/`Submap3D`) and the
//! optimization problem, and called by both pose graphs. Everything that
//! differs between 2D and 3D (pose representations, constraint frames,
//! constraint builder inputs, the global submap poses map) stays in
//! `two_d/pose_graph_2d.rs` and `three_d/pose_graph_3d.rs`. The functions
//! documented as "requires the state lock" correspond to the C++
//! `EXCLUSIVE_LOCKS_REQUIRED(mutex_)` members; they receive the locked data.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLockReadGuard};

use crate::common::fixed_ratio_sampler::FixedRatioSampler;
use crate::common::time::Time;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::optimization_problem_interface::OptimizationProblemInterface;
use crate::mapping::internal::pose_graph_data::{DeletionState, PoseGraphData, SubmapState};
use crate::mapping::pose_graph_interface::{Constraint, SubmapData, Tag, TrajectoryState};
use crate::mapping::pose_graph_trimmer::{PoseGraphTrimmer, Trimmable};
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::trajectory_node::{TrajectoryNode, TrajectoryNodeData};
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::metrics::gauge::Gauge;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::timestamped_transform::{TimestampedTransform, interpolate};

/// Locks `mutex`, ignoring poisoning (a panicking pool task aborts the
/// process anyway, see `ThreadPool`).
pub(crate) fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Read-locks a shared submap, ignoring poisoning.
pub(crate) fn read_submap<S>(submap: &SharedSubmap<S>) -> RwLockReadGuard<'_, S> {
    submap.read().unwrap_or_else(PoisonError::into_inner)
}

/// The constant data of a node (C++ `CHECK(constant_data != nullptr)`).
pub(crate) fn constant_data_of(node: &TrajectoryNode) -> &Arc<TrajectoryNodeData> {
    node.constant_data
        .as_ref()
        .expect("Check failed: constant_data != nullptr")
}

/// The pose graph gauges (the C++ file-level `static auto* k...Metric`
/// globals of `pose_graph_2d.cc` / `pose_graph_3d.cc`).
#[derive(Clone, Copy)]
pub(crate) struct PoseGraphMetrics {
    pub work_queue_delay: &'static dyn Gauge,
    pub work_queue_size: &'static dyn Gauge,
    pub constraints_same_trajectory: &'static dyn Gauge,
    pub constraints_different_trajectory: &'static dyn Gauge,
    pub active_submaps: &'static dyn Gauge,
    pub frozen_submaps: &'static dyn Gauge,
    pub deleted_submaps: &'static dyn Gauge,
}

impl PoseGraphMetrics {
    /// The null metrics (used until `RegisterMetrics` is called).
    pub fn null() -> Self {
        Self {
            work_queue_delay: <dyn Gauge>::null(),
            work_queue_size: <dyn Gauge>::null(),
            constraints_same_trajectory: <dyn Gauge>::null(),
            constraints_different_trajectory: <dyn Gauge>::null(),
            active_submaps: <dyn Gauge>::null(),
            frozen_submaps: <dyn Gauge>::null(),
            deleted_submaps: <dyn Gauge>::null(),
        }
    }

    /// C++ `PoseGraph{2D,3D}::RegisterMetrics` with metric names starting
    /// with `prefix` (`"mapping_2d_pose_graph"` / `"mapping_3d_pose_graph"`).
    pub fn register(family_factory: &mut dyn FamilyFactory, prefix: &str) -> Self {
        let labels = |pairs: &[(&str, &str)]| -> Labels {
            pairs
                .iter()
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let latency = family_factory.new_gauge_family(
            &format!("{prefix}_work_queue_delay"),
            "Age of the oldest entry in the work queue in seconds",
        );
        let work_queue_delay = latency.add(&Labels::new());
        let queue_size = family_factory.new_gauge_family(
            &format!("{prefix}_work_queue_size"),
            "Number of items in the work queue",
        );
        let work_queue_size = queue_size.add(&Labels::new());
        let constraints = family_factory.new_gauge_family(
            &format!("{prefix}_constraints"),
            "Current number of constraints in the pose graph",
        );
        let constraints_different_trajectory = constraints.add(&labels(&[
            ("tag", "inter_submap"),
            ("trajectory", "different"),
        ]));
        let constraints_same_trajectory =
            constraints.add(&labels(&[("tag", "inter_submap"), ("trajectory", "same")]));
        let submaps = family_factory.new_gauge_family(
            &format!("{prefix}_submaps"),
            "Number of submaps in the pose graph.",
        );
        let active_submaps = submaps.add(&labels(&[("state", "active")]));
        let frozen_submaps = submaps.add(&labels(&[("state", "frozen")]));
        let deleted_submaps = submaps.add(&labels(&[("state", "deleted")]));
        Self {
            work_queue_delay,
            work_queue_size,
            constraints_same_trajectory,
            constraints_different_trajectory,
            active_submaps,
            frozen_submaps,
            deleted_submaps,
        }
    }

    /// Updates the gauges that count the current number of inter-submap
    /// constraints (end of C++ `HandleWorkQueue`).
    pub fn update_constraint_gauges(&self, constraints: &[Constraint]) {
        let mut inter_constraints_same_trajectory = 0.;
        let mut inter_constraints_different_trajectory = 0.;
        for constraint in constraints {
            if constraint.tag == Tag::IntraSubmap {
                continue;
            }
            if constraint.node_id.trajectory_id == constraint.submap_id.trajectory_id {
                inter_constraints_same_trajectory += 1.;
            } else {
                inter_constraints_different_trajectory += 1.;
            }
        }
        self.constraints_same_trajectory
            .set(inter_constraints_same_trajectory);
        self.constraints_different_trajectory
            .set(inter_constraints_different_trajectory);
    }
}

/// The data guarded by the C++ `mutex_`.
pub(crate) struct PoseGraphState<S> {
    pub data: PoseGraphData<S>,
    /// List of all trimmers to consult when optimizations finish.
    pub trimmers: Vec<Box<dyn PoseGraphTrimmer>>,
    /// We globally localize a fraction of the nodes from each trajectory.
    pub global_localization_samplers: BTreeMap<i32, FixedRatioSampler>,
    /// Number of nodes added since last loop closure.
    pub num_nodes_since_last_loop_closure: i32,
}

impl<S> PoseGraphState<S> {
    /// The initial (empty) state.
    pub fn new() -> Self {
        Self {
            data: PoseGraphData::default(),
            trimmers: Vec::new(),
            global_localization_samplers: BTreeMap::new(),
            num_nodes_since_last_loop_closure: 0,
        }
    }

    /// Adds connectivity and sampler for a trajectory if it does not exist.
    pub fn add_trajectory_if_needed(
        &mut self,
        trajectory_id: i32,
        options: &proto::PoseGraphOptions,
    ) {
        let trajectory_state = self
            .data
            .trajectories_state
            .entry(trajectory_id)
            .or_default();
        assert!(
            trajectory_state.state != TrajectoryState::Finished,
            "Check failed: data_.trajectories_state.at(trajectory_id).state != TrajectoryState::FINISHED"
        );
        assert!(
            trajectory_state.state != TrajectoryState::Deleted,
            "Check failed: data_.trajectories_state.at(trajectory_id).state != TrajectoryState::DELETED"
        );
        assert!(
            trajectory_state.deletion_state == DeletionState::Normal,
            "Check failed: data_.trajectories_state.at(trajectory_id).deletion_state == \
             InternalTrajectoryState::DeletionState::NORMAL"
        );
        self.data.trajectory_connectivity_state.add(trajectory_id);
        // Make sure we have a sampler for this trajectory.
        self.global_localization_samplers
            .entry(trajectory_id)
            .or_insert_with(|| FixedRatioSampler::new(options.global_sampling_ratio));
    }
}

/// C++ `IsTrajectoryFinished` (requires the state lock).
pub(crate) fn is_trajectory_finished<S>(data: &PoseGraphData<S>, trajectory_id: i32) -> bool {
    data.trajectories_state
        .get(&trajectory_id)
        .is_some_and(|s| s.state == TrajectoryState::Finished)
}

/// C++ `IsTrajectoryFrozen` (requires the state lock).
pub(crate) fn is_trajectory_frozen<S>(data: &PoseGraphData<S>, trajectory_id: i32) -> bool {
    data.trajectories_state
        .get(&trajectory_id)
        .is_some_and(|s| s.state == TrajectoryState::Frozen)
}

/// C++ `GetTrajectoryStates` (requires the state lock).
pub(crate) fn trajectory_states<S>(data: &PoseGraphData<S>) -> BTreeMap<i32, TrajectoryState> {
    data.trajectories_state
        .iter()
        .map(|(&id, s)| (id, s.state))
        .collect()
}

/// C++ `CanAddWorkItemModifying`. Like C++ (`LOG(FATAL)`), modifying a
/// finished, deleted or to-be-deleted trajectory panics.
pub(crate) fn can_add_work_item_modifying<S>(data: &PoseGraphData<S>, trajectory_id: i32) -> bool {
    let Some(state) = data.trajectories_state.get(&trajectory_id) else {
        return true;
    };
    if state.state == TrajectoryState::Finished {
        // TODO(gaschler): Replace all FATAL to WARNING after some testing.
        panic!(
            "trajectory_id {trajectory_id} has finished but modification is requested, skipping."
        );
    }
    if state.deletion_state != DeletionState::Normal {
        panic!(
            "trajectory_id {trajectory_id} has been scheduled for deletion but modification is \
             requested, skipping."
        );
    }
    if state.state == TrajectoryState::Deleted {
        panic!(
            "trajectory_id {trajectory_id} has been deleted but modification is requested, \
             skipping."
        );
    }
    true
}

/// C++ `GetInterpolatedGlobalTrajectoryPose` (requires the state lock).
pub(crate) fn get_interpolated_global_trajectory_pose<S>(
    data: &PoseGraphData<S>,
    trajectory_id: i32,
    time: Time,
) -> Rigid3d {
    assert!(
        data.trajectory_nodes
            .size_of_trajectory_or_zero(trajectory_id)
            > 0,
        "Check failed: data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id) > 0"
    );
    let it = data.trajectory_nodes.lower_bound(trajectory_id, time);
    let begin = data.trajectory_nodes.begin_of_trajectory(trajectory_id);
    let end = data.trajectory_nodes.end_of_trajectory(trajectory_id);
    if it == begin {
        return begin.data().global_pose;
    }
    if it == end {
        return end.to_prev().data().global_pose;
    }
    let prev = it.to_prev();
    interpolate(
        &TimestampedTransform {
            time: prev.data().time(),
            transform: prev.data().global_pose,
        },
        &TimestampedTransform {
            time: it.data().time(),
            transform: it.data().global_pose,
        },
        time,
    )
    .transform
}

/// C++ `GetLatestNodeTime` (requires the state lock).
pub(crate) fn get_latest_node_time<S>(
    data: &PoseGraphData<S>,
    node_id: &NodeId,
    submap_id: &SubmapId,
) -> Time {
    let mut time = constant_data_of(data.trajectory_nodes.at(node_id)).time;
    let submap_data = data.submap_data.at(submap_id);
    if let Some(last_submap_node_id) = submap_data.node_ids.iter().next_back() {
        time = time.max(constant_data_of(data.trajectory_nodes.at(last_submap_node_id)).time);
    }
    time
}

/// Updates the trajectory connectivity structure with a new constraint
/// (requires the state lock).
pub(crate) fn update_trajectory_connectivity<S>(
    data: &mut PoseGraphData<S>,
    constraint: &Constraint,
) {
    assert_eq!(
        constraint.tag,
        Tag::InterSubmap,
        "Check failed: constraint.tag == Constraint::INTER_SUBMAP"
    );
    let time = get_latest_node_time(data, &constraint.node_id, &constraint.submap_id);
    data.trajectory_connectivity_state.connect(
        constraint.node_id.trajectory_id,
        constraint.submap_id.trajectory_id,
        time,
    );
}

/// Allows querying and manipulating the pose graph by the trimmers. The
/// pose graph's locks are held while this is used.
///
/// RUST PORT NOTE: C++ `PoseGraph{2D,3D}::TrimmingHandle` holds a pointer to
/// the pose graph; here it borrows the locked data, the locked optimization
/// problem and the constraint builder's `DeleteScanMatcher`. The only
/// dimension-specific part, the optimized global submap pose lookup
/// (`data_.global_submap_poses_{2d,3d}`), is passed as a function.
pub(crate) struct TrimmingHandle<'a, S, O: ?Sized> {
    pub data: &'a mut PoseGraphData<S>,
    pub optimization_problem: &'a mut O,
    /// C++ `constraint_builder_.DeleteScanMatcher`.
    pub delete_scan_matcher: &'a dyn Fn(&SubmapId),
    /// The last applied optimized global pose of a submap, if any.
    pub optimized_global_submap_pose: fn(&PoseGraphData<S>, &SubmapId) -> Option<Rigid3d>,
    pub metrics: PoseGraphMetrics,
}

impl<S, O> Trimmable for TrimmingHandle<'_, S, O>
where
    S: Submap + 'static,
    O: OptimizationProblemInterface + ?Sized,
{
    fn num_submaps(&self, trajectory_id: i32) -> i32 {
        self.optimization_problem
            .submap_data()
            .size_of_trajectory_or_zero(trajectory_id) as i32
    }

    fn get_submap_ids(&self, trajectory_id: i32) -> Vec<SubmapId> {
        self.optimization_problem
            .submap_data()
            .trajectory(trajectory_id)
            .map(|it| it.id)
            .collect()
    }

    fn get_optimized_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        let mut submaps = MapById::new();
        for submap_id_data in &self.data.submap_data {
            if submap_id_data.data.state != SubmapState::Finished {
                continue;
            }
            let Some(pose) = (self.optimized_global_submap_pose)(self.data, &submap_id_data.id)
            else {
                continue;
            };
            submaps.insert(
                &submap_id_data.id,
                SubmapData {
                    submap: Some(submap_id_data.data.submap_as_dyn()),
                    pose,
                },
            );
        }
        submaps
    }

    fn get_trajectory_nodes(&self) -> &MapById<NodeId, TrajectoryNode> {
        &self.data.trajectory_nodes
    }

    fn get_constraints(&self) -> &[Constraint] {
        &self.data.constraints
    }

    fn trim_submap(&mut self, submap_id: &SubmapId) {
        // TODO(hrapp): We have to make sure that the trajectory has been
        // finished if we want to delete the last submaps.
        assert!(
            self.data.submap_data.at(submap_id).state == SubmapState::Finished,
            "Check failed: parent_->data_.submap_data.at(submap_id).state == SubmapState::kFinished"
        );

        // Compile all nodes that are still INTRA_SUBMAP constrained to other
        // submaps once the submap with 'submap_id' is gone.
        // We need to use node_ids instead of constraints here to be also
        // compatible with frozen trajectories that don't have
        // intra-constraints.
        let mut nodes_to_retain: BTreeSet<NodeId> = BTreeSet::new();
        for submap_data in &self.data.submap_data {
            if submap_data.id != *submap_id {
                nodes_to_retain.extend(submap_data.data.node_ids.iter().copied());
            }
        }

        // Remove all nodes that are exlusively associated to 'submap_id'.
        let nodes_to_remove: BTreeSet<NodeId> = self
            .data
            .submap_data
            .at(submap_id)
            .node_ids
            .difference(&nodes_to_retain)
            .copied()
            .collect();

        // Remove all 'data_.constraints' related to 'submap_id'.
        self.data
            .constraints
            .retain(|constraint| constraint.submap_id != *submap_id);

        // Remove all 'data_.constraints' related to 'nodes_to_remove'.
        // If the removal lets other submaps lose all their inter-submap
        // constraints, delete their corresponding constraint submap matchers
        // to save memory.
        {
            let mut constraints = Vec::new();
            let mut other_submap_ids_losing_constraints: BTreeSet<SubmapId> = BTreeSet::new();
            for constraint in &self.data.constraints {
                if !nodes_to_remove.contains(&constraint.node_id) {
                    constraints.push(*constraint);
                } else {
                    // A constraint to another submap will be removed, mark it
                    // as affected.
                    other_submap_ids_losing_constraints.insert(constraint.submap_id);
                }
            }
            self.data.constraints = constraints;
            // Go through the remaining constraints to ensure we only delete
            // scan matchers of other submaps that have no inter-submap
            // constraints left.
            for constraint in &self.data.constraints {
                if constraint.tag == Tag::IntraSubmap {
                    continue;
                }
                // This submap still has inter-submap constraints - ignore it.
                other_submap_ids_losing_constraints.remove(&constraint.submap_id);
            }
            // Delete scan matchers of the submaps that lost all constraints.
            // TODO(wohe): An improvement to this implementation would be to
            // add the caching logic at the constraint builder which could keep
            // around only recently used scan matchers.
            for other_submap_id in &other_submap_ids_losing_constraints {
                (self.delete_scan_matcher)(other_submap_id);
            }
        }

        // Mark the submap with 'submap_id' as trimmed and remove its data.
        assert!(
            self.data.submap_data.at(submap_id).state == SubmapState::Finished,
            "Check failed: parent_->data_.submap_data.at(submap_id).state == SubmapState::kFinished"
        );
        self.data.submap_data.trim(submap_id);
        (self.delete_scan_matcher)(submap_id);
        self.optimization_problem.trim_submap(submap_id);

        // We have one submap less, update the gauge metrics.
        self.metrics.deleted_submaps.increment();
        if is_trajectory_frozen(self.data, submap_id.trajectory_id) {
            self.metrics.frozen_submaps.decrement();
        } else {
            self.metrics.active_submaps.decrement();
        }

        // Remove the 'nodes_to_remove' from the pose graph and the
        // optimization problem.
        for node_id in &nodes_to_remove {
            self.data.trajectory_nodes.trim(node_id);
            self.optimization_problem.trim_trajectory_node(node_id);
        }
    }

    fn is_finished(&self, trajectory_id: i32) -> bool {
        is_trajectory_finished(self.data, trajectory_id)
    }

    fn set_trajectory_state(&mut self, trajectory_id: i32, state: TrajectoryState) {
        self.data
            .trajectories_state
            .entry(trajectory_id)
            .or_default()
            .state = state;
    }
}

impl<S, O> TrimmingHandle<'_, S, O>
where
    S: Submap + 'static,
    O: OptimizationProblemInterface + ?Sized,
{
    /// C++ `DeleteTrajectoriesIfNeeded`: deletes trajectories waiting for
    /// deletion. Must not be called during constraint search. Requires the
    /// optimization problem and state locks.
    pub fn delete_trajectories_if_needed(&mut self) {
        let trajectory_ids: Vec<i32> = self
            .data
            .trajectories_state
            .iter()
            .filter(|(_, state)| state.deletion_state == DeletionState::WaitForDeletion)
            .map(|(&trajectory_id, _)| trajectory_id)
            .collect();
        for trajectory_id in trajectory_ids {
            // TODO(gaschler): Consider directly deleting from data_, which may
            // be more complete.
            let submap_ids = self.get_submap_ids(trajectory_id);
            for submap_id in &submap_ids {
                self.trim_submap(submap_id);
            }
            let state = self
                .data
                .trajectories_state
                .get_mut(&trajectory_id)
                .expect("trajectory state exists");
            state.state = TrajectoryState::Deleted;
            state.deletion_state = DeletionState::Normal;
        }
    }
}
