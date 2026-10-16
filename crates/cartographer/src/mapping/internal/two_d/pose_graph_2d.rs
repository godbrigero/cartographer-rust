//! Port of `cartographer/mapping/internal/2d/pose_graph_2d` (.h, .cc).
//!
//! # Design (the 3D port, `pose_graph_3d.rs`, should mirror it)
//!
//! RUST PORT NOTE: ownership, threading and locking.
//!
//! * Public handle vs. shared state. [`PoseGraph2D`] is a thin handle owning
//!   an `Arc<Inner>`. `Inner` holds everything the C++ class has; work items,
//!   the work-queue drain task and constraint builder callbacks run on thread
//!   pool threads and reach `Inner` through a `Weak<Inner>` (C++ captures
//!   `this`). Nothing but the handle owns a strong reference permanently, so
//!   there are no reference cycles. `PoseGraph2D` is `Send + Sync`; share it
//!   as `Arc<PoseGraph2D>` (trajectory builders) and/or `Arc<dyn PoseGraph>`.
//! * Locks (C++ -> Rust):
//!   - `mutex_` (guards `data_`, `trimmers_`, `global_localization_samplers_`,
//!     `num_nodes_since_last_loop_closure_`) -> `Inner::state: Mutex<State>`.
//!   - `work_queue_mutex_` + `work_queue_` -> `Inner::work_queue:
//!     Mutex<Option<WorkQueue<Inner>>>` (`None` <=> no drain/optimization in
//!     flight, exactly the C++ `nullptr` sentinel) plus a `Condvar` notified
//!     when it becomes `None` (C++ `AwaitWithTimeout`).
//!   - `optimization_problem_` is accessed without a lock in C++ (relying on
//!     the single logical worker, with documented races against getters).
//!     Here it is `Inner::optimization_problem: Mutex<Box<..>>`.
//!   - Lock order: `optimization_problem` -> `state` -> (constraint builder's
//!     internal mutex, via `delete_scan_matcher` during trimming). The
//!     work-queue mutex and the callback mutex are never held together with
//!     any other lock (except that `add_work_item` schedules the drain task on
//!     the thread pool while holding the work-queue mutex, like C++). Submap
//!     `RwLock`s are only read-locked briefly under these locks; the local
//!     trajectory builder write-locks submaps without holding pose graph
//!     locks.
//! * `Solve` runs while holding only the `optimization_problem` lock (C++:
//!   no lock at all). The constraints, trajectory states and landmark nodes it
//!   needs are copied out of `state` first (C++ reads `data_` unlocked).
//!   Foreground calls that only need `state` (`add_node`,
//!   `get_local_to_global_transform`, `get_trajectory_nodes`, ...) therefore
//!   never block on a running optimization; the getters that read the
//!   optimization problem (`get_trajectory_data`, `get_imu_data`, ...) wait
//!   for `Solve` instead of racing with it (C++ race §9.1 of the design doc).
//! * Background jobs receive `Arc` handles to submaps and node data (C++ raw
//!   pointers), so trimming while a search is in flight is memory safe.
//! * Work items are `FnOnce(&Inner) -> WorkItemResult` closures (see
//!   `work_queue.rs`); the queue protocol (drain on the pool until an item
//!   asks for an optimization, then `constraint_builder.when_done(
//!   handle_work_queue)`, which optimizes, trims and keeps draining) is the
//!   C++ one.
//! * Dropping [`PoseGraph2D`] runs `wait_for_all_computations` (C++
//!   destructor) and then waits until the pool tasks that are just returning
//!   have released their temporary strong references, so that `Inner` (and
//!   with it the constraint builder) is always dropped on the dropping thread,
//!   never on a pool thread. The last `PoseGraph2D` handle must therefore not
//!   be dropped from a thread pool thread (e.g. inside the global SLAM
//!   callback).
//! * 2D-specific parts the 3D port replaces: `NodeSpec2D`/`SubmapSpec2D`
//!   (gravity-aligned `Rigid2d` poses), `global_submap_poses_2d`,
//!   `compute_submap_pose`, the gravity (un)alignment of constraints in
//!   `constraints()` / `add_serialized_constraints`, the constraint builder
//!   inputs (`initial_relative_pose` vs. global poses/rotations),
//!   `set_trajectory_data_from_proto`, residual histograms (3D only) and the
//!   overlapping submaps trimmer (2D only). See §3.9 of
//!   `docs/architecture/pose_graph.md`.
//! * C++ `LOG(FATAL)`/`CHECK` become panics (also inside work items, where
//!   they take down the pool worker like the C++ process abort would).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock, Weak};
use std::time::{Duration as StdDuration, Instant};

use nalgebra::Quaternion;

use crate::common::task::Task;
use crate::common::thread_pool::ThreadPoolInterface;
use crate::common::time::{Time, from_seconds, from_universal};
use crate::mapping::constraints::{ConstraintBuilder2D, compute_submap_pose};
use crate::mapping::id::{ConstIterator, MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::optimization_problem_2d::{
    NodeSpec2D, OptimizationProblem2D, SubmapSpec2D,
};
use crate::mapping::internal::optimization::optimization_problem_interface::OptimizationProblemInterface;
use crate::mapping::internal::pose_graph_common::{
    PoseGraphMetrics, PoseGraphState, TrimmingHandle, can_add_work_item_modifying,
    constant_data_of, get_interpolated_global_trajectory_pose, get_latest_node_time,
    is_trajectory_finished, is_trajectory_frozen, lock, read_submap, trajectory_states,
    update_trajectory_connectivity,
};
use crate::mapping::internal::pose_graph_data::{
    DeletionState, InternalSubmapData, PoseGraphData, SubmapState,
};
use crate::mapping::internal::two_d::overlapping_submaps_trimmer_2d::OverlappingSubmapsTrimmer2D;
use crate::mapping::internal::work_queue::{WorkItem, WorkItemResult, WorkQueue};
use crate::mapping::pose_graph::{InitialTrajectoryPose, PoseGraph, pose_graph_to_proto};
use crate::mapping::pose_graph_interface::{
    Constraint, GlobalSlamOptimizationCallback, LandmarkNode, LandmarkObservation, Pose,
    PoseGraphInterface, SubmapData, SubmapPose, Tag, TrajectoryData, TrajectoryState,
};
use crate::mapping::pose_graph_trimmer::PoseGraphTrimmer;
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::trajectory_node::{
    ConstantPoseData, TrajectoryNode, TrajectoryNodeData, TrajectoryNodePose,
    from_proto as trajectory_node_data_from_proto,
};
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::metrics::family_factory::FamilyFactory;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{Rigid3d, quaternion_conjugate, quaternion_squared_norm};
use crate::transform::transform::{embed_3d, project_2d, to_rigid3};

// RUST PORT NOTE: the C++ `static auto* k...Metric` globals; `None` means the
// null metrics.
static METRICS: RwLock<Option<PoseGraphMetrics>> = RwLock::new(None);

fn metrics() -> PoseGraphMetrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(PoseGraphMetrics::null)
}

/// Eigen's `Quaternion::inverse()`: conjugate / squared norm (zero if the
/// norm is zero).
fn quaternion_inverse(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = quaternion_squared_norm(q);
    if n2 > 0. {
        let c = quaternion_conjugate(q);
        Quaternion::new(c.w / n2, c.i / n2, c.j / n2, c.k / n2)
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

/// The data guarded by the C++ `mutex_`.
type State = PoseGraphState<Submap2D>;

/// Computes the local to global map frame transform based on the given
/// `global_submap_poses` (requires the state lock).
fn compute_local_to_global_transform(
    data: &PoseGraphData<Submap2D>,
    global_submap_poses: &MapById<SubmapId, SubmapSpec2D>,
    trajectory_id: i32,
) -> Rigid3d {
    let begin_it = global_submap_poses.begin_of_trajectory(trajectory_id);
    let end_it = global_submap_poses.end_of_trajectory(trajectory_id);
    if begin_it == end_it {
        return match data.initial_trajectory_poses.get(&trajectory_id) {
            Some(initial_trajectory_pose) => {
                get_interpolated_global_trajectory_pose(
                    data,
                    initial_trajectory_pose.to_trajectory_id,
                    initial_trajectory_pose.time,
                ) * initial_trajectory_pose.relative_pose
            }
            None => Rigid3d::identity(),
        };
    }
    let last_optimized_submap_id = end_it.to_prev().id();
    // Accessing 'local_pose' in Submap is okay, since the member is const.
    embed_3d(
        &global_submap_poses
            .at(&last_optimized_submap_id)
            .global_pose,
    ) * read_submap(&data.submap_data.at(&last_optimized_submap_id).submap)
        .local_pose()
        .inverse()
}

fn get_submap_data_under_lock(data: &PoseGraphData<Submap2D>, submap_id: &SubmapId) -> SubmapData {
    let Some(internal_submap_data) = data.submap_data.get(submap_id) else {
        return SubmapData::default();
    };
    let submap = internal_submap_data.submap_as_dyn();
    if let Some(global_submap_pose) = data.global_submap_poses_2d.get(submap_id) {
        // We already have an optimized pose.
        return SubmapData {
            submap: Some(submap),
            pose: embed_3d(&global_submap_pose.global_pose),
        };
    }
    // We have to extrapolate.
    let pose = compute_local_to_global_transform(
        data,
        &data.global_submap_poses_2d,
        submap_id.trajectory_id,
    ) * read_submap(&internal_submap_data.submap).local_pose();
    SubmapData {
        submap: Some(submap),
        pose,
    }
}

fn get_all_submap_data_under_lock(data: &PoseGraphData<Submap2D>) -> MapById<SubmapId, SubmapData> {
    let mut submaps = MapById::new();
    for submap_id_data in &data.submap_data {
        submaps.insert(
            &submap_id_data.id,
            get_submap_data_under_lock(data, &submap_id_data.id),
        );
    }
    submaps
}

/// Grows the optimization problem to have an entry for every element of
/// `insertion_submaps`. Returns the IDs for the `insertion_submaps`.
fn initialize_global_submap_poses(
    optimization_problem: &mut OptimizationProblem2D,
    data: &mut PoseGraphData<Submap2D>,
    trajectory_id: i32,
    time: Time,
    insertion_submaps: &[SharedSubmap<Submap2D>],
) -> Vec<SubmapId> {
    assert!(
        !insertion_submaps.is_empty(),
        "Check failed: !insertion_submaps.empty()"
    );
    if insertion_submaps.len() == 1 {
        // If we don't already have an entry for the first submap, add one.
        if optimization_problem
            .submap_data()
            .size_of_trajectory_or_zero(trajectory_id)
            == 0
        {
            if let Some(initial_trajectory_pose) =
                data.initial_trajectory_poses.get(&trajectory_id).copied()
            {
                data.trajectory_connectivity_state.connect(
                    trajectory_id,
                    initial_trajectory_pose.to_trajectory_id,
                    time,
                );
            }
            let global_submap_pose = project_2d(
                &(compute_local_to_global_transform(
                    data,
                    &data.global_submap_poses_2d,
                    trajectory_id,
                ) * read_submap(&insertion_submaps[0]).local_pose()),
            );
            optimization_problem.add_submap(trajectory_id, &global_submap_pose);
        }
        assert_eq!(
            1,
            optimization_problem
                .submap_data()
                .size_of_trajectory_or_zero(trajectory_id),
            "Check failed: 1 == submap_data.SizeOfTrajectoryOrZero(trajectory_id)"
        );
        let submap_id = SubmapId::new(trajectory_id, 0);
        assert!(
            Arc::ptr_eq(
                &data.submap_data.at(&submap_id).submap,
                &insertion_submaps[0]
            ),
            "Check failed: data_.submap_data.at(submap_id).submap == insertion_submaps.front()"
        );
        return vec![submap_id];
    }
    assert_eq!(
        2,
        insertion_submaps.len(),
        "Check failed: 2 == insertion_submaps.size()"
    );
    let submap_data = optimization_problem.submap_data();
    let end_it = submap_data.end_of_trajectory(trajectory_id);
    assert!(
        submap_data.begin_of_trajectory(trajectory_id) != end_it,
        "Check failed: submap_data.BeginOfTrajectory(trajectory_id) != end_it"
    );
    let last_submap_id = end_it.to_prev().id();
    if Arc::ptr_eq(
        &data.submap_data.at(&last_submap_id).submap,
        &insertion_submaps[0],
    ) {
        // In this case, 'last_submap_id' is the ID of
        // 'insertions_submaps.front()' and 'insertions_submaps.back()' is new.
        let first_submap_pose = submap_data.at(&last_submap_id).global_pose;
        let global_submap_pose = first_submap_pose
            * compute_submap_pose(&read_submap(&insertion_submaps[0])).inverse()
            * compute_submap_pose(&read_submap(&insertion_submaps[1]));
        optimization_problem.add_submap(trajectory_id, &global_submap_pose);
        return vec![
            last_submap_id,
            SubmapId::new(trajectory_id, last_submap_id.submap_index + 1),
        ];
    }
    assert!(
        Arc::ptr_eq(
            &data.submap_data.at(&last_submap_id).submap,
            &insertion_submaps[1]
        ),
        "Check failed: data_.submap_data.at(last_submap_id).submap == insertion_submaps.back()"
    );
    let front_submap_id = SubmapId::new(trajectory_id, last_submap_id.submap_index - 1);
    assert!(
        Arc::ptr_eq(
            &data.submap_data.at(&front_submap_id).submap,
            &insertion_submaps[0]
        ),
        "Check failed: data_.submap_data.at(front_submap_id).submap == insertion_submaps.front()"
    );
    vec![front_submap_id, last_submap_id]
}

/// The optimized global pose of a submap (`data_.global_submap_poses_2d`),
/// for the `TrimmingHandle`.
fn optimized_global_submap_pose(
    data: &PoseGraphData<Submap2D>,
    submap_id: &SubmapId,
) -> Option<Rigid3d> {
    data.global_submap_poses_2d
        .get(submap_id)
        .map(|global_submap_pose| embed_3d(&global_submap_pose.global_pose))
}

/// Creates the C++ `TrimmingHandle(this)` from the locked parts.
fn trimming_handle<'a>(
    data: &'a mut PoseGraphData<Submap2D>,
    optimization_problem: &'a mut OptimizationProblem2D,
    delete_scan_matcher: &'a dyn Fn(&SubmapId),
) -> TrimmingHandle<'a, Submap2D, OptimizationProblem2D> {
    TrimmingHandle {
        data,
        optimization_problem,
        delete_scan_matcher,
        optimized_global_submap_pose,
        metrics: metrics(),
    }
}

/// The shared state of [`PoseGraph2D`] (everything the C++ class holds).
struct Inner {
    options: proto::PoseGraphOptions,
    /// Handle to ourselves, captured by tasks and callbacks (C++ `this`).
    weak_self: Weak<Inner>,
    /// RUST PORT NOTE: unguarded in C++ (set before use); a mutex here.
    global_slam_optimization_callback: Mutex<Option<Arc<GlobalSlamOptimizationCallback>>>,
    /// Current optimization problem. Lock order: before `state`.
    optimization_problem: Mutex<Box<OptimizationProblem2D>>,
    /// C++ `mutex_` and the members it guards.
    state: Mutex<State>,
    /// If it exists, further work items must be added to this queue, and
    /// will be considered later (C++ `work_queue_mutex_` + `work_queue_`).
    work_queue: Mutex<Option<WorkQueue<Inner>>>,
    /// Notified when `work_queue` becomes `None`.
    work_queue_drained: Condvar,
    constraint_builder: ConstraintBuilder2D,
    /// Thread pool used for handling the work queue.
    thread_pool: Arc<dyn ThreadPoolInterface>,
    conversion_tables: Arc<ValueConversionTables>,
}

impl Inner {
    fn lock_state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    fn lock_optimization_problem(&self) -> MutexGuard<'_, Box<OptimizationProblem2D>> {
        lock(&self.optimization_problem)
    }

    /// Handles a new work item.
    fn add_work_item(&self, work_item: impl FnOnce(&Inner) -> WorkItemResult + Send + 'static) {
        let mut work_queue = lock(&self.work_queue);
        if work_queue.is_none() {
            *work_queue = Some(WorkQueue::new());
            let task = Task::new();
            let weak_self = self.weak_self.clone();
            task.set_work_item(move || {
                if let Some(inner) = weak_self.upgrade() {
                    inner.drain_work_queue();
                }
            });
            self.thread_pool.schedule(task);
        }
        let queue = work_queue.as_mut().expect("work queue exists");
        let now = Instant::now();
        queue.push_back(WorkItem {
            time: now,
            task: Box::new(work_item),
        });
        let metrics = metrics();
        metrics.work_queue_size.set(queue.len() as f64);
        metrics.work_queue_delay.set(
            now.duration_since(queue.front().expect("non-empty").time)
                .as_secs_f64(),
        );
    }

    /// Appends the new node and submap (if needed) to the internal data
    /// structures.
    fn append_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Submap2D>],
        optimized_pose: &Rigid3d,
    ) -> NodeId {
        let mut state = self.lock_state();
        state.add_trajectory_if_needed(trajectory_id, &self.options);
        if !can_add_work_item_modifying(&state.data, trajectory_id) {
            log::warn!("AddNode was called for finished or deleted trajectory.");
        }
        let data = &mut state.data;
        let node_id = data.trajectory_nodes.append(
            trajectory_id,
            TrajectoryNode::new(constant_data, *optimized_pose),
        );
        data.num_trajectory_nodes += 1;
        let back = insertion_submaps
            .last()
            .expect("Check failed: !insertion_submaps.empty()");
        // Test if the 'insertion_submap.back()' is one we never saw before.
        if data.submap_data.size_of_trajectory_or_zero(trajectory_id) == 0
            || !Arc::ptr_eq(
                &data
                    .submap_data
                    .end_of_trajectory(trajectory_id)
                    .to_prev()
                    .data()
                    .submap,
                back,
            )
        {
            // We grow 'data_.submap_data' as needed. This code assumes that
            // the first time we see a new submap is as
            // 'insertion_submaps.back()'.
            let submap_id = data
                .submap_data
                .append(trajectory_id, InternalSubmapData::new(Arc::clone(back)));
            log::info!("Inserted submap {submap_id}.");
            metrics().active_submaps.increment();
        }
        node_id
    }

    /// Computes constraints for a node and submap pair.
    fn compute_constraint(&self, node_id: &NodeId, submap_id: &SubmapId) {
        let mut maybe_add_local_constraint = false;
        let mut maybe_add_global_constraint = false;
        let constant_data;
        let submap;
        {
            let mut state = self.lock_state();
            let State {
                data,
                global_localization_samplers,
                ..
            } = &mut *state;
            let internal_submap_data = data.submap_data.at(submap_id);
            assert!(
                internal_submap_data.state == SubmapState::Finished,
                "Check failed: data_.submap_data.at(submap_id).state == SubmapState::kFinished"
            );
            if !read_submap(&internal_submap_data.submap).insertion_finished() {
                // Uplink server only receives grids when they are finished, so
                // skip constraint search before that.
                return;
            }

            let node_time = get_latest_node_time(data, node_id, submap_id);
            let last_connection_time = data
                .trajectory_connectivity_state
                .last_connection_time(node_id.trajectory_id, submap_id.trajectory_id);
            if node_id.trajectory_id == submap_id.trajectory_id
                || node_time
                    < last_connection_time
                        + from_seconds(self.options.global_constraint_search_after_n_seconds)
            {
                // If the node and the submap belong to the same trajectory or
                // if there has been a recent global constraint that ties that
                // node's trajectory to the submap's trajectory, it suffices to
                // do a match constrained to a local search window.
                maybe_add_local_constraint = true;
            } else if global_localization_samplers
                .get_mut(&node_id.trajectory_id)
                .expect("global localization sampler exists")
                .pulse()
            {
                maybe_add_global_constraint = true;
            }
            constant_data = Arc::clone(constant_data_of(data.trajectory_nodes.at(node_id)));
            submap = Arc::clone(&data.submap_data.at(submap_id).submap);
        }

        if maybe_add_local_constraint {
            let initial_relative_pose = {
                let optimization_problem = self.lock_optimization_problem();
                optimization_problem
                    .submap_data()
                    .at(submap_id)
                    .global_pose
                    .inverse()
                    * optimization_problem.node_data().at(node_id).global_pose_2d
            };
            self.constraint_builder.maybe_add_constraint(
                submap_id,
                &submap,
                node_id,
                &constant_data,
                &initial_relative_pose,
            );
        } else if maybe_add_global_constraint {
            self.constraint_builder.maybe_add_global_constraint(
                submap_id,
                &submap,
                node_id,
                &constant_data,
            );
        }
    }

    /// Adds constraints for a node, and starts scan matching in the
    /// background.
    fn compute_constraints_for_node(
        &self,
        node_id: &NodeId,
        insertion_submaps: Vec<SharedSubmap<Submap2D>>,
        newly_finished_submap: bool,
    ) -> WorkItemResult {
        let submap_ids;
        let mut finished_submap_ids = Vec::new();
        let mut newly_finished_submap_node_ids = BTreeSet::new();
        {
            let mut optimization_problem = self.lock_optimization_problem();
            let mut state = self.lock_state();
            let data = &mut state.data;
            let constant_data = Arc::clone(constant_data_of(data.trajectory_nodes.at(node_id)));
            submap_ids = initialize_global_submap_poses(
                &mut optimization_problem,
                data,
                node_id.trajectory_id,
                constant_data.time,
                &insertion_submaps,
            );
            assert_eq!(
                submap_ids.len(),
                insertion_submaps.len(),
                "Check failed: submap_ids.size() == insertion_submaps.size()"
            );
            let matching_id = submap_ids[0];
            let local_pose_2d = project_2d(
                &(constant_data.local_pose
                    * Rigid3d::from_rotation(quaternion_inverse(&constant_data.gravity_alignment))),
            );
            let global_pose_2d = optimization_problem
                .submap_data()
                .at(&matching_id)
                .global_pose
                * compute_submap_pose(&read_submap(&insertion_submaps[0])).inverse()
                * local_pose_2d;
            optimization_problem.add_trajectory_node(
                matching_id.trajectory_id,
                &NodeSpec2D {
                    time: constant_data.time,
                    local_pose_2d,
                    global_pose_2d,
                    gravity_alignment: constant_data.gravity_alignment,
                },
            );
            for (submap_id, insertion_submap) in submap_ids.iter().zip(&insertion_submaps) {
                let internal_submap_data = data.submap_data.at_mut(submap_id);
                // Even if this was the last node added to 'submap_id', the
                // submap will only be marked as finished in
                // 'data_.submap_data' further below.
                assert!(
                    internal_submap_data.state == SubmapState::NoConstraintSearch,
                    "Check failed: data_.submap_data.at(submap_id).state == \
                     SubmapState::kNoConstraintSearch"
                );
                internal_submap_data.node_ids.insert(*node_id);
                let constraint_transform =
                    compute_submap_pose(&read_submap(insertion_submap)).inverse() * local_pose_2d;
                data.constraints.push(Constraint {
                    submap_id: *submap_id,
                    node_id: *node_id,
                    pose: Pose {
                        zbar_ij: embed_3d(&constraint_transform),
                        translation_weight: self.options.matcher_translation_weight,
                        rotation_weight: self.options.matcher_rotation_weight,
                    },
                    tag: Tag::IntraSubmap,
                });
            }

            // TODO(gaschler): Consider not searching for constraints against
            // trajectories scheduled for deletion.
            // TODO(danielsievers): Add a member variable and avoid having to
            // copy them out here.
            for submap_id_data in &data.submap_data {
                if submap_id_data.data.state == SubmapState::Finished {
                    assert!(
                        !submap_id_data.data.node_ids.contains(node_id),
                        "Check failed: submap_id_data.data.node_ids.count(node_id) == 0"
                    );
                    finished_submap_ids.push(submap_id_data.id);
                }
            }
            if newly_finished_submap {
                let newly_finished_submap_id = submap_ids[0];
                let finished_submap_data = data.submap_data.at_mut(&newly_finished_submap_id);
                assert!(
                    finished_submap_data.state == SubmapState::NoConstraintSearch,
                    "Check failed: finished_submap_data.state == SubmapState::kNoConstraintSearch"
                );
                finished_submap_data.state = SubmapState::Finished;
                newly_finished_submap_node_ids = finished_submap_data.node_ids.clone();
            }
        }

        for submap_id in &finished_submap_ids {
            self.compute_constraint(node_id, submap_id);
        }

        if newly_finished_submap {
            let newly_finished_submap_id = submap_ids[0];
            // We have a new completed submap, so we look into adding
            // constraints for old nodes.
            // RUST PORT NOTE: C++ iterates the optimization problem's node data
            // while calling ComputeConstraint (which locks `mutex_`); only this
            // worker modifies it, so collecting the IDs first is equivalent.
            let node_ids: Vec<NodeId> = self
                .lock_optimization_problem()
                .node_data()
                .iter()
                .map(|node_id_data| node_id_data.id)
                .collect();
            for other_node_id in &node_ids {
                if !newly_finished_submap_node_ids.contains(other_node_id) {
                    self.compute_constraint(other_node_id, &newly_finished_submap_id);
                }
            }
        }
        self.constraint_builder.notify_end_of_node();
        let mut state = self.lock_state();
        state.num_nodes_since_last_loop_closure += 1;
        if self.options.optimize_every_n_nodes > 0
            && state.num_nodes_since_last_loop_closure > self.options.optimize_every_n_nodes
        {
            return WorkItemResult::RunOptimization;
        }
        WorkItemResult::DoNotRunOptimization
    }

    /// Runs the optimization, executes the trimmers and processes the work
    /// queue.
    fn handle_work_queue(&self, result: &[Constraint]) {
        self.lock_state().data.constraints.extend_from_slice(result);
        self.run_optimization();

        let callback = lock(&self.global_slam_optimization_callback).clone();
        if let Some(callback) = callback {
            let mut trajectory_id_to_last_optimized_node_id = BTreeMap::new();
            let mut trajectory_id_to_last_optimized_submap_id = BTreeMap::new();
            {
                let optimization_problem = self.lock_optimization_problem();
                let submap_data = optimization_problem.submap_data();
                let node_data = optimization_problem.node_data();
                for trajectory_id in node_data.trajectory_ids() {
                    if node_data.size_of_trajectory_or_zero(trajectory_id) == 0
                        || submap_data.size_of_trajectory_or_zero(trajectory_id) == 0
                    {
                        continue;
                    }
                    trajectory_id_to_last_optimized_node_id.insert(
                        trajectory_id,
                        node_data.end_of_trajectory(trajectory_id).to_prev().id(),
                    );
                    trajectory_id_to_last_optimized_submap_id.insert(
                        trajectory_id,
                        submap_data.end_of_trajectory(trajectory_id).to_prev().id(),
                    );
                }
            }
            callback(
                &trajectory_id_to_last_optimized_submap_id,
                &trajectory_id_to_last_optimized_node_id,
            );
        }

        {
            let mut optimization_problem = self.lock_optimization_problem();
            let mut state = self.lock_state();
            let State {
                data,
                trimmers,
                num_nodes_since_last_loop_closure,
                ..
            } = &mut *state;
            for constraint in result {
                update_trajectory_connectivity(data, constraint);
            }
            let delete_scan_matcher =
                |submap_id: &SubmapId| self.constraint_builder.delete_scan_matcher(submap_id);
            {
                let mut trimming_handle =
                    trimming_handle(data, &mut optimization_problem, &delete_scan_matcher);
                trimming_handle.delete_trajectories_if_needed();
                for trimmer in trimmers.iter_mut() {
                    trimmer.trim(&mut trimming_handle);
                }
            }
            trimmers.retain(|trimmer| !trimmer.is_finished());

            *num_nodes_since_last_loop_closure = 0;

            // Update the gauges that count the current number of constraints.
            metrics().update_constraint_gauges(&data.constraints);
        }

        self.drain_work_queue();
    }

    /// Process pending tasks in the work queue on the calling thread, until
    /// the queue is either empty or an optimization is required.
    fn drain_work_queue(&self) {
        let mut process_work_queue = true;
        let mut work_queue_size = 0;
        while process_work_queue {
            let work_item = {
                let mut work_queue = lock(&self.work_queue);
                let queue = work_queue
                    .as_mut()
                    .expect("DrainWorkQueue requires a work queue");
                match queue.pop_front() {
                    None => {
                        *work_queue = None;
                        self.work_queue_drained.notify_all();
                        return;
                    }
                    Some(work_item) => {
                        work_queue_size = queue.len();
                        metrics().work_queue_size.set(work_queue_size as f64);
                        work_item
                    }
                }
            };
            process_work_queue = (work_item.task)(self) == WorkItemResult::DoNotRunOptimization;
        }
        log::info!("Remaining work items in queue: {work_queue_size}");
        // We have to optimize again.
        let weak_self = self.weak_self.clone();
        self.constraint_builder.when_done(move |result| {
            if let Some(inner) = weak_self.upgrade() {
                inner.handle_work_queue(result);
            }
        });
    }

    /// Waits until we caught up (i.e. nothing is waiting to be scheduled),
    /// and all computations have finished.
    fn wait_for_all_computations(&self) {
        let num_trajectory_nodes = self.lock_state().data.num_trajectory_nodes;
        let num_finished_nodes_at_start = self.constraint_builder.get_num_finished_nodes();

        let report_progress = || {
            // Log progress on nodes only when we are actually processing
            // nodes.
            if num_trajectory_nodes != num_finished_nodes_at_start {
                let progress =
                    100. * f64::from(
                        self.constraint_builder.get_num_finished_nodes()
                            - num_finished_nodes_at_start,
                    ) / f64::from(num_trajectory_nodes - num_finished_nodes_at_start);
                print!("\r\x1b[KOptimizing: {progress:.1}%...");
                let _ = std::io::stdout().flush();
            }
        };
        let timeout = StdDuration::from_secs(1);

        // First wait for the work queue to drain so that it's safe to
        // schedule a WhenDone() callback.
        {
            let mut work_queue = lock(&self.work_queue);
            while work_queue.is_some() {
                let (guard, wait_result) = self
                    .work_queue_drained
                    .wait_timeout(work_queue, timeout)
                    .unwrap_or_else(PoisonError::into_inner);
                work_queue = guard;
                if wait_result.timed_out() && work_queue.is_some() {
                    report_progress();
                }
            }
        }

        // Now wait for any pending constraint computations to finish.
        // RUST PORT NOTE: C++ waits on `mutex_` with an `absl::Condition`;
        // here a separate flag + condition variable is used, and the callback
        // releases its reference to the pose graph before signalling (see the
        // drop protocol in the module documentation).
        let notification = Arc::new((Mutex::new(false), Condvar::new()));
        {
            let notification = Arc::clone(&notification);
            let weak_self = self.weak_self.clone();
            self.constraint_builder.when_done(move |result| {
                if let Some(inner) = weak_self.upgrade() {
                    inner
                        .lock_state()
                        .data
                        .constraints
                        .extend_from_slice(result);
                }
                let (flag, condition) = &*notification;
                *lock(flag) = true;
                condition.notify_all();
            });
        }
        {
            let (flag, condition) = &*notification;
            let mut done = lock(flag);
            while !*done {
                let (guard, wait_result) = condition
                    .wait_timeout(done, timeout)
                    .unwrap_or_else(PoisonError::into_inner);
                done = guard;
                if wait_result.timed_out() && !*done {
                    report_progress();
                }
            }
        }
        assert_eq!(
            self.constraint_builder.get_num_finished_nodes(),
            num_trajectory_nodes,
            "Check failed: constraint_builder_.GetNumFinishedNodes() == num_trajectory_nodes"
        );
        println!("\r\x1b[KOptimizing: Done.     ");
    }

    /// Runs the optimization. Callers have to make sure, that there is only
    /// one optimization being run at a time.
    fn run_optimization(&self) {
        let mut optimization_problem = self.lock_optimization_problem();
        if optimization_problem.submap_data().empty() {
            return;
        }

        // No other thread is accessing the optimization_problem_,
        // data_.constraints, data_.frozen_trajectories and
        // data_.landmark_nodes when executing the Solve. Solve is time
        // consuming, so not taking the mutex before Solve to avoid blocking
        // foreground processing.
        // RUST PORT NOTE: the inputs are copied out of the state under its
        // lock; only the optimization problem lock is held during Solve.
        let (constraints, trajectories_state, landmark_nodes) = {
            let state = self.lock_state();
            (
                state.data.constraints.clone(),
                trajectory_states(&state.data),
                state.data.landmark_nodes.clone(),
            )
        };
        optimization_problem.solve(&constraints, &trajectories_state, &landmark_nodes);
        let mut state = self.lock_state();
        let data = &mut state.data;

        let submap_data = optimization_problem.submap_data();
        let node_data = optimization_problem.node_data();
        for trajectory_id in node_data.trajectory_ids() {
            for node in node_data.trajectory(trajectory_id) {
                let mutable_trajectory_node = data.trajectory_nodes.at_mut(&node.id);
                let gravity_alignment = constant_data_of(mutable_trajectory_node).gravity_alignment;
                mutable_trajectory_node.global_pose =
                    embed_3d(&node.data.global_pose_2d) * Rigid3d::from_rotation(gravity_alignment);
            }

            // Extrapolate all point cloud poses that were not included in the
            // 'optimization_problem_' yet.
            let local_to_new_global =
                compute_local_to_global_transform(data, submap_data, trajectory_id);
            let local_to_old_global = compute_local_to_global_transform(
                data,
                &data.global_submap_poses_2d,
                trajectory_id,
            );
            let old_global_to_new_global = local_to_new_global * local_to_old_global.inverse();

            let last_optimized_node_id = node_data.end_of_trajectory(trajectory_id).to_prev().id();
            let nodes_to_extrapolate: Vec<NodeId> =
                ConstIterator::at_id(&data.trajectory_nodes, &last_optimized_node_id)
                    .to_next()
                    .range_to(data.trajectory_nodes.end_of_trajectory(trajectory_id))
                    .map(|node| node.id)
                    .collect();
            for node_id in &nodes_to_extrapolate {
                let mutable_trajectory_node = data.trajectory_nodes.at_mut(node_id);
                mutable_trajectory_node.global_pose =
                    old_global_to_new_global * mutable_trajectory_node.global_pose;
            }
        }
        for (landmark_id, global_pose) in optimization_problem.landmark_data() {
            data.landmark_nodes
                .entry(landmark_id.clone())
                .or_default()
                .global_landmark_pose = Some(*global_pose);
        }
        data.global_submap_poses_2d = submap_data.clone();
    }
}

/// Implements the loop closure method called Sparse Pose Adjustment (SPA)
/// from Konolige, Kurt, et al. "Efficient sparse pose adjustment for 2d
/// mapping." Intelligent Robots and Systems (IROS), 2010 IEEE/RSJ
/// International Conference on (pp. 22--29). IEEE, 2010.
///
/// It is extended for submapping:
/// Each node has been matched against one or more submaps (adding a
/// constraint for each match), both poses of nodes and of submaps are to be
/// optimized. All constraints are between a submap i and a node j.
///
/// See the module documentation for the threading model.
pub struct PoseGraph2D {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for PoseGraph2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseGraph2D")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl PoseGraph2D {
    /// Creates the pose graph. Work items, constraint searches and
    /// optimizations run on `thread_pool`, which must outlive the pose graph
    /// (C++ `common::ThreadPool*`).
    pub fn new(
        options: &proto::PoseGraphOptions,
        optimization_problem: Box<OptimizationProblem2D>,
        thread_pool: Arc<dyn ThreadPoolInterface>,
    ) -> Self {
        let inner = Arc::new_cyclic(|weak_self| Inner {
            options: options.clone(),
            weak_self: weak_self.clone(),
            global_slam_optimization_callback: Mutex::new(None),
            optimization_problem: Mutex::new(optimization_problem),
            state: Mutex::new(PoseGraphState::new()),
            work_queue: Mutex::new(None),
            work_queue_drained: Condvar::new(),
            constraint_builder: ConstraintBuilder2D::new(
                &options
                    .constraint_builder_options
                    .clone()
                    .unwrap_or_default(),
                Arc::clone(&thread_pool),
            ),
            thread_pool,
            conversion_tables: Arc::new(ValueConversionTables::new()),
        });
        let pose_graph = PoseGraph2D { inner };
        if let Some(trimmer_options) = options.overlapping_submaps_trimmer_2d {
            // C++ narrows the int32 proto fields to uint16.
            pose_graph.add_trimmer(Box::new(OverlappingSubmapsTrimmer2D::new(
                trimmer_options.fresh_submaps_count as u16,
                trimmer_options.min_covered_area,
                trimmer_options.min_added_submaps_count as u16,
            )));
        }
        pose_graph
    }

    /// Adds a new node with `constant_data`. Its `constant_data.local_pose`
    /// was determined by scan matching against `insertion_submaps[0]` and
    /// the node data was inserted into the `insertion_submaps`. If
    /// `insertion_submaps[0].insertion_finished()` is `true`, data was
    /// inserted into this submap for the last time.
    pub fn add_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Submap2D>],
    ) -> NodeId {
        let optimized_pose =
            self.get_local_to_global_transform(trajectory_id) * constant_data.local_pose;

        let node_id = self.inner.append_node(
            constant_data,
            trajectory_id,
            insertion_submaps,
            &optimized_pose,
        );
        // We have to check this here, because it might have changed by the
        // time we execute the lambda.
        let newly_finished_submap = read_submap(
            insertion_submaps
                .first()
                .expect("Check failed: !insertion_submaps.empty()"),
        )
        .insertion_finished();
        let insertion_submaps = insertion_submaps.to_vec();
        self.inner.add_work_item(move |inner| {
            inner.compute_constraints_for_node(&node_id, insertion_submaps, newly_finished_submap)
        });
        node_id
    }

    /// Interpolates the global pose of `trajectory_id` at `time` from its
    /// nodes. Panics if the trajectory has no nodes.
    ///
    /// RUST PORT NOTE: C++ requires `mutex_` to be held by the caller; this
    /// public version takes the lock itself.
    pub fn get_interpolated_global_trajectory_pose(
        &self,
        trajectory_id: i32,
        time: Time,
    ) -> Rigid3d {
        let state = self.inner.lock_state();
        get_interpolated_global_trajectory_pose(&state.data, trajectory_id, time)
    }

    /// Waits until all work items, constraint searches and optimizations
    /// that were requested so far have finished (C++ private
    /// `WaitForAllComputations`; exposed for tests).
    #[cfg(test)]
    pub(crate) fn wait_for_all_computations(&self) {
        self.inner.wait_for_all_computations();
    }

    /// Registers the metrics with `family_factory`.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(
            PoseGraphMetrics::register(family_factory, "mapping_2d_pose_graph"),
        );
    }
}

impl Drop for PoseGraph2D {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.inner.wait_for_all_computations();
        assert!(
            lock(&self.inner.work_queue).is_none(),
            "Check failed: work_queue_ == nullptr"
        );
        // RUST PORT NOTE: pool tasks hold a temporary strong reference while
        // they run; the ones still returning after the wait above release it
        // momentarily. Wait for that so `Inner` is dropped on this thread.
        while Arc::strong_count(&self.inner) > 1 {
            std::thread::sleep(StdDuration::from_micros(50));
        }
    }
}

impl PoseGraphInterface for PoseGraph2D {
    fn run_final_optimization(&self) {
        {
            self.inner.add_work_item(|inner| {
                inner
                    .lock_optimization_problem()
                    .set_max_num_iterations(inner.options.max_num_final_iterations);
                WorkItemResult::RunOptimization
            });
            self.inner.add_work_item(|inner| {
                let max_num_iterations = inner
                    .options
                    .optimization_problem_options
                    .unwrap_or_default()
                    .ceres_solver_options
                    .unwrap_or_default()
                    .max_num_iterations;
                inner
                    .lock_optimization_problem()
                    .set_max_num_iterations(max_num_iterations);
                WorkItemResult::DoNotRunOptimization
            });
        }
        self.inner.wait_for_all_computations();
    }

    fn get_all_submap_data(&self) -> MapById<SubmapId, SubmapData> {
        get_all_submap_data_under_lock(&self.inner.lock_state().data)
    }

    fn get_submap_data(&self, submap_id: &SubmapId) -> SubmapData {
        get_submap_data_under_lock(&self.inner.lock_state().data, submap_id)
    }

    fn get_all_submap_poses(&self) -> MapById<SubmapId, SubmapPose> {
        let state = self.inner.lock_state();
        let mut submap_poses = MapById::new();
        for submap_id_data in &state.data.submap_data {
            let submap_data = get_submap_data_under_lock(&state.data, &submap_id_data.id);
            let version = submap_data
                .submap
                .as_ref()
                .expect("submap exists")
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .num_range_data();
            submap_poses.insert(
                &submap_id_data.id,
                SubmapPose {
                    version,
                    pose: submap_data.pose,
                },
            );
        }
        submap_poses
    }

    fn get_local_to_global_transform(&self, trajectory_id: i32) -> Rigid3d {
        let state = self.inner.lock_state();
        compute_local_to_global_transform(
            &state.data,
            &state.data.global_submap_poses_2d,
            trajectory_id,
        )
    }

    fn get_trajectory_nodes(&self) -> MapById<NodeId, TrajectoryNode> {
        self.inner.lock_state().data.trajectory_nodes.clone()
    }

    fn get_trajectory_node_poses(&self) -> MapById<NodeId, TrajectoryNodePose> {
        let mut node_poses = MapById::new();
        let state = self.inner.lock_state();
        for node_id_data in &state.data.trajectory_nodes {
            let constant_pose_data =
                node_id_data
                    .data
                    .constant_data
                    .as_ref()
                    .map(|constant_data| ConstantPoseData {
                        time: constant_data.time,
                        local_pose: constant_data.local_pose,
                    });
            node_poses.insert(
                &node_id_data.id,
                TrajectoryNodePose::new(node_id_data.data.global_pose, constant_pose_data),
            );
        }
        node_poses
    }

    fn get_trajectory_states(&self) -> BTreeMap<i32, TrajectoryState> {
        trajectory_states(&self.inner.lock_state().data)
    }

    fn get_landmark_poses(&self) -> BTreeMap<String, Rigid3d> {
        let state = self.inner.lock_state();
        state
            .data
            .landmark_nodes
            .iter()
            // Landmark without value has not been optimized yet.
            .filter_map(|(id, landmark)| {
                landmark.global_landmark_pose.map(|pose| (id.clone(), pose))
            })
            .collect()
    }

    fn set_landmark_pose(&self, landmark_id: &str, global_pose: &Rigid3d, frozen: bool) {
        let landmark_id = landmark_id.to_string();
        let global_pose = *global_pose;
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            let landmark_node = state.data.landmark_nodes.entry(landmark_id).or_default();
            landmark_node.global_landmark_pose = Some(global_pose);
            landmark_node.frozen = frozen;
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn delete_trajectory(&self, trajectory_id: i32) {
        {
            let mut state = self.inner.lock_state();
            let Some(trajectory_state) = state.data.trajectories_state.get_mut(&trajectory_id)
            else {
                log::warn!(
                    "Skipping request to delete non-existing trajectory_id: {trajectory_id}"
                );
                return;
            };
            trajectory_state.deletion_state = DeletionState::ScheduledForDeletion;
        }
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            let trajectory_state = state
                .data
                .trajectories_state
                .get_mut(&trajectory_id)
                .expect("trajectory state exists");
            assert!(
                trajectory_state.state != TrajectoryState::Active,
                "Check failed: data_.trajectories_state.at(trajectory_id).state != \
                 TrajectoryState::ACTIVE"
            );
            assert!(
                trajectory_state.state != TrajectoryState::Deleted,
                "Check failed: data_.trajectories_state.at(trajectory_id).state != \
                 TrajectoryState::DELETED"
            );
            assert!(
                trajectory_state.deletion_state == DeletionState::ScheduledForDeletion,
                "Check failed: data_.trajectories_state.at(trajectory_id).deletion_state == \
                 InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION"
            );
            trajectory_state.deletion_state = DeletionState::WaitForDeletion;
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn is_trajectory_finished(&self, trajectory_id: i32) -> bool {
        // RUST PORT NOTE: C++ requires (but its callers do not hold) `mutex_`.
        is_trajectory_finished(&self.inner.lock_state().data, trajectory_id)
    }

    fn is_trajectory_frozen(&self, trajectory_id: i32) -> bool {
        // RUST PORT NOTE: C++ requires (but its callers do not hold) `mutex_`.
        is_trajectory_frozen(&self.inner.lock_state().data, trajectory_id)
    }

    fn get_trajectory_data(&self) -> BTreeMap<i32, TrajectoryData> {
        self.inner
            .lock_optimization_problem()
            .trajectory_data()
            .clone()
    }

    fn constraints(&self) -> Vec<Constraint> {
        let state = self.inner.lock_state();
        state
            .data
            .constraints
            .iter()
            .map(|constraint| {
                let gravity_alignment =
                    constant_data_of(state.data.trajectory_nodes.at(&constraint.node_id))
                        .gravity_alignment;
                Constraint {
                    submap_id: constraint.submap_id,
                    node_id: constraint.node_id,
                    pose: Pose {
                        zbar_ij: constraint.pose.zbar_ij
                            * Rigid3d::from_rotation(gravity_alignment),
                        translation_weight: constraint.pose.translation_weight,
                        rotation_weight: constraint.pose.rotation_weight,
                    },
                    tag: constraint.tag,
                }
            })
            .collect()
    }

    fn to_proto(&self, include_unfinished_submaps: bool) -> proto::PoseGraph {
        pose_graph_to_proto(self, include_unfinished_submaps)
    }

    fn set_global_slam_optimization_callback(&self, callback: GlobalSlamOptimizationCallback) {
        *lock(&self.inner.global_slam_optimization_callback) = Some(Arc::new(callback));
    }
}

impl PoseGraph for PoseGraph2D {
    fn add_imu_data(&self, trajectory_id: i32, imu_data: &ImuData) {
        let imu_data = *imu_data;
        self.inner.add_work_item(move |inner| {
            let mut optimization_problem = inner.lock_optimization_problem();
            let state = inner.lock_state();
            if can_add_work_item_modifying(&state.data, trajectory_id) {
                optimization_problem.add_imu_data(trajectory_id, &imu_data);
            }
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_odometry_data(&self, trajectory_id: i32, odometry_data: &OdometryData) {
        let odometry_data = *odometry_data;
        self.inner.add_work_item(move |inner| {
            let mut optimization_problem = inner.lock_optimization_problem();
            let state = inner.lock_state();
            if can_add_work_item_modifying(&state.data, trajectory_id) {
                optimization_problem.add_odometry_data(trajectory_id, &odometry_data);
            }
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_fixed_frame_pose_data(
        &self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    ) {
        let fixed_frame_pose_data = *fixed_frame_pose_data;
        self.inner.add_work_item(move |inner| {
            let mut optimization_problem = inner.lock_optimization_problem();
            let state = inner.lock_state();
            if can_add_work_item_modifying(&state.data, trajectory_id) {
                optimization_problem
                    .add_fixed_frame_pose_data(trajectory_id, &fixed_frame_pose_data);
            }
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_landmark_data(&self, trajectory_id: i32, landmark_data: &LandmarkData) {
        let landmark_data = landmark_data.clone();
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            if can_add_work_item_modifying(&state.data, trajectory_id) {
                for observation in &landmark_data.landmark_observations {
                    state
                        .data
                        .landmark_nodes
                        .entry(observation.id.clone())
                        .or_default()
                        .landmark_observations
                        .push(LandmarkObservation {
                            trajectory_id,
                            time: landmark_data.time,
                            landmark_to_tracking_transform: observation
                                .landmark_to_tracking_transform,
                            translation_weight: observation.translation_weight,
                            rotation_weight: observation.rotation_weight,
                        });
                }
            }
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn finish_trajectory(&self, trajectory_id: i32) {
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            let data = &mut state.data;
            assert!(
                !is_trajectory_finished(data, trajectory_id),
                "Check failed: !IsTrajectoryFinished(trajectory_id)"
            );
            data.trajectories_state
                .entry(trajectory_id)
                .or_default()
                .state = TrajectoryState::Finished;

            for (_, submap) in data.submap_data.trajectory_mut(trajectory_id) {
                submap.state = SubmapState::Finished;
            }
            WorkItemResult::RunOptimization
        });
    }

    fn freeze_trajectory(&self, trajectory_id: i32) {
        self.inner
            .lock_state()
            .data
            .trajectory_connectivity_state
            .add(trajectory_id);
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            let data = &mut state.data;
            assert!(
                !is_trajectory_frozen(data, trajectory_id),
                "Check failed: !IsTrajectoryFrozen(trajectory_id)"
            );
            // Connect multiple frozen trajectories among each other.
            // This is required for localization against multiple frozen
            // trajectories because we lose inter-trajectory constraints when
            // freezing.
            let other_trajectory_ids: Vec<i32> = data.trajectories_state.keys().copied().collect();
            for other_trajectory_id in other_trajectory_ids {
                if !is_trajectory_frozen(data, other_trajectory_id) {
                    continue;
                }
                if data
                    .trajectory_connectivity_state
                    .transitively_connected(trajectory_id, other_trajectory_id)
                {
                    // Already connected, nothing to do.
                    continue;
                }
                data.trajectory_connectivity_state.connect(
                    trajectory_id,
                    other_trajectory_id,
                    from_universal(0),
                );
            }
            data.trajectories_state
                .entry(trajectory_id)
                .or_default()
                .state = TrajectoryState::Frozen;
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_submap_from_proto(&self, global_submap_pose: &Rigid3d, submap: &proto::Submap) {
        let Some(submap_2d) = submap.submap_2d.as_ref() else {
            return;
        };

        let submap_id = SubmapId::from_proto(&submap.submap_id.unwrap_or_default());

        let global_submap_pose_2d = project_2d(global_submap_pose);
        {
            let mut state = self.inner.lock_state();
            let submap_ptr = Arc::new(RwLock::new(Submap2D::from_proto(
                submap_2d,
                &self.inner.conversion_tables,
            )));
            state.add_trajectory_if_needed(submap_id.trajectory_id, &self.inner.options);
            if !can_add_work_item_modifying(&state.data, submap_id.trajectory_id) {
                return;
            }
            state
                .data
                .submap_data
                .insert(&submap_id, InternalSubmapData::new(submap_ptr));
            // Immediately show the submap at the 'global_submap_pose'.
            state.data.global_submap_poses_2d.insert(
                &submap_id,
                SubmapSpec2D {
                    global_pose: global_submap_pose_2d,
                },
            );
        }

        // TODO(MichaelGrupp): MapBuilder does freezing before deserializing
        // submaps, so this should be fine.
        if self.is_trajectory_frozen(submap_id.trajectory_id) {
            metrics().frozen_submaps.increment();
        } else {
            metrics().active_submaps.increment();
        }

        self.inner.add_work_item(move |inner| {
            let mut optimization_problem = inner.lock_optimization_problem();
            let mut state = inner.lock_state();
            state.data.submap_data.at_mut(&submap_id).state = SubmapState::Finished;
            optimization_problem.insert_submap(&submap_id, &global_submap_pose_2d);
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_node_from_proto(&self, global_pose: &Rigid3d, node: &proto::Node) {
        let node_id = NodeId::from_proto(&node.node_id.unwrap_or_default());
        let constant_data = Arc::new(trajectory_node_data_from_proto(
            &node.node_data.clone().unwrap_or_default(),
        ));

        {
            let mut state = self.inner.lock_state();
            state.add_trajectory_if_needed(node_id.trajectory_id, &self.inner.options);
            if !can_add_work_item_modifying(&state.data, node_id.trajectory_id) {
                return;
            }
            state
                .data
                .trajectory_nodes
                .insert(&node_id, TrajectoryNode::new(constant_data, *global_pose));
        }

        let global_pose = *global_pose;
        self.inner.add_work_item(move |inner| {
            let mut optimization_problem = inner.lock_optimization_problem();
            let state = inner.lock_state();
            let constant_data = constant_data_of(state.data.trajectory_nodes.at(&node_id));
            let gravity_alignment_inverse =
                Rigid3d::from_rotation(quaternion_inverse(&constant_data.gravity_alignment));
            optimization_problem.insert_trajectory_node(
                &node_id,
                &NodeSpec2D {
                    time: constant_data.time,
                    local_pose_2d: project_2d(
                        &(constant_data.local_pose * gravity_alignment_inverse),
                    ),
                    global_pose_2d: project_2d(&(global_pose * gravity_alignment_inverse)),
                    gravity_alignment: constant_data.gravity_alignment,
                },
            );
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn set_trajectory_data_from_proto(&self, data: &proto::TrajectoryData) {
        // gravity_constant and imu_calibration are omitted as its not used in
        // 2d
        if let Some(fixed_frame_origin_in_map) = data.fixed_frame_origin_in_map.as_ref() {
            let trajectory_data = TrajectoryData {
                fixed_frame_origin_in_map: Some(to_rigid3(fixed_frame_origin_in_map)),
                ..Default::default()
            };
            let trajectory_id = data.trajectory_id;
            self.inner.add_work_item(move |inner| {
                let mut optimization_problem = inner.lock_optimization_problem();
                let state = inner.lock_state();
                if can_add_work_item_modifying(&state.data, trajectory_id) {
                    optimization_problem.set_trajectory_data(trajectory_id, &trajectory_data);
                }
                WorkItemResult::DoNotRunOptimization
            });
        }
    }

    fn add_node_to_submap(&self, node_id: &NodeId, submap_id: &SubmapId) {
        let node_id = *node_id;
        let submap_id = *submap_id;
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            if can_add_work_item_modifying(&state.data, submap_id.trajectory_id) {
                state
                    .data
                    .submap_data
                    .at_mut(&submap_id)
                    .node_ids
                    .insert(node_id);
            }
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_serialized_constraints(&self, constraints: &[Constraint]) {
        let constraints = constraints.to_vec();
        self.inner.add_work_item(move |inner| {
            let mut state = inner.lock_state();
            let data = &mut state.data;
            for constraint in &constraints {
                assert!(
                    data.trajectory_nodes.contains(&constraint.node_id),
                    "Check failed: data_.trajectory_nodes.Contains(constraint.node_id)"
                );
                assert!(
                    data.submap_data.contains(&constraint.submap_id),
                    "Check failed: data_.submap_data.Contains(constraint.submap_id)"
                );
                let gravity_alignment =
                    constant_data_of(data.trajectory_nodes.at(&constraint.node_id))
                        .gravity_alignment;
                match constraint.tag {
                    Tag::IntraSubmap => {
                        assert!(
                            data.submap_data
                                .at_mut(&constraint.submap_id)
                                .node_ids
                                .insert(constraint.node_id),
                            "Check failed: data_.submap_data.at(constraint.submap_id)\
                             .node_ids.emplace(constraint.node_id).second"
                        );
                    }
                    Tag::InterSubmap => {
                        update_trajectory_connectivity(data, constraint);
                    }
                }
                let pose = Pose {
                    zbar_ij: constraint.pose.zbar_ij
                        * Rigid3d::from_rotation(quaternion_inverse(&gravity_alignment)),
                    translation_weight: constraint.pose.translation_weight,
                    rotation_weight: constraint.pose.rotation_weight,
                };
                data.constraints.push(Constraint {
                    submap_id: constraint.submap_id,
                    node_id: constraint.node_id,
                    pose,
                    tag: constraint.tag,
                });
            }
            log::info!("Loaded {} constraints.", constraints.len());
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn add_trimmer(&self, trimmer: Box<dyn PoseGraphTrimmer>) {
        // RUST PORT NOTE: C++11 cannot move a unique_ptr into a lambda and
        // releases it into a raw pointer; the Rust closure owns the box.
        self.inner.add_work_item(move |inner| {
            inner.lock_state().trimmers.push(trimmer);
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn get_connected_trajectories(&self) -> Vec<Vec<i32>> {
        self.inner
            .lock_state()
            .data
            .trajectory_connectivity_state
            .components()
    }

    fn get_imu_data(&self) -> MapByTime<ImuData> {
        self.inner.lock_optimization_problem().imu_data().clone()
    }

    fn get_odometry_data(&self) -> MapByTime<OdometryData> {
        self.inner
            .lock_optimization_problem()
            .odometry_data()
            .clone()
    }

    fn get_fixed_frame_pose_data(&self) -> MapByTime<FixedFramePoseData> {
        self.inner
            .lock_optimization_problem()
            .fixed_frame_pose_data()
            .clone()
    }

    fn get_landmark_nodes(&self) -> BTreeMap<String, LandmarkNode> {
        self.inner.lock_state().data.landmark_nodes.clone()
    }

    fn set_initial_trajectory_pose(
        &self,
        from_trajectory_id: i32,
        to_trajectory_id: i32,
        pose: &Rigid3d,
        time: Time,
    ) {
        self.inner
            .lock_state()
            .data
            .initial_trajectory_poses
            .insert(
                from_trajectory_id,
                InitialTrajectoryPose {
                    to_trajectory_id,
                    relative_pose: *pose,
                    time,
                },
            );
    }
}

#[cfg(test)]
mod tests {
    mod pose_graph_2d_test {
        use std::f64::consts::PI;
        use std::sync::{Arc, Mutex};

        use nalgebra::{DVector, Vector2, Vector3};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::common::thread_pool::{ThreadPool, ThreadPoolInterface};
        use crate::common::time::from_universal;
        use crate::mapping::id::{NodeId, SubmapId};
        use crate::mapping::internal::optimization::optimization_problem_2d::OptimizationProblem2D;
        use crate::mapping::internal::two_d::pose_graph_2d::PoseGraph2D;
        use crate::mapping::pose_graph::{PoseGraph, create_pose_graph_options};
        use crate::mapping::pose_graph_interface::{PoseGraphInterface, Tag, TrajectoryState};
        use crate::mapping::trajectory_node::TrajectoryNodeData;
        use crate::mapping::two_d::submap_2d::{ActiveSubmaps2D, create_submaps_options_2d};
        use crate::sensor::point_cloud::{PointCloud, transform_point_cloud};
        use crate::sensor::range_data::{RangeData, transform_range_data};
        use crate::sensor::rangefinder_point::RangefinderPoint;
        use crate::transform::rigid_transform::{Rigid2d, Rigid3d, quaternion_identity};
        use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
        use crate::transform::transform::{embed_3d, project_2d};

        const SUBMAPS_OPTIONS: &str = r#"
          return {
            num_range_data = 1,
            grid_options_2d = {
              grid_type = "PROBABILITY_GRID",
              resolution = 0.05,
            },
            range_data_inserter = {
              range_data_inserter_type = "PROBABILITY_GRID_INSERTER_2D",
              probability_grid_range_data_inserter = {
                insert_free_space = true,
                hit_probability = 0.53,
                miss_probability = 0.495,
              },
            tsdf_range_data_inserter = {
              truncation_distance = 0.3,
              maximum_weight = 10.,
              update_free_space = false,
              normal_estimation_options = {
                num_normal_samples = 4,
                sample_radius = 0.5,
              },
              project_sdf_distance_to_scan_normal = false,
              update_weight_range_exponent = 0,
              update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0,
              update_weight_distance_cell_to_hit_kernel_bandwidth = 0,
            },
          },
        }"#;

        const POSE_GRAPH_OPTIONS: &str = r#"
          return {
            optimize_every_n_nodes = 1000,
            constraint_builder = {
              sampling_ratio = 1.,
              max_constraint_distance = 6.,
              min_score = 0.5,
              global_localization_min_score = 0.6,
              loop_closure_translation_weight = 1.,
              loop_closure_rotation_weight = 1.,
              log_matches = true,
              fast_correlative_scan_matcher = {
                linear_search_window = 3.,
                angular_search_window = 0.1,
                branch_and_bound_depth = 3,
              },
              ceres_scan_matcher = {
                occupied_space_weight = 20.,
                translation_weight = 10.,
                rotation_weight = 1.,
                ceres_solver_options = {
                  use_nonmonotonic_steps = true,
                  max_num_iterations = 50,
                  num_threads = 1,
                },
              },
              fast_correlative_scan_matcher_3d = {
                branch_and_bound_depth = 3,
                full_resolution_depth = 3,
                min_rotational_score = 0.1,
                min_low_resolution_score = 0.5,
                linear_xy_search_window = 4.,
                linear_z_search_window = 4.,
                angular_search_window = 0.1,
              },
              ceres_scan_matcher_3d = {
                occupied_space_weight_0 = 20.,
                translation_weight = 10.,
                rotation_weight = 1.,
                only_optimize_yaw = true,
                ceres_solver_options = {
                  use_nonmonotonic_steps = true,
                  max_num_iterations = 50,
                  num_threads = 1,
                },
              },
            },
            matcher_translation_weight = 1.,
            matcher_rotation_weight = 1.,
            optimization_problem = {
              acceleration_weight = 1.,
              rotation_weight = 1e2,
              huber_scale = 1.,
              local_slam_pose_translation_weight = 0.,
              local_slam_pose_rotation_weight = 0.,
              odometry_translation_weight = 0.,
              odometry_rotation_weight = 0.,
              fixed_frame_pose_translation_weight = 1e1,
              fixed_frame_pose_rotation_weight = 1e2,
              fixed_frame_pose_use_tolerant_loss = false,
              fixed_frame_pose_tolerant_loss_param_a = 1,
              fixed_frame_pose_tolerant_loss_param_b = 1,
              log_solver_summary = true,
              use_online_imu_extrinsics_in_3d = true,
              fix_z_in_3d = false,
              ceres_solver_options = {
                use_nonmonotonic_steps = false,
                max_num_iterations = 200,
                num_threads = 1,
              },
            },
            max_num_final_iterations = 200,
            global_sampling_ratio = 0.01,
            log_residual_histograms = true,
            global_constraint_search_after_n_seconds = 10.0,
          }"#;

        /// The C++ fixture `PoseGraph2DTest`.
        ///
        /// Field order matters: the pose graph is dropped (and waits for its
        /// computations) before the thread pool, like the C++ members.
        struct PoseGraph2DTest {
            point_cloud: PointCloud,
            active_submaps: ActiveSubmaps2D,
            pose_graph: PoseGraph2D,
            _thread_pool: Arc<ThreadPool>,
            current_pose: Rigid2d,
        }

        impl PoseGraph2DTest {
            fn new() -> Self {
                Self::with_options(POSE_GRAPH_OPTIONS)
            }

            fn with_options(pose_graph_options: &str) -> Self {
                // Builds a wavy, irregularly circular point cloud that is
                // unique rotationally. This gives us good rotational texture
                // and avoids any possibility of the CeresScanMatcher2D
                // preferring free space (> kMinProbability) to unknown space
                // (== kMinProbability).
                let mut points = Vec::new();
                let mut t = 0.0f32;
                while f64::from(t) < 2. * PI {
                    let r = ((20. * t).sin() + 2.) * (t + 2.).sin();
                    points.push(RangefinderPoint {
                        position: Vector3::new(r * t.sin(), r * t.cos(), 0.),
                    });
                    t += 0.005;
                }
                let point_cloud = PointCloud::from_points(points);

                let thread_pool = Arc::new(ThreadPool::new(1));
                let options = create_pose_graph_options(&make_dictionary(pose_graph_options));
                let pose_graph = PoseGraph2D::new(
                    &options,
                    Box::new(OptimizationProblem2D::new(
                        &options.optimization_problem_options.unwrap_or_default(),
                    )),
                    Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
                );
                Self {
                    point_cloud,
                    active_submaps: new_active_submaps(),
                    pose_graph,
                    _thread_pool: thread_pool,
                    current_pose: Rigid2d::identity(),
                }
            }

            fn move_relative_with_noise_for(
                &mut self,
                movement: &Rigid2d,
                noise: &Rigid2d,
                trajectory_id: i32,
            ) -> NodeId {
                self.current_pose = self.current_pose * *movement;
                let new_point_cloud = transform_point_cloud(
                    &self.point_cloud,
                    &embed_3d(&self.current_pose.inverse().cast::<f32>()),
                );
                let range_data = RangeData {
                    origin: Vector3::zeros(),
                    returns: new_point_cloud,
                    misses: PointCloud::new(),
                };
                let pose_estimate = *noise * self.current_pose;
                self.active_submaps.insert_range_data(&transform_range_data(
                    &range_data,
                    &embed_3d(&pose_estimate.cast::<f32>()),
                ));
                let insertion_submaps = self.active_submaps.submaps();
                self.pose_graph.add_node(
                    Arc::new(TrajectoryNodeData {
                        time: from_universal(0),
                        gravity_alignment: quaternion_identity(),
                        filtered_gravity_aligned_point_cloud: range_data.returns,
                        high_resolution_point_cloud: PointCloud::new(),
                        low_resolution_point_cloud: PointCloud::new(),
                        rotational_scan_matcher_histogram: DVector::zeros(0),
                        local_pose: embed_3d(&pose_estimate),
                    }),
                    trajectory_id,
                    &insertion_submaps,
                )
            }

            fn move_relative_with_noise(&mut self, movement: &Rigid2d, noise: &Rigid2d) {
                const TRAJECTORY_ID: i32 = 0;
                self.move_relative_with_noise_for(movement, noise, TRAJECTORY_ID);
            }

            fn move_relative(&mut self, movement: &Rigid2d) {
                self.move_relative_with_noise(movement, &Rigid2d::identity());
            }
        }

        fn new_active_submaps() -> ActiveSubmaps2D {
            ActiveSubmaps2D::new(&create_submaps_options_2d(&make_dictionary(
                SUBMAPS_OPTIONS,
            )))
        }

        /// C++ `std::uniform_real_distribution<double>(-1., 1.)` (a seeded
        /// `StdRng` instead of `std::mt19937`).
        fn distribution(rng: &mut StdRng) -> f64 {
            rng.random_range(-1.0..1.0)
        }

        #[test]
        fn empty_map() {
            let test = PoseGraph2DTest::new();
            test.pose_graph.run_final_optimization();
            let nodes = test.pose_graph.get_trajectory_nodes();
            assert!(nodes.empty());
        }

        #[test]
        fn no_movement() {
            let mut test = PoseGraph2DTest::new();
            test.move_relative(&Rigid2d::identity());
            test.move_relative(&Rigid2d::identity());
            test.move_relative(&Rigid2d::identity());
            test.pose_graph.run_final_optimization();
            let nodes = test.pose_graph.get_trajectory_nodes();
            assert_eq!(nodes.trajectory_ids().collect::<Vec<_>>(), vec![0]);
            assert_eq!(nodes.size_of_trajectory_or_zero(0), 3);
            for i in 0..3 {
                assert_is_nearly(
                    &nodes.at(&NodeId::new(0, i)).global_pose,
                    &Rigid3d::identity(),
                    1e-2,
                );
            }
        }

        #[test]
        fn no_overlapping_nodes() {
            let mut test = PoseGraph2DTest::new();
            let mut rng = StdRng::seed_from_u64(0);
            let mut poses = Vec::new();
            for _ in 0..4 {
                let movement = Rigid2d::new(Vector2::new(0.25 * distribution(&mut rng), 5.), 0.);
                test.move_relative(&movement);
                poses.push(test.current_pose);
            }
            test.pose_graph.run_final_optimization();
            let nodes = test.pose_graph.get_trajectory_nodes();
            assert_eq!(nodes.trajectory_ids().collect::<Vec<_>>(), vec![0]);
            for (i, pose) in poses.iter().enumerate() {
                assert_is_nearly(
                    pose,
                    &project_2d(&nodes.at(&NodeId::new(0, i as i32)).global_pose),
                    1e-2,
                );
            }
        }

        #[test]
        fn consecutively_overlapping_nodes() {
            let mut test = PoseGraph2DTest::new();
            let mut rng = StdRng::seed_from_u64(0);
            let mut poses = Vec::new();
            for _ in 0..5 {
                let movement = Rigid2d::new(Vector2::new(0.25 * distribution(&mut rng), 2.), 0.);
                test.move_relative(&movement);
                poses.push(test.current_pose);
            }
            test.pose_graph.run_final_optimization();
            let nodes = test.pose_graph.get_trajectory_nodes();
            assert_eq!(nodes.trajectory_ids().collect::<Vec<_>>(), vec![0]);
            for (i, pose) in poses.iter().enumerate() {
                assert_is_nearly(
                    pose,
                    &project_2d(&nodes.at(&NodeId::new(0, i as i32)).global_pose),
                    1e-2,
                );
            }
        }

        #[test]
        fn overlapping_nodes() {
            let mut test = PoseGraph2DTest::new();
            let mut rng = StdRng::seed_from_u64(0);
            let mut ground_truth = Vec::new();
            let mut poses = Vec::new();
            for _ in 0..5 {
                let noise_x = 0.1 * distribution(&mut rng);
                let noise_y = 0.1 * distribution(&mut rng);
                let noise_orientation = 0.1 * distribution(&mut rng);
                let noise = Rigid2d::new(Vector2::new(noise_x, noise_y), noise_orientation);
                let movement = Rigid2d::new(Vector2::new(0.15 * distribution(&mut rng), 0.4), 0.);
                test.move_relative_with_noise(&movement, &noise);
                ground_truth.push(test.current_pose);
                poses.push(noise * test.current_pose);
            }
            test.pose_graph.run_final_optimization();
            let nodes = test.pose_graph.get_trajectory_nodes();
            assert_eq!(nodes.trajectory_ids().collect::<Vec<_>>(), vec![0]);
            let true_movement = ground_truth[0].inverse() * *ground_truth.last().unwrap();
            let movement_before = poses[0].inverse() * *poses.last().unwrap();
            let error_before = movement_before.inverse() * true_movement;
            let optimized_movement = nodes.begin_of_trajectory(0).data().global_pose.inverse()
                * nodes.end_of_trajectory(0).to_prev().data().global_pose;
            let optimized_error = project_2d(&optimized_movement).inverse() * true_movement;
            assert!(
                optimized_error.normalized_angle().abs() < error_before.normalized_angle().abs(),
                "{optimized_error:?} vs. {error_before:?}"
            );
            assert!(
                optimized_error.translation().norm() < error_before.translation().norm(),
                "{optimized_error:?} vs. {error_before:?}"
            );
        }

        // Additional Rust tests (no C++ counterpart).

        /// Adds `num_nodes` nodes moving 0.4 m each (consecutive nodes
        /// overlap; with `num_range_data = 1` a submap is finished after two
        /// nodes).
        fn add_straight_line(test: &mut PoseGraph2DTest, trajectory_id: i32, num_nodes: usize) {
            for _ in 0..num_nodes {
                test.move_relative_with_noise_for(
                    &Rigid2d::new(Vector2::new(0.05, 0.4), 0.),
                    &Rigid2d::identity(),
                    trajectory_id,
                );
            }
        }

        #[test]
        fn submaps_constraints_and_callback() {
            let mut test = PoseGraph2DTest::new();
            let callback_calls = Arc::new(Mutex::new(Vec::new()));
            {
                let callback_calls = Arc::clone(&callback_calls);
                test.pose_graph
                    .set_global_slam_optimization_callback(Box::new(move |submaps, nodes| {
                        callback_calls
                            .lock()
                            .unwrap()
                            .push((submaps.clone(), nodes.clone()));
                    }));
            }
            add_straight_line(&mut test, 0, 6);
            test.pose_graph.run_final_optimization();

            let submap_data = test.pose_graph.get_all_submap_data();
            let num_submaps = submap_data.size_of_trajectory_or_zero(0);
            assert!(num_submaps >= 3, "{num_submaps}");
            let submap_poses = test.pose_graph.get_all_submap_poses();
            assert_eq!(submap_poses.size(), num_submaps);
            for submap_id_data in &submap_data {
                let single = test.pose_graph.get_submap_data(&submap_id_data.id);
                assert_is_nearly(&single.pose, &submap_id_data.data.pose, 1e-12);
                let version = single.submap.unwrap().read().unwrap().num_range_data();
                assert_eq!(submap_poses.at(&submap_id_data.id).version, version);
            }
            assert!(
                test.pose_graph
                    .get_submap_data(&SubmapId::new(0, 100))
                    .submap
                    .is_none()
            );

            // Every node has intra-submap constraints to the submaps it was
            // inserted into; overlapping nodes found inter-submap constraints.
            let constraints = test.pose_graph.constraints();
            let num_intra = constraints
                .iter()
                .filter(|c| c.tag == Tag::IntraSubmap)
                .count();
            let num_inter = constraints.len() - num_intra;
            assert!(num_intra >= 6, "{num_intra}");
            assert!(num_inter > 0, "{num_inter}");

            let node_poses = test.pose_graph.get_trajectory_node_poses();
            assert_eq!(node_poses.size(), 6);
            assert!(
                node_poses
                    .iter()
                    .all(|node| node.data.constant_pose_data.is_some())
            );

            let calls = callback_calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            let (last_submaps, last_nodes) = &calls[0];
            assert_eq!(last_nodes[&0], NodeId::new(0, 5));
            assert_eq!(last_submaps[&0], SubmapId::new(0, num_submaps as i32 - 1));
            assert_eq!(test.pose_graph.get_connected_trajectories(), vec![vec![0]]);
        }

        #[test]
        fn trajectory_state_transitions() {
            let mut test = PoseGraph2DTest::new();
            add_straight_line(&mut test, 0, 5);
            test.pose_graph.wait_for_all_computations();
            assert_eq!(
                test.pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Active)].into()
            );
            assert!(!test.pose_graph.is_trajectory_finished(0));
            assert!(!test.pose_graph.is_trajectory_frozen(0));

            test.pose_graph.finish_trajectory(0);
            test.pose_graph.run_final_optimization();
            assert!(test.pose_graph.is_trajectory_finished(0));
            assert_eq!(
                test.pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Finished)].into()
            );
            assert!(test.pose_graph.get_all_submap_data().size() > 0);
            assert_eq!(test.pose_graph.get_trajectory_nodes().size(), 5);

            // Deleting an unknown trajectory is a no-op.
            test.pose_graph.delete_trajectory(42);
            test.pose_graph.delete_trajectory(0);
            // The trajectory is deleted by the next optimization.
            test.pose_graph.run_final_optimization();
            assert_eq!(
                test.pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Deleted)].into()
            );
            assert!(test.pose_graph.get_all_submap_data().empty());
            assert!(test.pose_graph.get_trajectory_nodes().empty());
            assert!(test.pose_graph.constraints().is_empty());

            // Another trajectory can be added and frozen.
            test.current_pose = Rigid2d::identity();
            test.active_submaps = new_active_submaps();
            add_straight_line(&mut test, 1, 3);
            test.pose_graph.freeze_trajectory(1);
            test.pose_graph.run_final_optimization();
            assert!(test.pose_graph.is_trajectory_frozen(1));
            assert_eq!(
                test.pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Deleted), (1, TrajectoryState::Frozen)].into()
            );
        }

        #[test]
        fn initial_trajectory_pose() {
            let mut test = PoseGraph2DTest::new();
            add_straight_line(&mut test, 0, 3);
            test.pose_graph.run_final_optimization();
            let relative_pose = Rigid3d::from_translation(Vector3::new(1., 2., 0.));
            test.pose_graph
                .set_initial_trajectory_pose(1, 0, &relative_pose, from_universal(0));
            // All nodes have the same time: the first node is used.
            let first_node_pose = test
                .pose_graph
                .get_trajectory_nodes()
                .begin()
                .data()
                .global_pose;
            assert_is_nearly(
                &test
                    .pose_graph
                    .get_interpolated_global_trajectory_pose(0, from_universal(0)),
                &first_node_pose,
                1e-12,
            );
            let expected = first_node_pose * relative_pose;
            assert_is_nearly(
                &test.pose_graph.get_local_to_global_transform(1),
                &expected,
                1e-12,
            );

            test.current_pose = Rigid2d::identity();
            test.active_submaps = new_active_submaps();
            let node_id =
                test.move_relative_with_noise_for(&Rigid2d::identity(), &Rigid2d::identity(), 1);
            assert_eq!(node_id, NodeId::new(1, 0));
            assert_is_nearly(
                &test
                    .pose_graph
                    .get_trajectory_nodes()
                    .at(&node_id)
                    .global_pose,
                &expected,
                1e-12,
            );
            test.pose_graph.wait_for_all_computations();
            // The initial pose connects the trajectories.
            assert_eq!(
                test.pose_graph.get_connected_trajectories(),
                vec![vec![0, 1]]
            );
        }

        #[test]
        fn landmarks_and_sensor_data() {
            use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
            use crate::sensor::landmark_data::{LandmarkData, LandmarkObservation};
            use crate::sensor::odometry_data::OdometryData;

            let test = PoseGraph2DTest::new();
            let pose = Rigid3d::from_translation(Vector3::new(1., 2., 3.));
            test.pose_graph.set_landmark_pose("landmark", &pose, true);
            test.pose_graph.add_landmark_data(
                0,
                &LandmarkData {
                    time: from_universal(5),
                    landmark_observations: vec![LandmarkObservation {
                        id: "landmark".to_string(),
                        landmark_to_tracking_transform: Rigid3d::identity(),
                        translation_weight: 1.,
                        rotation_weight: 2.,
                    }],
                },
            );
            test.pose_graph.add_odometry_data(
                0,
                &OdometryData {
                    time: from_universal(5),
                    pose: Rigid3d::identity(),
                },
            );
            test.pose_graph.add_fixed_frame_pose_data(
                0,
                &FixedFramePoseData {
                    time: from_universal(5),
                    pose: Some(Rigid3d::identity()),
                },
            );
            test.pose_graph.wait_for_all_computations();
            assert_eq!(
                test.pose_graph.get_landmark_poses(),
                [("landmark".to_string(), pose)].into()
            );
            let landmark_nodes = test.pose_graph.get_landmark_nodes();
            let landmark = &landmark_nodes["landmark"];
            assert!(landmark.frozen);
            assert_eq!(landmark.landmark_observations.len(), 1);
            assert_eq!(landmark.landmark_observations[0].trajectory_id, 0);
            assert_eq!(landmark.landmark_observations[0].rotation_weight, 2.);
            assert_eq!(test.pose_graph.get_odometry_data().trajectory(0).count(), 1);
            assert_eq!(
                test.pose_graph
                    .get_fixed_frame_pose_data()
                    .trajectory(0)
                    .count(),
                1
            );
            // The 2D optimization problem ignores IMU data.
            assert_eq!(test.pose_graph.get_imu_data().trajectory_ids().count(), 0);
        }

        /// Serializes a pose graph like `MapBuilder::SerializeState` and
        /// loads it into a fresh pose graph like `MapBuilder::LoadState`.
        #[test]
        fn to_proto_round_trip() {
            use crate::mapping::trajectory_node::to_proto as node_to_proto;
            use crate::mapping::{pose_graph, proto};
            use crate::transform::transform::to_rigid3;

            let mut test = PoseGraph2DTest::new();
            add_straight_line(&mut test, 0, 6);
            test.pose_graph.run_final_optimization();
            let original = test.pose_graph.to_proto(true);
            let submap_data = test.pose_graph.get_all_submap_data();
            assert_eq!(original.trajectory.len(), 1);
            assert_eq!(original.trajectory[0].trajectory_id, 0);
            assert_eq!(original.trajectory[0].node.len(), 6);
            assert_eq!(original.trajectory[0].submap.len(), submap_data.size());
            assert_eq!(
                original.constraint.len(),
                test.pose_graph.constraints().len()
            );

            // Unfinished submaps and their constraints are left out on request.
            let finished_only = test.pose_graph.to_proto(false);
            let unfinished: Vec<SubmapId> = submap_data
                .iter()
                .filter(|s| {
                    !s.data
                        .submap
                        .as_ref()
                        .unwrap()
                        .read()
                        .unwrap()
                        .insertion_finished()
                })
                .map(|s| s.id)
                .collect();
            assert!(!unfinished.is_empty());
            assert_eq!(
                finished_only.trajectory[0].submap.len(),
                original.trajectory[0].submap.len() - unfinished.len()
            );
            assert!(
                finished_only
                    .constraint
                    .iter()
                    .all(|c| !unfinished.contains(&SubmapId::from_proto(&c.submap_id.unwrap())))
            );
            assert!(finished_only.constraint.len() < original.constraint.len());
            assert_eq!(
                finished_only.trajectory[0].node,
                original.trajectory[0].node
            );

            // Load into a new pose graph.
            let loaded = PoseGraph2DTest::new();
            for submap_id_data in &submap_data {
                let mut submap_proto = submap_id_data
                    .data
                    .submap
                    .as_ref()
                    .unwrap()
                    .read()
                    .unwrap()
                    .to_proto(true);
                submap_proto.submap_id = Some(submap_id_data.id.to_proto());
                loaded
                    .pose_graph
                    .add_submap_from_proto(&submap_id_data.data.pose, &submap_proto);
            }
            for node_id_data in &test.pose_graph.get_trajectory_nodes() {
                let node_proto = proto::Node {
                    node_id: Some(node_id_data.id.to_proto()),
                    node_data: Some(node_to_proto(
                        node_id_data.data.constant_data.as_ref().unwrap(),
                    )),
                };
                loaded
                    .pose_graph
                    .add_node_from_proto(&node_id_data.data.global_pose, &node_proto);
            }
            loaded
                .pose_graph
                .add_serialized_constraints(&pose_graph::from_proto(&original.constraint));
            loaded.pose_graph.wait_for_all_computations();

            let reloaded = loaded.pose_graph.to_proto(true);
            assert_eq!(reloaded.constraint, original.constraint);
            assert_eq!(reloaded.trajectory.len(), 1);
            assert_eq!(reloaded.trajectory[0].node, original.trajectory[0].node);
            assert_eq!(
                reloaded.trajectory[0].submap.len(),
                original.trajectory[0].submap.len()
            );
            for (a, b) in reloaded.trajectory[0]
                .submap
                .iter()
                .zip(&original.trajectory[0].submap)
            {
                assert_eq!(a.submap_index, b.submap_index);
                assert_is_nearly(
                    &to_rigid3(a.pose.as_ref().unwrap()),
                    &to_rigid3(b.pose.as_ref().unwrap()),
                    1e-12,
                );
            }
            // The loaded graph can be optimized.
            loaded.pose_graph.run_final_optimization();
            assert_eq!(loaded.pose_graph.get_trajectory_nodes().size(), 6);
        }

        #[test]
        fn overlapping_submaps_trimmer_from_options() {
            let options = POSE_GRAPH_OPTIONS.replace(
                "global_constraint_search_after_n_seconds = 10.0,",
                "global_constraint_search_after_n_seconds = 10.0,\n\
                 overlapping_submaps_trimmer_2d = {\n\
                   fresh_submaps_count = 1,\n\
                   min_covered_area = 2,\n\
                   min_added_submaps_count = 1,\n\
                 },",
            );
            let mut test = PoseGraph2DTest::with_options(&options);
            // Nodes on top of each other: the submaps overlap completely and
            // only the freshest one is kept per cell.
            for _ in 0..8 {
                test.move_relative(&Rigid2d::identity());
            }
            test.pose_graph.finish_trajectory(0);
            test.pose_graph.run_final_optimization();
            let num_submaps = test.pose_graph.get_all_submap_data().size();
            assert!(num_submaps < 8, "{num_submaps}");
        }
    }
}
