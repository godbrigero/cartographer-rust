//! Port of `cartographer/mapping/internal/3d/pose_graph_3d` (.h, .cc).
//!
//! # Design
//!
//! RUST PORT NOTE: ownership, threading and locking are exactly those of
//! `two_d/pose_graph_2d.rs` (see its module documentation): a thin public
//! handle owning an `Arc<Inner>`, tasks and callbacks holding a
//! `Weak<Inner>`, `Inner::state: Mutex<PoseGraphState>` for the C++
//! `mutex_`, `Inner::work_queue: Mutex<Option<WorkQueue>>` + `Condvar` for
//! `work_queue_mutex_`/`work_queue_`, the optimization problem behind its own
//! mutex (lock order: optimization problem -> state -> constraint builder),
//! `Solve` running with only the optimization problem lock held (inputs
//! copied out of the state first), `Arc` handles for submaps and node data
//! given to background jobs, and the same drop protocol (the last handle
//! must not be dropped on a thread pool thread). The dimension-agnostic
//! helpers (trajectory states, connectivity, interpolation, the
//! `TrimmingHandle`, metrics) are shared with the 2D pose graph in
//! `pose_graph_common.rs`.
//!
//! 3D specifics (§3.9 of `docs/architecture/pose_graph.md`): full `Rigid3d`
//! node and submap poses (`NodeSpec3D`/`SubmapSpec3D`,
//! `global_submap_poses_3d`), constraints stored verbatim (no gravity
//! alignment), the constraint builder receives global node and submap poses
//! (local search) or rotations (global search), `SetTrajectoryDataFromProto`
//! always forwards gravity constant and IMU calibration, residual histograms
//! are logged after each optimization, and there is no overlapping submaps
//! trimmer.
//!
//! RUST PORT NOTE: C++ `OptimizationProblem3D` has virtual methods and the
//! C++ tests replace `Solve` by a mock. The pose graph therefore stores the
//! problem as a trait object of the private `OptimizationProblem3DApi`
//! (implemented by [`OptimizationProblem3D`]); the public constructor takes a
//! `Box<OptimizationProblem3D>` like C++ and like `PoseGraph2D`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock, Weak};
use std::time::{Duration as StdDuration, Instant};

use crate::common::histogram::Histogram;
use crate::common::math::normalize_angle_difference;
use crate::common::task::Task;
use crate::common::thread_pool::ThreadPoolInterface;
use crate::common::time::{Time, from_seconds, from_universal};
use crate::mapping::constraints::ConstraintBuilder3D;
use crate::mapping::id::{ConstIterator, MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::optimization_problem_3d::{
    NodeSpec3D, OptimizationProblem3D, SubmapSpec3D,
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
use crate::mapping::internal::work_queue::{WorkItem, WorkItemResult, WorkQueue};
use crate::mapping::pose_graph::{InitialTrajectoryPose, PoseGraph, pose_graph_to_proto};
use crate::mapping::pose_graph_interface::{
    Constraint, GlobalSlamOptimizationCallback, LandmarkNode, LandmarkObservation, Pose,
    PoseGraphInterface, SubmapData, SubmapPose, Tag, TrajectoryData, TrajectoryState,
};
use crate::mapping::pose_graph_trimmer::PoseGraphTrimmer;
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::trajectory_node::{
    ConstantPoseData, TrajectoryNode, TrajectoryNodeData, TrajectoryNodePose,
    from_proto as trajectory_node_data_from_proto,
};
use crate::metrics::family_factory::FamilyFactory;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{get_angle, to_rigid3};

// RUST PORT NOTE: the C++ `static auto* k...Metric` globals; `None` means the
// null metrics.
static METRICS: RwLock<Option<PoseGraphMetrics>> = RwLock::new(None);

fn metrics() -> PoseGraphMetrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(PoseGraphMetrics::null)
}

/// The virtual interface of C++ `optimization::OptimizationProblem3D` used by
/// the pose graph (see the module documentation).
trait OptimizationProblem3DApi:
    OptimizationProblemInterface<
        NodeData = NodeSpec3D,
        SubmapData = SubmapSpec3D,
        RigidTransform = Rigid3d,
    > + Send
{
    fn add_fixed_frame_pose_data(
        &mut self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    );
    fn set_trajectory_data(&mut self, trajectory_id: i32, trajectory_data: &TrajectoryData);
    fn fixed_frame_pose_data(&self) -> &MapByTime<FixedFramePoseData>;
    fn trajectory_data(&self) -> &BTreeMap<i32, TrajectoryData>;
}

impl OptimizationProblem3DApi for OptimizationProblem3D {
    fn add_fixed_frame_pose_data(
        &mut self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    ) {
        OptimizationProblem3D::add_fixed_frame_pose_data(
            self,
            trajectory_id,
            fixed_frame_pose_data,
        );
    }

    fn set_trajectory_data(&mut self, trajectory_id: i32, trajectory_data: &TrajectoryData) {
        OptimizationProblem3D::set_trajectory_data(self, trajectory_id, trajectory_data);
    }

    fn fixed_frame_pose_data(&self) -> &MapByTime<FixedFramePoseData> {
        OptimizationProblem3D::fixed_frame_pose_data(self)
    }

    fn trajectory_data(&self) -> &BTreeMap<i32, TrajectoryData> {
        OptimizationProblem3D::trajectory_data(self)
    }
}

type DynOptimizationProblem = dyn OptimizationProblem3DApi;

/// The data guarded by the C++ `mutex_`.
type State = PoseGraphState<Submap3D>;

/// Computes the local to global map frame transform based on the given
/// `global_submap_poses` (requires the state lock).
fn compute_local_to_global_transform(
    data: &PoseGraphData<Submap3D>,
    global_submap_poses: &MapById<SubmapId, SubmapSpec3D>,
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
    global_submap_poses
        .at(&last_optimized_submap_id)
        .global_pose
        * read_submap(&data.submap_data.at(&last_optimized_submap_id).submap)
            .local_pose()
            .inverse()
}

fn get_submap_data_under_lock(data: &PoseGraphData<Submap3D>, submap_id: &SubmapId) -> SubmapData {
    let Some(internal_submap_data) = data.submap_data.get(submap_id) else {
        return SubmapData::default();
    };
    let submap = internal_submap_data.submap_as_dyn();
    if let Some(global_submap_pose) = data.global_submap_poses_3d.get(submap_id) {
        // We already have an optimized pose.
        return SubmapData {
            submap: Some(submap),
            pose: global_submap_pose.global_pose,
        };
    }
    // We have to extrapolate.
    let pose = compute_local_to_global_transform(
        data,
        &data.global_submap_poses_3d,
        submap_id.trajectory_id,
    ) * read_submap(&internal_submap_data.submap).local_pose();
    SubmapData {
        submap: Some(submap),
        pose,
    }
}

fn get_all_submap_data_under_lock(data: &PoseGraphData<Submap3D>) -> MapById<SubmapId, SubmapData> {
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
    optimization_problem: &mut DynOptimizationProblem,
    data: &mut PoseGraphData<Submap3D>,
    trajectory_id: i32,
    time: Time,
    insertion_submaps: &[SharedSubmap<Submap3D>],
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
            let global_submap_pose = compute_local_to_global_transform(
                data,
                &data.global_submap_poses_3d,
                trajectory_id,
            ) * read_submap(&insertion_submaps[0]).local_pose();
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
            * read_submap(&insertion_submaps[0]).local_pose().inverse()
            * read_submap(&insertion_submaps[1]).local_pose();
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

/// The optimized global pose of a submap (`data_.global_submap_poses_3d`),
/// for the `TrimmingHandle`.
fn optimized_global_submap_pose(
    data: &PoseGraphData<Submap3D>,
    submap_id: &SubmapId,
) -> Option<Rigid3d> {
    data.global_submap_poses_3d
        .get(submap_id)
        .map(|global_submap_pose| global_submap_pose.global_pose)
}

/// Creates the C++ `TrimmingHandle(this)` from the locked parts.
fn trimming_handle<'a>(
    data: &'a mut PoseGraphData<Submap3D>,
    optimization_problem: &'a mut DynOptimizationProblem,
    delete_scan_matcher: &'a dyn Fn(&SubmapId),
) -> TrimmingHandle<'a, Submap3D, DynOptimizationProblem> {
    TrimmingHandle {
        data,
        optimization_problem,
        delete_scan_matcher,
        optimized_global_submap_pose,
        metrics: metrics(),
    }
}

/// Logs histograms for the translational and rotational residual of node
/// poses (requires the state lock).
fn log_residual_histograms(data: &PoseGraphData<Submap3D>) {
    let mut rotational_residual = Histogram::new();
    let mut translational_residual = Histogram::new();
    for constraint in &data.constraints {
        if constraint.tag == Tag::IntraSubmap {
            let optimized_node_to_map = data.trajectory_nodes.at(&constraint.node_id).global_pose;
            let node_to_submap_constraint = constraint.pose.zbar_ij;
            let optimized_submap_to_map = data
                .global_submap_poses_3d
                .at(&constraint.submap_id)
                .global_pose;
            let optimized_node_to_submap =
                optimized_submap_to_map.inverse() * optimized_node_to_map;
            let residual = node_to_submap_constraint.inverse() * optimized_node_to_submap;
            rotational_residual.add(normalize_angle_difference(get_angle(&residual)) as f32);
            translational_residual.add(residual.translation().norm() as f32);
        }
    }
    log::info!(
        "Translational residuals histogram:\n{}",
        translational_residual.to_string(10)
    );
    log::info!(
        "Rotational residuals histogram:\n{}",
        rotational_residual.to_string(10)
    );
}

/// The shared state of [`PoseGraph3D`] (everything the C++ class holds).
struct Inner {
    options: proto::PoseGraphOptions,
    /// Handle to ourselves, captured by tasks and callbacks (C++ `this`).
    weak_self: Weak<Inner>,
    /// RUST PORT NOTE: unguarded in C++ (set before use); a mutex here.
    global_slam_optimization_callback: Mutex<Option<Arc<GlobalSlamOptimizationCallback>>>,
    /// Current optimization problem. Lock order: before `state`.
    optimization_problem: Mutex<Box<DynOptimizationProblem>>,
    /// C++ `mutex_` and the members it guards.
    state: Mutex<State>,
    /// If it exists, further work items must be added to this queue, and
    /// will be considered later (C++ `work_queue_mutex_` + `work_queue_`).
    work_queue: Mutex<Option<WorkQueue<Inner>>>,
    /// Notified when `work_queue` becomes `None`.
    work_queue_drained: Condvar,
    constraint_builder: ConstraintBuilder3D,
    /// Thread pool used for handling the work queue.
    thread_pool: Arc<dyn ThreadPoolInterface>,
}

impl Inner {
    fn lock_state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    fn lock_optimization_problem(&self) -> MutexGuard<'_, Box<DynOptimizationProblem>> {
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
        insertion_submaps: &[SharedSubmap<Submap3D>],
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
        let (global_node_pose, global_submap_pose) = {
            let optimization_problem = self.lock_optimization_problem();
            (
                optimization_problem.node_data().at(node_id).global_pose,
                optimization_problem.submap_data().at(submap_id).global_pose,
            )
        };

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
                // In this situation, 'global_node_pose' and
                // 'global_submap_pose' have orientations agreeing on gravity.
                // Their relationship regarding yaw is arbitrary. Finding the
                // correct yaw component will be handled by the matching
                // procedure in the FastCorrelativeScanMatcher, and the given
                // yaw is essentially ignored.
                maybe_add_global_constraint = true;
            }
            constant_data = Arc::clone(constant_data_of(data.trajectory_nodes.at(node_id)));
            submap = Arc::clone(&data.submap_data.at(submap_id).submap);
        }

        if maybe_add_local_constraint {
            self.constraint_builder.maybe_add_constraint(
                submap_id,
                &submap,
                node_id,
                &constant_data,
                &global_node_pose,
                &global_submap_pose,
            );
        } else if maybe_add_global_constraint {
            self.constraint_builder.maybe_add_global_constraint(
                submap_id,
                &submap,
                node_id,
                &constant_data,
                global_node_pose.rotation(),
                global_submap_pose.rotation(),
            );
        }
    }

    /// Adds constraints for a node, and starts scan matching in the
    /// background.
    fn compute_constraints_for_node(
        &self,
        node_id: &NodeId,
        insertion_submaps: Vec<SharedSubmap<Submap3D>>,
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
                optimization_problem.as_mut(),
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
            let local_pose = constant_data.local_pose;
            let global_pose = optimization_problem
                .submap_data()
                .at(&matching_id)
                .global_pose
                * read_submap(&insertion_submaps[0]).local_pose().inverse()
                * local_pose;
            optimization_problem.add_trajectory_node(
                matching_id.trajectory_id,
                &NodeSpec3D {
                    time: constant_data.time,
                    local_pose,
                    global_pose,
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
                    read_submap(insertion_submap).local_pose().inverse() * local_pose;
                data.constraints.push(Constraint {
                    submap_id: *submap_id,
                    node_id: *node_id,
                    pose: Pose {
                        zbar_ij: constraint_transform,
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
                    trimming_handle(data, optimization_problem.as_mut(), &delete_scan_matcher);
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
        // drop protocol in the `PoseGraph2D` module documentation).
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
                data.trajectory_nodes.at_mut(&node.id).global_pose = node.data.global_pose;
            }

            // Extrapolate all point cloud poses that were not included in the
            // 'optimization_problem_' yet.
            let local_to_new_global =
                compute_local_to_global_transform(data, submap_data, trajectory_id);
            let local_to_old_global = compute_local_to_global_transform(
                data,
                &data.global_submap_poses_3d,
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
        data.global_submap_poses_3d = submap_data.clone();

        // Log the histograms for the pose residuals.
        if self.options.log_residual_histograms {
            log_residual_histograms(data);
        }
    }
}

/// Implements the loop closure method called Sparse Pose Adjustment (SPA)
/// from Konolige, Kurt, et al. "Efficient sparse pose adjustment for 2d
/// mapping." Intelligent Robots and Systems (IROS), 2010 IEEE/RSJ
/// International Conference on (pp. 22--29). IEEE, 2010.
///
/// It is extended for submapping in 3D:
/// Each node has been matched against one or more submaps (adding a
/// constraint for each match), both poses of nodes and of submaps are to be
/// optimized. All constraints are between a submap i and a node j.
///
/// See the module documentation for the threading model.
pub struct PoseGraph3D {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for PoseGraph3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseGraph3D")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl PoseGraph3D {
    /// Creates the pose graph. Work items, constraint searches and
    /// optimizations run on `thread_pool`, which must outlive the pose graph
    /// (C++ `common::ThreadPool*`).
    pub fn new(
        options: &proto::PoseGraphOptions,
        optimization_problem: Box<OptimizationProblem3D>,
        thread_pool: Arc<dyn ThreadPoolInterface>,
    ) -> Self {
        Self::with_optimization_problem(options, optimization_problem, thread_pool)
    }

    /// [`PoseGraph3D::new`] with any implementation of the optimization
    /// problem interface (C++ passes subclasses of `OptimizationProblem3D`,
    /// e.g. the tests' `MockOptimizationProblem3D`).
    fn with_optimization_problem(
        options: &proto::PoseGraphOptions,
        optimization_problem: Box<DynOptimizationProblem>,
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
            constraint_builder: ConstraintBuilder3D::new(
                &options
                    .constraint_builder_options
                    .clone()
                    .unwrap_or_default(),
                Arc::clone(&thread_pool),
            ),
            thread_pool,
        });
        PoseGraph3D { inner }
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
        insertion_submaps: &[SharedSubmap<Submap3D>],
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
    /// that were requested so far have finished (C++ protected
    /// `WaitForAllComputations`, exposed by the tests'
    /// `PoseGraph3DForTesting`).
    #[cfg(test)]
    pub(crate) fn wait_for_all_computations(&self) {
        self.inner.wait_for_all_computations();
    }

    /// Registers the metrics with `family_factory`.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(
            PoseGraphMetrics::register(family_factory, "mapping_3d_pose_graph"),
        );
    }
}

impl Drop for PoseGraph3D {
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

impl PoseGraphInterface for PoseGraph3D {
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
            &state.data.global_submap_poses_3d,
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
        self.inner.lock_state().data.constraints.clone()
    }

    fn to_proto(&self, include_unfinished_submaps: bool) -> proto::PoseGraph {
        pose_graph_to_proto(self, include_unfinished_submaps)
    }

    fn set_global_slam_optimization_callback(&self, callback: GlobalSlamOptimizationCallback) {
        *lock(&self.inner.global_slam_optimization_callback) = Some(Arc::new(callback));
    }
}

impl PoseGraph for PoseGraph3D {
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
        let Some(submap_3d) = submap.submap_3d.as_ref() else {
            return;
        };

        let submap_id = SubmapId::from_proto(&submap.submap_id.unwrap_or_default());
        let submap_ptr = Arc::new(RwLock::new(Submap3D::from_proto(submap_3d)));

        let global_submap_pose = *global_submap_pose;
        {
            let mut state = self.inner.lock_state();
            state.add_trajectory_if_needed(submap_id.trajectory_id, &self.inner.options);
            if !can_add_work_item_modifying(&state.data, submap_id.trajectory_id) {
                return;
            }
            state
                .data
                .submap_data
                .insert(&submap_id, InternalSubmapData::new(submap_ptr));
            // Immediately show the submap at the 'global_submap_pose'.
            state.data.global_submap_poses_3d.insert(
                &submap_id,
                SubmapSpec3D {
                    global_pose: global_submap_pose,
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
            optimization_problem.insert_submap(&submap_id, &global_submap_pose);
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
            optimization_problem.insert_trajectory_node(
                &node_id,
                &NodeSpec3D {
                    time: constant_data.time,
                    local_pose: constant_data.local_pose,
                    global_pose,
                },
            );
            WorkItemResult::DoNotRunOptimization
        });
    }

    fn set_trajectory_data_from_proto(&self, data: &proto::TrajectoryData) {
        let imu_calibration = data.imu_calibration.unwrap_or_default();
        let trajectory_data = TrajectoryData {
            gravity_constant: data.gravity_constant,
            imu_calibration: [
                imu_calibration.w,
                imu_calibration.x,
                imu_calibration.y,
                imu_calibration.z,
            ],
            fixed_frame_origin_in_map: data.fixed_frame_origin_in_map.as_ref().map(to_rigid3),
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
                assert!(
                    data.trajectory_nodes
                        .at(&constraint.node_id)
                        .constant_data
                        .is_some(),
                    "Check failed: data_.trajectory_nodes.at(constraint.node_id).constant_data \
                     != nullptr"
                );
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
                data.constraints.push(*constraint);
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
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::{OptimizationProblem3DApi, PoseGraph3D};
    use crate::common::thread_pool::{ThreadPool, ThreadPoolInterface};
    use crate::mapping::id::{MapById, NodeId, SubmapId};
    use crate::mapping::internal::optimization::optimization_problem_3d::{
        NodeSpec3D, OptimizationProblem3D, SubmapSpec3D,
    };
    use crate::mapping::internal::optimization::optimization_problem_interface::OptimizationProblemInterface;
    use crate::mapping::internal::testing::test_helpers::{
        add_to_proto_graph, create_fake_constraint, create_fake_landmark, create_fake_node,
        create_fake_node_default, create_fake_submap_3d, create_fake_submap_3d_default,
        resolve_lua_parameters,
    };
    use crate::mapping::pose_graph::{PoseGraph, create_pose_graph_options, from_proto};
    use crate::mapping::pose_graph_interface::{
        Constraint, LandmarkNode, PoseGraphInterface, TrajectoryData, TrajectoryState,
    };
    use crate::mapping::pose_graph_trimmer::{
        PoseGraphTrimmer, PureLocalizationTrimmer, Trimmable,
    };
    use crate::mapping::proto;
    use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
    use crate::sensor::imu_data::ImuData;
    use crate::sensor::map_by_time::MapByTime;
    use crate::sensor::odometry_data::OdometryData;
    use crate::transform::rigid_transform::Rigid3d;

    /// C++ `MockOptimizationProblem3D`: an `OptimizationProblem3D` with
    /// default options whose `Solve` does nothing.
    struct MockOptimizationProblem3D(OptimizationProblem3D);

    impl MockOptimizationProblem3D {
        fn new() -> Self {
            Self(OptimizationProblem3D::new(
                &crate::mapping::optimization::proto::OptimizationProblemOptions::default(),
            ))
        }
    }

    impl OptimizationProblemInterface for MockOptimizationProblem3D {
        type NodeData = NodeSpec3D;
        type SubmapData = SubmapSpec3D;
        type RigidTransform = Rigid3d;

        fn add_imu_data(&mut self, trajectory_id: i32, imu_data: &ImuData) {
            self.0.add_imu_data(trajectory_id, imu_data);
        }
        fn add_odometry_data(&mut self, trajectory_id: i32, odometry_data: &OdometryData) {
            self.0.add_odometry_data(trajectory_id, odometry_data);
        }
        fn add_trajectory_node(&mut self, trajectory_id: i32, node_data: &NodeSpec3D) {
            self.0.add_trajectory_node(trajectory_id, node_data);
        }
        fn insert_trajectory_node(&mut self, node_id: &NodeId, node_data: &NodeSpec3D) {
            self.0.insert_trajectory_node(node_id, node_data);
        }
        fn trim_trajectory_node(&mut self, node_id: &NodeId) {
            self.0.trim_trajectory_node(node_id);
        }
        fn add_submap(&mut self, trajectory_id: i32, global_submap_pose: &Rigid3d) {
            self.0.add_submap(trajectory_id, global_submap_pose);
        }
        fn insert_submap(&mut self, submap_id: &SubmapId, global_submap_pose: &Rigid3d) {
            self.0.insert_submap(submap_id, global_submap_pose);
        }
        fn trim_submap(&mut self, submap_id: &SubmapId) {
            self.0.trim_submap(submap_id);
        }
        fn set_max_num_iterations(&mut self, max_num_iterations: i32) {
            self.0.set_max_num_iterations(max_num_iterations);
        }
        // MOCK_METHOD3(Solve, ...) without expectations: does nothing.
        fn solve(
            &mut self,
            _constraints: &[Constraint],
            _trajectories_state: &BTreeMap<i32, TrajectoryState>,
            _landmark_nodes: &BTreeMap<String, LandmarkNode>,
        ) {
        }
        fn node_data(&self) -> &MapById<NodeId, NodeSpec3D> {
            self.0.node_data()
        }
        fn submap_data(&self) -> &MapById<SubmapId, SubmapSpec3D> {
            self.0.submap_data()
        }
        fn landmark_data(&self) -> &BTreeMap<String, Rigid3d> {
            self.0.landmark_data()
        }
        fn imu_data(&self) -> &MapByTime<ImuData> {
            self.0.imu_data()
        }
        fn odometry_data(&self) -> &MapByTime<OdometryData> {
            self.0.odometry_data()
        }
    }

    impl OptimizationProblem3DApi for MockOptimizationProblem3D {
        fn add_fixed_frame_pose_data(
            &mut self,
            trajectory_id: i32,
            fixed_frame_pose_data: &FixedFramePoseData,
        ) {
            self.0
                .add_fixed_frame_pose_data(trajectory_id, fixed_frame_pose_data);
        }
        fn set_trajectory_data(&mut self, trajectory_id: i32, trajectory_data: &TrajectoryData) {
            self.0.set_trajectory_data(trajectory_id, trajectory_data);
        }
        fn fixed_frame_pose_data(&self) -> &MapByTime<FixedFramePoseData> {
            self.0.fixed_frame_pose_data()
        }
        fn trajectory_data(&self) -> &BTreeMap<i32, TrajectoryData> {
            self.0.trajectory_data()
        }
    }

    /// The C++ fixture `PoseGraph3DTest` (`PoseGraph3DForTesting` is
    /// `PoseGraph3D` itself, whose `wait_for_all_computations` is visible
    /// to tests).
    ///
    /// Field order matters: the pose graph is dropped (and waits for its
    /// computations) before the thread pool, like the C++ members.
    struct PoseGraph3DTest {
        pose_graph: Option<PoseGraph3D>,
        pose_graph_options: proto::PoseGraphOptions,
        thread_pool: Arc<ThreadPool>,
    }

    impl PoseGraph3DTest {
        fn new() -> Self {
            const POSE_GRAPH_LUA: &str = r#"
              include "pose_graph.lua"
              return POSE_GRAPH"#;
            let pose_graph_parameters = resolve_lua_parameters(POSE_GRAPH_LUA);
            Self {
                pose_graph: None,
                pose_graph_options: create_pose_graph_options(&pose_graph_parameters),
                thread_pool: Arc::new(ThreadPool::new(1)),
            }
        }

        fn build_pose_graph(&mut self) {
            let optimization_problem = Box::new(OptimizationProblem3D::new(
                &self
                    .pose_graph_options
                    .optimization_problem_options
                    .unwrap_or_default(),
            ));
            self.pose_graph = Some(PoseGraph3D::new(
                &self.pose_graph_options,
                optimization_problem,
                Arc::clone(&self.thread_pool) as Arc<dyn ThreadPoolInterface>,
            ));
        }

        fn build_pose_graph_with_fake_optimization(&mut self) {
            let optimization_problem = Box::new(MockOptimizationProblem3D::new());
            self.pose_graph = Some(PoseGraph3D::with_optimization_problem(
                &self.pose_graph_options,
                optimization_problem,
                Arc::clone(&self.thread_pool) as Arc<dyn ThreadPoolInterface>,
            ));
        }

        fn pose_graph(&self) -> &PoseGraph3D {
            self.pose_graph.as_ref().expect("pose graph built")
        }
    }

    mod pose_graph_3d_test {
        use super::*;

        #[test]
        fn empty() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            pose_graph.wait_for_all_computations();
            assert!(pose_graph.get_all_submap_data().empty());
            assert!(pose_graph.constraints().is_empty());
            assert!(pose_graph.get_connected_trajectories().is_empty());
            assert!(pose_graph.get_all_submap_poses().empty());
            assert!(pose_graph.get_trajectory_nodes().empty());
            assert!(pose_graph.get_trajectory_node_poses().empty());
            assert!(pose_graph.get_trajectory_data().is_empty());
            let empty_proto = proto::PoseGraph::default();
            assert_eq!(
                pose_graph.to_proto(/*include_unfinished_submaps=*/ true),
                empty_proto
            );
        }

        #[test]
        fn basic_serialization() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let mut proto = proto::PoseGraph::default();
            let fake_node = create_fake_node_default();
            add_to_proto_graph(&fake_node, &mut proto);
            pose_graph.add_node_from_proto(&Rigid3d::identity(), &fake_node);
            let fake_submap = create_fake_submap_3d_default();
            add_to_proto_graph(&fake_submap, &mut proto);
            pose_graph.add_submap_from_proto(&Rigid3d::identity(), &fake_submap);
            add_to_proto_graph(
                &create_fake_constraint(&fake_node, &fake_submap),
                &mut proto,
            );
            pose_graph.add_serialized_constraints(&from_proto(&proto.constraint));
            add_to_proto_graph(
                &create_fake_landmark("landmark_id", &Rigid3d::identity()),
                &mut proto,
            );
            pose_graph.set_landmark_pose("landmark_id", &Rigid3d::identity(), false);
            pose_graph.wait_for_all_computations();
            let actual_proto = pose_graph.to_proto(/*include_unfinished_submaps=*/ true);
            assert_eq!(proto.constraint[0], actual_proto.constraint[0]);
            assert_eq!(
                proto.trajectory[0].node[0],
                actual_proto.trajectory[0].node[0]
            );
            assert_eq!(
                proto.trajectory[0].submap[0],
                actual_proto.trajectory[0].submap[0]
            );
            assert_eq!(proto.trajectory[0], actual_proto.trajectory[0]);
            assert_eq!(proto.landmark_poses[0], actual_proto.landmark_poses[0]);
            assert_eq!(proto, actual_proto);
        }

        #[test]
        fn serialization_with_unfinished_submaps() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let mut proto = proto::PoseGraph::default();

            // Create three nodes.
            let fake_node_1 = create_fake_node(1, 1);
            add_to_proto_graph(&fake_node_1, &mut proto);
            pose_graph.add_node_from_proto(&Rigid3d::identity(), &fake_node_1);
            let fake_node_2 = create_fake_node(1, 2);
            add_to_proto_graph(&fake_node_2, &mut proto);
            pose_graph.add_node_from_proto(&Rigid3d::identity(), &fake_node_2);
            let fake_node_3 = create_fake_node(1, 3);
            add_to_proto_graph(&fake_node_3, &mut proto);
            pose_graph.add_node_from_proto(&Rigid3d::identity(), &fake_node_3);

            // Create two submaps: one finished, the other not.
            let fake_submap_1 = create_fake_submap_3d(1, 1, true);
            add_to_proto_graph(&fake_submap_1, &mut proto);
            pose_graph.add_submap_from_proto(&Rigid3d::identity(), &fake_submap_1);
            let fake_submap_2 = create_fake_submap_3d(1, 2, false);
            add_to_proto_graph(&fake_submap_2, &mut proto);
            pose_graph.add_submap_from_proto(&Rigid3d::identity(), &fake_submap_2);

            // Connect node 1 to submap 1, node 2 to submaps 1 and 2, node 3
            // to submap 2.
            add_to_proto_graph(
                &create_fake_constraint(&fake_node_1, &fake_submap_1),
                &mut proto,
            );
            add_to_proto_graph(
                &create_fake_constraint(&fake_node_2, &fake_submap_1),
                &mut proto,
            );
            add_to_proto_graph(
                &create_fake_constraint(&fake_node_2, &fake_submap_2),
                &mut proto,
            );
            add_to_proto_graph(
                &create_fake_constraint(&fake_node_3, &fake_submap_2),
                &mut proto,
            );
            pose_graph.add_serialized_constraints(&from_proto(&proto.constraint));

            pose_graph.wait_for_all_computations();
            let actual_proto = pose_graph.to_proto(/*include_unfinished_submaps=*/ false);
            assert_eq!(actual_proto.constraint.len(), 2);
            assert_eq!(actual_proto.trajectory.len(), 1);
            assert_eq!(actual_proto.trajectory[0].node.len(), 3);
            assert_eq!(actual_proto.trajectory[0].submap.len(), 1);
            assert_eq!(proto.constraint[0], actual_proto.constraint[0]);
            assert_eq!(proto.constraint[1], actual_proto.constraint[1]);
            assert_eq!(
                proto.trajectory[0].node[0],
                actual_proto.trajectory[0].node[0]
            );
            assert_eq!(
                proto.trajectory[0].node[1],
                actual_proto.trajectory[0].node[1]
            );
            assert_eq!(
                proto.trajectory[0].submap[0],
                actual_proto.trajectory[0].submap[0]
            );
        }

        /// Adds one INTRA_SUBMAP constraint between `node` and `submap`.
        fn add_intra_submap_constraint(
            pose_graph: &PoseGraph3D,
            node: &proto::Node,
            submap: &proto::Submap,
        ) {
            let mut proto = proto::PoseGraph::default();
            let mut constraint = create_fake_constraint(node, submap);
            // TODO(gaschler): Also remove inter constraints when all
            // references are gone.
            constraint.set_tag(proto::pose_graph::constraint::Tag::IntraSubmap);
            add_to_proto_graph(&constraint, &mut proto);
            pose_graph.add_serialized_constraints(&from_proto(&proto.constraint));
        }

        /// The expectations shared by the trimmer tests.
        fn expect_sizes(
            pose_graph: &PoseGraph3D,
            trajectory_id: i32,
            num_submaps: usize,
            num_nodes: usize,
            num_constraints: usize,
        ) {
            assert_eq!(
                pose_graph
                    .get_all_submap_poses()
                    .size_of_trajectory_or_zero(trajectory_id),
                num_submaps
            );
            assert_eq!(
                pose_graph
                    .get_all_submap_data()
                    .size_of_trajectory_or_zero(trajectory_id),
                num_submaps
            );
            assert_eq!(
                pose_graph
                    .get_trajectory_nodes()
                    .size_of_trajectory_or_zero(trajectory_id),
                num_nodes
            );
            assert_eq!(
                pose_graph
                    .get_trajectory_node_poses()
                    .size_of_trajectory_or_zero(trajectory_id),
                num_nodes
            );
            assert_eq!(pose_graph.constraints().len(), num_constraints);
        }

        #[test]
        fn pure_localization_trimmer() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph_with_fake_optimization();
            let pose_graph = test.pose_graph();
            const TRAJECTORY_ID: i32 = 2;
            const NUM_SUBMAPS_TO_CREATE: i32 = 5;
            const NUM_SUBMAPS_TO_KEEP: i32 = 3;
            const NUM_NODES_PER_SUBMAP: i32 = 2;
            for i in 0..NUM_SUBMAPS_TO_CREATE {
                let submap_index = if i < 3 { 42 + i } else { 100 + i };
                let submap = create_fake_submap_3d(TRAJECTORY_ID, submap_index, true);
                pose_graph.add_submap_from_proto(&Rigid3d::identity(), &submap);
                for j in 0..NUM_NODES_PER_SUBMAP {
                    let node_index = 7 + NUM_NODES_PER_SUBMAP * submap_index + j;
                    let node = create_fake_node(TRAJECTORY_ID, node_index);
                    pose_graph.add_node_from_proto(&Rigid3d::identity(), &node);
                    add_intra_submap_constraint(pose_graph, &node, &submap);
                }
            }
            pose_graph.add_trimmer(Box::new(PureLocalizationTrimmer::new(
                TRAJECTORY_ID,
                NUM_SUBMAPS_TO_KEEP,
            )));
            pose_graph.wait_for_all_computations();
            let n = (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_CREATE) as usize;
            expect_sizes(
                pose_graph,
                TRAJECTORY_ID,
                NUM_SUBMAPS_TO_CREATE as usize,
                n,
                n,
            );
            for _ in 0..2 {
                pose_graph.run_final_optimization();
                let n = (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_KEEP) as usize;
                expect_sizes(
                    pose_graph,
                    TRAJECTORY_ID,
                    NUM_SUBMAPS_TO_KEEP as usize,
                    n,
                    n,
                );
            }
        }

        struct EvenSubmapTrimmer {
            trajectory_id: i32,
        }

        impl PoseGraphTrimmer for EvenSubmapTrimmer {
            fn trim(&mut self, pose_graph: &mut dyn Trimmable) {
                let submap_ids = pose_graph.get_submap_ids(self.trajectory_id);
                for submap_id in &submap_ids {
                    if submap_id.submap_index % 2 == 0 {
                        pose_graph.trim_submap(submap_id);
                    }
                }
            }

            fn is_finished(&self) -> bool {
                false
            }
        }

        #[test]
        fn even_submap_trimmer() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph_with_fake_optimization();
            let pose_graph = test.pose_graph();
            const TRAJECTORY_ID: i32 = 2;
            const NUM_SUBMAPS_TO_KEEP: i32 = 10;
            const NUM_SUBMAPS_TO_CREATE: i32 = 2 * NUM_SUBMAPS_TO_KEEP;
            const NUM_NODES_PER_SUBMAP: i32 = 3;
            for i in 0..NUM_SUBMAPS_TO_CREATE {
                let submap_index = 42 + i;
                let submap = create_fake_submap_3d(TRAJECTORY_ID, submap_index, true);
                pose_graph.add_submap_from_proto(&Rigid3d::identity(), &submap);
                for j in 0..NUM_NODES_PER_SUBMAP {
                    let node_index = 7 + NUM_NODES_PER_SUBMAP * i + j;
                    let node = create_fake_node(TRAJECTORY_ID, node_index);
                    pose_graph.add_node_from_proto(&Rigid3d::identity(), &node);
                    add_intra_submap_constraint(pose_graph, &node, &submap);
                }
            }
            pose_graph.add_trimmer(Box::new(EvenSubmapTrimmer {
                trajectory_id: TRAJECTORY_ID,
            }));
            pose_graph.wait_for_all_computations();
            let n = (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_CREATE) as usize;
            expect_sizes(
                pose_graph,
                TRAJECTORY_ID,
                NUM_SUBMAPS_TO_CREATE as usize,
                n,
                n,
            );
            for _ in 0..2 {
                pose_graph.run_final_optimization();
                let n = (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_KEEP) as usize;
                expect_sizes(
                    pose_graph,
                    TRAJECTORY_ID,
                    NUM_SUBMAPS_TO_KEEP as usize,
                    n,
                    n,
                );
            }
        }

        #[test]
        fn even_submap_trimmer_on_frozen_trajectory() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph_with_fake_optimization();
            let pose_graph = test.pose_graph();
            const TRAJECTORY_ID: i32 = 2;
            const NUM_SUBMAPS_TO_KEEP: i32 = 10;
            const NUM_SUBMAPS_TO_CREATE: i32 = 2 * NUM_SUBMAPS_TO_KEEP;
            const NUM_NODES_PER_SUBMAP: i32 = 3;
            for i in 0..NUM_SUBMAPS_TO_CREATE {
                let submap_index = 42 + i;
                let submap = create_fake_submap_3d(TRAJECTORY_ID, submap_index, true);
                pose_graph.add_submap_from_proto(&Rigid3d::identity(), &submap);
                for j in 0..NUM_NODES_PER_SUBMAP {
                    let node_index = 7 + NUM_NODES_PER_SUBMAP * i + j;
                    let node = create_fake_node(TRAJECTORY_ID, node_index);
                    pose_graph.add_node_from_proto(&Rigid3d::identity(), &node);
                    // This step is normally done by a MapBuilder when loading
                    // frozen state.
                    pose_graph.add_node_to_submap(
                        &NodeId::new(TRAJECTORY_ID, node_index),
                        &SubmapId::new(TRAJECTORY_ID, submap_index),
                    );
                }
            }
            pose_graph.freeze_trajectory(TRAJECTORY_ID);
            pose_graph.add_trimmer(Box::new(EvenSubmapTrimmer {
                trajectory_id: TRAJECTORY_ID,
            }));
            pose_graph.wait_for_all_computations();
            expect_sizes(
                pose_graph,
                TRAJECTORY_ID,
                NUM_SUBMAPS_TO_CREATE as usize,
                (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_CREATE) as usize,
                0,
            );
            for _ in 0..2 {
                pose_graph.run_final_optimization();
                expect_sizes(
                    pose_graph,
                    TRAJECTORY_ID,
                    NUM_SUBMAPS_TO_KEEP as usize,
                    (NUM_NODES_PER_SUBMAP * NUM_SUBMAPS_TO_KEEP) as usize,
                    0,
                );
            }
        }
    }

    /// Additional Rust tests (no C++ counterpart).
    mod pose_graph_3d_additional_test {
        use super::*;
        use crate::common::time::{Time, from_seconds, from_universal};
        use crate::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
        use crate::mapping::internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
        use crate::mapping::trajectory_node::to_proto as node_to_proto;
        use crate::sensor::landmark_data::{LandmarkData, LandmarkObservation};
        use crate::sensor::rangefinder_point::TimedRangefinderPoint;
        use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
        use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
        use crate::transform::transform::{to_proto as rigid_to_proto, to_rigid3};
        use cartographer_sim::{
            DatasetBuilder, ImuConfig, Lidar3dConfig, SensorData, Trajectory, World3d,
        };
        use nalgebra::Vector3;

        /// Simulation time 0 in Cartographer time.
        const BASE_UNIVERSAL: i64 = 638_000_000_000_000_000;

        fn to_time(seconds: f64) -> Time {
            from_universal(BASE_UNIVERSAL) + from_seconds(seconds)
        }

        /// Runs local SLAM (`trajectory_builder_3d.lua` with 20 range data
        /// per submap, so that submaps finish quickly) on `seconds` of a
        /// simulated walk through `World3d::room` and feeds nodes and IMU
        /// data of `trajectory_id` to `pose_graph`. Returns the number of
        /// nodes.
        fn add_simulated_trajectory(
            pose_graph: &PoseGraph3D,
            trajectory_id: i32,
            seconds: f64,
        ) -> usize {
            let options = create_local_trajectory_builder_options_3d(&resolve_lua_parameters(
                r#"
                include "trajectory_builder_3d.lua"
                TRAJECTORY_BUILDER_3D.submaps.num_range_data = 20
                return TRAJECTORY_BUILDER_3D"#,
            ));
            let dataset = DatasetBuilder::new(
                Trajectory::room_wander_3d(1.0, 0.8).with_start_from_rest(1.0),
                World3d::room(),
            )
            .lidar_3d("points", Lidar3dConfig::default())
            .imu("imu", ImuConfig::default())
            .seed(7)
            .time_range(0., seconds)
            .build();
            let mut builder = LocalTrajectoryBuilder3D::new(&options, &["points"]);
            let mut num_nodes = 0;
            for message in &dataset.messages {
                match &message.data {
                    SensorData::Imu(imu) => {
                        let imu_data = ImuData {
                            time: to_time(message.time),
                            linear_acceleration: imu.linear_acceleration,
                            angular_velocity: imu.angular_velocity,
                        };
                        builder.add_imu_data(&imu_data);
                        pose_graph.add_imu_data(trajectory_id, &imu_data);
                    }
                    SensorData::PointCloud(cloud) => {
                        let data = TimedPointCloudData {
                            time: to_time(message.time),
                            origin: cloud.origin,
                            ranges: cloud
                                .points
                                .iter()
                                .map(|p| TimedRangefinderPoint {
                                    position: p.position,
                                    time: p.time,
                                })
                                .collect(),
                            intensities: cloud.intensities.clone(),
                        };
                        if let Some(result) = builder.add_range_data("points", &data)
                            && let Some(insertion_result) = result.insertion_result
                        {
                            let node_id = pose_graph.add_node(
                                insertion_result.constant_data,
                                trajectory_id,
                                &insertion_result.insertion_submaps,
                            );
                            assert_eq!(node_id, NodeId::new(trajectory_id, num_nodes as i32));
                            num_nodes += 1;
                        }
                    }
                    _ => {}
                }
            }
            num_nodes
        }

        #[test]
        fn trajectory_state_transitions() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let num_nodes = add_simulated_trajectory(pose_graph, 0, 8.);
            pose_graph.wait_for_all_computations();
            assert_eq!(
                pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Active)].into()
            );
            assert!(!pose_graph.is_trajectory_finished(0));
            assert!(!pose_graph.is_trajectory_frozen(0));
            assert_eq!(pose_graph.get_trajectory_nodes().size(), num_nodes);
            assert!(pose_graph.get_all_submap_data().size() >= 3);

            pose_graph.finish_trajectory(0);
            pose_graph.run_final_optimization();
            assert!(pose_graph.is_trajectory_finished(0));
            assert_eq!(
                pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Finished)].into()
            );
            assert_eq!(pose_graph.get_trajectory_nodes().size(), num_nodes);
            // The optimization has seen the IMU data of the trajectory.
            assert!(pose_graph.get_trajectory_data().contains_key(&0));

            // Deleting an unknown trajectory is a no-op.
            pose_graph.delete_trajectory(42);
            pose_graph.delete_trajectory(0);
            // The trajectory is deleted by the next optimization.
            pose_graph.run_final_optimization();
            assert_eq!(
                pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Deleted)].into()
            );
            assert!(pose_graph.get_all_submap_data().empty());
            assert!(pose_graph.get_all_submap_poses().empty());
            assert!(pose_graph.get_trajectory_nodes().empty());
            assert!(pose_graph.constraints().is_empty());

            // Another trajectory can be added and frozen.
            let num_nodes = add_simulated_trajectory(pose_graph, 1, 3.);
            pose_graph.freeze_trajectory(1);
            pose_graph.run_final_optimization();
            assert!(pose_graph.is_trajectory_frozen(1));
            assert_eq!(
                pose_graph.get_trajectory_states(),
                [(0, TrajectoryState::Deleted), (1, TrajectoryState::Frozen)].into()
            );
            assert_eq!(
                pose_graph
                    .get_trajectory_nodes()
                    .size_of_trajectory_or_zero(1),
                num_nodes
            );
        }

        #[test]
        fn submaps_constraints_and_callback() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let callback_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
            {
                let callback_calls = Arc::clone(&callback_calls);
                pose_graph.set_global_slam_optimization_callback(Box::new(
                    move |submaps, nodes| {
                        callback_calls
                            .lock()
                            .unwrap()
                            .push((submaps.clone(), nodes.clone()));
                    },
                ));
            }
            let num_nodes = add_simulated_trajectory(pose_graph, 0, 8.);
            pose_graph.run_final_optimization();

            let submap_data = pose_graph.get_all_submap_data();
            let num_submaps = submap_data.size_of_trajectory_or_zero(0);
            assert!(num_submaps >= 3, "{num_submaps}");
            let submap_poses = pose_graph.get_all_submap_poses();
            assert_eq!(submap_poses.size(), num_submaps);
            for submap_id_data in &submap_data {
                let single = pose_graph.get_submap_data(&submap_id_data.id);
                assert_is_nearly(&single.pose, &submap_id_data.data.pose, 1e-12);
                let version = single.submap.unwrap().read().unwrap().num_range_data();
                assert_eq!(submap_poses.at(&submap_id_data.id).version, version);
            }
            assert!(
                pose_graph
                    .get_submap_data(&SubmapId::new(0, 100))
                    .submap
                    .is_none()
            );

            // Every node has intra-submap constraints to the submaps it was
            // inserted into; local constraint search found inter-submap
            // constraints. 3D constraints are stored and returned verbatim.
            let constraints = pose_graph.constraints();
            let num_intra = constraints
                .iter()
                .filter(|c| c.tag == crate::mapping::pose_graph_interface::Tag::IntraSubmap)
                .count();
            assert!(num_intra >= num_nodes, "{num_intra}");
            assert!(constraints.len() > num_intra, "{}", constraints.len());

            let node_poses = pose_graph.get_trajectory_node_poses();
            assert_eq!(node_poses.size(), num_nodes);
            assert!(
                node_poses
                    .iter()
                    .all(|node| node.data.constant_pose_data.is_some())
            );
            // The interpolated trajectory pose at a node's time is the node's
            // pose.
            let nodes = pose_graph.get_trajectory_nodes();
            let node = nodes.at(&NodeId::new(0, 10));
            assert_is_nearly(
                &pose_graph.get_interpolated_global_trajectory_pose(0, node.time()),
                &node.global_pose,
                1e-9,
            );

            let calls = callback_calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            let (last_submaps, last_nodes) = &calls[0];
            assert_eq!(last_nodes[&0], NodeId::new(0, num_nodes as i32 - 1));
            assert_eq!(last_submaps[&0], SubmapId::new(0, num_submaps as i32 - 1));
            assert_eq!(pose_graph.get_connected_trajectories(), vec![vec![0]]);
        }

        /// Serializes a pose graph like `MapBuilder::SerializeState` and
        /// loads it into a fresh pose graph like `MapBuilder::LoadState`.
        #[test]
        fn to_proto_round_trip() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let num_nodes = add_simulated_trajectory(pose_graph, 0, 6.);
            pose_graph.run_final_optimization();
            let original = pose_graph.to_proto(true);
            let submap_data = pose_graph.get_all_submap_data();
            assert_eq!(original.trajectory.len(), 1);
            assert_eq!(original.trajectory[0].trajectory_id, 0);
            assert_eq!(original.trajectory[0].node.len(), num_nodes);
            assert_eq!(original.trajectory[0].submap.len(), submap_data.size());
            assert_eq!(original.constraint.len(), pose_graph.constraints().len());

            // Unfinished submaps and their constraints are left out on
            // request.
            let finished_only = pose_graph.to_proto(false);
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
            let mut loaded = PoseGraph3DTest::new();
            loaded.build_pose_graph();
            let loaded_pose_graph = loaded.pose_graph();
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
                assert!(submap_proto.submap_3d.is_some());
                loaded_pose_graph.add_submap_from_proto(&submap_id_data.data.pose, &submap_proto);
            }
            for node_id_data in &pose_graph.get_trajectory_nodes() {
                let node_proto = proto::Node {
                    node_id: Some(node_id_data.id.to_proto()),
                    node_data: Some(node_to_proto(
                        node_id_data.data.constant_data.as_ref().unwrap(),
                    )),
                };
                loaded_pose_graph.add_node_from_proto(&node_id_data.data.global_pose, &node_proto);
            }
            for (trajectory_id, trajectory_data) in pose_graph.get_trajectory_data() {
                let [w, x, y, z] = trajectory_data.imu_calibration;
                loaded_pose_graph.set_trajectory_data_from_proto(&proto::TrajectoryData {
                    trajectory_id,
                    gravity_constant: trajectory_data.gravity_constant,
                    imu_calibration: Some(crate::transform::proto::Quaterniond { x, y, z, w }),
                    fixed_frame_origin_in_map: trajectory_data
                        .fixed_frame_origin_in_map
                        .as_ref()
                        .map(rigid_to_proto),
                });
            }
            let imu_data = pose_graph.get_imu_data();
            for imu in imu_data.trajectory(0) {
                loaded_pose_graph.add_imu_data(0, imu);
            }
            loaded_pose_graph.add_serialized_constraints(&from_proto(&original.constraint));
            loaded_pose_graph.wait_for_all_computations();

            let reloaded = loaded_pose_graph.to_proto(true);
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
            assert_eq!(
                loaded_pose_graph.get_trajectory_data(),
                pose_graph.get_trajectory_data()
            );
            assert_eq!(
                loaded_pose_graph.get_imu_data().trajectory(0).count(),
                imu_data.trajectory(0).count()
            );
            // The loaded graph can be optimized.
            loaded_pose_graph.run_final_optimization();
            assert_eq!(loaded_pose_graph.get_trajectory_nodes().size(), num_nodes);
            assert_eq!(
                loaded_pose_graph.constraints().len(),
                original.constraint.len()
            );
        }

        #[test]
        fn landmarks_and_sensor_data() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            let pose = Rigid3d::from_translation(Vector3::new(1., 2., 3.));
            pose_graph.set_landmark_pose("landmark", &pose, true);
            pose_graph.add_landmark_data(
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
            pose_graph.add_odometry_data(
                0,
                &OdometryData {
                    time: from_universal(5),
                    pose: Rigid3d::identity(),
                },
            );
            pose_graph.add_fixed_frame_pose_data(
                0,
                &FixedFramePoseData {
                    time: from_universal(5),
                    pose: Some(Rigid3d::identity()),
                },
            );
            pose_graph.add_imu_data(
                0,
                &ImuData {
                    time: from_universal(5),
                    linear_acceleration: Vector3::new(0., 0., 9.8),
                    angular_velocity: Vector3::zeros(),
                },
            );
            // 3D always forwards gravity constant and IMU calibration.
            pose_graph.set_trajectory_data_from_proto(&proto::TrajectoryData {
                trajectory_id: 3,
                gravity_constant: 9.81,
                imu_calibration: Some(crate::transform::proto::Quaterniond {
                    x: 0.,
                    y: 0.,
                    z: 1.,
                    w: 0.,
                }),
                fixed_frame_origin_in_map: None,
            });
            pose_graph.set_trajectory_data_from_proto(&proto::TrajectoryData {
                trajectory_id: 4,
                gravity_constant: 9.7,
                imu_calibration: None,
                fixed_frame_origin_in_map: Some(rigid_to_proto(&pose)),
            });
            pose_graph.wait_for_all_computations();
            assert_eq!(
                pose_graph.get_landmark_poses(),
                [("landmark".to_string(), pose)].into()
            );
            let landmark_nodes = pose_graph.get_landmark_nodes();
            let landmark = &landmark_nodes["landmark"];
            assert!(landmark.frozen);
            assert_eq!(landmark.landmark_observations.len(), 1);
            assert_eq!(landmark.landmark_observations[0].trajectory_id, 0);
            assert_eq!(landmark.landmark_observations[0].rotation_weight, 2.);
            assert_eq!(pose_graph.get_odometry_data().trajectory(0).count(), 1);
            assert_eq!(
                pose_graph.get_fixed_frame_pose_data().trajectory(0).count(),
                1
            );
            assert_eq!(pose_graph.get_imu_data().trajectory(0).count(), 1);
            let trajectory_data = pose_graph.get_trajectory_data();
            assert_eq!(trajectory_data.len(), 2);
            assert_eq!(trajectory_data[&3].gravity_constant, 9.81);
            assert_eq!(trajectory_data[&3].imu_calibration, [0., 0., 0., 1.]);
            assert!(trajectory_data[&3].fixed_frame_origin_in_map.is_none());
            assert_eq!(trajectory_data[&4].gravity_constant, 9.7);
            // A missing proto quaternion is all zeros (C++ default instance).
            assert_eq!(trajectory_data[&4].imu_calibration, [0., 0., 0., 0.]);
            assert_is_nearly(
                &trajectory_data[&4].fixed_frame_origin_in_map.unwrap(),
                &pose,
                1e-12,
            );
        }

        #[test]
        fn initial_trajectory_pose() {
            let mut test = PoseGraph3DTest::new();
            test.build_pose_graph();
            let pose_graph = test.pose_graph();
            add_simulated_trajectory(pose_graph, 0, 3.);
            pose_graph.run_final_optimization();
            let relative_pose = Rigid3d::from_translation(Vector3::new(1., 2., 0.));
            let nodes = pose_graph.get_trajectory_nodes();
            let node = nodes.at(&NodeId::new(0, 5));
            pose_graph.set_initial_trajectory_pose(1, 0, &relative_pose, node.time());
            let expected = node.global_pose * relative_pose;
            assert_is_nearly(
                &pose_graph.get_local_to_global_transform(1),
                &expected,
                1e-9,
            );
            // The initial pose connects the trajectories once trajectory 1
            // has nodes.
            add_simulated_trajectory(pose_graph, 1, 2.);
            pose_graph.wait_for_all_computations();
            assert_eq!(pose_graph.get_connected_trajectories(), vec![vec![0, 1]]);
            let first_node = pose_graph
                .get_trajectory_nodes()
                .at(&NodeId::new(1, 0))
                .clone();
            assert_is_nearly(
                &first_node.global_pose,
                &(expected * first_node.constant_data.unwrap().local_pose),
                1e-9,
            );
        }
    }
}
