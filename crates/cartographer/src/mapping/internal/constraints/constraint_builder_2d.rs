//! Port of `cartographer/mapping/internal/constraints/constraint_builder_2d` (.h, .cc).
//!
//! RUST PORT NOTE: threading and ownership model.
//!
//! * The C++ class is used through `this` from thread-pool tasks. Here the
//!   state touched by tasks lives in an `Arc<Inner>` (options, Ceres matcher,
//!   the `mutex_`-guarded `State` and the per-submap samplers) which the
//!   tasks capture. The thread pool (`Arc<dyn ThreadPoolInterface>`) is kept
//!   in the outer [`ConstraintBuilder2D`] only: tasks never schedule further
//!   work, and not capturing the pool avoids a task dropping the last handle
//!   of the pool on one of its own worker threads.
//! * The task graph is identical to C++: per submap one scan matcher
//!   construction task; per `maybe_add_*` call one constraint task depending
//!   on it; the current "finish node" task depends on all constraint tasks of
//!   the node and is scheduled by `notify_end_of_node`; the "when done" task
//!   depends on all finish node tasks and is scheduled by `when_done`.
//! * C++ passes raw `const Submap2D*` / `const TrajectoryNode::Data*`
//!   pointers which must stay valid until all computations are finished.
//!   Here the tasks own `SharedSubmap<Submap2D>` (`Arc<RwLock<Submap2D>>`) and
//!   `Arc<TrajectoryNodeData>` handles, so trimming a submap while a search is
//!   in flight is memory safe.
//! * The submap's read lock is held only while needed: to read the local pose,
//!   while building the `FastCorrelativeScanMatcher2D` (which keeps only its
//!   precomputed grids, not a reference to the submap grid) and during the
//!   Ceres refinement against the submap grid (C++ `*submap_scan_matcher.grid`).
//!   The branch-and-bound search runs without any lock.
//! * C++ `std::deque<std::unique_ptr<Constraint>>` (pointer-stable slots,
//!   `nullptr` = no match) becomes `Vec<Arc<OnceLock<Constraint>>>`: each
//!   constraint task owns its slot and fills it at most once. Results are
//!   collected in insertion order, exactly like C++.
//! * The scan matcher map stores `Arc<SubmapScanMatcher>`; the matcher itself
//!   is a `OnceLock` filled by the construction task (C++ assigns the
//!   `unique_ptr` from the task). `delete_scan_matcher` drops the map entry;
//!   in-flight tasks keep their own `Arc`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, Weak};

use ceres::SolverSummary;

use crate::common::fixed_ratio_sampler::FixedRatioSampler;
use crate::common::histogram::Histogram as ScoreHistogram;
use crate::common::task::Task;
use crate::common::thread_pool::ThreadPoolInterface;
use crate::mapping::constraints::proto;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::internal::two_d::scan_matching::ceres_scan_matcher_2d::CeresScanMatcher2D;
use crate::mapping::internal::two_d::scan_matching::fast_correlative_scan_matcher_2d::FastCorrelativeScanMatcher2D;
use crate::mapping::pose_graph_interface::{Constraint, Pose, Tag};
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::metrics::counter::Counter;
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::metrics::gauge::Gauge;
use crate::metrics::histogram::Histogram;
use crate::transform::rigid_transform::Rigid2d;
use crate::transform::transform::{embed_3d, project_2d};

/// The result handed to the [`ConstraintBuilder2D::when_done`] callback.
pub type Result = Vec<Constraint>;

/// Callback registered with [`ConstraintBuilder2D::when_done`]; executed in
/// the thread pool.
pub type WhenDoneCallback = Box<dyn FnOnce(&Result) + Send + 'static>;

/// The C++ file-level metric statics.
#[derive(Clone, Copy)]
struct Metrics {
    constraints_searched: &'static dyn Counter,
    constraints_found: &'static dyn Counter,
    global_constraints_searched: &'static dyn Counter,
    global_constraints_found: &'static dyn Counter,
    queue_length: &'static dyn Gauge,
    constraint_scores: &'static dyn Histogram,
    global_constraint_scores: &'static dyn Histogram,
    num_submap_scan_matchers: &'static dyn Gauge,
}

// RUST PORT NOTE: the C++ `static auto* k...Metric` globals; `None` means the
// null metrics.
static METRICS: std::sync::RwLock<Option<Metrics>> = std::sync::RwLock::new(None);

fn metrics() -> Metrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(|| Metrics {
            constraints_searched: <dyn Counter>::null(),
            constraints_found: <dyn Counter>::null(),
            global_constraints_searched: <dyn Counter>::null(),
            global_constraints_found: <dyn Counter>::null(),
            queue_length: <dyn Gauge>::null(),
            constraint_scores: <dyn Histogram>::null(),
            global_constraint_scores: <dyn Histogram>::null(),
            num_submap_scan_matchers: <dyn Gauge>::null(),
        })
}

/// Returns (map <- submap) where 'submap' is a coordinate system at the
/// origin of the Submap.
pub fn compute_submap_pose(submap: &Submap2D) -> Rigid2d {
    project_2d(&submap.local_pose())
}

fn read_submap(submap: &RwLock<Submap2D>) -> RwLockReadGuard<'_, Submap2D> {
    submap.read().unwrap_or_else(PoisonError::into_inner)
}

struct SubmapScanMatcher {
    // RUST PORT NOTE: C++ `const Grid2D* grid`; the grid is read through the
    // submap handle (under its read lock) for the Ceres refinement.
    submap: SharedSubmap<Submap2D>,
    // Set by the construction task; must only be accessed after
    // `creation_task_handle` has completed.
    fast_correlative_scan_matcher: OnceLock<FastCorrelativeScanMatcher2D>,
    creation_task_handle: Weak<Task>,
}

// The C++ members guarded by `mutex_`.
struct State {
    // 'callback' set by when_done().
    when_done: Option<WhenDoneCallback>,

    // TODO(gaschler): Use atomics instead of mutex to access these counters.
    // Number of the node in reaction to which computations are currently
    // added. This is always the number of nodes seen so far, even when older
    // nodes are matched against a new submap.
    num_started_nodes: i32,

    num_finished_nodes: i32,

    finish_node_task: Arc<Task>,

    when_done_task: Arc<Task>,

    // Constraints currently being computed in the background. Constraint
    // search results with below-threshold scores stay empty.
    constraints: Vec<Arc<OnceLock<Constraint>>>,

    // Map of dispatched or constructed scan matchers by 'submap_id'.
    submap_scan_matchers: BTreeMap<SubmapId, Arc<SubmapScanMatcher>>,

    // Histogram of scan matcher scores.
    score_histogram: ScoreHistogram,
}

// The part of the builder shared with thread-pool tasks (C++ `this`).
struct Inner {
    options: proto::ConstraintBuilderOptions,
    mutex: Mutex<State>,
    // RUST PORT NOTE: not guarded by `mutex_` in C++ (only touched from the
    // single pose graph worker and `DeleteScanMatcher`); a separate mutex here.
    per_submap_sampler: Mutex<BTreeMap<SubmapId, FixedRatioSampler>>,
    ceres_scan_matcher: CeresScanMatcher2D,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // Runs in a background thread and does computations for an additional
    // constraint, assuming 'submap' and 'constant_data' do not change
    // anymore. As output, it may create a new Constraint in 'constraint'.
    #[allow(clippy::too_many_arguments)]
    fn compute_constraint(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap2D>,
        node_id: &NodeId,
        match_full_submap: bool,
        constant_data: &TrajectoryNodeData,
        initial_relative_pose: &Rigid2d,
        submap_scan_matcher: &SubmapScanMatcher,
        constraint: &OnceLock<Constraint>,
    ) {
        let fast_correlative_scan_matcher = submap_scan_matcher
            .fast_correlative_scan_matcher
            .get()
            .expect("Check failed: submap_scan_matcher.fast_correlative_scan_matcher");
        // The local pose of a submap never changes.
        let submap_pose = compute_submap_pose(&read_submap(submap));
        let initial_pose = submap_pose * *initial_relative_pose;

        // The 'constraint_transform' (submap i <- node j) is computed from:
        // - a 'filtered_gravity_aligned_point_cloud' in node j,
        // - the initial guess 'initial_pose' for (map <- node j),
        // - the result 'pose_estimate' of Match() (map <- node j).
        // - the ComputeSubmapPose() (map <- submap i)
        let metrics = metrics();
        let score: f32;
        let mut pose_estimate: Rigid2d;

        // Compute 'pose_estimate' in three stages:
        // 1. Fast estimate using the fast correlative scan matcher.
        // 2. Prune if the score is too low.
        // 3. Refine.
        if match_full_submap {
            metrics.global_constraints_searched.increment();
            match fast_correlative_scan_matcher.match_full_submap(
                &constant_data.filtered_gravity_aligned_point_cloud,
                self.options.global_localization_min_score as f32,
            ) {
                Some((match_score, match_pose)) => {
                    score = match_score;
                    pose_estimate = match_pose;
                    assert!(
                        f64::from(score) > self.options.global_localization_min_score,
                        "Check failed: score > options_.global_localization_min_score()"
                    );
                    assert!(node_id.trajectory_id >= 0);
                    assert!(submap_id.trajectory_id >= 0);
                    metrics.global_constraints_found.increment();
                    metrics.global_constraint_scores.observe(f64::from(score));
                }
                None => return,
            }
        } else {
            metrics.constraints_searched.increment();
            match fast_correlative_scan_matcher.match_(
                &initial_pose,
                &constant_data.filtered_gravity_aligned_point_cloud,
                self.options.min_score as f32,
            ) {
                Some((match_score, match_pose)) => {
                    // We've reported a successful local match.
                    score = match_score;
                    pose_estimate = match_pose;
                    assert!(
                        f64::from(score) > self.options.min_score,
                        "Check failed: score > options_.min_score()"
                    );
                    metrics.constraints_found.increment();
                    metrics.constraint_scores.observe(f64::from(score));
                }
                None => return,
            }
        }
        self.lock().score_histogram.add(score);

        // Use the CSM estimate as both the initial and previous pose. This has
        // the effect that, in the absence of better information, we prefer the
        // original CSM estimate.
        let mut unused_summary = SolverSummary::default();
        {
            let grid_submap = read_submap(&submap_scan_matcher.submap);
            pose_estimate = self.ceres_scan_matcher.match_(
                pose_estimate.translation(),
                &pose_estimate,
                &constant_data.filtered_gravity_aligned_point_cloud,
                grid_submap.grid(),
                &mut unused_summary,
            );
        }

        let constraint_transform = submap_pose.inverse() * pose_estimate;
        let result = Constraint {
            submap_id: *submap_id,
            node_id: *node_id,
            pose: Pose {
                zbar_ij: embed_3d(&constraint_transform),
                translation_weight: self.options.loop_closure_translation_weight,
                rotation_weight: self.options.loop_closure_rotation_weight,
            },
            tag: Tag::InterSubmap,
        };
        assert!(
            constraint.set(result).is_ok(),
            "constraint slot filled twice"
        );

        if self.options.log_matches {
            let mut info = format!(
                "Node {} with {} points on submap {}",
                node_id,
                constant_data.filtered_gravity_aligned_point_cloud.size(),
                submap_id
            );
            if match_full_submap {
                info.push_str(" matches");
            } else {
                let difference = initial_pose.inverse() * pose_estimate;
                info.push_str(&format!(
                    " differs by translation {:.2} rotation {:.3}",
                    difference.translation().norm(),
                    difference.normalized_angle().abs()
                ));
            }
            info.push_str(&format!(" with score {:.1}%.", 100. * f64::from(score)));
            log::info!("{info}");
        }
    }

    fn run_when_done_callback(&self) {
        let mut result = Result::new();
        let callback;
        {
            let mut state = self.lock();
            assert!(
                state.when_done.is_some(),
                "Check failed: when_done_ != nullptr"
            );
            result.extend(state.constraints.iter().filter_map(|c| c.get().copied()));
            if self.options.log_matches {
                log::info!(
                    "{} computations resulted in {} additional constraints.",
                    state.constraints.len(),
                    result.len()
                );
                log::info!("Score histogram:\n{}", state.score_histogram.to_string(10));
            }
            state.constraints.clear();
            callback = state.when_done.take();
            metrics().queue_length.set(state.constraints.len() as f64);
        }
        if let Some(callback) = callback {
            callback(&result);
        }
    }
}

/// Asynchronously computes constraints.
///
/// Intermingle an arbitrary number of calls to `maybe_add_constraint`,
/// `maybe_add_global_constraint`, and `notify_end_of_node`, then call
/// `when_done` once. After all computations are done the `callback` will be
/// called with the result and another
/// `maybe_add(_global)_constraint()`/`when_done()` cycle can follow.
///
/// This class is thread-safe.
pub struct ConstraintBuilder2D {
    thread_pool: Arc<dyn ThreadPoolInterface>,
    inner: Arc<Inner>,
}

impl std::fmt::Debug for ConstraintBuilder2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintBuilder2D")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl ConstraintBuilder2D {
    /// Creates a builder scheduling its work on `thread_pool`.
    pub fn new(
        options: &proto::ConstraintBuilderOptions,
        thread_pool: Arc<dyn ThreadPoolInterface>,
    ) -> Self {
        Self {
            thread_pool,
            inner: Arc::new(Inner {
                options: options.clone(),
                mutex: Mutex::new(State {
                    when_done: None,
                    num_started_nodes: 0,
                    num_finished_nodes: 0,
                    finish_node_task: Task::new(),
                    when_done_task: Task::new(),
                    constraints: Vec::new(),
                    submap_scan_matchers: BTreeMap::new(),
                    score_histogram: ScoreHistogram::new(),
                }),
                per_submap_sampler: Mutex::new(BTreeMap::new()),
                ceres_scan_matcher: CeresScanMatcher2D::new(
                    &options.ceres_scan_matcher_options.unwrap_or_default(),
                ),
            }),
        }
    }

    /// Schedules exploring a new constraint between `submap` identified by
    /// `submap_id`, and the `constant_data` for `node_id`. The
    /// `initial_relative_pose` is relative to the `submap`.
    pub fn maybe_add_constraint(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap2D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
        initial_relative_pose: &Rigid2d,
    ) {
        if initial_relative_pose.translation().norm() > self.inner.options.max_constraint_distance {
            return;
        }
        if !self
            .inner
            .per_submap_sampler
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(*submap_id)
            .or_insert_with(|| FixedRatioSampler::new(self.inner.options.sampling_ratio))
            .pulse()
        {
            return;
        }

        self.add_constraint_task(
            submap_id,
            submap,
            node_id,
            constant_data,
            false, /* match_full_submap */
            *initial_relative_pose,
            "MaybeAddConstraint",
        );
    }

    /// Schedules exploring a new constraint between `submap` identified by
    /// `submap_id` and the `constant_data` for `node_id`. This performs
    /// full-submap matching.
    pub fn maybe_add_global_constraint(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap2D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
    ) {
        self.add_constraint_task(
            submap_id,
            submap,
            node_id,
            constant_data,
            true, /* match_full_submap */
            Rigid2d::identity(),
            "MaybeAddGlobalConstraint",
        );
    }

    // The common locked part of MaybeAddConstraint and
    // MaybeAddGlobalConstraint.
    #[allow(clippy::too_many_arguments)]
    fn add_constraint_task(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap2D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
        match_full_submap: bool,
        initial_relative_pose: Rigid2d,
        caller: &str,
    ) {
        let mut state = self.inner.lock();
        if state.when_done.is_some() {
            log::warn!("{caller} was called while WhenDone was scheduled.");
        }
        let constraint = Arc::new(OnceLock::new());
        state.constraints.push(Arc::clone(&constraint));
        metrics().queue_length.set(state.constraints.len() as f64);
        let scan_matcher = self.dispatch_scan_matcher_construction(&mut state, submap_id, submap);
        let constraint_task = Task::new();
        {
            let inner = Arc::clone(&self.inner);
            let submap_id = *submap_id;
            let submap = Arc::clone(submap);
            let node_id = *node_id;
            let constant_data = Arc::clone(constant_data);
            let scan_matcher = Arc::clone(&scan_matcher);
            constraint_task.set_work_item(move || {
                inner.compute_constraint(
                    &submap_id,
                    &submap,
                    &node_id,
                    match_full_submap,
                    &constant_data,
                    &initial_relative_pose,
                    &scan_matcher,
                    &constraint,
                );
            });
        }
        constraint_task.add_dependency(scan_matcher.creation_task_handle.clone());
        let constraint_task_handle = self.thread_pool.schedule(constraint_task);
        state
            .finish_node_task
            .add_dependency(constraint_task_handle);
    }

    /// Must be called after all computations related to one node have been
    /// added.
    pub fn notify_end_of_node(&self) {
        let mut state = self.inner.lock();
        {
            let inner = Arc::clone(&self.inner);
            state.finish_node_task.set_work_item(move || {
                inner.lock().num_finished_nodes += 1;
            });
        }
        let finish_node_task = std::mem::replace(&mut state.finish_node_task, Task::new());
        let finish_node_task_handle = self.thread_pool.schedule(finish_node_task);
        state.when_done_task.add_dependency(finish_node_task_handle);
        state.num_started_nodes += 1;
    }

    /// Registers the `callback` to be called with the results, after all
    /// computations triggered by `maybe_add_*_constraint` have finished.
    /// `callback` is executed in the thread pool.
    pub fn when_done(&self, callback: impl FnOnce(&Result) + Send + 'static) {
        let mut state = self.inner.lock();
        assert!(
            state.when_done.is_none(),
            "Check failed: when_done_ == nullptr"
        );
        state.when_done = Some(Box::new(callback));
        {
            let inner = Arc::clone(&self.inner);
            state
                .when_done_task
                .set_work_item(move || inner.run_when_done_callback());
        }
        let when_done_task = std::mem::replace(&mut state.when_done_task, Task::new());
        self.thread_pool.schedule(when_done_task);
    }

    /// Returns the number of consecutive finished nodes.
    pub fn get_num_finished_nodes(&self) -> i32 {
        self.inner.lock().num_finished_nodes
    }

    /// Delete data related to `submap_id`.
    pub fn delete_scan_matcher(&self, submap_id: &SubmapId) {
        let mut state = self.inner.lock();
        if state.when_done.is_some() {
            log::warn!("DeleteScanMatcher was called while WhenDone was scheduled.");
        }
        state.submap_scan_matchers.remove(submap_id);
        self.inner
            .per_submap_sampler
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(submap_id);
        metrics()
            .num_submap_scan_matchers
            .set(state.submap_scan_matchers.len() as f64);
    }

    // The returned 'fast_correlative_scan_matcher' must only be accessed
    // after 'creation_task_handle' has completed. Requires the state lock.
    fn dispatch_scan_matcher_construction(
        &self,
        state: &mut State,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap2D>,
    ) -> Arc<SubmapScanMatcher> {
        if let Some(submap_scan_matcher) = state.submap_scan_matchers.get(submap_id) {
            return Arc::clone(submap_scan_matcher);
        }
        let scan_matcher_task = Task::new();
        // RUST PORT NOTE: C++ stores the (default-constructed) map entry
        // first and lets the task write into it; here the entry is created
        // with the task handle before it is shared with the task.
        let submap_scan_matcher = Arc::new(SubmapScanMatcher {
            submap: Arc::clone(submap),
            fast_correlative_scan_matcher: OnceLock::new(),
            creation_task_handle: Arc::downgrade(&scan_matcher_task),
        });
        state
            .submap_scan_matchers
            .insert(*submap_id, Arc::clone(&submap_scan_matcher));
        metrics()
            .num_submap_scan_matchers
            .set(state.submap_scan_matchers.len() as f64);
        let scan_matcher_options = self
            .inner
            .options
            .fast_correlative_scan_matcher_options
            .unwrap_or_default();
        {
            let submap_scan_matcher = Arc::clone(&submap_scan_matcher);
            scan_matcher_task.set_work_item(move || {
                let submap = read_submap(&submap_scan_matcher.submap);
                let fast_correlative_scan_matcher =
                    FastCorrelativeScanMatcher2D::new(submap.grid(), &scan_matcher_options);
                drop(submap);
                assert!(
                    submap_scan_matcher
                        .fast_correlative_scan_matcher
                        .set(fast_correlative_scan_matcher)
                        .is_ok(),
                    "scan matcher constructed twice"
                );
            });
        }
        self.thread_pool.schedule(scan_matcher_task);
        submap_scan_matcher
    }

    /// Registers the metrics with `family_factory`.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        let labels = |pairs: &[(&str, &str)]| -> Labels {
            pairs
                .iter()
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let counts = family_factory.new_counter_family(
            "mapping_constraints_constraint_builder_2d_constraints",
            "Constraints computed",
        );
        let constraints_searched = counts.add(&labels(&[
            ("search_region", "local"),
            ("matcher", "searched"),
        ]));
        let constraints_found =
            counts.add(&labels(&[("search_region", "local"), ("matcher", "found")]));
        let global_constraints_searched = counts.add(&labels(&[
            ("search_region", "global"),
            ("matcher", "searched"),
        ]));
        let global_constraints_found = counts.add(&labels(&[
            ("search_region", "global"),
            ("matcher", "found"),
        ]));
        let queue_length_family = family_factory.new_gauge_family(
            "mapping_constraints_constraint_builder_2d_queue_length",
            "Queue length",
        );
        let queue_length = queue_length_family.add(&Labels::new());
        let boundaries = <dyn Histogram>::fixed_width(0.05, 20);
        let scores = family_factory.new_histogram_family(
            "mapping_constraints_constraint_builder_2d_scores",
            "Constraint scores built",
            &boundaries,
        );
        let constraint_scores = scores.add(&labels(&[("search_region", "local")]));
        let global_constraint_scores = scores.add(&labels(&[("search_region", "global")]));
        let num_matchers = family_factory.new_gauge_family(
            "mapping_constraints_constraint_builder_2d_num_submap_scan_matchers",
            "Current number of constructed submap scan matchers",
        );
        let num_submap_scan_matchers = num_matchers.add(&Labels::new());
        *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(Metrics {
            constraints_searched,
            constraints_found,
            global_constraints_searched,
            global_constraints_found,
            queue_length,
            constraint_scores,
            global_constraint_scores,
            num_submap_scan_matchers,
        });
    }
}

impl Drop for ConstraintBuilder2D {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let state = self.inner.lock();
        assert_eq!(
            state.finish_node_task.get_state(),
            crate::common::task::State::New,
            "Check failed: finish_node_task_->GetState() == common::Task::NEW"
        );
        assert_eq!(
            state.when_done_task.get_state(),
            crate::common::task::State::New,
            "Check failed: when_done_task_->GetState() == common::Task::NEW"
        );
        assert_eq!(state.constraints.len(), 0, "WhenDone() was not called");
        assert_eq!(
            state.num_started_nodes, state.num_finished_nodes,
            "Check failed: num_started_nodes_ == num_finished_nodes_"
        );
        assert!(
            state.when_done.is_none(),
            "Check failed: when_done_ == nullptr"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, RwLock};

    use nalgebra::{DVector, Vector2, Vector3};

    use crate::common::internal::testing::thread_pool_for_testing::ThreadPoolForTesting;
    use crate::common::thread_pool::ThreadPoolInterface;
    use crate::mapping::id::{NodeId, SubmapId};
    use crate::mapping::internal::constraints::constraint_builder::create_constraint_builder_options;
    use crate::mapping::internal::constraints::constraint_builder_2d::*;
    use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;
    use crate::mapping::pose_graph_interface::Tag;
    use crate::mapping::trajectory_node::TrajectoryNodeData;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::submap_2d::Submap2D;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::point_cloud::PointCloud;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::{Rigid2d, Rigid3d, quaternion_identity};

    /// Records the results passed to the callback (C++ `MockCallback`).
    #[derive(Clone, Default)]
    struct MockCallback {
        calls: Arc<Mutex<Vec<Result>>>,
    }

    impl MockCallback {
        fn callback(&self) -> impl FnOnce(&Result) + Send + 'static {
            let calls = Arc::clone(&self.calls);
            move |result: &Result| calls.lock().unwrap().push(result.clone())
        }

        fn calls(&self) -> Vec<Result> {
            self.calls.lock().unwrap().clone()
        }
    }

    struct Fixture {
        constraint_builder: ConstraintBuilder2D,
        mock: MockCallback,
        thread_pool: Arc<ThreadPoolForTesting>,
    }

    impl Fixture {
        fn new() -> Self {
            let constraint_builder_parameters = resolve_lua_parameters(
                r#"
            include "pose_graph.lua"
            POSE_GRAPH.constraint_builder.sampling_ratio = 1
            POSE_GRAPH.constraint_builder.min_score = 0
            POSE_GRAPH.constraint_builder.global_localization_min_score = 0
            return POSE_GRAPH.constraint_builder"#,
            );
            let thread_pool = Arc::new(ThreadPoolForTesting::new());
            let constraint_builder = ConstraintBuilder2D::new(
                &create_constraint_builder_options(&constraint_builder_parameters),
                Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
            );
            Self {
                constraint_builder,
                mock: MockCallback::default(),
                thread_pool,
            }
        }
    }

    fn node_data_with_point() -> Arc<TrajectoryNodeData> {
        let mut filtered_gravity_aligned_point_cloud = PointCloud::new();
        filtered_gravity_aligned_point_cloud.push_back(RangefinderPoint {
            position: Vector3::new(0.1, 0.2, 0.3),
        });
        Arc::new(TrajectoryNodeData {
            time: Default::default(),
            gravity_alignment: quaternion_identity(),
            filtered_gravity_aligned_point_cloud,
            high_resolution_point_cloud: PointCloud::new(),
            low_resolution_point_cloud: PointCloud::new(),
            rotational_scan_matcher_histogram: DVector::zeros(0),
            local_pose: Rigid3d::identity(),
        })
    }

    fn empty_submap() -> SharedSubmap<Submap2D> {
        let map_limits = MapLimits::new(1., Vector2::new(2., 3.), CellLimits::new(100, 110));
        let conversion_tables = Arc::new(ValueConversionTables::new());
        Arc::new(RwLock::new(Submap2D::new(
            &Vector2::new(4.0f32, 5.0f32),
            Box::new(ProbabilityGrid::new(map_limits, &conversion_tables)),
            &conversion_tables,
        )))
    }

    mod constraint_builder_2d_test {
        use super::*;

        #[test]
        fn calls_back() {
            let f = Fixture::new();
            assert_eq!(f.constraint_builder.get_num_finished_nodes(), 0);
            f.constraint_builder.notify_end_of_node();
            f.constraint_builder.when_done(f.mock.callback());
            f.thread_pool.wait_until_idle();
            assert_eq!(f.constraint_builder.get_num_finished_nodes(), 1);
            let calls = f.mock.calls();
            assert_eq!(calls.len(), 1);
            assert!(calls[0].is_empty());
        }

        #[test]
        fn finds_constraints() {
            let f = Fixture::new();
            let node_data = node_data_with_point();
            let submap_id = SubmapId::new(0, 1);
            let submap = empty_submap();
            let mut expected_nodes = 0;
            for i in 0..2 {
                assert_eq!(
                    f.constraint_builder.get_num_finished_nodes(),
                    expected_nodes
                );
                for _ in 0..2 {
                    f.constraint_builder.maybe_add_constraint(
                        &submap_id,
                        &submap,
                        &NodeId::new(0, 0),
                        &node_data,
                        &Rigid2d::identity(),
                    );
                }
                f.constraint_builder.maybe_add_global_constraint(
                    &submap_id,
                    &submap,
                    &NodeId::new(0, 0),
                    &node_data,
                );
                f.constraint_builder.notify_end_of_node();
                f.thread_pool.wait_until_idle();
                expected_nodes += 1;
                assert_eq!(
                    f.constraint_builder.get_num_finished_nodes(),
                    expected_nodes
                );
                f.constraint_builder.notify_end_of_node();
                f.thread_pool.wait_until_idle();
                expected_nodes += 1;
                assert_eq!(
                    f.constraint_builder.get_num_finished_nodes(),
                    expected_nodes
                );
                f.constraint_builder.when_done(f.mock.callback());
                f.thread_pool.wait_until_idle();
                let calls = f.mock.calls();
                assert_eq!(calls.len(), i + 1);
                let result = &calls[i];
                assert_eq!(result.len(), 3);
                assert!(result.iter().all(|c| c.tag == Tag::InterSubmap));
                f.constraint_builder.delete_scan_matcher(&submap_id);
            }
        }
    }

    // Additional Rust tests (no C++ counterpart).
    mod extra {
        use super::*;
        use crate::common::thread_pool::ThreadPool;

        // Results are reported in insertion order, independent of the order
        // in which a multi-threaded pool finishes the searches.
        #[test]
        fn results_are_in_insertion_order_with_real_thread_pool() {
            let parameters = resolve_lua_parameters(
                r#"
            include "pose_graph.lua"
            POSE_GRAPH.constraint_builder.sampling_ratio = 1
            POSE_GRAPH.constraint_builder.min_score = 0
            POSE_GRAPH.constraint_builder.global_localization_min_score = 0
            POSE_GRAPH.constraint_builder.log_matches = false
            return POSE_GRAPH.constraint_builder"#,
            );
            let thread_pool: Arc<dyn ThreadPoolInterface> = Arc::new(ThreadPool::new(4));
            let constraint_builder = ConstraintBuilder2D::new(
                &create_constraint_builder_options(&parameters),
                Arc::clone(&thread_pool),
            );
            let node_data = node_data_with_point();
            let submaps = [empty_submap(), empty_submap()];
            let (sender, receiver) = std::sync::mpsc::channel();
            const NUM_NODES: i32 = 8;
            for node_index in 0..NUM_NODES {
                for (submap_index, submap) in submaps.iter().enumerate() {
                    constraint_builder.maybe_add_constraint(
                        &SubmapId::new(0, submap_index as i32),
                        submap,
                        &NodeId::new(0, node_index),
                        &node_data,
                        &Rigid2d::identity(),
                    );
                }
                constraint_builder.notify_end_of_node();
            }
            constraint_builder
                .when_done(move |result: &Result| sender.send(result.clone()).unwrap());
            let result = receiver
                .recv_timeout(std::time::Duration::from_secs(60))
                .expect("callback not called");
            assert_eq!(result.len(), (NUM_NODES * 2) as usize);
            for (i, constraint) in result.iter().enumerate() {
                assert_eq!(constraint.node_id, NodeId::new(0, i as i32 / 2));
                assert_eq!(constraint.submap_id, SubmapId::new(0, i as i32 % 2));
            }
            // All finish node tasks ran before the when done task.
            assert_eq!(constraint_builder.get_num_finished_nodes(), NUM_NODES);
        }

        // Constraints beyond `max_constraint_distance` are not searched.
        #[test]
        fn skips_distant_constraints() {
            let f = Fixture::new();
            let node_data = node_data_with_point();
            let submap = empty_submap();
            f.constraint_builder.maybe_add_constraint(
                &SubmapId::new(0, 0),
                &submap,
                &NodeId::new(0, 0),
                &node_data,
                &Rigid2d::from_translation(Vector2::new(100., 0.)),
            );
            f.constraint_builder.notify_end_of_node();
            f.constraint_builder.when_done(f.mock.callback());
            f.thread_pool.wait_until_idle();
            assert!(f.mock.calls()[0].is_empty());
        }

        #[test]
        #[should_panic(expected = "WhenDone() was not called")]
        fn dropping_with_pending_constraints_panics() {
            let f = Fixture::new();
            let node_data = node_data_with_point();
            let submap = empty_submap();
            f.constraint_builder.maybe_add_global_constraint(
                &SubmapId::new(0, 0),
                &submap,
                &NodeId::new(0, 0),
                &node_data,
            );
            f.constraint_builder.notify_end_of_node();
            f.thread_pool.wait_until_idle();
            drop(f.constraint_builder);
        }
    }
}
