//! Port of `cartographer/mapping/internal/constraints/constraint_builder_3d` (.h, .cc).
//!
//! RUST PORT NOTE: the threading and ownership model is the one of
//! [`ConstraintBuilder2D`](super::constraint_builder_2d) (see the module
//! documentation there): shared `Arc<Inner>` captured by tasks, the C++ task
//! graph unchanged, owned `SharedSubmap<Submap3D>` / `Arc<TrajectoryNodeData>`
//! handles instead of raw pointers, `Arc<OnceLock<Constraint>>` result slots
//! collected in insertion order.
//!
//! 3D specifics: `FastCorrelativeScanMatcher3D<G>` keeps its low resolution
//! grid for the lifetime of the matcher. A read guard of the submap's
//! `RwLock` cannot be kept across tasks, so the matcher owns a copy of the
//! (small) low resolution grid: `G = Arc<HybridGrid>`, cloned once per submap
//! in the construction task while holding the submap's read lock (the high
//! resolution grid is only needed to build the precomputation grids, the
//! rotational histogram is copied by the matcher). The Ceres refinement reads
//! both grids of the submap under its read lock (C++ uses the
//! `high_resolution_hybrid_grid` / `low_resolution_hybrid_grid` pointers
//! stored in `SubmapScanMatcher`). The branch-and-bound search runs without
//! any lock.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard, Weak};

use ceres::SolverSummary;
use nalgebra::Quaternion;

use crate::common::fixed_ratio_sampler::FixedRatioSampler;
use crate::common::histogram::Histogram as ScoreHistogram;
use crate::common::task::Task;
use crate::common::thread_pool::ThreadPoolInterface;
use crate::mapping::constraints::proto;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::{
    CeresScanMatcher3D, PointCloudAndHybridGridsPointers,
};
use crate::mapping::internal::three_d::scan_matching::fast_correlative_scan_matcher_3d::{
    FastCorrelativeScanMatcher3D, FastCorrelativeScanMatcher3DResult,
};
use crate::mapping::pose_graph_interface::{Constraint, Pose, Tag};
use crate::mapping::submaps::SharedSubmap;
use crate::mapping::three_d::hybrid_grid::HybridGrid;
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::metrics::counter::Counter;
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::metrics::gauge::Gauge;
use crate::metrics::histogram::Histogram;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::get_angle;

/// The result handed to the [`ConstraintBuilder3D::when_done`] callback.
pub type Result = Vec<Constraint>;

/// Callback registered with [`ConstraintBuilder3D::when_done`]; executed in
/// the thread pool.
pub type WhenDoneCallback = Box<dyn FnOnce(&Result) + Send + 'static>;

/// The fast correlative scan matcher type used per submap (owning a copy of
/// the low resolution grid).
type SubmapFastCorrelativeScanMatcher3D = FastCorrelativeScanMatcher3D<Arc<HybridGrid>>;

/// The C++ file-level metric statics.
#[derive(Clone, Copy)]
struct Metrics {
    constraints_searched: &'static dyn Counter,
    constraints_found: &'static dyn Counter,
    global_constraints_searched: &'static dyn Counter,
    global_constraints_found: &'static dyn Counter,
    queue_length: &'static dyn Gauge,
    constraint_scores: &'static dyn Histogram,
    constraint_rotational_scores: &'static dyn Histogram,
    constraint_low_resolution_scores: &'static dyn Histogram,
    global_constraint_scores: &'static dyn Histogram,
    global_constraint_rotational_scores: &'static dyn Histogram,
    global_constraint_low_resolution_scores: &'static dyn Histogram,
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
            constraint_rotational_scores: <dyn Histogram>::null(),
            constraint_low_resolution_scores: <dyn Histogram>::null(),
            global_constraint_scores: <dyn Histogram>::null(),
            global_constraint_rotational_scores: <dyn Histogram>::null(),
            global_constraint_low_resolution_scores: <dyn Histogram>::null(),
            num_submap_scan_matchers: <dyn Gauge>::null(),
        })
}

fn read_submap(submap: &RwLock<Submap3D>) -> RwLockReadGuard<'_, Submap3D> {
    submap.read().unwrap_or_else(PoisonError::into_inner)
}

struct SubmapScanMatcher {
    // RUST PORT NOTE: C++ `const HybridGrid* high_resolution_hybrid_grid` and
    // `low_resolution_hybrid_grid`; the grids are read through the submap
    // handle (under its read lock) for the Ceres refinement.
    submap: SharedSubmap<Submap3D>,
    // Set by the construction task; must only be accessed after
    // `creation_task_handle` has completed.
    fast_correlative_scan_matcher: OnceLock<SubmapFastCorrelativeScanMatcher3D>,
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

    score_histogram: ScoreHistogram,
    rotational_score_histogram: ScoreHistogram,
    low_resolution_score_histogram: ScoreHistogram,
}

// The part of the builder shared with thread-pool tasks (C++ `this`).
struct Inner {
    options: proto::ConstraintBuilderOptions,
    mutex: Mutex<State>,
    // RUST PORT NOTE: not guarded by `mutex_` in C++; a separate mutex here.
    per_submap_sampler: Mutex<BTreeMap<SubmapId, FixedRatioSampler>>,
    ceres_scan_matcher: CeresScanMatcher3D,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // Runs in a background thread and does computations for an additional
    // constraint. As output, it may create a new Constraint in 'constraint'.
    #[allow(clippy::too_many_arguments)]
    fn compute_constraint(
        &self,
        submap_id: &SubmapId,
        node_id: &NodeId,
        match_full_submap: bool,
        constant_data: &TrajectoryNodeData,
        global_node_pose: &Rigid3d,
        global_submap_pose: &Rigid3d,
        submap_scan_matcher: &SubmapScanMatcher,
        constraint: &OnceLock<Constraint>,
    ) {
        let fast_correlative_scan_matcher = submap_scan_matcher
            .fast_correlative_scan_matcher
            .get()
            .expect("Check failed: submap_scan_matcher.fast_correlative_scan_matcher");
        // The 'constraint_transform' (submap i <- node j) is computed from:
        // - a 'high_resolution_point_cloud' in node j and
        // - the initial guess 'initial_pose' (submap i <- node j).
        let metrics = metrics();
        let match_result: FastCorrelativeScanMatcher3DResult;

        // Compute 'pose_estimate' in three stages:
        // 1. Fast estimate using the fast correlative scan matcher.
        // 2. Prune if the score is too low.
        // 3. Refine.
        if match_full_submap {
            metrics.global_constraints_searched.increment();
            match fast_correlative_scan_matcher.match_full_submap(
                global_node_pose.rotation(),
                global_submap_pose.rotation(),
                constant_data,
                self.options.global_localization_min_score as f32,
            ) {
                Some(result) => {
                    match_result = result;
                    assert!(
                        f64::from(match_result.score) > self.options.global_localization_min_score,
                        "Check failed: match_result->score > \
                         options_.global_localization_min_score()"
                    );
                    assert!(node_id.trajectory_id >= 0);
                    assert!(submap_id.trajectory_id >= 0);
                    metrics.global_constraints_found.increment();
                    metrics
                        .global_constraint_scores
                        .observe(f64::from(match_result.score));
                    metrics
                        .global_constraint_rotational_scores
                        .observe(f64::from(match_result.rotational_score));
                    metrics
                        .global_constraint_low_resolution_scores
                        .observe(f64::from(match_result.low_resolution_score));
                }
                None => return,
            }
        } else {
            metrics.constraints_searched.increment();
            match fast_correlative_scan_matcher.match_(
                global_node_pose,
                global_submap_pose,
                constant_data,
                self.options.min_score as f32,
            ) {
                Some(result) => {
                    // We've reported a successful local match.
                    match_result = result;
                    assert!(
                        f64::from(match_result.score) > self.options.min_score,
                        "Check failed: match_result->score > options_.min_score()"
                    );
                    metrics.constraints_found.increment();
                    metrics
                        .constraint_scores
                        .observe(f64::from(match_result.score));
                    metrics
                        .constraint_rotational_scores
                        .observe(f64::from(match_result.rotational_score));
                    metrics
                        .constraint_low_resolution_scores
                        .observe(f64::from(match_result.low_resolution_score));
                }
                None => return,
            }
        }
        {
            let mut state = self.lock();
            state.score_histogram.add(match_result.score);
            state
                .rotational_score_histogram
                .add(match_result.rotational_score);
            state
                .low_resolution_score_histogram
                .add(match_result.low_resolution_score);
        }

        // Use the CSM estimate as both the initial and previous pose. This has
        // the effect that, in the absence of better information, we prefer the
        // original CSM estimate.
        let mut unused_summary = SolverSummary::default();
        let mut constraint_transform = Rigid3d::identity();
        {
            let submap = read_submap(&submap_scan_matcher.submap);
            self.ceres_scan_matcher.match_(
                match_result.pose_estimate.translation(),
                &match_result.pose_estimate,
                &[
                    PointCloudAndHybridGridsPointers {
                        point_cloud: &constant_data.high_resolution_point_cloud,
                        hybrid_grid: submap.high_resolution_hybrid_grid(),
                        intensity_hybrid_grid: None,
                    },
                    PointCloudAndHybridGridsPointers {
                        point_cloud: &constant_data.low_resolution_point_cloud,
                        hybrid_grid: submap.low_resolution_hybrid_grid(),
                        intensity_hybrid_grid: None,
                    },
                ],
                &mut constraint_transform,
                &mut unused_summary,
            );
        }

        let result = Constraint {
            submap_id: *submap_id,
            node_id: *node_id,
            pose: Pose {
                zbar_ij: constraint_transform,
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
                constant_data.high_resolution_point_cloud.size(),
                submap_id
            );
            if match_full_submap {
                info.push_str(" matches");
            } else {
                // Compute the difference between (submap i <- node j)
                // according to loop closure ('constraint_transform') and
                // according to global SLAM state.
                let difference =
                    global_node_pose.inverse() * *global_submap_pose * constraint_transform;
                info.push_str(&format!(
                    " differs by translation {:.2} rotation {:.3}",
                    difference.translation().norm(),
                    get_angle(&difference)
                ));
            }
            info.push_str(&format!(
                " with score {:.1}%.",
                100. * f64::from(match_result.score)
            ));
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
                    "{} computations resulted in {} additional constraints.\n\
                     Score histogram:\n{}\n\
                     Rotational score histogram:\n{}\n\
                     Low resolution score histogram:\n{}",
                    state.constraints.len(),
                    result.len(),
                    state.score_histogram.to_string(10),
                    state.rotational_score_histogram.to_string(10),
                    state.low_resolution_score_histogram.to_string(10)
                );
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
pub struct ConstraintBuilder3D {
    thread_pool: Arc<dyn ThreadPoolInterface>,
    inner: Arc<Inner>,
}

impl std::fmt::Debug for ConstraintBuilder3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintBuilder3D")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl ConstraintBuilder3D {
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
                    rotational_score_histogram: ScoreHistogram::new(),
                    low_resolution_score_histogram: ScoreHistogram::new(),
                }),
                per_submap_sampler: Mutex::new(BTreeMap::new()),
                ceres_scan_matcher: CeresScanMatcher3D::new(
                    options.ceres_scan_matcher_options_3d.as_ref().expect(
                        "ConstraintBuilderOptions.ceres_scan_matcher_options_3d is required",
                    ),
                ),
            }),
        }
    }

    /// Schedules exploring a new constraint between `submap` identified by
    /// `submap_id`, and the `constant_data` for `node_id`.
    ///
    /// `global_node_pose` and `global_submap_pose` are initial estimates of
    /// poses in the global map frame, i.e. both are gravity aligned.
    pub fn maybe_add_constraint(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap3D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
        global_node_pose: &Rigid3d,
        global_submap_pose: &Rigid3d,
    ) {
        if (global_node_pose.translation() - global_submap_pose.translation()).norm()
            > self.inner.options.max_constraint_distance
        {
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
            *global_node_pose,
            *global_submap_pose,
            "MaybeAddConstraint",
        );
    }

    /// Schedules exploring a new constraint between `submap` identified by
    /// `submap_id` and the `constant_data` for `node_id`. This performs
    /// full-submap matching.
    ///
    /// `global_node_rotation` and `global_submap_rotation` are initial
    /// estimates of roll and pitch, i.e. their yaw is essentially ignored.
    pub fn maybe_add_global_constraint(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap3D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
        global_node_rotation: &Quaternion<f64>,
        global_submap_rotation: &Quaternion<f64>,
    ) {
        self.add_constraint_task(
            submap_id,
            submap,
            node_id,
            constant_data,
            true, /* match_full_submap */
            Rigid3d::from_rotation(*global_node_rotation),
            Rigid3d::from_rotation(*global_submap_rotation),
            "MaybeAddGlobalConstraint",
        );
    }

    // The common locked part of MaybeAddConstraint and
    // MaybeAddGlobalConstraint.
    #[allow(clippy::too_many_arguments)]
    fn add_constraint_task(
        &self,
        submap_id: &SubmapId,
        submap: &SharedSubmap<Submap3D>,
        node_id: &NodeId,
        constant_data: &Arc<TrajectoryNodeData>,
        match_full_submap: bool,
        global_node_pose: Rigid3d,
        global_submap_pose: Rigid3d,
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
            let node_id = *node_id;
            let constant_data = Arc::clone(constant_data);
            let scan_matcher = Arc::clone(&scan_matcher);
            constraint_task.set_work_item(move || {
                inner.compute_constraint(
                    &submap_id,
                    &node_id,
                    match_full_submap,
                    &constant_data,
                    &global_node_pose,
                    &global_submap_pose,
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
        submap: &SharedSubmap<Submap3D>,
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
            .fast_correlative_scan_matcher_options_3d
            .unwrap_or_default();
        {
            let submap_scan_matcher = Arc::clone(&submap_scan_matcher);
            scan_matcher_task.set_work_item(move || {
                let submap = read_submap(&submap_scan_matcher.submap);
                let fast_correlative_scan_matcher = SubmapFastCorrelativeScanMatcher3D::new(
                    submap.high_resolution_hybrid_grid(),
                    Arc::new(submap.low_resolution_hybrid_grid().clone()),
                    submap.rotational_scan_matcher_histogram(),
                    &scan_matcher_options,
                );
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
            "mapping_constraints_constraint_builder_3d_constraints",
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
            "mapping_constraints_constraint_builder_3d_queue_length",
            "Queue length",
        );
        let queue_length = queue_length_family.add(&Labels::new());
        let boundaries = <dyn Histogram>::fixed_width(0.05, 20);
        let scores = family_factory.new_histogram_family(
            "mapping_constraints_constraint_builder_3d_scores",
            "Constraint scores built",
            &boundaries,
        );
        let constraint_scores =
            scores.add(&labels(&[("search_region", "local"), ("kind", "score")]));
        let constraint_rotational_scores = scores.add(&labels(&[
            ("search_region", "local"),
            ("kind", "rotational_score"),
        ]));
        let constraint_low_resolution_scores = scores.add(&labels(&[
            ("search_region", "local"),
            ("kind", "low_resolution_score"),
        ]));
        let global_constraint_scores =
            scores.add(&labels(&[("search_region", "global"), ("kind", "score")]));
        let global_constraint_rotational_scores = scores.add(&labels(&[
            ("search_region", "global"),
            ("kind", "rotational_score"),
        ]));
        let global_constraint_low_resolution_scores = scores.add(&labels(&[
            ("search_region", "global"),
            ("kind", "low_resolution_score"),
        ]));
        let num_matchers = family_factory.new_gauge_family(
            "mapping_constraints_constraint_builder_3d_num_submap_scan_matchers",
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
            constraint_rotational_scores,
            constraint_low_resolution_scores,
            global_constraint_scores,
            global_constraint_rotational_scores,
            global_constraint_low_resolution_scores,
            num_submap_scan_matchers,
        });
    }
}

impl Drop for ConstraintBuilder3D {
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

    use nalgebra::{DVector, Vector3};

    use crate::common::internal::testing::thread_pool_for_testing::ThreadPoolForTesting;
    use crate::common::thread_pool::ThreadPoolInterface;
    use crate::mapping::id::{NodeId, SubmapId};
    use crate::mapping::internal::constraints::constraint_builder::create_constraint_builder_options;
    use crate::mapping::internal::constraints::constraint_builder_3d::*;
    use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;
    use crate::mapping::pose_graph_interface::Tag;
    use crate::mapping::three_d::submap_3d::Submap3D;
    use crate::mapping::trajectory_node::TrajectoryNodeData;
    use crate::sensor::point_cloud::PointCloud;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::{Rigid3d, quaternion_identity};

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
        constraint_builder: ConstraintBuilder3D,
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
    POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.min_low_resolution_score = 0
    POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.min_rotational_score = 0
    return POSE_GRAPH.constraint_builder"#,
            );
            let thread_pool = Arc::new(ThreadPoolForTesting::new());
            let constraint_builder = ConstraintBuilder3D::new(
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

    mod constraint_builder_3d_test {
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
            let point = RangefinderPoint {
                position: Vector3::new(0.1, 0.2, 0.3),
            };
            let mut high_resolution_point_cloud = PointCloud::new();
            high_resolution_point_cloud.push_back(point);
            let mut low_resolution_point_cloud = PointCloud::new();
            low_resolution_point_cloud.push_back(point);
            let node_data = Arc::new(TrajectoryNodeData {
                time: Default::default(),
                gravity_alignment: quaternion_identity(),
                filtered_gravity_aligned_point_cloud: PointCloud::new(),
                high_resolution_point_cloud,
                low_resolution_point_cloud,
                rotational_scan_matcher_histogram: DVector::zeros(3),
                local_pose: Rigid3d::identity(),
            });
            let submap_id = SubmapId::new(0, 1);
            let submap = Arc::new(RwLock::new(Submap3D::new(
                0.1,
                0.1,
                &Rigid3d::identity(),
                &DVector::zeros(3),
            )));
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
                        &Rigid3d::identity(),
                        &Rigid3d::identity(),
                    );
                }
                f.constraint_builder.maybe_add_global_constraint(
                    &submap_id,
                    &submap,
                    &NodeId::new(0, 0),
                    &node_data,
                    &quaternion_identity(),
                    &quaternion_identity(),
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
}
