# Pose Graph, Optimization & Map Builder — Architecture Report (Phase 0)

Source of truth: `cartographer/cartographer/mapping/` (C++, read-only). Paths below are relative to `cartographer/cartographer/`.
Scope: global SLAM (pose graph 2D/3D, constraint builders, SPA optimization problems + cost functions, trimmers, connectivity, work queue), trajectory-builder wiring (collated/global trajectory builders, range data collator, local SLAM result replay), `MapBuilder` incl. `LoadState`/`SerializeState`.
Out of scope (documented elsewhere, referenced here): scan matchers (FastCorrelativeScanMatcher2D/3D, CeresScanMatcher2D/3D), submaps/grids, LocalTrajectoryBuilder2D/3D, sensor collators, `common::ThreadPool`/`Task` (summarized in §9 because the pose graph's correctness depends on their semantics).

Notation: `T_a_b` = transform taking points in frame b into frame a. `Rigid3d` = (translation `t`, unit quaternion `q`, stored w,x,y,z in Ceres arrays). `Rigid2d` = (x, y, θ). `Project2D(Rigid3d)` = (t.x, t.y, yaw(q)); `Embed3D(Rigid2d)` = (x, y, 0, Rz(θ)).

---------------------------------------------------------------------------------------------------

## 0. File map

| File | Role |
|---|---|
| `mapping/pose_graph_interface.h` | Abstract read API + data structs (`Constraint`, `LandmarkNode`, `SubmapData`, `TrajectoryData`, `TrajectoryState`). |
| `mapping/pose_graph.{h,cc}` | `PoseGraph` abstract write API, `CreatePoseGraphOptions`, constraint proto (de)serialization, shared `PoseGraph::ToProto`. |
| `mapping/pose_graph_trimmer.{h,cc}` | `Trimmable`, `PoseGraphTrimmer`, `PureLocalizationTrimmer`. |
| `mapping/internal/2d/overlapping_submaps_trimmer_2d.{h,cc}` | Coverage-based submap trimmer (2D only). |
| `mapping/internal/pose_graph_data.h` | `PoseGraphData`, `InternalSubmapData`, `InternalTrajectoryState`, `SubmapState`. |
| `mapping/internal/work_queue.h` | `WorkItem{time, task}`, `WorkQueue = deque<WorkItem>`. |
| `mapping/internal/2d/pose_graph_2d.{h,cc}`, `3d/pose_graph_3d.{h,cc}` | Concrete pose graphs (~1300 LOC each, near-identical; diffs in §3.9). |
| `mapping/internal/constraints/constraint_builder{,_2d,_3d}.{h,cc}` | Options factory, async loop-closure search. |
| `mapping/internal/optimization/optimization_problem_{interface,options,2d,3d}.*` | SPA problems. |
| `mapping/internal/optimization/ceres_pose.{h,cc}` | Rigid3d ↔ two Ceres parameter blocks. |
| `mapping/internal/optimization/cost_functions/*` | SPA 2D/3D, landmark 2D/3D, IMU rotation/acceleration 3D, helpers. |
| `mapping/internal/3d/rotation_parameterization.h` | `YawOnlyQuaternionPlus`, `ConstantYawQuaternionPlus` (Ceres local parameterizations). |
| `mapping/internal/3d/imu_integration.h` | `IntegrateImu` (used by 3D optimization). |
| `mapping/internal/trajectory_connectivity_state.{h,cc}`, `connected_components.{h,cc}` | Union-find of trajectories + last-connection timestamps. |
| `mapping/internal/collated_trajectory_builder.{h,cc}` | Sensor → collator → wrapped builder. |
| `mapping/internal/global_trajectory_builder.{h,cc}` | Local SLAM → pose graph glue (template over 2D/3D). |
| `mapping/internal/range_data_collator.{h,cc}` | Merges multiple range sensors (used inside LocalTrajectoryBuilder). |
| `mapping/internal/submap_controller.{h,cc}`, `local_slam_result_data.h`, `2d|3d/local_slam_result_{2d,3d}.*` | Replay of serialized local SLAM results into the pose graph (cloud/uplink use). |
| `mapping/trajectory_builder_interface.{h,cc}`, `map_builder_interface.{h,cc}`, `map_builder.{h,cc}` | Public facade + wiring + load/save. |
| `io/internal/mapping_state_serialization.cc` (outside mapping, but `SerializeState` = this) | `WritePbStream` order. |
| `mapping/internal/testing/*` | Fakes, mocks, proto helpers. |

---------------------------------------------------------------------------------------------------

## 1. Public data model (`pose_graph_interface.h`, `pose_graph.h`)

```cpp
struct Constraint {
  struct Pose { Rigid3d zbar_ij; double translation_weight; double rotation_weight; };
  SubmapId submap_id;   // i
  NodeId   node_id;     // j
  Pose pose;            // zbar_ij = T_submap_node (node frame -> submap frame)
  enum Tag { INTRA_SUBMAP, INTER_SUBMAP } tag;
};
struct LandmarkNode {
  struct LandmarkObservation { int trajectory_id; Time time; Rigid3d landmark_to_tracking_transform;
                               double translation_weight, rotation_weight; };
  vector<LandmarkObservation> landmark_observations;
  optional<Rigid3d> global_landmark_pose;
  bool frozen = false;
};
struct SubmapPose   { int version; Rigid3d pose; };          // version = submap->num_range_data()
struct SubmapData   { shared_ptr<const Submap> submap; Rigid3d pose; };
struct TrajectoryData { double gravity_constant = 9.8; array<double,4> imu_calibration{1,0,0,0} /*w,x,y,z*/;
                        optional<Rigid3d> fixed_frame_origin_in_map; };
enum class TrajectoryState { ACTIVE, FINISHED, FROZEN, DELETED };
using GlobalSlamOptimizationCallback =
    function<void(const map<int,SubmapId>& last_optimized_submap, const map<int,NodeId>& last_optimized_node)>;
struct PoseGraph::InitialTrajectoryPose { int to_trajectory_id; Rigid3d relative_pose; Time time; };
```

`PoseGraphInterface` (read API, all virtual, non-copyable): `RunFinalOptimization()`, `GetAllSubmapData()`, `GetSubmapData(id)`, `GetAllSubmapPoses()`, `GetLocalToGlobalTransform(traj)`, `GetTrajectoryNodes()`, `GetTrajectoryNodePoses()`, `GetTrajectoryStates()`, `GetLandmarkPoses()`, `SetLandmarkPose(id, pose, frozen=false)`, `DeleteTrajectory(traj)`, `IsTrajectoryFinished(traj)`, `IsTrajectoryFrozen(traj)`, `GetTrajectoryData()`, `constraints()`, `ToProto(bool include_unfinished_submaps)`, `SetGlobalSlamOptimizationCallback(cb)`.

`PoseGraph : PoseGraphInterface` (write API): `AddImuData`, `AddOdometryData`, `AddFixedFramePoseData`, `AddLandmarkData`, `FinishTrajectory`, `FreezeTrajectory`, `AddSubmapFromProto(global_pose, proto::Submap)`, `AddNodeFromProto(global_pose, proto::Node)`, `SetTrajectoryDataFromProto`, `AddNodeToSubmap(NodeId, SubmapId)`, `AddSerializedConstraints(vector<Constraint>)`, `AddTrimmer(unique_ptr<PoseGraphTrimmer>)`, `GetConnectedTrajectories()`, `GetImuData()`, `GetOdometryData()`, `GetFixedFramePoseData()`, `GetLandmarkNodes()`, `SetInitialTrajectoryPose(from, to, pose, time)`. Concrete classes add `AddNode(...)` (typed by Submap2D/3D) and `GetInterpolatedGlobalTrajectoryPose`.

Free functions (`pose_graph.cc`): `FromProto(RepeatedPtrField<proto::PoseGraph::Constraint>) -> vector<Constraint>`, `ToProto(Constraint) -> proto`, tag mapping (`INTRA_SUBMAP=0`, `INTER_SUBMAP=1`, anything else → `LOG(FATAL)`).

### 1.1 `PoseGraph::ToProto(include_unfinished_submaps)` (shared by 2D/3D)
1. For each `GetAllSubmapData()` entry: create trajectory proto lazily (ordered by first appearance); if `!include_unfinished && !submap->insertion_finished()` → remember in `unfinished_submaps`, skip; else add `{submap_index, pose}` (pose = global pose).
2. `constraints()` copy: drop constraints whose `submap_id` is unfinished (when excluding) and collect their node ids as "orphaned" (**computed but never used — dead code; nodes are always all emitted**).
3. All `GetTrajectoryNodes()` → `{node_index, timestamp = ToUniversal(time), pose = global_pose}` (CHECK constant_data != null).
4. `GetLandmarkPoses()` → `landmark_poses {landmark_id, global_pose}` (only optimized landmarks).
Proto `PoseGraph{ repeated Constraint constraint=2; repeated Trajectory trajectory=4; repeated LandmarkPose landmark_poses=5 }`, `Constraint{submap_id=1,node_id=2,relative_pose=3,tag=5,translation_weight=6,rotation_weight=7}`.

### 1.2 Options (`CreatePoseGraphOptions`, `CreateConstraintBuilderOptions`, `CreateOptimizationProblemOptions`, defaults from `configuration_files/pose_graph.lua`)

| Option | Default | Used by |
|---|---|---|
| `optimize_every_n_nodes` | 90 | ComputeConstraintsForNode (≤0 → never optimize online) |
| `matcher_translation_weight` / `matcher_rotation_weight` | 5e2 / 1.6e3 | INTRA_SUBMAP constraint weights |
| `max_num_final_iterations` (CHECK > 0) | 200 | RunFinalOptimization |
| `global_sampling_ratio` | 0.003 | per-trajectory FixedRatioSampler for global search |
| `log_residual_histograms` | true | 3D only |
| `global_constraint_search_after_n_seconds` | 10 | local vs global search decision |
| `overlapping_submaps_trimmer_2d{fresh_submaps_count,min_covered_area,min_added_submaps_count}` | absent (commented 1, 2, 5) | PoseGraph2D ctor adds trimmer if present |
| `constraint_builder.sampling_ratio` | 0.3 | per-submap sampler (local only) |
| `.max_constraint_distance` | 15 | local search gate |
| `.min_score` / `.global_localization_min_score` | 0.55 / 0.6 | FastCSM thresholds |
| `.loop_closure_translation_weight` / `.loop_closure_rotation_weight` | 1.1e4 / 1e5 | INTER_SUBMAP weights |
| `.log_matches` | true | logging |
| `.fast_correlative_scan_matcher` (2D) | linear 7, angular rad(30), depth 7 | |
| `.ceres_scan_matcher` (2D) | occ 20, trans 10, rot 1, ceres{nonmono=true, iters 10, threads 1} | |
| `.fast_correlative_scan_matcher_3d` | depth 8, full_res_depth 3, min_rot 0.77, min_low_res 0.55, xy 5, z 1, angular rad(15) | |
| `.ceres_scan_matcher_3d` | occ_0 5, occ_1 30, trans 10, rot 1, only_optimize_yaw false, ceres{false,10,1} | |
| `optimization_problem.huber_scale` | 1e1 | Huber loss scale |
| `.acceleration_weight` / `.rotation_weight` | 1.1e2 / 1.6e4 | 3D IMU terms |
| `.local_slam_pose_translation_weight` / `_rotation_weight` | 1e5 / 1e5 | consecutive node terms |
| `.odometry_translation_weight` / `_rotation_weight` | 1e5 / 1e5 | odometry terms |
| `.fixed_frame_pose_translation_weight` / `_rotation_weight` | 1e1 / 1e2 | fixed-frame terms |
| `.fixed_frame_pose_use_tolerant_loss`, `_tolerant_loss_param_a`, `_b` | false, 1, 1 | TolerantLoss |
| `.log_solver_summary` | false | `summary.FullReport()` |
| `.use_online_imu_extrinsics_in_3d` | true | imu_calibration block free/constant |
| `.fix_z_in_3d` | false | 3D: z fixed + odometry/local-slam terms instead of IMU terms |
| `.ceres_solver_options{use_nonmonotonic_steps,max_num_iterations,num_threads}` | false, 50, 7 | Solver::Options (CHECK iters>0, threads>0) |

`MapBuilderOptions`: `use_trajectory_builder_2d` XOR `use_trajectory_builder_3d` (CHECK_NE), `num_background_threads` (default 4), `collate_by_trajectory` (default false), `pose_graph_options`.
`TrajectoryBuilderOptions`: `trajectory_builder_{2d,3d}_options`, `initial_trajectory_pose{relative_pose, to_trajectory_id, timestamp(int64 universal)}`, `pure_localization` (deprecated bool, → keep 3), `pure_localization_trimmer{max_submaps_to_keep}`, `collate_fixed_frame` (default true), `collate_landmarks` (default false), `pose_graph_odometry_motion_filter{max_time_seconds,max_distance_meters,max_angle_radians}`.

---------------------------------------------------------------------------------------------------

## 2. Internal state (`pose_graph_data.h`, `work_queue.h`)

```cpp
enum class SubmapState { kNoConstraintSearch, kFinished };
struct InternalTrajectoryState {
  enum class DeletionState { NORMAL, SCHEDULED_FOR_DELETION, WAIT_FOR_DELETION };
  TrajectoryState state = ACTIVE; DeletionState deletion_state = NORMAL; };
struct InternalSubmapData { shared_ptr<const Submap> submap; SubmapState state = kNoConstraintSearch;
                            set<NodeId> node_ids; /* nodes inserted into this submap (have INTRA constraints) */ };
struct PoseGraphData {
  MapById<SubmapId, InternalSubmapData> submap_data;
  MapById<SubmapId, SubmapSpec2D> global_submap_poses_2d;   // last *applied* optimized poses (display)
  MapById<SubmapId, SubmapSpec3D> global_submap_poses_3d;
  MapById<NodeId, TrajectoryNode> trajectory_nodes;         // {shared_ptr<const Data> constant_data, Rigid3d global_pose}
  map<string, LandmarkNode> landmark_nodes;
  TrajectoryConnectivityState trajectory_connectivity_state;
  int num_trajectory_nodes = 0;                             // only incremented by AppendNode (live nodes)
  map<int, InternalTrajectoryState> trajectories_state;
  map<int, InitialTrajectoryPose> initial_trajectory_poses;
  vector<Constraint> constraints;                           // 2D: zbar stored GRAVITY-ALIGNED (see §3.9)
};
struct WorkItem { enum class Result { kDoNotRunOptimization, kRunOptimization };
                  steady_clock::time_point time; function<Result()> task; };
using WorkQueue = deque<WorkItem>;
```

Key `MapById` semantics the port must replicate (`mapping/id.h`): per-trajectory `std::map<int index, T>`; `Append` uses `last_index+1` (0 if empty) and CHECKs `can_append_`; `Insert` sets `can_append_=false`; `Trim` erases, sets `can_append_=false` if it removed the highest index, and erases the trajectory entry when it becomes empty (after which `Append` would restart at 0!). Iteration order = (trajectory_id, index). `lower_bound(traj, time)` = first element with `time() >= time` (binary search). Indices are never renumbered after trimming.

---------------------------------------------------------------------------------------------------

## 3. PoseGraph2D / PoseGraph3D

### 3.1 Members & locks
| Member | Guard | Notes |
|---|---|---|
| `options_` | const | |
| `global_slam_optimization_callback_` | none (set before use) | |
| `mutex_` (mutable absl::Mutex) | — | guards `data_`, `trimmers_`, `global_localization_samplers_`, `num_nodes_since_last_loop_closure_` |
| `work_queue_mutex_` | — | guards `work_queue_` (unique_ptr; **null ⇔ no drain/optimization in flight**) |
| `optimization_problem_` (unique_ptr) | **no lock**: only touched from the single logical worker (drain/HandleWorkQueue) + some getters under `mutex_` | see §8 races |
| `constraint_builder_` (ConstraintBuilder2D/3D) | own internal mutex | |
| `thread_pool_` (borrowed `common::ThreadPool*`) | | shared with constraint builder |
| `trimmers_`, `data_`, `conversion_tables_` (2D only, for `Submap2D` from proto) | | |
| `TrimmingHandle : Trimmable` | requires `parent_->mutex_` held | constructed on the stack inside HandleWorkQueue / DeleteTrajectoriesIfNeeded |

Metrics (gauges, no-op unless `RegisterMetrics`): work queue delay/size, inter-submap constraint counts (same/different trajectory), submaps active/frozen/deleted. Port as optional.

### 3.2 Work queue protocol (the core concurrency model)
- `AddWorkItem(f)`: lock `work_queue_mutex_`; if `work_queue_ == null` → allocate, schedule a `Task{DrainWorkQueue}` on the thread pool; push `{now, f}`.
- `DrainWorkQueue()` (runs on a pool thread): loop { lock queue mutex; if empty → `work_queue_.reset()`, return; pop front; unlock; `r = item()` } until an item returns `kRunOptimization`. Then `constraint_builder_.WhenDone(HandleWorkQueue)` and return **without** resetting `work_queue_` (so producers keep enqueuing, nothing else drains).
- `HandleWorkQueue(result)` (runs on a pool thread inside the constraint builder's when-done task):
  1. lock `mutex_`: append `result` constraints to `data_.constraints`.
  2. `RunOptimization()` (§3.6).
  3. If callback set: under `mutex_`, for each trajectory in `optimization_problem_->node_data()` with ≥1 node and ≥1 submap in the optimization problem, record last node id and last submap id; unlock; invoke callback (outside lock).
  4. lock `mutex_`: `UpdateTrajectoryConnectivity(c)` for each new constraint (all are INTER_SUBMAP, CHECKed); `DeleteTrajectoriesIfNeeded()`; run every trimmer with a `TrimmingHandle`; erase trimmers with `IsFinished()`; `num_nodes_since_last_loop_closure_ = 0`; update metrics.
  5. `DrainWorkQueue()` on the same thread.
- Result: at most one logical worker executes work items / optimization at any time. Work items themselves take `mutex_` as needed.

`WaitForAllComputations()` (used by `RunFinalOptimization`, destructor, tests):
1. Read `num_trajectory_nodes` (under `mutex_`), `num_finished_nodes_at_start`.
2. Wait (1 s timeout loop, printing `"\r\x1b[KOptimizing: x.x%..."` to stdout if nodes are pending) until `work_queue_ == null`.
3. Lock `mutex_`; register `constraint_builder_.WhenDone(cb)` where `cb` locks `mutex_`, appends results to `data_.constraints` (**no optimization**), sets `notification=true`; `AwaitWithTimeout(notification)` in 1 s steps (releases mutex while waiting).
4. `CHECK_EQ(constraint_builder_.GetNumFinishedNodes(), num_trajectory_nodes)`; print `"Optimizing: Done."`.

`RunFinalOptimization()`: enqueue item A (`SetMaxNumIterations(max_num_final_iterations)`, return `kRunOptimization`), enqueue item B (`SetMaxNumIterations(ceres_solver_options.max_num_iterations)`, `kDoNotRunOptimization`), then `WaitForAllComputations()`.

Destructor: `WaitForAllComputations()`, then CHECK `work_queue_ == null`.

### 3.3 Node ingestion (live SLAM)
`AddNode(constant_data, trajectory_id, insertion_submaps)` (1 or 2 submaps; front = older/matching submap):
1. `optimized_pose = GetLocalToGlobalTransform(traj) * constant_data->local_pose` (takes mutex internally).
2. `AppendNode` (under `mutex_`): `AddTrajectoryIfNeeded(traj)`; if `!CanAddWorkItemModifying(traj)` → WARNING (but note `CanAddWorkItemModifying` itself LOG(FATAL)s on finished/deleted); `node_id = data_.trajectory_nodes.Append(traj, {constant_data, optimized_pose})`; `++num_trajectory_nodes`; if the trajectory has no submaps in `data_.submap_data` or the last one's pointer ≠ `insertion_submaps.back()` → `Append` new `InternalSubmapData{submap=back}` (assumes a new submap first appears as `back()`).
3. `newly_finished_submap = insertion_submaps.front()->insertion_finished()` (captured NOW, not at execution).
4. `AddWorkItem(ComputeConstraintsForNode(node_id, insertion_submaps, newly_finished_submap))`.

`AddTrajectoryIfNeeded(traj)` (needs `mutex_`): default-insert state (ACTIVE/NORMAL); CHECK state ∉ {FINISHED, DELETED} and deletion NORMAL; `connectivity.Add(traj)`; create `FixedRatioSampler(global_sampling_ratio)` if missing.

`InitializeGlobalSubmapPoses(traj, time, insertion_submaps)` (under `mutex_`; grows the optimization problem):
- size 1: if optimization problem has no submaps for traj: if `initial_trajectory_poses[traj]` exists → `connectivity.Connect(traj, to_traj, time)`; `AddSubmap(traj, G)` where 2D `G = Project2D(ComputeLocalToGlobalTransform(global_submap_poses, traj) * submap0.local_pose)`, 3D same without projection. CHECK exactly 1 submap; return `{(traj,0)}` (CHECK pointer equality).
- size 2: `last = last submap id in optimization problem`. If `data_.submap_data[last].submap == front` → new back: `AddSubmap(traj, global(last) * L(front)^-1 * L(back))` where 2D `L = ComputeSubmapPose = Project2D(local_pose)`, 3D `L = local_pose`; return `{last, last+1}`. Else CHECK `last == back`, `last-1 == front`; return `{last-1, last}`.

`ComputeConstraintsForNode(node_id, insertion_submaps, newly_finished_submap)`:
Under `mutex_`:
1. `submap_ids = InitializeGlobalSubmapPoses(...)`; `matching_id = submap_ids.front()`.
2. 2D: `local_pose_2d = Project2D(local_pose * Rot(gravity_alignment^-1))`; `global_pose_2d = optProblem.submap(matching_id).global_pose * ComputeSubmapPose(front)^-1 * local_pose_2d`; `AddTrajectoryNode(traj, NodeSpec2D{time, local_pose_2d, global_pose_2d, gravity_alignment})`.
   3D: `global = optProblem.submap(matching_id).global_pose * front.local_pose^-1 * local_pose`; `AddTrajectoryNode(traj, NodeSpec3D{time, local_pose, global})`.
3. For each insertion submap i: CHECK state == kNoConstraintSearch; `node_ids.insert(node_id)`; push INTRA constraint `zbar = L(submap_i)^-1 * local_pose(_2d)` (2D embedded to 3D), weights `matcher_translation/rotation_weight`.
4. `finished_submap_ids` = all `data_.submap_data` entries with state kFinished (CHECK node not in their node_ids).
5. If `newly_finished_submap`: mark `submap_ids.front()` kFinished (CHECK was kNoConstraintSearch), copy its `node_ids`.
Outside lock:
6. `ComputeConstraint(node_id, s)` for each finished submap.
7. If newly finished: for every node in `optimization_problem_->node_data()` (all trajectories, read without lock) not in that submap's node_ids → `ComputeConstraint(node, new_submap)`.
8. `constraint_builder_.NotifyEndOfNode()`.
9. Under `mutex_`: `++num_nodes_since_last_loop_closure_`; return `kRunOptimization` iff `optimize_every_n_nodes > 0 && counter > optimize_every_n_nodes` (strict).

`ComputeConstraint(node_id, submap_id)`:
- 3D only: read `global_node_pose`, `global_submap_pose` from optimization problem *before* locking.
- Under `mutex_`: CHECK submap state kFinished; **return if `!submap->insertion_finished()`** (loaded unfinished submaps). `node_time = GetLatestNodeTime(node, submap)` = max(node time, time of the highest NodeId in submap.node_ids). `last = connectivity.LastConnectionTime(node.traj, submap.traj)`.
  - LOCAL if `node.traj == submap.traj` or `node_time < last + global_constraint_search_after_n_seconds`.
  - else GLOBAL if `global_localization_samplers_[node.traj]->Pulse()`.
  - Grab raw pointers to `constant_data` and submap (lifetime: shared_ptrs in `data_`; trimming may free them — see §8).
- Outside lock: LOCAL → 2D `MaybeAddConstraint(submap_id, submap, node_id, data, initial_relative_pose = optP.submap(s).global_pose^-1 * optP.node(n).global_pose_2d)`; 3D `MaybeAddConstraint(..., global_node_pose, global_submap_pose)`. GLOBAL → 2D `MaybeAddGlobalConstraint(submap_id, submap, node_id, data)`; 3D `MaybeAddGlobalConstraint(..., global_node_pose.rotation(), global_submap_pose.rotation())`.

`UpdateTrajectoryConnectivity(c)`: CHECK INTER_SUBMAP; `connectivity.Connect(c.node.traj, c.submap.traj, GetLatestNodeTime(c.node, c.submap))`.

### 3.4 Other sensor inputs (all deferred via work items, all `kDoNotRunOptimization`, all lock `mutex_` and require `CanAddWorkItemModifying(traj)`)
| Call | Work item effect |
|---|---|
| `AddImuData` | `optP.AddImuData` (2D: ignored by problem) |
| `AddOdometryData` | `optP.AddOdometryData` |
| `AddFixedFramePoseData` | `optP.AddFixedFramePoseData` |
| `AddLandmarkData` | for each observation → `data_.landmark_nodes[obs.id].landmark_observations.push_back({traj, data.time, obs.landmark_to_tracking_transform, obs.translation_weight, obs.rotation_weight})` |
| `SetLandmarkPose(id, pose, frozen)` | `landmark_nodes[id].global_landmark_pose = pose; .frozen = frozen` (no modifiability check) |
| `AddTrimmer(t)` | push into `trimmers_` (raw pointer captured, ownership released) |
| `SetInitialTrajectoryPose` | **immediate**, under mutex: `initial_trajectory_poses[from] = {to, pose, time}` |

`CanAddWorkItemModifying(traj)`: unknown traj → true; FINISHED → **LOG(FATAL)**; deletion_state ≠ NORMAL → LOG(FATAL); DELETED → LOG(FATAL). (Comment says intended to become warnings; port should decide — recommend returning `false` + warn, but tests expect no crash only in valid sequences.)

### 3.5 Trajectory state machine
```
(absent) --AddTrajectoryIfNeeded--> ACTIVE/NORMAL
ACTIVE --FinishTrajectory (work item, returns kRunOptimization)--> FINISHED  [all its submaps -> kFinished]
any    --FreezeTrajectory (work item)--> FROZEN
FINISHED|FROZEN --DeleteTrajectory: immediate deletion_state=SCHEDULED_FOR_DELETION;
                  work item: CHECK state∉{ACTIVE,DELETED}, CHECK SCHEDULED -> WAIT_FOR_DELETION
WAIT_FOR_DELETION --next HandleWorkQueue: DeleteTrajectoriesIfNeeded--> DELETED/NORMAL (all submaps trimmed)
any    --Trimmable::SetTrajectoryState (PureLocalizationTrimmer)--> DELETED
```
- `FinishTrajectory` work item: CHECK not already finished.
- `FreezeTrajectory`: immediately (under mutex) `connectivity.Add(traj)`; work item: CHECK not frozen; for every other FROZEN trajectory not transitively connected → `Connect(traj, other, FromUniversal(0))` (epoch time ⇒ treated as "long ago"; global search remains in effect); set FROZEN.
- `DeleteTrajectory` on unknown traj → WARNING, return. Deleting an ACTIVE trajectory CHECK-fails inside the work item.
- `DeleteTrajectoriesIfNeeded` (in HandleWorkQueue, under mutex): for WAIT_FOR_DELETION trajectories: `TrimSubmap` every submap id the optimization problem has for it; state=DELETED, deletion=NORMAL.
- `IsTrajectoryFinished/Frozen`: map lookup; annotated "requires mutex_" but called unlocked from public API and from `AddSubmapFromProto` (benign race in C++; in Rust take the lock).

### 3.6 RunOptimization (result application)
```
if optP.submap_data().empty(): return
optP.Solve(data_.constraints, GetTrajectoryStates() /*locks*/, data_.landmark_nodes)   // NO mutex_ held during Solve
lock mutex_
for traj in optP.node_data().trajectory_ids():
   for node in optP.node_data().trajectory(traj):
       2D: data_.trajectory_nodes[node].global_pose = Embed3D(node.global_pose_2d) * Rot(gravity_alignment)
       3D: data_.trajectory_nodes[node].global_pose = node.global_pose
   new = ComputeLocalToGlobalTransform(optP.submap_data(), traj)
   old = ComputeLocalToGlobalTransform(data_.global_submap_poses, traj)
   for nodes in data_.trajectory_nodes AFTER the last optimized node of traj:   // nodes added while solving
       global_pose = (new * old^-1) * global_pose
for (id, pose) in optP.landmark_data(): data_.landmark_nodes[id].global_landmark_pose = pose
data_.global_submap_poses = optP.submap_data()     // copy
3D && log_residual_histograms: LogResidualHistograms()
```
`ComputeLocalToGlobalTransform(global_submap_poses, traj)`: if no submap poses for traj → if `initial_trajectory_poses[traj]` exists: `GetInterpolatedGlobalTrajectoryPose(to_traj, time) * relative_pose`, else Identity. Else with `last` = last submap id in the given map: 2D `Embed3D(global_pose(last)) * submap_data[last].submap->local_pose()^-1`; 3D `global_pose(last) * local_pose^-1`.

`GetInterpolatedGlobalTrajectoryPose(traj, time)` (CHECK traj has nodes): `it = trajectory_nodes.lower_bound(traj, time)`; at begin → first global pose; at end → last global pose; else `transform::Interpolate` (lerp translation, slerp rotation) between prev and it.

`LogResidualHistograms` (3D): for INTRA constraints: `residual = zbar^-1 * (global_submap^-1 * node_global)`; histograms (10 buckets) of `NormalizeAngleDifference(GetAngle(residual))` and `|t|`.

### 3.7 Queries
- `GetSubmapDataUnderLock(id)`: missing → `{}` (null submap). If `global_submap_poses` has id → that pose (2D embedded). Else extrapolate `ComputeLocalToGlobalTransform(global_submap_poses, traj) * submap->local_pose()`.
- `GetAllSubmapData` = above for all `data_.submap_data`. `GetAllSubmapPoses` = `{num_range_data, pose}`.
- `GetTrajectoryNodes` copy; `GetTrajectoryNodePoses` → `{global_pose, optional{time, local_pose}}` (null constant_data → none).
- `GetLandmarkPoses` skips landmarks without global pose. `GetTrajectoryStates` maps `.state`.
- `GetImuData/GetOdometryData/GetFixedFramePoseData/GetTrajectoryData` → copies from optimization problem under `mutex_` (2D `imu_data()` returns an always-empty map).
- `GetConnectedTrajectories` → `connectivity.Components()`.
- `constraints()`: 2D returns copies with `zbar = zbar * Rot(gravity_alignment(node))` (un-gravity-align); 3D returns `data_.constraints` verbatim.

### 3.8 Deserialization entry points
| Method | Immediate (under mutex) | Work item |
|---|---|---|
| `AddSubmapFromProto(global_pose, proto)` | return if proto lacks `submap_2d`/`submap_3d`; build `Submap2D(proto.submap_2d, &conversion_tables_)` / `Submap3D(proto.submap_3d)`; `AddTrajectoryIfNeeded`; return if !modifiable; `submap_data.Insert(id, {submap})`; `global_submap_poses.Insert(id, {pose (2D: Project2D)})` (visible immediately); metric frozen/active via unlocked `IsTrajectoryFrozen` | `state = kFinished; optP.InsertSubmap(id, pose)` |
| `AddNodeFromProto(global_pose, proto)` | `constant_data = FromProto(node_data)`; `AddTrajectoryIfNeeded`; return if !modifiable; `trajectory_nodes.Insert(id, {data, global_pose})` (does NOT bump `num_trajectory_nodes`) | `optP.InsertTrajectoryNode(id, spec)`; 2D spec = `{time, Project2D(local*Rot(g^-1)), Project2D(global*Rot(g^-1)), g}`; 3D `{time, local, global}` |
| `SetTrajectoryDataFromProto(proto)` | 2D: only if `fixed_frame_origin_in_map` present (gravity/imu ignored). 3D: always; copies gravity_constant, imu_calibration (w,x,y,z), optional fixed frame origin | `optP.SetTrajectoryData(traj, data)` if modifiable |
| `AddNodeToSubmap(node, submap)` | — | if modifiable(submap.traj): `submap_data[submap].node_ids.insert(node)` |
| `AddSerializedConstraints(cs)` | — | per constraint: CHECK node & submap exist with data; INTRA → CHECK `node_ids.emplace(node)` was new; INTER → `UpdateTrajectoryConnectivity`; 2D push `zbar * Rot(g(node)^-1)` (re-gravity-align), 3D push verbatim. LOG count. |

### 3.9 2D vs 3D differences (complete list)
| Aspect | 2D | 3D |
|---|---|---|
| Overlapping-submaps trimmer auto-added in ctor | yes (if option present) | no |
| Optimization poses | `Rigid2d` gravity-aligned; node spec carries `gravity_alignment` | full `Rigid3d` |
| Stored constraint frame | gravity-aligned (`zbar` 2D-embedded); converted on `constraints()` / `AddSerializedConstraints` | as-is |
| Submap pose helper | `ComputeSubmapPose = Project2D(local_pose)` | `local_pose` |
| Constraint builder inputs | `initial_relative_pose` (Rigid2d) | global node & submap poses (local) / rotations (global) |
| `SetTrajectoryDataFromProto` | only fixed-frame origin, only if present | gravity, imu calib, fixed frame; always enqueues |
| Residual histograms | no | yes |
| `ValueConversionTables` member | yes | no |
| `WaitForAllComputations` | private | protected (tests subclass) |

### 3.10 TrimmingHandle (Trimmable impl; parent `mutex_` held)
- `num_submaps(traj)`, `GetSubmapIds(traj)` → from **optimization problem** submap_data (only initialized submaps).
- `GetOptimizedSubmapData()` → `data_.submap_data` entries with state kFinished AND present in `global_submap_poses` (pose = optimized).
- `GetTrajectoryNodes()`, `GetConstraints()` → references into `data_` (2D constraints are gravity-aligned here).
- `IsFinished(traj)` → `IsTrajectoryFinished`; `SetTrajectoryState(traj, s)` → `trajectories_state[traj].state = s`.
- `TrimSubmap(id)`:
  1. CHECK state kFinished.
  2. `nodes_to_retain` = union of node_ids of all other submaps; `nodes_to_remove` = `submap.node_ids \ nodes_to_retain`.
  3. Remove all constraints with `submap_id == id`.
  4. Remove all constraints whose node ∈ nodes_to_remove; collect the submaps of removed constraints into `losing`; erase from `losing` any submap that still has an INTER constraint; `constraint_builder_.DeleteScanMatcher(s)` for the rest.
  5. `data_.submap_data.Trim(id)`; `DeleteScanMatcher(id)`; `optP.TrimSubmap(id)`; metrics.
  6. For each node in nodes_to_remove: `data_.trajectory_nodes.Trim(n)`; `optP.TrimTrajectoryNode(n)`.
  Note: `global_submap_poses` is NOT trimmed here (it is overwritten by the next RunOptimization copy).

---------------------------------------------------------------------------------------------------

## 4. Trimmers

### 4.1 `PureLocalizationTrimmer(trajectory_id, num_submaps_to_keep)` — CHECK keep ≥ 2
`Trim(pg)`: if `pg->IsFinished(traj)` → keep = 0. `ids = GetSubmapIds(traj)`; trim `ids[i]` for `i + keep < ids.size()` (oldest first). If keep == 0 → `finished_=true`, `SetTrajectoryState(traj, DELETED)`. `IsFinished()` → `finished_`.

### 4.2 `OverlappingSubmapsTrimmer2D(uint16 fresh_submaps_count, double min_covered_area, uint16 min_added_submaps_count)`
`Trim(pg)`:
1. `submap_data = GetOptimizedSubmapData()`; if `submap_data.size() - current_submap_count_ <= min_added_submaps_count` → return (size_t arithmetic; underflow wraps to a huge value ⇒ proceeds).
2. Coverage grid uses `MapLimits` of the **first** submap: `offset = limits.max()`, `res = limits.resolution()`; `cell = (RoundToInt64((offset.x - p.x)/res), RoundToInt64((offset.y - p.y)/res))`; cells store `vector<(SubmapId, Time)>` in `std::map<CellId,…>`.
3. Freshness: for INTRA constraints only, latest (max) NodeId per submap; submap must exist in submap_data (else WARNING) and node in trajectory nodes; freshness = node time.
4. For each submap with freshness and `insertion_finished()`: add to `all_submap_ids`; `grid.ComputeCroppedLimits(&offset, &cell_limits)`; empty → WARNING, skip; for each xy in cropped limits with `grid.IsKnown(xy+offset)`: cell center in submap-local frame `(max.x - res*(index.y+0.5), max.y - res*(index.x+0.5), 0)` (note x/y swap); global = `Project2D(submap_global_pose * local_pose^-1 * center)`; `AddPoint(global.translation, id, freshness)`.
5. Per cell: if > fresh_submaps_count entries, sort by time descending and keep the first `fresh_submaps_count`; count cells per submap.
6. `min_covered_cells_count = (uint16)(min_covered_area / res²)`; keep submaps with count ≥ that; result = `all_submap_ids \ keep` (sorted set difference).
7. `current_submap_count_ = submap_data.size() - result.size()`; `TrimSubmap` each. `IsFinished()` always false.

---------------------------------------------------------------------------------------------------

## 5. Constraint builders (`constraints/constraint_builder_{2d,3d}`)

### 5.1 API
```cpp
ConstraintBuilder2D(const ConstraintBuilderOptions&, ThreadPoolInterface*);
void MaybeAddConstraint(SubmapId, const Submap2D*, NodeId, const TrajectoryNode::Data*, const Rigid2d& initial_relative_pose);
void MaybeAddGlobalConstraint(SubmapId, const Submap2D*, NodeId, const TrajectoryNode::Data*);
// 3D: MaybeAddConstraint(..., const Rigid3d& global_node_pose, const Rigid3d& global_submap_pose);
//     MaybeAddGlobalConstraint(..., const Quaterniond& global_node_rotation, const Quaterniond& global_submap_rotation);
void NotifyEndOfNode();
void WhenDone(const function<void(const vector<Constraint>&)>&);
int  GetNumFinishedNodes();
void DeleteScanMatcher(SubmapId);
Rigid2d ComputeSubmapPose(const Submap2D&) /* = Project2D(submap.local_pose()) */;
```
State (`mutex_`): `when_done_` (unique_ptr<function>), `num_started_nodes_`, `num_finished_nodes_`, `finish_node_task_`, `when_done_task_` (both `unique_ptr<Task>`, pre-created), `constraints_` (deque<unique_ptr<Constraint>> — pointer-stable slots; null = no match), `submap_scan_matchers_` (map<SubmapId, {grid ptr(s), unique_ptr<FastCSM>, weak_ptr<Task> creation_task_handle}>), `score_histogram_` (+ rotational/low-res in 3D). `per_submap_sampler_` (map<SubmapId, FixedRatioSampler>) is **not** guarded (only touched from the single worker + DeleteScanMatcher). `ceres_scan_matcher_` (CeresScanMatcher2D/3D, stateless options).
Destructor CHECKs: both tasks still NEW, `constraints_` empty, started == finished, no pending `when_done_`.

### 5.2 Gating & sampling
- Local (`MaybeAddConstraint`): reject if distance > `max_constraint_distance` (2D: `|initial_relative_pose.t|`, i.e. node position relative to submap global origin; 3D: `|node.t - submap.t|`); then per-submap `FixedRatioSampler(sampling_ratio).Pulse()` (created on first use); reject on false.
- Global (`MaybeAddGlobalConstraint`): no distance gate, no per-submap sampler (sampling is the pose graph's per-trajectory `global_sampling_ratio`).
- `FixedRatioSampler::Pulse()`: `++pulses; if samples/pulses < ratio { ++samples; return true }` (deterministic; first pulse true if ratio > 0). CHECK 0 ≤ ratio ≤ 1.
- Then (locked): WARN if `when_done_` pending; append null slot; `DispatchScanMatcherConstruction` (once per submap: schedule task building `FastCorrelativeScanMatcher2D(*grid, options)` / `3D(high_res_grid, &low_res_grid, &submap.rotational_scan_matcher_histogram(), options_3d)`); schedule constraint task depending on construction task; `finish_node_task_->AddDependency(constraint_task)`.

### 5.3 ComputeConstraint (pool thread, unlocked)
2D:
1. `initial_pose = ComputeSubmapPose(submap) * initial_relative_pose` (node in submap's local/map frame; global search: identity relative).
2. Full: `MatchFullSubmap(filtered_gravity_aligned_point_cloud, global_localization_min_score, &score, &pose)`; local: `Match(initial_pose, cloud, min_score, &score, &pose)`. Fail → return (slot stays null). CHECK `score > threshold`.
3. Lock: `score_histogram_.Add(score)`.
4. `ceres_scan_matcher_.Match(target_translation = pose.translation(), initial = pose, cloud, *grid, &pose, &unused_summary)`.
5. `zbar = Embed3D(ComputeSubmapPose(submap)^-1 * pose)`, weights `loop_closure_*`, tag INTER_SUBMAP.
6. `log_matches`: log translation/rotation diff vs initial (local) and score %.
3D:
1. Full: `MatchFullSubmap(global_node_rot, global_submap_rot, constant_data, global_localization_min_score)`; local: `Match(global_node_pose, global_submap_pose, constant_data, min_score)`; returns `unique_ptr<Result{score, pose_estimate (T_submap_node), rotational_score, low_resolution_score}>` or null.
2. Histograms of all three scores.
3. `ceres_scan_matcher_.Match(pose_estimate.translation(), pose_estimate, {{&high_res_cloud, high_res_grid, nullptr}, {&low_res_cloud, low_res_grid, nullptr}}, &zbar, &unused)`.
4. Constraint `{zbar, loop_closure weights, INTER_SUBMAP}`; log diff = `global_node^-1 * global_submap * zbar`.

### 5.4 Task graph & callbacks
```
scan_matcher_task(submap)  ──► constraint_task(k) ──► finish_node_task(node n) ──► when_done_task
                                                     (NotifyEndOfNode schedules it,  (WhenDone schedules it,
                                                      new one created)                new one created)
```
- `NotifyEndOfNode()`: set work `{lock; ++num_finished_nodes_}` on current `finish_node_task_`, schedule it, replace with fresh Task, `when_done_task_->AddDependency(handle)`, `++num_started_nodes_`.
- `WhenDone(cb)`: CHECK no pending callback; store; set `when_done_task_` work = `RunWhenDoneCallback`; schedule; replace with fresh Task.
- `RunWhenDoneCallback()`: under lock collect non-null `*constraints_` in insertion order, optional logging, `constraints_.clear()`, move callback out; **invoke callback outside the lock** on the pool thread. Callback = `PoseGraph::HandleWorkQueue` or WaitForAllComputations' collector.
- `DeleteScanMatcher(id)`: lock; WARN if when_done pending; erase matcher and per-submap sampler.

---------------------------------------------------------------------------------------------------

## 6. Optimization problems

### 6.1 Interface (`OptimizationProblemInterface<NodeType, SubmapType, RigidType>`)
`AddImuData(traj, ImuData)`, `AddOdometryData(traj, OdometryData)`, `AddTrajectoryNode(traj, Node)` (Append; also default-creates `trajectory_data_[traj]`), `InsertTrajectoryNode(NodeId, Node)` (Insert; default-creates trajectory data), `TrimTrajectoryNode(NodeId)`, `AddSubmap(traj, pose)` (Append), `InsertSubmap(SubmapId, pose)`, `TrimSubmap(SubmapId)`, `SetMaxNumIterations(int32)` (mutates the stored ceres options), `Solve(constraints, trajectories_state, landmark_nodes)`, accessors `node_data()`, `submap_data()`, `landmark_data()` (map<string,Rigid3d>), `imu_data()`, `odometry_data()`. Concrete extras: `AddFixedFramePoseData`, `SetTrajectoryData`, `fixed_frame_pose_data()`, `trajectory_data()`.
```cpp
struct NodeSpec2D { Time time; Rigid2d local_pose_2d; Rigid2d global_pose_2d; Quaterniond gravity_alignment; };
struct SubmapSpec2D { Rigid2d global_pose; };
struct NodeSpec3D { Time time; Rigid3d local_pose; Rigid3d global_pose; };
struct SubmapSpec3D { Rigid3d global_pose; };
```
`TrimTrajectoryNode(id)`: `imu_data_.Trim(node_data_, id)`, `odometry_data_.Trim(...)`, `fixed_frame_pose_data_.Trim(...)` (MapByTime::Trim: drop sensor samples strictly inside the gap (prev node time, next node time), retaining the first and last sample of the gap; gap bounds open to ±∞ at trajectory ends), then `node_data_.Trim(id)`; if trajectory now has no nodes → erase `trajectory_data_[traj]`.

### 6.2 Solve — 2D (`OptimizationProblem2D::Solve`)
Return if `node_data_` empty. `frozen = {traj : state == FROZEN}`. Fresh `ceres::Problem` (default options) each call.

Parameter blocks:
| Block | Size / param | Init | Constant |
|---|---|---|---|
| `C_submaps[id]` = [x, y, θ] | 3, none | `FromPose(global_pose)` (θ = normalized_angle) | first submap in MapById order (global first), or frozen traj |
| `C_nodes[id]` = [x, y, θ] | 3, none | `global_pose_2d` | frozen traj |
| `C_landmarks[id]` (CeresPose) | t: 3 none; q: 4 `QuaternionParameterization` | `global_landmark_pose` or interpolated (§6.4) | `landmark.frozen` → both |
| `C_fixed_frames[traj]` = [x,y,θ] | 3, none | `Project2D(fixed_frame_origin_in_map)` if set, else `node.global_pose_2d * Project2D(zbar)^-1` at first node with a fixed-frame sample | never |

Residuals:
| Term | Cost | Blocks | Loss |
|---|---|---|---|
| each constraint | `SpaCost2D(c.pose)` (§7.1) | (C_submaps[i], C_nodes[j]) | INTER: `HuberLoss(huber_scale)`; INTRA: none |
| landmark observations | `LandmarkCost2D` | (prev node, next node, landmark q, landmark t) | `HuberLoss(huber_scale)` |
| odometry, consecutive nodes (index+1 only), non-frozen traj, both odom interpolations exist | `SpaCost2D({rel_odom, odometry_translation_weight, odometry_rotation_weight})`, `rel_odom = Rot(g1) * odom(t1)^-1 * odom(t2) * Rot(g2^-1)` | (C_nodes[n1], C_nodes[n2]) | none |
| local SLAM, same pairs | `SpaCost2D({Embed3D(local1^-1 * local2), local_slam_pose_translation/rotation_weight})` | (n1, n2) | none |
| fixed frame (all traj incl. frozen, every node with interpolable sample) | `SpaCost2D({ffp, fixed_frame_pose_translation/rotation_weight})` | (C_fixed_frames[traj], C_nodes[n]) | `TolerantLoss(a, b)` if `use_tolerant_loss` else none |

Solve: `ceres::Solve(CreateCeresSolverOptions(opts), &problem, &summary)`; if `log_solver_summary` → `LOG(INFO) << summary.FullReport()`. Store: submaps/nodes ← `ToPose(array)` (`Rigid2d({x,y}, θ)`), `trajectory_data_[traj].fixed_frame_origin_in_map = Embed3D(fixed)`, `landmark_data_[id] = C_landmark.ToRigid()`.

Interpolation helpers (both dims):
- Odometry `InterpolateOdometry(traj, t)`: `it = lower_bound(traj, t)`; end → null; begin → value only if `it.time == t`; else `Interpolate(prev, it, t)`.
- Fixed frame `Interpolate(map, traj, t)`: as above but null if `it->pose` empty, and null if `prev->pose` empty.

### 6.3 Solve — 3D (`OptimizationProblem3D::Solve`)
Return if `node_data_` empty; CHECK `submap_data_` non-empty. `translation_parameterization()` = `SubsetParameterization(3, {2})` if `fix_z_in_3d` else null (fresh instance per block).

Parameter blocks (`CeresPose` = separate translation[3] and rotation[4] (w,x,y,z) blocks, shared_ptr storage):
| Block | Translation param | Rotation param | Constant |
|---|---|---|---|
| first submap | translation_parameterization() | `AutoDiffLocalParameterization<ConstantYawQuaternionPlus,4,2>` | translation always; both if frozen |
| other submaps | translation_parameterization() | `QuaternionParameterization` | both if frozen |
| nodes | translation_parameterization() | `QuaternionParameterization` | both if frozen |
| landmarks | none | `QuaternionParameterization` | both if `frozen` |
| `trajectory_data.imu_calibration` (4) | — | `QuaternionParameterization` (raw `new`) | unless `use_online_imu_extrinsics_in_3d` |
| `trajectory_data.gravity_constant` (1) | implicitly added by residual | — | `SetParameterLowerBound(&g, 0, 0.0)` if any accel term |
| fixed frame per traj | none | `AutoDiffLocalParameterization<YawOnlyQuaternionPlus,4,1>`; init = (translation, yaw-only rotation `AngleAxis(GetYaw(q), Z)`) | never |

Residuals:
| Term | When | Cost | Blocks | Loss |
|---|---|---|---|---|
| constraints | always | `SpaCost3D(c.pose)` | (q_i, t_i, q_j, t_j) | INTER: Huber(huber_scale); INTRA none |
| landmarks | always | `LandmarkCost3D` | (q_prev, t_prev, q_next, t_next, q_lm, t_lm) | Huber(huber_scale) |
| IMU rotation | `!fix_z_in_3d`, non-frozen traj, consecutive pair (index+1) | `RotationCost3D(rotation_weight / Δt1, Δq_imu(t1→t2))` | (q_1, q_2, imu_calibration) | none |
| IMU acceleration | `!fix_z_in_3d`, non-frozen, consecutive triple (n, n+1, n+2) | `AccelCost3D(acceleration_weight / (Δt1+Δt2), Δv, Δt1, Δt2)` | (q_2, t_1, t_2, t_3, &gravity_constant, imu_calibration) | none |
| odometry | **only if `fix_z_in_3d`**, non-frozen, consecutive pair, both interpolations | `SpaCost3D({odom1^-1*odom2, odometry_*_weight})` | (q_1,t_1,q_2,t_2) | none |
| local SLAM | **only if `fix_z_in_3d`**, non-frozen, consecutive pair | `SpaCost3D({local1^-1*local2, local_slam_pose_*_weight})` | (q_1,t_1,q_2,t_2) | none |
| fixed frame | all traj with data, each node with interpolable sample | `SpaCost3D({ffp, fixed_frame_*_weight})`; init origin = `fixed_frame_origin_in_map` or `node.global * ffp^-1` | (q_ff, t_ff, q_n, t_n) | Tolerant(a,b) if enabled |

IMU bookkeeping (per non-frozen trajectory, `!fix_z`): CHECK `imu_data_.HasTrajectory(traj)` and non-empty (**3D without IMU data crashes unless fix_z_in_3d**). `imu_it = begin`. For each consecutive pair: advance `imu_it` while `next(imu_it).time <= t1`; `imu_it2 = imu_it`; `result = IntegrateImu(imu, t1, t2, &imu_it)`. If third node exists (index+1 again): `c1 = t1 + Δt1/2`, `c2 = t2 + Δt2/2`; `r_a = IntegrateImu(imu, t1, c1, &imu_it2)`; `r_b = IntegrateImu(imu, c1, c2, &imu_it2)`; `Δv = (result.delta_rotation^-1 * r_a.delta_rotation) * r_b.delta_velocity`; add accel term. Always add rotation term with `result.delta_rotation`.
`IntegrateImu(imu, start, end, &it)` (identity calibrations): CHECK `start <= end`, `it->time <= start`, `next(it)->time > start` (if exists). Piecewise-constant: for each segment `[cur, min(next_imu_time, end)]`, `Δt`: `q *= AngleAxisToQuat(ω_it * Δt)`; `v += q * (a_it * Δt)`; `p += Δt * v`; advance `it` when reaching next sample time.

Post-solve logging (`log_solver_summary`): FullReport + per trajectory "Gravity was: g" and "IMU correction was: rad2deg(2*acos(|w|)) deg (w,x,y,z)".
Store: submaps, nodes ← `CeresPose::ToRigid()` (`Rigid3d::FromArrays(rotation, translation)`, no renormalization beyond Ceres' manifold), fixed frame origins, landmarks. `imu_calibration`/`gravity_constant` are updated **in place** in `trajectory_data_` (they are the parameter blocks).

### 6.4 Landmark handling (2D & 3D `AddLandmarkCostFunctions`)
For each landmark, each observation (traj, time):
- `begin = node_data.BeginOfTrajectory(obs.traj)`; skip if `obs.time < begin.time` (**dereferences begin without checking the trajectory has nodes — UB if none**).
- `next = lower_bound(traj, obs.time)`; skip if end. If `next == begin` → `next = next+1` (**UB if the trajectory has exactly one node**). `prev = next-1`.
- Create landmark block on first use (init from `global_landmark_pose` or `GetInitialLandmarkPose` = interpolated node pose (2D: `InterpolateNodes2D` with gravity alignment, z=0; 3D: `InterpolateNodes3D`) `* landmark_to_tracking_transform`); frozen → constant.
- `interpolation_parameter = (obs.time − prev.time) / (next.time − prev.time)` (seconds, double).

### 6.5 `CeresPose`
`CeresPose(Rigid3d, unique_ptr<LocalParameterization> t_param, unique_ptr<LocalParameterization> q_param, Problem*)`: allocates `shared_ptr<Data{array<3> translation; array<4> rotation (w,x,y,z)}>`, `AddParameterBlock(t,3,t_param)`, `AddParameterBlock(q,4,q_param)` (null param = Euclidean). Copyable (shared data) — stored in `MapById`/`std::map`, so addresses must be stable while the Problem lives. `FromPose(Rigid3d) -> Data`.

---------------------------------------------------------------------------------------------------

## 7. Cost functions (exact residuals)

All are `ceres::AutoDiffCostFunction` (Jets) except `AnalyticalSpaCostFunction2D` (`SizedCostFunction<3,3,3>`, **test-only**, not used in Solve).

### 7.1 SPA 2D — residual 3, params (start[3], end[3])
`z = Project2D(zbar)`; `c=cos θi, s=sin θi, dx = xj−xi, dy = yj−yi`;
`h = (c dx + s dy, −s dx + c dy, θj − θi)`;
`e = (wt (z.x − h0), wt (z.y − h1), wr NormalizeAngleDifference(z.θ − h2))`, `z.θ = Rotation2D.angle()` of the projected yaw.
Analytical Jacobian (row-major 3×3, matches autodiff; NormalizeAngleDifference treated as identity):
`J_start = [[wt c, wt s, wt(s dx − c dy)], [−wt s, wt c, wt(c dx + s dy)], [0, 0, wr]]`, `J_end = [[−wt c, −wt s, 0], [wt s, −wt c, 0], [0, 0, −wr]]`.

### 7.2 SPA 3D — residual 6, params (q_i[4], t_i[3], q_j[4], t_j[3])
`h_t = q_i^-1 (t_j − t_i)` (q_i inverse built as conjugate (w,−x,−y,−z) — assumes unit);
`h_rot_inv = conj(q_j) * q_i`;
`e_rot = RotationQuaternionToAngleAxisVector(h_rot_inv * zbar.q)`;
`e = (wt (zbar.t − h_t), wr e_rot)`.
`RotationQuaternionToAngleAxisVector(q)`: normalize; if w<0 negate all; `angle = 2 atan2(|vec|, w)`; `scale = angle < 1e-7 ? 2 : angle / sin(angle/2)`; return `scale * vec`.

### 7.3 Helpers (`cost_helpers_impl.h`)
- `ScaleError3/6`: translation comps × wt, rotation comps × wr.
- `SlerpQuaternions(a, b, f)`: `cosθ = a·b`; `prev=1−f, next=f`; if `|cosθ| < 1−1e-5`: `θ = acos(|cosθ|)`, `prev = sin((1−f)θ)/sinθ`, `next = sin(fθ)/sinθ`; if `cosθ < 0` → `next = −next`; result `prev·a + next·b` (not renormalized).
- `InterpolateNodes3D(qa, ta, qb, tb, f)` = (Slerp(qa,qb,f), lerp(ta,tb,f)).
- `InterpolateNodes2D(pa, ga, pb, gb, f)`: `qa = normalize(AngleAxis(pa.θ, Z) * ga)`, same for b; return (Slerp, (lerp x, lerp y, 0)).

### 7.4 Landmark 2D — residual 6, params (prev[3], next[3], q_lm[4], t_lm[3])
`(q_int, t_int) = InterpolateNodes2D(prev, g_prev, next, g_next, f)`; `e = ScaleError(ComputeUnscaledError3D(landmark_to_tracking, q_int, t_int, q_lm, t_lm), obs.wt, obs.wr)`.
### 7.5 Landmark 3D — residual 6, params (q_prev, t_prev, q_next, t_next, q_lm, t_lm): same with `InterpolateNodes3D`.

### 7.6 IMU rotation 3D — residual 3, params (q_start[4], q_end[4], C[4])
`err = conj(q_end) * q_start * C * Δq_imu * conj(C)`; `e = s * (err.x, err.y, err.z)` with `s = rotation_weight / Δt1` (no ×2 factor, no w sign handling).

### 7.7 IMU acceleration 3D — residual 3, params (q_mid[4], p1[3], p2[3], p3[3], g[1], C[4])
`imu_dv = q_mid * C * Δv_imu − g * 0.5 (Δt1 + Δt2) * ẑ`; `v1 = (p2 − p1)/Δt1`; `v2 = (p3 − p2)/Δt2`; `e = s * (imu_dv − (v2 − v1))`, `s = acceleration_weight / (Δt1 + Δt2)`.

### 7.8 Local parameterizations (`rotation_parameterization.h`), x = (w,x,y,z)
- `YawOnlyQuaternionPlus` (global 4, local 1): `d = clamp(δ0, −0.5, 0.5)`; `q_δ = (sqrt(1 − d²), 0, 0, d)`; `x ⊞ δ = q_δ ⊗ x` (**left** multiply ⇒ yaw about world Z).
- `ConstantYawQuaternionPlus` (global 4, local 2): `n = |(δ0, δ1)|`; `q_δ = (n<1e-6 ? 1 : cos n, k δ0, k δ1, 0)` with `k = n<1e-6 ? 1 : sin(n)/n`; `x ⊞ δ = x ⊗ q_δ` (**right** multiply ⇒ roll/pitch in submap frame).
- Ceres `QuaternionParameterization` (global 4, local 3): `q_δ = (cos|δ|, sin|δ|/|δ| δ)` (identity if |δ|=0); `x ⊞ δ = q_δ ⊗ x` (left multiply). Jacobian is analytic 4×3 at δ=0.
- `SubsetParameterization(3, {2})`: z held constant (local size 2).

---------------------------------------------------------------------------------------------------

## 8. Exact Ceres feature inventory (for the focused Rust replacement)

Pose-graph scope (optimization problems + cost functions); scan matchers additionally use Ceres (separate doc) — constraint builder only passes an unused `Solver::Summary`.

| Feature | Where | Notes for replacement |
|---|---|---|
| `ceres::Problem` with default `Problem::Options` (owns cost, loss, local parameterizations) | Solve 2D/3D | rebuilt from scratch every Solve |
| `Problem::AddParameterBlock(double*, size)` / `(double*, size, LocalParameterization*)` | 2D: size 3 plain; 3D: 3 (+Subset), 4 (+Quaternion / AutoDiffLocal) | parameter blocks identified by pointer; blocks may also be implicitly added by `AddResidualBlock` (gravity, fixed frames 2D, landmarks) |
| `Problem::AddResidualBlock(cost, loss_or_null, blocks...)` | 2–6 blocks | block sizes per §6/§7 |
| `Problem::SetParameterBlockConstant` | first submap(s), frozen, landmarks, imu_calibration | constant blocks excluded from the tangent space |
| `Problem::SetParameterLowerBound(ptr, 0, 0.0)` | gravity constant | bounds handled by projection in Ceres TR |
| `AutoDiffCostFunction<F, nres, sizes...>` | all cost functions | needs forward-mode dual numbers supporting: `+ − * /`, `sin`, `cos`, `acos`, `atan2`, `sqrt`, `abs`, comparisons, Eigen-like quaternion multiply/conjugate/normalize, quaternion·vector rotation |
| `SizedCostFunction` (analytical) | test only | optional |
| `HuberLoss(a)` | INTER constraints, landmarks | ρ(s) = s if s ≤ a², else 2a√s − a² |
| `TolerantLoss(a, b)` | fixed frame (opt-in) | ρ(s) = b log(1 + e^{(s−a)/b}) − b log(1 + e^{−a/b}) |
| `QuaternionParameterization` | 3D rotations, landmarks (2D too), imu_calibration | see §7.8 |
| `SubsetParameterization(3, {2})` | `fix_z_in_3d` | |
| `AutoDiffLocalParameterization<F, 4, k>` (k=1,2) | 3D fixed frame, first submap | plus via functor, Jacobian via autodiff at δ=0 |
| `ceres::Solve(Solver::Options, Problem*, Summary*)` | | Options set: `use_nonmonotonic_steps`, `max_num_iterations`, `num_threads` only; everything else **Ceres defaults**: `minimizer_type=TRUST_REGION`, `trust_region_strategy_type=LEVENBERG_MARQUARDT`, `linear_solver_type=SPARSE_NORMAL_CHOLESKY` (when built with a sparse backend; else DENSE_QR), `max_consecutive_nonmonotonic_steps=5`, `initial_trust_region_radius=1e4`, `max_trust_region_radius=1e16`, `min_trust_region_radius=1e-32`, `min_relative_decrease=1e-3`, `min_lm_diagonal=1e-6`, `max_lm_diagonal=1e32`, `function_tolerance=1e-6`, `gradient_tolerance=1e-10`, `parameter_tolerance=1e-8`, `jacobi_scaling=true`, `use_inner_iterations=false` |
| `Solver::Summary` fields read | only `FullReport()` when `log_solver_summary` | no termination type / cost is consumed by logic |
| `ceres::QuaternionProduct`, `ceres::abs`, `ceres::sqrt/sin/cos` | parameterization functors, slerp | |

Problem sizes: 2D ≈ 3·(nodes+submaps) unknowns, sparse block structure (each residual touches 2 blocks except landmarks/fixed frames); a sparse LM with block-sparse normal equations + sparse Cholesky (e.g. faer/nalgebra-sparse) suffices. Exactness vs Ceres is not required by tests (tolerances ≥ 1e-2 on poses, see §11), but LM damping/accept logic should follow Ceres for comparable convergence.

---------------------------------------------------------------------------------------------------

## 9. Threading & ownership summary

- `MapBuilder` owns `common::ThreadPool thread_pool_(num_background_threads)` (declared before `pose_graph_` ⇒ destroyed after it), `unique_ptr<PoseGraph>`, `unique_ptr<CollatorInterface>`, `vector<unique_ptr<TrajectoryBuilderInterface>>`. Trajectory builders hold raw `PoseGraph2D/3D*` and `CollatorInterface*`.
- `ThreadPool`: N workers (Linux: `nice(10)`), FIFO `deque<shared_ptr<Task>>` of ready tasks + `tasks_not_ready_` map. `Schedule(unique_ptr<Task>) -> weak_ptr<Task>`; the pool owns the task until executed; dependencies on expired weak_ptrs are ignored (already done).
- `Task` states NEW → DISPATCHED → DEPENDENCIES_COMPLETED → RUNNING → COMPLETED; `AddDependency(weak_ptr)` only in NEW; completion notifies dependents.
- Locks (never nested except as listed): PoseGraph `mutex_`; PoseGraph `work_queue_mutex_` (never held while running work items; never held together with `mutex_` except WaitForAllComputations sequentially); ConstraintBuilder `mutex_` (PoseGraph calls into the builder without holding `mutex_`, except `DeleteScanMatcher` from TrimSubmap — **builder mutex nested inside pose-graph mutex**; and `WaitForAllComputations` calls `WhenDone` while holding `mutex_`); `ConnectedComponents::lock_` (inside pose-graph mutex); `Task::mutex_`, `ThreadPool::mutex_`.
- Callback threads: `HandleWorkQueue`, global-SLAM callback, and all work items run on pool threads. `LocalSlamResultCallback` runs on the caller of `AddSensorData` (inside the collator dispatch).
- Raw borrowed pointers crossing threads: constraint tasks hold `const Submap2D*`, `const TrajectoryNode::Data*`, `const SubmapScanMatcher&` (map node reference), and `this`.

Known races/hazards in the C++ (decide explicitly in Rust):
1. `Solve` runs without `mutex_` while foreground getters (`GetTrajectoryData`, `GetImuData`, …) read the optimization problem under `mutex_`; Solve writes `trajectory_data_` (imu_calibration/gravity in place) and node/submap data at the end → data race.
2. `ComputeConstraint` / `ComputeConstraintsForNode` read `optimization_problem_` unlocked (safe only because of the single-worker invariant).
3. Trimming can free a submap/node while a constraint task holds raw pointers (only if a trim races with in-flight searches; HandleWorkQueue runs after WhenDone so normally no in-flight tasks, but `DeleteScanMatcher` while a construction task runs would be use-after-free).
4. `WaitForAllComputations`' `WhenDone` can race with a newly scheduled drain that calls `MaybeAdd*` or a second `WhenDone` (CHECK) if producers add nodes concurrently.
5. `IsTrajectoryFinished/Frozen` read unlocked from public API.

---------------------------------------------------------------------------------------------------

## 10. Map builder & trajectory builder wiring

### 10.1 Data flow
```
user ─AddSensorData(sensor_id, data)─► CollatedTrajectoryBuilder
   ├─ FixedFramePose: if !collate_fixed_frame → wrapped.AddSensorData directly
   ├─ Landmark:       if !collate_landmarks   → wrapped.AddSensorData directly
   └─ else sensor::MakeDispatchable(...) → sensor_collator_->AddSensorData(traj, Data)
        (Collator: global time ordering across all trajectories; TrajectoryCollator if collate_by_trajectory)
        └─► callback HandleCollatedSensorData(sensor_id, data): RateTimer per sensor (log every 15 s),
            data->AddToTrajectoryBuilder(wrapped)   // double dispatch back to typed AddSensorData
GlobalTrajectoryBuilder<LocalTB, PG>:
   RANGE:  CHECK local_tb; m = local_tb->AddRangeData(sensor_id, cloud); null → return
           if m.insertion_result: node_id = pg->AddNode(constant_data, traj, insertion_submaps) (CHECK traj matches)
                                  InsertionResult{node_id, constant_data, submaps (upcast)}
           local_slam_result_callback(traj, m.time, m.local_pose, move(m.range_data_in_local), insertion_result|null)
   IMU:    local_tb?.AddImuData; pg->AddImuData
   ODOM:   CHECK pose valid; local_tb?.AddOdometryData; if motion filter set and IsSimilar(time,pose) → skip pg; else pg->AddOdometryData
   FIXED:  CHECK pose valid if present; pg->AddFixedFramePoseData
   LANDMARK: pg->AddLandmarkData
   LocalSlamResultData: CHECK !local_tb; data->AddToPoseGraph(traj, pg)
```
`CollatedTrajectoryBuilder` ctor: expected sensor id strings = all ids except LANDMARK (if !collate_landmarks) and FIXED_FRAME_POSE (if !collate_fixed_frame); `collator->AddTrajectory(traj, ids, callback)`. `AddLocalSlamResultData` → collated as data.
`MotionFilter::IsSimilar(time, pose)`: `++total`; similar iff `total > 1 && Δt ≤ max_time && |Δt| ≤ max_distance && angle(pose^-1 * last) ≤ max_angle`; otherwise update last and return false.

### 10.2 `MapBuilder`
- Ctor: CHECK 2D xor 3D; build `PoseGraph2D(options.pose_graph_options, make_unique<OptimizationProblem2D>(optimization_problem_options), &thread_pool_)` (or 3D); collator = `TrajectoryCollator` if `collate_by_trajectory` else `Collator`.
- `AddTrajectoryBuilder(expected_sensor_ids, options, callback)`: `traj = trajectory_builders_.size()`; optional `MotionFilter(pose_graph_odometry_motion_filter)`; local builder `LocalTrajectoryBuilder{2D,3D}(options.trajectory_builder_{2d,3d}_options, range sensor ids)` only if that sub-options message is present (else null ⇒ only LocalSlamResultData allowed); wrap: `CollatedTrajectoryBuilder(options, collator, traj, ids, CreateGlobalTrajectoryBuilder{2D,3D}(local, traj, pg, callback, filter))`; trimmer: deprecated `pure_localization` → `PureLocalizationTrimmer(traj, 3)` + WARNING, else `pure_localization_trimmer` → `PureLocalizationTrimmer(traj, max_submaps_to_keep)`; `initial_trajectory_pose` → `pg->SetInitialTrajectoryPose(traj, to_trajectory_id, ToRigid3(relative_pose), FromUniversal(timestamp))`; store `TrajectoryBuilderOptionsWithSensorIds{sensor_ids (ToProto), options}`; return traj.
- `AddTrajectoryForDeserialization(opts)`: push null builder + opts; return new id.
- `FinishTrajectory(traj)`: `collator->FinishTrajectory(traj)` (flushes queued data) then `pg->FinishTrajectory(traj)`.
- `GetTrajectoryBuilder(traj)` → `.at(traj).get()` (throws if out of range; null for deserialized).
- `SubmapToProto(id, response)`: error string if traj ∉ [0, num_builders) or submap null ("…maybe it has been trimmed."); else `submap->ToResponseProto(pose, response)`, return "".
- `SerializeState(include_unfinished, writer)` / `SerializeStateToFile(include_unfinished, filename)` (returns `writer.Close()`) → `io::WritePbStream`.
- `LoadStateFromFile(filename, frozen)`: WARN if suffix ≠ `.pbstream`; `ProtoStreamReader` → `LoadState`.

### 10.3 `WritePbStream` order (format version 2)
1. `SerializationHeader{format_version = 2}`.
2. `SerializedData{pose_graph = pg.ToProto(include_unfinished)}`.
3. `SerializedData{all_trajectory_builder_options}` = options for trajectory ids whose state ≠ DELETED (from `GetTrajectoryStates()`, ascending id).
4. Each submap (`GetAllSubmapData`, skip unfinished unless included): `submap->ToProto(include_probability_grid_data=true)` + submap_id.
5. Each node: `{node_id, node_data = ToProto(*constant_data)}`.
6. Each `GetTrajectoryData()` entry: `{trajectory_id, gravity_constant, imu_calibration (Quaterniond w,x,y,z), fixed_frame_origin_in_map?}`.
7. IMU, odometry, fixed-frame samples per trajectory (`{trajectory_id, data}`).
8. Landmarks: one `LandmarkData` message per observation `{trajectory_id, timestamp, observations[{id, landmark_to_tracking_transform, translation_weight, rotation_weight}]}`.

### 10.4 `LoadState(reader, load_frozen_state)`
1. `ProtoStreamDeserializer` reads header (CHECK version ∈ {1, 2}), then `pose_graph`, then `all_trajectory_builder_options` (required order).
2. For `i` in pose_graph.trajectory: `new_id = AddTrajectoryForDeserialization(all_options.options_with_sensor_ids(i))` (**positional pairing of trajectory i with options i**); `remap[old] = new` (CHECK unique); rewrite trajectory id; if frozen → `pg->FreezeTrajectory(new_id)` (enqueued before any submap/node).
3. Remap constraint submap/node trajectory ids (`.at` ⇒ throws on unknown).
4. `submap_poses[SubmapId]` and `node_poses[NodeId]` from the pose-graph proto (global poses).
5. Every `landmark_poses` → `pg->SetLandmarkPose(id, pose, frozen=true)` (**always frozen**, regardless of `load_frozen_state`).
6. 3D: CHECK format version ≠ 1 ("use pbstream migrate").
7. Stream loop over `SerializedData`: duplicate pose_graph/options → LOG(ERROR) and continue; `submap` → remap, `AddSubmapFromProto(submap_poses.at(id), submap)`; `node` → remap, `AddNodeFromProto(node_poses.at(id), node)`; `trajectory_data` → remap, `SetTrajectoryDataFromProto`; `imu/odometry/fixed_frame/landmark` → skipped if frozen, else remapped `pg->Add*Data(…, sensor::FromProto(...))`; unknown → WARNING.
8. Frozen: for INTRA constraints only → `AddNodeToSubmap(node, submap)` (no constraints loaded at all). Not frozen: `AddSerializedConstraints(FromProto(constraints))`.
9. CHECK `reader->eof()`; return remap.

### 10.5 Local SLAM result replay (`SubmapController`, `LocalSlamResult2D/3D`)
- `LocalSlamResultData : sensor::Data { Time GetTime(); virtual void AddToPoseGraph(int traj, PoseGraph*) const = 0; }`; time = `FromUniversal(proto.timestamp)`.
- `SubmapController<SubmapType>::UpdateSubmap(proto::Submap)`: if id not in `unfinished_submaps_`: `CreateSubmap` (returns null with log unless `num_range_data == 1`; 2D uses own `ValueConversionTables`); insert if non-null; return. Else `UpdateFromProto(proto)`; if now `insertion_finished()` → remove from map; else `set_num_range_data(0)`; return ptr.
- `LocalSlamResult2D::AddToPoseGraph`: CHECK ≥1 submap and first has `submap_2d`; collect non-null `UpdateSubmap` results ("Ignoring submap" otherwise); if none → "Ignoring node", return; `PoseGraph2D::AddNode(make_shared(FromProto(node_data)), traj, submaps)`.
- `LocalSlamResult3D::AddToPoseGraph`: same but **does not filter null submaps** (pushes nulls) — port should mirror 2D behaviour or document.
- `AddToTrajectoryBuilder(tb)`: `tb->AddLocalSlamResultData(make_unique copy of *this)`.

### 10.6 `RangeDataCollator` (used by LocalTrajectoryBuilder; in scope by request)
State: `expected_sensor_ids_` (set), `id_to_pending_data_` (map sensor→one TimedPointCloudData), `current_start_ = current_end_ = Time::min()`, `kDefaultIntensityValue = 0`.
`AddRangeData(sensor_id, data)`: CHECK expected; resize intensities to ranges size (fill 0). If sensor already pending: `current_start_ = current_end_; current_end_ = pending[sensor].time`; `result = CropAndMerge()`; then insert new data (emplace — the pending entry was erased by CropAndMerge because it's fully consumed up to its own time); return result. Else insert; if not all sensors pending → return `{}` (time default, empty). Else `current_start_ = current_end_; current_end_ = min pending time`; return `CropAndMerge()`.
`CropAndMerge()`: result `{time = current_end_, ranges, origins}`. Per pending sensor (map order): `overlap_begin` = first point with `data.time + p.time >= current_start_`; `overlap_end` = first point with `data.time + p.time > current_end_`; warn once if points dropped before begin; if overlap non-empty: push `origin` (index = origins.size()), copy points with `time += ToSeconds(data.time − current_end_)` and intensity, `origin_index`; if `overlap_end == end` erase sensor; else if `overlap_end == begin` keep; else keep tail `[overlap_end, end)` with same `time`/origin. Finally `std::sort` (unstable; ties may reorder) of all merged points by relative point time. Note: if a re-sent sensor's pending entry were not fully consumed (points with positive relative time), the subsequent `emplace` would silently keep the OLD data.

---------------------------------------------------------------------------------------------------

## 11. Connectivity

`ConnectedComponents` (thread-safe, `absl::Mutex lock_`): `forest_: map<int,int>` (union-find, path compression in `FindSet`, CHECKs presence), `connection_map_: map<pair<int,int> (minmax), int>`.
- `Add(t)`: `forest_.emplace(t,t)`. `Connect(a,b)`: emplace both, `forest[rep(a)] = rep(b)` (no rank), `++count[minmax]`.
- `TransitivelyConnected(a,b)`: a==b → true (no lock); either untracked → false; else reps equal.
- `Components()`: group by rep (flat_hash_map ⇒ unordered output). `GetComponent(t)`: all ids with same rep (**CHECK-fails if t untracked**). `ConnectionCount(a,b)` (0 if absent).
- `ToProto(components)`: sort each component, sort list of components lexicographically.
`TrajectoryConnectivityState` (thread-compatible): `Add`, `Connect(a,b,time)`: if already transitively connected → `last[minmax] = max(last, time)`; else for every `x ∈ comp(a)`, `y ∈ comp(b)`: `last[minmax(x,y)] = time` (overwrites); then `components.Connect(a,b)`. **Connect with an untracked id CHECK-fails in GetComponent** (callers always Add first). `LastConnectionTime(a,b)` → `last[minmax]` (inserts default `Time()` = universal epoch 0 ⇒ non-const). `TransitivelyConnected`, `Components` delegate.

---------------------------------------------------------------------------------------------------

## 12. Tests (all in scope)

Common Rust test infra needed: deterministic `std::mt19937` + `uniform_real_distribution` (seeded), Lua config resolution of `configuration_files/*.lua` (or equivalent defaults), fake protos (`testing/test_helpers`), `FakeTrimmable`, mock optimization problem (no-op Solve), thread pool with `WaitUntilIdle` (`ThreadPoolForTesting`).

### 12.1 `mapping/pose_graph_test.cc`
| Test | Checks |
|---|---|
| `PoseGraph.SerializeConstraint` | `CreateFakeConstraint(CreateFakeNode(1,2), CreateFakeSubmap3D(2,3))` → `FromProto` → `ToProto` equals original (MessageDifferencer). Fake constraint: relative pose t=(2,3,4), R=AngleAxis(π/8, X), translation_weight 0.2f, rotation_weight 0.1f, tag INTER. |

### 12.2 `mapping/pose_graph_trimmer_test.cc`
| `PureLocalizationTrimmerTest.MarksSubmapsAsExpected` | traj 42, keep 15, FakeTrimmable with 17 submaps → trimmed exactly [(42,0),(42,1)]. |

### 12.3 `internal/2d/pose_graph_2d_test.cc` (fixture `PoseGraph2DTest`)
Setup: `ThreadPool(1)`; point cloud: for `t` in `[0, 2π)` step 0.005f: `r = (sin(20t)+2)·sin(t+2)`, point `(r sin t, r cos t, 0)`; `ActiveSubmaps2D` with `num_range_data=1`, PROBABILITY_GRID res 0.05, inserter hit 0.53 miss 0.495 insert_free_space; pose graph options: optimize_every_n_nodes 1000, sampling_ratio 1, max_constraint_distance 6, min_score 0.5, global_localization_min_score 0.6, loop closure weights 1/1, FastCSM2D {linear 3, angular 0.1, depth 3}, Ceres2D {occ 20, trans 10, rot 1, nonmono true, 50 iters, 1 thread}, matcher weights 1/1, optimization {acceleration 1, rotation 1e2, huber 1, local_slam 0/0, odometry 0/0, fixed frame 1e1/1e2, ceres {false, 200, 1}}, max_num_final_iterations 200, global_sampling_ratio 0.01, global search after 10 s.
`MoveRelativeWithNoise(movement, noise)`: `current = current * movement`; cloud transformed by `Embed3D(current^-1)`; `pose_estimate = noise * current`; insert range data transformed by pose_estimate; `AddNode(Data{time=FromUniversal(0) (all nodes same time!), gravity=Identity, filtered cloud = returns, {}, {}, {}, local_pose = Embed3D(pose_estimate)}, traj 0, active submaps)`.
| Test | Input | Expectation / tolerance |
|---|---|---|
| `EmptyMap` | RunFinalOptimization | no nodes |
| `NoMovement` | 3× identity moves | traj ids == {0}; 3 nodes; each global pose ≈ Identity (`IsNearly`, 1e-2) |
| `NoOverlappingNodes` | rng(0), 4 moves `({0.25·U(-1,1), 5}, 0)` | each `poses[i]` ≈ `Project2D(global_pose)` within 1e-2 |
| `ConsecutivelyOverlappingNodes` | 5 moves `({0.25·U, 2}, 0)` | same, 1e-2 |
| `OverlappingNodes` | 5 moves `({0.15·U, 0.4}, 0)` with noise `({0.1U, 0.1U}, 0.1U)` (draw order: nx, ny, nθ, then movement x) | optimized first→last relative error < pre-optimization error in both |angle| and |translation| |

### 12.4 `internal/3d/pose_graph_3d_test.cc` (fixture with `ThreadPool(1)`, options from `pose_graph.lua` defaults; `PoseGraph3DForTesting` exposes `WaitForAllComputations`; `MockOptimizationProblem3D` = OptimizationProblem3D with default options and mocked no-op `Solve`)
| Test | Checks |
|---|---|
| `Empty` | after WaitForAllComputations all getters empty (`GetAllSubmapData`, `constraints`, `GetConnectedTrajectories`, `GetAllSubmapPoses`, `GetTrajectoryNodes`, `GetTrajectoryNodePoses`, `GetTrajectoryData`), `ToProto(true)` == empty proto |
| `BasicSerialization` | AddNodeFromProto(Identity, fake node (1,1)), AddSubmapFromProto(Identity, fake submap (1,1)), AddSerializedConstraints(fake constraint), SetLandmarkPose("landmark_id", Identity); `ToProto(true)` equals the hand-built proto (constraint, node{index, timestamp 42, pose}, submap{index, pose}, trajectory, landmark, whole message) |
| `SerializationWithUnfinishedSubmaps` | nodes (1,1..3), submaps (1,1) finished + (1,2) unfinished, constraints n1–s1, n2–s1, n2–s2, n3–s2; `ToProto(false)`: 2 constraints, 1 trajectory, 3 nodes, 1 submap; first two constraints/nodes/submap equal expected |
| `PureLocalizationTrimmer` | mock solve; traj 2; 5 submaps indices {42,43,44,104,105}; 2 nodes each, node index `7 + 2·submap_index + j`; one INTRA constraint per node; trimmer keep 3. After wait: 5 submaps (poses & data), 10 nodes (& poses), 10 constraints. After each of 2× RunFinalOptimization: 3 submaps, 6 nodes, 6 constraints |
| `EvenSubmapTrimmer` | custom trimmer trims even submap indices; 20 submaps (42..61), 3 nodes each (`7 + 3i + j`), INTRA constraints. Before: 20/60/60; after each of 2 RunFinalOptimization: 10 submaps, 30 nodes, 30 constraints |
| `EvenSubmapTrimmerOnFrozenTrajectory` | same data but `AddNodeToSubmap` instead of constraints, then FreezeTrajectory, AddTrimmer: before 20/60 and 0 constraints; after: 10/30, 0 constraints |

### 12.5 `internal/2d/overlapping_submaps_trimmer_2d_test.cc` (FakeTrimmable; square submaps: res 1.0, `max = corner`, num_cells², all cells value 1, known box full, `num_range_data=1`)
| Test | Setup | Expected trimmed |
|---|---|---|
| `EmptyPoseGraph` | trimmer(1, 0, 0) | none |
| `TrimOneOfTwoOverlappingSubmaps` | 2 identical 1-cell submaps at identity, nodes t=1000/2000, INTRA s0–n0, s1–n1 | [(0,0)] |
| `TestMinAddedSubmapsCountParam` | trimmer(1,0,2): with 2 submaps → none; add 3rd (n2 t=3000) and Trim again | [(0,0),(0,1)] |
| `DoNotTrimUnfinishedSubmap` | submap 0 unfinished | none |
| `UseOnlyIntraSubmapsToComputeFreshness` | INTER s0–n0, INTRA s0–n2(t=3000), INTRA s1–n1(t=2000) | [(0,1)] |
| `TrimSubmapWithLowCoveredCellsCount` | two 2×2 submaps, second global-translated (1,1); trimmer(1, 4, 0) | [(0,0)] |
| `TestTransformations` | second submap global=local=((1,1), π/2) | [(0,0)] |

### 12.6 Constraint builder tests (`constraint_builder_{2d,3d}_test.cc`; options from pose_graph.lua with sampling_ratio 1, min_score 0, global min 0; 3D also `min_low_resolution_score = 0`, `min_rotational_score = 0`; `ThreadPoolForTesting`)
| Test | Checks |
|---|---|
| `ConstraintBuilder{2D,3D}Test.CallsBack` | finished nodes 0; NotifyEndOfNode; WhenDone → callback invoked once with empty result; then finished nodes 1 |
| `…FindsConstraints` | node data: one point (0.1,0.2,0.3), identity gravity & local pose (3D: same point in high & low res clouds, histogram Zero(3)); 2D submap: origin (4,5), ProbabilityGrid MapLimits(1.0, max (2,3), cells 100×110) (empty grid); 3D: `Submap3D(0.1, 0.1, Identity, Zero(3))`. Loop 2×: 2× MaybeAddConstraint(identity rel / identity poses) + 1× MaybeAddGlobalConstraint; NotifyEndOfNode, wait, finished == ++expected; NotifyEndOfNode again, wait, ++expected; WhenDone → result size 3, all INTER_SUBMAP; wait; DeleteScanMatcher(submap). (Score threshold 0 with empty grid still "matches" because CHECK is `score > 0` — FastCSM must return a strictly positive min-probability score on an unknown grid.) |

### 12.7 Optimization tests
| Test | Checks |
|---|---|
| `OptimizationProblem3DTest.ReducesNoise` | options: accel 2e-5, rot 1e-3, huber 1, local slam/odom 1e-2, fixed frame 1e1/1e2, log summary, online imu extrinsics, fix_z false, ceres {false, 200 iters, 4 threads}; rng 45387. 100 nodes: ground truth `RandomTransform(10, 3)` (uniform t ∈ [-10,10]³, angle-axis ∈ [-3,3]³), noise `RandomYawOnlyTransform(0.2, 0.3)`; `AddNoise(T, N) = (T.t + N.t, N.q * T.q)`. Per node at t = 0, 0.1, …: IMU `{t, accel (0,0,9.81), ω 0}`, node `{t, noisy, noisy}`. Constraints per node: to submap 0 `{noisy, 1, 1}`, to submap 1 `{AddNoise(gt, RandomYawOnly(0.2,0.3)), 1, 1}`, to submap 2 `{Rz(π)^-1 * gt * RandomTransform(1e3, 3), 1e-9, 1e-9}`. Submaps: Identity, Identity, Rz(π). Solve (ACTIVE). Sum of translation errors and rotation angles after < 0.8 × before (both). |
| `SpaCostFunction2DTest.CompareAutoDiffAndAnalytical` | zbar = (t (1,1,1), q (1,1,−1,−1) **unnormalized**), wt 1, wr 10; start (1,1,1), end (10,1,100); residuals & jacobians equal within 1e-5 |
| `…EvaluateAnalyticalCost` / `…EvaluateAutoDiffCost` | residuals (−3.86272, 8.57324, −6.83333); J_start (0.540302, 0.841471, 7.57324, −0.841471, 0.540302, 4.86272, 0, 0, 10); J_end (−0.540302, −0.841471, 0, 0.841471, −0.540302, 0, 0, 0, −10); tolerance 1e-5 |
| `LandmarkCostFunctionTest.SmokeTest` (2D) | prev t=0, next t=10 (universal ticks), obs t=5, landmark_to_tracking = translation (1,1,1), wt 1, wr 2; prev (2,0,0), next (0,2,0), landmark q identity t (1,2,1) → residuals exactly (1,0,0,0,0,0) (DoubleEq) |
| `LandmarkCostFunction3DTest.SmokeTest` | same obs; prev pose identity at origin, next identity at (2,2,2), landmark t (1,2,2) → (1,0,0,0,0,0) |

### 12.8 Connectivity tests
`TrajectoryConnectivityStateTest`: `UnknownTrajectory` (Add 0; not connected to 1; LastConnectionTime(0,1) == Time()), `NotConnected`, `Connected` (Connect at FromUniversal(123456) → connected, time 123456), `UpdateConnectionTime` (123456 then 234567 → 234567; later 123456 doesn't lower it), `JoinTwoComponents` (0–1 and 2–3 at 123456, then 0–2 at 234567: (0,2),(0,3),(1,3) = 234567; (0,1) stays 123456).
`ConnectedComponentsTest`: `TransitivelyConnected` (10×10 reflexive-only initially, then 0–1, 8–9, 1–8 chain), `EmptyConnectedComponents` (0 components), `ConnectedComponents` (0..4 and 5..9 → two clusters, any order), `ConnectionCount` (10× Connect(0,1) → count increments, order-invariant; other pairs 0), `ReflexiveConnectivity` (0,0 connected and count 0 before/after Add).

### 12.9 `internal/range_data_collator_test.cc`
`CreateFakeRangeData(from, to, intensities)`: time = `to`, origin (0,1,2), 10 points `(1, 2, f)` with `f = i/9`, relative time `(1−f)·(−duration)` (duration = to−from in seconds; universal ticks are 100 ns), intensity = z if requested.
| Test | Expectations |
|---|---|
| `SingleSensor` | (200,300): time 300, 1 origin, 10 pts, sorted; (300,500): time 500, 10 pts, first point abs time ≈ 300 (±2 ticks); (−1000,510): time 510, 1 pt with time 0 |
| `SingleSensorEmptyData` | empty data at 300 → time 300, 0 pts; then as above |
| `TwoSensors` | s0 (200,300) → 0 pts; s1 (−1000,310) → 2 origins, time 300, 19 pts, first ≈ −1000, last time 0; s0 (300,500) → time 310, 2 origins, 2 pts, first ≈ 300, last 0; s0 (600,700) → time 500, total pts over outputs 1–3 = 30, last 0 |
| `ThreeSensors` | (100,200), (199,250) → 0 pts each; s2 (210,300) → 11 pts; s2 (400,500) → total 30 |
| `ThreeSensorsWithIntensities` | same + intensity == z (1e-6) |

### 12.10 `mapping/map_builder_test.cc`
Config: `map_builder.lua` with 2D, `optimize_every_n_nodes = 0`, `global_sampling_ratio 0.05`, `global_constraint_search_after_n_seconds 0`; trajectory builder `use_imu_data = false`, `submaps.num_range_data = 4` (2D & 3D). Fake data: `GenerateFakeRangeMeasurements(1.2 m, 4 s, 0.1 s)`: semicircle radius 5, angles [0, π) step 0.01, heights {−0.4,−0.2,0,0.2,0.4}, travel along normalize(2,1,0), times FromUniversal(123)+elapsed, cloud = `TransformTimedPointCloud(points, pose^-1)`, origin 0. Helpers: `SetOptionsTo3D`; `SetOptionsToTSDF2D` (TSDF inserter + grid, local ceres occupied 10, constraint builder ceres occupied 50); `SetOptionsEnableGlobalOptimization` (optimize_every_n_nodes 3, motion filter max_distance 0).
| Test (params) | Expectations |
|---|---|
| `TrajectoryAddFinish` (PG2D, PG3D, TSDF2D) | 1 builder, non-null builder & pose graph; Finish + RunFinalOptimization → IsTrajectoryFinished |
| `LocalSlam2D` (PG, TSDF) | #callbacks == #measurements; |last.t − first.t| ≈ 1.2 ± 0.12 |
| `LocalSlam3D` | + IMU (0,0,9.8) per measurement; same |
| `GlobalSlam2D` (PG, TSDF) | + ≥50 constraints, contains INTER, ≥20 nodes, ≥5 submaps, |GetLocalToGlobal * last local|.t ≈ 1.2 ± 0.12 |
| `GlobalSlam3D` | ≥10 constraints, INTER present, ≥5 nodes, ≥2 submaps, final ≈ 1.2 ± 0.12 |
| `DeleteFinishedTrajectory2D` (PG, TSDF) | finished, ≥50 constraints, ≥20 nodes, ≥5 submaps; DeleteTrajectory; second trajectory with +100 s; after optimization first traj has 0 nodes/submaps; delete second → 0 constraints, 0 nodes/submaps |
| `SaveLoadState` (PG2D, PG3D, TSDF2D) | 2D use_imu_data=true + IMU; serialize to `temp-SaveLoadState.pbstream` (include unfinished), new MapBuilder, LoadState(not frozen), RunFinalOptimization: same #constraints, remap size 1, same #nodes |
| `LocalizationOnFrozenTrajectory2D` (PG, TSDF) | build & save map; new builder with global optimization, LoadState(frozen); new trajectory with data shifted by `frozen_to_global = ((0.5,0.4,0), Rz(1.2))` and +100 s; ≈ travel ± 0.15·1.2; ≥50 constraints; ≥3 cross-trajectory constraints; INTER present; ≥20 nodes; ≥5 submaps; |GetLocalToGlobal.t| ≈ |frozen_to_global.t| ± 0.1; final global pose within 0.3 of `frozen_to_global * T(travel)` |

Untested in scope (port must add own tests): `OptimizationProblem2D` (no direct test), `SpaCostFunction3D`, IMU cost functions, `CeresPose`, `CollatedTrajectoryBuilder`, `GlobalTrajectoryBuilder`, `SubmapController`, `LocalSlamResult*`, `work_queue`.

Test helpers (`internal/testing`): `ResolveLuaParameters(code)` (resolver over `configuration_files`), `GenerateFakeRangeMeasurements` (2 overloads), `CreateFakeSubmap3D(traj=1, idx=1, finished=true)` (num_range_data 1, identity local pose), `CreateFakeNode(traj=1, idx=1)` (timestamp 42, identity local pose), `CreateFakeConstraint(node, submap)`, `CreateTrajectoryIfNeeded`, `CreateFakeLandmark(id, pose)`, `AddToProtoGraph` overloads (node pose = node local pose; submap pose = submap local pose). Mocks: `MockMapBuilder`, `MockPoseGraph` (PoseGraphInterface only), `MockTrajectoryBuilder` (`DoAddLocalSlamResultData` shim). `FakeTrimmable(traj, n)`: `num_submaps` = total − trimmed (ignores traj), `GetSubmapIds` returns all (including trimmed), `TrimSubmap` records only, `IsFinished` false, `SetTrajectoryState` no-op.

---------------------------------------------------------------------------------------------------

## 13. Rust porting notes & pitfalls

Architecture recommendation:
- `PoseGraph<D: Dim>` generic over 2D/3D (trait for spec types, submap type, constraint frame conversions, constraint builder, optimization problem) — the two C++ files are ~95% identical; encode the §3.9 differences in the trait.
- Split state: `Arc<Mutex<PoseGraphData>>` (+ trimmers, samplers, counter) for foreground-visible data; the optimization problem should be owned by the **single worker** (e.g. `Mutex<OptimizationProblem>` held by the worker only, or moved into a dedicated worker thread). Expose read-only snapshots of what getters need (imu/odom/fixed-frame/trajectory data) by copying into `PoseGraphData` after each Solve, eliminating race §9.1.
- Replace the drain-on-thread-pool + `work_queue_ == null` sentinel with an explicit worker state machine: `enum WorkerState { Idle, Draining, AwaitingConstraints }` under the queue mutex; `AddWorkItem` spawns a drain only from `Idle`. Keep FIFO semantics and the "stop draining at kRunOptimization until the constraint builder is done" rule — ordering of work items relative to optimizations is observable in tests (trimming happens only after an optimization; `RunFinalOptimization` relies on item A/B ordering).
- Work items: `Box<dyn FnOnce(&Worker) -> WorkResult + Send>`; avoid capturing `self` raw pointers — pass a context handle (`Arc<Inner>`).
- Constraint builder: replace the Task DAG with (a) a rayon/crossbeam pool, (b) per-node `AtomicUsize` pending counters or a `Vec<JoinHandle>` per node, (c) result slots `Arc<Mutex<Option<Constraint>>>` or a channel; `WhenDone` = barrier on all outstanding jobs issued since the previous `WhenDone`, then callback on a pool thread. Preserve: results ordered by issue order (deque order), `num_finished_nodes` incremented per node only after all that node's jobs finish, scan matcher construction memoized per submap (`Arc<OnceLock<FastCSM>>` keyed by SubmapId) and dropped by `DeleteScanMatcher`.
- Lifetimes: C++ passes raw `Submap*`/`Data*` to background jobs; in Rust pass `Arc<Submap2D>`/`Arc<TrajectoryNodeData>` (already shared_ptr in C++), removing use-after-free hazards on trim.
- Callbacks (`GlobalSlamOptimizationCallback`, `LocalSlamResultCallback`, collator callbacks): `Box<dyn Fn(..) + Send + Sync>`; always invoke **without** holding pose-graph locks (C++ does so); beware re-entrancy (a callback calling `AddNode` must not deadlock).
- `CollatedTrajectoryBuilder` ↔ collator ↔ `GlobalTrajectoryBuilder` is a cycle via callbacks capturing `this`; model as: collator owns per-trajectory sink `Box<dyn FnMut(SensorId, Data)>` which owns (or `Arc`s) the GlobalTrajectoryBuilder; `sensor::Data::AddToTrajectoryBuilder` double dispatch → a Rust `enum SensorData` + `match`.
- `MapById`: implement with `BTreeMap<i32, Trajectory{ data: BTreeMap<i32,T>, can_append: bool }>`; replicate Append/Insert/Trim/`lower_bound` exactly (index reuse after full trim!). Many algorithms depend on ordered iteration and `prev(end)`.
- Time: `common::Time` = i64 ticks of 100 ns since 0001-01-01 (universal); `Time()` default = 0 ticks; `FromUniversal(0)` equals the default. Use the same representation; LastConnectionTime default = epoch.
- Determinism: `FixedRatioSampler` is deterministic; keep per-submap / per-trajectory samplers and their creation timing identical (they're created lazily on first pulse) to reproduce loop-closure candidate selection.
- 2D gravity alignment: store constraints gravity-aligned internally and convert at the API boundary exactly as C++ (`constraints()` multiplies by `Rot(g)`, `AddSerializedConstraints` by `Rot(g^-1)`); serialized pbstreams contain non-aligned constraints.
- Angle normalization: `NormalizeAngleDifference` inside autodiff (piecewise, derivative 1); `Rigid2d::normalized_angle` on `FromPose`.
- LOG(FATAL)/CHECK semantics: decide per case between `panic!` (invariants: MapById misuse, state machine CHECKs used by tests) and `Result`/warn (`CanAddWorkItemModifying` FATALs are intended to become warnings). Documented crash cases to consider hardening: 3D optimization without IMU data and `!fix_z_in_3d`; IMU starting after first node (IntegrateImu CHECK); landmark observation on a trajectory with 0 or 1 nodes (UB); `TrajectoryConnectivityState::Connect` with untracked id; `GetTrajectoryBuilder` out-of-range; `LoadState` options/trajectory positional mismatch (deleted or empty trajectories make `pose_graph.trajectory` and `all_trajectory_builder_options` disagree); `LocalSlamResult3D` pushing null submaps.
- Unsigned arithmetic in `OverlappingSubmapsTrimmer2D` (`size_t - uint16` wrap, `double → uint16` truncation of min covered cells) — replicate with `wrapping_sub`/`as u16` if bit-exact behaviour is wanted.
- `PoseGraph::ToProto` orphaned-node logic is dead code; nodes are emitted even if all their constraints were dropped — keep for format parity.
- 3D `fix_z_in_3d=false` ⇒ **no odometry and no local-SLAM-pose terms** (only IMU terms); this asymmetry is intentional in upstream — don't "fix" it silently.
- `SetMaxNumIterations` mutates the optimization problem's stored options; final-iteration bump and restore happen as two separate work items.
- Ceres parameter blocks are raw `double*` into containers that must not reallocate during Solve (`MapById` of arrays, `std::map<int, TrajectoryData>` for imu_calibration/gravity). In Rust build the problem from a flat `Vec<f64>` state with block index maps, then write back.
- Quaternion conventions: Ceres arrays are (w,x,y,z); Eigen/nalgebra storage is (x,y,z,w) — convert explicitly at every boundary (`CeresPose::FromPose`, `Rigid3d::FromArrays`, `imu_calibration`, proto `Quaterniond`).
- Metrics are global statics in C++; make them optional injected handles.
