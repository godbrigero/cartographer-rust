# Architecture report: `cartographer/sensor/`

Phase 0 porting reference. Source (read-only): `cartographer/cartographer/sensor/` (21 headers/sources, 2 protos, 8 test files).
Namespace: `cartographer::sensor` (protos: `cartographer.sensor.proto`). Everything here is **thread-compatible** (no internal locking that matters), synchronous, and allocation-heavy (value semantics, copies everywhere).

Also covered (appendix A) because it was requested and is tightly coupled: `mapping/internal/range_data_collator.{h,cc}` (it lives in `mapping/`, not `sensor/`).

---

## 0. File inventory and dependency graph

| File | Kind | Depends on (in-repo) |
|---|---|---|
| `rangefinder_point.h` | header-only | `transform/transform.h`, `proto/sensor.pb.h` |
| `point_cloud.{h,cc}` | lib | rangefinder_point, `transform/rigid_transform.h` |
| `range_data.{h,cc}` | lib | point_cloud, compressed_point_cloud (include only, unused) |
| `compressed_point_cloud.{h,cc}` | lib | point_cloud, **`mapping/3d/hybrid_grid.h`** (reverse dep sensor -> mapping!), `common/math.h`/`port.h` |
| `imu_data.{h,cc}`, `odometry_data.{h,cc}`, `fixed_frame_pose_data.{h,cc}`, `landmark_data.{h,cc}`, `timed_point_cloud_data.{h,cc}` | lib | `common/time.h`, `transform/transform.h` |
| `data.h` | header-only | `common/time.h`; fwd-decl `mapping::TrajectoryBuilderInterface`. NB header guard is `CARTOGRAPHER_MAPPING_DATA_H_` (copy/paste quirk) |
| `collator_interface.h` | header-only | data.h, absl flat_hash_set/optional |
| `map_by_time.h` | header-only template | `mapping/id.h` (`MapById`, `NodeId`, `Range`) |
| `internal/dispatchable.h` | header-only template | `mapping/trajectory_builder_interface.h` |
| `internal/ordered_multi_queue.{h,cc}` | lib | `common/internal/blocking_queue.h`, dispatchable |
| `internal/collator.{h,cc}` | lib | ordered_multi_queue |
| `internal/trajectory_collator.{h,cc}` | lib | ordered_multi_queue, `metrics/counter.h`, `metrics/family_factory.h` |
| `internal/voxel_filter.{h,cc}` | lib | point_cloud, timed_point_cloud_data, `common/lua_parameter_dictionary.h`, adaptive_voxel_filter_options.pb |
| `internal/test_helpers.h` | test-only | gmock |
| `proto/sensor.proto`, `proto/adaptive_voxel_filter_options.proto` | proto3 | `transform/proto/transform.proto` |

Main consumers: `mapping/internal/collated_trajectory_builder.h` (wraps every sensor type with `MakeDispatchable`), `mapping/map_builder.cc` (`collate_by_trajectory ? TrajectoryCollator : Collator`), `mapping/internal/{2d,3d}/local_trajectory_builder_*.cc` (VoxelFilter / AdaptiveVoxelFilter), pose graphs (`MapByTime<ImuData|OdometryData|FixedFramePoseData>`), `cloud/` handlers. `mapping::LocalSlamResultData` is the only other `sensor::Data` subclass.

---

## 1. Protos

### `proto/sensor.proto` (proto3, `option java_outer_classname = "Sensor"`)

| Message | Fields (number: type name) |
|---|---|
| `RangefinderPoint` | 1: `transform.proto.Vector3f position` |
| `TimedRangefinderPoint` | 1: `Vector3f position`; 2: `float time` |
| `CompressedPointCloud` | 1: `int32 num_points`; 3: `repeated int32 point_data` (no field 2) |
| `TimedPointCloudData` | 1: `int64 timestamp`; 2: `Vector3f origin`; 3: `repeated Vector4f point_data_legacy`; 4: `repeated TimedRangefinderPoint point_data`; 5: `repeated float intensities` |
| `RangeData` | 1: `Vector3f origin`; 2: `repeated Vector3f returns_legacy`; 3: `repeated Vector3f misses_legacy`; 4: `repeated RangefinderPoint returns`; 5: `repeated RangefinderPoint misses` |
| `ImuData` | 1: `int64 timestamp`; 2: `Vector3d linear_acceleration`; 3: `Vector3d angular_velocity` |
| `OdometryData` | 1: `int64 timestamp`; 2: `Rigid3d pose` |
| `FixedFramePoseData` | 1: `int64 timestamp`; 2: `Rigid3d pose` (presence significant) |
| `LandmarkData` | 1: `int64 timestamp`; 2: `repeated LandmarkObservation landmark_observations` |
| `LandmarkData.LandmarkObservation` | 1: `bytes id`; 2: `Rigid3d landmark_to_tracking_transform`; 3: `double translation_weight`; 4: `double rotation_weight` |

Referenced transform protos: `Vector3f{float x=1,y=2,z=3}`, `Vector3d{double x,y,z}`, `Vector4f{float x=1,y=2,z=3,t=4}`, `Quaterniond{double x=1,y=2,z=3,w=4}`, `Rigid3d{Vector3d translation=1; Quaterniond rotation=2}`.

`timestamp` = `common::ToUniversal(time)`: int64 ticks of 100 ns since the Universal Time Scale epoch (0001-01-01).

### `proto/adaptive_voxel_filter_options.proto` (proto3)
`AdaptiveVoxelFilterOptions { float max_length = 1; float min_num_points = 2; float max_range = 3; }`
NB `min_num_points` is a **float** in the proto even though Lua reads it with `GetNonNegativeInt`; comparisons are `size_t` vs `float`.

---

## 2. Per-file API and semantics

### 2.1 `rangefinder_point.h`
```cpp
struct RangefinderPoint      { Eigen::Vector3f position; };
struct TimedRangefinderPoint { Eigen::Vector3f position; float time; };
template<class T> RangefinderPoint      operator*(const transform::Rigid3<T>&, const RangefinderPoint&);      // position = lhs*position
template<class T> TimedRangefinderPoint operator*(const transform::Rigid3<T>&, const TimedRangefinderPoint&); // time copied unchanged
bool operator==(RangefinderPoint, RangefinderPoint);           // exact float ==
bool operator==(TimedRangefinderPoint, TimedRangefinderPoint); // exact position and time
RangefinderPoint      FromProto(const proto::RangefinderPoint&);
proto::RangefinderPoint ToProto(const RangefinderPoint&);
TimedRangefinderPoint FromProto(const proto::TimedRangefinderPoint&);
proto::TimedRangefinderPoint ToProto(const TimedRangefinderPoint&);
RangefinderPoint      ToRangefinderPoint(const TimedRangefinderPoint&);          // drops time
TimedRangefinderPoint ToTimedRangefinderPoint(const RangefinderPoint&, float time);
```
- Rigid3<T> * Vector3f: in practice only `Rigid3f` is used (T=float).

### 2.2 `point_cloud.{h,cc}`
```cpp
class PointCloud {
  using PointType = RangefinderPoint;
  PointCloud(); explicit PointCloud(std::vector<PointType>);
  PointCloud(std::vector<PointType>, std::vector<float> intensities); // CHECK_EQ sizes iff intensities non-empty
  size_t size() const; bool empty() const;
  const std::vector<PointType>& points() const; const std::vector<float>& intensities() const;
  const PointType& operator[](size_t) const;   // unchecked
  ConstIterator begin()/end() const;           // iterates points only
  void push_back(PointType);                    // appends point ONLY (does not touch intensities)
  template<class Pred> PointCloud copy_if(Pred) const;
 private: std::vector<PointType> points_; std::vector<float> intensities_; // optional, same size or empty
};
using TimedPointCloud = std::vector<TimedRangefinderPoint>;
struct PointCloudWithIntensities { TimedPointCloud points; std::vector<float> intensities; }; // legacy, for cartographer_ros
PointCloud      TransformPointCloud(const PointCloud&, const transform::Rigid3f&);           // keeps intensities
TimedPointCloud TransformTimedPointCloud(const TimedPointCloud&, const transform::Rigid3f&); // keeps time
PointCloud      CropPointCloud(const PointCloud&, float min_z, float max_z);                // keeps min_z <= z <= max_z (inclusive both)
```
- `copy_if`: keeps points where predicate true, preserving order; copies matching intensities iff `intensities_` non-empty. Result built via 2-arg ctor (CHECK passes).
- TimedPointCloud time convention: seconds, relative to the **last** point (last point time 0, earlier points negative), increasing. 2D points have z = 0.
- Pitfall: `push_back` on a cloud with intensities silently breaks the size invariant (no CHECK). Rust: either forbid (make `push` only valid when intensities empty / debug_assert) or keep the same unchecked behavior.

### 2.3 `range_data.{h,cc}`
```cpp
struct RangeData { Eigen::Vector3f origin; PointCloud returns; PointCloud misses; };
RangeData TransformRangeData(const RangeData&, const transform::Rigid3f&); // origin, returns, misses all transformed
RangeData CropRangeData(const RangeData&, float min_z, float max_z);      // origin unchanged; returns & misses cropped (inclusive)
proto::RangeData ToProto(const RangeData&);
RangeData FromProto(const proto::RangeData&);
```
- `ToProto`: writes `origin`, `returns` (field 4), `misses` (field 5) only; never writes legacy fields. **Intensities are dropped.**
- `FromProto`: for each of returns/misses independently: if new field non-empty use it, else read legacy `Vector3f` list. Resulting PointClouds have no intensities.
- `misses`: points along rays without a return, inserted at a configured distance; free space assumed between origin and misses.

### 2.4 `compressed_point_cloud.{h,cc}` (lossy, 1 mm grid)
```cpp
class CompressedPointCloud {
  CompressedPointCloud();                                      // num_points_=0
  explicit CompressedPointCloud(const PointCloud&);            // encode
  explicit CompressedPointCloud(const proto::CompressedPointCloud&); // copies raw data, NO validation
  PointCloud Decompress() const;                               // no intensities
  bool empty() const; size_t size() const;                     // num_points_
  ConstIterator begin() const; ConstIterator end() const;      // forward iterator yielding RangefinderPoint BY VALUE
  bool operator==(const CompressedPointCloud&) const;          // point_data_ == && num_points_ ==
  proto::CompressedPointCloud ToProto() const;                 // num_points (int32) + point_data
 private: std::vector<int32> point_data_; size_t num_points_;
};
```
Constants (anon namespace in .cc):

| Name | Value | Meaning |
|---|---|---|
| `kPrecision` | `0.001f` m | grid spacing |
| `kBitsPerCoordinate` | `10` | bits per axis inside a block (block = 1024 mm = 1.024 m edge) |
| `kCoordinateMask` | `(1<<10)-1 = 1023` | |
| `kMaxBitsPerDirection` | `23` | `|coord|/kPrecision < 2^23` i.e. `|coord| < 8388.608 m`, CHECKed, else fatal "Point out of bounds" |

**Encoding** (per point, input order):
1. `CHECK_LT(max_i |p_i| / kPrecision, 1<<23)` (float division).
2. For each axis i: `r = RoundToInt(p_i / kPrecision)` (`std::lround` on float -> round half away from zero, then narrowed to int), `block_i = r >> 10` (arithmetic shift = floor division by 1024, works for negatives), `local_i = r & 1023` (always 0..1023, two's complement).
3. Append `{local, original_index}` to the bucket of `block` in a `mapping::HybridGridBase<std::vector<RasterPoint>>` (resolution arg kPrecision is irrelevant here). `num_blocks` counts newly non-empty buckets.

**Serialized layout** of `point_data_` (int32 stream), blocks in HybridGrid iteration order (see below), points inside a block in insertion (input) order:
```
for each non-empty block:
  [n_points_in_block] [block_x] [block_y] [block_z]
  n_points_in_block x  [ (local_z << 20) | (local_y << 10) | local_x ]   // computed as ((z<<10)+y)<<10)+x, value in [0, 2^30)
```
Total length = `4*num_blocks + num_points`. `CHECK_EQ(num_blocks, 0)` after the loop (consistency).

**Block iteration order** (needed for bit-exact `point_data` / `operator==` parity): HybridGridBase = `DynamicGrid<NestedGrid<FlatGrid<V,3>,3>>`. Each level iterates z-major flat index `((z<<bits)+y)<<bits)+x`, outer->inner. Because DynamicGrid's centering offset is always a multiple of 64, the order is independent of grid growth and equals lexicographic ascending order of the key
```
( floor(z/64), floor(y/64), floor(x/64),
  floor(mod(z,64)/8), floor(mod(y,64)/8), floor(mod(x,64)/8),
  mod(z,8), mod(y,8), mod(x,8) )        // mod = Euclidean, applied to block coords
```
(DynamicGrid max `bits_ = 8` -> block coords in [-8192, 8192), consistent with the 2^23 limit.) Rust: `BTreeMap<OrderKey, Vec<..>>` or HashMap + sort by this key. If byte-exact compatibility with C++-produced protos is not required, any deterministic order decodes correctly, but `operator==` between C++- and Rust-encoded clouds would differ.

**Decoding** (`ConstIterator::ReadNextPoint`): when `remaining_points_in_current_block_ == 0`, read count then 3 block coords, storing `block_i << 10`. Per point: `x = (bx + (v & 1023)) * kPrecision`, `y = (by + ((v >> 10) & 1023)) * kPrecision`, `z = (bz + (v >> 20)) * kPrecision` (z unmasked). Arithmetic: int sum converted to float, times `0.001f` (float multiply).
- Iterator state: `remaining_points_` (total), `remaining_points_in_current_block_`, `current_point_`, `current_block_coordinates_`, input cursor. Begin iterator eagerly reads the first point if `num_points_>0`. `EndIterator` = a begin iterator with `remaining_points_ = 0`. `operator!=` CHECKs same parent and compares only `remaining_points_`. `operator*` CHECKs `remaining_points_ > 0`. `operator++` decrements and reads next if any remain.
- Output order = grouped by block, NOT input order. Round-trip error <= 0.5 mm per axis (plus float error).
- Proto ctor does not validate (`TODO(wohe)`); malformed data -> out-of-bounds read in C++. Rust: return `Result`/bounds-check.

### 2.5 `imu_data.{h,cc}`
```cpp
struct ImuData { common::Time time; Eigen::Vector3d linear_acceleration; Eigen::Vector3d angular_velocity; };
proto::ImuData ToProto(const ImuData&); ImuData FromProto(const proto::ImuData&);
```
Straight field mapping; timestamp via `ToUniversal`/`FromUniversal`.

### 2.6 `odometry_data.{h,cc}`
```cpp
struct OdometryData { common::Time time; transform::Rigid3d pose; };
proto::OdometryData ToProto(const OdometryData&); OdometryData FromProto(const proto::OdometryData&);
```
`pose` via `transform::ToProto(Rigid3d)` / `transform::ToRigid3(proto)` (quaternion NOT normalized on read).

### 2.7 `fixed_frame_pose_data.{h,cc}`
```cpp
struct FixedFramePoseData { common::Time time; absl::optional<transform::Rigid3d> pose; };  // GPS etc., used in optimization
proto::FixedFramePoseData ToProto(const FixedFramePoseData&);  // pose field set only if has_value()
FixedFramePoseData FromProto(const proto::FixedFramePoseData&); // proto.has_pose() ? Some(ToRigid3) : None
```
Rust: `Option<Rigid3d>`; proto message field presence (`Option<Rigid3d>` in prost) maps directly.

### 2.8 `landmark_data.{h,cc}`
```cpp
struct LandmarkObservation { std::string id; transform::Rigid3d landmark_to_tracking_transform; double translation_weight; double rotation_weight; };
struct LandmarkData { common::Time time; std::vector<LandmarkObservation> landmark_observations; };
proto::LandmarkData ToProto(const LandmarkData&); LandmarkData FromProto(const proto::LandmarkData&);
```
- `id` is proto `bytes` <-> `std::string` (arbitrary bytes). Rust: if using `String`, decoding non-UTF-8 ids fails; prefer `Vec<u8>`/`bytes::Bytes` or document UTF-8 assumption.

### 2.9 `timed_point_cloud_data.{h,cc}`
```cpp
struct TimedPointCloudData { common::Time time; Eigen::Vector3f origin; TimedPointCloud ranges; std::vector<float> intensities; /* same size as ranges or empty */ };
struct TimedPointCloudOriginData {
  struct RangeMeasurement { TimedRangefinderPoint point_time; float intensity; size_t origin_index; };
  common::Time time; std::vector<Eigen::Vector3f> origins; std::vector<RangeMeasurement> ranges;
};
proto::TimedPointCloudData ToProto(const TimedPointCloudData&);
TimedPointCloudData FromProto(const proto::TimedPointCloudData&);
```
- `ToProto`: timestamp, origin, `point_data` (field 4) and `intensities`; never the legacy field.
- `FromProto`: `CHECK(intensities.size()==0 || intensities.size()==point_data.size())` - compares against the **new** field even when the legacy field is used, so legacy data with intensities CHECK-fails. If `point_data` non-empty use it, else legacy `Vector4f` -> `{xyz, time = .t}`. `time` of the message is the time of the last point (point times are relative, <= 0).
- No ToProto/FromProto for `TimedPointCloudOriginData` (internal only; produced by RangeDataCollator, consumed by local trajectory builders).

### 2.10 `data.h`
```cpp
class Data {                                   // polymorphic sensor packet, owned via std::unique_ptr<Data>
 public:
  explicit Data(const std::string& sensor_id); virtual ~Data();
  virtual common::Time GetTime() const = 0;
  const std::string& GetSensorId() const;
  virtual void AddToTrajectoryBuilder(mapping::TrajectoryBuilderInterface*) = 0;  // double dispatch
 protected: const std::string sensor_id_;
};
```

### 2.11 `internal/dispatchable.h`
```cpp
template<typename DataType> class Dispatchable : public Data {
  Dispatchable(const std::string& sensor_id, const DataType& data);  // copies data
  common::Time GetTime() const override { return data_.time; }      // DataType must have .time
  void AddToTrajectoryBuilder(TrajectoryBuilderInterface* tb) override { tb->AddSensorData(sensor_id_, data_); } // overload resolution on DataType
  const DataType& data() const;
 private: const DataType data_;
};
template<typename DataType> std::unique_ptr<Dispatchable<DataType>> MakeDispatchable(const std::string& sensor_id, const DataType& data);
```
Instantiated for: `TimedPointCloudData`, `ImuData`, `OdometryData`, `FixedFramePoseData`, `LandmarkData` (collated_trajectory_builder.h, cloud handlers). Other subclass: `mapping::LocalSlamResultData`.

### 2.12 `collator_interface.h`
```cpp
class CollatorInterface {  // non-copyable
  using Callback = std::function<void(const std::string& sensor_id, std::unique_ptr<Data>)>;
  virtual void AddTrajectory(int trajectory_id, const absl::flat_hash_set<std::string>& expected_sensor_ids, const Callback&) = 0;
  virtual void FinishTrajectory(int trajectory_id) = 0;
  virtual void AddSensorData(int trajectory_id, std::unique_ptr<Data>) = 0;  // per sensor_id in time order
  virtual void Flush() = 0;                                                  // once; no AddSensorData afterwards
  virtual absl::optional<int> GetBlockingTrajectoryId() const = 0;           // only if >=1 unfinished trajectory exists
};
```

### 2.13 `internal/ordered_multi_queue.{h,cc}` (core merge-sort engine)
```cpp
struct QueueKey { int trajectory_id; std::string sensor_id; bool operator<(...) const; }; // lexicographic (trajectory_id, sensor_id)
std::ostream& operator<<(std::ostream&, const QueueKey&);   // "(traj, sensor)" - inline in .cc
class OrderedMultiQueue {   // "thread-compatible"; movable
  using Callback = std::function<void(std::unique_ptr<Data>)>;
  void AddQueue(const QueueKey&, Callback);       // CHECK key not present
  void MarkQueueAsFinished(const QueueKey&);      // CHECK exists && !finished; sets finished; Dispatch()
  void Add(const QueueKey&, std::unique_ptr<Data>); // unknown key: LOG_EVERY_N(WARNING,1000) "Ignored data for queue" + DROP; else push back, Dispatch()
  void Flush();                                   // MarkQueueAsFinished on every unfinished queue (snapshot list first), in key order
  QueueKey GetBlocker() const;                    // CHECK(!queues_.empty()); returns blocker_
  ~OrderedMultiQueue();                           // CHECK every remaining queue is finished
 private:
  struct Queue { common::BlockingQueue<std::unique_ptr<Data>> queue; Callback callback; bool finished=false; };
  common::Time last_dispatched_time_ = Time::min();
  std::map<int, common::Time> common_start_time_per_trajectory_;
  std::map<QueueKey, Queue> queues_;   // ORDERED map: iteration order = key order (determinism + tie-breaking)
  QueueKey blocker_;                   // default-constructed: trajectory_id UNINITIALIZED until first CannotMakeProgress
};
const int kMaxQueueSize = 500;   // anon ns; only for logging
```
**`Dispatch()` exact algorithm** (loop until it returns):
1. Iterate `queues_` in key order. For each queue: `data = Peek()` (front or nullptr).
   - Empty and `finished` -> erase queue, continue.
   - Empty and not finished -> `CannotMakeProgress(key)`; **return** (the blocker is the first empty unfinished queue in key order).
   - Else if `next_data == nullptr || data.time < next_data.time` (strict `<`: ties go to the smaller key) select it.
   - `CHECK_LE(last_dispatched_time_, next_data->GetTime())` evaluated after each non-empty queue, against the current running minimum; fatal "Non-sorted data added to queue".
2. If nothing selected: `CHECK(queues_.empty())`, return.
3. `start = GetCommonStartTime(next_key.trajectory_id)`.
4. Three cases:
   - `next.time >= start` -> `last_dispatched_time_ = next.time`; `callback(Pop())`.
   - else if `queue.Size() < 2`: if not finished -> `CannotMakeProgress(next_key)`, return (cannot decide drop vs dispatch). If finished -> dispatch it (update last time, callback).
   - else (>=2 items): pop; if the new front's time `> start` (strict) dispatch the popped item (update last time), otherwise **drop** it (destroyed silently). I.e. keep the last packet before the common start so interpolation is possible.
5. Loop.

**`GetCommonStartTime(traj)`**: computed once per trajectory (first time it is needed, memoized forever in `common_start_time_per_trajectory_`): `max` over all queues currently in `queues_` with that trajectory_id of their front time (all are guaranteed non-empty at that point because step 1 passed). Logs INFO "All sensor data for trajectory N is available starting at 'T'.". Queues added later for that trajectory do not affect it.

**`CannotMakeProgress(key)`**: `blocker_ = key`; if any queue has `Size() > 500`, `LOG_EVERY_N(WARNING, 60) "Queue waiting for data: " << key` (logs the blocker key, once, then returns).

Semantics summary: global merge sort across **all trajectories**; waits until every unfinished queue has at least one item; output across queues globally non-decreasing in time; fatal if a queue receives data earlier than what was already dispatched. Callbacks run synchronously inside `Add` / `MarkQueueAsFinished` / `Flush` on the caller's thread.

BlockingQueue usage: only non-blocking ops are used (`Push` on an unbounded queue never blocks, `Peek<Data>()` returns nullptr when empty, `Pop()` only when non-empty, `Size()`). Its absl::Mutex is pure overhead here. Rust: `VecDeque<Box<dyn Data>>`.

### 2.14 `internal/collator.{h,cc}`
```cpp
class Collator : public CollatorInterface {
  OrderedMultiQueue queue_;                                    // one queue for ALL trajectories
  absl::flat_hash_map<int, std::vector<QueueKey>> queue_keys_;
};
```
- `AddTrajectory`: for each sensor_id in the (unordered) set: `queue_.AddQueue({traj, sensor_id}, [callback, sensor_id](data){ callback(sensor_id, move(data)); })`, record key. Duplicate key -> CHECK fail inside AddQueue.
- `FinishTrajectory(traj)`: `MarkQueueAsFinished` for each recorded key (`operator[]` -> unknown traj is a no-op). Calling twice -> CHECK fail (`!finished`).
- `AddSensorData(traj, data)`: key `{traj, data->GetSensorId()}`; unknown -> warning + drop.
- `Flush()`: `queue_.Flush()`.
- `GetBlockingTrajectoryId()`: always `optional(queue_.GetBlocker().trajectory_id)` (CHECKs non-empty queues).
- Deterministic output given deterministic per-queue input (std::map ordering).

### 2.15 `internal/trajectory_collator.{h,cc}`
```cpp
class TrajectoryCollator : public CollatorInterface {
  static void RegisterMetrics(metrics::FamilyFactory*);   // counter family "collator_input_total", help "Sensor data received"
  metrics::Counter* GetOrCreateSensorMetric(const std::string& sensor_id, int trajectory_id);
  static metrics::Family<metrics::Counter>* collator_metrics_family_;  // initialised to Family<Counter>::Null()
  absl::flat_hash_map<std::string, metrics::Counter*> metrics_map_;    // key "sensor_id/trajectory_id"
  absl::flat_hash_map<int, OrderedMultiQueue> trajectory_to_queue_;   // one queue PER trajectory
  absl::flat_hash_map<int, std::vector<QueueKey>> trajectory_to_queue_keys_;
};
```
- `AddTrajectory`: `CHECK_EQ(trajectory_to_queue_.count(traj), 0)`; adds per-sensor queues to that trajectory's own OrderedMultiQueue (NB: with an empty sensor set no queue entry is created).
- `FinishTrajectory`: marks each key finished via `.at(traj)`.
- `AddSensorData`: increments counter (labels `{sensor_id, trajectory_id}`, created lazily with `LOG(INFO) "Create metrics handler for key: ..."`), then `trajectory_to_queue_.at(traj).Add(...)` -> **unknown trajectory = crash** (`.at` throws/aborts), unlike Collator which drops.
- `Flush`: flushes every per-trajectory queue in flat_hash_map (unspecified) order.
- `GetBlockingTrajectoryId()`: always `nullopt`.
- Does not wait for other trajectories; `last_dispatched_time_` monotonicity is per trajectory. Interleaving across trajectories depends on input order (documented as non-deterministic).

### 2.16 `internal/voxel_filter.{h,cc}`
```cpp
std::vector<RangefinderPoint> VoxelFilter(const std::vector<RangefinderPoint>&, float resolution);
PointCloud                    VoxelFilter(const PointCloud&, float resolution);        // filters intensities with same mask
TimedPointCloud               VoxelFilter(const TimedPointCloud&, float resolution);   // ignores time for voxel key
std::vector<TimedPointCloudOriginData::RangeMeasurement> VoxelFilter(const std::vector<RangeMeasurement>&, float resolution); // key from point_time.position
proto::AdaptiveVoxelFilterOptions CreateAdaptiveVoxelFilterOptions(common::LuaParameterDictionary*); // GetDouble("max_length"), GetNonNegativeInt("min_num_points"), GetDouble("max_range")
PointCloud AdaptiveVoxelFilter(const PointCloud&, const proto::AdaptiveVoxelFilterOptions&);
```
**Voxel key** `GetVoxelCellIndex(p, res)`:
```
idx = p.array() / res                       // float
x,y,z = (uint64_t) common::RoundToInt(idx.*) // lround (half away from zero) -> int -> sign-extended to u64
key = (x << 42) + (y << 21) + z             // u64 wrapping arithmetic
```
- **Rounding, not flooring**: voxels are centred on integer multiples of `resolution` (cell = [ (k-0.5)res, (k+0.5)res ) ).
- No masking: negative or > 2^21 indices bleed into neighbouring bit fields -> distinct voxels can collide (e.g. `(x, y, -1)` and `(x, y-1, 2^21-1)`). Behaviourally part of the algorithm; replicate with `(i as i64 as u64).wrapping_shl(..).wrapping_add(..)` for parity. Index overflowing `int` (|p/res| > 2^31) is UB in C++.

**Which point is kept per voxel (reservoir sampling, k=1)** `RandomizedVoxelFilterIndices`:
```
std::minstd_rand0 generator;   // fresh per call, default seed 1 -> deterministic per input
flat_hash_map<u64, pair<int count, int index>> voxels;
for i in 0..n (input order):
  v = voxels[key(point_i)]; v.count++;
  if v.count == 1: v.index = i
  else: if uniform_int_distribution<int>(1, v.count)(generator) == v.count: v.index = i   // replace with prob 1/count
used[v.index] = true for each voxel
output = input elements with used[i], in ORIGINAL INPUT ORDER
```
- Each point in a voxel is kept with equal probability 1/N. One RNG draw per non-first point in its voxel, in input order (shared generator across voxels).
- `minstd_rand0`: LCG `x = 16807 * x mod (2^31-1)`, seed 1, outputs in [1, 2^31-2].
- `uniform_int_distribution` algorithm is **implementation-defined** (libstdc++ vs libc++ differ), so even C++ results differ between Linux/macOS builds. libstdc++ (reference Linux build) path for this engine: `urngrange = 2^31-3`, `urange = count-1`, `uerange = urange+1`, `scaling = urngrange / uerange`, `past = uerange * scaling`; `do { ret = g() - 1 } while (ret >= past); return 1 + ret / scaling`. Replicate this if bit-parity with Linux-produced results is desired; tests do not require it.
- `PointCloud` overload: `CHECK_LE(intensities.size(), points.size())`; intensities filtered by index for `i < intensities.size()`.

**AdaptiveVoxelFilter(cloud, opts)** = `AdaptivelyVoxelFiltered(opts, FilterByMaxRange(cloud, opts.max_range))`:
- `FilterByMaxRange`: `copy_if(point.position.norm() <= max_range)` (norm in whatever frame the caller passes; keeps intensities).
- `AdaptivelyVoxelFiltered`:
```
if size <= min_num_points: return cloud                       // already sparse
result = VoxelFilter(cloud, max_length); if result.size >= min_num_points: return result
for (high = max_length; high > 1e-2f*max_length; high /= 2.f):  // high in {L, L/2, ..., L/64}
  low = high/2.f; result = VoxelFilter(cloud, low)
  if result.size >= min_num_points:
    while ((high - low) / low > 1e-1f):                          // stop when within 10%
      mid = (low+high)/2.f; cand = VoxelFilter(cloud, mid)
      if cand.size >= min_num_points: low = mid; result = cand  else: high = mid
    return result
return result      // NB: returns the L/128 result, NOT the unfiltered cloud as the comment claims
```
All arithmetic in `float`. Comparisons `size_t` vs `float` (converted to float). Each VoxelFilter call restarts the RNG, so results are deterministic.

### 2.17 `map_by_time.h`
```cpp
template<typename DataType>   // DataType must have `common::Time time`
class MapByTime {
  void Append(int trajectory_id, const DataType&);   // CHECK_GE(traj,0); CHECK_GT(time, last time of that trajectory) (strictly increasing)
  template<typename NodeType> void Trim(const mapping::MapById<mapping::NodeId, NodeType>& nodes, const mapping::NodeId& node_id);
  bool HasTrajectory(int) const;
  ConstIterator BeginOfTrajectory(int) const; ConstIterator EndOfTrajectory(int) const;  // data_.at -> throws if absent
  mapping::Range<ConstTrajectoryIterator> trajectory_ids() const;   // ascending ids
  mapping::Range<ConstIterator> trajectory(int) const;              // ascending time
  ConstIterator lower_bound(int trajectory_id, common::Time) const; // first element with time >= t
  // ConstIterator: bidirectional over DataType; ConstTrajectoryIterator: bidirectional over int
 private: std::map<int, std::map<common::Time, DataType>> data_;
};
```
**Trim(nodes, node_id)** (call BEFORE removing node_id from `nodes`):
- No data for trajectory -> return. `CHECK(node in nodes)`.
- `gap_start` = time of previous node in same trajectory, or `Time::min()` if first; `gap_end` = time of next node, or `Time::max()` if last. `CHECK_LT(gap_start, gap_end)`.
- Range `[lower_bound(gap_start), upper_bound(gap_end))`; if empty return. If `gap_end != max`, exclude the last element of the range (retain last data inside gap); if then empty return. If `gap_start != min`, exclude the first element. Erase remaining. If trajectory map empty -> erase trajectory entry.
- Effect: keeps the boundary samples needed to interpolate across the gap.

Rust: `BTreeMap<i32, BTreeMap<Time, T>>`; use `range(..)` for bounds. Time is an i64-tick newtype with `min`/`max` sentinels.

---

## 3. Threading and ownership

- No class here is thread-safe; all are thread-compatible. Concurrency is handled by callers (MapBuilder / CollatedTrajectoryBuilder are driven from one thread in cartographer_ros; the cloud/gRPC path serializes per collator).
- `common::BlockingQueue` internally locks an `absl::Mutex`, but OrderedMultiQueue never blocks on it.
- Ownership: sensor packets travel as `std::unique_ptr<Data>` from `AddSensorData` -> queue -> callback (moved). Dropped packets (pre-common-start-time, unknown queue) are destroyed. Dispatchable stores a **copy** of the payload.
- Callbacks are invoked synchronously and re-entrantly from inside `Add`/`MarkQueueAsFinished`/`Flush`. A callback that calls back into the same queue would mutate `queues_` during iteration (UB in C++). Rust: callbacks stored in `Queue` as `Box<dyn FnMut(Box<dyn Data>)>`; borrowck forbids re-entrancy naturally; keep it that way.
- `TrajectoryCollator::collator_metrics_family_` is a mutable static (process-global; RegisterMetrics must happen before use). Rust: `OnceLock`/injected registry.
- `OrderedMultiQueue` is move-only (`std::function` + unique_ptr); destructor CHECKs all queues finished -> dropping an unfinished collator aborts. Rust: put the check in `Drop` behind `debug_assert!` or keep as `assert!` (panic in Drop during unwinding aborts; guard with `!std::thread::panicking()`).

---

## 4. External dependencies and exact usage

| Dep | Where | Usage |
|---|---|---|
| Eigen | everywhere | `Vector3f` (points, origins), `Vector3d` (IMU), `Vector4f` (legacy proto conversion `head<3>()`, `[3]`), `Array3i`/`Array3f` (raster/voxel math, `.cwiseAbs().maxCoeff()`, `.array()/res`), `Vector3i`, `isApprox` (tests), `Quaterniond` (landmark test). `Rigid3<T> * Vector3` via transform module. |
| absl | collator_interface, collator, trajectory_collator, voxel_filter, fixed_frame_pose_data, dispatchable | `flat_hash_set<std::string>` (expected sensor ids), `flat_hash_map` (queue key lists, metrics, voxel map), `optional` (FixedFramePoseData::pose, blocking trajectory id), `make_unique`, `StrCat` (metric key), `absl::Mutex` (via BlockingQueue). |
| glog | everywhere | `CHECK`, `CHECK_EQ/LE/LT/GT/GE`, `LOG(INFO)` (common start time, metrics creation), `LOG_EVERY_N(WARNING, 1000)` (ignored data), `LOG_EVERY_N(WARNING, 60)` (queue waiting). Rust: `assert!`/`panic!` for CHECK, `log`/`tracing` with rate limiting for EVERY_N. |
| protobuf | protos + all ToProto/FromProto | proto3 messages, `has_pose()` presence on message field, `Reserve`, `add_*`, repeated access. Rust: `prost` (message fields -> `Option<T>`; use `.unwrap_or_default()` to mirror C++ default-instance semantics for `origin`, `position`, etc.). |
| std | | `std::map` (ordering matters!), `std::minstd_rand0`, `std::uniform_int_distribution`, `std::lround`. |
| In-repo | | `common::Time` (`FromUniversal/ToUniversal`, `Time::min/max`), `common::RoundToInt` (= `std::lround`, result narrowed to int), `common::LuaParameterDictionary`, `common::BlockingQueue`, `transform::{ToEigen,ToProto,ToRigid3,Rigid3f,Rigid3d,Embed3D}`, `mapping::HybridGridBase` (compression), `mapping::MapById/NodeId/Range` (MapByTime), `metrics::{Counter,Family,FamilyFactory}`, `mapping::TrajectoryBuilderInterface::AddSensorData` overloads. |

---

## 5. Tests

Test helper `internal/test_helpers.h`:
- `MATCHER_P(Near, point, ...)`: `arg.isApprox(point, 0.001f)`.
- `testing::CollatorOutput = std::tuple<int traj, std::string sensor_id, common::Time>`.
- `testing::CollatorInput { const int trajectory_id; unique_ptr<Data> data; const CollatorOutput expected_output; }` with factories `CreateImuData(traj, sensor, int time)`, `CreateTimedPointCloudData(...)` (origin zero, no ranges), `CreateOdometryData(...)` (identity pose); times are `FromUniversal(time)` ticks; `MoveToCollator(CollatorInterface*)` calls `AddSensorData(trajectory_id, move(data))`.

| File | Test | Checks / inputs / tolerances |
|---|---|---|
| `compressed_point_cloud_test.cc` | `CompressPointCloudTest.CompressesPointsCorrectly` | `TestPoint(p)`: compress single point, size==1, decoded `(pos - p).isZero(0.001)` (Eigen isZero = each coeff abs <= prec), recompress decoded point, still within 0.001 of p. Points: (8000,7500,5000), (1000,2000,3000), (100,200,300), (10,20,30), (-0.00049,-0.0005,-0.0015), (0.05119,0.0512,0.05121), (-0.05119,-0.0512,-0.05121), (0.8405,0.84,0.8396), (0.8395,0.8394,0.8393), (0.839,0.8391,0.8392), (0.8389,0.8388,0.83985). Exercises half-way rounding, negatives, block boundary at 1024 mm (0.0512 ~ block-local values), large coords near the 8388 m limit. |
| | `CompressPointCloudTest.Compresses` | 3 points x = 0.838/0.839/0.840 (y=z=0): `!empty()`, size 3, decompressed size 3, each original `Contains` within 0.001 (order-agnostic). |
| | `CompressPointCloudTest.CompressesEmptyPointCloud` | default ctor: `empty()`, `size()==0`. |
| | `CompressPointCloudTest.CompressesNoGaps` | 3000 points x = 0.001*i - 1.5 (i in 0..3000, crosses negative/positive block boundaries); compress, decompress, recompress: sizes equal; iterate compressed, sort x, consecutive diffs `FloatNear(0.001, 1e-7)` (no gaps/duplicates). |
| `landmark_data_test.cc` | `LandmarkDataTest.LandmarkDataToAndFromProto` (TEST_F) | Fixture: obs {"ID1", Rigid3d((1,1,1), Quaterniond(w=1,x=1,y=-1,z=-1)), 1, 3}, {"ID2", ((2,2,2), Quaterniond(2,2,-2,-2)), 2, 4}. NB quaternions are **non-unit**. Round-trip `FromProto(ToProto(...))` at time FromUniversal(50): time exact; ids exact; transform `IsNearly(..., 1e-2)` (= `ToEigen(a).isApprox(ToEigen(b), 1e-2)` on the affine matrices); weights `DoubleNear(.., 0.01)`; `ElementsAre` order preserved. |
| `map_by_time_test.cc` | `MapByTimeTest.AppendAndViewTrajectory` | Append traj 0 @10ms, traj 42 @42ms, @43ms; `trajectory(42)` yields exactly 42, 43 ms in order. Times via `Time(FromMilliseconds(ms))`. |
| | `MapByTimeTest.Trimming` | Data @ {1,41,42,43,47,48,49,5000} ms in traj 42; nodes {0:42, 1:46, 2:48}. Trim node 1 -> remaining {1,41,42,48,49,5000}. Then trim node 2 (nodes now {0,2}) and node 0 -> `HasTrajectory(42)` false. |
| | `MapByTimeTest.TrimmingDoesNotCreateTrajectory` | Trim on a trajectory with no data does not create an entry. |
| `point_cloud_test.cc` | `PointCloudTest.TransformPointCloud` | Points (0.5,0.5,1), (3.5,0.5,42) rotated by `Embed3D(Rigid2f::Rotation(pi/2))` -> x,y = (-0.5,0.5), (-0.5,3.5); `EXPECT_NEAR 1e-6`. |
| | `PointCloudTest.TransformTimedPointCloud` | Same with TimedPointCloud (time 0). |
| | `PointCloudTest.CopyIf` | {(0,0,0),(1,1,1),(2,2,2)}, pred x>0.1 -> size 2, intensities empty, exact `ElementsAre`. |
| | `PointCloudTest.CopyIfWithIntensities` | Same with intensities {0,1,2} -> intensities {1,2} (`FloatNear 1e-6`), size 2. |
| `range_data_test.cc` | (none) | Only a fixture `RangeDataTest` (origin (1,1,1), returns (0,1,2),(4,5,6),(0,1,2), misses (7,8,9)) with **no TEST_F** - dead code. Rust: add own round-trip tests incl. legacy-field fallback. |
| `internal/collator_test.cc` | `Collator.Ordering` | Traj 0, sensors {"horizontal_rangefinder","vertical_rangefinder","imu","odometry"}; first all four @0, then inputs[4..9] = h@100, v@200, imu@300, h@400, v@500, odom@600, added in order 4,9,7,5,8,6. Expect blocking traj 0; exactly 7 received; received[4..6] == inputs[4..6]. After FinishTrajectory+Flush: all 10 received and received[i]==inputs[i] for i>=4 (first four are ties at t=0, order by key not checked). |
| | `Collator.OrderingMultipleTrajectories` | Trajs {8,5}, sensors {"my_points","some_imu"}; all four @0, then [4] t8 pts@100, [5] t5 imu@200, [6] t8 imu@300, [7],[8] t5 pts@400 (duplicate time, allowed: CHECK is <=), [9] t5 pts@500, [10] t5 imu@600. Add 4,6 -> blocker 5; add 7,8 -> blocker 5; add 5 -> blocker 8; add 10,9 -> blocker 8; 5 received, received[4]==input[4]. Finish both + Flush: 11 received, in input index order for i>=4. |
| `internal/ordered_multi_queue_test.cc` | Fixture: keys kFirst{1,"a"}, kSecond{1,"b"}, kThird{2,"b"}; callback asserts non-decreasing times; `MakeImu(t)` = Dispatchable "imu" ImuData @FromUniversal(t). | |
| | `OrderedMultiQueueTest.Ordering` | First {0,4,5,6}: none; Second {0,1}: none; Third {0,2}: 4 values; Second 3: 5 values; Second 7, Third 8, Flush: 11 values non-decreasing. |
| | `OrderedMultiQueueTest.MarkQueueAsFinished` | First {1,2,3}; finish First: 0; finish Second: 0 (Third empty unfinished blocks); finish Third: 3 values with times 1,2,3. |
| | `OrderedMultiQueueTest.CommonStartTimePerTrajectory` | First {0,1,2,3}, Second {2}: 0; Third {4}: 2 values (traj1 start=2: 0 and 1 dropped because next item <= 2; First@2 dispatched (tie -> first key), Second@2 dispatched; blocked on Second). Finish First: 2; finish Second: 4 (First@3, then Third@4 with traj2 start=4); finish Third: 4. |
| `internal/trajectory_collator_test.cc` | `TrajectoryCollator.OrderingMultipleTrajectories` | Trajs {4,7}, sensors {"my_points","some_imu"}. Inputs: [0] t4 pts@0, [1] t7 pts@0, [2] t7 imu@0, [3] t4 imu@0, [4] t7 pts@100, [5] t4 pts@50, [6] t4 imu@60, [7] t7 pts@150, [8] t7 imu@120. After 0,1: 0 received; after 2,3: 2 received = [1],[0]; after 4: 3 (=[2]) - doesn't wait for traj 4; after 5: 4 (=[3]); after 6: 5 (=[5]); after 7: 5; after 8: 7 (=[4],[8]). `GetBlockingTrajectoryId()` is nullopt. Finish both + Flush: 9 received. |
| `internal/voxel_filter_test.cc` | `VoxelFilterTest.ReturnsOnePointInEachVoxel` | {(0,0,0),(0.1,-0.1,0.1),(0.3,-0.1,0),(0,0,0.1)}, res 0.3 -> size 2 (rounding puts 3 points in voxel (0,0,0), (0.3,..) in (1,0,0)), intensities empty, both results are input points, result contains input[2]. |
| | `VoxelFilterTest.CorrectIntensities` | 100 points (-100, 0.3, 0.1*i) with intensity = z; res 0.3: intensities size == points size and each `|z - intensity| <= 1e-6`. |
| | `VoxelFilterTest.HandlesLargeCoordinates` | {(1e5,0,0),(100000.001,-1e-4,1e-4),(100000.003,-1e-4,0),(-2e5,0,0)}, res 0.01 -> size 2 (first three share a voxel; float 1e5 resolution swallows mm offsets), intensities empty, contains input[3]. |
| | `VoxelFilterTest.IgnoresTime` | 100 timed points same position (-100,0.3,0.4) times 0..99, res 0.3 -> exactly 1 point, which is one of the inputs (time not part of key). |

Not unit-tested here: `TransformRangeData`, `CropRangeData`, `CropPointCloud`, RangeData/Imu/Odometry/FixedFramePose/TimedPointCloudData proto round-trips, `AdaptiveVoxelFilter`, `CreateAdaptiveVoxelFilterOptions`, TrajectoryCollator metrics. The Rust port should add tests for these (especially AdaptiveVoxelFilter's fall-through behavior and legacy proto fields).

---

## 6. Rust porting notes and pitfalls

Suggested module layout: `sensor::{rangefinder_point, point_cloud, range_data, compressed_point_cloud, imu_data, odometry_data, fixed_frame_pose_data, landmark_data, timed_point_cloud_data, data, map_by_time, collator::{CollatorInterface trait, Collator, TrajectoryCollator, OrderedMultiQueue}, voxel_filter}`; protos via `prost-build` in a `proto` crate/module.

1. **Float semantics**: keep `f32` everywhere the C++ uses float (points, voxel index computation `p / res`, adaptive filter lengths, compression `p / 0.001f32`, decode `(i32 sum) as f32 * 0.001f32`). Do not replace division by multiplication with reciprocal (changes rounding at half-way cases tested in CompressesPointsCorrectly).
2. **Rounding**: `common::RoundToInt` = `std::lround` -> Rust `x.round() as i32` (round half away from zero; matches). Out-of-range: C++ UB vs Rust saturating cast; voxel filter on huge coords will differ - acceptable but document.
3. **Voxel key**: replicate the unmasked `(x<<42)+(y<<21)+z` over `u64` with sign-extended negatives and wrapping ops; otherwise collision behaviour differs. Consider a typed struct key if parity is not needed, but then the "which voxels merge" behaviour differs in edge cases.
4. **RNG parity**: implement `MinStdRand0` (16807, 2^31-1, seed 1) and the libstdc++ `uniform_int_distribution` downscaling (section 2.16) in a small module; new generator per filter call. Output order = input order, not hash order. Use `HashMap`/`FxHashMap` for voxel buckets - iteration order is irrelevant because results go through the `used` mask.
5. **AdaptiveVoxelFilter fall-through** returns the last (L/128) filtered result, not the input; the C++ comment is wrong. Preserve code behaviour. `min_num_points` is `f32`; compare `len as f32 >= min_num_points`.
6. **Compressed cloud**: ordering of blocks (section 2.4) must be replicated for bit-identical `point_data`. Avoid porting HybridGrid into `sensor` just for this (breaks layering); a `BTreeMap` keyed by the 9-tuple order key is equivalent. Validate proto input (lengths, counts) instead of trusting it. Iterator yields values (not references) -> implement `Iterator<Item = RangefinderPoint>`.
7. **OrderedMultiQueue**: must use an ordered map (`BTreeMap<QueueKey, Queue>`, `QueueKey: Ord` derived on `(i32, String)` field order) - determinism and tie-breaking depend on it. Iterating while erasing: collect keys to remove or use `retain` carefully - but note erasure happens lazily *during* the scan and the scan returns early at the first empty unfinished queue (queues after it are not erased in that pass). Erasure is not observable except through `GetBlocker`'s `CHECK(!queues_.empty())` and the destructor check, so a two-pass approach is fine.
8. **Blocker default**: `blocker_` has an uninitialized `trajectory_id` until the first `CannotMakeProgress`. In Rust use `Option<QueueKey>`; `get_blocker()` must still panic when `queues` is empty; return the stale key otherwise.
9. **Drop semantics**: data with time < common start time can be dropped silently (only keeps the last packet before start). Strictness: dispatch condition `>= start` for the head, but look-ahead uses `> start`.
10. **CHECK -> panic**: non-sorted input (`last_dispatched_time > next.time`), duplicate AddQueue, finishing twice, finishing unknown queue, destructor with unfinished queues, TrajectoryCollator unknown trajectory. Consider `Result` for API-boundary errors but keep panics for invariants to mirror behaviour.
11. **Data trait**: `trait Data { fn time(&self) -> Time; fn sensor_id(&self) -> &str; fn add_to_trajectory_builder(self: Box<Self>, tb: &mut dyn TrajectoryBuilderInterface); }` - C++ passes by const ref and copies (`AddSensorData(sensor_id_, data_)`); in Rust consuming `Box<Self>` avoids the copy. `Dispatchable<T>` generic with a trait bound like `T: Timed + DispatchToBuilder` (the C++ overload resolution becomes a trait impl per type). `LocalSlamResultData` (mapping) must also implement it - define the trait in `sensor` with `TrajectoryBuilderInterface` from `mapping` -> cyclic crate dependency; keep both in one crate or invert via a generic/trait object defined in a lower-level crate.
12. **Callbacks**: `CollatorInterface::Callback` receives `(sensor_id: &str, Box<dyn Data>)`. `Collator` wraps it per sensor: needs `Clone`/`Arc` of the user callback (C++ copies `std::function` per sensor) -> store `Arc<dyn Fn(&str, Box<dyn Data>)>` or `Rc<RefCell<dyn FnMut>>`.
13. **flat_hash_set of expected sensor ids** is unordered but irrelevant (queues are BTreeMap). TrajectoryCollator `Flush` order over trajectories is unspecified in C++; in Rust prefer deterministic (`BTreeMap<i32, OrderedMultiQueue>`) - strictly better.
14. **Proto presence**: prost message fields are `Option<_>`; C++ treats missing `origin`/`position`/`pose` as zero/default (except FixedFramePoseData.pose, which maps to `None`). Use `unwrap_or_default()` - note a default `Quaterniond` is all-zero (not identity), matching C++ default-instance behaviour.
15. **Legacy fields**: RangeData `returns_legacy/misses_legacy` (Vector3f) and TimedPointCloudData `point_data_legacy` (Vector4f, `t` = time) are read-only fallbacks; replicate the TimedPointCloudData intensity CHECK quirk or relax it deliberately (document).
16. **RangeData proto drops intensities**; CompressedPointCloud drops intensities and reorders points.
17. **Quaternions**: transform `ToRigid3` does not normalize; LandmarkData test uses non-unit quaternions. If the Rust `Rigid3d` stores `UnitQuaternion` (normalizes), round-trip comparisons still pass (both sides normalized via the same path) but values differ from C++; decide in the transform module port.
18. **Time**: `common::Time` = i64 ticks of 100 ns (UTS epoch); `FromUniversal(i64)`. Need `Time::MIN/MAX` sentinels and `Ord` for BTreeMap keys. RangeDataCollator uses `FromSeconds(f32 time)` conversions (double -> duration_cast truncation).
19. **PointCloud invariant**: intensities empty or same length. Make the field private, provide `push` that panics/debug_asserts when intensities are non-empty (or `push_with_intensity`).
20. **Metrics**: `collator_input_total{sensor_id, trajectory_id}` counter; Null family by default. Port behind a metrics trait with a no-op default.
21. **Logging**: rate-limited warnings (EVERY_N 1000 / 60) are per call-site counters; emulate with an `AtomicU64` counter per site.

---

## Appendix A. `mapping/internal/range_data_collator.{h,cc}` (namespace `mapping`)

```cpp
class RangeDataCollator {
  explicit RangeDataCollator(const std::vector<std::string>& expected_range_sensor_ids);
  sensor::TimedPointCloudOriginData AddRangeData(const std::string& sensor_id, sensor::TimedPointCloudData);
 private:
  sensor::TimedPointCloudOriginData CropAndMerge();
  const std::set<std::string> expected_sensor_ids_;
  std::map<std::string, sensor::TimedPointCloudData> id_to_pending_data_;  // at most one message per sensor
  common::Time current_start_ = Time::min(), current_end_ = Time::min();
  constexpr static float kDefaultIntensityValue = 0.f;
};
```
Purpose: merge TimedPointCloudData from several range sensors into one stream monotonic in per-point time; buffers <= 1 message per sensor (delay up to slowest sensor period).

`AddRangeData(id, data)`:
1. `CHECK(id in expected)`; `data.intensities.resize(data.ranges.size(), 0.f)` (pads OR truncates).
2. If a message from `id` is already pending: `current_start_ = current_end_; current_end_ = pending[id].time; result = CropAndMerge();` then `emplace(id, data)` (pending[id] has been consumed fully by CropAndMerge because its points are all <= its own time, so emplace succeeds); return result.
3. Else emplace; if not all expected sensors pending -> return `{}` (default: time = Time() epoch-zero, empty).
4. Else `current_start_ = current_end_; current_end_ = min over pending times; return CropAndMerge()`.

`CropAndMerge()`: `result = {time = current_end_, origins = {}, ranges = {}}`. For each pending sensor (std::map order by sensor id):
- `overlap_begin` = first point with `data.time + FromSeconds(p.time) >= current_start_` (points before are dropped; one `LOG(WARNING) "Dropped N earlier points."` per CropAndMerge call).
- `overlap_end` = first point (from overlap_begin) with absolute time `> current_end_`.
- If non-empty overlap: push `data.origin` -> `origin_index`; `time_correction = (float) ToSeconds(data.time - current_end_)`; push `RangeMeasurement{point (time += time_correction), intensity, origin_index}` (times now relative to `current_end_`).
- If `overlap_end == end` erase the pending entry; else if `overlap_end == begin` keep untouched; else replace with the remaining tail (same time/origin, points/intensities from overlap_end).
- Finally `std::sort` (NOT stable) ranges by `point_time.time` ascending. Rust: `sort_by` (stable) may order equal-time points differently from C++ (`std::sort` order for ties is unspecified anyway); use `sort_unstable_by(|a,b| a.partial_cmp(b))` is fine; neither guarantees parity.
- Note the scan conditions assume points are sorted by time within a message.

Tests (`mapping/internal/range_data_collator_test.cc`), `kNumSamples = 10`; `CreateFakeRangeData(from, to, fake_intensities)`: time = FromUniversal(to), origin (0,1,2), 10 points `(1,2,fraction)`, fraction = i/9, relative time `(1-fraction) * -duration_seconds`; intensity = z if enabled.

| Test | Checks |
|---|---|
| `RangeDataCollatorTest.SingleSensor` | (200,300): time 300, 1 origin, 10 ranges, sorted. (300,500): time 500, 10 ranges, first point abs time ~300 (`EXPECT_NEAR` 2 ticks). (-1000,510): time 510, 1 origin, 1 range with time exactly 0.f. |
| `RangeDataCollatorTest.SingleSensorEmptyData` | Empty msg @300 -> output time 300, 0 ranges; then same as above. |
| `RangeDataCollatorTest.TwoSensors` | s0 (200,300) -> 0 ranges; s1 (-1000,310) -> 2 origins, time 300, 19 ranges, first abs ~ -1000 (+-2), last time 0.f; s0 (300,500) -> 2 origins, time 310, 2 ranges, first abs ~300, last 0.f; s0 (600,700) -> time 500; total ranges over outputs 1..3 == 30; sorted. |
| `RangeDataCollatorTest.ThreeSensors` | s0 (100,200), s1 (199,250) -> 0 each; s2 (210,300) -> 11 ranges; s2 (400,500) -> output2+output3 == 30; sorted. |
| `RangeDataCollatorTest.ThreeSensorsWithIntensities` | Same with intensities; additionally each `|intensity - z| <= 1e-6`. |
