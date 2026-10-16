# Architecture report: 3D local mapping (Phase 0)

Source root: `cartographer/cartographer/mapping/` (C++ is read-only).
Scope: `3d/*`, `internal/3d/*` (except `pose_graph_3d*`, optimization, constraint builder), `internal/3d/scan_matching/*`, plus `internal/imu_based_pose_extrapolator.{h,cc}` (which uses `internal/3d/imu_integration.h` and `internal/3d/rotation_parameterization.h`).
Other teams own `pose_graph_3d`, `optimization/*` (incl. `SpaCostFunction3D`, `AccelerationCostFunction3D`, `RotationCostFunction3D`, `CeresPose`), and `constraints/constraint_builder_3d`. They are summarized below only where this module depends on them.

## 0. File inventory

| File | LOC | Kind | Notes |
|---|---|---|---|
| `3d/hybrid_grid.h` | 576 | header-only templates | FlatGrid / NestedGrid / DynamicGrid / HybridGridBase / HybridGrid / IntensityHybridGrid |
| `3d/range_data_inserter_3d.{h,cc}` | 170 | class | hit/miss lookup-table insertion, intensities |
| `3d/submap_3d.{h,cc}` | 489 | classes | `Submap3D`, `ActiveSubmaps3D`, X-ray texture |
| `internal/3d/imu_integration.h` | 163 | header templates | `IntegrateImu`, `ExtrapolatePoseWithImu` |
| `internal/3d/rotation_parameterization.h` | 67 | functors | `YawOnlyQuaternionPlus`, `ConstantYawQuaternionPlus` |
| `internal/3d/local_slam_result_3d.{h,cc}` | 98 | class | glue to pose graph / submap controller |
| `internal/3d/local_trajectory_builder_options_3d.{h,cc}` | 106 | fn | Lua to proto |
| `internal/3d/local_trajectory_builder_3d.{h,cc}` | 605 | class | local SLAM pipeline |
| `internal/3d/scan_matching/ceres_scan_matcher_3d.{h,cc}` | 230 | class | Ceres problem |
| `.../occupied_space_cost_function_3d.h` | 91 | functor | |
| `.../intensity_cost_function_3d.{h,cc}` | 123 | functor | |
| `.../rotation_delta_cost_functor_3d.h` | 80 | functor | |
| `.../translation_delta_cost_functor_3d.h` | 71 | functor | |
| `.../interpolated_grid.h` | 169 | template | tricubic (Hermite, zero slope) interpolation |
| `.../low_resolution_matcher.{h,cc}` | 76 | fn | closure scorer |
| `.../precomputation_grid_3d.{h,cc}` | 141 | class+fns | uint8 max-pyramids |
| `.../fast_correlative_scan_matcher_3d.{h,cc}` | 601 | class | 3D branch and bound |
| `.../real_time_correlative_scan_matcher_3d.{h,cc}` | 184 | class | exhaustive 6-DOF |
| `.../rotational_scan_matcher.{h,cc}` | 251 | class | yaw histograms |
| `internal/imu_based_pose_extrapolator.{h,cc}` | ~420 | class | Ceres sliding-window extrapolator |
| Tests | see section 17 | | 12 test files, 35 TEST/TEST_F |

Related protos: `mapping/proto/{hybrid_grid, range_data_inserter_options_3d, submaps_options_3d, local_trajectory_builder_options_3d, pose_extrapolator_options, submap (Submap3D)}.proto`, `mapping/proto/scan_matching/{ceres_scan_matcher_options_3d, fast_correlative_scan_matcher_options_3d, real_time_correlative_scan_matcher_options}.proto`.

Dependency DAG inside scope (arrow means "uses"):
```
LocalTrajectoryBuilder3D -> ActiveSubmaps3D -> Submap3D -> {HybridGrid, IntensityHybridGrid, RangeDataInserter3D, RotationalScanMatcher::RotateHistogram}
                         -> RealTimeCorrelativeScanMatcher3D -> HybridGrid
                         -> CeresScanMatcher3D -> {OccupiedSpaceCF3D, IntensityCF3D -> InterpolatedGrid -> HybridGrid}, {Translation,Rotation}DeltaCostFunctor3D, YawOnlyQuaternionPlus, optimization::CeresPose
                         -> RotationalScanMatcher::ComputeHistogram
                         -> PoseExtrapolatorInterface (PoseExtrapolator | ImuBasedPoseExtrapolator -> imu_integration, ConstantYawQuaternionPlus, Spa/Acceleration/RotationCostFunction3D)
FastCorrelativeScanMatcher3D (used by constraint_builder_3d) -> PrecomputationGridStack3D -> PrecomputationGrid3D; LowResolutionMatcher; RotationalScanMatcher
```

---

## 1. `3d/hybrid_grid.h`

### 1.1 API

```cpp
int ToFlatIndex(const Array3i& index, int bits);      // ((z<<bits)+y)<<bits)+x ; DCHECK 0<=idx<2^bits
Array3i To3DIndex(int index, int bits);                // (i&m, (i>>bits)&m, (i>>bits)>>bits), m=(1<<bits)-1
template<T> bool IsDefaultValue(const T& v);           // v == T()
template<E> bool IsDefaultValue(const std::vector<E>&);// v.empty()

template<TValueType,int kBits> class FlatGrid {        // non-copyable
  static int grid_size();                              // 1<<kBits
  ValueType value(Array3i) const; ValueType* mutable_value(Array3i);
  class Iterator { Iterator(); explicit Iterator(const FlatGrid&); void Next(); bool Done();
                   Array3i GetCellIndex(); const ValueType& GetValue(); };
  std::array<ValueType, 1<<(3*kBits)> cells_;          // default-constructed (zeroed)
};
template<WrappedGrid,int kBits> class NestedGrid {     // grid_size = Wrapped::grid_size()<<kBits
  value / mutable_value (lazily allocs child); Iterator (same API);
  std::array<std::unique_ptr<WrappedGrid>, 1<<(3*kBits)> meta_cells_;
};
template<WrappedGrid> class DynamicGrid {              // movable, not copyable
  DynamicGrid(): bits_(1), meta_cells_(8);
  int grid_size() const;                               // Wrapped::grid_size() << bits_
  value(index) const;  mutable_value(index);           // signed indices, grows on demand
  class Iterator { ...; void AdvanceToEnd(); pair<Array3i,ValueType> operator*(); operator++; operator!= (compares current_ only) };
  void Grow();                                         // private; CHECK_LE(new_bits, 8)
};
template<V> using GridBase = DynamicGrid<NestedGrid<FlatGrid<V,3>,3>>;
template<V> class HybridGridBase : public GridBase<V> {
  explicit HybridGridBase(float resolution);
  float resolution() const;
  Array3i GetCellIndex(const Vector3f& p) const;       // lround(p/res) per component
  static Array3i GetOctant(int i);                     // (i&1, i&2, i&4) as 0/1
  Vector3f GetCenterOfCell(const Array3i& idx) const;  // idx * res
  Iterator begin() const; Iterator end() const;
};
class HybridGrid : public HybridGridBase<uint16> {
  explicit HybridGrid(float resolution);
  explicit HybridGrid(const proto::HybridGrid&);       // CHECK sizes equal, SetProbability each
  void SetProbability(const Array3i&, float p);        // *mutable_value = ProbabilityToValue(p)
  void FinishUpdate();                                 // pops update_indices_, subtracts kUpdateMarker
  bool ApplyLookupTable(const Array3i&, const std::vector<uint16>& table);
  float GetProbability(const Array3i&) const;          // ValueToProbability(value)
  bool IsKnown(const Array3i&) const;                  // value != 0
  proto::HybridGrid ToProto() const;                   // CHECK(update_indices_.empty())
  std::vector<uint16*> update_indices_;                // raw pointers into cells
};
struct AverageIntensityData { float sum = 0.f; int count = 0; };
class IntensityHybridGrid : public HybridGridBase<AverageIntensityData> {
  void AddIntensity(const Array3i&, float);            // count+=1; sum+=intensity
  float GetIntensity(const Array3i&) const;            // count==0 ? 0 : sum/count
};
```

### 1.2 Structure and sizes

| Level | Type | cells/dim | Storage |
|---|---|---|---|
| leaf | `FlatGrid<V,3>` | 8 | inline `array<V,512>` (uint16: 1 KiB) |
| middle | `NestedGrid<FlatGrid,3>` | 64 | `array<unique_ptr<FlatGrid>,512>` (lazy) |
| root | `DynamicGrid<NestedGrid>` | `64 << bits_`; starts bits_=1 -> 128 | `vector<unique_ptr<NestedGrid>>` size `8^bits_` (lazy) |

- Root bits_ max is 8 -> 16384 cells/dim -> index range `[-8192, 8191]`. Exceeding it is a CHECK failure (process abort).
- Signed indices: root shifts by `grid_size()/2`: `shifted = index + (grid_size() >> 1)`. Bounds test uses unsigned cast: `(shifted.cast<unsigned>() >= grid_size()).any()` gives "outside".
- `value()` outside or unallocated returns `ValueType()` (0 / {0,0}). Reads never allocate.
- `mutable_value()` outside calls `Grow()` and recurses (may grow repeatedly).
- `Grow()`: `new_bits = bits_+1`; new vector of `8*size`; each old meta cell `(x,y,z)` moves to `(x,y,z) + (1 << (bits_-1))` in the new flat (z-major) layout. Old content stays centered. Child grids are moved, not copied.
- Meta index = `index / Wrapped::grid_size()` (indices are non-negative after shift, so truncating division is fine); inner = `index - meta*size`.

### 1.3 Iteration order (load-bearing for `ToProto` order and float summation order)

- Iterators skip cells whose value `== ValueType()` (for `uint16`, 0; for `AverageIntensityData` there is no `operator==`, so its iterator is never instantiated in practice; for `uint8` precomputation grids, 0).
- Order is hierarchical: root meta-cells in flat z-major order (x fastest, then y, then z), within each NestedGrid meta-cells in z-major order, within each FlatGrid cells z-major. It is **not** global lexicographic order over (z,y,x).
- `GetCellIndex()` at each level computes the flat position as `(1 << (3*bits)) - (end_ - current_)`, converts it with `To3DIndex`, then multiplies by the child grid size and adds the child index. Root subtracts `(1 << (bits_-1)) * 64`.
- The iterator snapshots `bits_` and raw pointers. It is invalidated by `Grow()` or by new allocations. Rust: borrow `&self`, so the type system already prevents mutation while iterating.
- Values with `kUpdateMarker` set are still yielded (raw value). `ToProto` CHECKs no pending update.
- `begin()/end()`: `end()` = `Iterator(*this)` + `AdvanceToEnd()`. `operator!=` compares only the root pointer.

### 1.4 Probability encoding (from `mapping/probability_values.*`, other module)

- `kMinProbability = 0.1`, `kMaxProbability = 0.9`, `kUnknownProbabilityValue = 0`, `kUpdateMarker = 1<<15`.
- `ProbabilityToValue(p) = lround((clamp(p,0.1,0.9)-0.1) * 32766/(0.8)) + 1` (in [1, 32767]).
- `ValueToProbability(v)`: table of 65536 entries (two repetitions, so values carrying the marker also map). Value 0 maps to **kMinProbability (0.1)**, not 0.5. `v -> v*s + (0.1 - s)`, `s = 0.8/32766`.
- `ComputeLookupTableToApplyOdds(odds)`: 32768 entries; `[0] = ProbabilityToValue(ProbabilityFromOdds(odds)) + kUpdateMarker` (unknown cell becomes the probability of `odds`); `[c] = ProbabilityToValue(ProbabilityFromOdds(odds * Odds(ValueToProbability(c)))) + kUpdateMarker`.

### 1.5 Update protocol

- `ApplyLookupTable(index, table)`: `cell = mutable_value(index)` (allocates); if `*cell >= kUpdateMarker`, return false (already updated this round); else push `cell` pointer to `update_indices_`, set `*cell = table[*cell]` (now >= marker), return true.
- `FinishUpdate()`: pop each pointer, subtract the marker.
- Pitfall: `update_indices_` holds raw `uint16*` into FlatGrid arrays. That is safe in C++ because FlatGrids are heap-allocated by unique_ptr and `Grow()` moves only pointers. In Rust, store **indices** (`Array3i`) or `(leaf ptr id, offset)` instead of references. Recommended: `Vec<[i32;3]>`, re-resolved in `finish_update` (cost: one extra lookup), or `NonNull<u16>` with Box-stable leaves (unsafe but equivalent).

### 1.6 Proto

- `proto::HybridGrid { float resolution; repeated sint32 x_indices, y_indices, z_indices; repeated int32 values; }`. `ToProto` emits cells in iteration order (1.3) and **strips nothing**: values written are raw `uint16` (no marker allowed). `FromProto` goes through `SetProbability(ValueToProbability(v))`, a lossy round trip that is idempotent for valid values.

---

## 2. `3d/range_data_inserter_3d.{h,cc}`

### API
```cpp
proto::RangeDataInserterOptions3D CreateRangeDataInserterOptions3D(LuaParameterDictionary*);
  // hit_probability (double, CHECK > 0.5), miss_probability (double, CHECK < 0.5),
  // num_free_space_voxels (int), intensity_threshold (double -> float field)
class RangeDataInserter3D {  // non-copyable
  explicit RangeDataInserter3D(const proto::RangeDataInserterOptions3D&);
  void Insert(const sensor::RangeData&, HybridGrid*, IntensityHybridGrid* /*nullable*/) const;
  // members: options_, hit_table_ = LUT(Odds(hit)), miss_table_ = LUT(Odds(miss))
};
```

### Algorithm `Insert`
1. `CHECK_NOTNULL(hybrid_grid)`.
2. **Hits**: for each return, `ApplyLookupTable(GetCellIndex(p), hit_table_)`.
3. **Misses** (no `FinishUpdate` in between, so hits take priority and a cell hit this round will not get a miss): `origin_cell = GetCellIndex(origin)`; for each return:
   - `delta = hit_cell - origin_cell` (int); `num_samples = max(|delta_i|)`; `CHECK_LT(num_samples, 1<<15)`.
   - For `position in [max(0, num_samples - num_free_space_voxels), num_samples)`: `miss_cell = origin_cell + delta * position / num_samples`. This is **integer** Eigen array math: `delta*position` component-wise, then C++ truncating division toward zero (negative deltas truncate toward 0). Rust: use `/` on i32 (also truncates toward zero). Do **not** use `div_euclid`.
   - Only the last `num_free_space_voxels` samples before the hit are updated (performance). `num_samples==0` gives no misses. The hit cell itself (`position == num_samples`) is excluded.
   - This is not a true voxel traversal (DDA). It samples on the dominant axis.
4. **Intensities** (only if grid non-null and `returns.intensities().size() > 0`): for each i, **skip if intensity > intensity_threshold**, else `AddIntensity(GetCellIndex(p_i), intensity_i)`.
5. `hybrid_grid->FinishUpdate()`.
- `range_data.misses` is **ignored** entirely in 3D.

---

## 3. `3d/submap_3d.{h,cc}`

### API
```cpp
proto::SubmapsOptions3D CreateSubmapsOptions3D(LuaParameterDictionary*);
  // high_resolution, high_resolution_max_range, low_resolution (double),
  // num_range_data (GetNonNegativeInt, CHECK_GT 0), range_data_inserter (dict)
class Submap3D : public Submap {   // Submap holds local_pose_, num_range_data_, insertion_finished_
  Submap3D(float high_res, float low_res, const Rigid3d& local_submap_pose, const VectorXf& histogram);
  explicit Submap3D(const proto::Submap3D&);   // intensity grid left nullptr
  proto::Submap ToProto(bool include_probability_grid_data) const override;
  void UpdateFromProto(const proto::Submap&) override;           // CHECK has_submap_3d
  void ToResponseProto(const Rigid3d& global_submap_pose, proto::SubmapQuery::Response*) const override;
  const HybridGrid& high_resolution_hybrid_grid() const; const HybridGrid& low_resolution_hybrid_grid() const;
  const IntensityHybridGrid& high_resolution_intensity_hybrid_grid() const;  // CHECK non-null
  void ForgetIntensityHybridGrid();                              // reset() to free memory
  const VectorXf& rotational_scan_matcher_histogram() const;
  void InsertData(const RangeData& range_data_in_local, const RangeDataInserter3D&,
                  float high_resolution_max_range, const Quaterniond& local_from_gravity_aligned,
                  const VectorXf& scan_histogram_in_gravity);  // CHECK !finished
  void Finish();                                                 // CHECK !finished; set finished
 private: unique_ptr<HybridGrid> high_, low_; unique_ptr<IntensityHybridGrid> high_intensity_; VectorXf histogram_;
};
class ActiveSubmaps3D {   // non-copyable
  explicit ActiveSubmaps3D(const proto::SubmapsOptions3D&);
  std::vector<std::shared_ptr<const Submap3D>> InsertData(const RangeData& range_data_in_local,
      const Quaterniond& local_from_gravity_aligned, const VectorXf& histogram_in_gravity);
  std::vector<std::shared_ptr<const Submap3D>> submaps() const;
 private: void AddSubmap(const Rigid3d& local_submap_pose, int histogram_size);
  options_; std::vector<std::shared_ptr<Submap3D>> submaps_; RangeDataInserter3D range_data_inserter_;
};
```

### `Submap3D::InsertData`
1. `transformed = TransformRangeData(range_data_in_local, local_pose().inverse().cast<float>())` (origin, returns, misses all transformed).
2. High-res: `Insert(FilterRangeDataByMaxRange(transformed, high_resolution_max_range), high, high_intensity)`. The filter keeps returns with `|p - origin| <= max_range` (origin = transformed origin), drops all misses, and keeps intensities (`PointCloud::copy_if` copies intensities).
3. Low-res: `Insert(transformed, low, nullptr)` (all returns, no max range).
4. `num_range_data += 1`.
5. `yaw = GetYaw(local_pose().inverse().rotation() * local_from_gravity_aligned)`; `histogram_ += RotateHistogram(scan_histogram_in_gravity, yaw)`.
- If constructed from proto, `high_intensity_` is null, and `Insert` with a null intensity grid simply skips intensities. However, `high_resolution_intensity_hybrid_grid()` CHECK-fails.

### `ActiveSubmaps3D::InsertData`
1. If `submaps_.empty() || back.num_range_data == options.num_range_data`: `AddSubmap(Rigid3d(range_data.origin.cast<double>(), local_from_gravity_aligned), histogram.size())`.
2. Insert into **every** submap in `submaps_` (at most 2).
3. If `front.num_range_data == 2*num_range_data`: `front.Finish()`.
4. Return `submaps()` (copy of shared_ptrs as const).
- `AddSubmap`: if `size >= 2`: CHECK front finished; `front.ForgetIntensityHybridGrid()`; erase front. Push a new `Submap3D(high_res, low_res, pose, VectorXf::Zero(size))`.
- Invariant: front = older (matching) submap; back = newer. The pair overlaps by `num_range_data` scans.

### `ToProto` / `UpdateFromProto`
- `ToProto`: `submap_3d.{local_pose, num_range_data, finished}`; grids only if `include_probability_grid_data`; histogram always (repeated float).
- `UpdateFromProto(Submap3D)`: set num_range_data, finished; replace grids **only if present** in proto (else keep old, which is null for the proto ctor, so a later `high_resolution_hybrid_grid()` call dereferences null; Rust: use `Option`); histogram replaced from proto.

### `ToResponseProto` (visualization, "X-ray")
- `response.submap_version = num_range_data`; adds two textures: high then low grid.
- `AddToTextureProto(grid, global_submap_pose, texture)`:
  - `ExtractVoxelData`: iterate the grid; skip cells with `ValueToProbability(v) < 0.501` (`kXrayObstructedCellProbabilityLimit`); `cell_center_global = global_pose.cast<float>() * GetCenterOfCell(idx)`; voxel = `(lround(cx/res), lround(cy/res), lround(cz/res), raw value)`. Track `min_index/max_index` of xy (init INT_MAX/INT_MIN).
  - `width = max.y - min.y + 1`, `height = max.x - min.x + 1` (if no voxels these overflow to weird values; same in C++).
  - `AccumulatePixelData(width, height, min, max, voxels)`: skip out-of-bounds; `x = max.x - px`, `y = max.y - py`; pixel `[x*width + y]`: `count++`, `min_z/max_z`, `probability_sum += ValueToProbability(v)`, `max_probability = max(max_probability(init 0.5), p)`.
  - `ComputePixelValues`: per pixel, `z_diff = count>0 ? max_z-min_z : 0`; if `z_diff < 3` push (0,0). Else `free_space = max(z_diff - count, 0)`, `fsw = 0.15*free_space`, `total = count + fsw`, `fsp = 1 - max_probability`, `avg = ClampProbability((sum + fsp*fsw)/total)`, `delta = 128 - ProbabilityToLogOddsInteger(avg)`, `alpha = delta>0 ? 0 : -delta`, `value = delta>0 ? delta : 0`; push `value`, then `(value||alpha) ? alpha : 1`.
  - `cells = FastGzipString(cell_data)`, `resolution`, `width`, `height`, `slice_pose = ToProto(global_pose.inverse() * Translation(max.x*res, max.y*res, global_pose.translation().z()))`.

---

## 4. `internal/3d/imu_integration.h`

```cpp
template<T> struct IntegrateImuResult { Matrix<T,3,1> delta_velocity, delta_translation; Quaternion<T> delta_rotation; };
template<T,RangeType,IteratorType> IntegrateImuResult<T> IntegrateImu(const RangeType& imu_data,
    const Transform<T,3,Affine>& linear_acceleration_calibration, const Transform<T,3,Affine>& angular_velocity_calibration,
    Time start_time, Time end_time, IteratorType* it);
template<RangeType,IteratorType> IntegrateImuResult<double> IntegrateImu(imu_data, start, end, it); // identity calibrations
template<T> struct ExtrapolatePoseResult { Rigid3<T> pose; Matrix<T,3,1> velocity; };
template<T,R,I> ExtrapolatePoseResult<T> ExtrapolatePoseWithImu(const Rigid3<T>& prev_from_tracking,
    const Matrix<T,3,1>& prev_velocity_in_tracking, Time prev_time, const Matrix<T,3,1>& gravity, Time time, const R& imu, I* imu_it);
template<T,R,I> ExtrapolatePoseResult<T> ExtrapolatePoseWithImu(const Rigid3<T>& prev, Time prev_time,
    const Rigid3<T>& prev_prev, Time prev_prev_time, const Matrix<T,3,1>& gravity, Time time, const R& imu, I* imu_it);
```

### `IntegrateImu` (zero-order hold, forward Euler)
- Preconditions (CHECK): `start <= end`; `*it != end`; `(*it)->time <= start`; if `next(*it)` exists then `next(*it)->time > start`.
- Init: `dv = 0`, `dp = 0`, `dq = I`. Loop while `current < end`:
  - `next_imu = next(*it)` exists ? its time : `Time::max()`; `next_time = min(next_imu, end)`; `dt = ToSeconds(next_time - current)`.
  - `dq *= AngleAxisVectorToRotationQuaternion((ang_cal * w_it) * dt)` (right-multiply, body frame).
  - `dv += dq * ((acc_cal * a_it) * dt)` (uses the **updated** dq).
  - `dp += dt * dv` (uses the **updated** dv).
  - `current = next_time`; if `current == next_imu` then `++*it`.
- The iterator is advanced in place (in/out). Callers reuse it across consecutive intervals.
- `AngleAxisVectorToRotationQuaternion(v)`: if `|v|^2 > 1e-8`: `w = cos(|v|/2)`, `xyz = sin(|v|/2)/|v| * v`; else `w=1`, `xyz = 0.5 v` (not normalized).

### `ExtrapolatePoseWithImu` (velocity form)
- `r = IntegrateImu(identity calibrations, prev_time, time)`; `dt = time - prev_time`.
- `translation = p_prev + R_prev*(dt*v_prev_tracking + r.dp) - 0.5*dt^2*gravity`.
- `rotation = R_prev * r.dq`; `velocity = R_prev*(v_prev_tracking + r.dv) - dt*gravity`. Note the returned velocity is in the parent (gravity/map) frame while the input velocity is in the tracking frame (asymmetric).
- Two-pose form: `v_prev_tracking = R_prev^{-1} * (p_prev - p_prev_prev) / (prev_time - prev_prev_time)`, then delegates.

---

## 5. `internal/3d/rotation_parameterization.h`

Ceres `AutoDiffLocalParameterization` plus-functors (quaternion stored `[w,x,y,z]`):

| Functor | Local size | Plus | Used by |
|---|---|---|---|
| `YawOnlyQuaternionPlus` | 1 | `c = clamp(δ0, -0.5, 0.5)`; `q_δ = [sqrt(1-c^2), 0, 0, c]`; `x⊕δ = q_δ ⊗ x` (**left**: yaw about the parent z axis; δ is sin(half-angle)) | `CeresScanMatcher3D` when `only_optimize_yaw` |
| `ConstantYawQuaternionPlus` | 2 | `n = sqrt(δ0^2+δ1^2)`; `s = n<1e-6 ? 1 : sin(n)/n`; `q_δ = [n<1e-6 ? 1 : cos(n), s*δ0, s*δ1, 0]`; `x⊕δ = x ⊗ q_δ` (**right**: roll/pitch in the local frame, so yaw is constant) | `ImuBasedPoseExtrapolator` last node; also optimization_problem_3d (other team) |

- Ceres `QuaternionParameterization` (used elsewhere): `n = |δ|` (3-vector); if n==0 then `x⊕δ = x`; else `q_δ = [cos n, sin(n)/n * δ]`, `x⊕δ = q_δ ⊗ x` (left multiply, **no 1/2 factor**). Analytic PlusJacobian at δ=0: rows `[-x1 -x2 -x3; x0 x3 -x2; -x3 x0 x1; x2 -x1 x0]`.
- `common::QuaternionProduct(z(double*), w(T*), zw)` is a Hamilton product with `[w,x,y,z]` order; `ceres::QuaternionProduct` is the same.

---

## 6. `internal/3d/scan_matching/interpolated_grid.h`

```cpp
template<class HybridGridType> class InterpolatedGrid {   // holds const HybridGridType& ; non-copyable
  explicit InterpolatedGrid(const HybridGridType&);
  template<T> T GetInterpolatedValue(const T& x, const T& y, const T& z) const;
};
using InterpolatedIntensityGrid = InterpolatedGrid<IntensityHybridGrid>;   // GetValue = GetIntensity (0 if empty)
using InterpolatedProbabilityGrid = InterpolatedGrid<HybridGrid>;          // GetValue = GetProbability (unknown -> 0.1)
```

Algorithm (piecewise tricubic Hermite with zero end-slopes, C1 continuous):
1. `CenterOfLowerVoxel(x,y,z)` uses the **scalar part** (`jet.a`) when T is a Jet (overload: a non-template `double` version, plus a template that forwards `.a`). Center of the containing cell is `GetCenterOfCell(GetCellIndex(Vector3f(x,y,z)))` (computed in float). Per axis: if `center > coord`, subtract `resolution`. Result: the largest voxel center `<= coord`.
2. `x1,y1,z1 = lower` (float to double); `x2 = lower.x + resolution` (float add, then to double).
3. `index1 = GetCellIndex(Vector3f(x1,y1,z1))`; read 8 corner values `q{ijk}` with `i,j,k in {1,2}` meaning offsets `(i-1, j-1, k-1)` on (x,y,z); values are `float` then promoted to double (constants with respect to the Jet).
4. `nx = (x - x1)/(x2 - x1)` etc. (T arithmetic; derivatives flow only through `x,y,z`).
5. Blend `f(A,B,t) = (A-B)*2t^3 + (B-A)*3t^2 + A`. Order: z first: `q11=f(q111,q112,nz)`, `q12=f(q121,q122,nz)`, `q21=f(q211,q212,nz)`, `q22=f(q221,q222,nz)`; then y: `q1=f(q11,q12,ny)`, `q2=f(q21,q22,ny)`; then x: `f(q1,q2,nx)`.
- At grid points the interpolation equals the voxel value exactly; between points it is monotonic (tests 17.6).
- Gradient vanishes at voxel centers: a scan exactly on centers has zero gradient (known Cartographer property).

---

## 7. Cost functions (all Ceres `AutoDiffCostFunction`, owned by the `ceres::Problem`)

| Functor | Residuals | Params | Residual |
|---|---|---|---|
| `OccupiedSpaceCostFunction3D` | `ceres::DYNAMIC` = N points | translation[3], rotation[4] (w,x,y,z) | `r_i = s * (1 - P_interp(T * p_i))` |
| `IntensityCostFunction3D` | DYNAMIC = N | translation[3], rotation[4] | if `intensity_i > threshold`: `r_i = 0`; else `r_i = s * (I_interp(T*p_i) - intensity_i)` |
| `TranslationDeltaCostFunctor3D` | 3 | translation[3] | `s * (t - target)` |
| `RotationDeltaCostFunctor3D` | 3 | rotation[4] | `δ = target^{-1} ⊗ q` (via `common::QuaternionProduct`, target inverse = conjugate stored as doubles); `r = s * δ.xyz` so `|r|^2 = s^2 sin^2(θ/2)` (if q is unit) |

- Occupied/intensity: `transform = Rigid3<T>(Map<translation>, Quaternion<T>(r[0],r[1],r[2],r[3]))`; Eigen `Quaternion(w,x,y,z)` ctor. The quaternion is not normalized inside (the parameterization keeps it unit). Eigen rotation of a vector by a non-unit quaternion uses the formula `v + 2w(u×v) + 2u×(u×v)`, which is not a scaled rotation. Rust port must use the **same formula** (nalgebra `UnitQuaternion` assumes unit; for jets implement manually).
- They hold `const PointCloud&` and `InterpolatedGrid` (holding `const Grid&`): **borrowed** data must outlive `ceres::Solve`. Rust: lifetimes `'a` on cost structs.
- `IntensityCostFunction3D` requires `point_cloud.intensities()` populated (indexes `intensities()[i]` unchecked).
- Constructors are private; only the `CreateAutoDiffCostFunction` factory exists. `IntensityCostFunction3D::CreateAutoDiffCostFunction` is defined in the `.cc` (same body pattern: `AutoDiffCostFunction<IntensityCostFunction3D, DYNAMIC, 3, 4>(new ..., point_cloud.size())`).

---

## 8. `ceres_scan_matcher_3d.{h,cc}`

### API
```cpp
proto::CeresScanMatcherOptions3D CreateCeresScanMatcherOptions3D(LuaParameterDictionary*);
  // occupied_space_weight_0.._k (loop until !HasKey), intensity_cost_function_options_0.._k
  //   {weight, huber_scale, intensity_threshold}, translation_weight, rotation_weight,
  //   only_optimize_yaw, ceres_solver_options {use_nonmonotonic_steps, max_num_iterations, num_threads}
struct PointCloudAndHybridGridsPointers { const PointCloud* point_cloud; const HybridGrid* hybrid_grid;
                                          const IntensityHybridGrid* intensity_hybrid_grid; /*nullable*/ };
class CeresScanMatcher3D {
  explicit CeresScanMatcher3D(const proto::CeresScanMatcherOptions3D&);  // solver opts + linear_solver_type = DENSE_QR
  void Match(const Vector3d& target_translation, const Rigid3d& initial_pose_estimate,
             const std::vector<PointCloudAndHybridGridsPointers>&, Rigid3d* pose_estimate,
             ceres::Solver::Summary* summary) const;
};
```

### Problem construction in `Match`
1. `ceres::Problem problem;` `CeresPose ceres_pose(initial_pose_estimate, nullptr /*translation: no parameterization*/, only_optimize_yaw ? AutoDiffLocalParameterization<YawOnlyQuaternionPlus,4,1> : QuaternionParameterization, &problem)`. `CeresPose` stores `{array<double,3> t; array<double,4> q(w,x,y,z)}` in a `shared_ptr` and calls `AddParameterBlock` for each.
2. `CHECK_EQ(occupied_space_weight_size, inputs.size())`. For each i: `CHECK_GT(w_i, 0)`; add `OccupiedSpace(w_i / sqrt(N_i), cloud_i, grid_i)`, no loss, params `(t, q)`. If `intensity_hybrid_grid` is non-null: CHECK `huber_scale>0`, `weight>0`, `intensity_threshold>0` of `intensity_cost_function_options(i)`; add `Intensity(weight/sqrt(N_i), threshold, cloud_i, igrid_i)` with **`new ceres::HuberLoss(huber_scale)`**.
3. `CHECK_GT(translation_weight,0)`: `TranslationDelta(translation_weight, target_translation)` on `t`.
4. `CHECK_GT(rotation_weight,0)`: `RotationDelta(rotation_weight, initial_pose_estimate.rotation())` on `q`.
5. `ceres::Solve(options, &problem, summary)`; `*pose_estimate = ceres_pose.ToRigid()` (`Rigid3d::FromArrays(q, t)`; it is not re-normalized explicitly).
- Note: the translation prior targets `target_translation` (the extrapolator prediction), while the rotation prior and initial value use `initial_pose_estimate` (possibly refined by RTCSM).
- Solver options from proto: only `use_nonmonotonic_steps`, `max_num_iterations`, `num_threads`; all else Ceres defaults (LM trust region, `function_tolerance=1e-6`, `gradient_tolerance=1e-10`, `parameter_tolerance=1e-8`, `initial_trust_region_radius=1e4`, ...). The Rust solver must reproduce LM + DENSE_QR semantics closely to keep test tolerances.

---

## 9. `real_time_correlative_scan_matcher_3d.{h,cc}`

```cpp
class RealTimeCorrelativeScanMatcher3D {  // options: linear_search_window, angular_search_window,
                                           // translation_delta_cost_weight, rotation_delta_cost_weight
  explicit RealTimeCorrelativeScanMatcher3D(const proto::RealTimeCorrelativeScanMatcherOptions&);
  float Match(const Rigid3d& initial_pose_estimate, const PointCloud&, const HybridGrid&, Rigid3d* pose_estimate) const;
};
// options factory lives in internal/scan_matching/real_time_correlative_scan_matcher.h (shared with 2D)
```
- `GenerateExhaustiveSearchTransforms(res, cloud)`: `L = lround(linear_window/res)`; `max_range = max(3*res, max |p|)`; `step = (1 - 1e-3) * acos(1 - res^2/(2*max_range^2))` (float); `A = lround(angular_window/step)`. Loops nested **z, y, x, rz, ry, rx** each over `[-L..L]` / `[-A..A]`; transform = `Rigid3f(Vector3f(x,y,z)*res, AngleAxisVectorToRotationQuaternion(Vector3f(rx,ry,rz)*step))`. Count = `(2L+1)^3 (2A+1)^3` (with default 0.15/0.1 m and 1 degree this can be very large; materialized as a Vec).
- `Match`: `best=-1`; for each δ: `candidate = initial.cast<float>() * δ`; `score = ScoreCandidate(grid, TransformPointCloud(cloud, candidate), δ)`; strict `>` update so the first maximum wins; `*pose_estimate = candidate.cast<double>()`. Returns best score.
- `ScoreCandidate`: mean of `GetProbability(GetCellIndex(p))` (nearest voxel; unknown = 0.1) times `exp(-(|δ.t|*w_t + GetAngle(δ)*w_r)^2)`; `CHECK_GT(score, 0)`. `GetAngle(r) = 2*atan2(|q.vec|, |q.w|)`.
- Single-threaded; all float math.

---

## 10. `rotational_scan_matcher.{h,cc}`

```cpp
class RotationalScanMatcher {
  static VectorXf RotateHistogram(const VectorXf& histogram, float angle);
  static VectorXf ComputeHistogram(const PointCloud& point_cloud, int histogram_size);
  explicit RotationalScanMatcher(const VectorXf* histogram);  // borrowed; must outlive
  std::vector<float> Match(const VectorXf& histogram, float initial_angle, const std::vector<float>& angles) const;
};
constexpr float kMinDistance = 0.2f, kMaxDistance = 0.9f, kSliceHeight = 0.2f;
```

### `ComputeHistogram` (input: gravity-aligned cloud)
1. `slices: std::map<int, PointCloud>` keyed by `lround(z / 0.2)`, **ascending key order**, points in input order.
2. For each slice: `AddPointCloudSliceToHistogram(SortSlice(slice))`.
- `SortSlice`: `c = centroid(slice)`; drop points with `|(p-c).xy| < 0.2`; sort by `atan2((p-c).y, (p-c).x)` with `std::sort` (**unstable**; ties are implementation-defined; Rust `sort_by` is stable; differences only for exact ties).
- `AddPointCloudSliceToHistogram(sorted)`: return if empty. `c = centroid(sorted)` (**recomputed on the filtered slice**). `last = sorted.front()`. For each p: `delta = (p - last).xy`, `dir = (p - c).xy`, `d = |delta|`;
  - if `d < 0.2 || |dir| < 0.2`: continue (last **not** updated);
  - if `d > 0.9`: `last = p`; continue;
  - else `angle = atan2(delta.y, delta.x)`, `value = max(0, 1 - |deltâ · dir̂|)`, `AddValueToHistogram(angle, value)` (last **not** updated: the reference point only advances after a gap > 0.9 m; replicate exactly).
- `AddValueToHistogram`: wrap angle into `[0, π]` (while `> π` subtract π, while `< 0` add π; float π); `bucket = clamp(lround(n*angle/π - 0.5), 0, n-1)`; `h[bucket] += value`.

### `RotateHistogram(h, angle)`
- If size 0, return h. `rb = -angle * n / M_PI` (float*long gives float, then / double gives double, stored as float). `full = lround(rb - 0.5)`; `frac = rb - full`; while `full < 0`: `full += n`. `h0[i] = h[(i+full) % n]`, `h1[i] = h[(i+1+full) % n]`; return `frac*h1 + (1-frac)*h0`. A shift by `n` buckets equals π (180 degrees), so the histogram has period π.

### `Match`
- For each `a` in `angles`: `s = RotateHistogram(histogram, initial_angle + a)`; score = `MatchHistograms(*histogram_, s)` = `dot / (|submap|*|scan|)`, or **1.0** if the norm product `< 1e-3` (empty histograms always pass).

---

## 11. `precomputation_grid_3d.{h,cc}` and `low_resolution_matcher.{h,cc}`

```cpp
class PrecomputationGrid3D : public HybridGridBase<uint8> {
  explicit PrecomputationGrid3D(float resolution);
  static float ToProbability(float value);   // 0.1 + value * (0.8/255)
};
PrecomputationGrid3D ConvertToPrecomputationGrid(const HybridGrid&);
PrecomputationGrid3D PrecomputeGrid(const PrecomputationGrid3D& grid, bool half_resolution, const Array3i& shift);
std::function<float(const Rigid3f&)> CreateLowResolutionMatcher(const HybridGrid* low_res_grid, const PointCloud* points);
```
- `ConvertToPrecomputationGrid`: iterate known cells: `v = lround((ValueToProbability(raw) - 0.1) * (255/0.8))`, CHECK `0..255`, store. Unknown stays 0 (maps to 0.1 = unknown probability, consistent).
- `PrecomputeGrid`: result has the **same `resolution()`** even when `half_resolution` (only indices are halved; resolution is metadata only). For each non-zero cell `c` and each octant `o in 0..8`: `target = c - shift * GetOctant(o)`; if half: `target >>= 1` per component (arithmetic shift = floor division); `result[target] = max(result[target], v)`. Equivalent: `result[i] = max over o of grid[i + shift*o]` (full-res case), a forward-looking max over a 2x2x2 dilation with stride `shift`.
- Zero cells are skipped (they cannot raise a max).
- `CreateLowResolutionMatcher`: closure capturing raw pointers; returns `mean_i GetProbability(GetCellIndex(pose * p_i))` (nearest voxel, unknown = 0.1). Division by `points->size()`: an empty cloud gives NaN.

---

## 12. `fast_correlative_scan_matcher_3d.{h,cc}` (3D branch and bound; used by constraint builder)

### API
```cpp
proto::FastCorrelativeScanMatcherOptions3D CreateFastCorrelativeScanMatcherOptions3D(LuaParameterDictionary*);
  // branch_and_bound_depth, full_resolution_depth (int), min_rotational_score, min_low_resolution_score,
  // linear_xy_search_window, linear_z_search_window, angular_search_window (double)
class PrecomputationGridStack3D { PrecomputationGridStack3D(const HybridGrid&, const Options&);
  const PrecomputationGrid3D& Get(int depth) const; int max_depth() const; /* = size-1 */ };
using MatchingFunction = std::function<float(const Rigid3f&)>;
class FastCorrelativeScanMatcher3D {
  struct Result { float score; Rigid3d pose_estimate; float rotational_score; float low_resolution_score; };
  FastCorrelativeScanMatcher3D(const HybridGrid& hybrid_grid, const HybridGrid* low_resolution_hybrid_grid,
      const VectorXf* rotational_scan_matcher_histogram, const Options&);
  std::unique_ptr<Result> Match(const Rigid3d& global_node_pose, const Rigid3d& global_submap_pose,
      const TrajectoryNode::Data& constant_data, float min_score) const;
  std::unique_ptr<Result> MatchFullSubmap(const Quaterniond& global_node_rotation,
      const Quaterniond& global_submap_rotation, const TrajectoryNode::Data&, float min_score) const;
 private: options_, resolution_ (=grid.resolution()), width_in_voxels_ (=grid.grid_size(), i.e. 64<<bits at construction),
          unique_ptr<PrecomputationGridStack3D>, const HybridGrid* low_resolution_hybrid_grid_, RotationalScanMatcher.
};
struct DiscreteScan3D { Rigid3f pose; std::vector<std::vector<Array3i>> cell_indices_per_depth; float rotational_score; };
struct Candidate3D { int scan_index; Array3i offset; float score = -inf; float low_resolution_score = 0;
                     static Unsuccessful() /* (0, 0) */; operator< / operator> on score };
```
`SearchParameters { int linear_xy_window_size; int linear_z_window_size; double angular_search_window; const MatchingFunction* low_resolution_matcher; }`

### Precomputation stack (D = branch_and_bound_depth, F = full_resolution_depth; CHECK D>=1, F>=1)
- `grids[0] = ConvertToPrecomputationGrid(hybrid_grid)`. `last_width = 1`.
- For `depth = 1 .. D-1`: `half = depth >= F`; `next_width = 1<<depth`; `k = 1 << max(0, depth - F)`; `shift = (next_width - last_width + (k-1)) / k` (int, all components equal); `grids[depth] = PrecomputeGrid(grids[depth-1], half, shift)`; `last_width = next_width`.
- So for depth < F the grid is full-res, with each cell the max over the `2^depth`-cube `[i, i+2^depth)`. From depth F onward, each level is halved once (index space `>> (depth-F+1)`).
- `max_depth = D-1`.

### `Match`
- `low_res_matcher = CreateLowResolutionMatcher(low_res_grid, &constant_data.low_resolution_point_cloud)`.
- `SearchParameters{ lround(xy_window/res), lround(z_window/res), angular_search_window, &matcher }`.
- `MatchWithSearchParameters(params, node.cast<float>(), submap.cast<float>(), constant_data.high_resolution_point_cloud, constant_data.rotational_scan_matcher_histogram, constant_data.gravity_alignment, min_score)`.

### `MatchFullSubmap`
- `max_d = max |p|` over the high-res cloud; `L = (width_in_voxels_+1)/2 + lround(max_d/res + 0.5)`; params `{L, L, M_PI, &matcher}`; poses = pure rotations `Rigid3f::Rotation(q.cast<float>())`.

### `GenerateDiscreteScans`
- `max_range = max(3*res, max |p|)`; `step = (1 - 1e-2) * acos(1 - res^2/(2 max_range^2))` (**safety 1e-2 here vs 1e-3 in RTCSM**); `A = lround(angular_window/step)`; `angles = [rz*step for rz in -A..=A]`.
- `node_to_submap = submap^{-1} * node`.
- `scores = rotational_matcher.Match(scan_histogram, GetYaw(node_to_submap.rotation() * gravity_alignment.inverse().cast<float>()), angles)`.
- For each i with `scores[i] >= min_rotational_score`: `pose = Rigid3f(node_to_submap.translation(), submap.rot^{-1} * AA(0,0,angles[i]) * node.rot)` (yaw applied in the map frame about the scan origin); push `DiscretizeScan(params, cloud, pose, scores[i])`.

### `DiscretizeScan`
- `full = [grids[0].GetCellIndex(pose * p) for p]`.
- `frd = min(F, D)`; the first `frd` depth entries are copies of `full`.
- `start = (-xy, -xy, -z)`; for `i in 0 .. D - frd`: `e = i+1`; per cell: `((cell + start) >> e) - (start >> e)` (arithmetic shifts). Total `D` depth entries.

### `GenerateLowestResolutionCandidates`
- `step = 1 << max_depth`; `n_xy = (2*xy + step)/step`, `n_z = (2*z + step)/step`; loops `scan_index`, then **z, y, x** from `-window` to `window` in steps of `step`; `CHECK_EQ(count, scans * n_xy^2 * n_z)`.

### `ScoreCandidates(depth, scans, candidates)`
- `e = max(0, depth - F + 1)`; for each candidate: `off = offset >> e`; `sum (int) = Σ grids[depth].value(cell + off)` over `cell_indices_per_depth[depth]` (CHECK depth < size); `score = ToProbability(sum / (float)count)`.
- Then `std::sort(..., std::greater<>)`, i.e. descending by score, **unstable**. Tie order affects which equal-score candidate is returned; for bit-exact parity sort by `(score desc)` with a deterministic tiebreak, and accept that C++ is implementation-defined.

### `BranchAndBound(params, scans, candidates, depth, min_score)`
- **depth == 0**: iterate sorted candidates; if `score <= min_score`, return `Unsuccessful()` (score -inf); `lr = (*low_res_matcher)(GetPoseFromCandidate)`; if `lr >= min_low_resolution_score`, return candidate with `low_resolution_score = lr`. After the loop, return `Unsuccessful()`.
- **depth > 0**: `best = Unsuccessful(); best.score = min_score;` for each candidate (sorted): if `score <= min_score`, **break** (`min_score` here is the function argument and does not tighten during the loop; the recursive call receives `best.score`). Children: `half = 1<<(depth-1)`; nested `z in {0,half}` (break if `off.z + z > z_window`), `y in {0,half}` (break if `> xy`), `x in {0,half}` (break if `> xy`); push `(scan_index, offset + (x,y,z))`. `ScoreCandidates(depth-1, children)`; `best = std::max(best, BranchAndBound(children, depth-1, best.score))`. `std::max(a,b)` returns `a` unless `a < b` (ties keep the existing best).
- Result: `MatchWithSearchParameters` returns `Result{score, GetPoseFromCandidate(...).cast<double>(), scans[idx].rotational_score, low_resolution_score}` iff `best.score > min_score`, else nullptr.
- `GetPoseFromCandidate = Translation(res * offset) * scans[scan_index].pose`.
- Threading: `const` and re-entrant. The constraint builder (other team) calls it from thread-pool workers concurrently on the same matcher. Rust: `Sync` (no interior mutability), `&self` methods.

---

## 13. `local_trajectory_builder_3d.{h,cc}` and options

### Options (`CreateLocalTrajectoryBuilderOptions3D`)
Keys: `min_range, max_range (float), num_accumulated_range_data (int), voxel_filter_size, high_resolution_adaptive_voxel_filter{max_length,min_num_points,max_range}, low_resolution_adaptive_voxel_filter{...}, use_online_correlative_scan_matching, real_time_correlative_scan_matcher{...}, ceres_scan_matcher{...}, motion_filter{max_time_seconds,max_distance_meters,max_angle_radians}, pose_extrapolator{use_imu_based, constant_velocity{imu_gravity_time_constant,pose_queue_duration}, imu_based{...}}, imu_gravity_time_constant, rotational_histogram_size, submaps{...}, use_intensities`. Proto also has `initial_poses` (repeated TimestampedTransform) and `initial_imu_data` (repeated ImuData), which are not read from Lua.

Defaults (`configuration_files/trajectory_builder_3d.lua`): min_range 1, voxel 0.15, hi-res AVF {2.0, 150, 15}, low-res AVF {4.0, 200, MAX}, RTCSM {0.15, 1 deg, 0.1, 0.1}, ceres {occ_w0=1, occ_w1=6, intensity_0 {0.5, 0.3, INTENSITY_THRESHOLD}, translation 5, rotation 4e2, yaw-only false, solver {false, 12 iters, 1 thread}}, motion filter {0.5 s, 0.1 m, 0.004 rad}, histogram 120, submaps {hi 0.10, hi_max_range 20, low 0.45, num_range_data 160, inserter {0.55, 0.49, 2 voxels}}. Pose graph FCSM3D defaults: depth 8, full_res_depth 3, min_rot 0.77, min_low_res 0.55, xy 5 m, z 1 m, 15 deg.

### API
```cpp
class LocalTrajectoryBuilder3D {   // non-copyable
  struct InsertionResult { shared_ptr<const TrajectoryNode::Data> constant_data; vector<shared_ptr<const Submap3D>> insertion_submaps; };
  struct MatchingResult { Time time; Rigid3d local_pose; RangeData range_data_in_local; unique_ptr<const InsertionResult> insertion_result /*null if motion-filtered*/; };
  LocalTrajectoryBuilder3D(const proto::LocalTrajectoryBuilderOptions3D&, const vector<string>& expected_range_sensor_ids);
  void AddImuData(const sensor::ImuData&);
  unique_ptr<MatchingResult> AddRangeData(const string& sensor_id, const sensor::TimedPointCloudData&);
  void AddOdometryData(const sensor::OdometryData&);
  static void RegisterMetrics(metrics::FamilyFactory*);
};
```
Members: `options_`, `ActiveSubmaps3D active_submaps_`, `MotionFilter`, `unique_ptr<RTCSM3D>`, `unique_ptr<CeresScanMatcher3D>`, `unique_ptr<PoseExtrapolatorInterface> extrapolator_` (null until first IMU), `num_accumulated_`, `vector<TimedPointCloudOriginData> accumulated_...`, `optional<steady_clock::time_point> last_wall_time_`, `optional<double> last_thread_cpu_time_seconds_`, `RangeDataCollator`, `optional<Time> last_sensor_time_`.

### `AddImuData`
- If the extrapolator exists, forward. Else build `initial_poses` and `initial_imu_data` from options, append this IMU sample, and `extrapolator_ = PoseExtrapolatorInterface::CreateWithImuData(...)` (`use_imu_based` selects `ImuBasedPoseExtrapolator::InitializeWithImu`, else `PoseExtrapolator::InitializeWithImu(pose_queue_duration, imu_gravity_time_constant, imu.back())`). The first IMU sample is only used for initialization.

### `AddRangeData` (step by step)
1. If `use_intensities`: CHECK `ranges.size() == intensities.size()`.
2. `sync = range_data_collator_.AddRangeData(id, data)`; empty: log, return null.
3. No extrapolator: return null.
4. CHECK `sync.ranges.back().point_time.time <= 0`. `t_first = sync.time + ranges.front().time`; if `< extrapolator.GetLastPoseTime()`, return null.
5. If `num_accumulated_ == 0`, clear the accumulator. `sync.ranges = VoxelFilter(sync.ranges, 0.5 * voxel_filter_size)`; push (moved); `++num_accumulated_`; if `< num_accumulated_range_data`, return null; reset to 0.
6. `hit_times`: start `prev = extrapolator.GetLastExtrapolatedTime()`; for every point in order `t = data.time + point.time`; if `t < prev`: log once and clamp `t = prev`; push; `prev = t`. Then push `accumulated.back().time`.
7. `res = extrapolator.ExtrapolatePosesWithGravity(hit_times)`; `hits_poses = res.previous_poses` + `res.current_pose.cast<float>()`; CHECK sizes equal.
8. For each point with pose `P_i`: `hit = P_i * p`; `origin = P_i * origins.at(origin_index)`; `range = |hit-origin|`; if `range >= min_range`: if `<= max_range` keep hit (+ intensity if `use_intensities`), else `misses.push(origin + max_range/range * delta)`. CHECK the iterator ends at the last pose (the `current_pose` entry is unused for points).
9. `sensor_duration = sync.time - last_sensor_time_` (if any). Note: `synchronized_data.time` is read **after** `std::move(sync)`; valid in C++ because `Time` is trivially copyable. In Rust, capture `time` before moving.
10. `current_time = hit_times.back()`. `filtered = RangeData{ current_pose.translation().cast<float>(), VoxelFilter(returns, voxel_size), VoxelFilter(misses, voxel_size) }`. Metric: voxel filter fraction.
11. `AddAccumulatedRangeData(current_time, TransformRangeData(filtered, current_pose.inverse()), sensor_duration, current_pose, res.gravity_from_tracking)`.

### `AddAccumulatedRangeData`
1. Empty returns: warn, null.
2. `hi = AdaptiveVoxelFilter(returns, hi_opts)`; empty: null. `lo = AdaptiveVoxelFilter(returns, lo_opts)`; empty: null.
3. `pose = ScanMatch(pose_prediction, lo, hi)` (never null in practice). `extrapolator.AddPose(time, pose)`.
4. `in_local = TransformRangeData(in_tracking, pose.cast<float>())`.
5. `insertion = InsertIntoSubmap(time, in_local, in_tracking, hi, lo, pose, gravity_alignment)`.
6. Metrics (latency, real-time ratio, CPU ratio via `common::GetThreadCpuTimeSeconds()`).
7. Return `MatchingResult{time, pose, in_local, insertion}`.

### `ScanMatch`
- No submaps: return the prediction. `submap = submaps().front()` (the older one). `initial = submap.local_pose^{-1} * prediction`.
- If `use_online_correlative_scan_matching`: `RTCSM.Match(initial, hi, submap.high_grid, &initial)` (in-place refine; score to metric).
- `ceres.Match(target_translation = (submap^{-1} * prediction).translation(), initial, {{&hi, &submap.high, use_intensities ? &submap.high_intensity : null}, {&lo, &submap.low, null}}, &pose_in_submap, &summary)`. So two occupied-space weights are required (`occupied_space_weight_0` for hi, `_1` for lo), and `intensity_cost_function_options_0` is required when `use_intensities`.
- Metrics: final cost, residual distance and angle (`angularDistance`). Return `submap.local_pose * pose_in_submap`.

### `InsertIntoSubmap`
- If `motion_filter.IsSimilar(time, pose)`: null.
- `hist = ComputeHistogram(TransformPointCloud(in_tracking.returns, Rigid3f::Rotation(gravity_alignment.cast<float>())), rotational_histogram_size)` (uses the full voxel-filtered returns, not the AVF clouds).
- `local_from_gravity_aligned = pose.rotation() * gravity_alignment.inverse()`.
- `submaps = active_submaps_.InsertData(in_local, local_from_gravity_aligned, hist)`.
- `TrajectoryNode::Data{time, gravity_alignment, {} /*2D only*/, hi, lo, hist, pose}`.

### Metrics
Gauges: `mapping_3d_local_trajectory_builder_{latency, voxel_filter_fraction, scan_matcher_fraction, insert_into_submap_fraction, real_time_ratio, cpu_real_time_ratio}`. Histograms: `..._scores` (FixedWidth(0.05, 20), label scan_matcher=real_time_correlative), `..._costs` (ScaledPowersOf(2, 0.01, 100), scan_matcher=ceres), `..._residuals` (ScaledPowersOf(2, 0.01, 10), component=distance|angle). Static globals default to `Null()`. Rust: use a `OnceLock` registry, or inject.

### `local_slam_result_3d.{h,cc}`
- `LocalSlamResult3D : LocalSlamResultData` (sensor_id, proto `LocalSlamResultData`, `SubmapController<Submap3D>*`).
- `AddToTrajectoryBuilder(tb)`: `tb->AddLocalSlamResultData(make_unique<LocalSlamResult3D>(*this))` (copy).
- `AddToPoseGraph(id, pg)`: CHECK at least 1 submap and `submaps(0).has_submap_3d()`; `submaps = [controller->UpdateSubmap(p) for p]`; `static_cast<PoseGraph3D*>(pg)->AddNode(make_shared<Data>(FromProto(node_data)), id, submaps)`. Coupled with pose_graph_3d (other team).

---

## 14. `internal/imu_based_pose_extrapolator.{h,cc}`

### API
```cpp
class ImuBasedPoseExtrapolator : public PoseExtrapolatorInterface {
  explicit ImuBasedPoseExtrapolator(const proto::ImuBasedPoseExtrapolatorOptions&);
  ~ImuBasedPoseExtrapolator() override;   // LOGs iteration histogram
  static unique_ptr<PoseExtrapolatorInterface> InitializeWithImu(const Options&, const vector<ImuData>&, const vector<TimestampedTransform>& initial_poses);
  Time GetLastPoseTime() const override;        // Time::min() if empty
  Time GetLastExtrapolatedTime() const override;
  void AddPose(Time, const Rigid3d&) override;
  void AddImuData(const ImuData&) override;      // CHECK time >= last pose time
  void AddOdometryData(const OdometryData&) override; // same CHECK
  Rigid3d ExtrapolatePose(Time) override;        // = ExtrapolatePosesWithGravity({t}).current_pose (runs a full solve)
  ExtrapolationResult ExtrapolatePosesWithGravity(const vector<Time>&) override;
  Quaterniond EstimateGravityOrientation(Time) override;  // .gravity_from_tracking (full solve)
 private: deque<TimestampedTransform> timed_pose_queue_, previous_solution_; deque<ImuData> imu_data_;
  deque<OdometryData> odometry_data_; Time last_extrapolated_time_ = min; Rigid3d gravity_from_local_ = I;
  options_; ceres::Solver::Options solver_options_; common::Histogram num_iterations_hist_;
};
```
Options: `pose_queue_duration, gravity_constant, pose_translation_weight, pose_rotation_weight, imu_acceleration_weight, imu_rotation_weight, solver_options, odometry_translation_weight, odometry_rotation_weight`.

### Initialization
- CHECK imu non-empty; copy all IMU into `imu_data_`. If `initial_poses` is non-empty: `AddPose` for those with `time > imu.front().time`. Else `AddPose(imu.back().time, Rotation(Quaterniond::FromTwoVectors(imu.back().linear_acceleration, UnitZ)))`.

### Queue maintenance
- `AddPose`: push; while `size > 3 && queue[1].time <= t - pose_queue_duration`: if `previous_solution_` is non-empty, CHECK front times equal and pop it; pop the queue front. Then `TrimImuData`.
- `TrimDequeData`: while `size > 1 && queue non-empty && data[1].time <= queue.front().time`: pop front (keeps one sample at or before the first pose).

### `ExtrapolatePosesWithGravity(times)` (own Ceres problem)
1. `time = times.back()`; CHECK `time >= newest.time`; `last_extrapolated_time_ = time`.
2. If `queue.size() < 3` or `time - newest < 1e-6 s`: return `{(n-1) copies of newest.cast<float>, newest, 0, newest.rotation()}`.
3. Parameters:
   - `gravity_from_local`: CeresPose(`gravity_from_local_`, no translation param, `QuaternionParameterization`).
   - For each queued pose i: init = `previous_solution_[i]` if available and not last (CHECK times match), else `gravity_from_local_ * pose_i`. Last queued node: rotation `AutoDiffLocalParameterization<ConstantYawQuaternionPlus,4,2>` and **translation constant** (`SetParameterBlockConstant`). Others: `QuaternionParameterization`. Nodes live in `std::deque<CeresPose>` (stable addresses; Rust: `Vec` pre-reserved, or boxed).
   - `gravity_constant` (double): 9.8, or `options.gravity_constant` if > 0 (then fixed).
   - Find `imu_it_prev_prev`: advance while `next->time <= newest.time`.
   - Initial guess for the new node at `time`: `ExtrapolatePoseWithImu<double>(node_last.ToRigid(), t_last, node_{last-1}.ToRigid(), t_{last-1}, g*UnitZ, time, imu, &it).pose`; add a node with `QuaternionParameterization`; `node_times.push(time)`.
4. Residuals:
   - For each queued pose i: `SpaCostFunction3D(Pose{pose_i, pose_translation_weight, pose_rotation_weight})` on `(gfl.rot, gfl.trans, node_i.rot, node_i.trans)`. SPA error: `h_t = R_i^{-1}(t_j - t_i)`, `h_r^{-1} = q_j^{-1} q_i`; `e = [z.t - h_t, AngleAxis(h_r^{-1} * z.q)]` scaled by `(wt, wt, wt, wr, wr, wr)`.
   - CHECK `imu.front().time <= queue.front().time`. `imu_calibration = [1,0,0,0]` added with QuaternionParameterization and set **constant**.
   - CHECK `imu.size()==1 || next(imu_it)->time > queue.front().time`.
   - For `i in 1..nodes.size()`: `result = IntegrateImu(imu, t_{i-1}, t_i, &imu_it)` (advances the shared iterator). If `i+1 < nodes.size()`: with a copy `imu_it2` (taken before the integration), `to_first_center = IntegrateImu(t_{i-1}, t_{i-1} + d1/2, &it2)`, `center_to_center = IntegrateImu(first_center, t_i + d2/2, &it2)`; `dv = (result.dq^{-1} * to_first_center.dq) * center_to_center.dv`; add `AccelerationCostFunction3D(imu_acceleration_weight, dv, d1, d2)` on `(node_i.rot, node_{i-1}.trans, node_i.trans, node_{i+1}.trans, &gravity_constant, imu_calibration)`: `r = w * (R_i * calib * dv - g * 0.5(d1+d2) * z - ((p_{i+1}-p_i)/d2 - (p_i-p_{i-1})/d1))`. Gravity block set constant (fixed) or lower bound 0 (re-applied each iteration, idempotent).
   - Always `RotationCostFunction3D(imu_rotation_weight, result.dq)` on `(node_{i-1}.rot, node_i.rot, calib)`: `err = q_end^* q_start calib dq calib^*`; `r = w*err.xyz`.
   - Odometry (if both `t_{i-1}`, `t_i` are strictly inside the odometry range, which needs at least 2 samples): `InterpolateOdometry` (upper_bound + `transform::Interpolate`), relative = `first^{-1} * second`, caching the last node odometry; `SpaCostFunction3D(Pose{relative, odom_t_w, odom_r_w})` on `(node_{i-1}, node_i)`.
5. `Solve(solver_options_)`. Log gravity every 20 when not fixed.
6. `gravity_estimate = nodes.back().rotation`. `extrapolated = newest.transform * nodes[n-2]^{-1} * nodes.back()` (relative motion from the last queued node, applied to the scan-matched pose). Histogram iterations. `gravity_from_local_ = gfl.ToRigid()`. `previous_solution_ = all nodes` (**including the extrapolated one**, so its size is queue+1 and the CHECK `size > i` holds).
7. `velocity = (extrapolated.t - newest.t) / (time - newest.time)`.
8. Return `{InterpolatePoses(newest, extrapolated, times[0..n-1]), extrapolated, velocity, gravity_estimate}`. `InterpolatePoses`: float lerp of translation, `Quaternionf::slerp(factor, end)`, `factor = (t - start)/(end - start)`.
- Note: `ExtrapolatePose`/`EstimateGravityOrientation` mutate state (`previous_solution_`, `gravity_from_local_`, `last_extrapolated_time_`); they are not pure.
- Note: the returned `gravity_from_tracking` is the full rotation of the new node in the gravity frame (not yaw-free).

---

## 15. Threading and ownership summary

| Component | Threading | Ownership |
|---|---|---|
| HybridGrid & co. | not thread-safe for writes; concurrent `const` reads OK | owns tree via `unique_ptr`; `update_indices_` raw ptrs into self |
| RangeDataInserter3D | const, stateless after ctor, shareable | owns two 32768-entry LUTs |
| Submap3D | written only by the local SLAM thread while unfinished; read concurrently by pose graph / constraint builder through `shared_ptr<const Submap3D>` (C++ relies on finished-before-reading semantics plus external mutexes; reads of unfinished submaps occur for the local matcher on the same thread) | `ActiveSubmaps3D` holds `shared_ptr<Submap3D>` (mutable); hands out `shared_ptr<const>`. Rust: `Arc<RwLock<Submap3D>>`, or a copy-on-finish/freeze design. `ForgetIntensityHybridGrid` mutates through the shared pointer after the submap leaves the active set. |
| LocalTrajectoryBuilder3D | single-threaded (called from the trajectory builder's sensor thread) | owns everything; Ceres `num_threads` from options (1 by default) |
| CeresScanMatcher3D / RTCSM3D | `const`, re-entrant | cost functions borrow clouds and grids for the `Solve` duration; Problem owns cost/loss/parameterization objects |
| FastCorrelativeScanMatcher3D | `const`, called concurrently from constraint builder thread pool | owns the precomputation stack; **borrows** the low-res grid and the histogram via raw pointers (the submap must outlive the matcher; constraint builder keeps the submap alive) |
| RotationalScanMatcher | const | borrows `const VectorXf*` |
| ImuBasedPoseExtrapolator | single-threaded; every extrapolate call mutates | owns deques |

## 16. External dependencies (exact usage)

- **Eigen**: `Array3i/Array3f/Array2i/Array4i` (component-wise ops, `.all()/.any()`, `cwiseMin/Max`, `cwiseAbs().maxCoeff()`, `.cast<>`, `.head<2>()`, `.matrix()`), `Vector3f/d`, `VectorXf` (histograms: `Zero`, `Unit`, `norm`, `dot`, arithmetic), `Quaterniond/f` (`FromTwoVectors`, `slerp`, `angularDistance`, `inverse`, `conjugate`, `Identity`), `AngleAxis`, `Transform<T,3,Affine>` (IMU calibration), `Map<const Matrix<T,3,1>>`.
- **Ceres**: `Problem`, `Problem::Options`, `Solve`, `Solver::Options` (`linear_solver_type = DENSE_QR`, `use_nonmonotonic_steps`, `max_num_iterations`, `num_threads`), `Solver::Summary` (`final_cost`, `iterations.size()`, `FullReport()`), `AutoDiffCostFunction<F, N|DYNAMIC, ...>` (DYNAMIC with runtime residual count), `CostFunction::Evaluate` (tests), `HuberLoss`, `QuaternionParameterization`, `AutoDiffLocalParameterization<F,4,1|2>`, `SetParameterBlockConstant`, `SetParameterLowerBound`, `AddParameterBlock`, `Jet` (`.a` scalar), `ceres::sqrt/sin/cos/atan2`, `ceres::QuaternionProduct`. (LocalParameterization was removed in Ceres 2.2 in favour of `Manifold`; the C++ targets the older API.)
- **cartographer::common**: `RoundToInt` (= `std::lround`, rounds half away from zero; Rust `f32::round() as i32` matches), `Clamp`, `Pow2`, `Power`, `atan2(Vector2)`, `QuaternionProduct`, `Time`/`Duration`/`ToSeconds`/`FromSeconds`/`FromUniversal`, `FastGzipString`, `Histogram`, `GetThreadCpuTimeSeconds`, `LuaParameterDictionary` (`GetDouble/GetInt/GetNonNegativeInt/GetBool/GetDictionary/HasKey`), `CreateCeresSolverOptions[Proto]`.
- **transform**: `Rigid3<T>` (`*`, `inverse`, `cast`, `Translation`, `Rotation`, `FromArrays`), `GetYaw` (atan2 of rotated UnitX), `GetAngle`, `AngleAxisVectorToRotationQuaternion`, `RotationQuaternionToAngleAxisVector`, `ToProto/ToRigid3`, `TimestampedTransform`, `Interpolate`.
- **sensor**: `PointCloud` (`copy_if`, `intensities()`, `points()`), `RangefinderPoint`, `RangeData`, `TransformRangeData`, `TransformPointCloud`, `TransformTimedPointCloud`, `VoxelFilter`, `AdaptiveVoxelFilter`, `ImuData`, `OdometryData`, `TimedPointCloudData`/`TimedPointCloudOriginData`, `RangeDataCollator`.
- **mapping (other modules)**: `probability_values`, `Submap` base, `MotionFilter`, `PoseExtrapolatorInterface`/`PoseExtrapolator`, `TrajectoryNode::Data`, `optimization::CeresPose`, `SpaCostFunction3D`, `AccelerationCostFunction3D`, `RotationCostFunction3D`, `ProbabilityToLogOddsInteger` (`submaps.h`).
- **absl**: `make_unique`, `optional`. **glog**: `CHECK*`, `DCHECK*` (debug-only bound checks in grid indexing), `LOG`, `LOG_IF_EVERY_N`. **metrics**: Gauge/Histogram families.

---

## 17. Tests (12 files, 35 cases)

### 17.1 `3d/hybrid_grid_test.cc`
| Test | Checks | Inputs / tolerance |
|---|---|---|
| `HybridGridTest.ApplyOdds` | 8 cells unknown initially; Set 0.5 then LUT(Odds 0.9) > 0.5; Set 0.5 then LUT(0.1) < 0.5; LUT(0.42) on unknown gives 0.42; second LUT before FinishUpdate is ignored; after FinishUpdate LUT(0.9) > 0.42 | res 1.0; NEAR 1e-4 |
| `HybridGridTest.GetProbability` | Set kMaxProbability at cell of (0,1,1); read back; neighbors (0,2,1),(1,1,1),(1,2,1) unknown | res 1.0; 1e-6 |
| `HybridGridTest.GetIntensity` | empty gives 0; add 58 gives 58; neighbors 0 | res 1.0; 1e-9 |
| `HybridGridTest.GetCellIndex` | res 2: (0,0,0)->(0,0,0); (0,26,10)->(0,13,5); (14,0,10)->(7,0,5); (14,26,0)->(7,13,0); rounding: (8.5,11.5,0.5)->(4,6,0); (7.5,12.5,1.5)->(4,6,1); (6.5,14.5,2.5)->(3,7,1); (5.5,13.5,3.5)->(3,7,2) | exact; confirms round-half-away-from-zero (4.25->4, 0.75->1, 1.25->1, 3.25->3, 1.75->2) |
| `HybridGridTest.GetCenterOfCell` | idx (3,2,1), res 2 -> (6,4,2); round trip | 1e-6 |
| `RandomHybridGridTest.TestIteration` (fixture: res 2, mt19937 seed 1285120005, 10000 random cells in [-3000,2999]^3 with uniform p in [0.1,0.9]; note x,y,z,value drawn in that order) | Iterator visits exactly the set cells; value matches GetProbability; within 1e-4 of the set value; range-for equals explicit iterator | needs grid growth to bits 7 |
| `RandomHybridGridTest.ToProto` | resolution equal; arrays equal length; map(proto) == map(range-for) | exact |
| `RandomHybridGridTest.FromProto` | `HybridGrid(ToProto())` has identical (index -> raw value) map | exact |

### 17.2 `3d/range_data_inserter_3d_test.cc` (fixture: hit 0.7, miss 0.4, free voxels 1000, intensity threshold 100; grids res 1; origin (0,0,-4); returns (-3,-1,4), (-2,0,4), (-1,1,4), (0,2,4); intensities 7,8,9,10)
| Test | Checks | Tol |
|---|---|---|
| `InsertPointCloud` | (0,0,-4),(0,0,-3),(0,0,-2) == miss prob; on z=4 plane x,y in [-4,4]: hits at y=x+2, x in [-3,0] == hit prob, else unknown | 1e-4 |
| `InsertPointCloudWithIntensities` | same, plus intensity == 10+x on hit cells, 0 elsewhere | 1e-4 / 1e-6 |
| `ProbabilityProgression` | after 1 insert: (-2,0,4)=hit, (-2,0,3)=miss, (0,0,-3)=miss; after 1000 more inserts: kMax / kMin / kMin | 1e-4 then 1e-3 |

### 17.3 `3d/submap_3d_test.cc`
| `SubmapsTest.ToFromProto` | Submap3D(0.05, 0.25, Rigid3d((1,2,0), Quaterniond(0,0,0,1))), histogram [1,2]; ToProto(true): has submap_3d not 2d; round-trip translation/rotation isApprox 1e-6, num_range_data, finished equal, resolutions 0.05/0.25 (checked on `expected`), histogram isApprox 1e-6 |

### 17.4 `internal/3d/local_trajectory_builder_3d_test.cc`
| `LocalTrajectoryBuilderTest.MoveInsideCubeUsingOnlyCeresScanMatcher` | End-to-end. Options: min 0.5, max 50, accum 1, voxel 0.2, both AVF {0.7, 200, 50}, RTCSM off, ceres {5, 20, t 0.1, r 0.3, yaw-only false, nonmonotonic, 20 iters, 1 thread}, motion filter {0.2, 0.02, 0.001}, histogram 120, constant-velocity extrapolator {gravity tc 10, queue 0.001}, submaps {0.2, 50, 0.5, 45000, inserter {0.7, 0.4, 0 free voxels, 100}}. World: 30 m cube centered at origin plus 100 bubbles (r 0.5) at radius 10 (mt19937 seed 42, uniform [-1,1]^3 normalized x10). Scans: 16 x 500 rays x 2 orthogonal rangefinders, all point times 0, origin 0, returned in sensor frame. IMU: linear acc = R^{-1}(0,0,9.81), zero gyro, one per node at node time. Trajectory: start `FromUniversal(12345678)`, 5 identity poses every 0.3 s, then t=0..0.6 step 0.05 (13 nodes) with translation (sin 4t, 1-cos 4t, t), rotation AngleAxis(0.3t, normalize(1,-1,2)), every 0.3 s. Each non-null result: `local_pose IsNearly(node.pose, 1e-1)` |

### 17.5 `scan_matching/ceres_scan_matcher_3d_test.cc` (fixture: grids res 1; 7 points (-3,2,0),(-4,2,0),(-5,2,0),(-6,2,0),(-6,3,1),(-6,4,2),(-7,3,1), intensity 50 each; map cells at expected_pose*(p) with expected = Translation(-1,0,0) set to p=1.0 (clamped to 0.9) and intensity 50; options: occ_w0 1, intensity_0 {0.5, huber 55, threshold 100}, t_w 0.01, r_w 0.1, nonmonotonic, 10 iters, 1 thread). `TestFromInitialPose(init)`: Match(target = init.translation, init, {{cloud, grid, igrid}}); EXPECT `final_cost` NEAR 0 (1e-2); pose IsNearly(expected, 3e-2).
| Test | Initial pose |
|---|---|
| `PerfectEstimate` | T(-1,0,0) |
| `AlongX` | T(-0.8,0,0) (recreates matcher) |
| `AlongZ` | T(-1,0,-0.2) |
| `AlongXYZ` | T(-0.9,-0.2,0.2) |
| `FullPoseCorrection` | cloud pre-rotated by Rz(0.05); expected = expected * Rz(0.05)^{-1}; init = ((-0.95,-0.05,0.05), Rx(0.05)) |

### 17.6 `scan_matching/interpolated_grid_test.cc` (grid res 0.1, same 7 points set to 1.0)
| Test | Checks |
|---|---|
| `InterpolatesGridPoints` | for z in [-1,3), y in [1,5), x in [-8,-2) stepping 0.1 (double accumulation): interpolated == GetProbability(GetCellIndex(x,y,z)), 1e-6 |
| `MonotonicBehaviorBetweenGridPointsInX` | where neighbor values differ (>1e-6), for samples s in [0.01, res-0.02) step res/10: `diff * (f(x+s+step) - f(x+s)) > 0` (strict monotonic) |

### 17.7 `scan_matching/intensity_cost_function_3d_test.cc`
| `IntensityCostFunction3DTest.SmokeTest` | cloud (0,0,0),(1,1,1),(2,2,2) with intensities 50,100,150; IntensityHybridGrid res 0.3 with 50 at cell(0,0,0); scaling 1, threshold 100; identity pose; residuals = [0, -100, 0] (1e-9). (Point 2 has intensity 100, not > threshold, interpolated 0, so -100; point 3 has 150 > 100, so 0.) |

### 17.8 `scan_matching/precomputation_grid_3d_test.cc`
| `PrecomputedGridGenerator3DTest.TestAgainstNaiveAlgorithm` | res 2; mt19937 seed 23847; 1000 cells in [-50,49]^3 (x,y,z, then value) with p uniform [0.1,0.9]; depths 0..3 built by `PrecomputeGrid(prev, false, (1<<(d-1))*Ones)`; 100 random queries per depth: max of GetProbability over `[x,x+w)^3`, w=2^d, NEAR ToProbability(grid.value) (1e-2) (max starts at 0 but unknown gives 0.1) |

### 17.9 `scan_matching/real_time_correlative_scan_matcher_3d_test.cc` (grid res 0.1; same 7 points at expected T(-1,0,0) set to 1; options linear 0.3, angular 1 deg, w_t 0.1, w_r 1). Each EXPECT pose IsNearly(expected, 1e-3).
| Test | Initial |
|---|---|
| `PerfectEstimate` | T(-1,0,0) |
| `AlongX` | T(-0.8,0,0) |
| `AlongZ` | T(-1,0,-0.2) |
| `AlongXYZ` | T(-0.9,-0.2,0.2) |
| `RotationAroundX` | (-1,0,0), 0.8 deg about X |
| `RotationAroundY` | 0.8 deg about Y |
| `RotationAroundYZ` | 0.8 deg about (0,1,1) (Eigen AngleAxisd with an **unnormalized** axis; Eigen's quaternion conversion uses axis*sin, so the quaternion is non-unit; reproduce, or normalize and accept a small difference) |

### 17.10 `scan_matching/rotation_delta_cost_functor_3d_test.cc` (`kPrecision = 1e-8`; sum of squared residuals)
| Test | Checks |
|---|---|
| `SameRotationGivesZeroCost` | identity vs identity is 0; R=AA(0.9, n(0.2,0.1,0.3)) vs itself is 0 |
| `ComputesCorrectCost` | s=1.2, R=AA(0.8, n(0.2,0.1,0.8)), target=AA(0.2, n(-0.5,0.3,0.4)); expected (1.2 sin 0.4)^2 for (R, I), (T*R, T), (R*T, T) |

### 17.11 `scan_matching/rotational_scan_matcher_test.cc`
| Test | Checks |
|---|---|
| `RotationalScanMatcher3DTest.OnlySameHistogramIsScoreOne` | h=[1,43,0.5,0.3123,23,42,0]; Match(h, 0, {0, 1}): [0] = 1 (1e-6), [1] < 1 |
| `RotationalScanMatcher3DTest.InterpolatesAsExpected` | 10 buckets, matcher histogram e3; for t in 0..1 step 0.1 (float accumulation): Match(e2, 0, {t*π/10}), Match(e2, 0, {(2-t)*π/10}), Match(e4, 0, {-tπ/10, (t-2)π/10}) all = t/hypot(t, 1-t) (1e-6). Pins the sign convention: a positive angle moves content to higher bucket indices (`rb = -angle*n/π`, reading `h[i+full]`). |

### 17.12 `scan_matching/fast_correlative_scan_matcher_3d_test.cc` (fixture: inserter {0.7, 0.4, 5 free voxels, 100}; options depth 6, full_res_depth 6, min_rot 0.1, min_low_res 0.15, xy 0.8, z 0.8, angular 0.3; cloud of 12 points along the axes at 4, 4.5, 5, 5.5; grid res 0.05 built by inserting the cloud at the random pose (origin = pose translation) + FinishUpdate; low-res grid = **same** grid; histogram = Zero(10) (so rotational score always 1); mt19937 seed 42, uniform [-1,1]; random pose = T(0.7*(x,y,z)) * Rz(0.2*θ) (x,y,z,θ drawn in that order); `kMinScore = 0.1`)
| Test | Checks |
|---|---|
| `CorrectPoseForMatch` | 20 iterations: Match(I, I, data(cloud), 0.1) non-null, score > 0.1, rotational > 0.09, low_res > 0.14, pose IsNearly(expected, 0.05); with low-res cloud {(42,42,42)}: null |
| `CorrectPoseForMatchFullSubmap` | 1 pose: MatchFullSubmap(I, I, ...) same checks; low-res (42,42,42) gives null |

Note: `IsNearly(a, eps)` (transform/rigid_transform_test_helpers.h) compares translation components and quaternion coefficients (sign-sensitive) within eps.

---

## 18. Rust porting notes and pitfalls

### Hybrid grid
- Implement `FlatGrid<T, const BITS>` as `Box<[T; 512]>`, `NestedGrid` as `[Option<Box<FlatGrid>>; 512]` (boxed array), `DynamicGrid` as `Vec<Option<Box<NestedGrid>>>`. Generic over `T: Copy + Default + PartialEq` (IntensityData: derive `PartialEq` or implement `is_default`).
- Keep **exact iteration order** (1.3). `ToProto` order, FCSM ties, and float summation order (`ConvertToPrecomputationGrid`, X-ray) depend on it. A `HashMap` replacement would break proto parity.
- Growth CHECK at bits 8 should be a panic (or `Result`). Indices are `i32`; the unsigned bounds trick maps to `(shifted as u32) >= size as u32`.
- `update_indices_`: avoid self-referential pointers; store indices.
- `GetCellIndex` must use round-half-away-from-zero (`f32::round`), **in f32** (`point / resolution` as f32 division, not multiplication by the inverse). The X-ray `ExtractVoxelData` uses `* resolution_inverse` (keep as is).
- `ValueToProbability(0) = 0.1` (unknown is treated as low probability everywhere in scoring). Port the precomputed tables (`kValueToProbability` double-length 65536 with marker wraparound).

### Ray casting
- Integer truncating division `delta * position / num_samples` per component: Rust `i32 /` matches C++. Overflow: `delta * position` is at most about 2^15 * 2^15 = 2^30, which fits in i32.
- Hits first, then misses, then intensities, then `finish_update`; the 3D inserter ignores `misses`.

### Ceres / autodiff
- Need: autodiff over `(t[3], q[4])` with dynamic residual count, LM solver with DENSE_QR, HuberLoss, local parameterizations (quaternion left-multiplicative with **no 1/2 factor**, yaw-only, constant-yaw), constant blocks, lower bounds (the IMU extrapolator's gravity). Options: `ceres-rs`/FFI to Ceres (closest parity), or `argmin`/`levenberg-marquardt` crate + `num-dual`/hand-rolled `Jet<const N>`. Tests with 3e-2 / 1e-1 tolerances are achievable with a faithful LM; the IMU extrapolator has no direct test.
- The interpolation **must branch on the scalar part** of the jet (`.a`), with corner values as constants. Write `trait Scalar { fn value(&self) -> f64; }` for f64 and Jet.
- Eigen `Quaternion * Vector3` with non-unit quaternions: replicate Eigen's formula (`v + 2w(u×v) + 2u×(u×v)`; Eigen `_transformVector`: `uv = 2*(u×v); return v + w*uv + u×uv`). nalgebra `UnitQuaternion::new_unchecked` uses the same formula: verify.
- `RotationDeltaCostFunctor3D` stores the conjugate (not a normalized inverse) of the target.
- `CeresPose::ToRigid` does not normalize; downstream uses `Rigid3d` with Eigen quaternion ops, so a near-unit quaternion is fine. If you wrap it in `UnitQuaternion`, normalize explicitly and accept tiny differences.
- Ceres LocalParameterization is the older API; if porting to a Manifold-style API, `Plus` is identical, and `Minus` is not needed by the LM solver.

### Scan matchers
- RTCSM: candidate order z, y, x, rz, ry, rx and strict `>`. Safety margin `1 - 1e-3`. FCSM: `1 - 1e-2`. All float32.
- FCSM: arithmetic right shifts on negative i32 (`>>` in Rust is arithmetic for i32: OK). `std::sort` with `greater` is unstable. Rust `sort_unstable_by(|a,b| b.score.partial_cmp(&a.score))` behaves equivalently but tie order may differ; tests do not depend on ties. Child loops use `break` (not `continue`) on window overflow. `std::max(best, child)` keeps `best` on ties. The early `break` uses the *original* `min_score` argument; do not "optimize" it to `best.score` (that changes pruning behavior and results).
- Precomputation `PrecomputeGrid` keeps the original resolution field even for half-res grids. `PrecomputationGrid3D` uses u8 with 0 = unknown = 0.1.
- `ScoreCandidates` sums into `i32` (max 255 * N).
- Low-res matcher closure: in Rust, `Box<dyn Fn(&Rigid3f) -> f32 + '_>` borrowing the grid and cloud.
- Rotational histogram: keep `BTreeMap<i32, Vec<Point>>` slices; centroid recomputed after filtering; `last_point` update quirk; the bucket formula `lround(n*z - 0.5)` and π wrap loops; `RotateHistogram` `rb` mixed float/double: compute as `((-angle * n as f32) as f64 / PI) as f32` to match. `MatchHistograms` returns 1 when the norm product < 1e-3.

### Local trajectory builder
- `extrapolator` is `Option<Box<dyn PoseExtrapolator>>`.
- The pipeline is single-threaded; `MatchingResult.insertion_result` is `Option<Box<..>>`.
- Read `synchronized_data.time` before moving the data.
- Two submaps: front = matching submap. The submaps must be shareable to the pose graph while still being mutated by local SLAM: choose `Arc<RwLock<Submap3D>>`, or (preferred) give local SLAM exclusive mutation and publish `Arc<Submap3D>` snapshots via copy-on-write (costly for large grids). Document the choice with the pose-graph team.
- `use_intensities` requires `intensity_cost_function_options_0` in the Ceres options (index i=0 corresponds to the hi-res cloud). Missing it panics in C++ (repeated-field out-of-range); in Rust, validate at construction.
- Metrics: use no-op default sinks.

### IMU
- `IntegrateImu` takes an in/out iterator into a deque: in Rust pass `&mut usize` (index into `VecDeque`), generic over scalar `T` for jets (only `T = f64` is used in scope; templating needed only if used inside cost functions, which it is not here).
- `ExtrapolatePoseWithImu` returns the velocity in the parent frame (input is in tracking): keep the asymmetry.
- `ImuBasedPoseExtrapolator`: every `extrapolate_*` call runs a Ceres solve and mutates state. `ExtrapolatePose` and `EstimateGravityOrientation` each trigger a solve.
- `previous_solution_` has queue+1 entries; the AddPose pop CHECK compares front times.
- `Quaterniond::FromTwoVectors(a, UnitZ)` must handle the antiparallel case like Eigen (SVD-based fallback).

### Misc
- `Submap3D` from proto: intensity grid absent; grids are optional until provided: model as `Option<Box<HybridGrid>>`, with accessor `expect`.
- The X-ray texture is visualization only (low priority) but is part of `ToResponseProto`. Width/height overflow when there are no obstructed voxels: guard in Rust (C++ produces nonsense/huge allocation; i.e. potential crash).
- glog `DCHECK`s become `debug_assert!`; `CHECK`s become `assert!`/`panic!` (or `Result` at API boundaries).
- Random-number-based tests use `std::mt19937` + `std::uniform_*_distribution`. Rust `rand` differs, so either port mt19937 and libstdc++'s distribution algorithms (uniform_real: `generate_canonical`; uniform_int: libstdc++ rejection scheme) for bit-exact fixtures, or regenerate fixtures and keep the assertions (tests are tolerance-based and should pass with any seed, except possibly the FCSM randomized poses and the LTB bubble world).
