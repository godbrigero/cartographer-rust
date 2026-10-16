# Architecture report: 2D mapping (local SLAM, grids, scan matching)

Phase 0 porting reference. Source root: `cartographer/cartographer/mapping/` (C++, read-only).
Excluded (covered elsewhere): `internal/2d/pose_graph_2d.*`, `internal/optimization/**`, `internal/constraints/**`.
All C++ snippets below are condensed. Unless noted, "float" = `f32` and "double" = `f64`.

---

## 0. Cross-cutting conventions (read this first)

### 0.1 Grid coordinate convention (the #1 porting pitfall)
- `MapLimits{resolution: f64, max: Vector2d, cell_limits: CellLimits{num_x_cells, num_y_cells}}`.
- Cell index `Array2i(ix, iy)` is **transposed and flipped** relative to world:
  - `ix = lround((max.y - p.y) / res - 0.5)`  (cell **x** index follows world **-y**)
  - `iy = lround((max.x - p.x) / res - 0.5)`  (cell **y** index follows world **-x**)
  - `GetCellCenter(ix,iy) = (max.x - res*(iy+0.5), max.y - res*(ix+0.5))` (computed in f64, returned as `Vector2f`).
- `GetCellIndex` takes a `Vector2f` but **promotes to f64** before arithmetic (`max_`, `resolution_` are double). Replicate: `((max.y - p.y as f64) / res - 0.5).round_half_away()`.
- `common::RoundToInt` = `std::lround` (round half away from zero) -> Rust `f64::round()` then `as i32` (Rust `round` is also half-away-from-zero). Do **not** use `round_ties_even`.
- Flat index: `ToFlatIndex(c) = num_x_cells * c.y + c.x` (row-major, x fastest). `ToFlatIndex` uses `CHECK(Contains)` (always on, not DCHECK).
- `Contains(c)`: `0 <= c.x < num_x && 0 <= c.y < num_y`.
- Consequence in Candidate2D: `x = -y_index_offset * res`, `y = -x_index_offset * res`.
- `GrowLimits` shifts `max` by `res * (y_offset, x_offset)` (note swap).

### 0.2 uint16 value encoding (shared by probability, correspondence cost, TSD, weight)
- `0` = unknown. `[1, 32767]` linearly maps to `[lower, upper]`. Bit 15 (`kUpdateMarker = 1<<15 = 32768`) = "updated during current insertion".
- Encode (f32 math): `v = lround((clamp(f, lower, upper) - lower) * (32766.f / (upper - lower))) + 1`.
- Decode (f32 math): `f = v * kScale + (lower - kScale)` where `kScale = (upper - lower) / 32766.f`; i.e. `lower + (v-1)*kScale`. `ValueConversionTablesTest.ValueConversion` checks with **exact float equality**, so compute literally `(v as f32) * scale + (lower - scale)` (no FMA, no reordering).
- Lookup tables are 65536 entries so indices with the marker set decode identically (`table[v | 0x8000] == table[v]`).

### 0.3 Error handling semantics
- `CHECK*` -> process abort (glog FATAL). `DCHECK*` -> debug only. Rust: `assert!` for CHECK, `debug_assert!` for DCHECK, unless the API is converted to `Result`. Several tests are `EXPECT_DEATH` (pose extrapolator), so the Rust port should `panic!` in those cases (test with `#[should_panic]`).

---

## 1. Shared primitives

### 1.1 `mapping/probability_values.{h,cc}`
| Item | Definition |
|---|---|
| `kMinProbability` | `0.1f` |
| `kMaxProbability` | `1 - 0.1f = 0.9f` |
| `kMinCorrespondenceCost` | `1 - kMaxProbability = 0.1f` (f32 arithmetic) |
| `kMaxCorrespondenceCost` | `1 - kMinProbability = 0.9f` |
| `kUnknownProbabilityValue` = `kUnknownCorrespondenceValue` | `0` |
| `kUpdateMarker` | `1u << 15` |
| `kValueCount` (cc file) | `32768` |
| `Odds(p)` | `p / (1 - p)` |
| `ProbabilityFromOdds(o)` | `o / (o + 1)` |
| `ProbabilityToCorrespondenceCost(p)` / inverse | `1 - x` |
| `ClampProbability`, `ClampCorrespondenceCost` | clamp to [0.1, 0.9] |
| `BoundedFloatToValue(f, lo, hi)` (anon ns, header) | encoding of §0.2, DCHECK 1..=32767 |
| `ProbabilityToValue(p)` | `BoundedFloatToValue(p, kMinProbability, kMaxProbability)` |
| `CorrespondenceCostToValue(c)` | `BoundedFloatToValue(c, kMinCC, kMaxCC)` |
| `kValueToProbability` | global `const vector<float>*` (65536): unknown -> `kMinProbability`, else decode over [0.1,0.9]. Built as two repetitions of 32768 values (equivalent to masking bit 15). |
| `kValueToCorrespondenceCost` | global (65536): unknown -> `kMaxCorrespondenceCost`, else decode over [0.1,0.9]. |
| `ValueToProbability(v)`, `ValueToCorrespondenceCost(v)` | table lookups (accept marker). |
| `ProbabilityValueToCorrespondenceCostValue(v)` | 0->0; strip marker if `v > kUpdateMarker` (note `>` not `>=`), `CorrespondenceCostToValue(1 - ValueToProbability(v))`, re-add marker. |
| `CorrespondenceCostValueToProbabilityValue(v)` | symmetric. |
| `ComputeLookupTableToApplyOdds(odds) -> Vec<u16>` (32768) | `[0] = ProbabilityToValue(ProbabilityFromOdds(odds)) + marker`; `[i] = ProbabilityToValue(ProbabilityFromOdds(odds * Odds(kValueToProbability[i]))) + marker`. |
| `ComputeLookupTableToApplyCorrespondenceCostOdds(odds)` (32768) | `[0] = CCToValue(1 - ProbabilityFromOdds(odds)) + marker`; `[i] = CCToValue(1 - ProbabilityFromOdds(odds * Odds(1 - kValueToCorrespondenceCost[i]))) + marker`. **This is the one used by the 2D inserter.** |
- Rust: globals -> `static VALUE_TO_PROBABILITY: LazyLock<Box<[f32; 65536]>>` (or `const fn` generation). Tables are big (256 KiB each) — heap allocate.

### 1.2 `mapping/value_conversion_tables.{h,cc}`
- `class ValueConversionTables { const vector<float>* GetConversionTable(float unknown_result, float lower, float upper); map<tuple<f,f,f>, unique_ptr<const vector<float>>> }`.
- Lazily builds a 65536-entry table: `table[v] = SlowValueToBoundedFloat(v & ~0x8000)`; `v==0 -> unknown_result`, else decode (§0.2). Returned raw pointer is stable for the lifetime of the `ValueConversionTables` object (map of unique_ptr).
- Not thread-safe (mutable map, no lock). One instance per `ActiveSubmaps2D` (and the pose graph has its own for deserialized submaps).
- Rust: `HashMap<(u32,u32,u32) /*to_bits*/, Arc<[f32]>>`; hand out `Arc<[f32]>` (avoids lifetime coupling between grids and the cache). Key on bit patterns (C++ `std::tuple<float>` ordering would treat -0.0 == 0.0; irrelevant in practice).

### 1.3 `mapping/grid_interface.h`
- `class GridInterface { virtual ~GridInterface() {} }` — empty marker base for `Grid2D`/`HybridGrid`. Inserters downcast with `static_cast`. Rust: a trait object is unnecessary; use an `enum Grid2D { Probability(ProbabilityGrid), Tsdf(Tsdf2D) }` or generics.

### 1.4 `mapping/submaps.h`
- `Logit(p) = ln(p / (1-p))`; `kMaxLogOdds = Logit(0.9)`, `kMinLogOdds = Logit(0.1)` (namespace-scope `const float`, dynamic init).
- `ProbabilityToLogOddsInteger(p) -> u8 = lround((Logit(p) - kMinLogOdds) * 254 / (kMaxLogOdds - kMinLogOdds)) + 1`, CHECK in [1,255]. Used for texture rendering.
- `class Submap` (abstract): ctor(`Rigid3d local_pose`); `ToProto(bool include_grid_data)`, `UpdateFromProto`, `ToResponseProto(global_pose, Response*)` pure virtual; `local_pose()` (const), `num_range_data()/set_`, `insertion_finished()/set_`. Defaults 0 / false.

### 1.5 `mapping/2d/xy_index.h`
- `struct CellLimits { int num_x_cells=0, num_y_cells=0; }` + ctor from proto; `ToProto(CellLimits)`.
- `XYIndexRangeIterator(min, max)` (inclusive box) / `(CellLimits)` = `(0,0)..=(nx-1, ny-1)`. Row-major: x increments fastest; on overflow x resets to `min.x`, y++. `end()` = `(min.x, max.y+1)`. `operator++` DCHECKs not at end. Rust: implement `Iterator<Item=[i32;2]>` with `for y in min.y..=max.y { for x in min.x..=max.x }`; note empty ranges (nx==0) are not handled specially in C++ (would produce garbage) — avoid.

### 1.6 `mapping/2d/map_limits.h`
- `MapLimits(res, max, cell_limits)` CHECKs `res > 0`, `nx > 0`, `ny > 0`. Proto ctor does **not** check.
- API: `resolution()`, `max()`, `cell_limits()`, `GetCellIndex(Vector2f) -> Array2i`, `GetCellCenter(Array2i) -> Vector2f`, `Contains(Array2i)`; `ToProto(MapLimits)`. See §0.1 for formulas.
- Copy-assignable (used in `GrowLimits`, `FastCorrelativeScanMatcher2D::limits_`).

### 1.7 `mapping/id.h`
- `GetTime(t)` SFINAE: uses `t.time()` if callable else `t.time` field.
- `NodeId{trajectory_id:int, node_index:int}`, `SubmapId{trajectory_id, submap_index}`: lexicographic `==`, `<`; `ToProto`; `operator<<` prints `(a, b)`. Rust: `#[derive(Copy, Eq, Ord, Hash)]` with field order (trajectory, index).
- `MapById<IdType, DataType>`: `map<int traj, MapByIndex{bool can_append_=true; map<int, Data> data_}>`.
  - `Append(traj, data)`: CHECK traj>=0, CHECK `can_append_`; index = last index+1 or 0.
  - `Insert(id, data)`: sets `can_append_=false`; CHECK not already present.
  - `Trim(id)`: must exist; if it is the highest index -> `can_append_=false`; drop trajectory when empty.
  - `find`, `Contains`, `at` (const/mut), `BeginOfTrajectory(t)` (= lower_bound on trajectory key), `EndOfTrajectory(t) = BeginOfTrajectory(t+1)`, `SizeOfTrajectoryOrZero`, `size()` (O(#traj)), `trajectory(t)` range, `trajectory_ids()` range, `begin()/end()` (end = BeginOfTrajectory(INT_MAX)), `empty()`.
  - Iterator is bidirectional; `--end()` works (test `MapByIdPrevIterator`). Deref yields `IdDataReference{id, const Data&}`.
  - `lower_bound(traj, time)`: first element of `traj` with `GetTime(data) >= time`, else `EndOfTrajectory(traj)`. Binary search over **index values** (not positions), handling gaps: `middle = left.idx + (right.idx-left.idx)/2`; `it = data.lower_bound(middle)`; if `it.idx > middle` step back one; compare time; `left = next(it)` or `right = it`.
  - Rust: `BTreeMap<i32, Trajectory{can_append, BTreeMap<i32,T>}>`; `lower_bound` via `range(..)`; iterators via `range()` chaining. A simpler `partition_point` over the BTreeMap values is acceptable as long as results equal (tests compare with std::lower_bound ground truth).

### 1.8 `mapping/trajectory_node.{h,cc}`
- `TrajectoryNodePose{ Rigid3d global_pose; optional<ConstantPoseData{Time time; Rigid3d local_pose}> }`.
- `TrajectoryNode::Data{ Time time; Quaterniond gravity_alignment; PointCloud filtered_gravity_aligned_point_cloud; PointCloud high_resolution_point_cloud; PointCloud low_resolution_point_cloud; VectorXf rotational_scan_matcher_histogram; Rigid3d local_pose; }` (2D fills only time, gravity_alignment, filtered cloud, local_pose).
- `TrajectoryNode{ shared_ptr<const Data> constant_data; Rigid3d global_pose; time() }`. Data must be shared (`Arc`) — comment: may be read for visualization while trimmed.
- `ToProto/FromProto(TrajectoryNodeData)`: point clouds go through `sensor::CompressedPointCloud` (lossy quantization; test compares after an initial compress/decompress round so values are already quantized), timestamp via `common::ToUniversal`, histogram as repeated float.

### 1.9 `mapping/trajectory_builder_interface.{h,cc}`
- `InsertionResult{ NodeId node_id; shared_ptr<const TrajectoryNode::Data> constant_data; vector<shared_ptr<const Submap>> insertion_submaps; }`.
- `LocalSlamResultCallback = fn(int trajectory_id, Time, Rigid3d local_pose, RangeData in_local, unique_ptr<const InsertionResult>)` (null InsertionResult if motion-filtered).
- `SensorId{ SensorType type; string id; }` with `SensorType {RANGE=0, IMU, ODOMETRY, FIXED_FRAME_POSE, LANDMARK, LOCAL_SLAM_RESULT}`, ordering by (type, id). `ToProto/FromProto` 1:1 enum mapping; unknown -> LOG(FATAL).
- Pure virtuals: `AddSensorData` overloads for `TimedPointCloudData`, `ImuData`, `OdometryData`, `FixedFramePoseData`, `LandmarkData`; `AddLocalSlamResultData(unique_ptr<LocalSlamResultData>)`.
- `CreateTrajectoryBuilderOptions(dict)`: `trajectory_builder_2d`, `trajectory_builder_3d`, `collate_fixed_frame`, `collate_landmarks`; optional `pure_localization_trimmer.max_submaps_to_keep`; optional `pose_graph_odometry_motion_filter.{max_time_seconds,max_distance_meters,max_angle_radians}` (only populated if key present).

### 1.10 `mapping/internal/local_slam_result_data.h`, `internal/2d/local_slam_result_2d.{h,cc}`
- `LocalSlamResultData : sensor::Data { ctor(sensor_id, Time); GetTime(); virtual AddToPoseGraph(int traj, PoseGraph*) const = 0; }`.
- `LocalSlamResult2D(sensor_id, proto::LocalSlamResultData, SubmapController<Submap2D>*)`:
  - `AddToTrajectoryBuilder(tb)`: `tb->AddLocalSlamResultData(make_unique<LocalSlamResult2D>(*this))` (copy).
  - `AddToPoseGraph`: CHECK >=1 submap and first has `submap_2d`; each submap proto -> `submap_controller->UpdateSubmap(proto)` (out of scope: `internal/submap_controller.h`), skip nulls; if none -> log and return; else `PoseGraph2D::AddNode(make_shared(FromProto(node_data)), traj, submaps)` (downcast via static_cast; DCHECK dynamic_cast).

### 1.11 `mapping/range_data_inserter_interface.{h,cc}`
- `RangeDataInserterInterface { virtual void Insert(const RangeData&, GridInterface*) const = 0; }`.
- `CreateRangeDataInserterOptions(dict)`: `range_data_inserter_type` string parsed as proto enum (`PROBABILITY_GRID_INSERTER_2D`, `TSDF_INSERTER_2D`, 3D types exist in proto); always parses **both** `probability_grid_range_data_inserter` and `tsdf_range_data_inserter` sub-dicts.

---

## 2. 2D grids

### 2.1 `mapping/2d/grid_2d.{h,cc}` — `Grid2D : GridInterface` (abstract)
State: `MapLimits limits_; vector<u16> correspondence_cost_cells_; float min_cc_, max_cc_; vector<int> update_indices_; Eigen::AlignedBox2i known_cells_box_ (inclusive, default empty); const vector<float>* value_to_correspondence_cost_table_`.

| API | Behavior |
|---|---|
| `CreateGridOptions2D(dict)` | `grid_type` string -> enum `PROBABILITY_GRID`/`TSDF` (CHECK parse), `resolution` |
| `enum class GridType { PROBABILITY_GRID, TSDF }` | |
| ctor `(limits, min_cc, max_cc, tables*)` | cells = nx*ny zeros; table = `GetConversionTable(unknown=max_cc, lower=min_cc, upper=max_cc)`; CHECK `min_cc < max_cc` |
| ctor `(proto::Grid2D, tables*)` | if proto min & max cc both `0.f` (legacy) -> use `kMinCC/kMaxCC` with LOG(WARNING); copies `known_cells_box` if present (`min_x,min_y,max_x,max_y`); cells copied with CHECK `<= 65535` |
| `FinishUpdate()` | pop all `update_indices_`, `cell -= kUpdateMarker` (DCHECK cell >= marker) |
| `GetCorrespondenceCost(c)` | out of bounds -> `max_cc_`; else `table[cells[flat]]` (marker tolerated) |
| `GetMin/MaxCorrespondenceCost()` | |
| `IsKnown(c)` | `Contains(c) && cells[flat] != 0` |
| `ComputeCroppedLimits(&offset, &limits)` | box empty -> offset 0, limits (1,1); else offset = box.min, limits = (sizes.x+1, sizes.y+1) |
| `GrowLimits(Vector2f)` (virtual) | calls protected `GrowLimits(point, {&cells}, {0})` |
| `GrowLimits(point, grids, unknown_values)` | see §2.1.1 |
| `ComputeCroppedGrid()`, `DrawToSubmapTexture()`, `GetGridType()` | pure virtual |
| `ToProto()` (virtual) | limits, cells, CHECK update_indices empty, known box if non-empty, min/max cc |
| protected accessors | `correspondence_cost_cells()`, `update_indices()`, `known_cells_box()`, `mutable_*`, `ToFlatIndex` |

#### 2.1.1 Grid growth algorithm
```
CHECK(update_indices_.empty())
while !limits.Contains(limits.GetCellIndex(point)):
    x_off = nx/2; y_off = ny/2                 // integer division
    new = MapLimits(res, max + res*(y_off, x_off), (2nx, 2ny))
    stride = 2nx; offset = x_off + stride*y_off
    for each grid g (with unknown value u): new_cells = [u; 4*nx*ny];
        for i in 0..ny, j in 0..nx: new_cells[offset + j + i*stride] = g[j + i*nx]
    limits = new; if box non-empty: box.translate(x_off, y_off)
```
- Only grows (doubles both axes) and keeps old content centered. Called for bbox min/max of each scan before inserting. TSDF overrides to grow both the TSD and weight arrays.

### 2.2 `mapping/2d/probability_grid.{h,cc}` — `ProbabilityGrid : Grid2D`
- Stores **correspondence cost** values (not probabilities). min/max cc = `kMinCC/kMaxCC` (0.1/0.9).
- ctor(limits, tables*); ctor(proto, tables*) CHECK `has_probability_grid_2d()`.
- `SetProbability(c, p)`: CHECK cell unknown (`== 0`); cell = `CorrespondenceCostToValue(1 - p)`; extend known box. No update marker.
- `ApplyLookupTable(c, table) -> bool`: DCHECK `table.size()==32768`; if cell `>= marker` return false; push flat index to update_indices; `cell = table[cell]` (result has marker); extend known box; return true.
- `GetProbability(c)`: out of bounds -> `kMinProbability`; else `1 - ValueToCorrespondenceCost(cell)` using the **global** table (not the per-grid table — same values for probability grids).
- `GetGridType() = PROBABILITY_GRID`; `ToProto()` = base + set empty `probability_grid_2d` submessage.
- `ComputeCroppedGrid()`: new grid with `max' = max - res*(offset.y, offset.x)`, limits = cropped; copy known cells via `SetProbability(xy, GetProbability(xy+offset))` (decode/re-encode round trip).
- `DrawToSubmapTexture(texture*, local_pose)`: over cropped region (row-major), per cell 2 bytes: unknown -> (0,0); else `delta = 128 - ProbabilityToLogOddsInteger(p)`; `alpha = delta>0 ? 0 : -delta`; `value = delta>0 ? delta : 0`; push `value`, then `(value||alpha) ? alpha : 1`. Gzip (`boost gzip_compressor(best_speed)`; Rust `flate2::GzEncoder` with `Compression::fast()` — bytes differ, decompressed content identical). width=nx, height=ny, resolution, `slice_pose = local_pose.inverse() * Translation(max.x - res*offset.y, max.y - res*offset.x, 0)`.

### 2.3 `mapping/2d/probability_grid_range_data_inserter_2d.{h,cc}`
- Options: `hit_probability` (CHECK > 0.5), `miss_probability` (CHECK < 0.5), `insert_free_space` (default **true** if key absent).
- ctor builds `hit_table_ = ComputeLookupTableToApplyCorrespondenceCostOdds(Odds(hit_p))`, `miss_table_` likewise.
- Constant `kSubpixelScale = 1000`.
- `Insert(range_data, grid)`: `static_cast<ProbabilityGrid*>`; `CastRays(...)`; `FinishUpdate()`.
- `GrowAsNeeded`: bbox (f32 `AlignedBox2f`) of origin.xy, all returns.xy, all misses.xy; `GrowLimits(min - 1e-6)`, `GrowLimits(max + 1e-6)`.
- `CastRays`:
  1. Grow. 2. `superscaled = MapLimits(res/1000, same max, (nx*1000, ny*1000))`. 3. `begin = superscaled.GetCellIndex(origin.xy)`.
  4. For each return: `end = superscaled.GetCellIndex(hit.xy)`; `ApplyLookupTable(end / 1000 /*integer div, both comps*/, hit_table)`.
  5. If `!insert_free_space` return.
  6. For each stored `end`: `for c in RayToPixelMask(begin, end, 1000): ApplyLookupTable(c, miss_table)`. Hit cells already carry the marker -> misses ignored there ("hits have priority").
  7. For each miss point: same with `RayToPixelMask(begin, GetCellIndex(miss.xy), 1000)` (end cell included).
  - No FinishUpdate between hits and misses; each cell updated at most once per scan.
- Note: integer division `/` on `Array2i` truncates toward zero; indices are guaranteed >= 0 after growth (RayToPixelMask CHECKs >= 0).

### 2.4 `mapping/internal/2d/ray_to_pixel_mask.{h,cc}`
`RayToPixelMask(scaled_begin, scaled_end, subpixel_scale) -> Vec<Array2i>`: all full pixels touched by the segment (supercover-like, exact integer arithmetic). Output pixel coords are unscaled. Algorithm (must be replicated exactly; tests check sequences):
```
if begin.x > end.x: return RayToPixelMask(end, begin, s)        // output always ordered by x ascending
CHECK begin.x>=0, begin.y>=0, end.y>=0
if begin.x/s == end.x/s:                                          // same pixel column
    cur = (begin.x/s, min(begin.y,end.y)/s); push cur
    for y in cur.y ..= max(begin.y,end.y)/s: push (x,y) if != last   // ascending y
    return
dx, dy = end - begin (i64); denom = 2*s*dx
cur = begin / s; push cur
sub_y = (2*(begin.y % s) + 1) * dx
first_pixel = 2*s - 2*(begin.x % s) - 1
last_pixel  = 2*(end.x % s) + 1
end_x = max(begin.x, end.x) / s
sub_y += dy * first_pixel
if dy > 0:
    loop:
        push cur if != last
        while sub_y > denom: sub_y -= denom; cur.y++; push cur if != last
        cur.x++
        if sub_y == denom: sub_y -= denom; cur.y++       // exact corner crossing: diagonal step, no extra pixel
        if cur.x == end_x: break
        sub_y += dy * 2*s
    sub_y += dy * last_pixel
    push cur if != last
    while sub_y > denom: sub_y -= denom; cur.y++; push cur if != last
    CHECK sub_y != denom; CHECK cur.y == end.y/s
else (dy <= 0):  mirror with `sub_y < 0` (+= denom, cur.y--), corner test `sub_y == 0`, CHECK sub_y != 0
```
- Use `i64` for `dx, dy, denom, sub_y` (products overflow i32 at scale 1000). `first_pixel/last_pixel` are `int`.
- Corner case: exactly diagonal lines produce `(1,1),(2,2),(3,3)` (no 4-connected filler).

### 2.5 `mapping/2d/submap_2d.{h,cc}`
- `CreateSubmapsOptions2D(dict)`: `num_range_data` (non-negative int, CHECK > 0), `grid_options_2d`, `range_data_inserter`; CHECK valid combos: (PROBABILITY_GRID, PROBABILITY_GRID_INSERTER_2D) or (TSDF, TSDF_INSERTER_2D).
- `Submap2D : Submap`:
  - ctor(origin `Vector2f`, `unique_ptr<Grid2D>`, tables*): `local_pose = Translation(origin.x, origin.y, 0)`.
  - ctor(proto::Submap2D, tables*): local_pose from proto; grid by oneof (`probability_grid_2d` / `tsdf_2d` / else FATAL) only if `has_grid`; num_range_data, finished.
  - `ToProto(include_grid)`: `submap_2d{local_pose, num_range_data, finished, grid?}` (CHECK grid if requested).
  - `UpdateFromProto(proto)`: CHECK has submap_2d; update counters; replace grid if present.
  - `ToResponseProto(_, response)`: if no grid return; `submap_version = num_range_data`; add texture; `grid->DrawToSubmapTexture(texture, local_pose)`.
  - `InsertRangeData(rd, inserter)`: CHECK grid & !finished; `inserter->Insert(rd, grid)`; `num_range_data++`.
  - `Finish()`: CHECK grid & !finished; `grid_ = grid_->ComputeCroppedGrid()`; `finished = true`.
- `ActiveSubmaps2D` (non-copyable):
  - Members: options, `vector<shared_ptr<Submap2D>> submaps_` (0..=2), `unique_ptr<RangeDataInserterInterface>`, **owned** `ValueConversionTables conversion_tables_` (grids keep raw pointers into it).
  - `InsertRangeData(rd) -> vector<shared_ptr<const Submap2D>>`:
    ```
    if submaps empty or back.num_range_data == N: AddSubmap(rd.origin.xy)
    for s in submaps: s.InsertRangeData(rd, inserter)
    if front.num_range_data == 2N: front.Finish()
    return submaps()
    ```
  - `AddSubmap(origin)`: if size >= 2: CHECK front finished, erase front. Push new Submap2D with `CreateGrid(origin)`.
  - `CreateGrid(origin)`: `kInitialSubmapSize = 100`; `MapLimits(res, origin + 0.5*100*res*(1,1), (100,100))`; Probability -> `ProbabilityGrid`; TSDF -> `TSDF2D(limits, truncation_distance, maximum_weight)` from tsdf inserter options.
  - Steady state: two submaps; front (older, N..2N inserts) is used for matching; back (newer) is being built. Each scan goes into both. Submap origin = origin of the first range data (gravity-aligned local frame).
  - `submaps()` returns const-converted copies of the shared_ptrs.

### 2.6 `mapping/internal/2d/tsd_value_converter.{h,cc}`
- `TSDValueConverter(max_tsd, max_weight, tables*)`: `min_tsd = -max_tsd`; `min_weight = 0` (static constexpr); `tsd_resolution = 32766/(max_tsd-min_tsd)`; `weight_resolution = 32766/max_weight`; `value_to_tsd = table(unknown=min_tsd, min_tsd, max_tsd)`; `value_to_weight = table(unknown=0, 0, max_weight)`.
- `TSDToValue(tsd) = lround((clamp(tsd) - min_tsd) * tsd_resolution) + 1`; `WeightToValue` likewise. `ValueToTSD/Weight` = table lookups.
- Static getters: unknown values 0, update marker `1<<15`; `getMax/MinTSD`, `getMax/MinWeight`.

### 2.7 `mapping/internal/2d/tsdf_2d.{h,cc}` — `TSDF2D : Grid2D`
- Base cells store TSD values. Base ctor called with `(min_cc=-truncation, max_cc=+truncation)` -> base table = `(unknown=+trunc, -trunc, +trunc)`: so `GetCorrespondenceCost` of unknown/out-of-bounds = **+truncation** whereas `GetTSD` of unknown = **-truncation** (converter table). Both tables live in the same `ValueConversionTables`.
- Extra: `unique_ptr<TSDValueConverter>`, `vector<u16> weight_cells_` (unknown 0; never has marker — marker lives only in TSD cell).
- ctor(proto): CHECK `has_tsdf_2d`; converter from `tsdf_2d.truncation_distance/max_weight`; weights copied.
- `CellIsUpdated(c)`: TSD cell `>= marker`.
- `SetCell(c, tsd, weight)`: if updated -> return (silently); push update index; extend box; `tsd_cell = TSDToValue(tsd) + marker`; `weight_cell = WeightToValue(weight)`.
- `GetTSD(c)`: OOB -> `min_tsd`; `GetWeight(c)`: OOB -> `0`; `GetTSDAndWeight(c)` -> pair (OOB -> (min_tsd, 0)).
- `GrowLimits` grows both arrays (unknown 0, 0).
- `ToProto`: base + `tsdf_2d{weight_cells, truncation_distance=max_tsd, max_weight}`.
- `ComputeCroppedGrid`: like probability grid using `SetCell(xy, GetTSD, GetWeight)` then `FinishUpdate()`. Returns `std::move(unique_ptr<TSDF2D>)` as `unique_ptr<Grid2D>`.
- `DrawToSubmapTexture`: `nt = sqrt(|tsd| / max_tsd)`; `nw = weight/max_weight`; `delta = (int) round(nw * (nt*255 - 128))` (f64 math); value/alpha as probability grid.

### 2.8 `mapping/internal/2d/normal_estimation_2d.{h,cc}`
- Options: `num_normal_samples` (int, CHECK > 0), `sample_radius` (double, CHECK > 0).
- `EstimateNormals(range_data, options) -> Vec<f32>` (angle in rad per return; returns must be sorted by bearing):
  ```
  for i in 0..n:
      hit = returns[i]
      b = i; while b > 0 && i-b < N/2 (integer) && |hit - returns[b-1]| < r: b--
      e = i; while e < n && e-i < ceil(N/2.0)+1 && |hit - returns[e]| < r: e++
      normals.push(EstimateNormal(returns, i, b, e, origin))
  EstimateNormal: if e-b < 2: return atan2 of (origin - p_i)          // faces sensor
      to_obs = origin - p_i; mean = 0
      for j in b..e, j != i: t = p_i - p_j; nrm = (-t.y, t.x, 0)
          if |nrm| < 1e-6 continue; if nrm·to_obs < 0: nrm = -nrm; mean += nrm.normalized()
      return atan2(mean.y, mean.x)
  ```
  (3D vectors; z ignored by construction. `atan2(0,0)=0` if all samples skipped.)

### 2.9 `mapping/internal/2d/tsdf_range_data_inserter_2d.{h,cc}`
- Options (`CreateTSDFRangeDataInserterOptions2D`): `truncation_distance`, `maximum_weight`, `update_free_space`, `normal_estimation_options`, `project_sdf_distance_to_scan_normal`, `update_weight_range_exponent` (int), `update_weight_angle_scan_normal_to_ray_kernel_bandwidth`, `update_weight_distance_cell_to_hit_kernel_bandwidth`.
- Constants: `kSubpixelScale = 1000`, `kMinRangeMeters = 1e-6f`, `kSqrtTwoPi = sqrt(2π)`.
- `GaussianKernel(x, σ) = 1/(sqrt(2π)σ) * exp(-0.5 x²/σ²)` (computed in double, returned f32).
- `ComputeRangeWeightFactor(range, exp) = |range| > 1e-6 ? 1/range^exp : 0`.
- `GrowAsNeeded`: bbox of origin.xy and, for each return, `hit + trunc * normalized(hit - origin)` (3D normalize, then xy). **Misses ignored.** Pad ±1e-6.
- `Insert(rd, grid)`:
  1. `static_cast<TSDF2D*>`; grow.
  2. If `project_sdf_distance_to_scan_normal || angle_bandwidth != 0`: copy returns, `std::sort` with `RangeDataSorter(origin)` and compute normals on the sorted copy. Sorter: `d = normalize(p.xy - origin.xy)`; points with `d.y < 0` first (ordered by `d.x` ascending), then `d.y >= 0` (ordered by `d.x` descending) == bearing ascending from -π to π. Not stable.
  3. For each (sorted) return i: `InsertHit(hit.xy, origin.xy, normal = normals.empty() ? NaN : normals[i])`.
  4. `FinishUpdate()`.
- `InsertHit`:
  ```
  ray = hit - origin; range = |ray|; if range < trunc: return     // whole hit skipped
  ratio = trunc / range
  ray_begin = update_free_space ? origin : origin + (1 - ratio)*ray
  ray_end   = origin + (1 + ratio)*ray
  mask = RayToPixelMask(superscaled(ray_begin), superscaled(ray_end), 1000)   // same superscaled MapLimits trick as §2.3
  w_angle = 1; if angle_bw != 0: w_angle = Gaussian(NormalizeAngleDifference(normal - atan2(-ray)), angle_bw)
  w_range = 1; if range_exp != 0: w_range = 1/range^exp
  for c in mask:
      if tsdf.CellIsUpdated(c): continue                 // first ray wins within one Insert()
      center = limits.GetCellCenter(c)
      tsd = range - |center - origin|
      if project_to_normal: tsd = (center - hit) · (cos normal, sin normal)
      tsd = clamp(tsd, -trunc, trunc)
      w = w_range * w_angle; if dist_bw != 0: w *= Gaussian(tsd, dist_bw)
      UpdateCell(c, tsd, w)
  UpdateCell: if w == 0 return; (t0,w0) = GetTSDAndWeight(c)
      w1 = w0 + w; t1 = (t0*w0 + tsd*w)/w1; w1 = min(w1, maximum_weight); SetCell(c, t1, w1)
  ```
  - Note: average uses the uncapped `w1`; cap only on store. Unknown cell has `(t0,w0) = (-trunc, 0)` so `t1 = tsd`.
  - `common::atan2(Vector2f)` = `atan2(v.y, v.x)`.

---

## 3. Scan matching (`mapping/internal/2d/scan_matching/`)

### 3.1 `correlative_scan_matcher_2d.{h,cc}` (shared CSM pieces)
- `DiscreteScan2D = Vec<Array2i>`.
- `SearchParameters{ int num_angular_perturbations; double angular_perturbation_step_size; double resolution; int num_scans; vec<LinearBounds{min_x,max_x,min_y,max_y}> linear_bounds /* per scan, inclusive, in cells */ }`.
  - ctor(linear_window, angular_window, point_cloud, resolution):
    ```
    max_range = max(3*res (f32), max over points |p.xy|)                       // f32
    step = (1 - 1e-3) * acos(1 - res² / (2*max_range²))                    // f64
    n_ang = ceil(angular_window / step); num_scans = 2*n_ang + 1
    n_lin = ceil(linear_window / res); every scan: [-n_lin, n_lin]²
    ```
  - Test ctor(n_lin, n_ang, step, res).
  - `ShrinkToFit(scans, cell_limits)`: CHECK sizes; per scan i: `min_b = min(0, min_k(-xy_k))`, `max_b = max(0, max_k((nx-1, ny-1) - xy_k))` (componentwise, starting from zero), then intersect with existing bounds.
- `GenerateRotatedScans(cloud, sp)`: scan k rotated about Z by `θ_k = -n_ang*step + k*step` (accumulated by repeated `+= step` in f64, then `AngleAxisf(θ as f32)`).
- `DiscretizeScans(limits, scans, Translation2f t)`: `GetCellIndex(t * p.xy)` for each point.
- `Candidate2D(scan_index, x_off, y_off, sp)`: `x = -y_off*res`, `y = -x_off*res`, `orientation = (scan_index - n_ang)*step`, `score = 0`. Ordering by `score` only (`<`, `>`).

### 3.2 `real_time_correlative_scan_matcher_2d.{h,cc}` — Olson exhaustive search
- Options (`proto::RealTimeCorrelativeScanMatcherOptions`, created in `internal/scan_matching/real_time_correlative_scan_matcher.cc`): `linear_search_window`, `angular_search_window`, `translation_delta_cost_weight` (>=0), `rotation_delta_cost_weight` (>=0).
- `Match(initial: Rigid2d, cloud, grid, &out) -> double score`:
  ```
  R0 = initial.rotation(); rotated = rotate cloud by R0.angle (f32)
  sp = SearchParameters(lin_win, ang_win, rotated, grid.res)          // no ShrinkToFit
  scans = GenerateRotatedScans(rotated, sp)
  dscans = DiscretizeScans(grid.limits, scans, Translation2f(initial.t))
  cands = all (scan, x in [min_x..=max_x], y in [min_y..=max_y])     // order: scan, x, y
  ScoreCandidates(...)
  best = max_element (first maximum wins on ties)
  out = Rigid2d(initial.t + (best.x, best.y), R0 * Rotation2D(best.orientation)); return best.score
  ```
- `ScoreCandidates(grid, dscans, sp, &cands)` (public for tests): per candidate, by grid type:
  - Probability: `mean_k GetProbability(xy_k + off)` (OOB -> 0.1); CHECK > 0.
  - TSDF: `Σ ((max_cc - |tsd|)/max_cc) * w / Σ w` with `(tsd,w) = GetTSDAndWeight(xy+off)`, `max_cc = truncation`; `Σw==0 -> 0`; CHECK >= 0.
  - Then `score *= exp(-(hypot(x,y)*tw + |orientation|*rw)²)`.

### 3.3 `fast_correlative_scan_matcher_2d.{h,cc}` — branch and bound (used by loop closure)
- Options: `linear_search_window`, `angular_search_window`, `branch_and_bound_depth` (int, CHECK >= 1 in stack ctor).
- `SlidingWindowMaximum` (monotone deque of floats): `AddValue(v)`: pop back while `v > back`; push. `RemoveValue(v)`: DCHECK `v <= front`; if `v == front` pop front (exact float equality — values are recomputed identically, so OK). `GetMaximum() = front`. `CheckIsEmpty()` CHECK size 0.
- `PrecomputationGrid2D(grid, limits, width, &reusable_intermediate)`:
  - `offset_ = (-width+1, -width+1)`; `wide_limits_ = (nx+width-1, ny+width-1)`; `min_score_ = 1 - grid.max_cc`, `max_score_ = 1 - grid.min_cc`; `cells_: Vec<u8>` of wide size.
  - Cell source value: `1 - |grid.GetCorrespondenceCost(x,y)|` (works for probability = p, unknown -> 0.1; and TSDF -> 1-|tsd|).
  - Pass 1 (rows, along x), `stride = wide_nx`, `intermediate.resize(wide_nx * ny)`:
    ```
    for y in 0..ny:
      W = new; W.add(val(0,y))
      for x in -width+1 .. 0:  inter[x+width-1 + y*stride] = W.max; if x+width < nx: W.add(val(x+width, y))
      for x in 0 .. nx-width:  inter[...] = W.max; W.remove(val(x,y)); W.add(val(x+width,y))
      for x in max(nx-width,0) .. nx: inter[...] = W.max; W.remove(val(x,y))
      W.check_empty()
    ```
  - Pass 2 (columns, along y) identical structure over `intermediate`, writing `cells_[x + (y+width-1)*stride] = ComputeCellValue(W.max)` for `x in 0..wide_nx`.
  - Result: `GetValue(xy)` = quantized max over original cells `[xy.x, xy.x+width) × [xy.y, xy.y+width)`; outside wide bounds -> 0 (single unsigned compare trick). `ComputeCellValue(p) = lround((p - min_score) * 255/(max_score - min_score))`, CHECK in [0,255]. `ToScore(v) = min_score + v*((max_score-min_score)/255)`.
- `PrecomputationGridStack2D(grid, options)`: depth D grids, widths `1<<i` for `i in 0..D`; `max_depth() = D-1`. Shares one intermediate buffer.
- `FastCorrelativeScanMatcher2D(grid, options)`: stores options, copy of `grid.limits()`, the stack. Grid not retained afterwards (only precomputed data) — immutable, safe to share across threads read-only.
- `Match(initial, cloud, min_score, &score, &pose) -> bool`: `SearchParameters(lin_win, ang_win, cloud, res)` (note: unrotated cloud for ranges) -> `MatchWithSearchParameters`.
- `MatchFullSubmap(cloud, min_score, ...)`: `SearchParameters(1e6*res, π, cloud, res)`; `center = Translation(max - 0.5*res*(ny, nx))`.
- `MatchWithSearchParameters(sp by value, initial, cloud, min_score, ...)`:
  ```
  rotate cloud by initial angle; rotated scans; discretize at initial.t
  sp.ShrinkToFit(dscans, limits.cell_limits)
  lowest = GenerateLowestResolutionCandidates(sp)  // step = 1<<max_depth, x,y from min to <=max by step; order scan,x,y
  ScoreCandidates(stack[max_depth], dscans, sp, &lowest)   // sorts descending
  best = BranchAndBound(dscans, sp, lowest, max_depth, min_score)
  if best.score > min_score: score=best.score; pose = (initial.t + (best.x,best.y), R0*Rot(best.orientation)); true
  else false
  ```
- `ScoreCandidates(pg, ...)`: `sum` (int) of `pg.GetValue(xy+off)`; `score = pg.ToScore(sum / (float)N)`; then `std::sort(desc)` (**unstable**).
- `BranchAndBound(dscans, sp, candidates, depth, min_score)`:
  ```
  if depth == 0: return candidates[0]                  // highest score (already sorted)
  best = Candidate2D(0,0,0,sp); best.score = min_score
  for c in candidates:                                 // sorted desc
      if c.score <= min_score: break                   // note: min_score is the *parameter*, not best.score
      half = 1 << (depth-1); children = []
      for xo in {0, half}: if c.x_off + xo > bounds[c.scan].max_x: break
          for yo in {0, half}: if c.y_off + yo > bounds[c.scan].max_y: break
              children.push(Candidate2D(c.scan, c.x_off+xo, c.y_off+yo))
      ScoreCandidates(stack[depth-1], ..., &children)
      best = max(best, BranchAndBound(dscans, sp, children, depth-1, best.score))   // std::max keeps first on tie
  return best
  ```
  - The recursive call passes `best.score` as the new pruning bound. Within one level the loop's break uses the *incoming* `min_score`.
  - Empty `children` at depth-1==0 would index `candidates[0]` of empty vec — cannot happen because offset 0 is always within bounds (lowest candidates start at min bounds). Rust: still guard.

### 3.4 `ceres_scan_matcher_2d.{h,cc}`
- Options (`CreateCeresScanMatcherOptions2D`): `occupied_space_weight`, `translation_weight`, `rotation_weight` (all CHECK > 0 at Match time), `ceres_solver_options{use_nonmonotonic_steps, max_num_iterations (>0), num_threads (>0)}` -> `common::CreateCeresSolverOptions` sets only those three fields on `ceres::Solver::Options`; ctor then forces `linear_solver_type = DENSE_QR`. Everything else = Ceres defaults (TRUST_REGION + LEVENBERG_MARQUARDT, function_tolerance 1e-6, gradient_tolerance 1e-10, parameter_tolerance 1e-8, initial_trust_region_radius 1e4, max 1e16, min_relative_decrease 1e-3, jacobi_scaling true, max_num_consecutive_invalid_steps 5, ...).
- `Match(target_translation: Vector2d, initial: Rigid2d, cloud, grid, &pose, &summary)`:
  ```
  x = [initial.t.x, initial.t.y, initial.rotation.angle()]    // single parameter block, size 3, no manifold
  problem.AddResidualBlock(cost_grid(w_occ / sqrt(N), cloud, grid), nullptr, x)    // PROBABILITY: OccupiedSpace, TSDF: TSDFMatch
  problem.AddResidualBlock(TranslationDelta(w_t, target_translation), nullptr, x) // 2 residuals
  problem.AddResidualBlock(RotationDelta(w_r, x[2] /*= initial angle*/), nullptr, x) // 1 residual
  ceres::Solve(options, &problem, summary); pose = Rigid2d((x0,x1), x2)
  ```
  - In local SLAM, `target_translation = pose_prediction.t` (extrapolated) while `initial` may be RT-CSM-refined: translation prior pulls toward the **prediction**, rotation prior toward the **initial (refined) angle**.
  - No loss functions. Angle not normalized after solve (Rigid2d ctor from angle).
- `TranslationDeltaCostFunctor2D` (AutoDiff, 2 residuals, 3 params): `r = w*(x0 - tx), w*(x1 - ty)`.
- `RotationDeltaCostFunctor2D` (AutoDiff, 1, 3): `r = w*(x2 - angle)` (no angle wrap).

### 3.5 `occupied_space_cost_function_2d.{h,cc}`
- `CreateOccupiedSpaceCostFunction2D(scale, cloud, grid) -> ceres::CostFunction*` = `new AutoDiffCostFunction<Functor, ceres::DYNAMIC, 3>(functor, N)`. Holds **references** to cloud and grid (caller keeps them alive for the solve).
- Functor (templated on `T` = double or `Jet<double,3>`):
  ```
  M = [R(x2) t; 0 0 1]; adapter = GridArrayAdapter(grid); interp = ceres::BiCubicInterpolator<GridArrayAdapter>(adapter)
  for i: w = M * (p_i.x, p_i.y, 1)
      interp.Evaluate((max.x - w.x)/res - 0.5 + kPadding,  (max.y - w.y)/res - 0.5 + kPadding, &r[i])  // (row, col)
      r[i] = scale * r[i]
  kPadding = INT_MAX/4 (= 536870911)
  GridArrayAdapter: DATA_DIMENSION = 1
      GetValue(row, col, double* v): if row < P || col < P || row >= NumRows()-P || col >= NumCols()-P: v = kMaxCorrespondenceCost (0.9, global const)
                                      else v = grid.GetCorrespondenceCost(Array2i(col - P, row - P))
      NumRows = ny + 2P; NumCols = nx + 2P
  ```
  - So "row" = continuous cell y index (world -x), "col" = continuous cell x index (world -y). Residual = interpolated correspondence cost (`1-p`); Ceres minimizes `0.5 Σ r²`.
  - kPadding exists only to keep `floor()` arguments positive in Ceres; a Rust port can drop it and evaluate `GetValue(r, c)` with signed indices directly (bounds check against `[0,ny)×[0,nx)` -> 0.9). Precision note: with padding, f64 coordinates near 5.4e8 lose ~1e-7 cell of precision; removing it changes results at that level only.
- **Exact `ceres::BiCubicInterpolator` semantics (Ceres 2.x `cubic_interpolation.h`)** — must be reimplemented:
  ```
  CubicHermiteSpline(p0,p1,p2,p3, x) -> (f, dfdx):        // Catmull-Rom
      a = 0.5*(-p0 + 3p1 - 3p2 + p3); b = 0.5*(2p0 - 5p1 + 4p2 - p3); c = 0.5*(-p0 + p2); d = p1
      f = d + x*(c + x*(b + x*a)); dfdx = c + x*(2b + 3a*x)
  Evaluate(r, c) -> (f, dfdr, dfdc):
      row = floor(r); col = floor(c)
      for k in -1..=2: (f_k, df_k/dc) = Spline(G(row+k, col-1), G(row+k,col), G(row+k,col+1), G(row+k,col+2), c - col)
      (f, dfdr) = Spline(f_-1, f_0, f_1, f_2, r - row)
      dfdc      = Spline(dfdc_-1, ..., dfdc_2, r - row).f
  Jet overload: (frc, dfdr, dfdc) = Evaluate(r.a, c.a); f.a = frc; f.v = dfdr*r.v + dfdc*c.v
  ```
  - `floor` returns `int` (C++ `const int row = std::floor(r)`), so Rust: `r.floor() as i32` (i64 if padding kept).
- `ceres::Grid2D` is **not** used; only the custom adapter with `GetValue(int,int,double*)` + `DATA_DIMENSION`.

### 3.6 `interpolated_tsdf_2d.h` — `InterpolatedTSDF2D` (bilinear, autodiff-aware)
- Holds `const TSDF2D&`. Template methods `GetCorrespondenceCost<T>(x, y)` and `GetWeight<T>(x, y)` for `T` = double or Jet.
- `CenterOfLowerPixel(x, y)` (on scalar part `jet.a`): `center = GetCellCenter(GetCellIndex((f32)x, (f32)y))`; if `center.x > x` -> `center.x -= res`; if `center.y > y` -> `center.y -= res`. Returns f32.
- `x1,y1 = lower center; x2 = x1+res; y2 = y1+res` (f32). `index1 = GetCellIndex((x1,y1))`. Neighbors: `q11 = idx1` (x1,y1), `q12 = idx1+(-1,0)` (x1,y2), `q21 = idx1+(0,-1)` (x2,y1), `q22 = idx1+(-1,-1)` (x2,y2) — because cell index axes are flipped/swapped (§0.1).
- `GetCorrespondenceCost`: if any of the 4 weights `== 0` -> return `T(tsdf.GetMaxCorrespondenceCost())` (= +truncation, zero gradient). Else bilinear of `tsdf.GetCorrespondenceCost` (base-table TSD values).
- `GetWeight`: bilinear of weights (no unknown check).
- `InterpolateBilinear(x,y,x1,y1,x2,y2,q11,q12,q21,q22)`: `nx = (x-x1)/(x2-x1)`, `ny = (y-y1)/(y2-y1)`, `q1 = (q12-q11)*ny + q11`, `q2 = (q22-q21)*ny + q21`, `return (q2-q1)*nx + q1` (gradient flows only through `x,y`; corner values are constants).

### 3.7 `tsdf_match_cost_function_2d.{h,cc}`
- `CreateTSDFMatchCostFunction2D(scale, cloud, tsdf)` = `AutoDiffCostFunction<Functor, DYNAMIC, 3>(.., N)`.
- Functor: `w_i = interp.GetWeight(world_i)`; `sw += w_i`; `r_i = N * scale * interp.GetCorrespondenceCost(world_i) * w_i`; if `sw == 0` **return false** (Ceres: evaluation failure / invalid step); else `r_i /= sw`.

### 3.8 Rust mapping for Ceres usage (for the `cartographer-ceres` crate)
| C++ Ceres API | Needed behavior |
|---|---|
| `ceres::Problem`, `AddResidualBlock(cost, nullptr, double* x)` | one 3-dim block, 3 cost blocks |
| `ceres::AutoDiffCostFunction<F, DYNAMIC or k, 3>` | forward-mode dual numbers (`Jet<f64,3>`); DYNAMIC residual count fixed at construction (N) |
| `ceres::Jet` (`.a`, `.v`) | scalar part used by interpolators |
| `ceres::BiCubicInterpolator<Grid>` + custom grid adapter | §3.5 |
| `ceres::Solver::Options` (`use_nonmonotonic_steps`, `max_num_iterations`, `num_threads`, `linear_solver_type = DENSE_QR`) | LM trust region with dense QR on a 3x3 normal system |
| `ceres::Solve`, `Solver::Summary.final_cost` | final_cost only used for metrics/tests (`< 1e-2`) |
| `CostFunction::Evaluate(params, residuals, jacobians)` -> bool | used directly by tests (occupied/tsdf cost tests) |

---

## 4. Motion / state estimation

### 4.1 `mapping/imu_tracker.{h,cc}` — `ImuTracker`
State: `gravity_time_constant` (const), `time_`, `last_linear_acceleration_time_ = Time::min()`, `orientation_ = Identity`, `gravity_vector_ = UnitZ`, `imu_angular_velocity_ = 0`. Copyable (copied by value in extrapolator).
- `Advance(t)`: CHECK `time_ <= t`; `dt = ToSeconds(t - time_)`; `q = AngleAxisVectorToRotationQuaternion(ω*dt)`; `orientation = (orientation*q).normalized()`; `gravity = q.conjugate() * gravity`; `time_ = t`.
- `AddImuLinearAccelerationObservation(a)`: `dt = last > Time::min() ? ToSeconds(time_ - last) : +inf`; `last = time_`; `α = 1 - exp(-dt/τ)`; `g = (1-α) g + α a`; `q = FromTwoVectors(g, orientation.conjugate()*UnitZ)`; `orientation = (orientation*q).normalized()`; CHECK `(orientation*g).z > 0` and `normalized().z > 0.99`.
- `AddImuAngularVelocityObservation(ω)`: store.
- `AngleAxisVectorToRotationQuaternion(v)` (transform.h): if `|v|² > 1e-8`: `s = sin(|v|/2)/|v|`, `w = cos(|v|/2)` else `s=0.5, w=1` (linearized, **not** normalized). Then `.normalized()` in Advance fixes it.
- `FromTwoVectors` (`internal/eigen_quaterniond_from_two_vectors.cc`) = `Eigen::Quaterniond::FromTwoVectors(a,b)`:
  ```
  v0=a.normalized(); v1=b.normalized(); c=v1·v0
  if c < -1 + 1e-12: c=max(c,-1); axis = right-singular vector of [v0ᵀ; v1ᵀ] for smallest SV (any unit vector ⟂ both)
                    w2=(1+c)/2; q = (w=sqrt(w2), vec=axis*sqrt(1-w2))
  else: axis=v0×v1; s=sqrt(2(1+c)); q = (w=s/2, vec=axis/s)
  ```
  `nalgebra::UnitQuaternion::rotation_between` returns `None` for antiparallel and uses a different formula — implement Eigen's version explicitly.
- Eigen `angularDistance(o) = 2*atan2(|d.vec|, |d.w|)`, `d = self * o.conjugate()` (used in tests).

### 4.2 `mapping/pose_extrapolator_interface.{h,cc}`, `mapping/pose_extrapolator.{h,cc}`
- Interface: `GetLastPoseTime`, `GetLastExtrapolatedTime`, `AddPose`, `AddImuData`, `AddOdometryData`, `ExtrapolatePose`, `ExtrapolatePosesWithGravity(times) -> {Vec<Rigid3f> previous_poses (N-1), Rigid3d current_pose, Vector3d current_velocity, Quaterniond gravity_from_tracking}`, `EstimateGravityOrientation(t)`. `CreateWithImuData(options, imu_data, initial_poses)` (CHECK non-empty IMU; `use_imu_based` -> `ImuBasedPoseExtrapolator` (out of scope) else `PoseExtrapolator::InitializeWithImu(..., imu_data.back())`).
- Options: `CreatePoseExtrapolatorOptions`: `use_imu_based`, `constant_velocity{pose_queue_duration, imu_gravity_time_constant}`, `imu_based{pose_queue_duration, gravity_constant, pose_translation_weight, pose_rotation_weight, imu_acceleration_weight, imu_rotation_weight, odometry_rotation_weight, odometry_translation_weight, solver_options}`.
- `PoseExtrapolator` state: `pose_queue_duration`, `deque<TimedPose{time,pose}> timed_pose_queue_`, `linear/angular_velocity_from_poses_ = 0`, `gravity_time_constant_`, `deque<ImuData> imu_data_`, `unique_ptr<ImuTracker> imu_tracker_, odometry_imu_tracker_, extrapolation_imu_tracker_`, `TimedPose cached_extrapolated_pose_ = {Time::min(), Identity}`, `deque<OdometryData> odometry_data_`, `linear/angular_velocity_from_odometry_ = 0`.
- `InitializeWithImu(dur, τ, imu)`: new; `AddImuData(imu)`; `imu_tracker_ = ImuTracker(τ, imu.time)`; add accel, add ω, `Advance(imu.time)`; `AddPose(imu.time, Rotation(imu_tracker.orientation))`.
- `GetLastPoseTime()`: back of queue or `Time::min()`. `GetLastExtrapolatedTime()`: `extrapolation_imu_tracker_->time()` or `Time::min()`.
- `AddPose(t, pose)`:
  ```
  if !imu_tracker_: start = t; if imu_data non-empty: start = min(start, imu_data.front.time); imu_tracker_ = ImuTracker(τ, start)
  queue.push_back({t,pose}); while queue.size > 2 && queue[1].time <= t - pose_queue_duration: pop_front
  UpdateVelocitiesFromPoses(); AdvanceImuTracker(t, imu_tracker_); TrimImuData(); TrimOdometryData()
  odometry_imu_tracker_ = copy(*imu_tracker_); extrapolation_imu_tracker_ = copy(*imu_tracker_)
  ```
- `UpdateVelocitiesFromPoses()`: need >= 2 poses; `Δ = newest.time - oldest.time` (s); if `Δ < pose_queue_duration`: LOG(WARNING) and keep old velocities; else `v = (t_new - t_old)/Δ`, `ω = RotationQuaternionToAngleAxisVector(q_old⁻¹ q_new)/Δ`.
- `AddImuData(imu)`: CHECK queue empty or `imu.time >= last pose time`; push; trim.
- `TrimImuData`: while `size > 1 && queue non-empty && imu[1].time <= last pose time`: pop_front (keeps one sample at/before last pose).
- `TrimOdometryData`: same with `size > 2`.
- `AddOdometryData(od)`: CHECK time >= last pose time; push; trim; if size < 2 return;
  ```
  dt = ToSeconds(oldest.time - newest.time)          // NEGATIVE
  delta = newest.pose⁻¹ * oldest.pose                 // also "backwards"; signs cancel
  ω_odo = AngleAxis(delta.rotation) / dt
  if queue empty return
  v_tracking = delta.translation / dt
  q_at_newest = last_pose.rotation * ExtrapolateRotation(newest.time, odometry_imu_tracker_)
  v_odo = q_at_newest * v_tracking
  ```
  (Port literally; do not "fix" sign conventions independently.)
- `ExtrapolatePose(t)`: `newest = queue.back()` (UB if empty — C++ test expects death; Rust: panic); CHECK `t >= newest.time`; if `cached.time != t`: `trans = ExtrapolateTranslation(t) + newest.t`; `rot = newest.q * ExtrapolateRotation(t, extrapolation_imu_tracker_)`; cache. Return cached pose.
- `EstimateGravityOrientation(t)`: copy `*imu_tracker_` (null deref -> death if no pose yet), `AdvanceImuTracker(t, &copy)`, return `copy.orientation()`.
- `AdvanceImuTracker(t, tracker)`:
  ```
  CHECK t >= tracker.time
  if imu_data empty or t < imu_data.front.time:
      tracker.Advance(t); tracker.AddAccel(UnitZ)   // fake gravity for 2D stability
      tracker.AddAngVel(odometry.size < 2 ? ω_poses : ω_odo); return
  if tracker.time < imu.front.time: tracker.Advance(imu.front.time)
  it = first imu with time >= tracker.time (lower_bound)
  while it != end && it.time < t: tracker.Advance(it.time); AddAccel(it.a); AddAngVel(it.ω); ++it
  tracker.Advance(t)
  ```
  Note: in the no-IMU branch the angular velocity set *after* advancing is used for the *next* advance.
- `ExtrapolateRotation(t, tracker)`: CHECK `t >= tracker.time`; advance; return `imu_tracker_.orientation⁻¹ * tracker.orientation`.
- `ExtrapolateTranslation(t)`: `(t - newest.time)s * (odometry.size < 2 ? v_poses : v_odo)`.
- `ExtrapolatePosesWithGravity(times)`: poses for all but last (f32), current = `ExtrapolatePose(last)`, velocity as above, gravity = `EstimateGravityOrientation(last)`.
- `RotationQuaternionToAngleAxisVector(q)`: normalize, flip sign if `w < 0`, `angle = 2 atan2(|vec|, w)`; `scale = angle < 1e-7 ? 2 : angle / sin(angle/2)`; return `scale * vec` (owned by the transform port).

### 4.3 `mapping/internal/motion_filter.{h,cc}`
- Options: `max_time_seconds`, `max_distance_meters`, `max_angle_radians`.
- State: `num_total_`, `num_different_`, `last_time_`, `last_pose_`.
- `IsSimilar(t, pose)`: every 500 calls (once total >= 500) logs reduction %. `++num_total`; if `num_total > 1 && t - last_time <= FromSeconds(max_time) && |Δtranslation| <= max_dist && GetAngle(pose⁻¹ * last_pose) <= max_angle` -> return **true** (no state update). Else update last time/pose, `++num_different`, return false.
- `transform::GetAngle(T) = 2*atan2(|q.vec|, |q.w|)` (in [0, π]).
- `FromSeconds` rounds to 100 ns ticks (`common::Duration`), compare as integer ticks.

### 4.4 `mapping/internal/range_data_collator.{h,cc}` — `RangeDataCollator`
- ctor(expected sensor ids) -> `set<string>`. State: `map<string, TimedPointCloudData> id_to_pending_data_` (sorted by id — determines `origins` order), `current_start_ = current_end_ = Time::min()`, `kDefaultIntensityValue = 0`.
- `AddRangeData(id, data) -> TimedPointCloudOriginData`:
  ```
  CHECK id expected; data.intensities.resize(data.ranges.size(), 0.0)
  if id already pending:
      current_start = current_end; current_end = pending[id].time
      result = CropAndMerge(); pending.emplace(id, data)   // emplace is a no-op if key still present
      return result
  pending.emplace(id, data)
  if pending.size != expected.size: return {}            // time = default(0), empty
  current_start = current_end; current_end = min over pending .time
  return CropAndMerge()
  ```
- `CropAndMerge()`: `result{time = current_end, origins = [], ranges = []}`. For each pending (id order):
  - `overlap_begin` = first point with `data.time + p.time >= current_start`; `overlap_end` = first after that with `data.time + p.time > current_end`. Points before `overlap_begin` are dropped (one LOG(WARNING) per call).
  - If overlap non-empty: `origin_index = origins.len()`; push `data.origin`; `corr = (float) ToSeconds(data.time - current_end)`; push each point with `point_time.time += corr`, its intensity, origin_index.
  - If `overlap_end == end`: erase entry; elif `overlap_end == begin`: keep; else keep only `[overlap_end, end)` (points + intensities).
  - Finally `std::sort` ranges by `point_time.time` (**unstable**; ties between sensors may reorder).
- Point times use `common::FromSeconds(float)` (100 ns ticks); `ToUniversal` comparisons in tests have tolerance 2 ticks.

---

## 5. `mapping/internal/2d/local_trajectory_builder_2d.{h,cc}` + options

### 5.1 Options (`local_trajectory_builder_options_2d.cc`, Lua defaults in `configuration_files/trajectory_builder_2d.lua`)
| Field | Default |
|---|---|
| `min_range`, `max_range` | 0, 30 |
| `min_z`, `max_z` | -0.8, 2 |
| `missing_data_ray_length` | 5 |
| `num_accumulated_range_data` | 1 |
| `voxel_filter_size` | 0.025 |
| `use_online_correlative_scan_matching` | false |
| `adaptive_voxel_filter` {max_length, min_num_points, max_range} | 0.5, 200, 50 |
| `loop_closure_adaptive_voxel_filter` | 0.9, 100, 50 (used by pose graph) |
| `real_time_correlative_scan_matcher` | lin 0.1, ang 20°, tw 0.1, rw 0.1 |
| `ceres_scan_matcher` | occ 1, trans 10, rot 40, solver {nonmonotonic false, iters 20, threads 1} |
| `motion_filter` | 5 s, 0.2 m, 1° |
| `imu_gravity_time_constant` (legacy) | 10 |
| `pose_extrapolator.constant_velocity` | τ 10, pose_queue_duration 0.001 |
| `submaps` | num_range_data 90, PROBABILITY_GRID res 0.05, hit 0.55, miss 0.49; TSDF trunc 0.3, max_w 10, normals {4, 0.5}, project true, range_exp 0, angle_bw 0.5, dist_bw 0.5 |
| `use_imu_data` | true |

### 5.2 Structure
- Members (construction order): `options_`, `ActiveSubmaps2D active_submaps_`, `MotionFilter`, `RealTimeCorrelativeScanMatcher2D`, `CeresScanMatcher2D`, `unique_ptr<PoseExtrapolator> extrapolator_` (lazy), `int num_accumulated_ = 0`, `RangeData accumulated_range_data_`, `optional<steady_clock::time_point> last_wall_time_`, `optional<double> last_thread_cpu_time_seconds_`, `optional<Time> last_sensor_time_`, `RangeDataCollator range_data_collator_(expected_range_sensor_ids)`.
- Results: `InsertionResult{ shared_ptr<const TrajectoryNode::Data> constant_data; vector<shared_ptr<const Submap2D>> insertion_submaps }`; `MatchingResult{ Time time; Rigid3d local_pose; RangeData range_data_in_local; unique_ptr<const InsertionResult> insertion_result /*null if motion-filtered*/ }`.

### 5.3 `AddRangeData(sensor_id, TimedPointCloudData)` -> `unique_ptr<MatchingResult>` (null = no output)
1. `sync = collator.AddRangeData(...)`; if `sync.ranges` empty -> LOG(INFO), null.
2. `time = sync.time`. If `!use_imu_data`: `InitializeExtrapolator(time)`.
3. If extrapolator null -> null (waiting for first IMU).
4. CHECK `sync.ranges.back().point_time.time <= 0`. `t_first = time + ranges.front.time`; if `t_first < extrapolator.GetLastPoseTime()` -> LOG, null. **Consequence:** without IMU the extrapolator is initialized with `AddPose(time, Identity)` on the first scan, so the first scan (whose first point is earlier than `time`) is always dropped.
5. Per point: `tp = time + p.time`; if `tp < GetLastExtrapolatedTime()` -> LOG(ERROR) once, clamp `tp`. `range_data_poses[i] = ExtrapolatePose(tp).cast<float>()` (monotone calls; cache).
6. If `num_accumulated_ == 0`: reset `accumulated_range_data_ = RangeData{{}, {}, {}}` (Eigen `{}` leaves the origin uninitialized; it is always overwritten in step 8 before use. Rust: zero it).
7. For each point i: `origin_local = pose_i * sync.origins[range.origin_index]`; `hit_local = pose_i * ToRangefinderPoint(hit)`; `delta = hit_local - origin_local`; `r = |delta|`; if `r >= min_range`: if `r <= max_range` -> returns.push(hit_local) else misses.push(`origin_local + missing_data_ray_length / r * delta`). (Intensity not propagated; `RangefinderPoint` = position only.)
8. `++num_accumulated_`; if `>= num_accumulated_range_data`:
   - sensor_duration = `time - last_sensor_time_` if set; update; reset counter.
   - `gravity_alignment = Rotation(extrapolator.EstimateGravityOrientation(time))`.
   - `accumulated.origin = range_data_poses.back().translation()` (tracking-frame origin at last point time, **not** the sensor origin).
   - `return AddAccumulatedRangeData(time, TransformToGravityAlignedFrameAndFilter(gravity_alignment.cast<float>() * range_data_poses.back().inverse(), accumulated), gravity_alignment, sensor_duration)`.
   - Else null.
- `TransformToGravityAlignedFrameAndFilter(T, rd)`: `CropRangeData(TransformRangeData(rd, T), min_z, max_z)` then `{origin, VoxelFilter(returns, voxel_filter_size), VoxelFilter(misses, voxel_filter_size)}`.

### 5.4 `AddAccumulatedRangeData(time, gravity_aligned_rd, gravity_alignment, sensor_duration)`
1. If returns empty -> LOG(WARNING) "Dropped empty horizontal range data", null.
2. `pose_prediction = Project2D(extrapolator.ExtrapolatePose(time) * gravity_alignment⁻¹)` (`Project2D` = xy + yaw via `GetYaw(q) = atan2` of rotated UnitX).
3. `filtered = AdaptiveVoxelFilter(returns, adaptive_voxel_filter_options)` (max-range filter on |p| then adaptive voxel). Empty -> null.
4. `ScanMatch(time, pose_prediction, filtered)`:
   - No submaps yet -> return prediction.
   - `matching = active_submaps.submaps().front()`.
   - `initial = prediction`; if `use_online_correlative_scan_matching`: `RT-CSM.Match(prediction, filtered, *matching.grid(), &initial)` (score -> metric).
   - `ceres.Match(prediction.translation(), initial, filtered, grid, &obs, &summary)`; metrics (final_cost, |Δt|, |Δangle|). Always returns non-null (the "failed" branch is dead code).
5. `pose_estimate = Embed3D(pose2d) * gravity_alignment`; `extrapolator.AddPose(time, pose_estimate)`.
6. `range_data_in_local = TransformRangeData(gravity_aligned_rd, Embed3D(pose2d.cast<float>()))` (voxel-filtered, not adaptive-filtered; misses included).
7. `InsertIntoSubmap(time, range_data_in_local, filtered, pose_estimate, gravity_alignment.rotation())`: if `motion_filter.IsSimilar(time, pose_estimate)` -> null; else `submaps = active_submaps.InsertRangeData(range_data_in_local)`; `Data{time, gravity_alignment, filtered, {}, {}, {}, pose_estimate}`.
8. Metrics: wall-clock latency, real-time ratio (sensor_duration / wall), CPU ratio (`common::GetThreadCpuTimeSeconds`). Return `MatchingResult{time, pose_estimate, range_data_in_local, insertion_result}`.

### 5.5 Other entry points
- `AddImuData(imu)`: CHECK `use_imu_data`; `InitializeExtrapolator(imu.time)`; `extrapolator.AddImuData(imu)`.
- `AddOdometryData(od)`: if no extrapolator -> LOG, drop; else forward.
- `InitializeExtrapolator(t)`: once; CHECK `!pose_extrapolator_options.use_imu_based`; `PoseExtrapolator(FromSeconds(constant_velocity.pose_queue_duration), constant_velocity.imu_gravity_time_constant)`; `AddPose(t, Identity)`. (Note: with IMU, first IMU sample time becomes first pose time; IMU sample is then added after the pose.)
- `RegisterMetrics(FamilyFactory*)`: gauges `mapping_2d_local_trajectory_builder_{latency,real_time_ratio,cpu_real_time_ratio}`; histograms `..._scores` (FixedWidth(0.05, 20), label scan_matcher=real_time_correlative), `..._costs` (ScaledPowersOf(2, 0.01, 100), scan_matcher=ceres), `..._residuals` (ScaledPowersOf(2, 0.01, 10), component=distance|angle). Globals default to Null metrics.

### 5.6 Sensor helpers used (owned by the sensor port; semantics that matter here)
- `VoxelFilter(points, size)`: key `(lround(x/s) as u64 << 42) + (lround(y/s) << 21) + lround(z/s)` (negative ints wrap — collisions possible, replicate); **randomized reservoir sampling** per voxel with `std::minstd_rand0` (default seed 1) and `std::uniform_int_distribution<>(1, count)`; output keeps original order of chosen points. Exact output depends on the standard library's `uniform_int_distribution` algorithm (libstdc++ vs libc++ differ) -> bit-exact parity with C++ is not achievable unless one stdlib's algorithm is reproduced; tests should not depend on which point is chosen.
- `AdaptiveVoxelFilter`: `FilterByMaxRange(|p| <= max_range)`; if size <= min_num_points return; try `max_length`; else halve `high_length` until `low=high/2` gives >= min points, then binary search until `(high-low)/low <= 0.1`.
- `CropRangeData(rd, min_z, max_z)`: keeps points with `min_z <= z <= max_z` (inclusive; returns and misses; origin untouched).

---

## 6. `mapping/internal/2d/overlapping_submaps_trimmer_2d.{h,cc}` (in `internal/2d`, pose-graph trimmer)
- `OverlappingSubmapsTrimmer2D(u16 fresh_submaps_count, double min_covered_area, u16 min_added_submaps_count) : PoseGraphTrimmer`; `IsFinished()` -> `finished_` (never set true).
- `Trim(Trimmable*)`:
  ```
  submap_data = pg.GetOptimizedSubmapData()
  if submap_data.size() - current_submap_count_ <= min_added_submaps_count_: return   // size_t arithmetic (unsigned!)
  coverage = SubmapCoverageGrid2D(first submap's grid limits)  // offset = max, resolution
  freshness = ComputeSubmapFreshness(submap_data, pg.GetTrajectoryNodes(), pg.GetConstraints())
  all_ids = AddSubmapsToSubmapCoverageGrid2D(freshness, submap_data, &coverage)
  to_remove = FindSubmapIdsToTrim(coverage, all_ids, fresh_submaps_count, min_covered_area / res²  /* implicit double->u16 */)
  current_submap_count_ = submap_data.size() - to_remove.size(); for id in to_remove: pg.TrimSubmap(id)
  ```
- `ComputeSubmapFreshness`: for INTRA_SUBMAP constraints, latest (max) NodeId per SubmapId; freshness = that node's time (skip missing submap/node).
- `AddSubmapsToSubmapCoverageGrid2D`: for each submap with freshness and `insertion_finished`: iterate cropped known cells; cell center in local frame `(max.x - res*(iy+0.5), max.y - res*(ix+0.5))`; global = `Project2D(global_pose * local_pose⁻¹ * Translation(center))`; `AddPoint`: key `(lround64((offset.x - p.x)/res), lround64((offset.y - p.y)/res))` -> push `(submap_id, time)`.
- `FindSubmapIdsToTrim`: per coverage cell keep the `fresh_submaps_count` freshest (sort by time desc, unstable) and count cells per submap (u16 counter); keep submaps with `count >= min_covered_cells`; result = `all_ids \ keep` (sorted set difference).
- Pitfall: `submap_data.size() - current_submap_count_` underflows if submaps were trimmed externally; Rust: use `saturating_sub` but document divergence or use `wrapping_sub` for parity.

---

## 7. Threading and ownership

| Object | Thread-safety | Ownership |
|---|---|---|
| `LocalTrajectoryBuilder2D` | Not thread-safe; all `Add*` calls serialized by the owning `GlobalTrajectoryBuilder`/collator dispatch (one call at a time, on the caller's thread). | Owned (unique_ptr) by `GlobalTrajectoryBuilder` for the trajectory lifetime. |
| `ActiveSubmaps2D` | Single-threaded writer. | Holds `shared_ptr<Submap2D>` (mutable) and hands out `shared_ptr<const Submap2D>` aliases to pose graph & `InsertionResult`. |
| `Submap2D` grids | Mutated in place by the local SLAM thread while the pose graph holds const aliases. Safety relies on protocol: pose graph only builds `FastCorrelativeScanMatcher2D` (in its thread pool) from submaps whose `insertion_finished()` is true; `Finish()` swaps `grid_` (cropped) *before* the submap is returned as finished. Visualization (`ToResponseProto`) on unfinished submaps from other threads is technically a data race in C++. | `unique_ptr<Grid2D>` inside Submap2D. |
| `ValueConversionTables` | Not thread-safe (lazy map). Grids keep raw pointers into it. | Owned by `ActiveSubmaps2D` (and separately by PoseGraph2D for deserialized submaps). Must outlive all grids created with it. |
| Global `kValueToProbability` / `kValueToCorrespondenceCost` | Immutable after static init. | Leaked static. |
| `FastCorrelativeScanMatcher2D` | Immutable after construction; `Match*` const and re-entrant. | Owned by constraint builder (out of scope). |
| `CeresScanMatcher2D`, `RealTimeCorrelativeScanMatcher2D` | `Match` is const; no internal mutable state. Ceres `num_threads` default 1. | Members of local builder. |
| `PoseExtrapolator`, `ImuTracker`, `MotionFilter`, `RangeDataCollator` | Not thread-safe. | Members of local builder. |
| Cost functors | Hold **references** to point cloud / grid; valid only during `Solve`. | Owned by `ceres::Problem` (takes ownership of `CostFunction*`). |
| Metrics globals | Set once in `RegisterMetrics` (before use). | Static. |

Rust recommendation:
- Active submap: `Submap2D { grid: Grid2D, ... }` owned by `ActiveSubmaps2D`, wrapped as `Arc<RwLock<Submap2D>>` (simple, correct), **or** freeze-on-finish: keep the growing grid uniquely owned and publish `Arc<Submap2D>` immutable snapshots when finished (plus cheap metadata for unfinished ones). The insertion result type must allow the pose graph to receive both active submaps each node; with `Arc<RwLock<_>>` readers take read locks only on finished submaps.
- Conversion tables: `Arc<[f32]>`, cache `ValueConversionTables` per `ActiveSubmaps2D` (or a global `Mutex<HashMap>`); no lifetimes.
- Use `rayon`-free code here; threading lives in the pose graph.

---

## 8. External dependencies — exact usage

| Dep | Usage in scope | Rust replacement |
|---|---|---|
| Eigen | `Array2i` (cell indices, componentwise `min/max`, `/` int division, `== .all()`), `Vector2f/2d/3f/3d`, `Matrix<T,2,2>/<T,3,3>/<T,3,1>` with Jet `T`, `Rotation2D<T>::toRotationMatrix`, `Rotation2Dd` composition and `.angle()`, `AngleAxisf/d`, `Quaterniond` (`FromTwoVectors`, `normalized`, `conjugate`, `inverse`, `angularDistance`, `isApprox`), `AlignedBox2i` (`extend`, `isEmpty`, `min/max`, `sizes`, `translate`), `AlignedBox2f` (bbox for growth), `Translation2f`, `Affine2f`, `VectorXf` (histogram), `lpNorm<1>` | `nalgebra` (`Vector2<i32>` for cell index or a tiny `[i32;2]` newtype; `UnitQuaternion`, `Rotation2`, `Isometry2/3`). Implement `AlignedBox2i` (min=i32::MAX, max=i32::MIN empty sentinel; `sizes = max - min`). Implement Eigen `FromTwoVectors` (§4.1). Beware nalgebra `UnitQuaternion::angle()` vs Eigen `angularDistance`. |
| Ceres | `Problem`, `AddResidualBlock`, `AutoDiffCostFunction<F, DYNAMIC, 3>` and `<F,1,3>`, `<F,2,3>`, `Jet` (`.a`, `.v`), `BiCubicInterpolator<Adapter>` with custom adapter (`DATA_DIMENSION`, `GetValue(int,int,double*)`), `Solver::Options` (`use_nonmonotonic_steps`, `max_num_iterations`, `num_threads`, `linear_solver_type=DENSE_QR`), `Solve`, `Solver::Summary` (`final_cost`, `FullReport`), `CostFunction::Evaluate` (tests) | In-house `cartographer-ceres` crate: dual numbers (const-generic `Jet<N>`), LM solver, bicubic interpolator replicating Catmull-Rom (§3.5). |
| absl | `absl::make_unique`, `absl::optional`, `absl::flat_hash_map` (voxel filter) | `Box::new`, `Option`, `HashMap` |
| glog | `CHECK*`, `DCHECK*`, `LOG(INFO/WARNING/ERROR/FATAL)`, `LOG_IF_EVERY_N` (motion filter) | `assert!`/`debug_assert!`/`panic!`, `log` crate; implement every-N counter manually |
| boost iostreams | `FastGzipString` (gzip, best_speed) for textures | `flate2` (already a dependency) |
| protobuf | options & serialization protos: `GridOptions2D`, `Grid2D` (`limits`, `cells` (repeated uint32), `known_cells_box`, `min/max_correspondence_cost`, oneof `probability_grid_2d`/`tsdf_2d{weight_cells, truncation_distance, max_weight}`), `MapLimits`, `CellLimits`, `Submap`/`Submap2D`, `SubmapQuery.Response.SubmapTexture`, `SubmapsOptions2D`, `RangeDataInserterOptions`, `ProbabilityGridRangeDataInserterOptions2D`, `TSDFRangeDataInserterOptions2D`, `NormalEstimationOptions2D`, `CeresScanMatcherOptions2D`, `RealTimeCorrelativeScanMatcherOptions`, `FastCorrelativeScanMatcherOptions2D`, `MotionFilterOptions`, `PoseExtrapolatorOptions`, `LocalTrajectoryBuilderOptions2D`, `TrajectoryNodeData`, `LocalSlamResultData`, `SensorId`, `NodeId`, `SubmapId`, `TrajectoryBuilderOptions` | `prost` (already set up via build.rs) |
| Lua (`common::LuaParameterDictionary`) | All `Create*Options` functions; tests build dicts with `common::MakeDictionary("return {...}")` | Owned by config port; Rust tests may construct prost structs directly. |
| `std::mt19937` + `uniform_*_distribution` | Tests only (random grids/poses) | `rand` with fixed seed; exact sequences will differ — tests must be value-agnostic (they are, except they rely on tolerance). |

---

## 9. Tests (every test in scope)

Tolerances quoted exactly. "Fixture" lines describe shared setup.

### 9.1 `mapping/probability_values_test.cc`
| Test | Checks |
|---|---|
| `ProbabilityValuesTest.OddsConversions` | `ProbabilityFromOdds(Odds(p)) ≈ p` for p = 0.1, 0.9, 0.5; tol 1e-6 |
| `.OddsConversionsCorrespondenceCost` | `1 - PFO(Odds(1 - c)) ≈ c` for c = kMaxCC, kMinCC, 0.5; 1e-6 |
| `.ProbabilityValueToCorrespondenceCostValueConversions` | for all i in 0..32768: both round trips (P->C->P, C->P->C) return exactly i |
| `.ProbabilityValueToCorrespondenceCostValueConversionsWithUpdateMarker` | same for i in 1..32768 with `i + kUpdateMarker`, exact |
| `.ConversionLookUpTable` | `ValueToProbability(0) ≈ 1 - ValueToCorrespondenceCost(0)` (0.1 vs 1-0.9); for i>=1 `ValueToProbability(i) ≈ ValueToCorrespondenceCost(i)` (same linear map); 1e-6 |
| `.CellUpdate` | tables for Odds(0.9): unknown cell post-update `p ≈ 1 - c` (1e-6); 5000 p samples in [0.1,0.9]: `ProbabilityToValue(p) ≈ 32768 - CCToValue(1-p)` within 1; post-update `p ≈ 1 - c` within 5e-5 |
| `.MultipleCellUpdate` | Odds(0.55), 20 repeated updates (strip marker each time) from each of 5000 p: `p ≈ 1-c` within 5e-5; initial unknown 1e-6 |
| `.EqualityLookupTableToApplyOdds` | odds 0.3: for all i in 0..32768, probability table vs correspondence table agree through value conversions within 1 (both directions) |

### 9.2 `mapping/value_conversion_tables_test.cc`
| Test | Checks |
|---|---|
| `ValueConversionTablesTest.EqualTables` | same bounds (0.1,0.1,0.5) -> same pointer |
| `.InequalTables` | different bounds -> different pointer |
| `.ValueConversion` | mt19937(42), 100 random (lower, upper, undefined) in [-10,10]: `table[0] == undefined`; `table[i] == i*scale + (lower - scale)` **exact** for i in 1..32768, `scale = (upper-lower)/32766` |

### 9.3 `mapping/submaps_test.cc`
- `SubmapsTest.LogOddsConversions`: `Expit(Logit(p)) ≈ p` for 0.1, 0.9, 0.5 (1e-6), with `Expit(x)=e^x/(1+e^x)` defined in test.

### 9.4 `mapping/2d/map_limits_test.cc`
| Test | Checks |
|---|---|
| `MapLimitsTest.ToProto` | MapLimits(42, (3,0), (2,3)) -> proto fields equal |
| `.ProtoConstructor` | proto(res 1, max (2,3), cells (4,5)) -> native equal |
| `.ConstructAndGet` | getters return ctor values |

### 9.5 `mapping/2d/xy_index_test.cc`
| Test | Checks |
|---|---|
| `XYIndexTest.CellLimitsToProto` / `.CellLimitsProtoConstructor` | field round trip (1,2) |
| `.XYIndexRangeIterator` | min (1,2), max (3,4): begin == min, end == (1,5), 9 indices all within [min,max] |

### 9.6 `mapping/2d/probability_grid_test.cc`
| Test | Checks |
|---|---|
| `ProbabilityGridTest.ProtoConstructor` | proto with limits (1,(2,3),(4,5)), 6 cells (6..11; fewer than 20!), inverted known box; constructs; limits equal; type PROBABILITY_GRID. (Rust: proto ctor must not validate cell count.) |
| `.ConstructorGridType` | type |
| `.ToProto` | limits round trip |
| `.ApplyOdds` | 2x2 grid res 1 max (1,1): all Contains, none known; SetProbability((1,0),0.5) + table Odds(0.9) + Finish -> p > 0.5; (0,1) 0.5 + Odds(0.1) -> < 0.5; unknown (1,1) + Odds(0.42) -> ≈0.42 (1e-4); second ApplyLookupTable(Odds 0.9) before Finish ignored (≈0.42); after Finish, update -> > 0.42 |
| `.GetProbability` | 2x2 max (1,2): Set kMaxProbability at cell of (-0.5,0.5) -> read ≈0.9 (1e-6); other three cells (via GetCellIndex of (-0.5,1.5),(0.5,0.5),(0.5,1.5)) contained & unknown |
| `.GetCellIndex` | res 2, max (8,14), cells (14,8): (7,13)->(0,0); (7,-13)->(13,0); (-7,13)->(0,7); (-7,-13)->(13,7); (0.5,0.5)->(6,3); (1.5,1.5)->(6,3); (0.5,-0.5)->(7,3); (-0.5,0.5)->(6,4); (-0.5,-0.5)->(7,4) |
| `.CorrectCropping` | 400x400 res 0.05 max (10,10), set random p in box (100..=299)²: cropped offset (100,100), limits 200x200 |

### 9.7 `mapping/2d/range_data_inserter_2d_test.cc` (fixture `RangeDataInserterTest2D`)
- Fixture: ProbabilityGrid res 1, max (1,5), 5x5; options `insert_free_space=true, hit=0.7, miss=0.4`. `InsertPointCloud()`: origin (-0.5, 0.5), returns (-3.5,0.5), (-2.5,1.5), (-1.5,2.5), (-0.5,3.5); Insert then FinishUpdate.
| Test | Checks |
|---|---|
| `InsertPointCloud` | limits unchanged (no growth): max (1,5) (1e-9), 5x5. Expected states indexed `expected[column][row]` for `cell_index(row, column)` = `{{U,U,U,U,U},{U,H,M,M,M},{U,U,H,M,M},{U,U,U,H,M},{U,U,U,U,H}}`; HIT ≈ 0.7, MISS ≈ 0.4 (1e-4), UNKNOWN !IsKnown |
| `ProbabilityProgression` | after one insert hit cell (-3.5,0.5) ≈ 0.7 and miss (-2.5,0.5) ≈ 0.4 (1e-4); after 1000 more inserts ≈ kMaxProbability / kMinProbability (1e-3) |

### 9.8 `mapping/2d/submap_2d_test.cc`
| Test | Checks |
|---|---|
| `Submap2DTest.TheRightNumberOfRangeDataAreInserted` | options num_range_data=10, PROBABILITY_GRID 0.05, hit 0.53 / miss 0.495 (TSDF block present but unused); 1000 inserts of empty RangeData at origin; whenever >1 submap, front has >= 10; end: 2 submaps; among all distinct submaps returned, all but one have 20 range data and exactly one has 10 |
| `Submap2DTest.ToFromProto` | Submap2D(origin (4,5), ProbabilityGrid(1, (2,3), (100,110))) -> ToProto(true): has submap_2d, not submap_3d; from proto: local pose translation/rotation isApprox 1e-6; num_range_data, finished equal; resolution (1e-6), max isApprox, num_x_cells equal |

### 9.9 `mapping/internal/2d/ray_to_pixel_mask_test.cc` (exact sequences)
| Test | Input (scale 1 unless noted) -> expected |
|---|---|
| `SingleCell` | (1,1)->(1,1): [(1,1)] |
| `AxisAlignedX` | (1,1)->(3,1) and reversed: [(1,1),(2,1),(3,1)] |
| `AxisAlignedY` | (1,1)->(1,3) and reversed: [(1,1),(1,2),(1,3)] |
| `Diagonal` | (1,1)->(3,3) both orders: [(1,1),(2,2),(3,3)]; (1,3)->(3,1) both orders: [(1,3),(2,2),(3,1)] |
| `SteepLine` | (1,1)->(2,5): [(1,1),(1,2),(1,3),(2,3),(2,4),(2,5)]; (1,1)->(2,4): [(1,1),(1,2),(2,3),(2,4)] |
| `FlatLine` | (1,1)->(5,2): [(1,1),(2,1),(3,1),(3,2),(4,2),(5,2)]; (1,1)->(4,2): [(1,1),(2,1),(3,2),(4,2)] |
| `MultiScaleAxisAlignedX` | res 0.1, max (1,1), 20x20, scales 1,2,4,...,8192: world (0.05,0.05)->(0.35,0.05) via superscaled GetCellIndex: [(9,6),(9,7),(9,8),(9,9)] |
| `MultiScaleSkewedLine` | world (0.01,0.09)->(0.21,0.19): scale 1 -> [(8,7),(8,8),(9,8),(9,9)]; scale 20 -> [(8,7),(8,8),(8,9),(9,9)] |

### 9.10 `mapping/internal/2d/tsd_value_converter_test.cc` (fixture trunc 0.1, max_weight 10)
| Test | Checks |
|---|---|
| `DefaultValues` | unknown values 0; min/max TSD ±0.1; min/max weight 0/10 (exact) |
| `ValueToTSDConversions` / `...WithUpdateMarker` | `TSDToValue(ValueToTSD(i [+marker])) == i` for i in 1..32768 |
| `ValueToWeightConversions` / `...WithUpdateMarker` | same for weights |
| `TSDToValueConversions` | 1000 samples in [-0.1,0.1): round trip within `2*0.1/32767` |
| `WeightToValueConversions` | 1000 samples in [0,10): within `10/32767` |
| `WeightToValueOutOfRangeConversions` | 20 -> 10, -10 -> 0 (tol 10/32767) |
| `TSDToValueOutOfRangeConversions` | ±0.2 -> ±0.1 (tol 0.2/32767) |

### 9.11 `mapping/internal/2d/tsdf_2d_test.cc`
| Test | Checks |
|---|---|
| `TSDF2DTest.ProtoConstructor` | proto with tsdf_2d (max_weight 10, trunc 1), 6 cells & 6 weights, min/max cc ±1 -> type TSDF, limits equal |
| `.ConstructorGridType`, `.ToProto` | type, limits round trip |
| `.GetCellIndex` | identical cases to §9.6 |
| `.WriteRead` | 2x2 res 1 max (1,2), trunc 1, max_w 10: one random (tsd, w) set at cell of (-0.5,0.5) without FinishUpdate; GetTSDAndWeight/GetTSD within `2/32768`, weight within `10/32768`; other 3 cells unknown |
| `.CorrectCropping` | as probability grid version |

### 9.12 `mapping/internal/2d/tsdf_range_data_inserter_2d_test.cc`
- Fixture: TSDF2D(res 1, max (1,7), cells (8,1)) = 8 cells along cell-x (world y in [-1,7]) by 1 cell along cell-y (world x in [0,1]); trunc 2, max_w 10. The insert grows it. Options: trunc 2, max_w 10, update_free_space false, normals {2, 10}, project false, range_exp 0, both bandwidths 0. `InsertPoint()`: origin (-0.5,-0.5), one return (-0.5,3.5); Insert; FinishUpdate. Grid grows on insert. Matcher `EqualCellProperties(known, tsd, w)`: tsd tol 1e-4, weight tol 1e-2.
| Test | Checks |
|---|---|
| `InsertPoint` | for y = 1.5..5.5: cell (-0.5,y) known, tsd = clamp(3.5-y, ±2), w=1; cells at x=0.5,1.5 unknown (tsd -2, w 0); (0.5,6.5) and (-0.5,-1.5) unknown. After 1000 more inserts: same tsd, weight = 10 (capped) |
| `InsertPointWithFreeSpaceUpdate` | update_free_space=true: y from -0.5..5.5 known with same formula; (-0.5,6.5) and (-0.5,-1.5) unknown; after 1000: weight 10 |
| `InsertPointLinearWeight` | range_exp=1: w = 1/4 (range 4) |
| `InsertPointQuadraticWeight` | range_exp=2: w = 1/16 |
| `InsertSmallAnglePointWithoutNormalProjection` | returns (-0.5,3.5),(5.5,3.5),(10.5,3.5); cell at (4.5,2.5): tsd = |(-0.5,-0.5)-(5.5,3.5)| - |(4.5,2.5)-(-0.5,-0.5)|, w 1 |
| `InsertSmallAnglePointWitNormalProjection` | project=true, returns (-0.5,3.5),(5.5,3.5): (4.5,2.5) tsd 1, w 1; (6.5,4.5) tsd -1, w 1 |
| `InsertPointsWithAngleScanNormalToRayWeight` | angle bw 10: weights at (-0.5,3.5) and (6.5,4.5) ≈ `1/(sqrt(2π)*10)` and ≈ `that*exp(+atan(7/5)²/(2*100))` (test uses **positive** exponent; passes only due to 1e-3 tolerance — keep implementation's negative exponent) |
| `InsertPointsWithDistanceCellToHit` | dist bw 10: along ray tsd as InsertPoint, w = `1/(sqrt(2π)10) * exp(+tsd²/200)` (same sign quirk, tol 1e-2) |

### 9.13 `mapping/internal/2d/normal_estimation_2d_test.cc`
| Test | Checks |
|---|---|
| `NormalEstimation2DTest.SinglePoint` | single return on unit circle at angle a (100 angles in [-π,π)): normal ≈ a+π within `2π/100 + 1e-4` (via NormalizeAngleDifference) |
| `.StraightLineGeometry` | {(-1,1),(0,1),(1,1)} -> all -π/2; {(1,1),(1,0),(1,-1)} -> |n| = π; {(1,-1),(0,-1),(-1,-1)} -> π/2; {(-1,-1),(-1,0),(-1,1)} -> 0; tol 1e-4; options {2, 10} |
| `CircularGeometry2DTest.NumSamplesPerNormal` (param N = 1,2,4,5,8) | 100 points on the unit circle, point i at bearing `i*2π/100 - π`, origin 0, radius 10: normal (pointing inward, = bearing + π) checked as `NormalizeAngleDifference(normal[i] - i*2π/100) ≈ 0` within `2π/100 * N/2 + 1e-4` |

### 9.14 `mapping/internal/2d/overlapping_submaps_trimmer_2d_test.cc` (uses `testing::FakeTrimmable` from `internal/testing/`)
- Helper `AddSquareSubmap(global_from_submap, local_from_submap, corner max, index, num_cells, finished)`: proto grid res 1, all cells value 1, known box full, `num_range_data = 1`.
| Test | Setup -> expected trimmed |
|---|---|
| `EmptyPoseGraph` | (1,0,0) -> none |
| `TrimOneOfTwoOverlappingSubmaps` | two identical 1-cell submaps, nodes t=1000/2000 intra -> trims {0,0} |
| `TestMinAddedSubmapsCountParam` | min_added=2: first Trim none; add third submap/node/constraint; Trim -> {0,0},{0,1} |
| `DoNotTrimUnfinishedSubmap` | submap 0 unfinished -> none |
| `UseOnlyIntraSubmapsToComputeFreshness` | submap 0: INTER to node0 + INTRA to node2 (t 3000); submap 1 INTRA node1 (2000) -> trims {0,1} |
| `TrimSubmapWithLowCoveredCellsCount` | 2x2 submaps, second shifted by (1,1); min_covered_area 4 -> trims {0,0} |
| `TestTransformations` | second submap global = local = ((1,1), π/2) -> overlaps exactly -> trims {0,0} |

### 9.15 `mapping/internal/2d/scan_matching/` tests
| Test | Setup / checks |
|---|---|
| `CeresScanMatcherTest.testPerfectEstimate` / `testOptimizeAlongX` / `AlongY` / `AlongXY` | ProbabilityGrid res 1, max (10,10), 20x20; one cell at (-3.5,2.5) set to 0.9; cloud {(-3,2)}; options occ 1, trans 0.1, rot 1.5, solver {nonmonotonic true, 50 iters, 1 thread}. Initial (and target) translation (-0.5,0.5) / (-0.3,0.5) / (-0.45,0.3) / (-0.3,0.3), angle 0. Expect `final_cost ≈ 0` (1e-2) and pose ≈ Translation(-0.5,0.5) (`IsNearly` 1e-2). |
| `SearchParameters.Construction` | test ctor (4, 5, 0.03, 0.05): n_ang 5, step 0.03, res 0.05, 11 scans, all bounds ±4 |
| `Candidate.Construction` | Candidate(3, 4, -5): x 0.25, y -0.2, orientation -0.06, score 0 (1e-9); ordering by score |
| `GenerateRotatedScans.GenerateRotatedScans` | point (-1,1), sp(0,1,π/2,0): 3 scans -> (1,1), (-1,1), (-1,-1) (1e-6) |
| `DiscretizeScans.DiscretizeScans` | 7 points (0.025,0.175),(−0.025,0.175),(−0.075,0.175),(−0.125,0.175),(−0.125,0.125),(−0.125,0.075),(−0.125,0.025); limits (0.05, (0.05,0.25), 6x6): indices (1,0),(1,1),(1,2),(1,3),(2,3),(3,3),(4,3) |
| `PrecomputationGridTest.CorrectValues` | 250x250 res 0.05 max (5,5); random exact-uint8 scores in cells (50..=249)²; widths 1,2,3,8: `ToScore(GetValue(xy)) ≈ max_{dx,dy<width} GetProbability(xy+(dx,dy))` (1e-4) for all xy |
| `PrecomputationGridTest.TinyProbabilityGrid` | 4x4 grid fully random; widths 1,2,3,8,200 (width > grid) same check |
| `FastCorrelativeScanMatcherTest.CorrectPose` | depth 3, lin 3, ang 1; inserter hit 0.7/miss 0.4; 6-point cloud; 50 random poses (t ∈ [-2,2]², θ ∈ [-0.5,0.5]); fresh 200x200 res 0.05 grid each; `Match(Identity, cloud, 0.1)` true, score > 0.1, pose ≈ expected (IsNearly 0.03) |
| `FastCorrelativeScanMatcherTest.FullSubmapMatching` | depth 6; 20 iterations; cloud perturbed by random (±10 m, ±1.6 rad); `MatchFullSubmap` true, score > 0.1, pose within 0.03 |
| `InterpolatedTSDF2DTest.InterpolatesGridPoints` | TSDF res 1, max (5.5,5.5), 10x10, trunc 1, max_w 10; 4 inner + ring cells set (0.1, 1); at inner grid points interpolated == raw cc/weight (1e-4); at (3,2) (border neighbor unknown) cost == max cc (1.0) and weight == raw |
| `InterpolatedTSDF2DTest.InterpolatesWithinCell` | 4 cells at (0,0),(0,1),(1,0),(1,1) with tsd 0.1/0.2/0.3/0.4, w 1/2/3/4; loops start at res/100 = 0.01 and step by res (1.0) while < 1, so only (0.01, 0.01) is sampled: tsd and weight equal the bilinear formula (1e-3) |
| `OccupiedSpaceCostFunction2DTest.SmokeTest` | empty 2x2 grid, point (0,0), pose 0: residual == 0.9 (`DoubleEq(kMaxProbability)`, i.e. kMaxCorrespondenceCost); jacobian computed but unchecked |
| `RealTimeCorrelativeScanMatcherTest.ScorePerfectHighResolutionCandidateProbabilityGrid` | grid res 0.05 max (0.05,0.25) 6x6, insert 7-point cloud (same as DiscretizeScans) from origin 0, hit 0.7; candidate (0,0,0) score ≈ 0.7 (1e-2) |
| `...ScorePerfectHighResolutionCandidateTSDF` | TSDF res 0.05 max (0.3,0.5) 20x20 trunc 0.3 max_w 1.0; inserter (trunc 0.3, max_w 10, normals {4,0.5}, project true, bandwidths 0.5/0.5), origin (0.5,-0.5); score ≈ 1.0 (0.1) and > 0.95 |
| `...ScorePartiallyCorrectHighResolutionCandidateProbabilityGrid` | candidate (0,0,1): `0.7*3/7 < score < 0.7` |
| `...ScorePartiallyCorrectHighResolutionCandidateTSDF` | candidate (0,0,1): `1 - 4/42 < score < 1` |
| (RT options for all: lin 0.6, ang 0.16, both delta weights 0) | |
| `TSDFSpaceCostFunction2DTest.MatchEmptyTSDF` | TSDF res 0.1 max (2.05,2.05) 40x40 trunc 0.3 max_w 1; point (0,0): `Evaluate` returns false |
| `.ExactInitialPose` | insert returns x ∈ {-0.5,-0.4,..} at y=1 from origin (-0.5,-0.5), project true; point (0,1), pose 0: residual ≈ 0 (1e-3), jacobian ≈ (0,-1,0) (1e-3) |
| `.PertubatedInitialPose` | pose y=+0.1 -> residual ≈ -0.1, y=-0.1 -> +0.1; jacobian (0,-1,0) |
| `.InvalidInitialPose` | pose y=±0.4 -> Evaluate false (weights zero outside truncation band) |

### 9.16 Other shared tests
| File / Test | Checks |
|---|---|
| `pose_extrapolator_test` consts | queue 0.5 s, τ 0.1, extrapolate 0.1 s, precision 1e-8, extrapolation precision 1e-2 |
| `PoseExtrapolatorDeathTest.IncompleteInitialization` | no pose: `EstimateGravityOrientation` and `ExtrapolatePose` die; still die after AddImuData |
| `PoseExtrapolatorDeathTest.ExtrapolateInPast` | pose at t: extrapolate at t == pose (1e-8); t-10 s dies (both calls); after extrapolating to t+20 s, extrapolating at t dies (tracker can't go back) |
| `PoseExtrapolatorTest.EstimateGravityOrientationWithIMU` | InitializeWithImu accel (1.6,2,8): gravity orientation == FromTwoVectors(a, Z) (angularDistance 1e-8); 10 more samples every τ: still equal |
| `PoseExtrapolatorTest.ExtrapolateWithPoses` | poses each 1 s composed with motion (v=(0,0.1,0), ω=(0.01,0,0.1)): ExtrapolatePose(t_i) == pose (1e-8); at +0.1 s ≈ pose*motion(0.1) (1e-2); GetLastExtrapolatedTime == t+0.1; gravity axis ≈ Z (1e-2) |
| `PoseExtrapolatorTest.ExtrapolateWithIMU` | init with g=(0,0,9.8); 2 s of rest poses+IMU; then 3x10 IMU samples of yaw 0.1 rad/s every 0.1 s: ExtrapolatePose ≈ expected (1e-2), then AddPose; gravity axis ≈ Z (1e-8) |
| `ImuTrackerTest.IntegrateYawRotation` | τ=0.3, 10 steps over 3 s with accel (0,0,9.9), ω (0,0,0.3): orientation == AngleAxis(0.9, Z) (1e-8) |
| `ImuTrackerTest.IntegrateFullRotation` | τ=3e10, ω (0.1,0.4,0.1): orientation == AngleAxis(3|ω|, ω̂) (1e-8) |
| `ImuTrackerTest.LearnGravityVector` | accel (0.5,0.3,9.5), ω 0: orientation == FromTwoVectors(a, Z) (1e-8) |
| `MotionFilterTest.NotInitialized` | first call false |
| `.NoChange` | same time/pose second call true |
| `.TimeElapsed` | +1 s (> 0.5) false, repeat true |
| `.LinearMotion` | 0.3 m false, 0.45 (0.15 from last) true, 0.6 false (0.3 from last kept 0.3), (0.6,0.15) true |
| `.RotationalMotion` | about Y: 1.9 true, 2.1 false, 4.0 true, 5.9 false, identity true (GetAngle wraps via |w|) |
| `RangeDataCollatorTest.SingleSensor` | 10-point fake data (times spread over [from,to], end time `to`, origin (0,1,2), z = fraction): outputs at 300 (10 pts, 1 origin, sorted), 500 (10 pts, first point at ≈300 ±2 ticks), data (-1000..510) -> only 1 point (time 0) at 510 |
| `.SingleSensorEmptyData` | empty first message returns time 300 & 0 ranges; then as above |
| `.TwoSensors` | s0 (200..300) -> empty; s1 (-1000..310) -> 2 origins, time 300, 19 ranges, first ≈ -1000, last time 0, sorted; s0 (300..500) -> time 310, 2 ranges...; s0 (600..700) -> time 500; total 30 points across outputs |
| `.ThreeSensors` / `.ThreeSensorsWithIntensities` | s0 100..200, s1 199..250, s2 210..300 -> 11 points; s2 again 400..500 -> totals 30; intensities follow points (z == intensity, 1e-6) |
| `TrajectoryNodeTest.ToAndFromProto` | full Data round trip: time equal, gravity isApprox, point clouds equal (pre-quantized), histogram equal, local_pose (1e-9) |
| `IdTest.*` (13 tests) | `EmptyMapById`, `DeleteTrajectory`, `MapByIdIterator` (order (0,0),(0,1),(7,0),(42,0)), `MapByIdTrajectoryRange`, `MapByIdTrajectoryIdRange` (0,7,42), `MapByIdIterateByTrajectories`, `MapByIdPrevIterator`, `InsertIntoMapById` (Append then Insert(42,5) -> size 2), `FindNodeId`, `FindSubmapId`, `LowerBoundEdgeCases`, `LowerBound` (times 1,2,4,5: t3->idx2, t2->idx1, t4->idx2), `LowerBoundFuzz`, `LowerBoundTrimmedTrajectory` (gap after trimming, t=0), `LowerBoundFuzzWithStruct` (field `time` instead of method) |

No tests exist for `LocalTrajectoryBuilder2D` (TODO in header), `ProbabilityGridRangeDataInserter2D` growth, `DrawToSubmapTexture`, or `LocalSlamResult2D`. Recommend adding Rust integration tests (synthetic scans of a box room) for the full local SLAM loop.

---

## 10. Rust porting notes and pitfalls

1. **Axis convention** (§0.1): wrap in `MapLimits::cell_index(p) -> CellIndex{x,y}` and never compute indices ad hoc. Write the `GetCellIndex` unit test table (§9.6) first.
2. **Float precision fidelity**: keep f32 where C++ uses float (cell values, tables, point clouds, TSDF math, normals) and f64 where C++ uses double (MapLimits arithmetic, poses, Ceres). Mixed promotions (`Vector2f` point vs `double` max) must be reproduced to get identical cell indices at cell boundaries.
3. **Rounding**: `lround` = round-half-away-from-zero; `std::floor` -> `int` in the interpolator; integer `/` on indices truncates toward zero (`i32 / i32` in Rust also truncates — OK).
4. **Update marker discipline**: every insertion = (grow) -> apply (marks) -> `finish_update()` (clears). `GrowLimits` asserts no pending updates. TSDF `set_cell` silently ignores already-updated cells. Model the "pending update" state explicitly (e.g. `update_indices: Vec<usize>`).
5. **Table identity**: `GetCorrespondenceCost` uses the per-grid table (unknown -> max cc), `ProbabilityGrid::GetProbability` uses the global table (unknown -> 0.1 prob). TSDF: base table unknown -> +trunc; converter table unknown -> -trunc. Do not unify.
6. **Unstable sorts**: `std::sort` in Fast CSM scoring, range-data collator, TSDF bearing sort, trimmer. Rust `sort_by` is stable; results may differ on ties. For B&B ties, C++ ordering is implementation-defined anyway — use `sort_unstable_by` with `partial_cmp().unwrap()` (scores are never NaN) and accept tie differences; document in tests.
7. **Tie-breaking**: RT-CSM `max_element` = first maximum (Rust `max_by` returns the **last** maximum — use a manual loop or `reduce` keeping the first). B&B `std::max(best, x)` keeps `best` on ties.
8. **Branch-and-bound pruning bound**: loop break uses the incoming `min_score`, recursion receives updated `best.score` (§3.3). Getting this wrong changes performance, not correctness.
9. **SlidingWindowMaximum** uses exact float equality on removal; values must be recomputed by the same function (`1 - |cc|`), so cache them in a row buffer to be safe.
10. **Ceres replication**: bicubic Catmull-Rom with Jet chain rule (§3.5); `AutoDiffCostFunction` with dynamic residual count; TSDF cost returns `false` -> solver must treat as failed evaluation (LM rejects step / initial evaluation failure aborts). LM default parameters must match Ceres defaults for parity in tests (`final_cost < 1e-2`, pose within 1e-2).
11. **Ceres scan matcher priors**: translation prior = prediction, rotation prior = initial (possibly RT-refined) angle. Weights: occupied weight divided by `sqrt(N)`.
12. **Angles**: no normalization in Ceres result or rotation residual; `Rigid2d` stores `Rotation2D` (angle kept as given; `.angle()` returns stored value). Make sure the Rust `Rigid2` type does not silently normalize if parity matters (it affects `RotationDeltaCostFunctor2D` target).
13. **Extrapolator death semantics**: methods panic if no pose was added; `ExtrapolatePose` cannot go backwards in time past the last extrapolation (tracker monotonic). The local builder clamps per-point times to `GetLastExtrapolatedTime()` to avoid this.
14. **First scan dropped** in no-IMU mode (§5.3 step 4) — not a bug in the port.
15. **Odometry velocity sign trick** (§4.2): port literally.
16. **Voxel filter randomness** (§5.6): no bit-exact parity; design tests to be robust.
17. **Overflow**: `RayToPixelMask` intermediates (`dx, dy, denominator, sub_y`) must be i64 (`dy * 2*s * ...` exceeds i32 at s=1000). Superscaled indices themselves are i32 (`num_cells * 1000` overflows only beyond ~2.1M cells per axis); `CellLimits(nx*1000, ny*1000)` is also computed in i32. Keep i32 for indices, i64 for the accumulator.
18. **u16 narrowing** in the trimmer (`min_covered_area / res²` -> u16, cell counts in u16) — replicate with explicit `as u16` (saturating in Rust vs truncating/UB in C++ for out-of-range; document).
19. **Growth padding**: probability inserter pads bbox by 1e-6 and includes misses; TSDF inserter extends returns by truncation distance and ignores misses.
20. **Hit priority**: probability inserter applies all hits before any misses, without FinishUpdate in between. Ray endpoints for returns are the hit cells (already marked). Miss-ray endpoints get miss updates.
21. **ComputeCroppedGrid** re-encodes through float (lossy but stable); TSDF cropped grid needs `finish_update()`; probability grid does not.
22. **Submap lifecycle**: insert into both active submaps; finish front at `2N`; drop front when a third would be created (front must be finished). Matching uses `submaps().front()`.
23. **Ownership**: replace `shared_ptr<const Submap2D>` aliasing of mutable submaps with `Arc<RwLock<Submap2D>>` or immutable snapshots (§7). Replace raw `ValueConversionTables*` with `Arc` tables.
24. **Proto compatibility**: `Grid2D.cells` are `repeated uint32` in proto but must be <= 65535; legacy protos with min/max cc both 0 -> defaults. Proto constructors do not validate cell counts (tests rely on this).
25. **Logging-only branches** (metrics, "every 500" log) can be no-ops but keep counters (`num_total_`, `num_different_`) since they influence only logging.
26. **`ExtrapolatePose` caching** is keyed on exact `Time` equality; repeated calls with the same time must not re-advance the tracker.
27. **`FromTwoVectors`**: implement Eigen's algorithm including the antiparallel SVD branch (any unit vector orthogonal to v0 suffices; Eigen's choice is the smallest right singular vector — for ImuTracker the antiparallel case is unreachable because of the CHECKs).
28. **`MotionFilter` time comparison** uses integer `Duration` ticks (`FromSeconds` rounds to 100 ns) — compare `i64` ticks, not floats.
