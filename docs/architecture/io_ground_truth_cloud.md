# Architecture report: `io/`, `ground_truth/`, `cloud/` (Phase 0)

Source: `cartographer/cartographer/{io,ground_truth,cloud}` (read-only). Namespaces: `cartographer::io`, `cartographer::ground_truth`, `cartographer::cloud`.
All paths below are relative to `cartographer/cartographer/`.

---

## 0. Overview and dependency map

| Area | Purpose | Depends on (internal) |
|---|---|---|
| `io/points_*`, `*_points_processor` | Offline "assets writer" pipeline: stream `PointsBatch`es through a chain of processors configured by Lua; write PLY/PCD/XYZ/PNG/proto. Used by `cartographer_ros` `assets_writer_main` (not in this repo). | `common` (Lua dict, FixedRatioSampler, math), `mapping` (HybridGrid, ProbabilityGrid, RangeDataInserters, DetectFloors), `sensor::RangefinderPoint/PointCloud`, `transform` |
| `io/image`, `io/color`, `io/draw_trajectories`, `io/submap_painter` | Cairo-based raster image helpers. | Cairo, `mapping::Submap2D/3D` |
| `io/proto_stream*`, `io/internal/*` | `.pbstream` binary format reader/writer, deserializer, full-state serializer (`WritePbStream`), in-memory streams, `pbstream` CLI tool. | protobuf, boost::iostreams gzip (via `common/port.h`) |
| `io/serialization_format_migration` | pbstream v1 -> v2 (adds 3D submap rotational histograms). | `mapping::PoseGraph3D`, `RotationalScanMatcher`, Lua config `map_builder.lua` |
| `ground_truth/` | Relation-based SLAM accuracy metric (Kuemmerle et al. 2009): generate relations from a pose graph, read relations text file, compute error statistics. Two CLI binaries. | `io::DeserializePoseGraphFromFile`, `transform::TransformInterpolationBuffer` |
| `cloud/` | gRPC server (`MapBuilderServer`) exposing `MapBuilderInterface`; client stubs that implement `MapBuilderInterface`/`PoseGraphInterface`/`TrajectoryBuilderInterface` over gRPC; optional uplink (edge -> cloud) uploader; Prometheus metrics. Only built with `BUILD_GRPC=ON` (Prometheus with `BUILD_PROMETHEUS=ON`). | `async_grpc` (Cartographer's own wrapper lib), grpc++, prometheus-cpp |

Binaries (CMake `google_binary`):
- `cartographer_pbstream` <- `io/pbstream_main.cc` (subcommands `info`, `migrate`)
- `cartographer_autogenerate_ground_truth` <- `ground_truth/autogenerate_ground_truth_main.cc`
- `cartographer_compute_relations_metrics` <- `ground_truth/compute_relations_metrics_main.cc`
- `cartographer_grpc_server` <- `cloud/map_builder_server_main.cc` (gRPC only)

---

## 1. `io/` — Points processing pipeline

### 1.1 Core types

**`points_batch.h`**
```cpp
struct PointsBatch {
  common::Time start_time;                 // time of first point
  Eigen::Vector3f origin = Zero;           // sensor location in world (map frame)
  std::string frame_id;                    // sensor frame or ""
  int trajectory_id = 0;
  std::vector<sensor::RangefinderPoint> points;  // map frame; RangefinderPoint{Vector3f position}
  std::vector<float> intensities;          // optional (empty or same len as points)
  std::vector<FloatColor> colors;          // optional RGB floats in [0,1]
};
void RemovePoints(absl::flat_hash_set<int> to_remove, PointsBatch* batch);
```
`RemovePoints`: rebuilds `points`, and `intensities`/`colors` only if non-empty, skipping indices in set; preserves order. (Assumes `intensities`/`colors` length == points length when non-empty; indexes `[i]` unchecked.)

**`points_processor.h`**
```cpp
class PointsProcessor { // non-copyable
  enum class FlushResult { kRestartStream, kFinished };
  virtual void Process(std::unique_ptr<PointsBatch>) = 0;
  virtual FlushResult Flush() = 0;
};
```
Contract: each processor holds raw `PointsProcessor* next_` (non-owning) and forwards batches/flush. Driver loop (seen in tests and in cartographer_ros):
```
do { for batch in source: pipeline.back()->Process(batch) } while (pipeline.back()->Flush() == kRestartStream);
```
Multi-pass processors return `kRestartStream` from Flush. Writers (and HybridGrid/XRay/ProbabilityGrid) `LOG(FATAL)` if `next_->Flush()` returns `kRestartStream`, so every multi-pass stage (`voxel_filter_and_remove_moving_objects`) must come **before** them in the config. Writers are flushed (and close their files) once per Flush call, so they must only ever see one final pass.

**`null_points_processor.h`**: `NullPointsProcessor` — `Process` drops; `Flush` -> `kFinished`. Always the tail of the pipeline.

**`points_processor_pipeline_builder.{h,cc}`**
```cpp
class PointsProcessorPipelineBuilder {
  using FactoryFunction = std::function<std::unique_ptr<PointsProcessor>(common::LuaParameterDictionary*, PointsProcessor* next)>;
  void Register(const std::string& name, FactoryFunction);   // CHECK unique name
  std::vector<std::unique_ptr<PointsProcessor>> CreatePipeline(common::LuaParameterDictionary* dictionary) const;
};
void RegisterBuiltInPointsProcessors(const std::vector<mapping::proto::Trajectory>& trajectories,
                                     const FileWriterFactory&, PointsProcessorPipelineBuilder*);
```
- `CreatePipeline`: `dictionary` is a Lua array of tables. Result vector: `[0]` = `NullPointsProcessor`, then processors constructed **in reverse config order** (last config entry first), each getting `pipeline.back().get()` as `next`. So `pipeline.back()` is the **first** stage in the config; feed batches to `pipeline.back()`. Unknown `action` -> CHECK fail ("Unknown action ... Did you register the correspoinding PointsProcessor?").
- Three registration flavors: plain (`FromDictionary(dict, next)`), file-writing (`FromDictionary(file_writer_factory, dict, next)`), file-writing-with-trajectories (`FromDictionary(trajectories, factory, dict, next)`). NOTE: the with-trajectories lambda captures `trajectories` **by reference** — caller must keep the vector alive while building.

### 1.2 File writer abstraction (`file_writer.{h,cc}`, `fake_file_writer.{h,cc}`)
```cpp
class FileWriter { virtual bool WriteHeader(const char*, size_t) = 0;  // overwrite at file start
                   virtual bool Write(const char*, size_t) = 0;
                   virtual bool Close() = 0; virtual std::string GetFilename() = 0; };
class StreamFileWriter : FileWriter { StreamFileWriter(const std::string& filename); }  // std::ofstream out|binary
using FileWriterFactory = std::function<std::unique_ptr<FileWriter>(const std::string& filename)>;
class FakeFileWriter : FileWriter { FakeFileWriter(const std::string& filename, std::shared_ptr<std::vector<char>> content); }
```
- `StreamFileWriter`: file opened (truncated) in ctor. `Write`: false if `bad()` before/after. `WriteHeader`: `flush(); seekp(0); Write(...)` — overwrites the first `len` bytes in place (relies on header being same length each time; this is why counts are zero-padded to 15 digits). `Close`: `close()`, returns `!bad()`.
- `FakeFileWriter` (test helper, uses gtest `EXPECT_FALSE(is_closed_)` in each op): `Write` appends; `WriteHeader` resizes content to `len` **only if smaller**, then copies into beginning (overwrites prefix; never shrinks); `Close` marks closed. All return true.

### 1.3 Processor catalogue

All processors forward `Process` to `next_` (unless noted) and `Flush` -> `next_->Flush()` (unless noted).

| Action name (Lua `action`) | Class / file | Lua options | Behaviour |
|---|---|---|---|
| `dump_num_points` | `CountingPointsProcessor` / `counting_points_processor.*` | none | Adds `points.size()` to `num_points_` (int64). Flush: calls next; on `kFinished` logs "Processed N and finishing."; on `kRestartStream` logs "... and restarting stream." and resets counter to 0; propagates result. |
| `fixed_ratio_sampler` | `FixedRatioSamplingPointsProcessor` | `sampling_ratio` (double; CHECK `0 < r < 1` strictly) | Per point calls `common::FixedRatioSampler::Pulse()`; removes points where Pulse false. Pulse: `++num_pulses; if (double(num_samples)/num_pulses < ratio) {++num_samples; return true;} return false;` (deterministic; first point always kept). On `kRestartStream` from next, recreates sampler (deterministic repeat). |
| `frame_id_filter` | `FrameIdFilteringPointsProcessor` | `keep_frames` (string array, optional), `drop_frames` (string array, optional). CHECK exactly one non-empty. | Batch passed iff (keep non-empty && frame_id in keep) or (drop non-empty && frame_id not in drop). Otherwise batch is **dropped entirely** (not forwarded). |
| `min_max_range_filter` | `MinMaxRangeFilteringPointsProcessor` | `min_range`, `max_range` (double) | Stores squared ranges (double). Removes point unless `min^2 <= |p - origin|^2 <= max^2` (range_squared computed as float). Inclusive bounds. |
| `vertical_range_filter` | `VerticalRangeFilteringPointsProcessor` | `min_z`, `max_z` (double) | `dz = p.z - origin.z` (float); keep iff `min_z <= dz <= max_z`. |
| `voxel_filter_and_remove_moving_objects` | `OutlierRemovingPointsProcessor` | `voxel_size` (double), `miss_per_hit_limit` (double, optional; default 3 with LOG(INFO)) | 3-pass state machine on `mapping::HybridGridBase<VoxelData{int hits=0; int rays=0}>(voxel_size)`. **Phase 1** (Process): `++hits` at `GetCellIndex(p)`; nothing forwarded. Flush -> state=Phase2, return `kRestartStream` (does NOT call next->Flush). **Phase 2**: for each point, `delta=p-origin; length=|delta|; for (float x=0; x<length; x+=voxel_size) idx=GetCellIndex(origin + (x/length)*delta); if value(idx).hits>0: ++rays` (note: sampled ray march, includes the start, excludes endpoint unless step lands inside; `value()` on missing cell returns default without allocating; `x` is float, `voxel_size` double -> float accumulation). Flush -> Phase3, `kRestartStream`. **Phase 3**: remove point unless `rays < miss_per_hit_limit * hits` of its voxel; forward. Flush: `CHECK(next_->Flush()==kFinished)`, return kFinished. Logs "Marking hits...", "Counting rays...", "Filtering outliers...". |
| `color_points` | `ColoringPointsProcessor` | `frame_id` (string), `color` (array of 3 numbers 0..255) | Color values cast `static_cast<uint8>(double)` (truncation) then `ToFloatColor` (/255). If `batch.frame_id == frame_id`: replace `colors` with `points.size()` copies. |
| `intensity_to_color` | `IntensityToColorPointsProcessor` | `min_intensity`, `max_intensity` (double -> float), `frame_id` (optional, "" = all) | If intensities non-empty and frame matches: `colors = [gray,gray,gray]` with `gray = clamp((i-min)/(max-min), 0, 1)` per intensity (float). (Header comment says "*255" but value is in [0,1] FloatColor.) |
| `write_pcd` | `PcdWritingPointsProcessor` | `filename` | See 1.4. |
| `write_ply` | `PlyWritingPointsProcessor` | `filename` | See 1.4. Comments vector always empty from Lua. |
| `write_xyz` | `XyzWriterPointsProcessor` | `filename` | See 1.4. |
| `write_hybrid_grid` | `HybridGridPointsProcessor` | `voxel_size` (double), `range_data_inserter` (dict -> `mapping::CreateRangeDataInserterOptions3D`), `filename` | Each batch: `RangeDataInserter3D::Insert({origin, PointCloud(points), /*misses*/{}}, &hybrid_grid_, /*intensity_hybrid_grid=*/nullptr)`. Flush: `hybrid_grid_.ToProto()` (`mapping::proto::HybridGrid`) `SerializeToString` -> `Write` (raw protobuf bytes, no framing, no compression) -> `CHECK(Close())`; `LOG(FATAL)` if next returns restart. |
| `write_xray_image` | `XRayPointsProcessor` | `voxel_size`, `filename`, `transform` (dict: `translation={x,y,z}`, `rotation={roll,pitch,yaw}` or `{w=,x=,y=,z=}`, quaternion CHECK norm within 1e-9), `separate_floors` (bool, opt, default false), `draw_trajectories` (bool, opt, default **true**), `saturation_factor` (double, opt, default 1.0) | See 1.5. |
| `write_probability_grid` | `ProbabilityGridPointsProcessor` | `resolution`, `range_data_inserter` (dict -> `CreateProbabilityGridRangeDataInserterOptions2D`: `insert_free_space`, `hit_probability`, `miss_probability`), `filename`, `draw_trajectories` (opt, default true), `output_type` ("png" \| "pb", opt, default png; other -> LOG(FATAL)) | See 1.6. |

Registration order in `RegisterBuiltInPointsProcessors`: Counting, FixedRatioSampling, FrameIdFiltering, MinMaxRange, VerticalRange, OutlierRemoving, Coloring, IntensityToColor (plain); Pcd, Ply, Xyz, HybridGrid (file); XRay, ProbabilityGrid (file+trajectories).

### 1.4 Point-cloud file formats

**PCD (`write_pcd`)** — header is ASCII, data binary little-endian (host memcpy).
```
# generated by Cartographer\n
VERSION .7\n
FIELDS x y z[ rgb]\n
SIZE 4 4 4[ 4]\n
TYPE F F F[ U]\n
COUNT 1 1 1[ 1]\n
WIDTH %015d\n          (setw(15) setfill('0'))
HEIGHT 1\n
VIEWPOINT 0 0 0 1 0 0 0\n
POINTS %015d\n
DATA binary\n
```
- Per point: 12 bytes `float x,y,z` (memcpy, native endian); if batch has colors: 4 bytes `[B, G, R, 0]` from `ToUint8Color` (i.e. little-endian uint32 0x00RRGGBB).
- `has_colors_` decided by **first non-empty batch**; header written with count 0 at that time via `WriteHeader`; rewritten in `Flush` with final count (via `WriteHeader`, return value not checked) then `CHECK(Close())`. Empty batches are forwarded without touching file. Color written per batch only if that batch has colors (no consistency CHECK, unlike PLY -> can produce a corrupt file if batches mix).
- If no points ever: file is header-only with `has_colors_=false`, count 0.

**PLY (`write_ply`)** — ASCII header + binary little-endian body.
```
ply\n
format binary_little_endian 1.0\n
comment generated by Cartographer\n
[comment <c>\n ...]
element vertex %015d\n
property float x\nproperty float y\nproperty float z\n
[property uchar red\nproperty uchar green\nproperty uchar blue\n]
[property float intensity\n]
end_header\n
```
- Per point: 12 bytes xyz float; then if `has_colors_` 3 bytes R,G,B (uint8 via `ToUint8Color`); then if `has_intensities_` 4 bytes float.
- `has_colors_`/`has_intensities_` fixed by first non-empty batch; subsequent batches CHECK `colors.size()==points.size()` / `intensities.size()==points.size()` ("First PointsBatch had colors, but encountered one without. frame_id: ...").
- Header via `CHECK(WriteHeader)` on first batch (count 0) and in Flush (final count), then `CHECK(Close())` ("Closing PLY file_writer failed.").
- BUG: `has_intensities_` is **not initialized** in ctor; Flush with zero points reads it uninitialized (UB). Rust: initialize to `false`.

**XYZ (`write_xyz`)** — ASCII, one line per point: `stream << setprecision(6) << x << " " << y << " " << z << "\n"` (default float formatting = `%g` with 6 significant digits, e.g. `1e-07`, `123457`). Flush: `CHECK(Close())` ("Closing XYZ file failed."). Rust: must emulate C++ `%g`-style 6-significant-digit formatting (not `{}`), including exponent format `1e-07` (two-digit min exponent, sign).

### 1.5 X-ray images (`xray_points_processor.{h,cc}`)
Ctor: one `Aggregation{HybridGridBase<bool>(voxel_size), std::map<pair<int,int>, ColumnData{float sum_r,sum_g,sum_b; uint32 count}>}` per floor (or one if no floors). `bounding_box_` (`Eigen::AlignedBox3i`) is **shared across all floors**.
- `FromDictionary`: if `separate_floors`, CHECK `trajectories.size()==1`, `floors = mapping::DetectFloors(trajectories[0])` (`Floor{vector<Timespan{start,end}> timespans; double z}`).
- `Process`: no floors -> Insert into aggregation 0; else for each floor i where `start_time` in any timespan (`start <= t <= end` inclusive) Insert into aggregation i. Forward.
- `Insert`: per point: `camera_point = transform_ * point` (Rigid3f); `cell = voxels.GetCellIndex(pos)`; set voxel true; `bounding_box_.extend(cell)`; column key `(cell[1], cell[2])` (Y,Z); add color (default black `{0,0,0}` if batch has no colors; `colors.at(i)`) to sums; `++count`. Note: count is per point (not per voxel).
- `Flush`: no floors -> `WriteVoxels(agg0, factory(filename + ".png"))`; floors -> `factory(StrCat(filename, i, ".png"))` e.g. `xray0.png`. Then `LOG(FATAL)` on restart.
- `WriteVoxels`: if bbox empty -> LOG(WARNING) "Not writing output: bounding box is empty." and return **without Close** (file still created by factory). Image size `xsize = bbox.sizes()[1]+1`, `ysize = bbox.sizes()[2]+1`. Pixel of voxel index: `(bbox.max()[1] - idx[1], bbox.max()[2] - idx[2])` (projects along X axis, i.e. a side view looking along +X; Y flipped to image-right, Z up). For each occupied voxel (HybridGridBase iterator: only `true`-valued cells... iterator yields cells whose value != default i.e. true): pixel.mean_rgb = column sums / count; `++num_occupied_cells_in_column`.
- `IntoImage(matrix, saturation_factor)`: `max = numeric_limits<float>::min()` (smallest positive normal ~1.18e-38, NOT lowest); `max = max(log(n))` over non-empty pixels. Empty pixel -> white (255,255,255). Else `saturation = min(1.0, log(n)/max * saturation_factor)`; color = `Mix(1, mean, s) = 1*(1-s) + s*mean` per channel -> `ToUint8Color`. Edge case: if all columns have n=1, `log(1)/max = 0` -> white-ish = mean mixed 0 -> white. NaN not possible since max>0.
- If `draw_trajectories`: for each trajectory i, `DrawTrajectory(traj, GetColor(i), pose -> voxel_index_to_pixel(voxels.GetCellIndex((transform_ * pose.cast<float>()).translation())), image.GetCairoSurface())`.
- `image.WritePng(file_writer); CHECK(Close())`.
- Public accessor `bounding_box()`.

### 1.6 Probability grid (`probability_grid_points_processor.{h,cc}`)
```cpp
std::unique_ptr<Image> DrawProbabilityGrid(const mapping::ProbabilityGrid&, Eigen::Array2i* offset);
mapping::ProbabilityGrid CreateProbabilityGrid(double resolution, mapping::ValueConversionTables*);
```
- `CreateProbabilityGrid`: `kInitialProbabilityGridSize = 100`; `max = 0.5*100*resolution*(1,1)`; `ProbabilityGrid(MapLimits(resolution, max, CellLimits(100,100)), conversion_tables)` (grid grows as needed).
- Ctor: file opened via `factory(filename + ".png" | ".pb")`. WARNING if `pb` and draw_trajectories.
- Process: `ProbabilityGridRangeDataInserter2D::Insert({origin, PointCloud(points), {}}, &grid)` (3D points; inserter projects to 2D, ignores z).
- Flush PNG: `DrawProbabilityGrid`; if non-null, optionally draw trajectories with pixel = `grid.limits().GetCellIndex(pose.translation().head<2>().cast<float>) - offset`; `WritePng`, `CHECK(Close())`. Flush PB: `grid.ToProto()` (`mapping::proto::Grid2D`) serialized raw -> `Write` -> `CHECK(Close())`. Then LOG(FATAL) on restart.
- `DrawProbabilityGrid`: `grid.ComputeCroppedLimits(offset, &limits)` (offset = min of known-cells box; NB: for an empty grid it returns `CellLimits(1,1)`, so the "empty -> nullptr" branch is effectively unreachable and a 1x1 image is written). Image `num_x_cells x num_y_cells`; for each `xy` in `XYIndexRangeIterator(limits)`: `index = xy + offset`; value = known ? `ProbabilityToColor(GetProbability(index))` : `kUnknownValue = 128`; `SetPixel(xy.x, xy.y, {v,v,v})`.
- `ProbabilityToColor(p) = RoundToInt(255 * ((1-p) - kMinProbability) / (kMaxProbability - kMinProbability))` with `kMinProbability=0.1f`, `kMaxProbability=0.9f` (occupied -> dark). `RoundToInt` = `lround` (half away from zero).

### 1.7 Colors (`color.{h,cc}`)
```cpp
using Uint8Color = std::array<uint8,3>; using FloatColor = std::array<float,3>;
FloatColor GetColor(int id);                       // CHECK id >= 0
uint8 FloatComponentToUint8(float c) = (uint8) RoundToInt(Clamp(c,0,1)*255);
float Uint8ComponentToFloat(uint8 c) = c / 255.f;
Uint8Color ToUint8Color(FloatColor); FloatColor ToFloatColor(Uint8Color);
```
`GetColor(id)`: `hue = fmod(0.69f + 0.6180339887498949f * id, 1.f)` (float arithmetic), `HsvToRgb(hue, s=0.85f, v=0.77f)`. `HsvToRgb`: `h6 = (h==1)?0:6h; hi=floor(h6); f=h6-hi; p=v(1-s); q=v(1-fs); t=v(1-(1-f)s)`; hi 0..5 -> (v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q); else black. Use f32 throughout to bit-match.

### 1.8 Image (`image.{h,cc}`) — Cairo wrapper
```cpp
constexpr cairo_format_t kCairoFormat = CAIRO_FORMAT_ARGB32;
using UniqueCairoSurfacePtr = std::unique_ptr<cairo_surface_t, void(*)(cairo_surface_t*)>;  // cairo_surface_destroy
using UniqueCairoPtr        = std::unique_ptr<cairo_t, void(*)(cairo_t*)>;                  // cairo_destroy
UniqueCairoSurfacePtr MakeUniqueCairoSurfacePtr(cairo_surface_t*); UniqueCairoPtr MakeUniqueCairoPtr(cairo_t*);
class Image {
  explicit Image(UniqueCairoSurfacePtr surface);  // copies pixels; CHECK format ARGB32, stride == 4*width
  Image(int width, int height);                   // pixels zero (transparent black)
  const Uint8Color GetPixel(int x, int y) const;  void SetPixel(int x, int y, const Uint8Color&);
  void WritePng(FileWriter*);  void Rotate90DegreesClockwise();
  UniqueCairoSurfacePtr GetCairoSurface();        // cairo_image_surface_create_for_data over pixels_ (aliasing!)
  int width() const; int height() const;
 private: int width_, height_; std::vector<uint32> pixels_;   // row-major, index y*width+x
};
```
- Pixel encoding: `0xFF<<24 | r<<16 | g<<8 | b` (Cairo ARGB32 premultiplied, native-endian uint32; alpha always 255 when set via SetPixel). `GetPixel` ignores alpha.
- `WritePng`: creates aliasing surface, CHECK status success, `cairo_surface_write_to_png_stream(surface, CairoWriteCallback, file_writer)` CHECK success. Callback forwards to `FileWriter::Write`, returns `CAIRO_STATUS_WRITE_ERROR` on false. Output PNG is 8-bit RGBA (Cairo writes ARGB32 as RGBA PNG, un-premultiplied).
- `Rotate90DegreesClockwise`: new pixels in order `for x in 0..w: for y in h-1..=0: old[y*w+x]`; swap w/h.
- Drawing (DrawTrajectory) on the surface from `GetCairoSurface()` writes directly into `pixels_` (alpha-blended with premultiplied math). Image pixels from `Image(w,h)` start at 0 (transparent) — pixels not explicitly set remain transparent in the PNG.

### 1.9 `draw_trajectories.{h,cc}`
```cpp
using PoseToPixelFunction = std::function<Eigen::Array2i(const transform::Rigid3d& pose)>;
void DrawTrajectory(const mapping::proto::Trajectory&, const FloatColor&, const PoseToPixelFunction&, cairo_surface_t*);
```
Exact Cairo ops: return if `node_size()==0`. `cr = cairo_create(surface)`; `cairo_set_source_rgba(cr, c0,c1,c2, 0.7)`; `cairo_set_line_width(cr, 4.0)`; for each node `cairo_line_to(cr, px.x, px.y)` (first line_to on an empty path acts as move_to); `cairo_stroke`. Start marker: `set_source_rgba(0,1,0,0.7)`; `cairo_arc(cr, x0, y0, 6.0, 0, 2*M_PI)`; `cairo_fill`. End marker: `set_source_rgba(1,0,0,0.7)`; arc radius 6 at last node; `cairo_fill`. `cairo_surface_flush(surface)`; `cairo_destroy`. Defaults: antialias default, line cap BUTT, line join MITER, operator OVER. Pixel coords are integer (cell index), not pixel centers.

### 1.10 `submap_painter.{h,cc}`
```cpp
struct PaintSubmapSlicesResult { UniqueCairoSurfacePtr surface; Eigen::Array2f origin; /* top-left pixel of surface in map frame */ };
struct SubmapSlice { int width, height, version; double resolution; transform::Rigid3d slice_pose;
                     UniqueCairoSurfacePtr surface /*nullptr by default*/; std::vector<uint32_t> cairo_data /*must outlive surface*/;
                     transform::Rigid3d pose; int metadata_version = -1; };
struct SubmapTexture { struct Pixels { std::vector<char> intensity, alpha; } pixels; int width, height; double resolution; transform::Rigid3d slice_pose; };
struct SubmapTextures { int version; std::vector<SubmapTexture> textures; };
PaintSubmapSlicesResult PaintSubmapSlices(const std::map<mapping::SubmapId, SubmapSlice>&, double resolution);
void FillSubmapSlice(const transform::Rigid3d& global_submap_pose, const mapping::proto::Submap&, SubmapSlice*, mapping::ValueConversionTables*);
void DeserializeAndFillSubmapSlices(ProtoStreamDeserializer*, std::map<mapping::SubmapId, SubmapSlice>*, mapping::ValueConversionTables*);
SubmapTexture::Pixels UnpackTextureData(const std::string& compressed_cells, int width, int height);
UniqueCairoSurfacePtr DrawTexture(const std::vector<char>& intensity, const std::vector<char>& alpha, int width, int height, std::vector<uint32_t>* cairo_data);
```
(Used by cartographer_ros `pbstream_to_ros_map`/`occupancy_grid_node` and rviz; not used by anything else in this scope.)

- `CairoPaintSubmapSlices(scale, submaps, cr, callback)` (anon): `cairo_scale(cr, scale, scale)`; for each slice in map order: **if `surface == nullptr` -> `return`** (stops processing all remaining slices — upstream bug/behaviour; preserve or document); `homo = (pose * slice_pose)` as 4x4 affine (`Translation3d(t) * rotation`); `cairo_save`; `cairo_matrix_init(&m, xx=homo(1,0), yx=homo(0,0), xy=-homo(1,1), yy=-homo(0,1), x0=homo(0,3), y0=-homo(1,3))`; `cairo_transform(cr,&m)`; `cairo_scale(cr, res, res)`; callback(slice); `cairo_restore`. (Cairo matrix maps `x' = xx*x + xy*y + x0`, `y' = yx*x + yy*y + y0`; this encodes map->image axis swap: texture x along map -y... preserve exactly.)
- `PaintSubmapSlices(submaps, resolution)`:
  1. Bounding pass: 1x1 ARGB32 surface + context; callback calls `cairo_user_to_device(cr,&x,&y)` for 4 corners `(0,0),(w,0),(0,h),(w,h)` and extends `AlignedBox2f`. Scale = `1/resolution`.
  2. `kPaddingPixel = 5`; size = `(ceil(bbox.sizes().x) + 10, ceil(bbox.sizes().y) + 10)`; `origin = (-bbox.min.x + 5, -bbox.min.y + 5)`.
  3. `cairo_image_surface_create(ARGB32, size)`; `cr`: `cairo_set_source_rgba(0.5, 0, 0, 1)`; `cairo_paint` (dark red background = "unknown" marker); `cairo_translate(origin)`; paint pass with callback `cairo_set_source_surface(cr, slice.surface, 0, 0); cairo_paint(cr)` (default filter = GOOD/bilinear, operator OVER); `cairo_surface_flush`.
  4. Return `{surface, origin}`.
- `FillSubmapSlice`: build `Submap3D(proto.submap_3d())` if `has_submap_3d()` else `Submap2D(proto.submap_2d(), conversion_tables)`; `local_pose = submap.local_pose()` (unused); `submap.ToResponseProto(global_submap_pose, &response)` (`SubmapQuery::Response`); `slice.pose = global_submap_pose`; uses `response.textures(0)` (for 3D this is the high-res slice): `UnpackTextureData(cells,w,h)`; set width/height/resolution/slice_pose (`ToRigid3(texture.slice_pose)`); `surface = DrawTexture(...)`. Does not set `version`/`metadata_version`.
- `DeserializeAndFillSubmapSlices`: build `map<SubmapId, Rigid3d>` from `deserializer->pose_graph().trajectory[].submap[]` (`SubmapId(trajectory_id, submap_index)`, `ToRigid3(pose)`); then for each `SerializedData` with `has_submap()` and (`Has2DGrid` = has submap_2d && has grid; or `Has3DGrids` = has submap_3d && has low && high res grids): `FillSubmapSlice(submap_poses.at(id), ...)` (`.at` throws if missing).
- `UnpackTextureData`: `common::FastGunzipString(cells)`; CHECK size == `2*w*h`; interleaved bytes `[intensity, alpha]` per cell, row-major.
- `DrawTexture`: CHECK `cairo_data` empty; CHECK stride == 4*w. Per cell: `observed = (intensity==0 && alpha==0) ? 0 : 255`; pixel = `(alpha<<24) | (intensity<<16) | (observed<<8) | 0` (R=intensity, G=observed flag, B=0, A=alpha). NOTE: `intensity`/`alpha` stored as `char` (signed on x86) and converted to `uint8_t` first, so no sign-extension issue. Surface via `cairo_image_surface_create_for_data` over `cairo_data` (aliasing; data must outlive surface). CHECK status SUCCESS. These values are **not premultiplied-valid** in general (R may exceed A); Cairo treats them as premultiplied anyway — consumers (ROS) decode R/G/A channels directly.

---

## 2. `io/` — pbstream format and serialization

### 2.1 Interfaces (`proto_stream_interface.h`)
```cpp
class ProtoStreamWriterInterface { virtual void WriteProto(const google::protobuf::Message&) = 0; virtual bool Close() = 0; };
class ProtoStreamReaderInterface { virtual bool ReadProto(google::protobuf::Message*) = 0; virtual bool eof() const = 0; };
```

### 2.2 Binary format (`proto_stream.{h,cc}`) — EXACT
```
file   := magic:u64le  record*
magic  := 0x7b1d1f7b5bf501db   (bytes on disk: db 01 f5 5b 7b 1f 1d 7b)
record := len:u64le  payload[len]
payload := gzip( proto.SerializeToString() )   // one independent gzip member per message
```
- Little-endian u64 written byte-by-byte (`put(size & 0xff); size >>= 8`), portable.
- Compression: `common::FastGzipString` = `boost::iostreams::gzip_compressor(zlib::best_speed)` -> **gzip container (RFC 1952) with level 1**, written via `filtering_ostream` + `back_inserter`. boost's gzip header: magic 1f 8b, CM=8, flags=0 (no filename/comment), mtime=0, XFL (0 or per level), OS = boost default (`gzip::os_unknown`=255 per boost; any value is fine for readers). Decompression `FastGunzipString` = `gzip_decompressor`. Byte-exact compressed output is NOT required for interop (any valid gzip member is accepted); but readers must accept boost-produced gzip.
- `ProtoStreamWriter(filename)`: opens `ofstream(out|binary)`, writes magic immediately. `WriteProto`: serialize -> gzip -> write len + data. `Close()`: `out_.close(); return !out_.fail()`. No error checking on open beyond Close.
- `ProtoStreamReader(filename)`: opens `ifstream(in|binary)`; reads 8 bytes magic (`ReadSizeAsLittleEndian`: accumulates `size >>= 8; size += (u64)in.get() << 56` 8 times; returns `!in.fail()`); if fail or magic mismatch sets failbit; `CHECK(in_.good())` "Failed to open proto stream '<f>'." (process abort).
- `ReadProto(msg)`: read len (false on EOF/fail), read `len` bytes (false if short), gunzip, `ParseFromString` (false if parse fails). Note: `in->get()` at EOF returns `EOF(-1)` cast to u64 — irrelevant since fail is checked.
- `eof()`: `in_.eof()` — becomes true only **after** an attempted read hits EOF (i.e. after a failed `ReadProto`). Callers like `MigrateStreamVersion1ToVersion2` CHECK `input->eof()` after the read loop finishes.
- No trailer, no index, no checksums beyond gzip CRC32.

### 2.3 Stream layout (`internal/mapping_state_serialization.{h,cc}`)
```cpp
static constexpr int kMappingStateSerializationFormatVersion = 2;
static constexpr int kFormatVersionWithoutSubmapHistograms = 1;
void WritePbStream(const mapping::PoseGraph&, const std::vector<mapping::proto::TrajectoryBuilderOptionsWithSensorIds>& builder_options,
                   ProtoStreamWriterInterface* writer, bool include_unfinished_submaps);
```
Message order written by `WritePbStream` (each is one record; record 0 is `SerializationHeader`, all others are `SerializedData` with the given oneof):
1. `SerializationHeader{format_version = 2}`
2. `SerializedData.pose_graph = pose_graph.ToProto(include_unfinished_submaps)`
3. `SerializedData.all_trajectory_builder_options` = `options_with_sensor_ids[id]` for each id in `GetValidTrajectoryIds(GetTrajectoryStates())` (state != DELETED, ascending trajectory id). Indexing `builder_options[id]` assumes vector indexed by trajectory id.
4. Submaps: for each `GetAllSubmapData()` (MapById order: trajectory, then index): skip if `!include_unfinished && !submap->insertion_finished()`; `submap = submap->ToProto(/*include_probability_grid_data=*/true)`; set `submap_id{trajectory_id, submap_index}`.
5. Nodes: each `GetTrajectoryNodes()`: `node_id{traj, node_index}`, `node_data = ToProto(*constant_data)`.
6. TrajectoryData: per `GetTrajectoryData()` entry: `trajectory_id`, `gravity_constant`, `imu_calibration = Quaterniond(c[0],c[1],c[2],c[3])` (array is w,x,y,z), `fixed_frame_origin_in_map` if present.
7. IMU data: per trajectory id in `GetImuData().trajectory_ids()`, each datum -> `imu_data{trajectory_id, imu_data=sensor::ToProto}`.
8. Odometry data (same pattern).
9. Fixed-frame pose data (same pattern).
10. Landmark data: for each landmark node (map<string, LandmarkNode>, sorted by id) for each observation: one record `landmark_data{trajectory_id=obs.trajectory_id, landmark_data{timestamp=ToUniversal(obs.time), landmark_observations=[{id, landmark_to_tracking_transform, translation_weight, rotation_weight}]}}` (one observation per record).

`SerializedData` oneof field numbers: pose_graph=1, all_trajectory_builder_options=2, submap=3, node=4, trajectory_data=5, imu_data=6, odometry_data=7, fixed_frame_pose_data=8, landmark_data=9. `SerializationHeader{uint32 format_version=1}`.

### 2.4 `proto_stream_deserializer.{h,cc}`
```cpp
mapping::proto::PoseGraph DeserializePoseGraphFromFile(const std::string& file_name);
class ProtoStreamDeserializer {  // non-copyable, non-movable
  explicit ProtoStreamDeserializer(ProtoStreamReaderInterface* reader);
  mapping::proto::SerializationHeader& header();
  mapping::proto::PoseGraph& pose_graph(); const mapping::proto::PoseGraph& pose_graph() const;
  const mapping::proto::AllTrajectoryBuilderOptions& all_trajectory_builder_options();
  bool ReadNextSerializedData(mapping::proto::SerializedData* data);  // = reader->ReadProto(data)
};
```
Ctor (all failures are CHECK aborts): read header ("Failed to read SerializationHeader."); version must be 1 or 2 ("Unsupported serialization format \"N\""); read next -> must be `has_pose_graph()` ("Serialized stream misses PoseGraph." / "Serialized stream order corrupt. Expecting `PoseGraph` after `SerializationHeader`, but got field tag N"); read next -> must be `has_all_trajectory_builder_options()`; CHECK `pose_graph.trajectory_size() == options_with_sensor_ids_size()`.
`DeserializePoseGraphFromFile`: open `ProtoStreamReader`, construct deserializer, return pose graph copy (remaining stream unread).

### 2.5 In-memory streams (`internal/in_memory_proto_stream.{h,cc}`)
```cpp
class ForwardingProtoStreamWriter : ProtoStreamWriterInterface {
  using WriterCallback = std::function<bool(const google::protobuf::Message* proto)>;
  explicit ForwardingProtoStreamWriter(WriterCallback);
  void WriteProto(const Message& p) { CHECK(cb(&p)); }   bool Close() { return cb(nullptr); }
};
class InMemoryProtoStreamReader : ProtoStreamReaderInterface {
  explicit InMemoryProtoStreamReader(std::queue<std::unique_ptr<Message>>&&); InMemoryProtoStreamReader();
  template <typename M> void AddProto(const M&);   // pushes copy
  bool ReadProto(Message* p);   // false if empty; else p->CopyFrom(front) (CHECKs type match inside protobuf); pop
  bool eof() const { return queue.empty(); }        // differs from file reader: true BEFORE a failed read
};
```
Rust: `ReadProto` copies a type-erased message into a typed target — model as `enum` of the two stream message types (`SerializationHeader | SerializedData`), or store encoded bytes and decode into target (prost `Message::decode`), which is equivalent to `CopyFrom` for same-type messages.

### 2.6 Test helpers (`internal/testing/test_helpers.{h,cc}`)
`template<T> T ProtoFromStringOrDie(const std::string& textproto)` (TextFormat parse, CHECK). `ProtoReaderFromStrings(header_textpb, {data_textpbs...})` -> `InMemoryProtoStreamReader` with one `SerializationHeader` followed by `SerializedData`s. Rust: needs protobuf **text format** parsing for tests (prost lacks it; use `protobuf` crate v3 text_format, or `prost-reflect` with `text_format` feature, or hand-build messages in tests).

### 2.7 Serialization format migration (`serialization_format_migration.{h,cc}`)
```cpp
void MigrateStreamVersion1ToVersion2(ProtoStreamReaderInterface* input, ProtoStreamWriterInterface* output, bool include_unfinished_submaps);
mapping::MapById<SubmapId, proto::Submap> MigrateSubmapFormatVersion1ToVersion2(
    const MapById<SubmapId, proto::Submap>& submap_id_to_submaps, MapById<NodeId, proto::Node>& node_id_to_nodes,
    const proto::PoseGraph& pose_graph_proto);
```
**What v1 -> v2 means**: v2 3D submaps (`Submap3D.rotational_scan_matcher_histogram`, repeated float) carry a histogram; v1 didn't. 2D is unchanged.

`MigrateStreamVersion1ToVersion2` algorithm:
1. Load Lua options: `ConfigurationFileResolver({kSourceDirectory + "/configuration_files"})` (compile-time source dir!), code `include "map_builder.lua"; MAP_BUILDER.use_trajectory_builder_3d = true; return MAP_BUILDER` -> `CreateMapBuilderOptions`. `ThreadPool(1)`. CHECK not 2D. Build `PoseGraph3D(pose_graph_options, OptimizationProblem3D(...), &thread_pool)`.
2. `ProtoStreamDeserializer(input)` (accepts v1 or v2). Copy pose_graph proto. For each trajectory i: push `options_with_sensor_ids(i)` to builder options; CHECK `trajectory_id == i` (contiguous ids required).
3. Constraint trajectory-id "remapping" loop is a no-op (sets each id to itself).
4. Build `submap_poses`, `node_poses` (MapById) from pose graph trajectories. For each `landmark_poses` entry: `pose_graph.SetLandmarkPose(id, pose, /*frozen=*/true)`.
5. Read loop over SerializedData: pose_graph / all_trajectory_builder_options again -> LOG(FATAL) ("Found multiple serialized ..."). `submap`: CHECK `has_submap_3d()` ("Converting to the new submap format only makes sense for 3D."), collect. `node`: `pose_graph.AddNodeFromProto(node_poses.at(id), node)`; collect. `trajectory_data`: `SetTrajectoryDataFromProto`. imu/odometry/fixed_frame/landmark: `AddImuData` etc. with `sensor::FromProto`. default: LOG(WARNING) skip.
6. If header version == 1: `submaps = MigrateSubmapFormatVersion1ToVersion2(submaps, nodes, pose_graph_proto)`.
7. `AddSubmapFromProto(submap_poses.at(id), submap)` for each; `AddSerializedConstraints(FromProto(pose_graph_proto.constraint()))`; CHECK `input->eof()`; `WritePbStream(pose_graph, builder_options, output, include_unfinished_submaps)` (writes header version 2).

`MigrateSubmapFormatVersion1ToVersion2`: if input empty or **first** submap has `submap_2d` -> return input unchanged. Else copy; for each constraint with tag `INTRA_SUBMAP`: node must exist (CHECK) -> `node_data = FromProto(node.node_data())` (`TrajectoryNode::Data`); submap must exist & be 3D (CHECK). `yaw = transform::GetYaw( ToRigid3(submap_3d.local_pose).inverse().rotation() * node_data.local_pose.rotation() * node_data.gravity_alignment.inverse() )` where `GetYaw(q) = atan2((q*UnitX).y, (q*UnitX).x)`. `h = RotationalScanMatcher::RotateHistogram(node_hist, yaw)` (float angle). If submap histogram empty: set to `h`; else elementwise `+=` (assumes equal sizes; `Mutable(i)` unchecked).
`RotateHistogram(h, angle)`: empty -> return; `r = -angle * n / M_PI` (float); `full = RoundToInt(r - 0.5f)` (lround); `frac = r - full`; `while full<0: full += n`; `out = frac * h[(i+1+full)%n] + (1-frac) * h[(i+full)%n]`.

### 2.8 `pbstream` CLI (`pbstream_main.cc`, `internal/pbstream_info.*`, `internal/pbstream_migrate.*`)
- `main`: `InitGoogleLogging`, `FLAGS_logtostderr=true`, `ParseCommandLineFlags(remove_flags=true)`. `argv[1]`: `info` -> `pbstream_info(argc, argv)`; `migrate` -> `pbstream_migrate`; none/unknown -> usage ("Swiss Army knife for pbstreams...") and `EXIT_FAILURE`.
- `pbstream info <pbstream_filename> [--all_debug_strings=false]` (`int pbstream_info(int argc, char* argv[])`; needs argc>=3): logs header DebugString; each trajectory options DebugString; per trajectory "Trajectory id: X has #nodes N has #submaps M"; pose graph DebugString if flag; iterates all SerializedData counting by case name (`submap,node,trajectory_data,imu_data,odometry_data,fixed_frame_pose_data,landmark_data`) plus pre-initialized-to-0 `submap_2d, submap_2d_grid, submap_3d, submap_3d_high_resolution_hybrid_grid`; prints "SerializedData package contains #<name>: <count>" in std::map (alphabetical) order. BUG: unknown data case logs warning then dereferences `end()` iterator (UB) — Rust: skip.
- `pbstream migrate <input> <output> [--include_unfinished_submaps=true]` (argc>=4): `ProtoStreamReader`, `ProtoStreamWriter`, `MigrateStreamVersion1ToVersion2`, `CHECK(output.Close())`.
- Output goes to glog (stderr). Rust: `clap` subcommands + `log`/`tracing` to stderr.

---

## 3. `ground_truth/`

### 3.1 Proto (`proto/relations.proto`, proto3, package `cartographer.ground_truth.proto`)
```proto
message Relation { int64 timestamp1 = 1; int64 timestamp2 = 2;
                   transform.proto.Rigid3d expected = 3;   // relative transform of tracking frame from timestamp2 to timestamp1 (pose1^-1 * pose2)
                   double covered_distance = 4; }
message GroundTruth { repeated Relation relation = 1; }
```
Timestamps are Cartographer universal ticks (100 ns since 0001-01-01, `common::ToUniversal`).
GroundTruth file on disk = **raw serialized `GroundTruth` protobuf** (no pbstream framing, no gzip).

### 3.2 `relations_text_file.{h,cc}`
`proto::GroundTruth ReadRelationsTextFile(const std::string& relations_filename);`
- Whitespace-separated doubles, 8 per relation: `time1 time2 x y z roll pitch yaw` (Unix epoch seconds; Freiburg relations format). Read via `ifstream >> double` repeatedly until extraction fails; then `CHECK(stream.eof())` (any non-numeric garbage / partial record not at EOF -> abort; a trailing partial record at EOF is silently dropped).
- `UnixToCommonTime(t) = FromUniversal(kUtsEpochOffsetFromUnixEpochInSeconds * 10'000'000) + FromSeconds(t)`, `kUtsEpochOffsetFromUnixEpochInSeconds = 719162*24*60*60 = 62135596800`. `FromSeconds(double)` -> `duration_cast<Duration>(duration<double>(s))` (truncates toward zero to 100ns ticks).
- `expected = Rigid3d(Vector(x,y,z), RollPitchYaw(roll,pitch,yaw))` where `RollPitchYaw = AngleAxis(yaw,Z) * AngleAxis(pitch,Y) * AngleAxis(roll,X)`.
- `covered_distance` not set (0).

### 3.3 `autogenerate_ground_truth.{h,cc}`
`proto::GroundTruth GenerateGroundTruth(const mapping::proto::PoseGraph&, double min_covered_distance, double outlier_threshold_meters, double outlier_threshold_radians);`
Uses **only trajectory 0** (`pose_graph.trajectory(0)`; constraints must all be trajectory 0 — CHECKs).
1. `ComputeCoveredDistance(traj)`: CHECK node_size>0; `cd[0]=0; cd[i] = cd[i-1] + |(pose[i-1]^-1 * pose[i]).translation|` (indexed by position in `trajectory.node()` list, NOT by node_index — assumes node_index == position).
2. `ComputeSubmapRepresentativeNode(pg)`: iterate constraints with tag INTRA_SUBMAP in proto order: `next = vec.size(); s = submap_index; if (s <= next) continue; CHECK_EQ(s, next+1); vec.push_back(node_index)`. Result: `submap_to_node_index[i]` = node of the first INTRA constraint seen for submap `i+1` (the node where submap i+1 begins, which is roughly the middle of submap i, since consecutive submaps overlap by half). Requires constraints in proto order to introduce submaps 1,2,3,... without gaps (CHECK). Port exactly; do not "fix".
3. For each constraint with tag != INTRA_SUBMAP (i.e. INTER_SUBMAP): CHECK traj ids 0; skip if `submap_index >= vec.size()`. `matched = node_index; rep = vec[submap_index]`. `cd_c = |cd[matched] - cd[rep]|`; skip if `< min_covered_distance`.
   - `solution_pose1 = node(rep).pose; solution_pose2 = node(matched).pose; solution = p1^-1 * p2`
   - `submap_solution = trajectory.submap(submap_index).pose` (indexed by position in submap list)
   - `expected = (p1^-1 * submap_solution) * constraint.relative_pose`
   - `error = solution * expected^-1`; outlier if `|error.t| > outlier_threshold_meters || GetAngle(error) > outlier_threshold_radians` -> `++num_outliers`, skip.
   - Else add relation `{timestamp1 = node(rep).timestamp, timestamp2 = node(matched).timestamp, expected = ToProto(expected), covered_distance = cd_c}`.
4. LOG(INFO) "Generated N relations and ignored M outliers."
`GetAngle(T) = 2 * atan2(|q.vec|, |q.w|)`.

### 3.4 `autogenerate_ground_truth_main.cc`
Flags: `--pose_graph_filename` (""), `--output_filename` (""), `--min_covered_distance` (100.0 m), `--outlier_threshold_meters` (0.15), `--outlier_threshold_radians` (0.02). Missing either filename -> usage + EXIT_FAILURE. Reads `io::DeserializePoseGraphFromFile` (pbstream), generates, writes `GroundTruth.SerializeToOstream` to binary ofstream; CHECKs success. Returns 0 implicitly.

### 3.5 `compute_relations_metrics_main.cc`
Flags: `--pose_graph_filename`, `--relations_filename`, `--read_text_file_with_unix_timestamps` (false), `--write_relation_metrics` (false).
Algorithm:
1. `pose_graph = DeserializePoseGraphFromFile`; `TransformInterpolationBuffer buf(pose_graph.trajectory(0))` (node timestamps -> poses; slerp/lerp interpolation).
2. Ground truth: text file via `ReadRelationsTextFile` or binary proto `ParseFromIstream` (CHECK).
3. For each relation: `pose1 = LookupTransform(buf, FromUniversal(ts1))`, `pose2` likewise, where `LookupTransform`: if `buf.Has(t)` -> `Lookup(t)`; elif `t < earliest_time` -> `Lookup(earliest)`; else `Lookup(latest)` (clamping extrapolation). `error = (pose1^-1 * pose2) * expected^-1`; record `Error{translational_squared = |error.t|^2, rotational_squared = GetAngle(error)^2}`.
4. If `--write_relation_metrics`: CSV at `<pose_graph_filename>.relation_metrics.csv`, header
   `translational_error,squared_translational_error,rotational_errors_degree,squared_rotational_errors_degree,expected_translation_x,expected_translation_y,expected_translation_z,expected_rotation_w,expected_rotation_x,expected_rotation_y,expected_rotation_z,covered_distance` then one row per relation using default ostream double formatting (`%g`, 6 significant digits).
5. LOG(INFO) "Result:\n" + `StatisticsString`:
   ```
   Abs translational error <mean> +/- <sd> m
   Sqr translational error <mean> +/- <sd> m^2
   Abs rotational error <mean> +/- <sd> deg
   Sqr rotational error <mean> +/- <sd> deg^2
   ```
   where translational = sqrt(t2), rotational degrees = RadToDeg(sqrt(r2)), squared degrees = (deg)^2. `MeanAndStdDevString`: CHECK `values.size() >= 2` (aborts with <2 relations!); mean; **sample** std dev (divide by n-1); `std::fixed << setprecision(5)`: `"%.5f +/- %.5f"`.
Note TODO in source: translational error is asymmetric w.r.t. swapping pose1/pose2.

---

## 4. `cloud/` (summary-level)

### 4.1 Architecture
```
 [client process]                                   [server process: cartographer_grpc_server]
 MapBuilderStub : MapBuilderInterface  --gRPC-->  async_grpc::Server (N grpc threads, M event threads)
   PoseGraphStub : PoseGraphInterface               |-- 26 RpcHandlers (one class per RPC)
   TrajectoryBuilderStub : TrajectoryBuilderInterface|-- ExecutionContext = MapBuilderContext<Submap2D|Submap3D>
     (client-streaming Add*Data RPCs,                |        (access to MapBuilderServer internals; mutex-guarded by async_grpc
      server-streaming ReceiveLocalSlamResults        |         for GetContext(), unguarded for GetUnsynchronizedContext())
      read in a std::thread)                          |-- incoming_data_queue_ (BlockingQueue<Data{traj_id, unique_ptr<sensor::Data>}>)
                                                      |-- SLAM thread: PopWithTimeout(100ms) -> data->AddToTrajectoryBuilder(map_builder->GetTrajectoryBuilder(id))
                                                      |-- local/global SLAM subscription maps (absl::Mutex)
                                                      `-- optional LocalTrajectoryUploader (edge->cloud uplink, own thread, batches AddSensorDataBatch)
```
- `MapBuilderServerInterface { Start(); WaitForShutdown(); WaitUntilIdle(); Shutdown(); }`; `CreateMapBuilderServer(options, unique_ptr<MapBuilderInterface>)`; `RegisterMapBuilderServerMetrics(metrics::FamilyFactory*)`.
- `MapBuilderServer` ctor: `kMaxMessageSize = 100 MiB` send/recv; registers 26 handlers; if `uplink_server_address` non-empty creates uploader; execution context `MapBuilderContext<Submap2D>` if `use_trajectory_builder_2d`, `<Submap3D>` if 3d, else FATAL; sets pose graph `GlobalSlamOptimizationCallback` -> `OnGlobalSlamOptimizations`.
- `Start`: uploader->Start, start SLAM thread, grpc Start. `Shutdown`: flag, grpc Shutdown, join SLAM thread, uploader Shutdown. `WaitUntilIdle`: `incoming_data_queue_.WaitUntilEmpty(); pose_graph->RunFinalOptimization()`.
- `OnLocalSlamResult(traj, client_id, time, local_pose, range_data, insertion_result)`: if insertion_result && uploader: build `proto::SensorData` via `CreateSensorDataForLocalSlamResult(sensor_id="local_slam_result_<traj>", traj, client_id, time, starting_submap_index_, ...)` (each insertion submap -> `ToProto(include_grid = insertion_finished())`, submap_id `{traj, starting_index+k}`); if `insertion_submaps.front()->insertion_finished()` `++starting_submap_index_` (global counter, not per trajectory — known fragility); enqueue to uploader. Then under lock, call every local subscription for that trajectory with a copy; callback returning false -> erase (NB: erases from map while range-iterating — iterator invalidation bug in C++; Rust: use `retain`).
- `NotifyFinishTrajectory(traj)`: call each local subscription callback with `nullptr` (stream ends via `WritesDone`).
- `MapBuilderContext<SubmapType>`: holds `mapping::SubmapController<SubmapType>` (reassembles submaps from `LocalSlamResultData` for uplinked results) and `client_ids_` map (traj -> client_id; `RegisterClientIdForTrajectory` CHECKs not already registered; `CheckClientIdForTrajectory` exact match). `EnqueueLocalSlamResultData` creates `LocalSlamResult2D/3D(sensor_id, proto, &submap_controller_)` and pushes to queue. `GetLocalSlamResultCallbackForSubscriptions` returns callback that looks up client_id (unknown -> LOG(ERROR) ignore) and calls `OnLocalSlamResult`.
- Options (`map_builder_server_options.{h,cc}`, `proto/map_builder_server_options.proto`): `MapBuilderServerOptions{string server_address=1; int32 num_grpc_threads=2; int32 num_event_threads=3; mapping.proto.MapBuilderOptions map_builder_options=4; string uplink_server_address=5; int32 upload_batch_size=6; bool enable_ssl_encryption=7; bool enable_google_auth=9;}`. `CreateMapBuilderServerOptions(LuaParameterDictionary*)` reads keys `server_address, num_grpc_threads, num_event_threads, map_builder (dict), uplink_server_address, upload_batch_size, enable_ssl_encryption, enable_google_auth`. `LoadMapBuilderServerOptions(dir, basename)`. Default Lua `configuration_files/map_builder_server.lua`: threads 4/4, address `0.0.0.0:50051`, uplink "", batch 100, ssl/auth false, `MAP_BUILDER.collate_by_trajectory = true`.
- `map_builder_server_main.cc`: flags `--configuration_directory`, `--configuration_basename`; if `USE_PROMETHEUS`: `prometheus::FamilyFactory`, `RegisterAllMetrics`, `RegisterMapBuilderServerMetrics`, `prometheus::Exposer("0.0.0.0:9100")` ("/metrics"). Then `CreateMapBuilder(options.map_builder_options())`, server Start + WaitForShutdown.

### 4.2 Service `cartographer.cloud.proto.MapBuilderService` (`proto/map_builder_service.proto`, proto3)
Imports: `mapping/proto/{pose_graph,serialization,submap_visualization,trajectory_builder_options}.proto`, `sensor/proto/sensor.proto`, `transform/proto/transform.proto`, `google/protobuf/empty.proto`.

| RPC (path `/cartographer.cloud.proto.MapBuilderService/<Name>`) | Req -> Resp (stream) | Handler behaviour |
|---|---|---|
| AddTrajectory | `AddTrajectoryRequest{repeated SensorId expected_sensor_ids=3; TrajectoryBuilderOptions trajectory_builder_options=2; string client_id=4}` -> `{int32 trajectory_id=1}` | `map_builder.AddTrajectoryBuilder(ids, opts, local_slam_cb)`; register client_id; if uploader: clear `trajectory_builder_2d_options`, `_3d_options`, `pure_localization_trimmer`, `initial_trajectory_pose` and `uploader->AddTrajectory(...)` (error logged only). |
| AddSensorDataBatch | `AddSensorDataBatchRequest{repeated SensorData sensor_data=1}` -> Empty | For each: client-id check (NOT_FOUND "Unknown trajectory" -> Finish, abort rest); per-client Prometheus counters (`cartographer_sensor_data_total{client_id,trajectory_id,sensor}`); enqueue via `MakeDispatchable(sensor_id, FromProto(...))`; `local_slam_result_data` -> `EnqueueLocalSlamResultData`; unknown -> FATAL. |
| AddOdometryData / AddImuData / AddFixedFramePoseData / AddLandmarkData | `stream Add*DataRequest{SensorMetadata sensor_metadata=1; <data>=2}` -> Empty | `AddSensorDataHandlerBase`: per message client-id check (NOT_FOUND); enqueue; if uploader, forward as `SensorData` to uploader. Response Empty on `OnReadsDone`. |
| AddRangefinderData | `stream AddRangefinderDataRequest{metadata; TimedPointCloudData=2}` -> Empty | Same but **never** forwarded to uploader (range data is uplinked as LocalSlamResult instead). |
| ReceiveLocalSlamResults | `{int32 trajectory_id=1}` -> `stream ReceiveLocalSlamResultsResponse{trajectory_id=1; int64 timestamp=2; Rigid3d local_pose=3; RangeData range_data=4; LocalSlamInsertionResult{NodeId node_id=1} insertion_result=5}` | Subscribe; write per result; `nullptr` -> `WritesDone`; write failure -> unsubscribe (return false). `OnFinish` unsubscribes. |
| ReceiveGlobalSlamOptimizations | Empty -> `stream {map<int32,NodeId> last_optimized_node_ids=1; map<int32,SubmapId> last_optimized_submap_ids=2}` | Subscribe; unsubscribe on finish/write failure. |
| FinishTrajectory | `{trajectory_id=1; client_id=2}` -> Empty | client check; `map_builder.FinishTrajectory`; `NotifyFinishTrajectory`; uploader FinishTrajectory (error logged). |
| DeleteTrajectory | `{trajectory_id=1; client_id=2}` -> Empty | client check; `pose_graph->DeleteTrajectory`. |
| GetSubmap | `{SubmapId submap_id=1}` -> `{SubmapQuery.Response submap_query_response=1; string error_msg=2}` | `map_builder.SubmapToProto`. |
| GetTrajectoryNodePoses | Empty -> `{repeated TrajectoryNodePose{NodeId node_id=1; Rigid3d global_pose=2; ConstantPoseData{int64 timestamp=1; Rigid3d local_pose=2} constant_pose_data=3}}` | |
| GetTrajectoryStates | Empty -> `{map<int32, TrajectoryState{ACTIVE=0,FINISHED=1,FROZEN=2,DELETED=3}> trajectories_state=1}` | |
| GetLandmarkPoses | Empty -> `{repeated PoseGraph.LandmarkPose landmark_poses=1}` | |
| SetLandmarkPose | `{PoseGraph.LandmarkPose landmark_pose=1}` -> Empty | `SetLandmarkPose(id, pose)` (frozen defaults false). |
| GetAllSubmapPoses | Empty -> `{repeated SubmapPose{SubmapId=1; int32 submap_version=2; Rigid3d global_pose=3}}` | |
| GetLocalToGlobalTransform | `{trajectory_id=1}` -> `{Rigid3d local_to_global=1}` | |
| GetConstraints | Empty -> `{repeated PoseGraph.Constraint constraints=1}` | |
| IsTrajectoryFinished / IsTrajectoryFrozen | `{trajectory_id=1}` -> `{bool is_finished=1}` / `{bool is_frozen=1}` | |
| RunFinalOptimization | Empty -> Empty | |
| LoadState | `stream LoadStateRequest{oneof state_chunk{SerializedData serialized_data=1; SerializationHeader serialization_header=2; string client_id=3}; bool load_frozen_state=4}` -> `{TrajectoryRemapping{map<int32,int32> serialized_trajectories_to_trajectories=1} trajectory_remapping=1}` | Accumulates into `InMemoryProtoStreamReader`; on ReadsDone `map_builder.LoadState(&reader, load_frozen_state)` (last message's flag wins); register client_id for each new trajectory id. |
| LoadStateFromFile | `{string file_path=1; client_id=2; bool load_frozen_state=3}` -> `{trajectory_remapping=1}` | `LoadStateFromFile` on **server** filesystem. |
| WriteState | Empty -> `stream WriteStateResponse{oneof{SerializationHeader header=1; SerializedData serialized_data=2}}` | `ForwardingProtoStreamWriter` dispatching by `GetTypeName()`; `SerializeState(include_unfinished_submaps=false)`; Close -> WritesDone. |
| WriteStateToFile | `{string filename=1}` -> `{bool success=1}` | empty filename -> INVALID_ARGUMENT; `SerializeStateToFile(false, filename)`. |

Other messages: `enum SensorType{RANGE=0, IMU=1, ODOMETRY=2, FIXED_FRAME_POSE=3, LANDMARK=4, LOCAL_SLAM_RESULT=5}`, `SensorId{string id=1; SensorType type=2}`, `SensorMetadata{int32 trajectory_id=1; string sensor_id=2; string client_id=3}`, `SensorData{metadata=1; oneof{odometry_data=2, imu_data=3, timed_point_cloud_data=4, fixed_frame_pose_data=5, landmark_data=6, local_slam_result_data=7}}`.

### 4.3 Client stubs
- `MapBuilderStub(server_address, client_id)` (`client/map_builder_stub.*`): insecure channel; `WaitForConnected` 10 s else FATAL. Constants: `kChannelTimeoutSeconds=10`, `kRetryBaseDelayMilliseconds=500`, `kRetryDelayFactor=2.0`, `kMaxRetries=5` (AddTrajectory uses limited exponential backoff). Implements: AddTrajectoryBuilder (creates `TrajectoryBuilderStub`), GetTrajectoryBuilder (nullptr if unknown), FinishTrajectory (erases stub on success), SubmapToProto, SerializeState (reads WriteState stream into writer; warns if include_unfinished), SerializeStateToFile, LoadState (streams: client_id msg, header, pose_graph, all_trajectory_builder_options, then rest; CHECK reader eof), LoadStateFromFile, num_trajectory_builders, pose_graph. `AddTrajectoryForDeserialization`, `GetAllTrajectoryBuilderOptions` -> FATAL "Not implemented".
- `PoseGraphStub` (`internal/client/pose_graph_stub.*`): RunFinalOptimization, GetAllSubmapPoses, GetLocalToGlobalTransform, GetTrajectoryNodePoses, GetTrajectoryStates, GetLandmarkPoses, SetLandmarkPose (ignores `frozen`), DeleteTrajectory, IsTrajectoryFinished/Frozen, constraints. FATAL: GetAllSubmapData, GetSubmapData, GetTrajectoryNodes, GetTrajectoryData, ToProto, SetGlobalSlamOptimizationCallback.
- `TrajectoryBuilderStub(channel, trajectory_id, client_id, local_slam_result_callback)`: lazily opens one client-streaming RPC per sensor kind on first `AddSensorData`; if callback set, opens `ReceiveLocalSlamResults` and a reader `std::thread` invoking callback per response; dtor joins thread then `WritesDone`+`Finish` (CHECK ok) each open stream. `AddLocalSlamResultData` -> FATAL.
- Serialization helpers: `internal/sensor/serialization.*` (`CreateSensorMetadata`, `CreateAdd*Request`, `CreateSensorDataForLocalSlamResult`, `ToProto/FromProto(SensorId)`), `internal/mapping/serialization.*` (`ToProto/FromProto(TrajectoryState)`, `ToProto/FromProto(map<int,int> TrajectoryRemapping)`).

### 4.4 LocalTrajectoryUploader (`internal/local_trajectory_uploader.*`)
Interface: `Start(); Shutdown(); EnqueueSensorData(unique_ptr<proto::SensorData>); grpc::Status AddTrajectory(client_id, local_traj_id, expected_sensor_ids, options); grpc::Status FinishTrajectory(client_id, local_traj_id); SensorId GetLocalSlamResultSensorId(local_traj_id)` -> `{LOCAL_SLAM_RESULT, "local_slam_result_<id>"}`. Factory `CreateLocalTrajectoryUploader(addr, batch_size, ssl, google_auth)`.
- Credentials: google_auth -> `GoogleDefaultCredentials`; else ssl -> `SslCredentials({})`; else insecure. Connect wait 10 s (error logged, not fatal).
- Constants: `kConnectionTimeoutInSeconds=10`, `kConnectionRecoveryTimeoutInSeconds=60`, `kTokenRefreshIntervalInSeconds=60` (unused), `kPopTimeout=100ms`, unrecoverable codes `{DEADLINE_EXCEEDED, NOT_FOUND, UNAVAILABLE, UNKNOWN}`.
- `RegisterTrajectory`: AddTrajectory to uplink with expected sensors **minus RANGE** plus the local-slam-result sensor id; stores uplink trajectory id.
- `ProcessSendQueue` (thread): pop (100ms); translate trajectory id to uplink id (failure -> clear batch, `TryRecovery`); rewrite `submap_id.trajectory_id` in local slam result submaps; when batch size == `batch_size_` send `AddSensorDataBatch` with `UnlimitedConstantDelayStrategy(1s, unrecoverable codes)`; failure -> clear batch, `TryRecovery`. Partial batches are never flushed on shutdown.
- `TryRecovery`: if channel not READY, wait up to 60 s; drop queue items until the head is a LocalSlamResult whose last submap is new (`num_range_data == 1`); reset all uplink ids and re-register each trajectory.

### 4.5 Metrics (`metrics/prometheus/family_factory.*`)
Adapter implementing `cartographer::metrics::FamilyFactory` (`NewCounterFamily`, `NewGaugeFamily`, `NewHistogramFamily(name, desc, BucketBoundaries)`) on top of `prometheus::Registry`; `GetCollectable()`. Server gauge `cloud_internal_map_builder_server_incoming_data_queue_length`.

### 4.6 Porting cloud with tonic
- Codegen: `tonic-build` (prost) over `map_builder_service.proto` + all imported mapping/sensor/transform protos; `google.protobuf.Empty` -> `()` via prost-types (`compile_well_known_types` or extern path `::prost_types`/`()`). Service path names stay identical (`/cartographer.cloud.proto.MapBuilderService/<Rpc>`), so wire-compatible with C++ clients/servers.
- async_grpc concepts -> tonic: `RpcHandler::OnRequest` for unary -> `async fn`; `Stream<Req>` -> `tonic::Streaming<Req>` (loop `message().await`, respond at end = OnReadsDone); server-streaming -> return `ReceiverStream` from an `mpsc` channel; subscription callback writes into the channel, `send` error == client closed (return false -> unsubscribe); `nullptr` "finished" -> drop sender. `Finish(Status)` -> `Err(Status::not_found(...))`. ExecutionContext (`GetContext` = locked) -> `Arc<Mutex<Context>>` (or `Arc<Context>` with internal locks); `GetUnsynchronizedContext` -> lock-free accessors (queue is already thread-safe).
- SLAM thread consuming `incoming_data_queue_` -> dedicated `std::thread` (MapBuilder is sync/CPU-bound) fed by `crossbeam_channel` (recv_timeout 100ms) or `tokio::sync::mpsc` + `spawn_blocking`. Avoid calling blocking MapBuilder methods directly on tokio worker threads (`spawn_blocking`).
- Max message size: `.max_decoding_message_size(100<<20).max_encoding_message_size(100<<20)` on server and client.
- Client stubs: tonic client is async; `MapBuilderInterface` is sync -> wrap with a dedicated runtime and `block_on`, or make stubs async. Retry/backoff (async_grpc `CreateLimitedBackoffStrategy`, `CreateUnlimitedConstantDelayStrategy`) must be reimplemented (e.g., `tower::retry` or manual loop) with the same codes.
- Credentials: insecure / TLS (`tonic` `tls` feature, `ClientTlsConfig`) / Google default creds (needs `gcp_auth` crate + interceptor adding bearer token; refresh every 60 s).
- Metrics: `prometheus` crate + small HTTP exposer on `0.0.0.0:9100/metrics`.
- Recommend putting cloud behind a cargo feature (`grpc`), mirroring `BUILD_GRPC`.

---

## 5. External dependencies — exact usage

| Dep | Where | Exact usage | Rust replacement |
|---|---|---|---|
| **Cairo** (>=1.12.16) | `io/image.*`, `draw_trajectories.cc`, `submap_painter.*`, xray/probability grid processors | Format **ARGB32 only**. Calls: `cairo_image_surface_create`, `cairo_image_surface_create_for_data`, `cairo_format_stride_for_width` (CHECK == 4*w), `cairo_image_surface_get_{width,height,format,data}`, `cairo_surface_status`, `cairo_surface_write_to_png_stream`, `cairo_surface_flush`, `cairo_surface_destroy`, `cairo_create`, `cairo_destroy`, `cairo_set_source_rgba`, `cairo_set_source_surface`, `cairo_set_line_width`, `cairo_line_to`, `cairo_stroke`, `cairo_arc`, `cairo_fill`, `cairo_paint`, `cairo_scale`, `cairo_translate`, `cairo_save`, `cairo_restore`, `cairo_matrix_init`, `cairo_transform`, `cairo_user_to_device`, `cairo_status_to_string`. | Options: `cairo-rs` (exact behaviour, system lib), or pure-Rust `tiny-skia` (paths/stroke/fill/affine transforms/pattern draw with bilinear filter, premultiplied RGBA8; supports PNG via `png` crate). With tiny-skia, pixel output will differ slightly in antialiasing — acceptable unless golden-image tests are added. Pixel-only paths (SetPixel images, `DrawTexture`) need no rasterizer. PNG encode: `png` crate (un-premultiply ARGB32 -> RGBA8 like Cairo). |
| **boost::iostreams + zlib** | `common/port.h` `FastGzipString` / `FastGunzipString` used by proto_stream and `UnpackTextureData` (texture cells are gzip produced by `Submap::ToResponseProto`) | gzip compressor with `zlib::best_speed` (level 1); decompressor default. | `flate2::write::GzEncoder::new(.., Compression::fast())` / `GzDecoder` (use `MultiGzDecoder` defensively). |
| **protobuf** (>=3.0) | everywhere | Binary `SerializeToString`/`ParseFromString`/`SerializeToOstream`/`ParseFromIstream`; `CopyFrom`; `DebugString` (pbstream info); `TextFormat::ParseFromString` (tests); `MessageDifferencer::Equals` (tests); `GetTypeName()` (WriteState dispatch); `ByteSize`. | `prost` (+ `prost-build`). For DebugString use `{:#?}`; for text format in tests use `prost-reflect` text format or construct messages in code. Replace `GetTypeName` dispatch with a Rust enum. |
| **gflags** | 4 `main`s | `DEFINE_string/bool/double`, `ParseCommandLineFlags(&argc,&argv,true)` (removes flags), `SetUsageMessage`, `ShowUsageWithFlagsRestrict`. | `clap` derive. Keep flag names (`--pose_graph_filename` etc.; gflags also accepts `-flag` and `--flag value`). |
| **glog** | all | `LOG(INFO/WARNING/ERROR/FATAL)`, `CHECK*` (abort), `FLAGS_logtostderr`. | `log`/`tracing`; CHECK -> `assert!`/`panic!` or `Result` errors at API boundaries. |
| **gRPC / async_grpc** | `cloud/` | See §4.6. | `tonic`, `tokio`. |
| **prometheus-cpp** | `cloud/metrics/prometheus`, server main | Registry, Exposer on 9100. | `prometheus` crate + `hyper`/`axum`. |
| **Lua** (via `common::LuaParameterDictionary`) | pipeline builder, migration, server options | `GetString/GetDouble/GetInt/GetBool/HasKey/GetDictionary/GetArrayValuesAs{Dictionaries,Doubles,Strings}`. | Whatever `common` port provides (mlua). |
| **absl** | flat_hash_set/map, make_unique, StrCat, Mutex, optional | | std/`hashbrown`. |

---

## 6. Tests (every `*_test.cc` in scope)

### 6.1 io
| File | Test | Checks / inputs / tolerances |
|---|---|---|
| `io/fake_file_writer_test.cc` | `FakeFileWriter.Filename` | `GetFilename()=="file"`. |
| | `FakeFileWriter.CloseStream` | Close true; content "". |
| | `FakeFileWriter.WriteHeader` | WriteHeader("dummy header") -> content == "dummy header"; WriteHeader("dummy header 2") (longer) -> "dummy header 2"; after Close unchanged. |
| | `FakeFileWriter.Write` | Writes "data 1","data 2" -> "data 1data 2"; unchanged after Close. |
| | `FakeFileWriter.HeaderAndWrite` | header then two writes -> "dummy headerdata 1data 2". |
| | `FakeFileWriter.WriteTerminatedString` | Write `{'d','a','t','a','\0',' ','1'}` -> content equals exactly (embedded NUL preserved). |
| | `FakeFileWriter.WriteTerminatedHeaderString` | WriteHeader `{'h','e','a','d','\0',' ','e','r'}` -> exact. |
| | `FakeFileWriter.HeaderAndWriteTerminatedString` | WriteHeader 7 bytes w/ NUL, Write 8 bytes w/ NUL -> concatenation; unchanged after Close. |
| `io/points_processor_pipeline_builder_test.cc` | `PointsProcessorPipelineBuilderTest.RegisterBuiltInPointsProcessors` | `RegisterBuiltInPointsProcessors({}, factory returning nullptr, &builder)` doesn't crash (no duplicate names). |
| `io/probability_grid_points_processor_test.cc` | `ProbabilityGridPointsProcessorTest.WriteProto` (fixture) | Lua pipeline: one `write_probability_grid` with `resolution=0.05`, `range_data_inserter={insert_free_space=true, hit_probability=0.55, miss_probability=0.49}`, `draw_trajectories=false`, `output_type="pb"`, `filename="map"`. Factory asserts filename == `"map.pb"`. Batch: origin (0,0,0), points (0,0,0),(0,1,2),(1,2,4),(0,3,5),(3,0,6). Runs driver loop (`Process` then `Flush` until not restart). Expected bytes = independently built grid via `CreateProbabilityGrid(0.05)` + `ProbabilityGridRangeDataInserter2D::Insert` + `ToProto().SerializePartialToArray`. `ContainerEq` exact byte equality. Rust note: requires deterministic protobuf encoding of `Grid2D` identical between the two paths (same encoder — fine). |
| `io/proto_stream_test.cc` | `ProtoStreamTest.WriteAndReadBack` | Writes 10 `mapping::proto::Trajectory` each with one node `timestamp=i` to `./test_trajectory.pbstream`; Close true; reads back 10, each node_size 1 & timestamp i; 11th `ReadProto` false; removes file. |
| `io/proto_stream_deserializer_test.cc` | `ProtoStreamDeserializerTest.WorksOnGoldenTextStream` | In-memory reader: header `format_version: 1`; data `pose_graph {}`, `all_trajectory_builder_options {}`, `submap {}`, `node {}`, `trajectory_data {}`, `imu_data {}`, `odometry_data {}`, `fixed_frame_pose_data {}`, `landmark_data {}`. Checks header/pose_graph/options equal (MessageDifferencer), then 7 `ReadNextSerializedData` equal in order, then false and `reader.eof()`. Note `submap {}` text means oneof set with empty message — Rust prost must preserve "oneof set to empty message" (it does: `Some(Data::Submap(Default))`). |
| | `ProtoStreamDeserializerTest.FailsIfVersionNotSupported` | header `format_version: 123` -> death test, message contains "Unsupported serialization format". Rust: `#[should_panic(expected=...)]` or `Err`. |
| `io/serialization_format_migration_test.cc` | `SubmapHistogramMigrationTest.SubmapHistogramGenerationFromTrajectoryNodes` | Submap = `CreateFakeSubmap3D()` (traj 1, idx 1, num_range_data 1, identity local_pose, finished=true) with histogram cleared; Node = `CreateFakeNode()` (traj 1, idx 1, timestamp 42, identity local_pose, **gravity_alignment unset => zero quaternion**) + 10 histogram entries all 1.0; PoseGraph with one INTRA_SUBMAP constraint node->submap, identity relative pose. After migration: same size; submap has 3D not 2D; histogram size == 10 and each `EXPECT_EQ` (exact float) == 1.0. Pitfall: `gravity_alignment.inverse()` of a zero quaternion — Eigen returns zero quaternion (n2==0 branch), then `q*UnitX` with zero q yields UnitX (`v + w*uv + vec×uv` = v), yaw = 0; `RotateHistogram(h, 0)` gives `full=lround(-0.5)=-1 -> 9`, `frac=1` -> identity. Rust port must reproduce: quaternion inverse of zero = zero (not NaN), rotate-vector formula `v + 2w(q×v) + 2 q×(q×v)` (not via normalized matrix), and `lround` half-away-from-zero. |
| `io/internal/in_memory_proto_stream_test.cc` | `InMemoryProtoStreamTest.ReadStreamInitializedFromQueue` | Queue [PoseGraph(trajectory id 1), SerializedData(odometry_data.trajectory_id 2)]: eof false, read PG -> trajectory(0).id==1, eof false, read SD -> 2, eof true. |
| | `InMemoryProtoStreamTest.ReadStreamInitializedIncrementally` | Same via `AddProto`. |

No `ground_truth/*_test.cc` exist. No test data files (no `.pbstream`, `.ply`, `.pcd`, golden PNGs) are checked in for these modules; tests build inputs in code. The migration code path (`MigrateStreamVersion1ToVersion2`) and Bazel tests depend on `//:configuration_files` (`map_builder.lua` etc.) at `kSourceDirectory/configuration_files`.

### 6.2 cloud (gRPC-only)
| File | Tests | Notes |
|---|---|---|
| `cloud/internal/handlers/add_imu_data_handler_test.cc` | `AddImuDataHandlerTest.NoLocalSlamUploader`, `.WithMockLocalSlamUploader` | `HandlerTest` harness (`internal/testing/handler_test.h`): `async_grpc::testing::RpcHandlerTestServer` + `MockMapBuilderContext`, `MockLocalTrajectoryUploader`, `MockMapBuilder`, `MockPoseGraph`. Parses text request; expects `CheckClientIdForTrajectory` -> true, `DoEnqueueSensorData(trajectory_id, Pointee(Eq(expected dispatchable)))`, and with uploader `DoEnqueueSensorData(Pointee(EqualsProto(sensor_data)))`. |
| `add_odometry_data_handler_test.cc`, `add_fixed_frame_pose_data_handler_test.cc`, `add_landmark_data_handler_test.cc` | `*.NoLocalSlamUploader`, `*.WithMockLocalSlamUploader` | Same pattern. |
| `add_rangefinder_data_handler_test.cc` | `AddRangefinderDataHandlerTest.NoLocalSlamUploader` | Only enqueue checked. |
| `add_trajectory_handler_test.cc` | `AddTrajectoryHandlerTest.NoLocalSlamUploader`, `.WithLocalSlamUploader` | Mock `AddTrajectoryBuilder` returns 13; response trajectory_id == 13; with uploader expects `AddTrajectory` with options stripped of 2D/3D builder options etc. |
| `get_landmark_poses_handler_test.cc` | `GetLandmarkPosesHandlerTest.NoLocalSlamUploader` | Mock pose graph returns 2 landmarks; response EqualsProto expected text. |
| `set_landmark_pose_handler_test.cc` | `SetLandmarkPoseHandlerTest.SetLandmarkPose` | Expects `SetLandmarkPose("some_landmark", pose, false)`. |
| `local_trajectory_uploader_test.cc` | `LocalTrajectoryUploaderTest.HandlesInvalidUplink` | Uplink "localhost:99" (nothing listening); AddTrajectory status not ok; `GetLocalSlamResultSensorId` non-empty; FinishTrajectory not ok; Start/Shutdown. |
| `metrics/prometheus/metrics_test.cc` | `MetricsTest.CollectCounter`, `.CollectGauge`, `.CollectHistogram`, `.RunExposerServer` | Collect from registry: counter value, gauge value, histogram sample_count and bucket cumulative counts; exposer on a port. |
| `internal/client_server_test.cc` | `ClientServerTest.StartAndStopServer`, `.GlobalSlam3D`, `.DelayedConnectionToUplinkServer`; parameterized over `GridType{PROBABILITY_GRID, TSDF}` (`ClientServerTestByGridType`): `AddTrajectoryBuilder`, `AddTrajectoryBuilderWithMock`, `AddSensorData`, `AddSensorDataWithMock`, `LocalSlam2D`, `LocalSlamAndDelete2D`, `StartAndStopUploadingServerAndServer`, `AddTrajectoryBuilderWithUploadingServer`, `LocalSlam2DWithUploadingServer`, `LocalSlam2DUplinkServerRestarting`, `LoadStateAndDelete`, `LoadUnfrozenStateAndDelete`, `LocalSlam2DHandlesInvalidRequests` | Real servers on `0.0.0.0:50051` (main) and `0.0.0.0:50052` (uploading, uplink `localhost:50051`), client id `"CLIENT_ID"`. Fake range measurements `kDuration=4s`, `kTimeStep=0.1s`, `kTravelDistance=1.2m`; asserts local SLAM result count == measurements and `EXPECT_NEAR(1.2, |last.t - first.t|, 0.1*1.2)`. Load tests use text pbstream (`format_version: 2`, pose graph with one trajectory) and check remapping size 1, frozen/unfrozen state, delete -> DELETED, submap/node poses cleared. Invalid-request test: unknown trajectory -> IsTrajectoryFinished/Frozen false, `GetTrajectoryBuilder` nullptr, non-empty `SubmapToProto` error. |

---

## 7. Rust porting notes and pitfalls

**Pipeline design**
- Replace raw `next_` pointers with ownership by value: `Box<dyn PointsProcessor>` where each processor owns its `next: Box<dyn PointsProcessor>`; builder constructs from last config entry backwards (same as C++). Expose the head. `Process(Box<PointsBatch>)` -> `fn process(&mut self, batch: Box<PointsBatch>)`; `Flush` -> `fn flush(&mut self) -> FlushResult`.
- Factory registry: `HashMap<String, Box<dyn Fn(&mut LuaDict, Box<dyn PointsProcessor>) -> Box<dyn PointsProcessor>>>`. Trajectories: pass `Arc<[Trajectory]>` instead of borrowed ref.
- `FileWriter` trait with `write_header(&mut self, &[u8])`, `write`, `close`, `filename`. `StreamFileWriter` = `File` + `Seek(Start(0))` for header. `FileWriterFactory = Arc<dyn Fn(&str) -> Box<dyn FileWriter> + Send + Sync>`.
- Preserve LOG(FATAL)/CHECK semantics for misconfiguration (panic) but consider `Result` at the builder level.

**Bit/format fidelity**
- PLY/PCD floats: write `f32::to_le_bytes` (C++ writes host-endian; on x86/ARM LE identical). PCD color byte order B,G,R,0.
- 15-digit zero-padded counts: `format!("{:015}", n)`. If n is negative (impossible) differs; fine.
- XYZ and CSV use C++ default stream formatting (`%g`, precision 6): implement a `fmt_g(x, 6)` helper (strip trailing zeros, exponent form `e-07`/`e+06` when exp < -5 or >= precision). Stats strings use `%.5f`.
- `RoundToInt` == `f32::round` (half away from zero) then cast. `Clamp` as usual.
- `GetColor`: compute in f32 including `fmod` (`%` on f32 in Rust == C `fmodf`).
- `std::numeric_limits<float>::min()` in xray = `f32::MIN_POSITIVE`, NOT `f32::MIN`.
- Xray `ColumnData.count` is u32; `mean = sum / count` in f32.
- OutlierRemoving phase-2 ray loop: `x: f32` incremented by `voxel_size as f32`... in C++ `x += voxel_size_` promotes to double then narrows to float each iteration; replicate `x = (x as f64 + voxel_size) as f32` for exactness. `x/length` in float.
- PLY `has_intensities_` uninitialized -> use `false`.
- `DrawProbabilityGrid` never returns None for an empty grid (1x1 image) — replicate.
- Xray: writer file created even when bbox empty (no Close); bounding box shared across floors so per-floor images all have the same size.
- `CairoPaintSubmapSlices` returns (stops) at first slice with null surface — replicate or document deviation.
- `DrawTexture` stores non-premultiplied-consistent ARGB; if using tiny-skia (which requires valid premultiplied colors, `PremultipliedColorU8::from_rgba` fails if channel > alpha), you cannot store these values verbatim — keep a raw `Vec<u32>` and implement compositing manually, or use cairo-rs for this path.
- Image memory layout: native-endian u32 `0xAARRGGBB`. When converting for PNG, un-premultiply (Cairo does `c*255+a/2)/a`). For images from SetPixel alpha is 255 or 0.

**pbstream**
- Reader: read 8-byte LE magic, compare `0x7b1d1f7b5bf501db`; then loop `(u64 len, bytes)`; gunzip each with `flate2::read::GzDecoder` (single member); `prost::Message::decode`. Treat short read/EOF as end (false), set an `eof` flag only once a read hits EOF, to match `CHECK(input->eof())` semantics in migration. Guard huge `len` (C++ allocates `len` bytes blindly).
- Writer: `GzEncoder::new(Vec::new(), Compression::new(1))`. Byte-for-byte output will differ from boost (header OS byte, deflate impl) — round-trip tests only, never compare compressed bytes.
- Header is a different message type than the rest; model reader API as generic `read_proto<M: Message + Default>()`.
- `DeserializePoseGraphFromFile` only reads the first 3 records.
- `WritePbStream` indexes `builder_options[trajectory_id]` — requires options vector indexed by trajectory id including deleted trajectories.

**Migration**
- Depends on `PoseGraph3D` full machinery and compile-time `kSourceDirectory` config path; in Rust take config dir as a parameter / env var / embed `map_builder.lua` via `include_str!`.
- Zero-quaternion handling (see test notes) and `GetYaw` via rotating UnitX with unnormalized quaternion formula; `nalgebra::UnitQuaternion` will normalize/panic — use raw `Quaternion<f64>` math for this path.
- `RotateHistogram` uses f32 angle (`yaw as f32`).

**Ground truth**
- `ComputeCoveredDistance` and node lookups index by list position, not `node_index`; submap pose by list position. Keep.
- `ComputeSubmapRepresentativeNode` index semantics (off-by-one-ish) must be preserved exactly.
- `MeanAndStdDevString` panics with < 2 relations; preserve or return error.
- `UnixToCommonTime`: `FromSeconds(double)` truncation to 100 ns ticks: `(t * 1e7) as i64` toward zero (C++ `duration_cast` truncates); offset 62135596800 s.
- `LookupTransform` clamps outside range (no extrapolation). Needs `TransformInterpolationBuffer` port (from `transform/`).
- Relation proto file: plain `GroundTruth::decode(&bytes)`; `ParseFromIstream` of an empty file yields empty GroundTruth.
- Text relations: parse whitespace-separated f64 tokens; stop at first parse failure; error if not at EOF (C++ `CHECK(eof)`); drop incomplete trailing record.

**CLI**
- Use `clap` but accept gflags-style `--flag=value`; booleans as `--flag`/`--noflag` in gflags — document deviation if not supported.
- `pbstream info` unknown-data-case UB -> just warn & skip.

**Cloud**
- Iterator invalidation in `OnLocalSlamResult`/`OnGlobalSlamOptimizations` (erase while iterating) — use `retain`.
- `starting_submap_index_` is a single server-wide counter shared across trajectories — preserve behaviour (or fix with per-trajectory map and note it).
- Uploader never flushes partial batches; recovery drops data until next new submap.
- `LoadState` handler: `load_frozen_state` taken from last request; client_id must arrive via a dedicated oneof message (first message from stub).
- `WriteState` dispatches on protobuf type name strings `cartographer.mapping.proto.SerializationHeader` / `...SerializedData` — in Rust use a typed enum on `ForwardingProtoStreamWriter`.
- Thread-safety model: `GetContext` (mutex) vs `GetUnsynchronizedContext` matters for MapBuilder calls (MapBuilder is not thread-safe) — wrap MapBuilder in a `Mutex`, keep queues lock-free.
- Ports 50051/50052/9100 hardcoded in tests/defaults.
