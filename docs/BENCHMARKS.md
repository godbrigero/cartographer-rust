# Benchmarks

Criterion benchmarks of the hot paths of the `cartographer` crate live in
`crates/cartographer/benches/`. All inputs are generated deterministically
(fixed seeds) from the `cartographer-sim` simulator or from synthetic pose
graphs, outside the timed closures. The shared input generation is in
`benches/common/mod.rs`. Options come from the shipped Lua configuration
files (`trajectory_builder_2d.lua`, `trajectory_builder_3d.lua`,
`pose_graph.lua`) unless noted.

## Reproduce

```sh
cargo bench -p cartographer                          # everything, ~1.5 min on an M3
cargo bench -p cartographer --bench scan_matching    # one bench target
cargo bench -p cartographer -- scan_matching_2d/     # filter by id prefix
cargo bench -p cartographer --no-run                 # build only
```

Every bench uses `sample_size = 10`, 0.5 s warm-up and 2 s measurement time
(`common::criterion_config`), and plots are disabled.

## Result files

Criterion writes one directory per benchmark id:

```
target/criterion/<group>/<function>/new/estimates.json              # bench_function
target/criterion/<group>/<function>/<parameter>/new/estimates.json  # BenchmarkId
```

The ids below are exactly the path components (`<group>/<function>[/<parameter>]`).
`estimates.json` holds `mean`, `median`, `std_dev`, `median_abs_dev` and
(where meaningful) `slope`, each with `point_estimate` and
`confidence_interval.{lower_bound,upper_bound}` in **nanoseconds per
iteration**. Benches marked with a throughput below also write a
`benchmark.json` with the element count per iteration.

## Results

Apple M3, `cargo bench` (release profile), median of 10 samples with the 95 %
confidence interval of the median. The machine was otherwise loaded
(load average ~12-16 from other builds), so expect a few percent of noise.

| id | median | 95 % CI |
|---|---|---|
| `scan_matching_2d/ceres_match` | 309 µs | 308 µs – 317 µs |
| `scan_matching_2d/real_time_correlative_match` | 1.07 ms | 1.07 ms – 1.07 ms |
| `scan_matching_2d/fast_correlative_new` | 36.2 ms | 35.5 ms – 36.7 ms |
| `scan_matching_2d/fast_correlative_match` | 2.08 ms | 2.07 ms – 2.09 ms |
| `scan_matching_2d/fast_correlative_match_full_submap` | 69.9 ms | 69.2 ms – 70.7 ms |
| `scan_matching_3d/ceres_match` | 740 µs | 735 µs – 743 µs |
| `scan_matching_3d/real_time_correlative_match` | 545 µs | 539 µs – 556 µs |
| `scan_matching_3d/fast_correlative_new` | 8.96 ms | 8.93 ms – 9.06 ms |
| `scan_matching_3d/fast_correlative_match` | 526 µs | 523 µs – 534 µs |
| `grid_insertion_2d/probability_grid_720_beams` | 215 µs | 212 µs – 217 µs |
| `grid_insertion_2d/tsdf_720_beams` | 149 µs | 147 µs – 149 µs |
| `grid_insertion_3d/hybrid_grid_16_rings_raw` | 394 µs | 389 µs – 396 µs |
| `grid_insertion_3d/hybrid_grid_16_rings_voxel_filtered` | 95.1 µs | 94.2 µs – 95.7 µs |
| `grid_insertion_3d/low_resolution_hybrid_grid_16_rings_voxel_filtered` | 63.2 µs | 62.9 µs – 63.4 µs |
| `voxel_filter_3d/voxel_filter` | 85.4 µs | 85.3 µs – 85.8 µs |
| `voxel_filter_3d/adaptive_voxel_filter_high_resolution` | 111 µs | 111 µs – 111 µs |
| `voxel_filter_3d/adaptive_voxel_filter_low_resolution` | 131 µs | 130 µs – 132 µs |
| `optimization_problem_2d/solve/500_nodes` | 10.7 ms | 10.5 ms – 10.8 ms |
| `optimization_problem_2d/solve/2000_nodes` | 63.8 ms | 62.9 ms – 64.9 ms |
| `optimization_problem_3d/solve_with_imu/200_nodes` | 26.3 ms | 26.2 ms – 26.8 ms |
| `ceres_spa_2d_sparse_normal_cholesky/5_iterations/1000_poses` | 3.71 ms | 3.69 ms – 3.73 ms |
| `ceres_spa_2d_sparse_normal_cholesky/5_iterations/4000_poses` | 25.4 ms | 25.0 ms – 25.7 ms |
| `ceres_spa_2d_sparse_normal_cholesky/5_iterations/16000_poses` | 129 ms | 129 ms – 131 ms |
| `local_slam_2d/probability_grid_imu/100_scans` | 58.1 ms (1720 scans/s) | 57.6 ms – 58.3 ms |
| `local_slam_2d/probability_grid_imu_online_correlative/100_scans` | 170 ms (588 scans/s) | 169 ms – 170 ms |
| `local_slam_2d/tsdf_imu/100_scans` | 30.7 ms (3260 scans/s) | 30.5 ms – 31.0 ms |

## Benchmark descriptions

### `scan_matching` (`benches/scan_matching.rs`)

**2D fixture**: `LocalTrajectoryBuilder2D` (default 2D config, IMU on) runs
over 8 s of `World2d::rect_room` / `Trajectory::room_wander` (0.5 m/s, 720-beam
10 Hz LiDAR with 1 cm noise, 200 Hz IMU). The last inserted node provides the
scan (`filtered_gravity_aligned_point_cloud`, 205 points) and its local pose;
the matching submap (53 nodes, 5 cm, grown to 800 x 800 cells) the grid.

* `scan_matching_2d/ceres_match`: `CeresScanMatcher2D::match_`
  (`TRAJECTORY_BUILDER_2D.ceres_scan_matcher`, <= 20 iterations), initial
  estimate 3 cm / 2 cm / 0.6 deg off.
* `scan_matching_2d/real_time_correlative_match`:
  `RealTimeCorrelativeScanMatcher2D::match_` (0.1 m / 20 deg window).
* `scan_matching_2d/fast_correlative_new`: `FastCorrelativeScanMatcher2D::new`
  (precomputation of the 7-level grid stack, `POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher`).
* `scan_matching_2d/fast_correlative_match`: `match_` with a 7 m / 30 deg
  window, initial estimate (0.4 m, -0.3 m, 5.7 deg) off, `min_score` 0.55
  (found with score 0.85).
* `scan_matching_2d/fast_correlative_match_full_submap`: `match_full_submap`
  (global localization), `min_score` 0.6.

**3D fixture**: 80 simulated VLP-16-like scans (16 rings x 900 columns, 2 cm
noise, instantaneous) along `Trajectory::room_wander_3d` (1 m/s, handheld
motion) in `World3d::room`, at their ground-truth poses, are voxel filtered
(0.15 m) and inserted into `ActiveSubmaps3D` (`TRAJECTORY_BUILDER_3D.submaps`).
The 81st scan is the query, prepared like `LocalTrajectoryBuilder3D`
(adaptive voxel filters: 166 high / 207 low resolution points; rotational
histogram of the voxel-filtered cloud in the gravity frame).

* `scan_matching_3d/ceres_match`: `CeresScanMatcher3D::match_` on the high and
  low resolution grids (`TRAJECTORY_BUILDER_3D.ceres_scan_matcher`, <= 12
  iterations, no intensity grid), initial estimate (5, -3, 2) cm / 1 deg off.
* `scan_matching_3d/real_time_correlative_match`:
  `RealTimeCorrelativeScanMatcher3D::match_` (0.15 m / 1 deg window) on the
  high resolution grid.
* `scan_matching_3d/fast_correlative_new`: `FastCorrelativeScanMatcher3D::new`
  (precomputation, `POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d`,
  depth 8).
* `scan_matching_3d/fast_correlative_match`: `match_` with node pose
  (0.3, -0.2, 0.05) m / 2.9 deg off, `min_score` 0.55 (found with score 0.73,
  rotational 0.94, low resolution 0.84).

### `grid_insertion` (`benches/grid_insertion.rs`)

Each insertion bench inserts the same scan repeatedly into a grid that
already holds 20 other scans of the sequence (so grid growth is done before
timing; each iteration is a full insert + `finish_update`). Throughput is
reported per ray/point.

* `grid_insertion_2d/probability_grid_720_beams`:
  `ProbabilityGridRangeDataInserter2D::insert_into_probability_grid` of one
  unfiltered 720-beam scan (rect room, all beams return) into a 5 cm grid
  (`insert_free_space = true`).
* `grid_insertion_2d/tsdf_720_beams`: `TSDFRangeDataInserter2D::insert_into_tsdf`
  of the same scan (shipped TSDF options: normal estimation and projection on).
* `grid_insertion_3d/hybrid_grid_16_rings_raw`: `RangeDataInserter3D::insert`
  of a raw 16-ring scan (14400 points) into a 0.10 m `HybridGrid`.
* `grid_insertion_3d/hybrid_grid_16_rings_voxel_filtered`: the same scan
  after the 0.15 m voxel filter (3248 points), as local SLAM inserts it.
* `grid_insertion_3d/low_resolution_hybrid_grid_16_rings_voxel_filtered`:
  the voxel-filtered scan into a 0.45 m grid.
* `voxel_filter_3d/voxel_filter`: `voxel_filter` (0.15 m) of the raw
  14400-point scan.
* `voxel_filter_3d/adaptive_voxel_filter_{high,low}_resolution`:
  `adaptive_voxel_filter` of the voxel-filtered scan with the
  `TRAJECTORY_BUILDER_3D.{high,low}_resolution_adaptive_voxel_filter` options.

### `optimization` (`benches/optimization.rs`)

The problem is rebuilt in the untimed setup of every iteration
(`iter_batched`, `PerIteration`) because `solve` writes the result back.

* `optimization_problem_2d/solve/{500,2000}_nodes`:
  `OptimizationProblem2D::solve` (`POSE_GRAPH.optimization_problem`, <= 50
  iterations). Synthetic graph: nodes 0.2 m apart on laps of a 10 m radius
  circle, local SLAM drift of 1 cm / 0.1 deg per node, one submap per 20
  nodes, every node constrained to the two active submaps (intra), and
  Huber-robustified loop closures (2 cm / 0.3 deg noise) from every third node
  to the closest submap of an earlier lap within 3 m. 500 nodes: 25 submaps,
  1047 constraints (67 loop closures); 2000 nodes: 100 submaps, 4547
  constraints (567 loop closures). Odometry-free, so local SLAM relative-pose
  terms connect consecutive nodes.
* `optimization_problem_3d/solve_with_imu/200_nodes`:
  `OptimizationProblem3D::solve` with IMU terms (`fix_z_in_3d = false`): 20 s
  of `Trajectory::corridor_loops_3d` (1.2 m/s, handheld motion) with a 200 Hz
  simulated IMU (4001 samples), 199 nodes at 10 Hz with 3D local SLAM drift,
  10 submaps, intra constraints plus an inter-submap constraint from every
  5th node to the submap two back (410 constraints).
* `ceres_spa_2d_sparse_normal_cholesky/5_iterations/{1000,4000,16000}_poses`:
  a raw `ceres::Problem` 2D SPA (analytical SPA cost function from
  `cost_functions::spa_cost_function_2d`): poses on laps of a 20 m circle
  (500 per lap), noisy odometry edges and Huber loop closures to the pose one
  lap earlier (every second pose), first pose fixed, `SPARSE_NORMAL_CHOLESKY`,
  exactly 5 LM iterations (all tolerances 0) to show the per-iteration scaling
  (~linear: 3.7 / 25 / 129 ms).

### `local_slam` (`benches/local_slam.rs`)

`LocalTrajectoryBuilder2D` throughput: a 10 s sequence (`rect_room`,
`room_wander` at 0.5 m/s, 100 scans of 720 beams, 2000 IMU samples) is
converted to Cartographer inputs once; each iteration feeds a clone of all
inputs (cloned in the untimed setup) to a fresh builder. Every scan yields a
matching result (asserted). Throughput is reported in scans/s.

* `local_slam_2d/probability_grid_imu/100_scans`: shipped
  `trajectory_builder_2d.lua` defaults (probability grid, IMU, Ceres only).
* `local_slam_2d/probability_grid_imu_online_correlative/100_scans`: plus
  `use_online_correlative_scan_matching = true`.
* `local_slam_2d/tsdf_imu/100_scans`: TSDF grid + TSDF inserter.

## Optimizations made while adding the benches

Both keep results bit-for-bit identical (all `cargo test -p cartographer`
tests pass unchanged, including the simulated local SLAM accuracy tests; a
new proptest checks the buffer-reusing ray caster against the original).

1. **Ray casting buffer reuse** (`mapping/internal/two_d/ray_to_pixel_mask.rs`):
   new `ray_to_pixel_mask_into(begin, end, scale, &mut Vec)` that clears and
   fills a caller-provided buffer; `ray_to_pixel_mask` is now a thin wrapper.
   `ProbabilityGridRangeDataInserter2D` (`cast_rays`) and
   `TSDFRangeDataInserter2D` reuse one buffer for all rays of a scan instead of
   allocating (and regrowing) a `Vec` per ray.
   * `grid_insertion_2d/probability_grid_720_beams`: 383 µs -> 215 µs (-44 %)
   * `grid_insertion_2d/tsdf_720_beams`: 233 µs -> 149 µs (-36 %)
2. **Voxel filter hashing** (`sensor/internal/voxel_filter.rs`): the voxel
   map uses a splitmix64-finalizer hasher for the packed `u64` voxel keys
   instead of SipHash and is pre-sized to the point count. Which point is kept
   per voxel depends only on the reservoir sampling in input order, not on
   the hash, so outputs are identical.
   * `voxel_filter_3d/voxel_filter`: 446 µs -> 85 µs (-81 %)
   * `voxel_filter_3d/adaptive_voxel_filter_high_resolution`: 439 µs -> 111 µs (-75 %)
   * `voxel_filter_3d/adaptive_voxel_filter_low_resolution`: 555 µs -> 131 µs (-76 %)

Combined effect on local SLAM (before -> after, same machine):

* `local_slam_2d/probability_grid_imu/100_scans`: 80.5 ms -> 58.1 ms (-28 %)
* `local_slam_2d/probability_grid_imu_online_correlative/100_scans`: 194 ms -> 170 ms (-12 %)
* `local_slam_2d/tsdf_imu/100_scans`: 42.4 ms -> 30.7 ms (-28 %)
