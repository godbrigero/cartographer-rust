# 2D local SLAM stalls on a slow robot with sparse, noise-free scans

**Question.** `examples/faithful_api.rs` builds options from the shipped Lua
(`map_builder.lua` + `trajectory_builder.lua`, 2D, `use_imu_data = false`,
`use_online_correlative_scan_matching = true`). It feeds 100 synthetic,
noise-free, 360-beam (1 deg) planar scans of the room `[-5, 5] x [-3, 3]`,
taken from `(x0 + 0.2 t, 0)` at 10 Hz. That is 2 cm (0.4 cells) per scan,
with no IMU and no odometry. With `x0 = 0` local SLAM tracks the motion
(final local x 1.93 m, true value 1.98 m). With `x0 = -1` it does not track
at all: the pose stays near 0 and only 2 nodes pass the motion filter. Is
this a port bug?

**Answer: this is how Cartographer itself behaves. No port bug was found.**
The input is degenerate, and the shipped configuration has no way to start
the extrapolator moving.

1. **The map is a single scan.** The first scan goes into the submap at the
   identity pose. After that the motion filter (0.2 m / 1 deg / 5 s) drops
   every scan until the estimate moves 0.2 m or 5 s pass. So every scan is
   matched against the grid of that one first scan.
2. **The side walls anchor "no motion".** The walls `y = +-3` run parallel
   to the motion, and the beams have fixed angles. So the robot-frame points
   on those walls are *identical* in every scan. Beyond about 1.5 m along
   the wall, 1 deg beams hit it more than one cell (5 cm) apart. The first
   scan therefore leaves a sparse comb of hit cells (p = 0.55). Between the
   teeth are unknown cells (correspondence cost 0.9). The instrumented run
   counted 117 hit cells along `y = 3` for `x0 = -1`, with gaps of up to
   0.25 m. At the "no motion" pose every side-wall point of a later scan
   lands exactly on a tooth. Any shift along x moves the points towards the
   gaps.
3. **Only the end walls see the motion, and they see it weakly.** Only the
   walls `x = +-5` (local `x = -4` and `x = +6` for `x0 = -1`) carry
   information about x. They are one cell thick, and the motion per scan is
   0.4 cells. How much they pull depends on where the wall sits inside its
   cell. For `x0 = -1` both end walls lie exactly on cell boundaries.
4. **Priors and coarse search keep the estimate there.** The Ceres scan
   matcher pulls the translation towards the *prediction* with
   `translation_weight = 10`. The extrapolator starts with zero velocity, and
   nothing else (IMU or odometry) can give it one. The real-time correlative
   scan matcher searches whole-cell offsets (5 cm) only, and the comb makes
   offset 0 the best one.

## Evidence (instrumented run, `x0 = -1`, 360 beams)

* **The grid is correct.** After the first insertion there are 360
  occupied cells, one per beam. All of them are within half a cell of the
  expected walls, and none is off a wall. All other known cells are free.
  The grid grows symmetrically from 100 to 400 cells (`max = (10, 10)`),
  the same for `x0 = 0` and `x0 = -1`. So asymmetric growth plays no role.
* **The cost landscape does not point to the truth.** The occupied-space
  cost along x for the 2nd scan (true offset +0.02 m) is lowest at about
  +0.005 m. Its curvature is about 24 /m^2, against 100 /m^2 for the
  translation prior. The combined optimum is about 0.001 m, which is what
  Ceres returns (0.0011). For `x0 = 0` the minimum lies at about +0.013 m.
  Ceres returns 0.004, and that is enough to give the extrapolator a
  velocity and to bootstrap tracking.
* **The correlative scan matcher always returns the prediction.** Per-scan
  mean probabilities, split into side-wall and end-wall points, show the
  side walls scoring best at offset 0 at every step (for example 0.324
  against 0.268 at +1 cell). The end walls cannot outvote them.
* **Removing the degeneracy fixes it everywhere.** These are final local x
  values (true value 1.98) for `x0` in {-1, -0.7, -0.5, -0.3, 0, 0.3, 0.5,
  1}:
  * 360 beams, as shipped: -1 gives 0.00, -0.5 gives 1.00, 0.5 gives
    1.17. The result depends erratically on `x0`.
  * 360 beams with a per-scan beam-angle phase in [0, 1) deg, which breaks
    the comb: 1.83 to 1.97 for every `x0` tested.
  * 360 beams with the end walls moved 1.3 cm off the cell boundaries: 1.99
    for every `x0`. With the *side* walls moved off the boundaries instead,
    tracking gets worse (stalls at 0 for `x0` <= 0). A sharper comb is a
    stronger anchor.
  * 720 or 1440 beams (0.5 or 0.25 deg), noise-free or with 1 cm Gaussian
    range noise: 1.89 to 2.00 for every `x0`.
  * 360 beams with `motion_filter.max_time_seconds = 0` (every scan is
    inserted): 1.91.

## Code audit (Rust vs C++)

Each stage on this path was compared with the C++, and the instrumentation
backs up the comparison:

* `Grid2D::GrowLimits` and `MapLimits::GetCellIndex`/`GetCellCenter`
  (including the `x <-> -y` cell axis swap and `RoundToInt` rounding half
  away from zero) match.
* `OccupiedSpaceCostFunction2D` matches: the `GridArrayAdapter` padding of
  `INT_MAX / 4` with `kMaxCorrespondenceCost`, the row/column order passed
  to `BiCubicInterpolator::Evaluate`, and the `limits.max()` offsets. The
  Ceres bicubic interpolator (Keys kernel) matches as well.
* `CeresScanMatcher2D` matches: the translation prior targets the
  *predicted* translation, the rotation prior targets the initial angle,
  and the scaling factor is `weight / sqrt(n)`.
* `RealTimeCorrelativeScanMatcher2D` matches: candidate scoring and the
  `exp(-(|t| w_t + |theta| w_r)^2)` delta cost.
* `RandomizedVoxelFilter` matches: reservoir sampling with
  `std::minstd_rand0`.
* The motion filter and the constant-velocity extrapolator behave as
  expected: the predictions are the linear extrapolation of the last two
  poses.
* The shipped `trajectory_builder_2d.lua` and `map_builder.lua` are
  byte-identical to the C++ ones.

Every quantity above is computed by the same formulas in C++. The C++ would
therefore stall the same way on this input.

## Tests and example

`crates/cartographer/tests/local_slam_2d_slow_room.rs`:

* `dense_scans_track_slow_motion_for_all_start_positions`: 1440 beams,
  `x0` in {-1, -0.5, 0, 0.5, 1}, final local x within 0.1 m of 1.98.
* `sparse_noise_free_scans_stall_from_minus_one`: pins the behaviour
  described above (360 beams, `x0 = -1`, local x stays near 0).

`examples/faithful_api.rs` now uses 1440 beams (0.25 deg, as typical 2D
LiDARs have) and keeps `x0 = -1`. A comment there explains why it does not
use 360 beams.
