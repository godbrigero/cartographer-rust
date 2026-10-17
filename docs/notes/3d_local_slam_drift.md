# Vertical drift of 3D local SLAM in simulation

**Question.** In `cartographer-sim` (`World3d::room()`, 12 x 8 x 3 m, and the
loop corridor), `LocalTrajectoryBuilder3D` with the shipped
`trajectory_builder_3d.lua` drifts about 30 cm over 60 m, mostly vertically,
even with ideal (noise-free) sensors and no walking motion. Is this a port
bug or the behaviour of Cartographer itself?

**Answer: this is how Cartographer itself behaves. No port bug was found.**
The drift comes from how the scan matcher's cost function is designed. The
interpolated grid (`InterpolatedGrid`) peaks at voxel centres. That pulls
horizontal surfaces (the floor and the ceiling) onto the nearest layer of
voxel centres of the **0.45 m low-resolution grid**. The pull is up to a
third of a voxel (15 cm). With the shipped weights (`occupied_space_weight_1
= 6`, so 36x in the squared cost), this term outweighs the 0.1 m
high-resolution term along z. z is the weakly observed direction here: a
+-15 deg LiDAR at 1.2 m only sees the floor beyond 4.5 m and the ceiling
beyond 6.7 m. Each submap's grid is anchored at the sensor position when the
submap is created. The sensor height stays constant, so the floor lands at
the same offset from the voxel centres in every submap, and every submap
handover pulls the trajectory again in the same direction. The drift
therefore ratchets one way instead of averaging out.

All experiments are in `crates/cartographer/tests/local_slam_3d_drift.rs`.
Run them with
`cargo test -p cartographer-slam --test local_slam_3d_drift -- --ignored --nocapture`.
Two fast, non-ignored tests pin down the mechanism:
`occupied_space_cost_snaps_horizontal_plane_to_voxel_centres` and
`static_sensor_height_bias_follows_low_resolution_voxel_centres`.

## Code audit (Rust vs C++)

Each file below was compared line by line with the C++ (and with Ceres 2.0
for the parameterization). Everything matches:

* `local_trajectory_builder_3d.cc`: per-point extrapolation, range
  cropping, voxel filter (half size before accumulation), adaptive filters,
  `ScanMatch` (the translation prior targets the *predicted* translation,
  and the rotation prior targets the initial rotation),
  `local_from_gravity_aligned = pose.rotation * gravity_alignment^-1`, and
  the motion filter.
* `voxel_filter.cc`: the randomized reservoir sampling with `minstd_rand0`,
  `uniform_int_distribution` downscaling, and the unmasked `uint64` key
  packing.
* `pose_extrapolator.cc` and `imu_tracker.cc`: the
  `FromTwoVectors(gravity, orientation^-1 * z)` correction, the exponential
  gravity average, rotation from the tracker delta, and translation from
  the constant velocity of the last two poses. `use_imu_based = false` is
  the default, so `ImuBasedPoseExtrapolator` is not used.
* `ceres_scan_matcher_3d.cc`: weights divided by `sqrt(N)`, `DENSE_QR`, and
  `QuaternionParameterization`. `ceres::QuaternionParameterization`'s
  `Plus` (with the full angle |delta|, not the half angle) and
  `ComputeJacobian` match Ceres 2.0.
* `occupied_space_cost_function_3d.h`, `interpolated_grid.h`: the `(w, x,
  y, z)` order and Eigen's unnormalized `q * v` formula. `CenterOfLowerVoxel`
  computes `GetCenterOfCell(GetCellIndex(p))` and then moves one cell down
  per axis if needed. The tricubic blending order is z, then y, then x. Both
  versions use `float` for the cell centres and `double` for the weights.
* `hybrid_grid.h`: `GetCellIndex` is `lround(p / resolution)`, rounding half
  away from zero, and `GetCenterOfCell` is `index * resolution`.
  `ApplyLookupTable`/`FinishUpdate` use the update marker.
* `range_data_inserter_3d.cc`: the hits come first, then the last
  `num_free_space_voxels` miss cells using integer `delta * position /
  num_samples`.
* `submap_3d.cc`: the submap origin is `(range_data.origin,
  local_from_gravity_aligned)`, high-resolution data is cropped to 20 m,
  and the low-resolution grid receives all returns.

Numerical checks (experiment f):

* The autodiff Jacobians of the occupied-space cost match central finite
  differences. The largest difference is 2.3e-5 against |J| up to 216 for
  the high-resolution grid, and 2.6e-7 against 49 for the low-resolution
  grid.
* The solver result is a stationary point of the objective. The shipped 12
  iterations stop at cost 0.887937, and 100 allowed iterations converge
  after 15 iterations to 0.887933 at the same pose (within 0.2 mm).

## Experiments (ideal LiDAR, 16 rings over +-15 deg, instantaneous; ideal IMU)

### (a) Static sensor, room, 1.2 m height, 30 s

| | final x / y / z error | roll/pitch |
|---|---|---|
| defaults | -1.1 / +8.2 / **-4.2 cm** | 8.6 mrad |
| `low_resolution = 0.15` | +1.5 / +1.7 / **+0.3 cm** | 0.7 mrad |
| `occupied_space_weight_1 = 1` | -4.2 / +5.2 / -0.6 cm | 1.1 mrad |
| `translation_weight 0.5`, `rotation_weight 40` | +1.7 / +7.4 / -3.5 cm | 12.3 mrad |

A static sensor seeing identical noise-free scans still gets a
centimetre-level offset. Most of it appears in the first second, which
makes it a bias rather than noise. The offset almost disappears with a finer
low-resolution grid, and lowering the prior weights does not help.

### Static sensor at several heights (10 s): the sign follows the voxel-centre offsets

The first submap's origin is at the sensor, so the floor sits at `-h` and
the ceiling at `3 - h`. The predicted pull is the shift of each surface to
its nearest voxel-centre layer.

| h | 0.45 m grid: floor / ceiling pull | observed z | 0.15 m grid: floor / ceiling pull | observed z |
|---|---|---|---|---|
| 1.20 | -0.15 / 0 | **-3.3 cm** | 0 / 0 | **+0.2 cm** |
| 1.35 | 0 / +0.15 | **+2.0 cm** | 0 / 0 | **-0.2 cm** |
| 1.40 | +0.05 / +0.20 | **+3.6 cm** | +0.05 / +0.05 | **+3.6 cm** |
| 1.10 | +0.20 / -0.10 | **+6.8 cm** | +0.05 / +0.05 | **+4.2 cm** |

All eight signs match the prediction. Wherever both surfaces lie on
centre layers, the bias vanishes. The 0.1 m high-resolution grid has both
surfaces on centre layers at every height listed, so it is not the cause.

### (c)/(d) Perfect map built from ground-truth poses (room, 60 scans), then matching scans of the same tour

* Starting exactly at ground truth, the matcher moves z to **-3.1 cm** on
  average, so the minimum of its objective is not at ground truth.
* Starting from GT +-5 cm in z, it ends at -1.5 / -4.7 cm. This is
  asymmetric: the low-resolution cost is flat over about 10 cm around its
  minimum, so the result is a compromise with the translation prior.
* The cost profile of one scan along z is minimal at dz = 0.00 for the
  0.1 m grid and at **dz = -0.08** for the 0.45 m grid. The full 0.15 m
  pull toward the floor's centre layer is diluted by wall points.
* x offsets of +-5 / +-10 cm shrink to +2.3 / +0.2 and +3.4 / -0.8 cm.
  The translation prior (weight 5) pulls toward the initial guess, which in
  local SLAM is the constant-velocity prediction.

### Room tour (two laps of `Trajectory::room_wander`, 60 m, 1 m/s, constant height)

| | final z | max \|z\| | final xy | roll/pitch |
|---|---|---|---|---|
| defaults, h = 1.2 m | **-28.8 cm** | 31.4 cm | 8.5 cm | 15.1 mrad |
| (e) gravity-only IMU (exact gravity direction) | -20.3 cm | 21.3 cm | 7.4 cm | 4.6 mrad |
| `low_resolution = 0.15` | **-4.5 cm** | 9.4 cm | 12.0 cm | 2.0 mrad |
| `occupied_space_weight_1 = 1` | -9.1 cm | 13.1 cm | 6.0 cm | 10.4 mrad |
| `translation_weight 0.5`, `rotation_weight 40` | -30.0 cm | 32.6 cm | 14.7 cm | 15.2 mrad |
| gravity-only IMU, h = 1.35 m | **+11.9 cm** | 15.8 cm | | |
| gravity-only IMU, h = 1.40 m | **+23.5 cm** | 26.1 cm | | |
| gravity-only IMU, h = 1.10 m | +5.4 cm | 25.7 cm | | |

The drift direction reverses with the sensor height exactly as the
voxel-centre offsets predict. With defaults at 1.2 m, the z error at each
new submap's creation is -13.3 cm (17.3 s), -15.3 cm (33.7 s) and -29.4 cm
(49.9 s). The jumps follow the switch of the matching submap: the older,
front submap is used until it holds 2 x 160 range data.

The IMU contributes the remaining part. With the measured IMU, centripetal
acceleration in the turns tilts the `ImuTracker` gravity estimate (time
constant 10 s). The rotation prediction (last pose x tracker delta) carries
that tilt into the scan matcher's rotation prior. An IMU that reports the
exact gravity direction lowers roll/pitch error from 15 to 4.6 mrad and z
drift from 29 to 20 cm. This also matches the C++ behaviour.

### (b) Straight corridor (32 m along the south corridor, 1.2 m/s, from rest)

The estimate never picks up the motion along the corridor axis: the final x
error is -31.9 m, while y and z stay within 4 cm and 11 cm. This holds for
every option variant tried, including `translation_weight = 0.5`. This is a
separate effect, the axial degeneracy of a long, plain corridor combined
with the zero-velocity start of the constant-velocity extrapolator:

* The x constraint comes only from the far end wall and a few features.
* The weak map (`hit_probability` 0.55) is re-inserted from the lagging
  poses, so the map moves along with the estimate.
* The velocity estimate taken from those poses stays close to zero.

The `e2e_3d` corridor runs avoid this by starting next to features and
with handheld motion.

## Why the 2D pipeline is not affected

2D matching uses a single 5 cm grid with Ceres' bicubic interpolator. It
has no coarse grid with a 6x weight and no zero-derivative plateaus at
0.45 m voxel centres, and it does not estimate z at all.

## Consequences for the tests

* No tolerance changes are needed. The documented local-SLAM drift in
  `e2e_3d.rs` ("local SLAM drifts 33-37 cm, mostly in z, even with ideal
  sensors") is what Cartographer does here. Global SLAM (the 3D optimization
  with IMU, plus loop closure) removes most of it.
* Configuration (not port) levers: a finer `submaps.low_resolution` (for
  example 0.15) or a lower `occupied_space_weight_1` removes most of the
  vertical drift in these small, low-ceilinged worlds. A wider vertical field
  of view helps a little, because more floor and ceiling points reach the
  0.1 m grid.
