# cartographer-demos

End-to-end demos of the Cartographer Rust port on recorded data. This crate is
not published (`publish = false`). Its library is only a tiny pure-Rust raster
(`canvas`: lines, discs, a 3×5 bitmap font, PNG output) used for the plots.

## `unitree_l1_bag`: 3D SLAM on a Unitree L1 ROS1 bag

The example reads a ROS1 bag with the pure-Rust `cartographer_rosbag` reader.
It runs `cartographer::Slam` (`Mode::ThreeD`) on the LiDAR and IMU data and
compares the result with the ground truth stored in the bag.

```bash
cargo run --release -p cartographer-demos --example unitree_l1_bag -- \
    <BAG> [--out DIR] [--lua FILE] [--record FILE]
```

- `--out DIR`: output directory. Default: `target/unitree_l1/`.
- `--record FILE`: also record the run over time for the video renderer
  (see [Video of a run](#video-of-a-run)).
- `--lua FILE`: a cartographer_ros-style Lua file that replaces the built-in
  configuration. The file must return `{ map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER, ... }` with
  `MAP_BUILDER.use_trajectory_builder_3d = true`. Inside it,
  `include "map_builder.lua"` and `include "trajectory_builder.lua"` always
  resolve. Other keys, such as frame names, are ignored.

### Data

The bags come from the MuJoCo simulator in `tools/unitree_l1_sim` (branch
`unitree-l1-sim`; see its `README.md` and `SPECS.md`). A simulated Unitree
4D LiDAR L1 is mounted upright 0.55 m above the floor. It runs 90 s, about 3
laps (98.9 m) of an ellipse around a partition, inside a 20 × 12 × 3 m room.

| Topic | Content |
|---|---|
| `/unilidar/cloud` | `PointCloud2`, 10 Hz, 2,160 points per cloud, frame `unilidar_lidar`. `header.stamp` is the time of the **first** point; the per-point `time` field is seconds since the stamp (0 to 0.1). |
| `/unilidar/imu` | 250 Hz, frame `unilidar_imu`. The noisy bag adds white noise and a constant gyro/accel bias. |
| `/tf_static` | `unilidar_imu` → `unilidar_lidar`: translation (0.0077, 0.0147, −0.0067) m, no rotation |
| `/ground_truth/pose` | 100 Hz pose of `unilidar_imu` in `map` |

The L1 sees only the upper hemisphere. Mounted upright, it never sees the
floor, and about 70 % of its points land on the ceiling.

### What the example does

1. **Frames.**
   - The tracking frame is `unilidar_imu`. Cartographer 3D requires the IMU
     frame as the tracking frame.
   - The LiDAR extrinsics are looked up in `/tf_static` as
     `tracking_from_lidar` (the TF `parent_from_child`, inverted if the edge
     points the other way).
   - The IMU frame is the tracking frame, so the IMU extrinsics are the
     identity.
2. **Times.**
   - Each cloud is converted with `cartographer::slam::points_relative_to_first(header.stamp, (xyz, time))`.
     The cloud's timestamp becomes the time of its last point, and point
     times become ≤ 0, which is Cartographer's convention. This is what
     cartographer_ros does for `PointCloud2` with a `time` field.
   - IMU samples use their `header.stamp`.
   - Messages are fed in bag-record order. A cloud is recorded when its last
     point is taken, so all sensors arrive in time order.
3. **SLAM.** Data goes through `Slam::push_imu` and `Slam::push_lidar`, then
   `Slam::finish()` runs. The example prints progress every 10 s of bag time,
   the realtime factor, and pose graph statistics. Constraints are counted by
   type:
   - intra-submap;
   - inter-submap;
   - loop-closing constraints, a subset of inter-submap: those against a
     submap older than the node's previous submap;
   - inter-trajectory.
4. **Outputs** in `--out`:

| File | Content |
|---|---|
| `map.ply` | Occupied voxels (p ≥ 0.6) of the high-resolution submaps, in Cartographer's map frame (binary PLY: x, y, z, probability) |
| `map.pbstream` | Full SLAM state |
| `trajectory_tum.txt` | Optimized node poses, TUM format (`t x y z qx qy qz qw`) |
| `local_trajectory_tum.txt` | Local-SLAM node poses (no loop closure) |
| `ground_truth_tum.txt` | `/ground_truth/pose` in TUM format |
| `metrics.json` | Timing, statistics, the main configuration values and the evaluation |
| `top_down.png` | Top-down voxel density with the ceiling removed. The ceiling height is the most populated 10 cm z slice; everything within 30 cm below it is cut. Ground truth (green), optimized (red) and local-only (blue) trajectories are overlaid. |
| `side_xz.png`, `side_yz.png` | Side views (all heights) with the same trajectories |
| `error_vs_time.png` | Translation error against time after SE(3) alignment, optimized and local |
| `loop_closures.png` | Loop-closure constraints drawn as lines from the node to the submap origin, on the top-down map. Submap origins are labeled. |

   The plots are drawn in the ground-truth `map` frame. The map and the
   trajectories are moved there with the SE(3) alignment of the evaluation.
5. **Evaluation**, run when `/ground_truth/pose` exists:
   - **Association:** the ground truth is linearly interpolated (slerp for
     rotation) at each node time, if its two bracketing samples are at most
     50 ms apart.
   - **ATE:** RMSE and max of the translation error after an SE(3)
     Umeyama/Horn alignment on positions. This uses
     `cartographer_sim::metrics`.
   - **RPE:** over 1 m of ground-truth path.
   - **Final drift:** the position error of the last node with both
     trajectories aligned at their first pose only, with no global fit. It
     is also given as a percentage of the path length.
   - All metrics are computed for the optimized trajectory and for the local
     (local-SLAM-only) trajectory.

### Exact commands used for the results

The bags are generated on branch `unitree-l1-sim`:

```bash
cd tools/unitree_l1_sim
.venv/bin/python generate_bag.py                                               # out/unitree_l1_loop.bag
.venv/bin/python generate_bag.py --no-noise --output out/unitree_l1_loop_nonoise.bag
```

Then, from the workspace root:

```bash
cargo run --release -p cartographer-demos --example unitree_l1_bag -- \
    tools/unitree_l1_sim/out/unitree_l1_loop.bag --out target/unitree_l1/noisy
cargo run --release -p cartographer-demos --example unitree_l1_bag -- \
    tools/unitree_l1_sim/out/unitree_l1_loop_nonoise.bag --out target/unitree_l1/nonoise
```

### Results

Machine: Apple M3 Max, release build. Pose graph: 899 nodes, 8 submaps.

| Bag | Trajectory | ATE RMSE | ATE max | Rot. RMSE | RPE 1 m (transl. / rot.) | Final drift |
|---|---|---|---|---|---|---|
| noisy | **optimized** | **1.8 cm** | 9.2 cm | 0.19° | 1.8 cm / 0.09° | 3.3 cm (0.03 %) |
| noisy | local SLAM only | 7.2 cm | 26.8 cm | 0.78° | 5.9 cm / 0.68° | 27.9 cm (0.28 %) |
| noise-free | **optimized** | **2.8 cm** | 7.1 cm | 0.15° | 1.8 cm / 0.08° | 2.6 cm (0.03 %) |
| noise-free | local SLAM only | 18.7 cm | 50.1 cm | 0.92° | 6.1 cm / 0.72° | 69.5 cm (0.70 %) |

| Bag | Constraints | Loop closures | Global optimizations | Wall time (90 s bag) | Realtime factor |
|---|---|---|---|---|---|
| noisy | 2765 (1678 intra + 1087 inter) | 406 | 11 | 2.2 s (feed 2.0 s, `finish()` 0.2 s) | **~41×** |
| noise-free | 2719 (1678 intra + 1041 inter) | 390 | 11 | 2.1 s | **~43×** |

- The local-SLAM numbers vary with any change to the configuration. Across
  the robustness runs below they range from 5 to 19 cm. The optimized
  numbers stay between 1.7 and 4.2 cm.
- Loop closure runs on 4 background threads, so the optimized numbers can
  change by a few millimeters between runs. Local SLAM is deterministic.

### Video of a run

`--record FILE` writes a JSON-lines log of the run as it happens (about
9 MB for the 90 s bag), and `scripts/render_slam_video.py` turns it into a
1600×900, 30 fps video (about 40 s, one frame per node plus a short
animation after each global optimization):

```bash
cargo run --release -p cartographer-demos --example unitree_l1_bag -- \
    tools/unitree_l1_sim/out/unitree_l1_loop.bag --out target/unitree_l1 \
    --record target/unitree_l1/slam_run.jsonl
# Python with numpy and matplotlib (e.g. the simulator's venv); ffmpeg on PATH.
tools/unitree_l1_sim/.venv/bin/python crates/demos/scripts/render_slam_video.py \
    target/unitree_l1/slam_run.jsonl -o target/unitree_l1/slam_run.mp4 \
    --keyframes target/unitree_l1/keyframes
```

`--frames 100,400` renders only those frames as PNGs into `--keyframes`,
which is useful to check the layout. `--jobs N` sets the number of rendering
processes. Rendering takes about 40 s on 14 processes.

The log has one JSON object per line. Poses are `[x, y, z, qx, qy, qz, qw]`
and times are bag seconds.

| `type` | Content |
|---|---|
| `header` | Bag start time `t0` and the frame conventions |
| `ground_truth` | `/ground_truth/pose` as `[t, x, y, z, qx, qy, qz, qw]` |
| `node` | One per node, in order. It holds the node id, the time, the local pose, and the live global estimate (local-to-global transform at that moment × local pose). It also holds up to 440 scan points in the **node** (tracking) frame: up to 360 below the ceiling and 80 on it. The node frame lets a renderer move the points when the node's pose is optimized. |
| `optimization` | One per global optimization the main thread observes. It holds the bag and wall time, the last optimized node, and the local-to-global transform. It also holds every node's global pose, every submap's global pose, the inter-submap constraints as `[node, submap, closes_loop]`, and the constraint counts. |
| `final` | The rigid alignments `world_from_map` and `world_from_local` (from the ATE evaluation) and the final metrics |

The callbacks only queue data. The optimization callback runs on a
background thread while Cartographer holds its locks. The pose graph is read
on the main thread between `push_*` calls. Recording does not change the
results: `metrics.json` (except timing), the trajectories, and `map.ply` are
byte-identical with and without `--record`. Two optimizations can land
between two reads. The final optimization of `finish()` is one example. A
snapshot then covers both, and its `index` is the optimization count.

The video has four panels:

- **Top-down view (left).** Each node's points, minus the ceiling and
  coloured by height, are drawn at the node's current best pose. That pose
  comes from the latest optimization snapshot if one covers the node,
  otherwise from the live estimate. Overlaid are the ground truth (green),
  the current trajectory estimate (red), local SLAM only (dashed blue), and
  the current scan (white). After each optimization the map and trajectory
  animate from the old poses to the new ones. The constraints flash from
  node to submap origin (loop closures in magenta, other inter-submap
  constraints in cyan), with a banner such as "global optimization #k — N
  loop closures".
- **3D view and zoom (right top).** A rotating 3D view of the map, next to a
  ±40 cm zoom around the robot. The drift and the corrections are a few
  centimeters, which cannot be seen at room scale.
- **Translation error against ground truth (right middle).** Three curves:
  local SLAM only, the live estimate when each scan arrived, and the whole
  trajectory as currently estimated. The last one is redrawn every frame:
  it climbs for nodes that no optimization covers yet and drops after each
  optimization.
- **Statistics (right bottom).** Sim time, nodes, submaps, optimizations,
  constraints, and the ATE so far.

All panels use the ground-truth frame through the fixed alignments in
`final`. Only the frame changes; each estimate is the one the system had at
that moment.

### Configuration

The configuration starts from Cartographer's shipped defaults
(`trajectory_builder_3d.lua`, `pose_graph.lua`). Each override is justified
in `unitree_l1_config()` in the example. There are three overrides:

| Override | Default → used | Why |
|---|---|---|
| `use_online_correlative_scan_matching` | false → **true** | **Poor motion prediction.** With no odometry, the 3D extrapolator predicts with constant velocity; the IMU only provides rotation. **Small matcher basin.** The L1's sparse, ceiling-dominated clouds give the Ceres matcher a small convergence basin, so it sometimes "sticks" when the robot accelerates from rest. Local SLAM then lags by meters, which loop closure cannot repair. **Gyro bias.** The correlative search also corrects the yaw drift that comes from the gyro bias. With the defaults (`rotation_weight = 400`), local SLAM almost follows the gyro: about 0.2 °/s of yaw drift on the noisy bag (16° after 76 s), local ATE 0.44 m. The cartographer_ros tuning guide gives this advice when the motion prior is poor. |
| `real_time_correlative_scan_matcher.angular_search_window` | 1° → **0.5°** | With the default window the yaw ran away by about 10° in the first 2 s, while the robot stood still (see below). With 0.5° it stays within about 1.5°. Here 0.5° is ±1 angular step of about 0.5°; the step depends on the scan range. The final accuracy is essentially unchanged (2.8 / 1.8 cm against 2.6 / 1.7 cm). |
| `submaps.num_range_data` | 160 → **120** | That is 12 s or about 13 m per submap. Loop closure only matches finished submaps, which need 2 × `num_range_data` insertions. With 120, the first submap is finished after 24 s, before the first lap (about 30 s) ends, and there are more, smaller submaps to close against. Optimized ATE is 1.7–2.9 cm instead of 2.6–3.6 cm; 160 also works. |

These knobs were considered and left at their defaults, with the evidence for
each:

- **`min_range` / `max_range` (1 / 60 m).** Setting them to the L1's limits
  (0.3 / 30 m) changed nothing: the results were bit-identical, because no
  returns in this room fall in between. On a real robot, keep `min_range`
  above the robot body.
- **`voxel_filter_size` (0.15 m).** 0.10, 0.15 and 0.20 all work. These three
  values were used as the robustness check:
  - optimized ATE noisy: 1.7 / 1.8 / 1.7 cm;
  - optimized ATE noise-free: 4.2 / 2.8 / 1.8 cm;
  - ATE max ≤ 15 cm in all six runs.
- **`num_accumulated_range_data` (1).** 2 was comparable. 3 and 4 were
  clearly worse: optimized ATE 4–9 cm, max up to 0.6 m.
- **Ceres `occupied_space_weight_0/1` (1 / 6), `translation_weight` /
  `rotation_weight` (5 / 400).** Changing these fixes a z bias (see below),
  but made the system fragile. In about a third of the weight combinations
  tried, local SLAM lost track by several meters at the start of motion, and
  neighbouring settings flipped between good and broken.
- **`imu_gravity_time_constant` (10 s), loop closure settings
  (`optimize_every_n_nodes` 90, `min_score` 0.55, `sampling_ratio` 0.3).**
  About 400 loop closures are found with the defaults. Nothing indicated a
  need to change them.
- **`pose_extrapolator.use_imu_based = true`.** This one is not usable here.
  It worked in one setting but diverged in others, and then hit a
  `CHECK`-failure in `HybridGrid` ("index out of the DynamicGrid range"),
  just like in C++.

For comparison, all on the noisy and noise-free bags:

| Configuration | Optimized ATE noisy / noise-free | Local ATE noisy / noise-free |
|---|---|---|
| Shipped defaults, no override | 7.9 / 3.4 cm | 44.3 / 14.0 cm |
| Defaults + correlative matching | 1.6 / 3.7 cm | 7.2 / 11.8 cm |
| **This demo** | **1.8 / 2.8 cm** | **7.2 / 18.7 cm** |

### Investigation: conventions and root causes

Before any tuning, the frame and time conventions were checked by breaking
each one on purpose. Each row changes one thing in the final configuration.
The correct conventions give the best optimized trajectory:

| Variant | Optimized ATE noisy / noise-free | Loop closures noisy |
|---|---|---|
| Correct (the demo) | 1.8 / 2.8 cm | 406 |
| Point times ignored (no de-skewing) | 3.7 / 3.7 cm | 340 |
| Point times reversed (stamp treated as the *last* point) | 2.8 / 4.0 cm (max 19 cm) | 250 |
| LiDAR extrinsics inverted | 3.9 / 4.7 cm | 409 |

The differences are small because the effects are small in this setup:
- motion distortion is at most about 0.1 m per cloud;
- the lever arm is 1.7 cm.

Gravity was checked as well:
- the IMU reads +9.81 m/s² on z at rest;
- the first scan puts the ceiling at +2.45 m in the local frame, which
  matches 3 m minus 0.55 m.

So the gravity sign and direction are right.

The problems were in how Cartographer's 3D local SLAM behaves with this
sensor. All of them reproduce the C++ algorithm; none is a bug in the port:

1. **Z bias from the low-resolution grid.**
   - **Cause.** The Ceres cost weighs the 0.45 m low-resolution grid 6× and
     the 0.10 m grid 1×. Interpolation in the hybrid grid peaks at voxel
     centers. A flat ceiling, which here is 70 % of the points, is pulled
     towards the center of its 0.45 m low-resolution voxel.
   - **Observed.** In the first submap, the ceiling at 2.45 m was matched at
     about 2.25 m. The estimate moved about 0.2–0.28 m down while the robot
     stood still.
   - **It compounds.** Each new submap has its own voxel grid, so this was a
     z step at submap changes. The noise-free local trajectory drifted
     −0.9 m in z over 90 s with the first trial configuration.
   - **Fix.** Raising the high-resolution weight (for example 6 / 1)
     removes the bias, but makes tracking fragile (see above), so the
     weights were left at default.
   - **What remains.** With the final configuration, the noisy bag keeps a
     roughly constant offset of −0.1 to −0.24 m. The noise-free bag still
     steps down, to −0.6 m after 75 s. Loop closure removes both: the
     optimized ATE stays at 1.8 / 2.8 cm.
2. **Yaw runaway on a sparse map ("spokes").**
   - **Cause.** Each cloud is 18 arcs through the zenith, 22° apart in
     azimuth, and the pattern precesses from cloud to cloud. A sparse map,
     such as the first submap while the robot stands still, is therefore a
     spoke pattern on the ceiling. Matchers that reward overlap rotate each
     new scan towards the previous spokes.
   - **Observed.** About 5 °/s of yaw while standing still. The noisy gyro's
     error is only about 0.1 °/s.
   - **Fix.** The correlative matcher went to its ±1° window edge on every
     scan. The ±0.5° window bounds it, and once the map is dense the effect
     disappears.
3. **Sticking at the start of motion.** The constant-velocity prediction is
   zero when the robot starts. The Ceres matcher, with no correlative search,
   sometimes failed to follow the 2 s cosine speed ramp. The estimate then
   lagged 0.7 to 6 m behind, depending on the configuration. Online
   correlative matching fixes it.
4. **Gyro bias.** With `rotation_weight = 400` the scan matcher fixes only a
   few percent of each scan's small rotation error. The per-scan error from
   gyro bias is about 0.03°, which is about 2 mm at 5 m, well under a 0.1 m
   voxel. Local SLAM therefore integrated the bias: 22° of yaw over 90 s on
   the noisy bag with the first trial configuration, and about 18° with the
   shipped defaults. Correlative matching fixes it: the final configuration
   stays within ±0.2° of yaw.
5. **Noise-free is not easier.**
   - **Observed.** In almost all runs with correlative matching, local SLAM
     on the noise-free bag is worse than on the noisy one: typically
     12–19 cm against 5–11 cm. The noise-free local trajectory also keeps
     the z steps of item 1.
   - **Likely explanation.** The room's planes are axis-aligned, and so is
     the map frame at the start. Without range noise, each plane falls
     exactly into one voxel layer, sometimes right on a voxel boundary: the
     ceiling at 2.45 m sits on the 0.1 m grid. That makes the quantization
     effects above deterministic and sharp. The 1 cm range noise dithers
     them.
6. **Local jitter.** With correlative matching, local poses show a
   few-centimeter, roughly 1 Hz jitter. The cause is the discrete search
   grid (0.1 m, about 0.5°). The pose graph smooths it out: optimized
   RPE (1 m) is 1.8 cm.

No change to the `cartographer` crate was needed.

### Limitations

- This is simulated data with one perfect clock and exact extrinsics.
  - Real L1 data adds timing jitter, calibration errors, and self- and
    mount-occlusion.
  - Real data also brings environments where the ceiling is out of range.
- The ATE alignment hides the offsets of local SLAM: 0.1 to 0.6 m in z, and
  0–3° of yaw from the start-up phase. "Final drift" (alignment at the first
  pose) shows them.
- The error and drift statistics use node poses only. Scans dropped by the
  motion filter are not evaluated.

## Promo media (map videos and summary image)

The clips and image at the top of the repository README are produced by:

```sh
# 2D: occupancy grid snapshots during a simulated office tour (needs ffmpeg).
cargo run --release -p cartographer-demos --example map_2d_video -- target/media/cartographer_rust_2d_map.mp4

# 3D: camera orbit around the Unitree L1 map; input is the --record log of unitree_l1_bag.
python crates/demos/scripts/render_map_3d_video.py target/unitree_l1/slam_run.jsonl \
    --out target/media/cartographer_rust_3d_map.mp4

# Summary image: 1080x1350 portrait, or 1920x1080 with --landscape (video thumbnail).
python crates/demos/scripts/linkedin_chart.py target/unitree_l1/slam_run.jsonl \
    --landscape --out target/media/cartographer_rust_thumbnail_1920x1080.png
```

The Python scripts need `numpy`, `matplotlib` and `Pillow` (the venv of the
`unitree-l1-sim` tools works). The README copies in `docs/media/` are
downscaled: MP4s re-encoded at 720p, GIF previews at 640–800 px.
