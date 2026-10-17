# Visual report

`index.html` is a static, self-contained report on the Rust port:
architecture diagrams, porting progress, 2D/3D end-to-end SLAM results
against simulated ground truth, the Unitree L1 MuJoCo demo, criterion
benchmarks and the build environment. Open it in a browser straight from
disk; it needs no server. Every chart is inline SVG and is also saved as a
standalone file in `assets/`. The only network request loads Mermaid from a
CDN to redraw the two architecture diagrams. Offline, the report shows its
own pre-rendered layout of the same diagrams, and the Mermaid source is
included under each one.

## Regenerate

From the workspace root:

```sh
cargo run --release -p xtask -- report
```

This runs the e2e acceptance tests
(`cargo test -p cartographer-slam --test e2e_2d --test e2e_3d`), which write
`target/e2e/<scenario>.json` and `<scenario>_map.png`. It reuses criterion
results already in `target/criterion`, computes the porting status and
rewrites `report/` (`index.html` and `assets/`).

| Flag | Effect |
|---|---|
| `--bench` | also runs the four criterion benches first (~1.5 min) |
| `--bench-quick` | same with criterion's `--quick` mode (seconds; noisier) |
| `--unitree-bag <bag>` | also runs `cartographer-demos`' `unitree_l1_bag` example on the bag (outputs in `target/unitree_l1/`) |
| `--unitree-dir <dir>` | reads the demo outputs from `<dir>` instead of `target/unitree_l1/` |
| `--no-tests` | reuses `target/e2e` as is |
| `--cpp <dir>` | C++ `cartographer/cartographer` tree for the porting status. By default it is looked up next to the workspace (or next to the main checkout for a git worktree); if it is missing, `PORTING_STATUS.md` is parsed instead |
| `--out <dir>` | output directory (default `report/`) |

The report honors `CARGO_TARGET_DIR`. A section whose inputs are missing
says so and names the flag that produces them.

## Contents

1. **Overview**: headline numbers and the module and data-flow diagrams
   from `docs/architecture/OVERVIEW.md`.
2. **Porting progress**: C++ units and C++ tests ported, per top-level
   module, plus the Rust test count.
3. **2D results**: for each scenario, the occupancy map, the estimated vs
   ground-truth trajectory and the pose graph with its inter-submap and
   loop-closure constraints. Also ATE and RPE over time, and every
   pass/fail check against its threshold.
4. **3D results**: for each scenario, the occupied voxels projected
   top-down and from the side with both trajectories, the height over time,
   the pose graph, error over time and the checks.
5. **Unitree L1 demo**: the demo's plots, metrics (global vs local SLAM)
   and video, if `target/unitree_l1/**` exists.
6. **Benchmarks**: criterion medians with 95 % CIs on a log scale, compared
   with the medians recorded in `docs/BENCHMARKS.md`.
7. **Environment**: rustc, target, commit, date and crate versions.

The e2e JSON files hold everything the report draws: per-pose errors
(`errors_over_time`), the optimized pose graph (`pose_graph`), the
inter-submap constraints with their kind, and, for 3D, a subsampled set of
occupied voxels (`map_points`). The tests write all of this from
`crates/cartographer/tests/common/mod.rs`.

## Notes

- The simulated datasets use fixed seeds, so the trajectory metrics are
  reproducible on one platform. Wall-clock times and benchmark numbers
  depend on the machine and its load.
- Videos (`assets/*.mp4`) are git-ignored to keep the repository small.
  They are copied in when you regenerate the report, if smaller than 50 MB.
  Larger videos are linked by path instead.
- The light/dark theme follows the system setting. The button in the
  header overrides it.
