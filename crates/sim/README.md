# cartographer-sim

Deterministic synthetic-data simulator for testing SLAM systems, written for
the [`cartographer-slam`](https://crates.io/crates/cartographer-slam) Rust
port of Google Cartographer but independent of it (plain `nalgebra` data).

* **Worlds:** 2D line-segment worlds (room, loop corridor, office) and 3D
  box/plane worlds, with ray casting and ground-truth occupancy rasters.
* **Trajectories:** smooth spline trajectories with analytic velocity,
  acceleration and angular velocity, optional handheld motion, and an
  optional smooth start from rest.
* **Sensors:** 2D/3D LiDAR (with per-point timing / motion distortion),
  IMU (specific force, noise, bias), odometry with drift, GPS-like
  fixed-frame poses and landmarks, each with configurable extrinsics.
* **Metrics:** Umeyama/Horn alignment, ATE, RPE and occupancy-map
  precision/recall.

Same seed, same data.

```rust
use cartographer_sim::DatasetBuilder;

let dataset = DatasetBuilder::corridor_2d(1.0).seed(7).build();
println!("{} messages", dataset.messages.len());
```

License: Apache-2.0.
