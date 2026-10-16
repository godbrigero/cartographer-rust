# cartographer

A native, pure-Rust port of [Google Cartographer], the real-time 2D and 3D
SLAM library (lidar, IMU, odometry, fixed-frame poses and landmarks). No C/C++
dependencies, no `protoc` (protos are compiled with the pure-Rust `protox`),
no Lua interpreter to install.

The crate has two layers:

- **High-level API** (`Slam`, `SlamBuilder`, `SlamConfig`, ...): declare
  sensors, push timestamped data, get optimized poses, occupancy grids
  (PNG / ROS map), 3D point maps (PLY) and `.pbstream` files. Typed
  configuration with Cartographer's defaults.
- **Faithful port** (`mapping`, `sensor`, `transform`, `io`, `common`, ...):
  Cartographer's C++ API one-to-one (`MapBuilder`, option protos, Lua
  configuration dictionaries), wire-compatible `.pbstream` serialization.

```rust,no_run
use cartographer::{Extrinsics, Mode, Slam, SlamError};

fn main() -> Result<(), SlamError> {
    let mut slam = Slam::builder(Mode::TwoD)
        .lidar("scan", Extrinsics::from_translation(0.1, 0.0, 0.2))
        .imu("imu", Extrinsics::identity())
        .config(|c| c.max_range = 20.0)
        .build()?;
    // For each sensor message (times in seconds):
    // slam.push_imu(t, imu)?;
    // slam.push_lidar("scan", t, &points)?;
    slam.finish()?;
    slam.map_2d()?.to_png("map.png")?;
    slam.save("map.pbstream")?;
    Ok(())
}
```

Cartographer's stock Lua configuration files ship in `configuration_files/`
and are also compiled into the library, so `SlamConfig::from_lua` works
without any files on disk.

Related crates: [`cartographer-ceres`](https://crates.io/crates/cartographer-ceres)
(the optimization backend) and
[`cartographer-cloud`](https://crates.io/crates/cartographer-cloud) (gRPC
map builder server and client).

## License

Apache License 2.0, like Cartographer (see `LICENSE` and `NOTICE`). This is
an independent port and is not affiliated with or endorsed by Google or the
Cartographer project.

[Google Cartographer]: https://github.com/cartographer-project/cartographer
