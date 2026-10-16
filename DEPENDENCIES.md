# Dependencies

Policy (in order): Rust standard library → a well-maintained **pure-Rust**
crate → a focused port of the upstream library's own source code. No crate in
the default build compiles C/C++ code or links system libraries, so
`cargo build` works on any Rust tier-1 target without extra toolchains.

Versions were checked on crates.io on 2026-10-16.

## C++ dependency → Rust replacement

| C++ dependency | Used for in Cartographer | Rust replacement | Why |
|---|---|---|---|
| Eigen 3 | vectors, matrices, quaternions, transforms | [`nalgebra`] 0.35 | The most widely used pure-Rust linear algebra crate, actively maintained. It has fixed-size `Vector3`/`Matrix3`/`UnitQuaternion` matching Eigen's fixed-size types, and its generic `RealField` scalar lets the same code run on `f32`, `f64` and Ceres-style `Jet`s (like Eigen templates do). |
| Ceres Solver | nonlinear least squares: Ceres scan matchers (2D/3D), pose graph SPA, landmarks, IMU terms; `jet.h`, `rotation.h`, `cubic_interpolation.h` | **`cartographer-ceres`** (in this workspace: a focused port of Ceres 2.0.0's own source) | We evaluated `tiny-solver` 0.18, `factrs` 0.3, `argmin` and `levenberg-marquardt`. None reproduces Ceres semantics closely enough for a faithful port: local parameterizations with constant blocks, `DYNAMIC` autodiff residual counts, Ceres' LM trust-region strategy and step evaluator (including non-monotonic steps), the loss-function corrector, and `BiCubicInterpolator` over Jets. They also use dynamically sized duals (`num-dual` `DualDVec`), which allocate in the scan-matching hot path. The port is about the size of the subset Cartographer actually calls. See `crates/ceres/DESIGN.md`. |
| Ceres sparse backend (SuiteSparse/CHOLMOD or Eigen sparse) | `SPARSE_NORMAL_CHOLESKY` for the pose graph | see `crates/ceres/DESIGN.md` | Chosen by the solver port (pure Rust). |
| Protocol Buffers (libprotobuf + `protoc`) | option protos, `.pbstream` serialization | [`prost`] 0.14 + [`prost-build`] 0.14 + [`protox`] 0.10 | `prost` is the standard pure-Rust protobuf runtime and is wire-compatible with libprotobuf. `protox` is a pure-Rust `.proto` compiler, so the build needs no `protoc` binary. The `.proto` files are byte-identical copies of the C++ ones; a test enforces this. |
| Lua 5.2 (C library) | `.lua` configuration files, `LuaParameterDictionary` | **dedicated pure-Rust Lua 5.2 subset interpreter** (`cartographer::common::lua`) | `mlua` with the `vendored` feature compiles C Lua, which the policy rules out. `piccolo` 0.3 (pure Rust) is experimental, and its stackless, GC-arena API makes the dictionary semantics awkward to build on. Configs only need a small, well-defined subset of Lua: tables, locals, `include`, arithmetic, `math.*`, simple control flow and functions. A dedicated interpreter lets us match Lua 5.2 number formatting and the C++ reference-counting semantics exactly. |
| glog | logging, `CHECK` | [`log`] 0.4 + `assert!`/`panic!` | `log` is the standard logging facade. Binaries and examples install a simple logger. |
| gflags | flags of `*_main.cc` tools | [`clap`] 4 | The de-facto standard CLI parser. |
| Boost iostreams + zlib | gzip of `.pbstream` chunks | [`flate2`] 1.1 with the `rust_backend` (`miniz_oxide`) | Pure-Rust DEFLATE. Its output is standard gzip, which C++ reads. |
| absl (Mutex, time, strings, containers, `make_unique`, `optional`) | utilities | `std` | `std::sync::{Mutex, Condvar}`, `Option`, `HashMap`/`BTreeMap`, and `String` utilities cover all uses. |
| Cairo | submap painting, trajectory drawing, X-ray/probability-grid PNGs | TBD (pure Rust) | Chosen when `io/` is ported. |
| gRPC / async_grpc | `cloud/` map builder server | `tonic` (optional, last) | Chosen when `cloud/` is ported. |
| gtest / gmock | tests | built-in `#[test]`, [`approx`], [`proptest`] | Hand-written fakes replace gmock mocks. |

## Direct dependencies

| Crate | Version | Scope | Pure Rust |
|---|---|---|---|
| nalgebra | 0.35 | runtime | yes |
| prost | 0.14 | runtime | yes |
| flate2 (`rust_backend`) | 1.1 | runtime | yes (miniz_oxide) |
| log | 0.4 | runtime | yes |
| prost-build | 0.14 | build | yes (we never invoke `protoc`; descriptors come from protox) |
| protox | 0.10 | build | yes |
| approx | 0.5 | dev | yes |
| proptest | 1 | dev | yes |
| rand | 0.9 | dev | yes |

[`nalgebra`]: https://crates.io/crates/nalgebra
[`prost`]: https://crates.io/crates/prost
[`prost-build`]: https://crates.io/crates/prost-build
[`protox`]: https://crates.io/crates/protox
[`log`]: https://crates.io/crates/log
[`clap`]: https://crates.io/crates/clap
[`flate2`]: https://crates.io/crates/flate2
[`approx`]: https://crates.io/crates/approx
[`proptest`]: https://crates.io/crates/proptest
