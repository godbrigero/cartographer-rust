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
| Ceres sparse backend (SuiteSparse/CHOLMOD or Eigen sparse) | `SPARSE_NORMAL_CHOLESKY` for the pose graph | [`faer`] 0.24 (`sparse-linalg`, no rayon) | Pure-Rust, actively maintained; supernodal sparse Cholesky with AMD fill-reducing ordering and reusable symbolic analysis. A 16,000-pose 2D SPA problem solves 5 LM iterations in ~0.13 s. Rationale in `crates/ceres/DESIGN.md`. |
| Protocol Buffers (libprotobuf + `protoc`) | option protos, `.pbstream` serialization | [`prost`] 0.14 + [`prost-build`] 0.14 + [`protox`] 0.10 | `prost` is the standard pure-Rust protobuf runtime and is wire-compatible with libprotobuf. `protox` is a pure-Rust `.proto` compiler, so the build needs no `protoc` binary. The `.proto` files are byte-identical copies of the C++ ones; a test enforces this. |
| Lua 5.2 (C library) | `.lua` configuration files, `LuaParameterDictionary` | **dedicated pure-Rust Lua 5.2 subset interpreter** (`cartographer::common::lua`) | `mlua` with the `vendored` feature compiles C Lua, which the policy rules out. `piccolo` 0.3 (pure Rust) is experimental, and its stackless, GC-arena API makes the dictionary semantics awkward to build on. Configs only need a small, well-defined subset of Lua: tables, locals, `include`, arithmetic, `math.*`, simple control flow and functions. A dedicated interpreter lets us match Lua 5.2 number formatting and the C++ reference-counting semantics exactly. |
| glog | logging, `CHECK` | [`log`] 0.4 + `assert!`/`panic!` | `log` is the standard logging facade. Binaries and examples install a simple logger. |
| gflags | flags of `*_main.cc` tools | [`clap`] 4 | The de-facto standard CLI parser. |
| Boost iostreams + zlib | gzip of `.pbstream` chunks | [`flate2`] 1.1 with the `rust_backend` (`miniz_oxide`) | Pure-Rust DEFLATE. Its output is standard gzip, which C++ reads. |
| absl (Mutex, time, strings, containers, `make_unique`, `optional`) | utilities | `std` | `std::sync::{Mutex, Condvar}`, `Option`, `HashMap`/`BTreeMap`, and `String` utilities cover all uses. |
| Cairo | submap painting, trajectory drawing, X-ray/probability-grid PNGs | `cartographer::io::cairo` (own pure-Rust code on raw ARGB32 buffers) + [`png`] 0.18 | See [Cairo](#cairo) below. |
| gRPC / async_grpc | `cloud/` map builder server, client stubs, uploader | [`tonic`] 0.14 (`default-features = false`, features `transport`, `codegen`, `router`) + [`tonic-prost`] 0.14 + [`tokio`] 1 (`rt-multi-thread`, `net`, `sync`, `time`, `macros`, `io-util`) + [`tokio-stream`] 0.1; codegen with [`tonic-prost-build`] 0.14 (`default-features = false`, `transport`) from protox file descriptors | Only in the separate `cartographer-cloud` crate. tonic is the standard pure-Rust gRPC stack (hyper/h2); it is wire-compatible with C++ gRPC (same `/cartographer.cloud.proto.MapBuilderService/<Rpc>` paths). **No TLS feature is enabled** (`tls-ring`/`tls-aws-lc` would pull C/assembly via ring or aws-lc), so channels are plaintext; `enable_ssl_encryption`/`enable_google_auth` are rejected with a clear panic. `cargo tree -p cartographer-cloud -e normal,build` contains no `cc`, `cmake`, `bindgen`, `pkg-config`, `ring`, `aws-lc*` or `openssl*` (only `libc`, Rust FFI declarations without C compilation, as used by tokio/mio). The `async_grpc` handler/retry framework is reimplemented in `internal/async_grpc.rs`. |
| prometheus-cpp | `cloud/metrics/prometheus`, `/metrics` exposer of the gRPC server | [`prometheus-client`] 0.25 (default features, i.e. no protobuf) behind the default-on `prometheus` feature of `cartographer-cloud` | The official pure-Rust Prometheus client; its `Family<LabelSet, M>` supports arbitrary label sets per family like prometheus-cpp (the tikv `prometheus` crate needs label names up front). The HTTP exposer is a ~60-line tokio responder instead of an HTTP framework. |
| gtest / gmock | tests | built-in `#[test]`, [`approx`], [`proptest`] | Hand-written fakes replace gmock mocks. |

## Cairo

Cartographer uses a small, fixed subset of Cairo, always on
`CAIRO_FORMAT_ARGB32` image surfaces: create surfaces (also over existing
pixel data), `paint` with a solid color or with another surface under an
affine transform (`scale`/`translate`/`transform`, `save`/`restore`,
`user_to_device`), stroke a 4 px polyline, fill two small circles, and write a
PNG.

**Decision: re-implement these operations in `crates/cartographer/src/io/cairo.rs`
on raw `u32` ARGB32 buffers, and encode PNGs with the pure-Rust [`png`]
crate.** No C dependency (`cairo-rs` links the system library, which the
policy rules out).

Why not [`tiny-skia`] (0.11, pure Rust)? It covers paths, strokes and
bilinear pattern drawing, but:

* `submap_painter`'s `DrawTexture` stores pixels that are *not* valid
  premultiplied colors (red = intensity can exceed alpha). Cairo/pixman store
  and composite them verbatim (with saturating adds), and downstream tools
  decode the red/green/alpha channels directly. tiny-skia's
  `PremultipliedColorU8` requires `c <= a`, so the data would be clamped or
  rejected.
* Matching Cairo's pixel math matters more than general 2D drawing. The custom
  code reproduces exactly: Cairo's matrix semantics, color conversion
  (`double` -> 16 bit -> 8 bit), pixman's `OVER` (`MUL_UN8` rounding,
  saturating add), pixman's bilinear filter (7-bit weights, 16.16 fixed point
  sample positions, `EXTEND_NONE`), Cairo's arc-to-Bezier construction and
  spline flattening (tolerance 0.1, 24.8 fixed point), and the PNG writer's
  un-premultiplication (including the 8-bit wrap-around for invalid pixels)
  and RGB-vs-RGBA choice.
* Only anti-aliased edge coverage is approximated (15 sub-scanlines x 1/256
  px like Cairo's tor scan converter, but not its exact edge stepping), so
  stroke/arc edge pixels may differ from Cairo by a few alpha levels.
  Cairo >= 1.14's box filter for down-scaled surface painting is not
  reproduced (bilinear is used; exact when submaps are painted at their own
  resolution, the common case).
* It is ~1600 lines including tests, less than wiring a general renderer
  and then working around its premultiplication invariants.

[`png`] (pure Rust, depends on `flate2` with `miniz_oxide`, `fdeflate`,
`crc32fast`) is the standard PNG codec of the Rust ecosystem; PNG bytes are
not byte-identical to libpng's output (different deflate), but decode to the
same pixels, and we write the same `bKGD` chunk as Cairo.

[`tiny-skia`]: https://crates.io/crates/tiny-skia

## Direct dependencies

| Crate | Version | Scope | Pure Rust |
|---|---|---|---|
| nalgebra | 0.35 | runtime | yes |
| prost | 0.14 | runtime | yes |
| flate2 (`rust_backend`) | 1.1 | runtime | yes (miniz_oxide) |
| log | 0.4 | runtime | yes |
| png | 0.18 | runtime | yes (flate2/miniz_oxide, fdeflate, crc32fast) |
| prost-build | 0.14 | build | yes (we never invoke `protoc`; descriptors come from protox) |
| protox | 0.10 | build | yes |
| approx | 0.5 | dev | yes |
| proptest | 1 | dev | yes |
| rand | 0.9 | dev | yes |
| tonic (no TLS) | 0.14 | runtime, `cartographer-cloud` only | yes (hyper, h2, tower, axum router) |
| tonic-prost | 0.14 | runtime, `cartographer-cloud` only | yes |
| tokio | 1 | runtime, `cartographer-cloud` only | yes (mio; `libc`/`windows-sys` bindings only) |
| tokio-stream | 0.1 | runtime, `cartographer-cloud` only | yes |
| prometheus-client | 0.25 | runtime, `cartographer-cloud` feature `prometheus` (default) | yes |
| tonic-prost-build | 0.14 | build, `cartographer-cloud` only | yes (fed by protox; never invokes `protoc`) |
| faer (`std`, `sparse-linalg`) | 0.24 | runtime, `cartographer-ceres` | yes |
| simba / num-traits | 0.10 / 0.2 | runtime, `cartographer-ceres` (implementing nalgebra's `RealField` for `Jet`) | yes |
| libm | 0.2 | runtime, `cartographer-ceres` (`erf`, Bessel functions of `jet.h`) | yes |
| clap | 4 | runtime (binaries) | yes |
| cpu-time | 1 | runtime (`common::time::get_thread_cpu_time_seconds`, avoids our own `unsafe` FFI) | yes (OS API bindings only) |
| num-traits | 0.2 | runtime | yes |
| criterion | 0.8 | dev (benches) | yes |
| rand_chacha / rand_distr | 0.9 / 0.5 | `cartographer-sim` | yes |
| lz4_flex | 0.14 | `cartographer-rosbag` (lz4 bag chunks) | yes (`safe-decode`) |
| bzip2-rs | 0.1 | `cartographer-rosbag`, feature `bz2` | yes (the `rosbag` crate on crates.io was rejected: it always builds `bzip2-sys`/`lz4-sys` C code) |

## `unsafe` policy

The workspace denies `unsafe_code`. The single exception is in
`crates/ceres/src/jet.rs`: nalgebra's `RealField` requires simba's
`SimdValue` trait, which *declares* two `unsafe fn` methods
(`extract_unchecked`, `replace_unchecked`). Implementing them for `Jet`
needs a narrow `#[allow(unsafe_code)]`; their bodies contain no unsafe
operations.

[`faer`]: https://crates.io/crates/faer

[`tonic`]: https://crates.io/crates/tonic
[`tonic-prost`]: https://crates.io/crates/tonic-prost
[`tonic-prost-build`]: https://crates.io/crates/tonic-prost-build
[`tokio`]: https://crates.io/crates/tokio
[`tokio-stream`]: https://crates.io/crates/tokio-stream
[`prometheus-client`]: https://crates.io/crates/prometheus-client

[`nalgebra`]: https://crates.io/crates/nalgebra
[`prost`]: https://crates.io/crates/prost
[`prost-build`]: https://crates.io/crates/prost-build
[`protox`]: https://crates.io/crates/protox
[`log`]: https://crates.io/crates/log
[`clap`]: https://crates.io/crates/clap
[`flate2`]: https://crates.io/crates/flate2
[`png`]: https://crates.io/crates/png
[`approx`]: https://crates.io/crates/approx
[`proptest`]: https://crates.io/crates/proptest
