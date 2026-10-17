# Releasing to crates.io

Publishing is irreversible (versions can only be yanked, never deleted)
and is done by the maintainer, not by automation.

## Crates

Published, in dependency order:

| Package              | Library (`use ...`)  | Depends on (published)                         |
|----------------------|----------------------|------------------------------------------------|
| `cartographer-ceres` | `ceres`              | –                                              |
| `cartographer-sim`   | `cartographer_sim`   | – (dev-dependency of `cartographer-slam`)      |
| `cartographer-slam`  | `cartographer`       | `cartographer-ceres`; dev: `cartographer-sim`  |
| `cartographer-cloud` | `cartographer_cloud` | `cartographer-slam`                            |

Not published (`publish = false`): `cartographer-rosbag`, `cartographer-demos`,
`xtask`.

The main package is named `cartographer-slam` because `cartographer` is
taken on crates.io (0.0.1, an unrelated "namespace reservation", April
2026); its library keeps the name `cartographer`, so user code writes
`use cartographer::...`. Dependents use
`cartographer = { package = "cartographer-slam", version = "0.1" }`.

Name check (2026-10-16): `cartographer-slam`, `cartographer-ceres`,
`cartographer-cloud`, `cartographer-sim` were available; `cartographer` and
`cartographer-rs` are taken.

## Steps

1. Make sure the tree is clean and green:

   ```sh
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   for c in cartographer-ceres cartographer-sim cartographer-slam cartographer-cloud; do
     RUSTDOCFLAGS=-D warnings cargo doc --no-deps -p $c
   done
   ```

2. Build all four packages from their tarballs (cargo resolves the
   inter-crate dependencies against the other freshly packaged tarballs):

   ```sh
   cargo package -p cartographer-ceres -p cartographer-sim -p cartographer-slam -p cartographer-cloud
   ```

3. Dry-run, then publish, in order (recent cargo waits for each crate to
   appear in the index before returning):

   ```sh
   cargo publish --dry-run -p cartographer-ceres && cargo publish -p cartographer-ceres
   cargo publish --dry-run -p cartographer-sim   && cargo publish -p cartographer-sim
   cargo publish --dry-run -p cartographer-slam  && cargo publish -p cartographer-slam
   cargo publish --dry-run -p cartographer-cloud && cargo publish -p cartographer-cloud
   ```

   `cargo publish --dry-run` of a crate fails until its published
   dependencies are actually on crates.io; step 2 is the pre-publish check
   for those.

## Versions

All crates share `[workspace.package] version`. Bump it together; the
inter-crate `version = "..."` requirements must be bumped with it. Use semver:
breaking API changes bump the minor version while at 0.x.

## What is in each package

`include` in each `Cargo.toml` is an allowlist:

- `cartographer-ceres`: `src/`, `tests/`, `README.md`, `DESIGN.md`,
  `LICENSE`, `LICENSE-CERES`, `NOTICE`.
- `cartographer-sim`: `src/`, `examples/`, `README.md`, `LICENSE`, `NOTICE`.
- `cartographer-slam`: `src/`, `tests/`, `examples/`, `benches/`, `proto/`
  (compiled by `build.rs` with the pure-Rust `protox`, no `protoc`),
  `configuration_files/*.lua`, `build.rs`, `README.md`, `LICENSE`, `NOTICE`.
  Tests that compare against the C++ source tree skip when it is absent.
- `cartographer-cloud`: `src/`, `proto/`, `build.rs`, `README.md`,
  `LICENSE`, `NOTICE`.

All are far below the 10 MB crates.io limit (largest: `cartographer-slam`,
about 0.7 MB compressed).

## Build-time paths

- `cartographer-slam` declares `links = "cartographer"` and its build script
  emits `cargo::metadata=proto_dir=<abs path of its proto/>`.
  `cartographer-cloud`'s build script reads it as
  `DEP_CARTOGRAPHER_PROTO_DIR`; the in-repo `../cartographer/proto` is only a
  fallback.
- `common::config::CONFIGURATION_FILES_DIRECTORY` is
  `$CARGO_MANIFEST_DIR/configuration_files`, which for a registry build is
  the unpacked source in `~/.cargo/registry/src/...` (it is shipped). The
  high-level `SlamConfig::from_lua` additionally falls back to compiled-in
  copies of the stock Lua files.

## Licensing

Apache-2.0 (`LICENSE`, copied from upstream Cartographer) with a `NOTICE`
crediting the Cartographer authors (Google Inc. and other contributors).
`cartographer-ceres` is `Apache-2.0 AND BSD-3-Clause`: its sources are ported
from Ceres Solver (New BSD), whose license text is in
`crates/ceres/LICENSE-CERES`. `cartographer-sim` is original code
(Apache-2.0).

## Minimum Rust version

`rust-version` is 1.89 (required by `nalgebra` 0.35). Check it before each
release:

```sh
rustup toolchain install 1.89 --profile minimal
CARGO_TARGET_DIR=target/msrv cargo +1.89 check --all-features --all-targets \
  -p cartographer-ceres -p cartographer-sim -p cartographer-slam -p cartographer-cloud
```
