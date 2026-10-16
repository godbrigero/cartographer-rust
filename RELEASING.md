# Releasing to crates.io

Three crates are published, in dependency order:

| Package              | Library name         | Depends on                     |
|----------------------|----------------------|--------------------------------|
| `cartographer-ceres` | `ceres`              | -                              |
| `cartographer`       | `cartographer`       | `cartographer-ceres`           |
| `cartographer-cloud` | `cartographer_cloud` | `cartographer`                 |

Not published (`publish = false`): `cartographer-sim`, `cartographer-rosbag`,
`xtask` (and the demos). `cartographer-sim` is a path-only dev-dependency of
`cartographer`; `cargo package` strips it from the published manifest.

## Blocker: the `cartographer` name is taken

Checked 2026-10-16 (`cargo search`, `https://crates.io/api/v1/crates/<name>`):

| Name                   | Status                                                        |
|------------------------|---------------------------------------------------------------|
| `cartographer`         | **taken** (0.0.1, "Provenance-first dataset platform (namespace reservation)", April 2026) |
| `cartographer-ceres`   | available                                                     |
| `cartographer-cloud`   | available                                                     |
| `cartographer-rs`      | **taken** (unrelated TUI menu crate)                          |
| `cartographer-slam`    | available                                                     |
| `cartographer-core`    | available                                                     |
| `cartographer-sim`     | available                                                     |
| `cartographer-rosbag`  | available                                                     |

Before publishing, either rename the `cartographer` *package* (e.g. to
`cartographer-slam`, keeping `[lib] name = "cartographer"` so `use
cartographer::...` keeps working; dependents then write `cartographer = {
package = "cartographer-slam", ... }`, like `ceres` does today), or ask the
owner of the `cartographer` crate to transfer it. Renaming touches
`crates/cartographer/Cargo.toml`, the `cartographer` dependency of
`cartographer-cloud`, the `links`/`DEP_CARTOGRAPHER_PROTO_DIR` pair (the
`links` value can stay `"cartographer"`), docs.rs URLs and the READMEs.

## Checklist

1. All of these pass on a clean tree:

   ```sh
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo build --workspace --all-features
   cargo test --workspace
   RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features \
       -p cartographer-ceres -p cartographer -p cartographer-cloud
   ```

2. Package and verify all three crates together. Cargo puts the earlier
   packages in a temporary local registry, so each tarball is built standalone
   against the *packaged* versions of its siblings (not the source tree):

   ```sh
   cargo package -p cartographer-ceres -p cartographer -p cartographer-cloud
   ls -l target/package/*.crate
   ```

   The `warning: ignoring test/benchmark ...` lines are expected (tests,
   examples and benches of `cartographer` are not shipped, see below).

3. Dry-run, then publish, in order. Wait until each crate is visible in the
   index before publishing the next one (recent cargo waits automatically):

   ```sh
   cargo publish --dry-run -p cartographer-ceres
   cargo publish -p cartographer-ceres

   cargo publish --dry-run -p cartographer
   cargo publish -p cartographer

   cargo publish --dry-run -p cartographer-cloud
   cargo publish -p cartographer-cloud
   ```

   (`cargo publish --dry-run -p cartographer` fails until
   `cartographer-ceres` is on crates.io, because a single-package dry run
   resolves dependencies from the registry; step 2 is the pre-publish check
   for it. With cargo >= 1.90, `cargo publish --workspace` publishes all three
   in order in one command.)

4. Tag the release: `git tag v0.1.0`.

## Version policy

- All crates share `workspace.package.version` and are released together.
- Semver, pre-1.0: bump the minor version (`0.1.x` -> `0.2.0`) for any
  breaking API change in any crate, the patch version otherwise.
- When bumping, also update the `version = "..."` of the internal
  dependencies (`cartographer` -> `cartographer-ceres`,
  `cartographer-cloud` -> `cartographer`).
- `rust-version` (MSRV) changes are minor bumps.

## What is in each package

`include` in each `Cargo.toml` is an allowlist:

- `cartographer-ceres`: `src/`, `tests/` (self-contained), `README.md`,
  `DESIGN.md`, `LICENSE`, `LICENSE-CERES`, `NOTICE`.
  40 files, 738 KiB (164 KiB compressed).
- `cartographer`: `src/`, `proto/` (compiled by `build.rs` with the
  pure-Rust `protox`, no `protoc`), `configuration_files/*.lua`, `build.rs`,
  `README.md`, `LICENSE`, `NOTICE`.
  282 files, 2.8 MiB (598 KiB compressed).
  Tests, examples and benches are **not** shipped: they use the unpublished
  `cartographer-sim`. Some unit tests inside `src/` also use it, so
  `cargo test` on the registry tarball does not compile; the library, the
  binaries and the docs do.
- `cartographer-cloud`: `src/`, `proto/`, `build.rs`, `README.md`,
  `LICENSE`, `NOTICE`. 67 files, 375 KiB (80 KiB compressed).

All are far below the 10 MB crates.io limit.

## Build-time paths

- `cartographer` declares `links = "cartographer"` and its build script emits
  `cargo::metadata=proto_dir=<abs path of its proto/>`.
  `cartographer-cloud`'s build script reads it as
  `DEP_CARTOGRAPHER_PROTO_DIR` to resolve the imported core protos; the
  in-repo `../cartographer/proto` is only a fallback.
- `common::config::CONFIGURATION_FILES_DIRECTORY` is
  `$CARGO_MANIFEST_DIR/configuration_files`, which for a registry build is
  the unpacked source in `~/.cargo/registry/src/...` (it is shipped). The
  high-level `SlamConfig::from_lua` additionally falls back to compiled-in
  copies of the stock Lua files when that directory is gone.

## Licensing

Apache-2.0 (`LICENSE`, copied from upstream Cartographer) with a `NOTICE`
crediting the Cartographer authors (Google Inc. and other contributors).
`cartographer-ceres` is `Apache-2.0 AND BSD-3-Clause`: its sources are ported
from Ceres Solver (New BSD, Google Inc.), whose license text is in
`crates/ceres/LICENSE-CERES`.

`repository` is intentionally unset (the workspace used to point at the
upstream C++ repository, which is not where this code lives). Add
`repository = "..."` to `[workspace.package]` (and `repository.workspace =
true` to the crates) once the Rust repository is public.
