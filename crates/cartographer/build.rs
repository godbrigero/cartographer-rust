//! Compiles the Cartographer `.proto` schemas (copied byte-for-byte from the
//! C++ tree into `proto/`) with `protox` (a pure-Rust protobuf compiler, so no
//! `protoc` binary is needed) and generates Rust types with `prost-build`.

use std::path::{Path, PathBuf};

fn collect_protos(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_protos(&path, out);
        } else if path.extension().is_some_and(|e| e == "proto") {
            out.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("proto");
    let mut protos = Vec::new();
    collect_protos(&root, &mut protos);
    println!("cargo:rerun-if-changed={}", root.display());
    // `links = "cartographer"`: dependents' build scripts (cartographer-cloud)
    // see this as `DEP_CARTOGRAPHER_PROTO_DIR`, so they can import these
    // protos from a registry checkout too.
    println!("cargo::metadata=proto_dir={}", root.display());
    let file_descriptors =
        protox::compile(&protos, [&root]).expect("failed to compile Cartographer protos");
    let mut config = prost_build::Config::new();
    config.include_file("protos.rs");
    config
        .compile_fds(file_descriptors)
        .expect("failed to generate Rust code for Cartographer protos");
}
