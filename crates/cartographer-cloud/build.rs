//! Compiles the `cartographer/cloud/proto/*.proto` schemas (copied
//! byte-for-byte from the C++ tree into `proto/`) with `protox` (a pure-Rust
//! protobuf compiler, so no `protoc` binary is needed) and generates the
//! messages and the `MapBuilderService` gRPC client/server with
//! `tonic-prost-build`.
//!
//! The imported `cartographer.{common,mapping,sensor,transform}` packages are
//! resolved from the core crate's `proto/` directory and mapped onto the
//! types already generated in `cartographer::protos`, so both crates share one
//! set of message types. `google.protobuf.Empty` maps to `()`.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let cloud_proto_root = manifest_dir.join("proto");
    let core_proto_root = manifest_dir.join("../cartographer/proto");
    let protos = [
        cloud_proto_root.join("cartographer/cloud/proto/map_builder_server_options.proto"),
        cloud_proto_root.join("cartographer/cloud/proto/map_builder_service.proto"),
    ];
    println!("cargo:rerun-if-changed={}", cloud_proto_root.display());
    println!("cargo:rerun-if-changed={}", core_proto_root.display());

    let file_descriptors = protox::compile(&protos, [&cloud_proto_root, &core_proto_root])
        .expect("failed to compile Cartographer cloud protos");

    let core = "::cartographer::protos::cartographer";
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .btree_map(".")
        .extern_path(".cartographer.common", format!("{core}::common"))
        .extern_path(".cartographer.mapping", format!("{core}::mapping"))
        .extern_path(".cartographer.sensor", format!("{core}::sensor"))
        .extern_path(".cartographer.transform", format!("{core}::transform"))
        .compile_fds(file_descriptors)
        .expect("failed to generate Rust code for Cartographer cloud protos");
}
