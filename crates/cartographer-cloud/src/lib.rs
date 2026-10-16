//! Port of `cartographer/cloud`: a gRPC `MapBuilderService` server
//! ([`internal::map_builder_server::MapBuilderServer`]), client stubs that
//! implement `MapBuilderInterface` / `PoseGraphInterface` /
//! `TrajectoryBuilderInterface` over gRPC ([`client::map_builder_stub`]), and
//! the edge-to-cloud [`internal::local_trajectory_uploader`].
//!
//! The C++ code is built on the `async_grpc` handler framework (one
//! `RpcHandler` class per RPC, an `ExecutionContext`, retry strategies). This
//! port reproduces the behaviour with `tonic` + `tokio`: each handler is a
//! module in [`internal::handlers`] with the same name as the C++ class, the
//! execution context is the [`internal::map_builder_context_interface`]
//! trait, and the retry strategies live in [`internal::async_grpc`]. The wire
//! format and RPC paths (`/cartographer.cloud.proto.MapBuilderService/<Rpc>`)
//! are identical, so Rust and C++ clients/servers interoperate.

/// Generated protobuf/gRPC types of the `cartographer.cloud.proto` package.
/// Imported Cartographer messages are the types from
/// [`cartographer::protos`]; `google.protobuf.Empty` is `()`.
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/cartographer.cloud.proto.rs"));
}

pub mod client;
pub mod internal;
pub mod map_builder_server_interface;
pub mod map_builder_server_options;
#[cfg(feature = "prometheus")]
pub mod metrics;

pub use client::map_builder_stub::MapBuilderStub;
pub use internal::map_builder_server::MapBuilderServer;
pub use map_builder_server_interface::{
    MapBuilderServerInterface, create_map_builder_server, register_map_builder_server_metrics,
};
pub use map_builder_server_options::{
    create_map_builder_server_options, load_map_builder_server_options,
};
