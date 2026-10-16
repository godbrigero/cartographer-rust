//! Native Rust port of Google Cartographer.

/// Generated protobuf types, laid out by proto package
/// (`cartographer.mapping.proto` -> `protos::cartographer::mapping::proto`).
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod protos {
    include!(concat!(env!("OUT_DIR"), "/protos.rs"));
}

pub mod common;
pub mod ground_truth;
pub mod io;
pub mod mapping;
pub mod metrics;
pub mod sensor;
pub mod transform;
