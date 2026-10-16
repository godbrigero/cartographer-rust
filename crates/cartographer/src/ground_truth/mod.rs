//! Port of `cartographer/ground_truth/`.

pub mod autogenerate_ground_truth;
pub mod compute_relations_metrics;
pub mod relations_text_file;

/// Generated protobuf types of package `cartographer.ground_truth.proto`.
pub mod proto {
    pub use crate::protos::cartographer::ground_truth::proto::*;
}

pub use autogenerate_ground_truth::generate_ground_truth;
pub use relations_text_file::read_relations_text_file;
