//! Port of `cartographer/io/internal/testing/test_helpers` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ helpers parse protobuf text format
//! (`TextFormat::ParseFromString`), which prost does not support. We
//! implement the small subset of the text format used by the io tests:
//!
//! * `SerializationHeader`: `format_version: <number>` (or empty);
//! * `SerializedData`: `<oneof field name> {}` (or empty), i.e. setting one
//!   of the `data` oneof fields to an empty message.
//!
//! Anything else panics (like `ProtoFromStringOrDie` on parse errors).
//! Tests needing richer messages construct them in code and use
//! [`proto_reader_from_protos`].

use std::collections::VecDeque;

use crate::io::internal::in_memory_proto_stream::InMemoryProtoStreamReader;
use crate::io::proto_stream_interface::ProtoMessage;
use crate::mapping::proto::{self, SerializationHeader, SerializedData, serialized_data::Data};

/// Messages that can be parsed from the supported text format subset.
pub trait FromTextProto: Sized {
    /// Parses `text`; `None` if it is not in the supported subset.
    fn from_text_proto(text: &str) -> Option<Self>;
}

impl FromTextProto for SerializationHeader {
    fn from_text_proto(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Some(Self::default());
        }
        let (field, value) = text.split_once(':')?;
        if field.trim() != "format_version" {
            return None;
        }
        Some(Self {
            format_version: value.trim().parse().ok()?,
        })
    }
}

impl FromTextProto for SerializedData {
    fn from_text_proto(text: &str) -> Option<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Some(Self::default());
        }
        let field = text.strip_suffix('}')?.trim_end().strip_suffix('{')?.trim();
        let data = match field {
            "pose_graph" => Data::PoseGraph(proto::PoseGraph::default()),
            "all_trajectory_builder_options" => {
                Data::AllTrajectoryBuilderOptions(proto::AllTrajectoryBuilderOptions::default())
            }
            "submap" => Data::Submap(proto::Submap::default()),
            "node" => Data::Node(proto::Node::default()),
            "trajectory_data" => Data::TrajectoryData(proto::TrajectoryData::default()),
            "imu_data" => Data::ImuData(proto::ImuData::default()),
            "odometry_data" => Data::OdometryData(proto::OdometryData::default()),
            "fixed_frame_pose_data" => {
                Data::FixedFramePoseData(proto::FixedFramePoseData::default())
            }
            "landmark_data" => Data::LandmarkData(proto::LandmarkData::default()),
            _ => return None,
        };
        Some(Self { data: Some(data) })
    }
}

/// Parses `proto_string` (see the module docs for the supported subset);
/// panics on failure.
pub fn proto_from_string_or_die<T: FromTextProto>(proto_string: &str) -> T {
    T::from_text_proto(proto_string).unwrap_or_else(|| {
        panic!(
            "Check failed: TextFormat::ParseFromString(proto_string, &msg) \
             (unsupported text proto: {proto_string:?})"
        )
    })
}

/// Creates an in-memory reader returning a `SerializationHeader` parsed from
/// `header_textpb` followed by the `SerializedData` messages parsed from
/// `data_textpbs`.
pub fn proto_reader_from_strings(
    header_textpb: &str,
    data_textpbs: &[&str],
) -> Box<InMemoryProtoStreamReader> {
    proto_reader_from_protos(
        proto_from_string_or_die(header_textpb),
        data_textpbs
            .iter()
            .map(|text| proto_from_string_or_die(text))
            .collect(),
    )
}

/// Like [`proto_reader_from_strings`] with already constructed messages.
pub fn proto_reader_from_protos(
    header: SerializationHeader,
    data: Vec<SerializedData>,
) -> Box<InMemoryProtoStreamReader> {
    let mut proto_queue: VecDeque<Box<dyn ProtoMessage>> = VecDeque::new();
    proto_queue.push_back(Box::new(header));
    for d in data {
        proto_queue.push_back(Box::new(d));
    }
    Box::new(InMemoryProtoStreamReader::new(proto_queue))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::proto_stream_interface::ProtoStreamReaderInterface;

    #[test]
    fn parses_supported_subset() {
        let header: SerializationHeader = proto_from_string_or_die("format_version: 123");
        assert_eq!(header.format_version, 123);
        let data: SerializedData = proto_from_string_or_die("imu_data {}");
        assert!(matches!(data.data, Some(Data::ImuData(_))));
        let data: SerializedData = proto_from_string_or_die("  pose_graph  { } ");
        assert!(matches!(data.data, Some(Data::PoseGraph(_))));
    }

    #[test]
    #[should_panic(expected = "unsupported text proto")]
    fn rejects_unsupported_text() {
        let _: SerializedData = proto_from_string_or_die("pose_graph { trajectory {} }");
    }

    #[test]
    fn reader_from_strings() {
        let mut reader = proto_reader_from_strings("format_version: 2", &["node {}"]);
        let mut header = SerializationHeader::default();
        assert!(reader.read_proto(&mut header));
        assert_eq!(header.format_version, 2);
        let mut data = SerializedData::default();
        assert!(reader.read_proto(&mut data));
        assert!(matches!(data.data, Some(Data::Node(_))));
        assert!(reader.eof());
    }
}
