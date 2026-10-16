//! Port of `cartographer/io/proto_stream_deserializer` (.h, .cc).

use crate::io::internal::mapping_state_serialization::{
    FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS, MAPPING_STATE_SERIALIZATION_FORMAT_VERSION,
};
use crate::io::proto_stream::ProtoStreamReader;
use crate::io::proto_stream_interface::ProtoStreamReaderInterface;
use crate::mapping::proto::{
    AllTrajectoryBuilderOptions, PoseGraph, SerializationHeader, SerializedData,
    serialized_data::Data,
};

fn read_header_or_die(reader: &mut dyn ProtoStreamReaderInterface) -> SerializationHeader {
    let mut header = SerializationHeader::default();
    assert!(
        reader.read_proto(&mut header),
        "Check failed: reader->ReadProto(&header) Failed to read SerializationHeader."
    );
    header
}

fn is_version_supported(header: &SerializationHeader) -> bool {
    header.format_version == MAPPING_STATE_SERIALIZATION_FORMAT_VERSION
        || header.format_version == FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS
}

/// The protobuf field number of the set `data` oneof field, 0 if unset (C++
/// `SerializedData::data_case()`).
pub fn data_case(data: &SerializedData) -> i32 {
    match &data.data {
        None => 0,
        Some(Data::PoseGraph(_)) => 1,
        Some(Data::AllTrajectoryBuilderOptions(_)) => 2,
        Some(Data::Submap(_)) => 3,
        Some(Data::Node(_)) => 4,
        Some(Data::TrajectoryData(_)) => 5,
        Some(Data::ImuData(_)) => 6,
        Some(Data::OdometryData(_)) => 7,
        Some(Data::FixedFramePoseData(_)) => 8,
        Some(Data::LandmarkData(_)) => 9,
    }
}

/// Helper function for deserializing the PoseGraph from a proto stream file.
pub fn deserialize_pose_graph_from_file(file_name: &str) -> PoseGraph {
    let mut reader = ProtoStreamReader::new(file_name);
    let deserializer = ProtoStreamDeserializer::new(&mut reader);
    deserializer.pose_graph().clone()
}

/// Helper for deserializing a previously serialized mapping state from a
/// proto stream, abstracting away the format parsing logic.
///
/// The constructor reads the `SerializationHeader` (format version 1 or 2),
/// the `PoseGraph` and the `AllTrajectoryBuilderOptions`; all failures panic
/// (C++ CHECK).
pub struct ProtoStreamDeserializer<'a> {
    reader: &'a mut dyn ProtoStreamReaderInterface,
    header: SerializationHeader,
    pose_graph: SerializedData,
    all_trajectory_builder_options: SerializedData,
}

impl<'a> ProtoStreamDeserializer<'a> {
    /// Reads the stream prefix from `reader`.
    pub fn new(reader: &'a mut dyn ProtoStreamReaderInterface) -> Self {
        let header = read_header_or_die(reader);
        assert!(
            is_version_supported(&header),
            "Unsupported serialization format \"{}\"",
            header.format_version
        );
        let mut deserializer = Self {
            reader,
            header,
            pose_graph: SerializedData::default(),
            all_trajectory_builder_options: SerializedData::default(),
        };

        let mut pose_graph = SerializedData::default();
        assert!(
            deserializer.read_next_serialized_data(&mut pose_graph),
            "Serialized stream misses PoseGraph."
        );
        assert!(
            matches!(pose_graph.data, Some(Data::PoseGraph(_))),
            "Serialized stream order corrupt. Expecting `PoseGraph` after \
             `SerializationHeader`, but got field tag {}",
            data_case(&pose_graph)
        );

        let mut all_trajectory_builder_options = SerializedData::default();
        assert!(
            deserializer.read_next_serialized_data(&mut all_trajectory_builder_options),
            "Serialized stream misses `AllTrajectoryBuilderOptions`."
        );
        assert!(
            matches!(
                all_trajectory_builder_options.data,
                Some(Data::AllTrajectoryBuilderOptions(_))
            ),
            "Serialized stream order corrupt. Expecting `AllTrajectoryBuilderOptions` after \
             PoseGraph, got field tag {}",
            data_case(&all_trajectory_builder_options)
        );

        deserializer.pose_graph = pose_graph;
        deserializer.all_trajectory_builder_options = all_trajectory_builder_options;
        let num_trajectories = deserializer.pose_graph().trajectory.len();
        let num_options = deserializer
            .all_trajectory_builder_options()
            .options_with_sensor_ids
            .len();
        assert_eq!(
            num_trajectories, num_options,
            "Check failed: pose_graph_.pose_graph().trajectory_size() == \
             all_trajectory_builder_options_.all_trajectory_builder_options()\
             .options_with_sensor_ids_size()"
        );
        deserializer
    }

    /// The serialization header.
    pub fn header(&self) -> &SerializationHeader {
        &self.header
    }

    /// Mutable access to the serialization header.
    pub fn header_mut(&mut self) -> &mut SerializationHeader {
        &mut self.header
    }

    /// The pose graph.
    pub fn pose_graph(&self) -> &PoseGraph {
        match &self.pose_graph.data {
            Some(Data::PoseGraph(pose_graph)) => pose_graph,
            _ => unreachable!("checked in the constructor"),
        }
    }

    /// Mutable access to the pose graph.
    pub fn pose_graph_mut(&mut self) -> &mut PoseGraph {
        match &mut self.pose_graph.data {
            Some(Data::PoseGraph(pose_graph)) => pose_graph,
            _ => unreachable!("checked in the constructor"),
        }
    }

    /// The options of all trajectory builders.
    pub fn all_trajectory_builder_options(&self) -> &AllTrajectoryBuilderOptions {
        match &self.all_trajectory_builder_options.data {
            Some(Data::AllTrajectoryBuilderOptions(options)) => options,
            _ => unreachable!("checked in the constructor"),
        }
    }

    /// Reads the next `SerializedData` message of the ProtoStream into
    /// `data`. Returns `true` if the message was successfully read or `false`
    /// in case there are no-more messages or an error occurred.
    pub fn read_next_serialized_data(&mut self, data: &mut SerializedData) -> bool {
        self.reader.read_proto(data)
    }
}

#[cfg(test)]
mod tests {
    mod proto_stream_deserializer_test {
        use crate::io::internal::testing::test_helpers::{
            proto_from_string_or_die, proto_reader_from_strings,
        };
        use crate::io::proto_stream_deserializer::*;
        use crate::io::proto_stream_interface::ProtoStreamReaderInterface;

        const SERIALIZATION_HEADER_PROTO_STRING: &str = "format_version: 1";
        const UNSUPPORTED_SERIALIZATION_HEADER_PROTO_STRING: &str = "format_version: 123";
        const POSE_GRAPH_PROTO_STRING: &str = "pose_graph {}";
        const ALL_TRAJECTORY_BUILDER_OPTIONS_PROTO_STRING: &str =
            "all_trajectory_builder_options {}";
        const SUBMAP_PROTO_STRING: &str = "submap {}";
        const NODE_PROTO_STRING: &str = "node {}";
        const TRAJECTORY_DATA_PROTO_STRING: &str = "trajectory_data {}";
        const IMU_DATA_PROTO_STRING: &str = "imu_data {}";
        const ODOMETRY_DATA_PROTO_STRING: &str = "odometry_data {}";
        const FIXED_FRAME_POSE_DATA_PROTO_STRING: &str = "fixed_frame_pose_data {}";
        const LANDMARK_DATA_PROTO_STRING: &str = "landmark_data {}";

        fn pose_graph_of(data: SerializedData) -> PoseGraph {
            match data.data {
                Some(Data::PoseGraph(p)) => p,
                _ => panic!(),
            }
        }

        fn options_of(data: SerializedData) -> AllTrajectoryBuilderOptions {
            match data.data {
                Some(Data::AllTrajectoryBuilderOptions(o)) => o,
                _ => panic!(),
            }
        }

        // This test checks if the serialization works.
        #[test]
        fn works_on_golden_text_stream() {
            // Load text proto into in_memory_reader.
            let mut reader = proto_reader_from_strings(
                SERIALIZATION_HEADER_PROTO_STRING,
                &[
                    POSE_GRAPH_PROTO_STRING,
                    ALL_TRAJECTORY_BUILDER_OPTIONS_PROTO_STRING,
                    SUBMAP_PROTO_STRING,
                    NODE_PROTO_STRING,
                    TRAJECTORY_DATA_PROTO_STRING,
                    IMU_DATA_PROTO_STRING,
                    ODOMETRY_DATA_PROTO_STRING,
                    FIXED_FRAME_POSE_DATA_PROTO_STRING,
                    LANDMARK_DATA_PROTO_STRING,
                ],
            );

            {
                let mut deserializer = ProtoStreamDeserializer::new(reader.as_mut());

                assert_eq!(
                    *deserializer.header(),
                    proto_from_string_or_die::<SerializationHeader>(
                        SERIALIZATION_HEADER_PROTO_STRING
                    )
                );
                assert_eq!(
                    *deserializer.pose_graph(),
                    pose_graph_of(proto_from_string_or_die(POSE_GRAPH_PROTO_STRING))
                );
                assert_eq!(
                    *deserializer.all_trajectory_builder_options(),
                    options_of(proto_from_string_or_die(
                        ALL_TRAJECTORY_BUILDER_OPTIONS_PROTO_STRING
                    ))
                );

                let mut serialized_data = SerializedData::default();
                for expected in [
                    SUBMAP_PROTO_STRING,
                    NODE_PROTO_STRING,
                    TRAJECTORY_DATA_PROTO_STRING,
                    IMU_DATA_PROTO_STRING,
                    ODOMETRY_DATA_PROTO_STRING,
                    FIXED_FRAME_POSE_DATA_PROTO_STRING,
                    LANDMARK_DATA_PROTO_STRING,
                ] {
                    assert!(deserializer.read_next_serialized_data(&mut serialized_data));
                    assert_eq!(
                        serialized_data,
                        proto_from_string_or_die::<SerializedData>(expected)
                    );
                }

                assert!(!deserializer.read_next_serialized_data(&mut serialized_data));
            }
            assert!(reader.eof());
        }

        #[test]
        #[should_panic(expected = "Unsupported serialization format")]
        fn fails_if_version_not_supported() {
            let mut reader =
                proto_reader_from_strings(UNSUPPORTED_SERIALIZATION_HEADER_PROTO_STRING, &[]);
            ProtoStreamDeserializer::new(reader.as_mut());
        }

        #[test]
        #[should_panic(
            expected = "Expecting `PoseGraph` after `SerializationHeader`, but got field tag 3"
        )]
        fn fails_on_wrong_order() {
            let mut reader = proto_reader_from_strings(
                SERIALIZATION_HEADER_PROTO_STRING,
                &[SUBMAP_PROTO_STRING],
            );
            ProtoStreamDeserializer::new(reader.as_mut());
        }

        #[test]
        #[should_panic(expected = "Serialized stream misses `AllTrajectoryBuilderOptions`.")]
        fn fails_on_missing_options() {
            let mut reader = proto_reader_from_strings(
                SERIALIZATION_HEADER_PROTO_STRING,
                &[POSE_GRAPH_PROTO_STRING],
            );
            ProtoStreamDeserializer::new(reader.as_mut());
        }

        #[test]
        fn data_case_numbers_match_field_tags() {
            for (text, tag) in [
                ("", 0),
                (POSE_GRAPH_PROTO_STRING, 1),
                (ALL_TRAJECTORY_BUILDER_OPTIONS_PROTO_STRING, 2),
                (SUBMAP_PROTO_STRING, 3),
                (NODE_PROTO_STRING, 4),
                (TRAJECTORY_DATA_PROTO_STRING, 5),
                (IMU_DATA_PROTO_STRING, 6),
                (ODOMETRY_DATA_PROTO_STRING, 7),
                (FIXED_FRAME_POSE_DATA_PROTO_STRING, 8),
                (LANDMARK_DATA_PROTO_STRING, 9),
            ] {
                assert_eq!(data_case(&proto_from_string_or_die(text)), tag, "{text}");
            }
        }
    }

    mod file_round_trip {
        use crate::io::proto_stream::ProtoStreamWriter;
        use crate::io::proto_stream_deserializer::*;
        use crate::io::proto_stream_interface::ProtoStreamWriterInterface;
        use crate::mapping::proto;

        #[test]
        fn deserialize_pose_graph_from_file_reads_written_stream() {
            let path = std::env::temp_dir().join(format!(
                "cartographer_{}_deserializer.pbstream",
                std::process::id()
            ));
            let filename = path.to_str().unwrap();
            let pose_graph = PoseGraph {
                trajectory: vec![
                    proto::Trajectory {
                        trajectory_id: 0,
                        ..Default::default()
                    },
                    proto::Trajectory {
                        trajectory_id: 1,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            };
            {
                let mut writer = ProtoStreamWriter::new(filename);
                writer.write_proto(&SerializationHeader { format_version: 2 });
                writer.write_proto(&SerializedData {
                    data: Some(Data::PoseGraph(pose_graph.clone())),
                });
                writer.write_proto(&SerializedData {
                    data: Some(Data::AllTrajectoryBuilderOptions(
                        AllTrajectoryBuilderOptions {
                            options_with_sensor_ids: vec![Default::default(); 2],
                        },
                    )),
                });
                writer.write_proto(&SerializedData {
                    data: Some(Data::Node(proto::Node::default())),
                });
                assert!(writer.close());
            }
            assert_eq!(deserialize_pose_graph_from_file(filename), pose_graph);
            std::fs::remove_file(&path).unwrap();
        }
    }
}
