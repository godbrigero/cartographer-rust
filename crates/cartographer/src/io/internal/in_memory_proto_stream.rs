//! Port of `cartographer/io/internal/in_memory_proto_stream` (.h, .cc).

use std::collections::VecDeque;

use crate::io::proto_stream_interface::{
    ProtoMessage, ProtoStreamReaderInterface, ProtoStreamWriterInterface,
};

/// A callback that is invoked anytime `write_proto()` is called on the
/// [`ForwardingProtoStreamWriter`]. When `close()` is called on the
/// `ForwardingProtoStreamWriter` the callback is invoked with `None` (C++
/// `nullptr`).
///
/// Use [`ProtoMessage::as_any`] to dispatch on the message type (C++
/// `GetTypeName()`).
pub type WriterCallback = Box<dyn FnMut(Option<&dyn ProtoMessage>) -> bool + Send>;

/// A [`ProtoStreamWriterInterface`] forwarding all messages to a callback.
pub struct ForwardingProtoStreamWriter {
    writer_callback: WriterCallback,
}

impl ForwardingProtoStreamWriter {
    /// Creates the writer.
    pub fn new(writer_callback: WriterCallback) -> Self {
        Self { writer_callback }
    }
}

impl ProtoStreamWriterInterface for ForwardingProtoStreamWriter {
    fn write_proto(&mut self, proto: &dyn ProtoMessage) {
        assert!(
            (self.writer_callback)(Some(proto)),
            "Check failed: writer_callback_(&proto)"
        );
    }

    fn close(&mut self) -> bool {
        (self.writer_callback)(None)
    }
}

/// A [`ProtoStreamReaderInterface`] reading from an in-memory queue of
/// messages.
#[derive(Default)]
pub struct InMemoryProtoStreamReader {
    state_chunks: VecDeque<Box<dyn ProtoMessage>>,
}

impl InMemoryProtoStreamReader {
    /// Creates a reader returning the messages of `state_chunks` in order.
    pub fn new(state_chunks: VecDeque<Box<dyn ProtoMessage>>) -> Self {
        Self { state_chunks }
    }

    /// Appends a copy of `proto`.
    pub fn add_proto<M: ProtoMessage + Clone>(&mut self, proto: &M) {
        self.state_chunks.push_back(Box::new(proto.clone()));
    }
}

impl ProtoStreamReaderInterface for InMemoryProtoStreamReader {
    /// Copies the next message into `proto` (which must have the same type,
    /// like protobuf's `CopyFrom`). Returns false if the queue is empty.
    fn read_proto(&mut self, proto: &mut dyn ProtoMessage) -> bool {
        let Some(front) = self.state_chunks.pop_front() else {
            return false;
        };
        proto.copy_from(front.as_ref());
        true
    }

    /// True if the queue is empty. Unlike the file reader this is already
    /// true *before* a read fails.
    fn eof(&self) -> bool {
        self.state_chunks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    mod in_memory_proto_stream_test {
        use std::collections::VecDeque;
        use std::sync::{Arc, Mutex};

        use crate::io::internal::in_memory_proto_stream::*;
        use crate::mapping::proto::{self, PoseGraph, SerializedData};

        struct Fixture {
            pose_graph: PoseGraph,
            serialized_data: SerializedData,
        }

        fn set_up() -> Fixture {
            let mut pose_graph = PoseGraph::default();
            pose_graph.trajectory.push(proto::Trajectory {
                trajectory_id: 1,
                ..Default::default()
            });
            let serialized_data = SerializedData {
                data: Some(proto::serialized_data::Data::OdometryData(
                    proto::OdometryData {
                        trajectory_id: 2,
                        ..Default::default()
                    },
                )),
            };
            Fixture {
                pose_graph,
                serialized_data,
            }
        }

        fn odometry_trajectory_id(data: &SerializedData) -> i32 {
            match &data.data {
                Some(proto::serialized_data::Data::OdometryData(o)) => o.trajectory_id,
                _ => panic!("not odometry data"),
            }
        }

        #[test]
        fn read_stream_initialized_from_queue() {
            let f = set_up();
            let mut proto_queue: VecDeque<Box<dyn ProtoMessage>> = VecDeque::new();
            proto_queue.push_back(Box::new(f.pose_graph.clone()));
            proto_queue.push_back(Box::new(f.serialized_data.clone()));
            let mut reader = InMemoryProtoStreamReader::new(proto_queue);

            let mut actual_pose_graph = PoseGraph::default();
            assert!(!reader.eof());
            assert!(reader.read_proto(&mut actual_pose_graph));
            assert_eq!(1, actual_pose_graph.trajectory[0].trajectory_id);

            let mut actual_serialized_data = SerializedData::default();
            assert!(!reader.eof());
            assert!(reader.read_proto(&mut actual_serialized_data));
            assert_eq!(2, odometry_trajectory_id(&actual_serialized_data));

            assert!(reader.eof());
        }

        #[test]
        fn read_stream_initialized_incrementally() {
            let f = set_up();
            let mut reader = InMemoryProtoStreamReader::default();
            reader.add_proto(&f.pose_graph);
            reader.add_proto(&f.serialized_data);

            let mut actual_pose_graph = PoseGraph::default();
            assert!(!reader.eof());
            assert!(reader.read_proto(&mut actual_pose_graph));
            assert_eq!(1, actual_pose_graph.trajectory[0].trajectory_id);

            let mut actual_serialized_data = SerializedData::default();
            assert!(!reader.eof());
            assert!(reader.read_proto(&mut actual_serialized_data));
            assert_eq!(2, odometry_trajectory_id(&actual_serialized_data));

            assert!(reader.eof());
        }

        #[test]
        fn read_from_empty_reader_fails() {
            let mut reader = InMemoryProtoStreamReader::default();
            assert!(reader.eof());
            assert!(!reader.read_proto(&mut PoseGraph::default()));
        }

        #[test]
        fn forwarding_writer_forwards_and_signals_close() {
            let received = Arc::new(Mutex::new(Vec::new()));
            let mut writer = {
                let received = Arc::clone(&received);
                ForwardingProtoStreamWriter::new(Box::new(move |proto| {
                    let entry = match proto {
                        None => "closed".to_string(),
                        Some(p) => match p.as_any().downcast_ref::<proto::SerializationHeader>() {
                            Some(h) => format!("header {}", h.format_version),
                            None => "data".to_string(),
                        },
                    };
                    received.lock().unwrap().push(entry);
                    true
                }))
            };
            writer.write_proto(&proto::SerializationHeader { format_version: 2 });
            writer.write_proto(&SerializedData::default());
            assert!(writer.close());
            assert_eq!(
                *received.lock().unwrap(),
                vec![
                    "header 2".to_string(),
                    "data".to_string(),
                    "closed".to_string()
                ]
            );
        }

        #[test]
        #[should_panic(expected = "writer_callback_")]
        fn forwarding_writer_panics_if_callback_fails() {
            let mut writer = ForwardingProtoStreamWriter::new(Box::new(|_| false));
            writer.write_proto(&SerializedData::default());
        }
    }
}
