//! Port of `cartographer/io/proto_stream_interface` (.h).

use std::any::Any;

/// An object safe view of a protobuf message (the parts of
/// `google::protobuf::Message` used by the proto streams).
///
/// RUST PORT NOTE: `prost::Message` is not usable as a trait object, so the
/// stream interfaces take `&dyn ProtoMessage`, which is implemented for every
/// generated prost message. Callers pass concrete messages
/// (`writer.write_proto(&trajectory)`, `reader.read_proto(&mut header)`).
pub trait ProtoMessage: Any + Send {
    /// `SerializeToString`.
    fn serialize_to_bytes(&self) -> Vec<u8>;
    /// `ParseFromString`: replaces the contents with the parsed `bytes`.
    /// Returns false if `bytes` is not a valid encoding.
    fn parse_from_bytes(&mut self, bytes: &[u8]) -> bool;
    /// `CopyFrom`. Panics if `other` is a different message type (protobuf
    /// CHECK-fails in that case too).
    fn copy_from(&mut self, other: &dyn ProtoMessage);
    /// The Rust type name of the message (C++ `GetTypeName()` returns the
    /// protobuf full name instead).
    fn type_name(&self) -> &'static str;
    /// Upcast for downcasting to the concrete message type.
    fn as_any(&self) -> &dyn Any;
    /// Mutable upcast for downcasting to the concrete message type.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Boxed copy of the message.
    fn clone_message(&self) -> Box<dyn ProtoMessage>;
}

impl<M> ProtoMessage for M
where
    M: prost::Message + Default + Clone + Send + 'static,
{
    fn serialize_to_bytes(&self) -> Vec<u8> {
        self.encode_to_vec()
    }

    fn parse_from_bytes(&mut self, bytes: &[u8]) -> bool {
        match M::decode(bytes) {
            Ok(message) => {
                *self = message;
                true
            }
            Err(_) => false,
        }
    }

    fn copy_from(&mut self, other: &dyn ProtoMessage) {
        let Some(other_message) = other.as_any().downcast_ref::<M>() else {
            panic!(
                "Tried to copy from a message with a different type. to: {}, from: {}",
                std::any::type_name::<M>(),
                other.type_name()
            );
        };
        *self = other_message.clone();
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<M>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_message(&self) -> Box<dyn ProtoMessage> {
        Box::new(self.clone())
    }
}

/// A writer for writing proto messages to a pbstream.
pub trait ProtoStreamWriterInterface {
    /// Serializes, compressed and writes the `proto` to the file.
    fn write_proto(&mut self, proto: &dyn ProtoMessage);

    /// This should be called to check whether writing was successful.
    fn close(&mut self) -> bool;
}

/// A reader of the format produced by `ProtoStreamWriter`.
pub trait ProtoStreamReaderInterface {
    /// Deserialize compressed proto from the pb stream.
    fn read_proto(&mut self, proto: &mut dyn ProtoMessage) -> bool;

    /// 'End-of-file' marker for the pb stream.
    fn eof(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::proto;

    #[test]
    fn copy_from_same_type() {
        let source = proto::SerializationHeader { format_version: 7 };
        let mut target = proto::SerializationHeader::default();
        target.copy_from(&source);
        assert_eq!(target.format_version, 7);
    }

    #[test]
    #[should_panic(expected = "different type")]
    fn copy_from_different_type_panics() {
        let source = proto::SerializationHeader { format_version: 7 };
        let mut target = proto::SerializedData::default();
        target.copy_from(&source);
    }

    #[test]
    fn serialize_and_parse() {
        let source = proto::SerializationHeader { format_version: 2 };
        let bytes = source.serialize_to_bytes();
        let mut target = proto::SerializationHeader { format_version: 9 };
        assert!(target.parse_from_bytes(&bytes));
        assert_eq!(target, source);
        assert!(!target.parse_from_bytes(&[0xff, 0xff, 0xff]));
    }
}
