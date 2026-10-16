//! Port of `cartographer/cloud/internal/handlers/write_state_handler` (.h, .cc).

use cartographer::io::internal::in_memory_proto_stream::ForwardingProtoStreamWriter;
use cartographer::io::proto_stream_interface::{ProtoMessage, ProtoStreamWriterInterface};
use cartographer::mapping::proto::{SerializationHeader, SerializedData};
use tokio::sync::mpsc::UnboundedSender;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;
use crate::proto::write_state_response::StateChunk;

/// Converts a message written by `SerializeState` to the streamed response.
/// Panics on other message types (C++ `LOG(FATAL) << "Unsupported message
/// type"`).
///
/// RUST PORT NOTE: dispatches by downcasting instead of `GetTypeName()`.
pub fn generate_response(message: &dyn ProtoMessage) -> proto::WriteStateResponse {
    let state_chunk = if let Some(header) = message.as_any().downcast_ref::<SerializationHeader>() {
        StateChunk::Header(*header)
    } else if let Some(serialized_data) = message.as_any().downcast_ref::<SerializedData>() {
        StateChunk::SerializedData(serialized_data.clone())
    } else {
        panic!("Unsupported message type: {}", message.type_name());
    };
    proto::WriteStateResponse {
        state_chunk: Some(state_chunk),
    }
}

/// Handles `WriteState`: serializes the state (without unfinished submaps)
/// into the response stream `writer`; dropping the writer ends the stream
/// (C++ `WritesDone`). Runs on a blocking thread.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    writer: UnboundedSender<Result<proto::WriteStateResponse, Status>>,
) {
    let mut writer = Some(writer);
    let mut proto_stream_writer =
        ForwardingProtoStreamWriter::new(Box::new(move |proto: Option<&dyn ProtoMessage>| {
            match proto {
                None => {
                    writer = None;
                }
                Some(proto) => {
                    if let Some(writer) = &writer {
                        // A failed write means the client went away; like the
                        // C++ the serialization still runs to completion.
                        let _ = writer.send(Ok(generate_response(proto)));
                    }
                }
            }
            true
        }));
    context.map_builder().serialize_state(
        /*include_unfinished_submaps=*/ false,
        &mut proto_stream_writer,
    );
    proto_stream_writer.close();
}
