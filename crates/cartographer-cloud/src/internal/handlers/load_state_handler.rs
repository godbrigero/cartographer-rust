//! Port of `cartographer/cloud/internal/handlers/load_state_handler` (.h, .cc).

use cartographer::io::internal::in_memory_proto_stream::InMemoryProtoStreamReader;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::mapping::serialization::trajectory_remapping_to_proto;
use crate::proto;
use crate::proto::load_state_request::StateChunk;

/// The state accumulated over the streamed `LoadStateRequest`s (the C++
/// handler members `reader_`, `client_id_`, `load_frozen_state_`).
#[derive(Default)]
pub struct LoadStateHandler {
    reader: InMemoryProtoStreamReader,
    client_id: String,
    load_frozen_state: bool,
}

impl LoadStateHandler {
    /// Handles one streamed message (C++ `OnRequest`). The
    /// `load_frozen_state` flag of the last message wins.
    ///
    /// RUST PORT NOTE: a message without `state_chunk` is answered with
    /// `INVALID_ARGUMENT` instead of the C++ `LOG(FATAL)`.
    pub fn on_request(&mut self, request: &proto::LoadStateRequest) -> Result<(), Status> {
        match &request.state_chunk {
            Some(StateChunk::SerializedData(serialized_data)) => {
                self.reader.add_proto(serialized_data);
            }
            Some(StateChunk::SerializationHeader(serialization_header)) => {
                self.reader.add_proto(serialization_header);
            }
            Some(StateChunk::ClientId(client_id)) => self.client_id = client_id.clone(),
            None => {
                log::error!("Unhandled proto::LoadStateRequest case.");
                return Err(Status::invalid_argument(
                    "Unhandled proto::LoadStateRequest case.",
                ));
            }
        }
        self.load_frozen_state = request.load_frozen_state;
        Ok(())
    }

    /// Loads the accumulated state and registers the client id for every
    /// loaded trajectory (C++ `OnReadsDone`).
    pub fn on_reads_done(
        mut self,
        context: &dyn MapBuilderContextInterface,
    ) -> Result<proto::LoadStateResponse, Status> {
        let trajectory_remapping = context
            .map_builder()
            .load_state(&mut self.reader, self.load_frozen_state);
        for trajectory_id in trajectory_remapping.values() {
            context.register_client_id_for_trajectory(&self.client_id, *trajectory_id);
        }
        Ok(proto::LoadStateResponse {
            trajectory_remapping: Some(trajectory_remapping_to_proto(&trajectory_remapping)),
        })
    }
}
