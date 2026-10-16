//! Port of `cartographer/cloud/internal/handlers/load_state_from_file_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::mapping::serialization::trajectory_remapping_to_proto;
use crate::proto;

/// Handles `LoadStateFromFile` (the file is read on the server's host) and
/// registers the client id for every loaded trajectory.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::LoadStateFromFileRequest,
) -> Result<proto::LoadStateFromFileResponse, Status> {
    // TODO(gaschler): This blocks a handler thread, consider working in
    // background.
    let trajectory_remapping = context
        .map_builder()
        .load_state_from_file(&request.file_path, request.load_frozen_state);
    for trajectory_id in trajectory_remapping.values() {
        context.register_client_id_for_trajectory(&request.client_id, *trajectory_id);
    }
    Ok(proto::LoadStateFromFileResponse {
        trajectory_remapping: Some(trajectory_remapping_to_proto(&trajectory_remapping)),
    })
}
