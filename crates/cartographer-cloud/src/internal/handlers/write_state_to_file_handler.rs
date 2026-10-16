//! Port of `cartographer/cloud/internal/handlers/write_state_to_file_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `WriteStateToFile` (written on the server's host, without
/// unfinished submaps). An empty filename is `INVALID_ARGUMENT`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::WriteStateToFileRequest,
) -> Result<proto::WriteStateToFileResponse, Status> {
    if request.filename.is_empty() {
        return Err(Status::invalid_argument("Filename empty."));
    }
    let success = context.map_builder().serialize_state_to_file(
        /*include_unfinished_submaps=*/ false,
        &request.filename,
    );
    Ok(proto::WriteStateToFileResponse { success })
}
