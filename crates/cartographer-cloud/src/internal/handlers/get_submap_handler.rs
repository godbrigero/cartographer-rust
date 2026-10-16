//! Port of `cartographer/cloud/internal/handlers/get_submap_handler` (.h, .cc).

use cartographer::mapping::id::SubmapId;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetSubmap`; a failure is reported in `error_msg`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::GetSubmapRequest,
) -> Result<proto::GetSubmapResponse, Status> {
    let submap_id = request.submap_id.unwrap_or_default();
    let mut submap_query_response = Default::default();
    let error_msg = context.map_builder().submap_to_proto(
        &SubmapId::new(submap_id.trajectory_id, submap_id.submap_index),
        &mut submap_query_response,
    );
    Ok(proto::GetSubmapResponse {
        submap_query_response: Some(submap_query_response),
        error_msg,
    })
}
