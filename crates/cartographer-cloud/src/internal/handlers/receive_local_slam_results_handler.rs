//! Port of `cartographer/cloud/internal/handlers/receive_local_slam_results_handler` (.h, .cc).

use std::sync::{Arc, Mutex};

use cartographer::common::time::to_universal;
use cartographer::sensor::range_data;
use cartographer::transform::transform::to_proto;
use tokio::sync::mpsc;
use tonic::Status;

use crate::internal::handlers::SubscriptionStream;
use crate::internal::map_builder_context_interface::{LocalSlamResult, MapBuilderContextInterface};
use crate::internal::map_builder_server::lock;
use crate::proto;

/// Converts a local SLAM result to the streamed response.
pub fn generate_response(
    local_slam_result: &LocalSlamResult,
) -> proto::ReceiveLocalSlamResultsResponse {
    proto::ReceiveLocalSlamResultsResponse {
        trajectory_id: local_slam_result.trajectory_id,
        timestamp: to_universal(local_slam_result.time),
        local_pose: Some(to_proto(&local_slam_result.local_pose)),
        range_data: local_slam_result
            .range_data
            .as_ref()
            .map(|range_data| range_data::to_proto(range_data)),
        insertion_result: local_slam_result
            .insertion_result
            .as_ref()
            .map(|insertion_result| proto::LocalSlamInsertionResult {
                node_id: Some(insertion_result.node_id.to_proto()),
            }),
    }
}

/// Handles `ReceiveLocalSlamResults`: subscribes to the local SLAM results of
/// the trajectory and streams them until the trajectory is finished (the
/// stream ends) or the client goes away (the subscription is removed).
pub fn on_request(
    context: Arc<dyn MapBuilderContextInterface>,
    request: &proto::ReceiveLocalSlamResultsRequest,
) -> SubscriptionStream<proto::ReceiveLocalSlamResultsResponse> {
    let (sender, receiver) = mpsc::unbounded_channel();
    // `None` once the trajectory finished (C++ `writer.WritesDone()`).
    let writer = Mutex::new(Some(sender));
    let subscription_id = context.subscribe_local_slam_results(
        request.trajectory_id,
        Arc::new(move |local_slam_result: Option<Box<LocalSlamResult>>| {
            let mut writer = lock(&writer);
            match local_slam_result {
                Some(local_slam_result) => {
                    let written = writer.as_ref().is_some_and(|sender| {
                        sender
                            .send(Ok::<_, Status>(generate_response(&local_slam_result)))
                            .is_ok()
                    });
                    if !written {
                        // Client closed connection.
                        log::info!("Client closed connection.");
                        return false;
                    }
                }
                None => {
                    // Callback with 'None' signals that the trajectory
                    // finished.
                    *writer = None;
                }
            }
            true
        }),
    );
    SubscriptionStream::new(
        receiver,
        Box::new(move || context.unsubscribe_local_slam_results(&subscription_id)),
    )
}
