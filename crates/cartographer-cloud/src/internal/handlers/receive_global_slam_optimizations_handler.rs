//! Port of `cartographer/cloud/internal/handlers/receive_global_slam_optimizations_handler` (.h, .cc).

use std::collections::BTreeMap;
use std::sync::Arc;

use cartographer::mapping::id::{NodeId, SubmapId};
use tokio::sync::mpsc;
use tonic::Status;

use crate::internal::handlers::SubscriptionStream;
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Converts the last optimized ids per trajectory to the streamed response.
pub fn generate_response(
    last_optimized_submap_ids: &BTreeMap<i32, SubmapId>,
    last_optimized_node_ids: &BTreeMap<i32, NodeId>,
) -> proto::ReceiveGlobalSlamOptimizationsResponse {
    proto::ReceiveGlobalSlamOptimizationsResponse {
        last_optimized_node_ids: last_optimized_node_ids
            .iter()
            .map(|(trajectory_id, node_id)| (*trajectory_id, node_id.to_proto()))
            .collect(),
        last_optimized_submap_ids: last_optimized_submap_ids
            .iter()
            .map(|(trajectory_id, submap_id)| (*trajectory_id, submap_id.to_proto()))
            .collect(),
    }
}

/// Handles `ReceiveGlobalSlamOptimizations`: streams a message after every
/// global optimization until the client goes away.
pub fn on_request(
    context: Arc<dyn MapBuilderContextInterface>,
) -> SubscriptionStream<proto::ReceiveGlobalSlamOptimizationsResponse> {
    let (sender, receiver) = mpsc::unbounded_channel();
    let subscription_index = context.subscribe_global_slam_optimizations(Arc::new(
        move |last_optimized_submap_ids: &BTreeMap<i32, SubmapId>,
              last_optimized_node_ids: &BTreeMap<i32, NodeId>| {
            if sender
                .send(Ok::<_, Status>(generate_response(
                    last_optimized_submap_ids,
                    last_optimized_node_ids,
                )))
                .is_err()
            {
                // Client closed connection.
                log::info!("Client closed connection.");
                return false;
            }
            true
        },
    ));
    log::info!("Added subscription: {subscription_index}");
    SubscriptionStream::new(
        receiver,
        Box::new(move || {
            log::info!("Removing subscription {subscription_index}");
            context.unsubscribe_global_slam_optimizations(subscription_index);
        }),
    )
}
