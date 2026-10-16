//! Port of `cartographer/cloud/map_builder_server_interface` (.h, .cc).

use cartographer::metrics::family_factory::FamilyFactory;

use crate::internal::map_builder_context_interface::BoxedMapBuilder;
use crate::internal::map_builder_server::MapBuilderServer;
use crate::proto;

/// A map builder server.
///
/// RUST PORT NOTE: all methods take `&self` so that one thread can block in
/// [`wait_for_shutdown`](Self::wait_for_shutdown) while another calls
/// [`shutdown`](Self::shutdown).
pub trait MapBuilderServerInterface: Send + Sync {
    /// Starts the gRPC server, the `LocalTrajectoryUploader` and the SLAM
    /// thread.
    fn start(&self);

    /// Waits for the `MapBuilderServer` to shut down. Note: The server must
    /// be either shutting down or some other thread must call `shutdown()`
    /// for this function to ever return.
    fn wait_for_shutdown(&self);

    /// Waits until all computation is finished (for testing).
    fn wait_until_idle(&self);

    /// Shuts down the gRPC server, the `LocalTrajectoryUploader` and the
    /// SLAM thread.
    fn shutdown(&self);
}

/// Registers the metrics of the map builder server.
pub fn register_map_builder_server_metrics(factory: &mut dyn FamilyFactory) {
    MapBuilderServer::register_metrics(factory);
}

/// Creates a [`MapBuilderServer`] serving `map_builder`.
pub fn create_map_builder_server(
    map_builder_server_options: &proto::MapBuilderServerOptions,
    map_builder: BoxedMapBuilder,
) -> Box<dyn MapBuilderServerInterface> {
    Box::new(MapBuilderServer::new(
        map_builder_server_options,
        map_builder,
    ))
}
