//! Port of `cartographer/cloud/internal/handlers/add_rangefinder_data_handler` (.h, .cc).

use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::timed_point_cloud_data;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddRangefinderData` RPC.
#[derive(Debug)]
pub struct AddRangefinderDataHandler;

impl AddSensorDataHandler for AddRangefinderDataHandler {
    type Request = proto::AddRangefinderDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddRangefinderDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.timed_point_cloud_data.clone().unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                timed_point_cloud_data::from_proto(&payload),
            ),
        );
        // Range data is never forwarded to the uploader: it is uplinked as
        // part of the local SLAM results instead.
    }
}
