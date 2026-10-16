//! Port of `cartographer/cloud/internal/handlers/add_odometry_data_handler` (.h, .cc).

use cartographer::sensor::internal::dispatchable::make_dispatchable;
use cartographer::sensor::odometry_data;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddOdometryData` RPC.
#[derive(Debug)]
pub struct AddOdometryDataHandler;

impl AddSensorDataHandler for AddOdometryDataHandler {
    type Request = proto::AddOdometryDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddOdometryDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.odometry_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                odometry_data::from_proto(&payload),
            ),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::OdometryData(payload)),
            }));
        }
    }
}
