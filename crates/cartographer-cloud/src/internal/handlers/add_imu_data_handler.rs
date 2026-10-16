//! Port of `cartographer/cloud/internal/handlers/add_imu_data_handler` (.h, .cc).

use cartographer::sensor::imu_data;
use cartographer::sensor::internal::dispatchable::make_dispatchable;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddImuData` RPC.
#[derive(Debug)]
pub struct AddImuDataHandler;

impl AddSensorDataHandler for AddImuDataHandler {
    type Request = proto::AddImuDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddImuDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.imu_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(&sensor_metadata.sensor_id, imu_data::from_proto(&payload)),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::ImuData(payload)),
            }));
        }
    }
}
