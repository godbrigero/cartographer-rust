//! Port of `cartographer/cloud/internal/handlers/add_fixed_frame_pose_data_handler` (.h, .cc).

use cartographer::sensor::fixed_frame_pose_data;
use cartographer::sensor::internal::dispatchable::make_dispatchable;

use crate::internal::handlers::add_sensor_data_handler_base::{
    AddSensorDataHandler, SensorDataRequest,
};
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handler of the client-streaming `AddFixedFramePoseData` RPC.
#[derive(Debug)]
pub struct AddFixedFramePoseDataHandler;

impl AddSensorDataHandler for AddFixedFramePoseDataHandler {
    type Request = proto::AddFixedFramePoseDataRequest;

    fn on_sensor_data(
        context: &dyn MapBuilderContextInterface,
        request: &proto::AddFixedFramePoseDataRequest,
    ) {
        let sensor_metadata = request.sensor_metadata();
        let payload = request.fixed_frame_pose_data.unwrap_or_default();
        // The `BlockingQueue` returned by `sensor_data_queue()` is already
        // thread-safe.
        context.enqueue_sensor_data(
            sensor_metadata.trajectory_id,
            make_dispatchable(
                &sensor_metadata.sensor_id,
                fixed_frame_pose_data::from_proto(&payload),
            ),
        );
        // The `BlockingQueue` in `LocalTrajectoryUploader` is thread-safe.
        if let Some(uploader) = context.local_trajectory_uploader() {
            uploader.enqueue_sensor_data(Box::new(proto::SensorData {
                sensor_metadata: Some(sensor_metadata),
                sensor_data: Some(proto::sensor_data::SensorData::FixedFramePoseData(payload)),
            }));
        }
    }
}
