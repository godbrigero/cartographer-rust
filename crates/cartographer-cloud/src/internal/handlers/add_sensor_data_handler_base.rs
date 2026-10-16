//! Port of `cartographer/cloud/internal/handlers/add_sensor_data_handler_base` (.h).
//!
//! RUST PORT NOTE: the CRTP-like `AddSensorDataHandlerBase<Signature>` class
//! becomes the [`AddSensorDataHandler`] trait plus the generic stream driver
//! [`handle_add_sensor_data_stream`] (`OnRequest` per message, `OnReadsDone`
//! sends the empty response).

use std::sync::Arc;

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// A client-streamed `Add*DataRequest` carrying `sensor_metadata`.
pub trait SensorDataRequest: prost::Message + Default + Send + 'static {
    /// The `sensor_metadata` field (default instance if unset).
    fn sensor_metadata(&self) -> proto::SensorMetadata;
}

macro_rules! impl_sensor_data_request {
    ($($request:ty),*) => {$(
        impl SensorDataRequest for $request {
            fn sensor_metadata(&self) -> proto::SensorMetadata {
                self.sensor_metadata.clone().unwrap_or_default()
            }
        }
    )*};
}

impl_sensor_data_request!(
    proto::AddImuDataRequest,
    proto::AddOdometryDataRequest,
    proto::AddRangefinderDataRequest,
    proto::AddFixedFramePoseDataRequest,
    proto::AddLandmarkDataRequest
);

/// A handler of a client-streaming `Add*Data` RPC.
pub trait AddSensorDataHandler {
    /// The streamed request message.
    type Request: SensorDataRequest;

    /// Processes one message of a known trajectory (C++ `OnSensorData`).
    fn on_sensor_data(context: &dyn MapBuilderContextInterface, request: &Self::Request);
}

/// Checks the client id of a single streamed message and processes it (C++
/// `AddSensorDataHandlerBase::OnRequest`). Returns `NOT_FOUND` for an unknown
/// trajectory, which finishes the RPC.
pub fn on_request<H: AddSensorDataHandler>(
    context: &dyn MapBuilderContextInterface,
    request: &H::Request,
) -> Result<(), Status> {
    let sensor_metadata = request.sensor_metadata();
    if !context
        .check_client_id_for_trajectory(&sensor_metadata.client_id, sensor_metadata.trajectory_id)
    {
        log::error!(
            "Unknown trajectory with ID {} and client_id {}",
            sensor_metadata.trajectory_id,
            sensor_metadata.client_id
        );
        return Err(Status::not_found("Unknown trajectory"));
    }
    H::on_sensor_data(context, request);
    Ok(())
}

/// Drives a client-streaming `Add*Data` RPC: handles every message, and
/// answers with `Empty` once the client is done writing (C++
/// `OnReadsDone`).
pub async fn handle_add_sensor_data_stream<H: AddSensorDataHandler>(
    context: Arc<dyn MapBuilderContextInterface>,
    mut stream: tonic::Streaming<H::Request>,
) -> Result<(), Status> {
    while let Some(request) = stream.message().await? {
        on_request::<H>(context.as_ref(), &request)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tonic::Code;

    use crate::internal::testing::handler_test::HandlerTest;
    use crate::proto;

    /// An unknown trajectory finishes the stream with `NOT_FOUND` and
    /// nothing is enqueued.
    #[test]
    fn unknown_trajectory_is_not_found() {
        let test = HandlerTest::new();
        test.mock_map_builder_context
            .set_check_client_id_result(false);
        let request = proto::AddImuDataRequest {
            sensor_metadata: Some(proto::SensorMetadata {
                trajectory_id: 4,
                sensor_id: "imu".to_string(),
                client_id: "CLIENT_ID".to_string(),
            }),
            imu_data: Some(Default::default()),
        };
        let status = test
            .call(|mut client| async move {
                client
                    .add_imu_data(tokio_stream::iter(vec![request.clone(), request]))
                    .await
            })
            .unwrap_err();
        assert_eq!(status.code(), Code::NotFound);
        assert_eq!(status.message(), "Unknown trajectory");
        assert!(
            test.mock_map_builder_context
                .enqueued_sensor_data()
                .is_empty()
        );
    }

    /// Every message of the stream is handled; the response comes after the
    /// client finished writing.
    #[test]
    fn handles_every_streamed_message() {
        let test = HandlerTest::new();
        let requests: Vec<_> = (0..5)
            .map(|i| proto::AddImuDataRequest {
                sensor_metadata: Some(proto::SensorMetadata {
                    trajectory_id: 2,
                    sensor_id: "imu".to_string(),
                    client_id: "CLIENT_ID".to_string(),
                }),
                imu_data: Some(cartographer::sensor::proto::ImuData {
                    timestamp: i,
                    ..Default::default()
                }),
            })
            .collect();
        test.call(
            |mut client| async move { client.add_imu_data(tokio_stream::iter(requests)).await },
        )
        .unwrap();
        assert_eq!(
            test.mock_map_builder_context.enqueued_sensor_data().len(),
            5
        );
    }
}
