//! Port of `cartographer/cloud/internal/handlers/`: one module per C++
//! `RpcHandler`, plus the `tonic` service ([`MapBuilderServiceHandlers`])
//! that dispatches the RPCs to them (the C++ `RegisterHandler<...>()` calls in
//! the `MapBuilderServer` constructor).
//!
//! Handlers that call into the (blocking, CPU-bound) map builder run on
//! tokio's blocking thread pool.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::internal::async_grpc::MAX_MESSAGE_SIZE;
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;
use crate::proto::map_builder_service_server::{MapBuilderService, MapBuilderServiceServer};

pub mod add_fixed_frame_pose_data_handler;
pub mod add_imu_data_handler;
pub mod add_landmark_data_handler;
pub mod add_odometry_data_handler;
pub mod add_rangefinder_data_handler;
pub mod add_sensor_data_batch_handler;
pub mod add_sensor_data_handler_base;
pub mod add_trajectory_handler;
pub mod delete_trajectory_handler;
pub mod finish_trajectory_handler;
pub mod get_all_submap_poses;
pub mod get_constraints_handler;
pub mod get_landmark_poses_handler;
pub mod get_local_to_global_transform_handler;
pub mod get_submap_handler;
pub mod get_trajectory_node_poses_handler;
pub mod get_trajectory_states_handler;
pub mod is_trajectory_finished_handler;
pub mod is_trajectory_frozen_handler;
pub mod load_state_from_file_handler;
pub mod load_state_handler;
pub mod receive_global_slam_optimizations_handler;
pub mod receive_local_slam_results_handler;
pub mod run_final_optimization_handler;
pub mod set_landmark_pose_handler;
pub mod write_state_handler;
pub mod write_state_to_file_handler;

use add_sensor_data_handler_base::handle_add_sensor_data_stream;

/// A server stream fed by a subscription callback. Dropping it (the RPC
/// finished or the client went away) runs `on_finish`, which removes the
/// subscription (C++ `RpcHandler::OnFinish`).
pub struct SubscriptionStream<T> {
    receiver: mpsc::UnboundedReceiver<Result<T, Status>>,
    on_finish: Option<Box<dyn FnOnce() + Send>>,
}

impl<T> SubscriptionStream<T> {
    /// Creates the stream.
    pub fn new(
        receiver: mpsc::UnboundedReceiver<Result<T, Status>>,
        on_finish: Box<dyn FnOnce() + Send>,
    ) -> Self {
        Self {
            receiver,
            on_finish: Some(on_finish),
        }
    }
}

impl<T> Stream for SubscriptionStream<T> {
    type Item = Result<T, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().receiver.poll_recv(cx)
    }
}

impl<T> Drop for SubscriptionStream<T> {
    fn drop(&mut self) {
        if let Some(on_finish) = self.on_finish.take() {
            on_finish();
        }
    }
}

/// Runs `f` with the context on the blocking thread pool. A panic in `f`
/// (a C++ `CHECK` failure) becomes an `INTERNAL` status.
async fn run_blocking<T, F>(
    context: &Arc<dyn MapBuilderContextInterface>,
    f: F,
) -> Result<T, Status>
where
    T: Send + 'static,
    F: FnOnce(&dyn MapBuilderContextInterface) -> Result<T, Status> + Send + 'static,
{
    let context = Arc::clone(context);
    match tokio::task::spawn_blocking(move || f(context.as_ref())).await {
        Ok(result) => result,
        Err(join_error) => {
            let message = match join_error.try_into_panic() {
                Ok(panic) => panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "handler panicked".to_string()),
                Err(e) => e.to_string(),
            };
            log::error!("Handler failed: {message}");
            Err(Status::internal(message))
        }
    }
}

/// The `MapBuilderService` implementation dispatching to the handler
/// modules.
pub struct MapBuilderServiceHandlers {
    context: Arc<dyn MapBuilderContextInterface>,
}

impl MapBuilderServiceHandlers {
    /// Creates the service for `context`.
    pub fn new(context: Arc<dyn MapBuilderContextInterface>) -> Self {
        Self { context }
    }
}

/// Creates the gRPC service with the 100 MB message size limits of the C++
/// server.
pub fn create_map_builder_service(
    context: Arc<dyn MapBuilderContextInterface>,
) -> MapBuilderServiceServer<MapBuilderServiceHandlers> {
    MapBuilderServiceServer::new(MapBuilderServiceHandlers::new(context))
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
}

#[tonic::async_trait]
impl MapBuilderService for MapBuilderServiceHandlers {
    async fn add_trajectory(
        &self,
        request: Request<proto::AddTrajectoryRequest>,
    ) -> Result<Response<proto::AddTrajectoryResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            add_trajectory_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn add_sensor_data_batch(
        &self,
        request: Request<proto::AddSensorDataBatchRequest>,
    ) -> Result<Response<()>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            add_sensor_data_batch_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn add_odometry_data(
        &self,
        request: Request<Streaming<proto::AddOdometryDataRequest>>,
    ) -> Result<Response<()>, Status> {
        handle_add_sensor_data_stream::<add_odometry_data_handler::AddOdometryDataHandler>(
            Arc::clone(&self.context),
            request.into_inner(),
        )
        .await
        .map(Response::new)
    }

    async fn add_imu_data(
        &self,
        request: Request<Streaming<proto::AddImuDataRequest>>,
    ) -> Result<Response<()>, Status> {
        handle_add_sensor_data_stream::<add_imu_data_handler::AddImuDataHandler>(
            Arc::clone(&self.context),
            request.into_inner(),
        )
        .await
        .map(Response::new)
    }

    async fn add_rangefinder_data(
        &self,
        request: Request<Streaming<proto::AddRangefinderDataRequest>>,
    ) -> Result<Response<()>, Status> {
        handle_add_sensor_data_stream::<add_rangefinder_data_handler::AddRangefinderDataHandler>(
            Arc::clone(&self.context),
            request.into_inner(),
        )
        .await
        .map(Response::new)
    }

    async fn add_fixed_frame_pose_data(
        &self,
        request: Request<Streaming<proto::AddFixedFramePoseDataRequest>>,
    ) -> Result<Response<()>, Status> {
        handle_add_sensor_data_stream::<
            add_fixed_frame_pose_data_handler::AddFixedFramePoseDataHandler,
        >(Arc::clone(&self.context), request.into_inner())
        .await
        .map(Response::new)
    }

    async fn add_landmark_data(
        &self,
        request: Request<Streaming<proto::AddLandmarkDataRequest>>,
    ) -> Result<Response<()>, Status> {
        handle_add_sensor_data_stream::<add_landmark_data_handler::AddLandmarkDataHandler>(
            Arc::clone(&self.context),
            request.into_inner(),
        )
        .await
        .map(Response::new)
    }

    type ReceiveLocalSlamResultsStream = SubscriptionStream<proto::ReceiveLocalSlamResultsResponse>;

    async fn receive_local_slam_results(
        &self,
        request: Request<proto::ReceiveLocalSlamResultsRequest>,
    ) -> Result<Response<Self::ReceiveLocalSlamResultsStream>, Status> {
        Ok(Response::new(
            receive_local_slam_results_handler::on_request(
                Arc::clone(&self.context),
                request.get_ref(),
            ),
        ))
    }

    type ReceiveGlobalSlamOptimizationsStream =
        SubscriptionStream<proto::ReceiveGlobalSlamOptimizationsResponse>;

    async fn receive_global_slam_optimizations(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::ReceiveGlobalSlamOptimizationsStream>, Status> {
        Ok(Response::new(
            receive_global_slam_optimizations_handler::on_request(Arc::clone(&self.context)),
        ))
    }

    async fn finish_trajectory(
        &self,
        request: Request<proto::FinishTrajectoryRequest>,
    ) -> Result<Response<()>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            finish_trajectory_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn delete_trajectory(
        &self,
        request: Request<proto::DeleteTrajectoryRequest>,
    ) -> Result<Response<()>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            delete_trajectory_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn get_submap(
        &self,
        request: Request<proto::GetSubmapRequest>,
    ) -> Result<Response<proto::GetSubmapResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            get_submap_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn get_trajectory_node_poses(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::GetTrajectoryNodePosesResponse>, Status> {
        run_blocking(&self.context, get_trajectory_node_poses_handler::on_request)
            .await
            .map(Response::new)
    }

    async fn get_trajectory_states(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::GetTrajectoryStatesResponse>, Status> {
        run_blocking(&self.context, get_trajectory_states_handler::on_request)
            .await
            .map(Response::new)
    }

    async fn get_landmark_poses(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::GetLandmarkPosesResponse>, Status> {
        run_blocking(&self.context, get_landmark_poses_handler::on_request)
            .await
            .map(Response::new)
    }

    async fn get_all_submap_poses(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::GetAllSubmapPosesResponse>, Status> {
        run_blocking(&self.context, get_all_submap_poses::on_request)
            .await
            .map(Response::new)
    }

    async fn get_local_to_global_transform(
        &self,
        request: Request<proto::GetLocalToGlobalTransformRequest>,
    ) -> Result<Response<proto::GetLocalToGlobalTransformResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            get_local_to_global_transform_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn get_constraints(
        &self,
        _request: Request<()>,
    ) -> Result<Response<proto::GetConstraintsResponse>, Status> {
        run_blocking(&self.context, get_constraints_handler::on_request)
            .await
            .map(Response::new)
    }

    async fn set_landmark_pose(
        &self,
        request: Request<proto::SetLandmarkPoseRequest>,
    ) -> Result<Response<()>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            set_landmark_pose_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn is_trajectory_finished(
        &self,
        request: Request<proto::IsTrajectoryFinishedRequest>,
    ) -> Result<Response<proto::IsTrajectoryFinishedResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            is_trajectory_finished_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn is_trajectory_frozen(
        &self,
        request: Request<proto::IsTrajectoryFrozenRequest>,
    ) -> Result<Response<proto::IsTrajectoryFrozenResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            is_trajectory_frozen_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    async fn run_final_optimization(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        run_blocking(&self.context, run_final_optimization_handler::on_request)
            .await
            .map(Response::new)
    }

    async fn load_state(
        &self,
        request: Request<Streaming<proto::LoadStateRequest>>,
    ) -> Result<Response<proto::LoadStateResponse>, Status> {
        let mut stream = request.into_inner();
        let mut handler = load_state_handler::LoadStateHandler::default();
        while let Some(request) = stream.message().await? {
            handler.on_request(&request)?;
        }
        run_blocking(&self.context, move |context| handler.on_reads_done(context))
            .await
            .map(Response::new)
    }

    async fn load_state_from_file(
        &self,
        request: Request<proto::LoadStateFromFileRequest>,
    ) -> Result<Response<proto::LoadStateFromFileResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            load_state_from_file_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }

    type WriteStateStream = UnboundedReceiverStream<Result<proto::WriteStateResponse, Status>>;

    async fn write_state(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WriteStateStream>, Status> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let context = Arc::clone(&self.context);
        // The serialization runs in the background and streams the chunks.
        tokio::task::spawn_blocking(move || {
            let error_sender = sender.clone();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                write_state_handler::on_request(context.as_ref(), sender)
            }))
            .is_err()
            {
                let _ = error_sender.send(Err(Status::internal("WriteState failed")));
            }
        });
        Ok(Response::new(UnboundedReceiverStream::new(receiver)))
    }

    async fn write_state_to_file(
        &self,
        request: Request<proto::WriteStateToFileRequest>,
    ) -> Result<Response<proto::WriteStateToFileResponse>, Status> {
        let request = request.into_inner();
        run_blocking(&self.context, move |context| {
            write_state_to_file_handler::on_request(context, &request)
        })
        .await
        .map(Response::new)
    }
}
