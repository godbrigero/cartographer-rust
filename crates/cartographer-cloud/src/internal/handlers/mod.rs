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

/// Tests of the handlers without a C++ handler test (the C++ covers them only
/// through `client_server_test.cc` with a real map builder).
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use cartographer::common::time::from_universal;
    use cartographer::mapping::id::{NodeId, SubmapId};
    use cartographer::mapping::pose_graph_interface::{
        Constraint, Pose, SubmapPose, Tag, TrajectoryState,
    };
    use cartographer::mapping::proto as mapping_proto;
    use cartographer::mapping::trajectory_builder_interface::InsertionResult;
    use cartographer::mapping::trajectory_node::{ConstantPoseData, TrajectoryNodePose};
    use cartographer::sensor::range_data::RangeData;
    use cartographer::transform::rigid_transform::Rigid3d;
    use cartographer::transform::transform::to_proto;
    use nalgebra::{Quaternion, Vector3};
    use tonic::Code;

    use crate::internal::map_builder_context_interface::LocalSlamResult;
    use crate::internal::map_builder_server::lock;
    use crate::internal::testing::fake_map_builder::{MapBuilderCall, PoseGraphCall};
    use crate::internal::testing::handler_test::HandlerTest;
    use crate::internal::testing::mock_map_builder_context::ContextCall;
    use crate::internal::testing::test_helpers::DispatchedData;
    use crate::proto;
    use crate::proto::load_state_request::StateChunk;
    use crate::proto::write_state_response::StateChunk as WriteStateChunk;

    fn metadata(trajectory_id: i32, sensor_id: &str) -> Option<proto::SensorMetadata> {
        Some(proto::SensorMetadata {
            trajectory_id,
            sensor_id: sensor_id.to_string(),
            client_id: "CLIENT_ID".to_string(),
        })
    }

    fn pose(x: f64) -> Rigid3d {
        Rigid3d::new(Vector3::new(x, 2., 3.), Quaternion::new(1., 0., 0., 0.))
    }

    fn wait_until(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn finish_trajectory_unknown_is_not_found() {
        let test = HandlerTest::new();
        test.mock_map_builder_context
            .set_check_client_id_result(false);
        let status = test
            .call(|mut client| async move {
                client
                    .finish_trajectory(proto::FinishTrajectoryRequest {
                        trajectory_id: 3,
                        client_id: "CLIENT_ID".to_string(),
                    })
                    .await
            })
            .unwrap_err();
        assert_eq!(status.code(), Code::NotFound);
        assert!(lock(&test.mock_map_builder).calls.is_empty());
    }

    #[test]
    fn finish_trajectory_notifies_subscribers_and_uplink() {
        let test = HandlerTest::new();
        test.set_mock_local_trajectory_uploader();
        test.call(|mut client| async move {
            client
                .finish_trajectory(proto::FinishTrajectoryRequest {
                    trajectory_id: 3,
                    client_id: "CLIENT_ID".to_string(),
                })
                .await
        })
        .unwrap();
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::FinishTrajectory(3)]
        );
        assert!(
            test.mock_map_builder_context
                .calls()
                .contains(&ContextCall::NotifyFinishTrajectory(3))
        );
        assert_eq!(
            test.mock_local_trajectory_uploader
                .finish_trajectory_calls(),
            vec![("CLIENT_ID".to_string(), 3)]
        );
    }

    #[test]
    fn delete_trajectory() {
        let test = HandlerTest::new();
        let request = proto::DeleteTrajectoryRequest {
            trajectory_id: 5,
            client_id: "CLIENT_ID".to_string(),
        };
        let r = request.clone();
        test.call(|mut client| async move { client.delete_trajectory(r).await })
            .unwrap();
        assert_eq!(
            test.mock_pose_graph.calls(),
            vec![PoseGraphCall::DeleteTrajectory(5)]
        );
        test.mock_map_builder_context
            .set_check_client_id_result(false);
        let status = test
            .call(|mut client| async move { client.delete_trajectory(request).await })
            .unwrap_err();
        assert_eq!(status.code(), Code::NotFound);
    }

    #[test]
    fn add_sensor_data_batch_enqueues_all_types() {
        use proto::sensor_data::SensorData as Case;
        let test = HandlerTest::new();
        let local_slam_result_data = mapping_proto::LocalSlamResultData {
            timestamp: 6,
            ..Default::default()
        };
        let request = proto::AddSensorDataBatchRequest {
            sensor_data: vec![
                proto::SensorData {
                    sensor_metadata: metadata(1, "odometry"),
                    sensor_data: Some(Case::OdometryData(Default::default())),
                },
                proto::SensorData {
                    sensor_metadata: metadata(1, "imu"),
                    sensor_data: Some(Case::ImuData(Default::default())),
                },
                proto::SensorData {
                    sensor_metadata: metadata(1, "range"),
                    sensor_data: Some(Case::TimedPointCloudData(Default::default())),
                },
                proto::SensorData {
                    sensor_metadata: metadata(1, "fixed"),
                    sensor_data: Some(Case::FixedFramePoseData(Default::default())),
                },
                proto::SensorData {
                    sensor_metadata: metadata(1, "landmark"),
                    sensor_data: Some(Case::LandmarkData(Default::default())),
                },
                proto::SensorData {
                    sensor_metadata: metadata(1, "local_slam_result_1"),
                    sensor_data: Some(Case::LocalSlamResultData(local_slam_result_data.clone())),
                },
            ],
        };
        test.call(|mut client| async move { client.add_sensor_data_batch(request).await })
            .unwrap();
        let enqueued = test.mock_map_builder_context.enqueued_sensor_data();
        assert_eq!(enqueued.len(), 5);
        assert!(matches!(&enqueued[0].1, DispatchedData::Odometry(id, _) if id == "odometry"));
        assert!(matches!(&enqueued[1].1, DispatchedData::Imu(id, _) if id == "imu"));
        assert!(matches!(&enqueued[2].1, DispatchedData::TimedPointCloud(id, _) if id == "range"));
        assert!(matches!(&enqueued[3].1, DispatchedData::FixedFramePose(id, _) if id == "fixed"));
        assert!(matches!(&enqueued[4].1, DispatchedData::Landmark(id, _) if id == "landmark"));
        assert_eq!(
            test.mock_map_builder_context
                .enqueued_local_slam_result_data(),
            vec![(1, "local_slam_result_1".to_string(), local_slam_result_data)]
        );
    }

    #[test]
    fn add_sensor_data_batch_rejects_unknown_trajectory_and_empty_data() {
        let test = HandlerTest::new();
        let request = proto::AddSensorDataBatchRequest {
            sensor_data: vec![proto::SensorData {
                sensor_metadata: metadata(1, "imu"),
                sensor_data: None,
            }],
        };
        let r = request.clone();
        let status = test
            .call(|mut client| async move { client.add_sensor_data_batch(r).await })
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        test.mock_map_builder_context
            .set_check_client_id_result(false);
        let status = test
            .call(|mut client| async move { client.add_sensor_data_batch(request).await })
            .unwrap_err();
        assert_eq!(status.code(), Code::NotFound);
        assert!(
            test.mock_map_builder_context
                .enqueued_sensor_data()
                .is_empty()
        );
    }

    #[test]
    fn write_state_to_file() {
        let test = HandlerTest::new();
        let status = test
            .call(|mut client| async move {
                client
                    .write_state_to_file(proto::WriteStateToFileRequest {
                        filename: String::new(),
                    })
                    .await
            })
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), "Filename empty.");
        let response = test
            .call(|mut client| async move {
                client
                    .write_state_to_file(proto::WriteStateToFileRequest {
                        filename: "state.pbstream".to_string(),
                    })
                    .await
            })
            .unwrap()
            .into_inner();
        assert!(response.success);
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::SerializeStateToFile(
                false,
                "state.pbstream".to_string()
            )]
        );
    }

    #[test]
    fn write_state_streams_header_and_serialized_data() {
        let test = HandlerTest::new();
        let serialized_data = mapping_proto::SerializedData {
            data: Some(mapping_proto::serialized_data::Data::PoseGraph(
                mapping_proto::PoseGraph::default(),
            )),
        };
        lock(&test.mock_map_builder).serialized_state = vec![serialized_data.clone()];
        let responses = test.call(|mut client| async move {
            let mut stream = client.write_state(()).await.unwrap().into_inner();
            let mut responses = Vec::new();
            while let Some(response) = stream.message().await.unwrap() {
                responses.push(response);
            }
            responses
        });
        assert_eq!(
            responses,
            vec![
                proto::WriteStateResponse {
                    state_chunk: Some(WriteStateChunk::Header(
                        mapping_proto::SerializationHeader { format_version: 2 }
                    )),
                },
                proto::WriteStateResponse {
                    state_chunk: Some(WriteStateChunk::SerializedData(serialized_data)),
                },
            ]
        );
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::SerializeState(false)]
        );
    }

    #[test]
    fn load_state_registers_client_id_for_loaded_trajectories() {
        let test = HandlerTest::new();
        lock(&test.mock_map_builder).trajectory_remapping = [(0, 7)].into_iter().collect();
        let requests = vec![
            proto::LoadStateRequest {
                state_chunk: Some(StateChunk::ClientId("CLIENT_ID".to_string())),
                load_frozen_state: false,
            },
            proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializationHeader(
                    mapping_proto::SerializationHeader { format_version: 2 },
                )),
                load_frozen_state: true,
            },
            proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializedData(Default::default())),
                load_frozen_state: true,
            },
            proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializedData(Default::default())),
                load_frozen_state: true,
            },
        ];
        let response = test
            .call(|mut client| async move { client.load_state(tokio_stream::iter(requests)).await })
            .unwrap()
            .into_inner();
        assert_eq!(
            response
                .trajectory_remapping
                .unwrap()
                .serialized_trajectories_to_trajectories,
            [(0, 7)].into_iter().collect::<BTreeMap<_, _>>()
        );
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::LoadState(true, 3)]
        );
        assert!(test.mock_map_builder_context.calls().contains(
            &ContextCall::RegisterClientIdForTrajectory("CLIENT_ID".to_string(), 7)
        ));
    }

    #[test]
    fn load_state_rejects_empty_chunk() {
        let test = HandlerTest::new();
        let status = test
            .call(|mut client| async move {
                client
                    .load_state(tokio_stream::iter(vec![proto::LoadStateRequest::default()]))
                    .await
            })
            .unwrap_err();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(lock(&test.mock_map_builder).calls.is_empty());
    }

    #[test]
    fn load_state_from_file() {
        let test = HandlerTest::new();
        lock(&test.mock_map_builder).trajectory_remapping = [(1, 4)].into_iter().collect();
        let response = test
            .call(|mut client| async move {
                client
                    .load_state_from_file(proto::LoadStateFromFileRequest {
                        file_path: "map.pbstream".to_string(),
                        client_id: "CLIENT_ID".to_string(),
                        load_frozen_state: true,
                    })
                    .await
            })
            .unwrap()
            .into_inner();
        assert_eq!(
            response
                .trajectory_remapping
                .unwrap()
                .serialized_trajectories_to_trajectories,
            [(1, 4)].into_iter().collect::<BTreeMap<_, _>>()
        );
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::LoadStateFromFile(
                "map.pbstream".to_string(),
                true
            )]
        );
        assert!(test.mock_map_builder_context.calls().contains(
            &ContextCall::RegisterClientIdForTrajectory("CLIENT_ID".to_string(), 4)
        ));
    }

    #[test]
    fn get_submap_reports_error_message() {
        let test = HandlerTest::new();
        lock(&test.mock_map_builder).submap_to_proto_error = "Requested submap 2".to_string();
        let response = test
            .call(|mut client| async move {
                client
                    .get_submap(proto::GetSubmapRequest {
                        submap_id: Some(mapping_proto::SubmapId {
                            trajectory_id: 1,
                            submap_index: 2,
                        }),
                    })
                    .await
            })
            .unwrap()
            .into_inner();
        assert_eq!(response.error_msg, "Requested submap 2");
        assert!(response.submap_query_response.is_some());
        assert_eq!(
            lock(&test.mock_map_builder).calls,
            vec![MapBuilderCall::SubmapToProto(SubmapId::new(1, 2))]
        );
    }

    #[test]
    fn pose_graph_queries() {
        let test = HandlerTest::new();
        let constraint = Constraint {
            submap_id: SubmapId::new(0, 1),
            node_id: NodeId::new(0, 2),
            pose: Pose {
                zbar_ij: pose(4.),
                translation_weight: 5.,
                rotation_weight: 6.,
            },
            tag: Tag::InterSubmap,
        };
        {
            let mut script = test.mock_pose_graph.lock();
            script.trajectory_states = [
                (0, TrajectoryState::Active),
                (1, TrajectoryState::Finished),
                (2, TrajectoryState::Frozen),
                (3, TrajectoryState::Deleted),
            ]
            .into_iter()
            .collect();
            script.submap_poses = vec![(
                SubmapId::new(0, 1),
                SubmapPose {
                    version: 3,
                    pose: pose(1.),
                },
            )];
            script.node_poses = vec![
                (
                    NodeId::new(0, 0),
                    TrajectoryNodePose::new(
                        pose(2.),
                        Some(ConstantPoseData {
                            time: from_universal(42),
                            local_pose: pose(3.),
                        }),
                    ),
                ),
                (NodeId::new(0, 1), TrajectoryNodePose::new(pose(5.), None)),
            ];
            script.constraints = vec![constraint];
            script.local_to_global.insert(1, pose(7.));
        }

        let states = test
            .call(|mut client| async move { client.get_trajectory_states(()).await })
            .unwrap()
            .into_inner()
            .trajectories_state;
        assert_eq!(
            states,
            [(0, 0), (1, 1), (2, 2), (3, 3)].into_iter().collect()
        );

        let submap_poses = test
            .call(|mut client| async move { client.get_all_submap_poses(()).await })
            .unwrap()
            .into_inner()
            .submap_poses;
        assert_eq!(
            submap_poses,
            vec![proto::SubmapPose {
                submap_id: Some(mapping_proto::SubmapId {
                    trajectory_id: 0,
                    submap_index: 1,
                }),
                submap_version: 3,
                global_pose: Some(to_proto(&pose(1.))),
            }]
        );

        let node_poses = test
            .call(|mut client| async move { client.get_trajectory_node_poses(()).await })
            .unwrap()
            .into_inner()
            .node_poses;
        assert_eq!(node_poses.len(), 2);
        assert_eq!(
            node_poses[0].constant_pose_data,
            Some(proto::trajectory_node_pose::ConstantPoseData {
                timestamp: 42,
                local_pose: Some(to_proto(&pose(3.))),
            })
        );
        assert_eq!(node_poses[0].global_pose, Some(to_proto(&pose(2.))));
        assert!(node_poses[1].constant_pose_data.is_none());

        let constraints = test
            .call(|mut client| async move { client.get_constraints(()).await })
            .unwrap()
            .into_inner()
            .constraints;
        assert_eq!(
            cartographer::mapping::pose_graph::from_proto(&constraints),
            vec![constraint]
        );

        let local_to_global = test
            .call(|mut client| async move {
                client
                    .get_local_to_global_transform(proto::GetLocalToGlobalTransformRequest {
                        trajectory_id: 1,
                    })
                    .await
            })
            .unwrap()
            .into_inner()
            .local_to_global;
        assert_eq!(local_to_global, Some(to_proto(&pose(7.))));

        for (trajectory_id, finished, frozen) in
            [(0, false, false), (1, true, false), (2, false, true)]
        {
            let is_finished = test
                .call(|mut client| async move {
                    client
                        .is_trajectory_finished(proto::IsTrajectoryFinishedRequest {
                            trajectory_id,
                        })
                        .await
                })
                .unwrap()
                .into_inner()
                .is_finished;
            let is_frozen = test
                .call(|mut client| async move {
                    client
                        .is_trajectory_frozen(proto::IsTrajectoryFrozenRequest { trajectory_id })
                        .await
                })
                .unwrap()
                .into_inner()
                .is_frozen;
            assert_eq!((is_finished, is_frozen), (finished, frozen));
        }

        test.call(|mut client| async move { client.run_final_optimization(()).await })
            .unwrap();
        assert_eq!(
            test.mock_pose_graph.calls(),
            vec![PoseGraphCall::RunFinalOptimization]
        );
    }

    #[test]
    fn receive_local_slam_results_streams_until_trajectory_finished() {
        let test = HandlerTest::new();
        let mut stream = test
            .call(|mut client| async move {
                client
                    .receive_local_slam_results(proto::ReceiveLocalSlamResultsRequest {
                        trajectory_id: 2,
                    })
                    .await
            })
            .unwrap()
            .into_inner();
        assert_eq!(test.mock_map_builder_context.num_subscriptions(), (1, 0));
        let result = LocalSlamResult {
            trajectory_id: 2,
            time: from_universal(17),
            local_pose: pose(1.),
            range_data: Some(Arc::new(RangeData::default())),
            insertion_result: Some(InsertionResult {
                node_id: NodeId::new(2, 9),
                constant_data: Arc::new(cartographer::mapping::trajectory_node::from_proto(
                    &Default::default(),
                )),
                insertion_submaps: Vec::new(),
            }),
        };
        assert_eq!(
            test.mock_map_builder_context
                .run_local_slam_subscriptions(2, Some(Box::new(result.clone()))),
            vec![true]
        );
        // Other trajectories are not delivered.
        assert!(
            test.mock_map_builder_context
                .run_local_slam_subscriptions(3, Some(Box::new(result)))
                .is_empty()
        );
        let response = test.runtime.block_on(stream.message()).unwrap().unwrap();
        assert_eq!(response.trajectory_id, 2);
        assert_eq!(response.timestamp, 17);
        assert_eq!(response.local_pose, Some(to_proto(&pose(1.))));
        assert!(response.range_data.is_some());
        assert_eq!(
            response.insertion_result,
            Some(proto::LocalSlamInsertionResult {
                node_id: Some(mapping_proto::NodeId {
                    trajectory_id: 2,
                    node_index: 9,
                }),
            })
        );
        // 'None' signals that the trajectory finished: the stream ends.
        test.mock_map_builder_context
            .run_local_slam_subscriptions(2, None);
        assert!(test.runtime.block_on(stream.message()).unwrap().is_none());
        drop(stream);
        // OnFinish removes the subscription.
        wait_until(|| test.mock_map_builder_context.num_subscriptions() == (0, 0));
    }

    #[test]
    fn receive_global_slam_optimizations_streams_until_client_goes_away() {
        let test = HandlerTest::new();
        let mut stream = test
            .call(|mut client| async move { client.receive_global_slam_optimizations(()).await })
            .unwrap()
            .into_inner();
        assert_eq!(test.mock_map_builder_context.num_subscriptions(), (0, 1));
        let submap_ids = [(0, SubmapId::new(0, 3))].into_iter().collect();
        let node_ids = [(0, NodeId::new(0, 8))].into_iter().collect();
        assert_eq!(
            test.mock_map_builder_context
                .run_global_slam_subscriptions(&submap_ids, &node_ids),
            vec![true]
        );
        let response = test.runtime.block_on(stream.message()).unwrap().unwrap();
        assert_eq!(
            response.last_optimized_submap_ids,
            [(
                0,
                mapping_proto::SubmapId {
                    trajectory_id: 0,
                    submap_index: 3,
                }
            )]
            .into_iter()
            .collect()
        );
        assert_eq!(
            response.last_optimized_node_ids,
            [(
                0,
                mapping_proto::NodeId {
                    trajectory_id: 0,
                    node_index: 8,
                }
            )]
            .into_iter()
            .collect()
        );
        drop(stream);
        wait_until(|| test.mock_map_builder_context.num_subscriptions() == (0, 0));
    }
}
