//! Port of `cartographer/cloud/internal/testing/handler_test` (.h).
//!
//! RUST PORT NOTE: the C++ `HandlerTest` fixture drives a single handler
//! through `async_grpc::testing::RpcHandlerTestServer`. Here the complete
//! `MapBuilderService` is served on an ephemeral localhost port with a
//! [`MockMapBuilderContext`], and tests talk to it through the generated
//! client, which exercises the same handler code plus the wire format.

use std::future::Future;
use std::sync::{Arc, Mutex};

use cartographer::mapping::internal::testing::mock_trajectory_builder::MockTrajectoryBuilder;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Channel;

use crate::internal::async_grpc::{RuntimeHolder, create_channel, create_endpoint};
use crate::internal::handlers::add_sensor_data_handler_base::SensorDataRequest;
use crate::internal::handlers::create_map_builder_service;
use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::testing::fake_map_builder::{FakeMapBuilder, FakePoseGraph, MapBuilderScript};
use crate::internal::testing::mock_local_trajectory_uploader::MockLocalTrajectoryUploader;
use crate::internal::testing::mock_map_builder_context::ContextCall;
use crate::internal::testing::mock_map_builder_context::MockMapBuilderContext;
use crate::internal::testing::test_helpers::DataPredicateEquals;
use crate::proto::map_builder_service_client::MapBuilderServiceClient;

/// Test fixture serving the `MapBuilderService` with a mock context.
pub struct HandlerTest {
    /// The runtime of server and client.
    pub runtime: RuntimeHolder,
    /// The mock execution context.
    pub mock_map_builder_context: Arc<MockMapBuilderContext>,
    /// The uploader installed by [`Self::set_mock_local_trajectory_uploader`].
    pub mock_local_trajectory_uploader: Arc<MockLocalTrajectoryUploader>,
    /// Script of the fake map builder returned by `map_builder()`.
    pub mock_map_builder: Arc<Mutex<MapBuilderScript>>,
    /// The fake pose graph of the fake map builder.
    pub mock_pose_graph: Arc<FakePoseGraph>,
    /// A client connected to the test server.
    pub client: MapBuilderServiceClient<Channel>,
    shutdown_sender: Option<oneshot::Sender<()>>,
}

impl Default for HandlerTest {
    fn default() -> Self {
        Self::new()
    }
}

impl HandlerTest {
    /// C++ `SetUp()`: creates the mocks and starts the test server.
    pub fn new() -> Self {
        Self::with_trajectory_builder(None)
    }

    /// Like [`Self::new`]; `GetTrajectoryBuilder` of the fake map builder
    /// returns `trajectory_builder`.
    pub fn with_trajectory_builder(trajectory_builder: Option<MockTrajectoryBuilder>) -> Self {
        let runtime = RuntimeHolder::new("handler-test", 2);
        let map_builder = FakeMapBuilder::new(trajectory_builder);
        let mock_map_builder = map_builder.script();
        let mock_pose_graph = map_builder.fake_pose_graph();
        let mock_map_builder_context = Arc::new(MockMapBuilderContext::new(Box::new(map_builder)));
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
            .expect("failed to bind an ephemeral port");
        let address = listener.local_addr().unwrap();
        let service = create_map_builder_service(
            Arc::clone(&mock_map_builder_context) as Arc<dyn MapBuilderContextInterface>
        );
        let (shutdown_sender, shutdown_receiver) = oneshot::channel::<()>();
        runtime.handle().spawn(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown_receiver.await;
                }),
        );
        let endpoint = create_endpoint(&address.to_string()).unwrap();
        let client = MapBuilderServiceClient::new(create_channel(runtime.handle(), &endpoint));
        Self {
            runtime,
            mock_map_builder_context,
            mock_local_trajectory_uploader: Arc::new(MockLocalTrajectoryUploader::new()),
            mock_map_builder,
            mock_pose_graph,
            client,
            shutdown_sender: Some(shutdown_sender),
        }
    }

    /// C++ `SetNoLocalTrajectoryUploader()`.
    pub fn set_no_local_trajectory_uploader(&self) {
        self.mock_map_builder_context
            .set_local_trajectory_uploader(None);
    }

    /// C++ `SetMockLocalTrajectoryUploader()`.
    pub fn set_mock_local_trajectory_uploader(&self) {
        self.mock_map_builder_context
            .set_local_trajectory_uploader(Some(Arc::clone(&self.mock_local_trajectory_uploader)));
    }

    /// Runs `f` with a clone of the client on the test runtime.
    pub fn call<T, F, Fut>(&self, f: F) -> T
    where
        F: FnOnce(MapBuilderServiceClient<Channel>) -> Fut,
        Fut: Future<Output = T>,
    {
        self.runtime.block_on(f(self.client.clone()))
    }

    /// The common part of the `Add*DataHandlerTest`s: streams `request`
    /// once through `rpc` (C++ `SendWrite` + `SendWritesDone` +
    /// `SendFinish`) and checks the C++ expectations
    /// `CheckClientIdForTrajectory(client_id, trajectory_id)` and
    /// `DoEnqueueSensorData(trajectory_id, Pointee(Truly(
    /// BuildDataPredicateEquals(request))))`.
    pub fn send_sensor_data_and_expect_enqueued<R, F, Fut>(&self, request: &R, rpc: F)
    where
        R: SensorDataRequest + DataPredicateEquals + Clone,
        F: FnOnce(MapBuilderServiceClient<Channel>, Vec<R>) -> Fut,
        Fut: Future<Output = Result<tonic::Response<()>, tonic::Status>>,
    {
        let sensor_metadata = request.sensor_metadata();
        self.call(|client| rpc(client, vec![request.clone()]))
            .expect("Add*Data RPC failed");
        assert!(self.mock_map_builder_context.calls().contains(
            &ContextCall::CheckClientIdForTrajectory(
                sensor_metadata.client_id.clone(),
                sensor_metadata.trajectory_id,
            )
        ));
        let enqueued = self.mock_map_builder_context.enqueued_sensor_data();
        assert_eq!(enqueued.len(), 1, "{enqueued:?}");
        assert_eq!(enqueued[0].0, sensor_metadata.trajectory_id);
        assert!(request.data_equals(&enqueued[0].1), "{:?}", enqueued[0].1);
    }
}

impl Drop for HandlerTest {
    fn drop(&mut self) {
        if let Some(shutdown_sender) = self.shutdown_sender.take() {
            let _ = shutdown_sender.send(());
        }
    }
}
