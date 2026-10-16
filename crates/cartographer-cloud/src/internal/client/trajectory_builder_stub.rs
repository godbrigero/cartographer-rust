//! Port of `cartographer/cloud/internal/client/trajectory_builder_stub` (.h, .cc).

use std::future::Future;
use std::sync::Arc;
use std::thread::JoinHandle;

use cartographer::common::time::from_universal;
use cartographer::mapping::id::NodeId;
use cartographer::mapping::internal::local_slam_result_data::LocalSlamResultData;
use cartographer::mapping::trajectory_builder_interface::{
    InsertionResult, LocalSlamResultCallback, TrajectoryBuilderInterface,
};
use cartographer::mapping::trajectory_node::TrajectoryNodeData;
use cartographer::sensor::fixed_frame_pose_data::{self, FixedFramePoseData};
use cartographer::sensor::imu_data::{self, ImuData};
use cartographer::sensor::landmark_data::{self, LandmarkData};
use cartographer::sensor::odometry_data::{self, OdometryData};
use cartographer::sensor::range_data;
use cartographer::sensor::timed_point_cloud_data::{self, TimedPointCloudData};
use cartographer::transform::transform::to_rigid3;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;
use tonic::{Response, Status, Streaming};

use crate::internal::async_grpc::RuntimeHolder;
use crate::internal::sensor::serialization::{
    create_add_fixed_frame_pose_data_request, create_add_imu_data_request,
    create_add_landmark_data_request, create_add_odometry_data_request,
    create_add_range_finder_data_request,
};
use crate::proto;
use crate::proto::map_builder_service_client::MapBuilderServiceClient;

/// Number of buffered messages per sensor stream before `add_*` blocks.
const STREAM_BUFFER_SIZE: usize = 64;

/// An open client-streaming `Add*Data` RPC (C++
/// `async_grpc::Client<Add*DataSignature>` in streaming mode).
struct StreamWriter<R> {
    sender: Option<mpsc::Sender<R>>,
    response: tokio::task::JoinHandle<Result<Response<()>, Status>>,
}

impl<R: Send + 'static> StreamWriter<R> {
    /// Starts the RPC; `rpc` gets the request stream.
    fn new<F, Fut>(runtime: &RuntimeHolder, rpc: F) -> Self
    where
        F: FnOnce(ReceiverStream<R>) -> Fut,
        Fut: Future<Output = Result<Response<()>, Status>> + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel(STREAM_BUFFER_SIZE);
        let response = runtime.handle().spawn(rpc(ReceiverStream::new(receiver)));
        Self {
            sender: Some(sender),
            response,
        }
    }

    /// Writes one message (C++ `client->Write(request)`); blocks while the
    /// buffer is full. Returns false if the RPC already ended.
    fn write(&self, request: R) -> bool {
        self.sender
            .as_ref()
            .is_some_and(|sender| sender.blocking_send(request).is_ok())
    }

    /// Half-closes the stream and waits for the response (C++
    /// `StreamWritesDone()` + `StreamFinish()`).
    fn finish(mut self, runtime: &RuntimeHolder) -> Result<(), Status> {
        self.sender = None;
        match runtime.block_on(self.response) {
            Ok(result) => result.map(|_| ()),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// A `TrajectoryBuilderInterface` that streams sensor data to a remote map
/// builder server and (if a callback is given) receives the local SLAM
/// results on a reader thread.
///
/// One client-streaming RPC per sensor type is opened lazily on first use and
/// closed on drop. Like all stub methods, the `add_*` methods block and must
/// not be called from within an async context.
pub struct TrajectoryBuilderStub {
    runtime: Arc<RuntimeHolder>,
    client: MapBuilderServiceClient<Channel>,
    trajectory_id: i32,
    client_id: String,
    add_rangefinder_client: Option<StreamWriter<proto::AddRangefinderDataRequest>>,
    add_imu_client: Option<StreamWriter<proto::AddImuDataRequest>>,
    add_odometry_client: Option<StreamWriter<proto::AddOdometryDataRequest>>,
    add_fixed_frame_pose_client: Option<StreamWriter<proto::AddFixedFramePoseDataRequest>>,
    add_landmark_client: Option<StreamWriter<proto::AddLandmarkDataRequest>>,
    receive_local_slam_results_thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for TrajectoryBuilderStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrajectoryBuilderStub")
            .field("trajectory_id", &self.trajectory_id)
            .field("client_id", &self.client_id)
            .finish_non_exhaustive()
    }
}

impl TrajectoryBuilderStub {
    /// Creates the stub. If `local_slam_result_callback` is set, subscribes
    /// to the local SLAM results of the trajectory before returning and
    /// starts the reader thread that invokes the callback.
    ///
    /// RUST PORT NOTE: the subscription RPC is established synchronously
    /// (the C++ starts it asynchronously), so no results are missed by data
    /// sent right after construction.
    pub fn new(
        runtime: Arc<RuntimeHolder>,
        client: MapBuilderServiceClient<Channel>,
        trajectory_id: i32,
        client_id: &str,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> Self {
        let receive_local_slam_results_thread = local_slam_result_callback.map(|callback| {
            let mut subscription_client = client.clone();
            let stream = runtime
                .block_on(subscription_client.receive_local_slam_results(
                    proto::ReceiveLocalSlamResultsRequest { trajectory_id },
                ))
                .map(Response::into_inner);
            let runtime = Arc::clone(&runtime);
            std::thread::Builder::new()
                .name("cartographer-local-slam-results".to_string())
                .spawn(move || match stream {
                    Ok(stream) => Self::run_local_slam_results_reader(&runtime, stream, &callback),
                    Err(status) => {
                        log::error!("ReceiveLocalSlamResults failed: {}", status.message())
                    }
                })
                .expect("failed to spawn the local SLAM results reader thread")
        });
        Self {
            runtime,
            client,
            trajectory_id,
            client_id: client_id.to_string(),
            add_rangefinder_client: None,
            add_imu_client: None,
            add_odometry_client: None,
            add_fixed_frame_pose_client: None,
            add_landmark_client: None,
            receive_local_slam_results_thread,
        }
    }

    /// Reads the local SLAM results stream until it ends and invokes the
    /// callback for every result (outside of the async runtime).
    fn run_local_slam_results_reader(
        runtime: &RuntimeHolder,
        mut stream: Streaming<proto::ReceiveLocalSlamResultsResponse>,
        local_slam_result_callback: &LocalSlamResultCallback,
    ) {
        while let Ok(Some(response)) = runtime.block_on(stream.message()) {
            let trajectory_id = response.trajectory_id;
            let time = from_universal(response.timestamp);
            let local_pose = to_rigid3(&response.local_pose.unwrap_or_default());
            let range_data = range_data::from_proto(&response.range_data.unwrap_or_default());
            // RUST PORT NOTE: the C++ `InsertionResult` built here only has a
            // node id; the Rust struct also requires the constant data, which
            // is left empty, and no insertion submaps.
            let insertion_result = response.insertion_result.map(|insertion_result| {
                let node_id = insertion_result.node_id.unwrap_or_default();
                InsertionResult {
                    node_id: NodeId::new(node_id.trajectory_id, node_id.node_index),
                    constant_data: Arc::new(empty_node_data()),
                    insertion_submaps: Vec::new(),
                }
            });
            local_slam_result_callback(
                trajectory_id,
                time,
                local_pose,
                range_data,
                insertion_result,
            );
        }
    }
}

fn empty_node_data() -> TrajectoryNodeData {
    TrajectoryNodeData {
        time: from_universal(0),
        gravity_alignment: nalgebra::Quaternion::identity(),
        filtered_gravity_aligned_point_cloud: Default::default(),
        high_resolution_point_cloud: Default::default(),
        low_resolution_point_cloud: Default::default(),
        rotational_scan_matcher_histogram: nalgebra::DVector::zeros(0),
        local_pose: Default::default(),
    }
}

impl TrajectoryBuilderInterface for TrajectoryBuilderStub {
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) {
        let writer = self.add_rangefinder_client.get_or_insert_with(|| {
            let mut client = self.client.clone();
            StreamWriter::new(&self.runtime, |stream| async move {
                client.add_rangefinder_data(stream).await
            })
        });
        writer.write(create_add_range_finder_data_request(
            sensor_id,
            self.trajectory_id,
            &self.client_id,
            &timed_point_cloud_data::to_proto(timed_point_cloud_data),
        ));
    }

    fn add_imu_data(&mut self, sensor_id: &str, imu_data: &ImuData) {
        let writer = self.add_imu_client.get_or_insert_with(|| {
            let mut client = self.client.clone();
            StreamWriter::new(&self.runtime, |stream| async move {
                client.add_imu_data(stream).await
            })
        });
        writer.write(create_add_imu_data_request(
            sensor_id,
            self.trajectory_id,
            &self.client_id,
            &imu_data::to_proto(imu_data),
        ));
    }

    fn add_odometry_data(&mut self, sensor_id: &str, odometry_data: &OdometryData) {
        let writer = self.add_odometry_client.get_or_insert_with(|| {
            let mut client = self.client.clone();
            StreamWriter::new(&self.runtime, |stream| async move {
                client.add_odometry_data(stream).await
            })
        });
        writer.write(create_add_odometry_data_request(
            sensor_id,
            self.trajectory_id,
            &self.client_id,
            &odometry_data::to_proto(odometry_data),
        ));
    }

    fn add_fixed_frame_pose_data(
        &mut self,
        sensor_id: &str,
        fixed_frame_pose: &FixedFramePoseData,
    ) {
        let writer = self.add_fixed_frame_pose_client.get_or_insert_with(|| {
            let mut client = self.client.clone();
            StreamWriter::new(&self.runtime, |stream| async move {
                client.add_fixed_frame_pose_data(stream).await
            })
        });
        writer.write(create_add_fixed_frame_pose_data_request(
            sensor_id,
            self.trajectory_id,
            &self.client_id,
            &fixed_frame_pose_data::to_proto(fixed_frame_pose),
        ));
    }

    fn add_landmark_data(&mut self, sensor_id: &str, landmark_data: &LandmarkData) {
        let writer = self.add_landmark_client.get_or_insert_with(|| {
            let mut client = self.client.clone();
            StreamWriter::new(&self.runtime, |stream| async move {
                client.add_landmark_data(stream).await
            })
        });
        writer.write(create_add_landmark_data_request(
            sensor_id,
            self.trajectory_id,
            &self.client_id,
            &landmark_data::to_proto(landmark_data),
        ));
    }

    fn add_local_slam_result_data(
        &mut self,
        _local_slam_result_data: Box<dyn LocalSlamResultData>,
    ) {
        panic!("Not implemented");
    }
}

impl Drop for TrajectoryBuilderStub {
    /// Joins the reader thread, then half-closes all open sensor streams and
    /// waits for their responses.
    ///
    /// RUST PORT NOTE: the C++ `CHECK`s that every stream finished OK; a
    /// panic in `drop` would abort, so failures are logged instead.
    fn drop(&mut self) {
        if let Some(thread) = self.receive_local_slam_results_thread.take() {
            let _ = thread.join();
        }
        fn finish<R: Send + 'static>(
            writer: Option<StreamWriter<R>>,
            runtime: &RuntimeHolder,
            name: &str,
        ) {
            if let Some(writer) = writer
                && let Err(status) = writer.finish(runtime)
            {
                log::error!("{name} stream finished with an error: {status}");
            }
        }
        finish(
            self.add_rangefinder_client.take(),
            &self.runtime,
            "AddRangefinderData",
        );
        finish(self.add_imu_client.take(), &self.runtime, "AddImuData");
        finish(
            self.add_odometry_client.take(),
            &self.runtime,
            "AddOdometryData",
        );
        finish(
            self.add_landmark_client.take(),
            &self.runtime,
            "AddLandmarkData",
        );
        finish(
            self.add_fixed_frame_pose_client.take(),
            &self.runtime,
            "AddFixedFramePoseData",
        );
    }
}
