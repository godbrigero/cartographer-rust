//! Port of `cartographer/cloud/client/map_builder_stub` (.h, .cc).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cartographer::io::proto_stream_deserializer::ProtoStreamDeserializer;
use cartographer::io::proto_stream_interface::{
    ProtoStreamReaderInterface, ProtoStreamWriterInterface,
};
use cartographer::mapping::id::SubmapId;
use cartographer::mapping::map_builder_interface::{
    LocalSlamResultCallback, MapBuilderInterface, SensorId,
};
use cartographer::mapping::pose_graph_interface::PoseGraphInterface;
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use tonic::transport::Channel;

use crate::internal::async_grpc::{
    MAX_MESSAGE_SIZE, RetryStrategy, RuntimeHolder, call_with_retry, create_channel,
    create_endpoint, request_with_timeout, wait_for_connected,
};
use crate::internal::client::pose_graph_stub::PoseGraphStub;
use crate::internal::client::trajectory_builder_stub::TrajectoryBuilderStub;
use crate::internal::mapping::serialization::trajectory_remapping_from_proto;
use crate::internal::sensor::serialization;
use crate::proto;
use crate::proto::load_state_request::StateChunk;
use crate::proto::map_builder_service_client::MapBuilderServiceClient;
use crate::proto::write_state_response::StateChunk as WriteStateChunk;

const CHANNEL_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
const RETRY_DELAY_FACTOR: f64 = 2.0;
const MAX_RETRIES: u32 = 5;

/// A `MapBuilderInterface` forwarding to a remote map builder server
/// (`cartographer_grpc_server`).
///
/// RUST PORT NOTE: the Cartographer interfaces are synchronous, so the stub
/// owns a tokio runtime and blocks on every RPC; its methods (and those of the
/// pose graph and trajectory builder stubs) must not be called from within an
/// async context. RPC failures panic like the C++ `CHECK`s.
pub struct MapBuilderStub {
    runtime: Arc<RuntimeHolder>,
    client: MapBuilderServiceClient<Channel>,
    pose_graph_stub: Arc<PoseGraphStub>,
    trajectory_builder_stubs: BTreeMap<i32, TrajectoryBuilderStub>,
    client_id: String,
}

impl std::fmt::Debug for MapBuilderStub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapBuilderStub")
            .field("client_id", &self.client_id)
            .field(
                "trajectory_ids",
                &self.trajectory_builder_stubs.keys().collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl MapBuilderStub {
    /// Connects to `server_address` (`host:port`); panics if no connection
    /// can be established within 10 s (C++ `LOG(FATAL)`).
    pub fn new(server_address: &str, client_id: &str) -> Self {
        Self::try_new(server_address, client_id)
            .unwrap_or_else(|status| panic!("{}", status.message()))
    }

    /// Like [`Self::new`], returning an error instead of panicking.
    ///
    /// RUST PORT NOTE: addition to the C++ API.
    pub fn try_new(server_address: &str, client_id: &str) -> Result<Self, Status> {
        log::info!("Connecting to SLAM process at {server_address} with client_id {client_id}");
        let runtime = Arc::new(RuntimeHolder::new("cartographer-stub", 2));
        let endpoint = create_endpoint(server_address)?;
        if !runtime.block_on(wait_for_connected(
            &endpoint,
            Instant::now() + CHANNEL_TIMEOUT,
        )) {
            return Err(Status::unavailable(format!(
                "Failed to connect to {server_address}"
            )));
        }
        let channel = create_channel(runtime.handle(), &endpoint);
        let client = MapBuilderServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE);
        Ok(Self {
            pose_graph_stub: Arc::new(PoseGraphStub::new(
                Arc::clone(&runtime),
                client.clone(),
                client_id,
            )),
            runtime,
            client,
            trajectory_builder_stubs: BTreeMap::new(),
            client_id: client_id.to_string(),
        })
    }

    fn check<T>(rpc_name: &str, result: Result<tonic::Response<T>, Status>) -> T {
        result
            .unwrap_or_else(|status| panic!("Check failed: {rpc_name} RPC: {status}"))
            .into_inner()
    }
}

impl MapBuilderInterface for MapBuilderStub {
    fn add_trajectory_builder(
        &mut self,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &mapping_proto::TrajectoryBuilderOptions,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> i32 {
        let request = proto::AddTrajectoryRequest {
            expected_sensor_ids: expected_sensor_ids
                .iter()
                .map(serialization::to_proto)
                .collect(),
            trajectory_builder_options: Some(trajectory_options.clone()),
            client_id: self.client_id.clone(),
        };
        let strategy = RetryStrategy::LimitedBackoff {
            min_delay: RETRY_BASE_DELAY,
            backoff_factor: RETRY_DELAY_FACTOR,
            max_attempts: MAX_RETRIES,
        };
        let client = self.client.clone();
        let response = self.runtime.block_on(call_with_retry(
            &strategy,
            || false,
            || {
                let mut client = client.clone();
                let request = request_with_timeout(request.clone(), Some(CHANNEL_TIMEOUT));
                async move { client.add_trajectory(request).await }
            },
        ));
        let trajectory_id = Self::check("AddTrajectory", response).trajectory_id;
        // Construct trajectory builder stub.
        self.trajectory_builder_stubs.insert(
            trajectory_id,
            TrajectoryBuilderStub::new(
                Arc::clone(&self.runtime),
                self.client.clone(),
                trajectory_id,
                &self.client_id,
                local_slam_result_callback,
            ),
        );
        trajectory_id
    }

    fn add_trajectory_for_deserialization(
        &mut self,
        _options_with_sensor_ids_proto: &mapping_proto::TrajectoryBuilderOptionsWithSensorIds,
    ) -> i32 {
        panic!("Not implemented");
    }

    fn get_trajectory_builder(
        &mut self,
        trajectory_id: i32,
    ) -> Option<&mut dyn TrajectoryBuilderInterface> {
        self.trajectory_builder_stubs
            .get_mut(&trajectory_id)
            .map(|stub| stub as &mut dyn TrajectoryBuilderInterface)
    }

    fn finish_trajectory(&mut self, trajectory_id: i32) {
        let request = proto::FinishTrajectoryRequest {
            trajectory_id,
            client_id: self.client_id.clone(),
        };
        let mut client = self.client.clone();
        if let Err(status) = self
            .runtime
            .block_on(async move { client.finish_trajectory(request).await })
        {
            log::error!(
                "Failed to finish trajectory {trajectory_id} for client_id {}: {}",
                self.client_id,
                status.message()
            );
            return;
        }
        self.trajectory_builder_stubs.remove(&trajectory_id);
    }

    fn submap_to_proto(
        &mut self,
        submap_id: &SubmapId,
        submap_query_response: &mut mapping_proto::submap_query::Response,
    ) -> String {
        let request = proto::GetSubmapRequest {
            submap_id: Some(submap_id.to_proto()),
        };
        let mut client = self.client.clone();
        let response = Self::check(
            "GetSubmap",
            self.runtime
                .block_on(async move { client.get_submap(request).await }),
        );
        *submap_query_response = response.submap_query_response.unwrap_or_default();
        response.error_msg
    }

    fn serialize_state(
        &mut self,
        include_unfinished_submaps: bool,
        writer: &mut dyn ProtoStreamWriterInterface,
    ) {
        if include_unfinished_submaps {
            log::warn!(
                "Serializing unfinished submaps is currently unsupported. Proceeding to write \
                 the state without them."
            );
        }
        let mut client = self.client.clone();
        let mut stream = Self::check(
            "WriteState",
            self.runtime
                .block_on(async move { client.write_state(()).await }),
        );
        loop {
            let response = match self.runtime.block_on(stream.message()) {
                Ok(Some(response)) => response,
                Ok(None) => break,
                Err(status) => {
                    log::error!("WriteState stream failed: {status}");
                    break;
                }
            };
            match response.state_chunk {
                Some(WriteStateChunk::Header(header)) => writer.write_proto(&header),
                Some(WriteStateChunk::SerializedData(serialized_data)) => {
                    writer.write_proto(&serialized_data)
                }
                None => panic!("Unhandled message type"),
            }
        }
    }

    fn serialize_state_to_file(
        &mut self,
        include_unfinished_submaps: bool,
        filename: &str,
    ) -> bool {
        if include_unfinished_submaps {
            log::warn!(
                "Serializing unfinished submaps is currently unsupported. Proceeding to write \
                 the state without them."
            );
        }
        let request = proto::WriteStateToFileRequest {
            filename: filename.to_string(),
        };
        let mut client = self.client.clone();
        match self
            .runtime
            .block_on(async move { client.write_state_to_file(request).await })
        {
            Ok(response) => response.into_inner().success,
            Err(status) => {
                log::error!(
                    "WriteStateToFileRequest failed - code: {} reason: {}",
                    status.code(),
                    status.message()
                );
                false
            }
        }
    }

    fn load_state(
        &mut self,
        reader: &mut dyn ProtoStreamReaderInterface,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        let (sender, receiver) = mpsc::channel::<proto::LoadStateRequest>(16);
        let mut client = self.client.clone();
        let response = self
            .runtime
            .handle()
            .spawn(async move { client.load_state(ReceiverStream::new(receiver)).await });
        let write = |request: proto::LoadStateRequest| {
            assert!(
                sender.blocking_send(request).is_ok(),
                "Check failed: client.Write(request)"
            );
        };
        write(proto::LoadStateRequest {
            state_chunk: Some(StateChunk::ClientId(self.client_id.clone())),
            load_frozen_state: false,
        });
        {
            let mut deserializer = ProtoStreamDeserializer::new(&mut *reader);
            // Request with the SerializationHeader proto is sent first.
            write(proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializationHeader(*deserializer.header())),
                load_frozen_state,
            });
            // Request with a PoseGraph proto is sent second.
            write(proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializedData(mapping_proto::SerializedData {
                    data: Some(mapping_proto::serialized_data::Data::PoseGraph(
                        deserializer.pose_graph().clone(),
                    )),
                })),
                load_frozen_state,
            });
            // Request with an AllTrajectoryBuilderOptions should be third.
            write(proto::LoadStateRequest {
                state_chunk: Some(StateChunk::SerializedData(mapping_proto::SerializedData {
                    data: Some(
                        mapping_proto::serialized_data::Data::AllTrajectoryBuilderOptions(
                            deserializer.all_trajectory_builder_options().clone(),
                        ),
                    ),
                })),
                load_frozen_state,
            });
            // Multiple requests with SerializedData are sent after.
            let mut serialized_data = mapping_proto::SerializedData::default();
            while deserializer.read_next_serialized_data(&mut serialized_data) {
                write(proto::LoadStateRequest {
                    state_chunk: Some(StateChunk::SerializedData(std::mem::take(
                        &mut serialized_data,
                    ))),
                    load_frozen_state,
                });
            }
        }
        assert!(reader.eof(), "Check failed: reader->eof()");
        // Half-close the stream (C++ `StreamWritesDone`).
        drop(sender);
        let response = match self.runtime.block_on(response) {
            Ok(response) => response,
            Err(e) => Err(Status::internal(e.to_string())),
        };
        let response = Self::check("LoadState", response);
        trajectory_remapping_from_proto(&response.trajectory_remapping.unwrap_or_default())
    }

    fn load_state_from_file(
        &mut self,
        filename: &str,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        let request = proto::LoadStateFromFileRequest {
            file_path: filename.to_string(),
            client_id: self.client_id.clone(),
            load_frozen_state,
        };
        let mut client = self.client.clone();
        let response = Self::check(
            "LoadStateFromFile",
            self.runtime
                .block_on(async move { client.load_state_from_file(request).await }),
        );
        trajectory_remapping_from_proto(&response.trajectory_remapping.unwrap_or_default())
    }

    fn num_trajectory_builders(&self) -> i32 {
        self.trajectory_builder_stubs.len() as i32
    }

    fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
        Arc::clone(&self.pose_graph_stub) as Arc<dyn PoseGraphInterface>
    }

    fn get_all_trajectory_builder_options(
        &self,
    ) -> &[mapping_proto::TrajectoryBuilderOptionsWithSensorIds] {
        panic!("Not implemented");
    }
}
