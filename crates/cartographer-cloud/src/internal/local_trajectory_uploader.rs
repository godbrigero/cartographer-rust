//! Port of `cartographer/cloud/internal/local_trajectory_uploader` (.h, .cc).
//!
//! Uploads sensor data batches to an uplink server. Gracefully handles
//! interruptions of the connection.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cartographer::common::internal::blocking_queue::BlockingQueue;
use cartographer::common::time::{Duration as CommonDuration, from_milliseconds};
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::{SensorId, SensorType};
use prost::Message;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};

use crate::internal::async_grpc::{
    MAX_MESSAGE_SIZE, RetryStrategy, RuntimeHolder, call_with_retry, create_channel,
    create_endpoint, request_with_timeout, wait_for_connected,
};
use crate::internal::sensor::serialization;
use crate::proto;
use crate::proto::map_builder_service_client::MapBuilderServiceClient;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECTION_RECOVERY_TIMEOUT: Duration = Duration::from_secs(60);
fn pop_timeout() -> CommonDuration {
    from_milliseconds(100)
}

/// The status codes that are considered unrecoverable errors and hence no
/// retries will be attempted by the client.
pub const UNRECOVERABLE_STATUS_CODES: [Code; 4] = [
    Code::DeadlineExceeded,
    Code::NotFound,
    Code::Unavailable,
    Code::Unknown,
];

/// Interface of the uploader of local SLAM results and sensor data to an
/// uplink server.
///
/// RUST PORT NOTE: `grpc::Status` return values become
/// `Result<(), tonic::Status>`. All methods take `&self` (the uploader is
/// shared between the server handlers and its upload thread).
pub trait LocalTrajectoryUploaderInterface: Send + Sync {
    /// Starts the upload thread.
    fn start(&self);

    /// Shuts down the upload thread. This method blocks until the shutdown
    /// is complete.
    fn shutdown(&self);

    /// Enqueue an `Add*DataRequest` message to be uploaded.
    fn enqueue_sensor_data(&self, sensor_data: Box<proto::SensorData>);

    /// Creates a new trajectory with the specified settings in the uplink. An
    /// `Err` indicates that the creation failed.
    fn add_trajectory(
        &self,
        client_id: &str,
        local_trajectory_id: i32,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &mapping_proto::TrajectoryBuilderOptions,
    ) -> Result<(), Status>;

    /// Finishes the uplink trajectory of `local_trajectory_id`.
    fn finish_trajectory(&self, client_id: &str, local_trajectory_id: i32) -> Result<(), Status>;

    /// The sensor id under which local SLAM results of `local_trajectory_id`
    /// are uploaded.
    fn get_local_slam_result_sensor_id(&self, local_trajectory_id: i32) -> SensorId;
}

/// Returns whether `submap` received exactly one range data so far.
fn is_new_submap(submap: &mapping_proto::Submap) -> bool {
    submap
        .submap_2d
        .as_ref()
        .is_some_and(|s| s.num_range_data == 1)
        || submap
            .submap_3d
            .as_ref()
            .is_some_and(|s| s.num_range_data == 1)
}

/// What the uploader knows about a local trajectory.
#[derive(Clone, Debug)]
struct TrajectoryInfo {
    /// `None` if uplink has not yet responded to `AddTrajectoryRequest`.
    uplink_trajectory_id: Option<i32>,
    expected_sensor_ids: BTreeSet<SensorId>,
    trajectory_options: mapping_proto::TrajectoryBuilderOptions,
    client_id: String,
}

struct Inner {
    runtime: RuntimeHolder,
    endpoint: Option<Endpoint>,
    client_channel: Mutex<Option<Channel>>,
    batch_size: usize,
    local_trajectory_id_to_trajectory_info: Mutex<BTreeMap<i32, TrajectoryInfo>>,
    send_queue: BlockingQueue<Box<proto::SensorData>>,
    shutting_down: AtomicBool,
}

/// The gRPC implementation of [`LocalTrajectoryUploaderInterface`].
pub struct LocalTrajectoryUploader {
    inner: Arc<Inner>,
    upload_thread: Mutex<Option<JoinHandle<()>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl LocalTrajectoryUploader {
    /// Creates the uploader and waits up to 10 s for the uplink connection
    /// (a failure is only logged).
    ///
    /// RUST PORT NOTE: only plaintext channels are supported (no TLS backend
    /// is compiled in, see `DEPENDENCIES.md`); `enable_ssl_encryption` and
    /// `enable_google_auth` panic with an explanation instead of silently
    /// falling back to an insecure channel.
    pub fn new(
        uplink_server_address: &str,
        batch_size: i32,
        enable_ssl_encryption: bool,
        enable_google_auth: bool,
    ) -> Self {
        assert!(
            !enable_google_auth,
            "enable_google_auth is not supported: the Rust port has no Google credentials support"
        );
        assert!(
            !enable_ssl_encryption,
            "enable_ssl_encryption is not supported: the Rust port is built without a TLS backend"
        );
        let runtime = RuntimeHolder::new("cartographer-uploader", 1);
        log::info!("Connecting to uplink {uplink_server_address}");
        let endpoint = match create_endpoint(uplink_server_address) {
            Ok(endpoint) => Some(endpoint),
            Err(status) => {
                log::error!("Invalid uplink address: {}", status.message());
                None
            }
        };
        let client_channel = endpoint.as_ref().map(|endpoint| {
            if !runtime.block_on(wait_for_connected(
                endpoint,
                Instant::now() + CONNECTION_TIMEOUT,
            )) {
                log::error!("Failed to connect to {uplink_server_address}");
            }
            create_channel(runtime.handle(), endpoint)
        });
        Self {
            inner: Arc::new(Inner {
                runtime,
                endpoint,
                client_channel: Mutex::new(client_channel),
                batch_size: usize::try_from(batch_size).unwrap_or(0),
                local_trajectory_id_to_trajectory_info: Mutex::new(BTreeMap::new()),
                send_queue: BlockingQueue::new(),
                shutting_down: AtomicBool::new(false),
            }),
            upload_thread: Mutex::new(None),
        }
    }
}

impl Inner {
    fn client(&self) -> Result<MapBuilderServiceClient<Channel>, Status> {
        let channel = lock(&self.client_channel)
            .clone()
            .ok_or_else(|| Status::unavailable("No valid uplink address."))?;
        Ok(MapBuilderServiceClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE))
    }

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::SeqCst)
    }

    fn try_recovery(&self) {
        // RUST PORT NOTE: tonic exposes no connectivity state, so instead of
        // `GetState() != READY` + `WaitForConnected`, a fresh connection
        // attempt is made (for up to 60 s) whenever recovery is needed.
        if let Some(endpoint) = &self.endpoint {
            log::info!("Trying to re-connect to uplink...");
            let deadline = Instant::now() + CONNECTION_RECOVERY_TIMEOUT;
            let connected = self.runtime.block_on(async {
                loop {
                    if self.is_shutting_down() {
                        return false;
                    }
                    let attempt_deadline = (Instant::now() + Duration::from_secs(1)).min(deadline);
                    if wait_for_connected(endpoint, attempt_deadline).await {
                        return true;
                    }
                    if Instant::now() >= deadline {
                        return false;
                    }
                }
            });
            if !connected {
                log::error!("Failed to re-connect to uplink prior to recovery attempt.");
                return;
            }
            *lock(&self.client_channel) = Some(create_channel(self.runtime.handle(), endpoint));
        } else {
            log::error!("Failed to re-connect to uplink prior to recovery attempt.");
            return;
        }
        log::info!("Uplink channel ready, trying recovery.");
        // Wind the sensor_data_queue forward to the next new submap.
        log::info!("LocalTrajectoryUploader tries to recover with next submap.");
        loop {
            if self.is_shutting_down() {
                return;
            }
            let is_new_submap_result =
                self.send_queue
                    .peek_with_timeout(pop_timeout(), |sensor_data| {
                        match &sensor_data.sensor_data {
                            Some(proto::sensor_data::SensorData::LocalSlamResultData(data)) => {
                                data.submaps.last().is_some_and(is_new_submap)
                            }
                            _ => false,
                        }
                    });
            match is_new_submap_result {
                Some(true) => break,
                Some(false) => {
                    self.send_queue.pop();
                }
                None => {}
            }
        }
        // Because the trajectories may be interrupted on the uplink side, we
        // can no longer upload to those.
        let local_trajectory_ids: Vec<i32> = {
            let mut infos = lock(&self.local_trajectory_id_to_trajectory_info);
            for info in infos.values_mut() {
                info.uplink_trajectory_id = None;
            }
            infos.keys().copied().collect()
        };
        // TODO(gaschler): If the uplink did not restart but only the
        // connection was interrupted, this leaks trajectories in the uplink.
        // Attempt to recreate the trajectories.
        for local_trajectory_id in local_trajectory_ids {
            if let Err(status) = self.register_trajectory(local_trajectory_id) {
                log::error!(
                    "Failed to create trajectory. Aborting recovery attempt. {}",
                    status.message()
                );
                return;
            }
        }
        log::info!("LocalTrajectoryUploader recovered.");
    }

    fn process_send_queue(&self) {
        log::info!("Starting uploader thread.");
        let mut batch_request = proto::AddSensorDataBatchRequest::default();
        let strategy = RetryStrategy::UnlimitedConstantDelay {
            delay: Duration::from_secs(1),
            unrecoverable_codes: UNRECOVERABLE_STATUS_CODES.to_vec(),
        };
        while !self.is_shutting_down() {
            let Some(mut sensor_data) = self.send_queue.pop_with_timeout(pop_timeout()) else {
                continue;
            };
            let metadata = sensor_data
                .sensor_metadata
                .get_or_insert_with(Default::default);
            if !self.translate_trajectory_id(metadata) {
                batch_request.sensor_data.clear();
                self.try_recovery();
                continue;
            }
            let uplink_trajectory_id = metadata.trajectory_id;
            // A submap also holds a trajectory id that must be translated to
            // uplink's trajectory id.
            if let Some(proto::sensor_data::SensorData::LocalSlamResultData(data)) =
                &mut sensor_data.sensor_data
            {
                for submap in &mut data.submaps {
                    submap
                        .submap_id
                        .get_or_insert_with(Default::default)
                        .trajectory_id = uplink_trajectory_id;
                }
            }
            batch_request.sensor_data.push(*sensor_data);
            if batch_request.sensor_data.len() == self.batch_size {
                let result = self.client().map(|client| {
                    self.runtime.block_on(call_with_retry(
                        &strategy,
                        || self.is_shutting_down(),
                        || {
                            let mut client = client.clone();
                            let request = request_with_timeout(
                                batch_request.clone(),
                                Some(CONNECTION_TIMEOUT),
                            );
                            async move { client.add_sensor_data_batch(request).await }
                        },
                    ))
                });
                if let Ok(Ok(_)) = result {
                    log::info!(
                        "Uploaded {} bytes of sensor data.",
                        batch_request.encoded_len()
                    );
                    batch_request.sensor_data.clear();
                    continue;
                }
                // Unrecoverable error occurred. Attempt recovery.
                batch_request.sensor_data.clear();
                self.try_recovery();
            }
        }
    }

    /// Returns `false` for failure.
    fn translate_trajectory_id(&self, sensor_metadata: &mut proto::SensorMetadata) -> bool {
        let infos = lock(&self.local_trajectory_id_to_trajectory_info);
        let Some(info) = infos.get(&sensor_metadata.trajectory_id) else {
            return false;
        };
        let Some(cloud_trajectory_id) = info.uplink_trajectory_id else {
            // Could not yet register trajectory with uplink server.
            return false;
        };
        sensor_metadata.trajectory_id = cloud_trajectory_id;
        true
    }

    fn register_trajectory(&self, local_trajectory_id: i32) -> Result<(), Status> {
        let info = lock(&self.local_trajectory_id_to_trajectory_info)
            .get(&local_trajectory_id)
            .cloned()
            .unwrap_or_else(|| panic!("map::at: unknown local trajectory {local_trajectory_id}"));
        let mut request = proto::AddTrajectoryRequest {
            client_id: info.client_id.clone(),
            trajectory_builder_options: Some(info.trajectory_options.clone()),
            expected_sensor_ids: Vec::new(),
        };
        for sensor_id in &info.expected_sensor_ids {
            // Range sensors are not forwarded, but combined into a
            // LocalSlamResult.
            if sensor_id.r#type != SensorType::Range {
                request
                    .expected_sensor_ids
                    .push(serialization::to_proto(sensor_id));
            }
        }
        request
            .expected_sensor_ids
            .push(serialization::to_proto(&local_slam_result_sensor_id(
                local_trajectory_id,
            )));
        let response = self.client().and_then(|mut client| {
            self.runtime.block_on(
                client.add_trajectory(request_with_timeout(request, Some(CONNECTION_TIMEOUT))),
            )
        });
        match response {
            Err(status) => {
                log::error!(
                    "Failed to register local_trajectory_id {local_trajectory_id} with uplink \
                     server. {}",
                    status.message()
                );
                Err(status)
            }
            Ok(response) => {
                let uplink_trajectory_id = response.into_inner().trajectory_id;
                log::info!(
                    "Created trajectory for client_id: {} local trajectory_id: \
                     {local_trajectory_id} uplink trajectory_id: {uplink_trajectory_id}",
                    info.client_id
                );
                if let Some(info) =
                    lock(&self.local_trajectory_id_to_trajectory_info).get_mut(&local_trajectory_id)
                {
                    info.uplink_trajectory_id = Some(uplink_trajectory_id);
                }
                Ok(())
            }
        }
    }
}

fn local_slam_result_sensor_id(local_trajectory_id: i32) -> SensorId {
    SensorId {
        r#type: SensorType::LocalSlamResult,
        id: format!("local_slam_result_{local_trajectory_id}"),
    }
}

impl LocalTrajectoryUploaderInterface for LocalTrajectoryUploader {
    fn start(&self) {
        assert!(
            !self.inner.is_shutting_down(),
            "Check failed: !shutting_down_"
        );
        let mut upload_thread = lock(&self.upload_thread);
        assert!(upload_thread.is_none(), "Check failed: !upload_thread_");
        let inner = Arc::clone(&self.inner);
        *upload_thread = Some(
            std::thread::Builder::new()
                .name("cartographer-uploader".to_string())
                .spawn(move || inner.process_send_queue())
                .expect("failed to spawn the uploader thread"),
        );
    }

    fn shutdown(&self) {
        assert!(
            !self.inner.is_shutting_down(),
            "Check failed: !shutting_down_"
        );
        let upload_thread = lock(&self.upload_thread)
            .take()
            .expect("Check failed: upload_thread_");
        self.inner.shutting_down.store(true, Ordering::SeqCst);
        if upload_thread.join().is_err() {
            log::error!("The uploader thread panicked.");
        }
    }

    fn enqueue_sensor_data(&self, sensor_data: Box<proto::SensorData>) {
        self.inner.send_queue.push(sensor_data);
    }

    fn add_trajectory(
        &self,
        client_id: &str,
        local_trajectory_id: i32,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &mapping_proto::TrajectoryBuilderOptions,
    ) -> Result<(), Status> {
        {
            let mut infos = lock(&self.inner.local_trajectory_id_to_trajectory_info);
            assert!(
                !infos.contains_key(&local_trajectory_id),
                "Check failed: local_trajectory_id_to_trajectory_info_.count({local_trajectory_id}) == 0"
            );
            infos.insert(
                local_trajectory_id,
                TrajectoryInfo {
                    uplink_trajectory_id: None,
                    expected_sensor_ids: expected_sensor_ids.clone(),
                    trajectory_options: trajectory_options.clone(),
                    client_id: client_id.to_string(),
                },
            );
        }
        self.inner.register_trajectory(local_trajectory_id)
    }

    fn finish_trajectory(&self, client_id: &str, local_trajectory_id: i32) -> Result<(), Status> {
        let cloud_trajectory_id = {
            let infos = lock(&self.inner.local_trajectory_id_to_trajectory_info);
            let Some(info) = infos.get(&local_trajectory_id) else {
                return Err(Status::invalid_argument(
                    "local_trajectory_id has not been registered with AddTrajectory.",
                ));
            };
            info.uplink_trajectory_id
        };
        let Some(cloud_trajectory_id) = cloud_trajectory_id else {
            return Err(Status::unavailable(
                "trajectory_id has not been created in uplink, ignoring.",
            ));
        };
        let request = proto::FinishTrajectoryRequest {
            trajectory_id: cloud_trajectory_id,
            client_id: client_id.to_string(),
        };
        let mut client = self.inner.client()?;
        self.inner
            .runtime
            .block_on(
                client.finish_trajectory(request_with_timeout(request, Some(CONNECTION_TIMEOUT))),
            )
            .map(|_| ())
    }

    fn get_local_slam_result_sensor_id(&self, local_trajectory_id: i32) -> SensorId {
        local_slam_result_sensor_id(local_trajectory_id)
    }
}

impl Drop for LocalTrajectoryUploader {
    fn drop(&mut self) {
        // RUST PORT NOTE: the C++ destructor does not stop a running upload
        // thread; here it is stopped so that the thread cannot outlive us.
        if let Some(upload_thread) = lock(&self.upload_thread).take() {
            self.inner.shutting_down.store(true, Ordering::SeqCst);
            let _ = upload_thread.join();
        }
    }
}

/// Returns a `LocalTrajectoryUploader` with the actual implementation.
pub fn create_local_trajectory_uploader(
    uplink_server_address: &str,
    batch_size: i32,
    enable_ssl_encryption: bool,
    enable_google_auth: bool,
) -> Arc<dyn LocalTrajectoryUploaderInterface> {
    Arc::new(LocalTrajectoryUploader::new(
        uplink_server_address,
        batch_size,
        enable_ssl_encryption,
        enable_google_auth,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIENT_ID: &str = "CLIENT_ID";
    const LOCAL_TRAJECTORY_ID: i32 = 3;

    /// Returns `localhost:<port>` of an ephemeral port nothing listens on.
    fn unused_local_address() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        format!("127.0.0.1:{port}")
    }

    mod local_trajectory_uploader_test {
        use super::*;

        #[test]
        fn handles_invalid_uplink() {
            let imu_sensor_id = SensorId {
                r#type: SensorType::Imu,
                id: "imu".to_string(),
            };
            let range_sensor_id = SensorId {
                r#type: SensorType::Range,
                id: "range".to_string(),
            };
            let uploader = create_local_trajectory_uploader(
                &unused_local_address(),
                /*batch_size=*/ 1,
                false,
                false,
            );
            uploader.start();
            let options = mapping_proto::TrajectoryBuilderOptions::default();
            let status = uploader.add_trajectory(
                CLIENT_ID,
                LOCAL_TRAJECTORY_ID,
                &[range_sensor_id, imu_sensor_id.clone()]
                    .into_iter()
                    .collect(),
                &options,
            );
            assert!(status.is_err());
            let sensor_data = proto::SensorData {
                sensor_metadata: Some(proto::SensorMetadata {
                    trajectory_id: LOCAL_TRAJECTORY_ID,
                    sensor_id: imu_sensor_id.id.clone(),
                    client_id: CLIENT_ID.to_string(),
                }),
                sensor_data: Some(proto::sensor_data::SensorData::ImuData(
                    cartographer::sensor::proto::ImuData {
                        timestamp: 1,
                        ..Default::default()
                    },
                )),
            };
            uploader.enqueue_sensor_data(Box::new(sensor_data));
            let sensor_id = uploader.get_local_slam_result_sensor_id(LOCAL_TRAJECTORY_ID);
            assert!(!sensor_id.id.is_empty());
            let status = uploader.finish_trajectory(CLIENT_ID, LOCAL_TRAJECTORY_ID);
            assert!(status.is_err());
            uploader.shutdown();
        }
    }

    #[test]
    fn local_slam_result_sensor_id_format() {
        assert_eq!(
            local_slam_result_sensor_id(3),
            SensorId {
                r#type: SensorType::LocalSlamResult,
                id: "local_slam_result_3".to_string(),
            }
        );
    }

    #[test]
    fn new_submap_detection() {
        let mut submap = mapping_proto::Submap::default();
        assert!(!is_new_submap(&submap));
        submap.submap_2d = Some(mapping_proto::Submap2D {
            num_range_data: 1,
            ..Default::default()
        });
        assert!(is_new_submap(&submap));
        submap.submap_2d.as_mut().unwrap().num_range_data = 2;
        assert!(!is_new_submap(&submap));
        submap.submap_3d = Some(mapping_proto::Submap3D {
            num_range_data: 1,
            ..Default::default()
        });
        assert!(is_new_submap(&submap));
    }

    #[test]
    #[should_panic(expected = "enable_ssl_encryption is not supported")]
    fn ssl_is_rejected() {
        let _ = LocalTrajectoryUploader::new("127.0.0.1:1", 1, true, false);
    }

    #[test]
    fn finish_unknown_trajectory_is_invalid_argument() {
        let uploader = LocalTrajectoryUploader::new(&unused_local_address(), 1, false, false);
        assert_eq!(
            uploader.finish_trajectory(CLIENT_ID, 1).unwrap_err().code(),
            Code::InvalidArgument
        );
    }
}
