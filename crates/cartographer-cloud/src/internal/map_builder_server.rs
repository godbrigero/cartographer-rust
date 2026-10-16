//! Port of `cartographer/cloud/internal/map_builder_server` (.h, .cc).
//!
//! RUST PORT NOTE: `async_grpc::Server` becomes a `tonic` server running on
//! a tokio runtime owned by the [`MapBuilderServer`]
//! (`num_grpc_threads` worker threads; `num_event_threads` has no tonic
//! equivalent and is ignored). Blocking map builder calls run on tokio's
//! blocking thread pool. The state that the C++ `MapBuilderContext` reaches
//! through its `MapBuilderServer*` friend pointer lives in [`ServerState`],
//! shared via `Arc`.

use std::collections::BTreeMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use cartographer::common::internal::blocking_queue::BlockingQueue;
use cartographer::common::time::{Time, from_milliseconds};
use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::three_d::submap_3d::Submap3D;
use cartographer::mapping::trajectory_builder_interface::InsertionResult;
use cartographer::mapping::two_d::submap_2d::Submap2D;
use cartographer::metrics::family_factory::{FamilyFactory, Labels};
use cartographer::metrics::gauge::Gauge;
use cartographer::sensor::range_data::RangeData;
use cartographer::transform::rigid_transform::Rigid3d;
use tokio::sync::oneshot;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::TcpListenerStream;

use crate::internal::async_grpc::RuntimeHolder;
use crate::internal::handlers;
use crate::internal::local_trajectory_uploader::{
    LocalTrajectoryUploaderInterface, create_local_trajectory_uploader,
};
use crate::internal::map_builder_context_impl::MapBuilderContext;
use crate::internal::map_builder_context_interface::{
    BoxedMapBuilder, Data, GlobalSlamOptimizationCallback, LocalSlamResult,
    LocalSlamSubscriptionCallback, LocalSlamSubscriptionId, MapBuilderContextInterface,
};
use crate::internal::sensor::serialization::create_sensor_data_for_local_slam_result;
use crate::map_builder_server_interface::MapBuilderServerInterface;
use crate::proto;

static INCOMING_DATA_QUEUE_METRIC: Mutex<Option<&'static dyn Gauge>> = Mutex::new(None);

fn incoming_data_queue_metric() -> &'static dyn Gauge {
    lock(&INCOMING_DATA_QUEUE_METRIC).unwrap_or_else(<dyn Gauge>::null)
}

/// How long `shutdown` waits for in-flight RPCs before aborting them.
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

type LocalSlamResultHandlerSubscriptions =
    BTreeMap<i32 /* subscription_index */, LocalSlamSubscriptionCallback>;

#[derive(Default)]
struct Subscriptions {
    current_subscription_index: i32,
    local_slam_subscriptions:
        BTreeMap<i32 /* trajectory ID */, LocalSlamResultHandlerSubscriptions>,
    global_slam_subscriptions:
        BTreeMap<i32 /* subscription_index */, GlobalSlamOptimizationCallback>,
}

/// The server state shared by the server, its execution context, the SLAM
/// thread and the map builder callbacks.
pub struct ServerState {
    shutting_down: AtomicBool,
    pub(crate) map_builder: Mutex<BoxedMapBuilder>,
    pub(crate) incoming_data_queue: BlockingQueue<Box<Data>>,
    subscriptions: Mutex<Subscriptions>,
    pub(crate) local_trajectory_uploader: RwLock<Option<Arc<dyn LocalTrajectoryUploaderInterface>>>,
    starting_submap_index: Mutex<i32>,
}

impl ServerState {
    fn process_sensor_data_queue(&self, context: &dyn MapBuilderContextInterface) {
        log::info!("Starting SLAM thread.");
        let pop_timeout = from_milliseconds(100);
        while !self.shutting_down.load(Ordering::SeqCst) {
            incoming_data_queue_metric().set(self.incoming_data_queue.size() as f64);
            if let Some(sensor_data) = self.incoming_data_queue.pop_with_timeout(pop_timeout) {
                context.add_sensor_data_to_trajectory(sensor_data);
            }
        }
    }

    /// The uploader, if any (C++ `local_trajectory_uploader_.get()`).
    pub(crate) fn local_trajectory_uploader(
        &self,
    ) -> Option<Arc<dyn LocalTrajectoryUploaderInterface>> {
        self.local_trajectory_uploader
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub(crate) fn on_local_slam_result(
        &self,
        trajectory_id: i32,
        client_id: &str,
        time: Time,
        local_pose: Rigid3d,
        range_data: RangeData,
        insertion_result: Option<InsertionResult>,
    ) {
        let shared_range_data = Arc::new(range_data);

        // If there is an uplink server and a submap insertion happened,
        // enqueue this local SLAM result for uploading.
        if let (Some(insertion_result), Some(uploader)) =
            (&insertion_result, self.local_trajectory_uploader())
        {
            let sensor_id = uploader.get_local_slam_result_sensor_id(trajectory_id);
            let mut starting_submap_index = lock(&self.starting_submap_index);
            let sensor_data = create_sensor_data_for_local_slam_result(
                &sensor_id.id,
                trajectory_id,
                client_id,
                time,
                *starting_submap_index,
                insertion_result,
            );
            // TODO(cschuet): Make this more robust.
            if insertion_result
                .insertion_submaps
                .first()
                .expect("insertion_submaps must not be empty")
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .insertion_finished()
            {
                *starting_submap_index += 1;
            }
            drop(starting_submap_index);
            uploader.enqueue_sensor_data(Box::new(sensor_data));
        }

        let mut subscriptions = lock(&self.subscriptions);
        if let Some(trajectory_subscriptions) = subscriptions
            .local_slam_subscriptions
            .get_mut(&trajectory_id)
        {
            // RUST PORT NOTE: the C++ erases from the map while iterating
            // over it; `retain` has the intended semantics.
            trajectory_subscriptions.retain(|subscription_index, callback| {
                let keep = callback(Some(Box::new(LocalSlamResult {
                    trajectory_id,
                    time,
                    local_pose,
                    range_data: Some(Arc::clone(&shared_range_data)),
                    insertion_result: insertion_result.clone(),
                })));
                if !keep {
                    log::info!("Removing subscription with index: {subscription_index}");
                }
                keep
            });
        }
    }

    fn on_global_slam_optimizations(
        &self,
        last_optimized_submap_ids: &BTreeMap<i32, SubmapId>,
        last_optimized_node_ids: &BTreeMap<i32, NodeId>,
    ) {
        lock(&self.subscriptions).global_slam_subscriptions.retain(
            |subscription_index, callback| {
                let keep = callback(last_optimized_submap_ids, last_optimized_node_ids);
                if !keep {
                    log::info!("Removing subscription with index: {subscription_index}");
                }
                keep
            },
        );
    }

    pub(crate) fn subscribe_local_slam_results(
        &self,
        trajectory_id: i32,
        callback: LocalSlamSubscriptionCallback,
    ) -> LocalSlamSubscriptionId {
        let mut subscriptions = lock(&self.subscriptions);
        let subscription_index = subscriptions.current_subscription_index;
        subscriptions
            .local_slam_subscriptions
            .entry(trajectory_id)
            .or_default()
            .insert(subscription_index, callback);
        subscriptions.current_subscription_index += 1;
        LocalSlamSubscriptionId {
            trajectory_id,
            subscription_index,
        }
    }

    /// RUST PORT NOTE: the C++ CHECKs that exactly one subscription was
    /// erased; here a subscription that was already removed (because its
    /// callback returned false or the server shut down) is ignored.
    pub(crate) fn unsubscribe_local_slam_results(&self, subscription_id: &LocalSlamSubscriptionId) {
        if let Some(trajectory_subscriptions) = lock(&self.subscriptions)
            .local_slam_subscriptions
            .get_mut(&subscription_id.trajectory_id)
        {
            trajectory_subscriptions.remove(&subscription_id.subscription_index);
        }
    }

    pub(crate) fn subscribe_global_slam_optimizations(
        &self,
        callback: GlobalSlamOptimizationCallback,
    ) -> i32 {
        let mut subscriptions = lock(&self.subscriptions);
        let subscription_index = subscriptions.current_subscription_index;
        subscriptions
            .global_slam_subscriptions
            .insert(subscription_index, callback);
        subscriptions.current_subscription_index += 1;
        subscription_index
    }

    /// See [`Self::unsubscribe_local_slam_results`] for the deviation from
    /// the C++ CHECK.
    pub(crate) fn unsubscribe_global_slam_optimizations(&self, subscription_index: i32) {
        lock(&self.subscriptions)
            .global_slam_subscriptions
            .remove(&subscription_index);
    }

    pub(crate) fn notify_finish_trajectory(&self, trajectory_id: i32) {
        if let Some(trajectory_subscriptions) = lock(&self.subscriptions)
            .local_slam_subscriptions
            .get(&trajectory_id)
        {
            for callback in trajectory_subscriptions.values() {
                // 'None' signals subscribers that the trajectory finished.
                callback(None);
            }
        }
    }

    /// Drops all subscriptions, which ends the corresponding server streams.
    fn clear_subscriptions(&self) {
        let mut subscriptions = lock(&self.subscriptions);
        subscriptions.local_slam_subscriptions.clear();
        subscriptions.global_slam_subscriptions.clear();
    }
}

struct GrpcServer {
    shutdown_sender: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    local_address: SocketAddr,
}

/// The gRPC map builder server.
pub struct MapBuilderServer {
    server_address: String,
    num_grpc_threads: i32,
    state: Arc<ServerState>,
    context: Arc<dyn MapBuilderContextInterface>,
    runtime: Mutex<Option<RuntimeHolder>>,
    grpc_server: Mutex<Option<GrpcServer>>,
    grpc_stopped: Mutex<bool>,
    grpc_stopped_condition: Condvar,
    slam_thread: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for MapBuilderServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapBuilderServer")
            .field("server_address", &self.server_address)
            .field("local_address", &self.local_address())
            .finish_non_exhaustive()
    }
}

impl MapBuilderServer {
    /// Creates the server (not started yet). Creates the uploader if
    /// `uplink_server_address` is set and registers the global SLAM callback
    /// with the map builder's pose graph. Panics unless exactly one of
    /// `use_trajectory_builder_2d`/`_3d` is set.
    pub fn new(
        map_builder_server_options: &proto::MapBuilderServerOptions,
        map_builder: BoxedMapBuilder,
    ) -> Self {
        let local_trajectory_uploader =
            (!map_builder_server_options.uplink_server_address.is_empty()).then(|| {
                create_local_trajectory_uploader(
                    &map_builder_server_options.uplink_server_address,
                    map_builder_server_options.upload_batch_size,
                    map_builder_server_options.enable_ssl_encryption,
                    map_builder_server_options.enable_google_auth,
                )
            });
        let pose_graph = map_builder.pose_graph();
        let state = Arc::new(ServerState {
            shutting_down: AtomicBool::new(false),
            map_builder: Mutex::new(map_builder),
            incoming_data_queue: BlockingQueue::new(),
            subscriptions: Mutex::new(Subscriptions::default()),
            local_trajectory_uploader: RwLock::new(local_trajectory_uploader),
            starting_submap_index: Mutex::new(0),
        });
        let map_builder_options = map_builder_server_options
            .map_builder_options
            .clone()
            .unwrap_or_default();
        let context: Arc<dyn MapBuilderContextInterface> =
            if map_builder_options.use_trajectory_builder_2d {
                MapBuilderContext::<Submap2D>::new(Arc::clone(&state))
            } else if map_builder_options.use_trajectory_builder_3d {
                MapBuilderContext::<Submap3D>::new(Arc::clone(&state))
            } else {
                panic!("Set either use_trajectory_builder_2d or use_trajectory_builder_3d");
            };
        let weak_state: Weak<ServerState> = Arc::downgrade(&state);
        pose_graph.set_global_slam_optimization_callback(Box::new(
            move |last_optimized_submap_ids, last_optimized_node_ids| {
                if let Some(state) = weak_state.upgrade() {
                    state.on_global_slam_optimizations(
                        last_optimized_submap_ids,
                        last_optimized_node_ids,
                    );
                }
            },
        ));
        Self {
            server_address: map_builder_server_options.server_address.clone(),
            num_grpc_threads: map_builder_server_options.num_grpc_threads,
            state,
            context,
            runtime: Mutex::new(None),
            grpc_server: Mutex::new(None),
            grpc_stopped: Mutex::new(false),
            grpc_stopped_condition: Condvar::new(),
            slam_thread: Mutex::new(None),
        }
    }

    /// The execution context shared by the handlers.
    pub fn context(&self) -> &Arc<dyn MapBuilderContextInterface> {
        &self.context
    }

    /// The address the gRPC server is bound to while running (useful with
    /// port 0, i.e. an ephemeral port).
    ///
    /// RUST PORT NOTE: addition to the C++ API.
    pub fn local_address(&self) -> Option<SocketAddr> {
        lock(&self.grpc_server)
            .as_ref()
            .map(|server| server.local_address)
    }

    /// Registers the server metrics (C++ `RegisterMetrics`).
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        let queue_length = family_factory.new_gauge_family(
            "cloud_internal_map_builder_server_incoming_data_queue_length",
            "Incoming SLAM Data Queue length",
        );
        *lock(&INCOMING_DATA_QUEUE_METRIC) = Some(queue_length.add(&Labels::new()));
    }

    fn start_slam_thread(&self) {
        let mut slam_thread = lock(&self.slam_thread);
        assert!(slam_thread.is_none(), "Check failed: !slam_thread_");
        // Start the SLAM processing thread.
        let state = Arc::clone(&self.state);
        let context = Arc::clone(&self.context);
        *slam_thread = Some(
            std::thread::Builder::new()
                .name("cartographer-slam".to_string())
                .spawn(move || state.process_sensor_data_queue(context.as_ref()))
                .expect("failed to spawn the SLAM thread"),
        );
    }

    fn start_grpc_server(&self) {
        let address = self
            .server_address
            .to_socket_addrs()
            .ok()
            .and_then(|mut addresses| addresses.next())
            .unwrap_or_else(|| panic!("Invalid server address '{}'", self.server_address));
        let listener = std::net::TcpListener::bind(address)
            .unwrap_or_else(|e| panic!("Failed to bind to '{}': {e}", self.server_address));
        let local_address = listener
            .local_addr()
            .expect("bound listener has an address");
        listener
            .set_nonblocking(true)
            .expect("failed to make the listener non-blocking");

        let mut runtime = lock(&self.runtime);
        let runtime = runtime.get_or_insert_with(|| {
            RuntimeHolder::new(
                "cartographer-grpc",
                usize::try_from(self.num_grpc_threads).unwrap_or(1),
            )
        });
        let incoming = {
            let _guard = runtime.handle().enter();
            tokio::net::TcpListener::from_std(listener)
                .expect("failed to register the listener with tokio")
        };
        let incoming = TcpListenerStream::new(incoming).map(|stream| {
            stream.inspect(|stream| {
                let _ = stream.set_nodelay(true);
            })
        });
        let service = handlers::create_map_builder_service(Arc::clone(&self.context));
        let (shutdown_sender, shutdown_receiver) = oneshot::channel::<()>();
        let task = runtime.handle().spawn(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = shutdown_receiver.await;
                }),
        );
        *lock(&self.grpc_stopped) = false;
        *lock(&self.grpc_server) = Some(GrpcServer {
            shutdown_sender,
            task,
            local_address,
        });
        log::info!("Server listening on {local_address}");
    }

    fn shutdown_grpc_server(&self) {
        let Some(server) = lock(&self.grpc_server).take() else {
            return;
        };
        let _ = server.shutdown_sender.send(());
        // Ending all subscriptions ends the server streams, so that the
        // graceful shutdown does not wait for them.
        self.state.clear_subscriptions();
        if let Some(runtime) = lock(&self.runtime).as_ref() {
            let abort_handle = server.task.abort_handle();
            match runtime.block_on(tokio::time::timeout(GRACEFUL_SHUTDOWN_TIMEOUT, server.task)) {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(e))) => log::error!("gRPC server error: {e}"),
                Ok(Err(e)) => log::error!("gRPC server task failed: {e}"),
                Err(_) => {
                    log::warn!("Aborting in-flight RPCs after {GRACEFUL_SHUTDOWN_TIMEOUT:?}.");
                    abort_handle.abort();
                }
            }
        }
        *lock(&self.grpc_stopped) = true;
        self.grpc_stopped_condition.notify_all();
    }
}

impl MapBuilderServerInterface for MapBuilderServer {
    fn start(&self) {
        self.state.shutting_down.store(false, Ordering::SeqCst);
        if let Some(uploader) = self.state.local_trajectory_uploader() {
            uploader.start();
        }
        self.start_slam_thread();
        self.start_grpc_server();
    }

    fn wait_for_shutdown(&self) {
        {
            let stopped = lock(&self.grpc_stopped);
            let _stopped = self
                .grpc_stopped_condition
                .wait_while(stopped, |stopped| !*stopped)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if let Some(slam_thread) = lock(&self.slam_thread).take() {
            let _ = slam_thread.join();
        }
        let uploader = self
            .state
            .local_trajectory_uploader
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(uploader) = uploader {
            uploader.shutdown();
        }
    }

    fn wait_until_idle(&self) {
        self.state.incoming_data_queue.wait_until_empty();
        let pose_graph = lock(&self.state.map_builder).pose_graph();
        pose_graph.run_final_optimization();
    }

    fn shutdown(&self) {
        self.state.shutting_down.store(true, Ordering::SeqCst);
        self.shutdown_grpc_server();
        if let Some(slam_thread) = lock(&self.slam_thread).take()
            && slam_thread.join().is_err()
        {
            log::error!("The SLAM thread panicked.");
        }
        let uploader = self
            .state
            .local_trajectory_uploader
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(uploader) = uploader {
            uploader.shutdown();
        }
    }
}

impl Drop for MapBuilderServer {
    /// RUST PORT NOTE: the C++ destructor does nothing (destroying a running
    /// server terminates the process); here a running server is shut down.
    fn drop(&mut self) {
        if lock(&self.grpc_server).is_some() || lock(&self.slam_thread).is_some() {
            self.shutdown();
        }
    }
}
