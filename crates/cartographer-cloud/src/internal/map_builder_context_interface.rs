//! Port of `cartographer/cloud/internal/map_builder_context_interface` (.h).
//!
//! RUST PORT NOTE: in C++ this is an `async_grpc::ExecutionContext`; handlers
//! reach it through `GetContext<T>()` (which locks a mutex around the whole
//! context) or `GetUnsynchronizedContext<T>()`. Here the context is a
//! `Send + Sync` trait object shared as `Arc<dyn MapBuilderContextInterface>`
//! and every implementation synchronizes internally: the map builder sits
//! behind a mutex ([`MapBuilderContextInterface::map_builder`] returns the
//! guard), the client-id map and subscriptions have their own locks, and the
//! queues are thread-safe.

use std::collections::BTreeMap;
use std::sync::{Arc, MutexGuard};

use cartographer::common::internal::blocking_queue::BlockingQueue;
use cartographer::common::time::Time;
use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::map_builder_interface::MapBuilderInterface;
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::{
    InsertionResult, LocalSlamResultCallback,
};
use cartographer::sensor::data::Data as SensorData;
use cartographer::sensor::range_data::RangeData;
use cartographer::transform::rigid_transform::Rigid3d;

use crate::internal::local_trajectory_uploader::LocalTrajectoryUploaderInterface;

/// The map builder type served by the cloud server.
pub type BoxedMapBuilder = Box<dyn MapBuilderInterface + Send>;

/// A local SLAM result delivered to subscribers (C++
/// `MapBuilderContextInterface::LocalSlamResult`).
#[derive(Clone, Debug)]
pub struct LocalSlamResult {
    /// The trajectory.
    pub trajectory_id: i32,
    /// Time of the range data.
    pub time: Time,
    /// Local pose estimate.
    pub local_pose: Rigid3d,
    /// Range data in the local frame (shared between subscribers).
    pub range_data: Option<Arc<RangeData>>,
    /// Set if the range data was inserted into submaps.
    pub insertion_result: Option<InsertionResult>,
}

/// Calling with `None` (C++ `nullptr`) signals subscribers that the
/// subscription has ended, e.g. this happens when the corresponding
/// trajectory was finished and hence no more local SLAM updates will occur.
/// The callback can return `false` to indicate that the client is not
/// interested in more local SLAM updates and `MapBuilderServer` will end the
/// subscription.
pub type LocalSlamSubscriptionCallback =
    Arc<dyn Fn(Option<Box<LocalSlamResult>>) -> bool + Send + Sync>;

/// The callback can return `false` to indicate that the client is not
/// interested in more global SLAM runs and `MapBuilderServer` will end the
/// subscription. Arguments: last optimized submap and node id per
/// trajectory.
pub type GlobalSlamOptimizationCallback = Arc<
    dyn Fn(
            &BTreeMap<i32 /* trajectory_id */, SubmapId>,
            &BTreeMap<i32 /* trajectory_id */, NodeId>,
        ) -> bool
        + Send
        + Sync,
>;

/// A sensor data packet queued for the SLAM thread (C++
/// `MapBuilderContextInterface::Data`).
pub struct Data {
    /// The trajectory the data belongs to.
    pub trajectory_id: i32,
    /// The packet.
    pub data: Box<dyn SensorData>,
}

impl std::fmt::Debug for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Data")
            .field("trajectory_id", &self.trajectory_id)
            .field("sensor_id", &self.data.get_sensor_id())
            .field("time", &self.data.get_time())
            .finish()
    }
}

/// Identifies a local SLAM subscription.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalSlamSubscriptionId {
    /// The subscribed trajectory.
    pub trajectory_id: i32,
    /// Server-wide subscription index.
    pub subscription_index: i32,
}

/// The execution context shared by all RPC handlers of the map builder
/// server.
pub trait MapBuilderContextInterface: Send + Sync {
    /// Locks and returns the map builder (C++ `map_builder()` through the
    /// locked `GetContext`). Keep the guard short-lived.
    fn map_builder(&self) -> MutexGuard<'_, BoxedMapBuilder>;

    /// The incoming sensor data queue consumed by the SLAM thread.
    fn sensor_data_queue(&self) -> &BlockingQueue<Box<Data>>;

    /// Returns the callback that forwards local SLAM results of a trajectory
    /// to the subscribers (and the uploader). `None` corresponds to an empty
    /// C++ `std::function`.
    fn get_local_slam_result_callback_for_subscriptions(&self) -> Option<LocalSlamResultCallback>;

    /// Hands `sensor_data` to the trajectory builder of its trajectory.
    fn add_sensor_data_to_trajectory(&self, sensor_data: Box<Data>);

    /// Subscribes to local SLAM results of `trajectory_id`.
    fn subscribe_local_slam_results(
        &self,
        trajectory_id: i32,
        callback: LocalSlamSubscriptionCallback,
    ) -> LocalSlamSubscriptionId;

    /// Ends a local SLAM subscription.
    fn unsubscribe_local_slam_results(&self, subscription_id: &LocalSlamSubscriptionId);

    /// Subscribes to global SLAM optimizations; returns the subscription
    /// index.
    fn subscribe_global_slam_optimizations(&self, callback: GlobalSlamOptimizationCallback) -> i32;

    /// Ends a global SLAM subscription.
    fn unsubscribe_global_slam_optimizations(&self, subscription_index: i32);

    /// Signals the end of the local SLAM subscriptions of `trajectory_id`.
    fn notify_finish_trajectory(&self, trajectory_id: i32);

    /// The uploader to the uplink server, if configured.
    fn local_trajectory_uploader(&self) -> Option<Arc<dyn LocalTrajectoryUploaderInterface>>;

    /// Queues sensor data for the SLAM thread.
    fn enqueue_sensor_data(&self, trajectory_id: i32, data: Box<dyn SensorData>);

    /// Queues an uploaded local SLAM result for the SLAM thread.
    fn enqueue_local_slam_result_data(
        &self,
        trajectory_id: i32,
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
    );

    /// Associates `trajectory_id` with `client_id`. Panics if the trajectory
    /// is already registered.
    fn register_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32);

    /// Returns whether `trajectory_id` is registered to `client_id`.
    fn check_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32) -> bool;
}
