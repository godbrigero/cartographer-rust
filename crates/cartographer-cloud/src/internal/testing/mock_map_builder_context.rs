//! Port of `cartographer/cloud/internal/testing/mock_map_builder_context` (.h).
//!
//! RUST PORT NOTE: the gmock mock becomes a hand-written fake: calls are
//! recorded, return values are configurable, enqueued sensor data is
//! dispatched into a recorder (see [`super::test_helpers::dispatch`]) and
//! subscriptions are stored so tests can drive them.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use cartographer::common::internal::blocking_queue::BlockingQueue;
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::trajectory_builder_interface::LocalSlamResultCallback;
use cartographer::sensor::data::Data as SensorData;

use crate::internal::local_trajectory_uploader::LocalTrajectoryUploaderInterface;
use crate::internal::map_builder_context_interface::{
    BoxedMapBuilder, Data, GlobalSlamOptimizationCallback, LocalSlamSubscriptionCallback,
    LocalSlamSubscriptionId, MapBuilderContextInterface,
};
use crate::internal::map_builder_server::lock;
use crate::internal::testing::mock_local_trajectory_uploader::MockLocalTrajectoryUploader;
use crate::internal::testing::test_helpers::{DispatchedData, dispatch};

/// A recorded context call without payload of interest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextCall {
    /// `GetLocalSlamResultCallbackForSubscriptions()`.
    GetLocalSlamResultCallbackForSubscriptions,
    /// `CheckClientIdForTrajectory(client_id, trajectory_id)`.
    CheckClientIdForTrajectory(String, i32),
    /// `RegisterClientIdForTrajectory(client_id, trajectory_id)`.
    RegisterClientIdForTrajectory(String, i32),
    /// `NotifyFinishTrajectory(trajectory_id)`.
    NotifyFinishTrajectory(i32),
    /// `local_trajectory_uploader()`.
    LocalTrajectoryUploader,
    /// `AddSensorDataToTrajectory(trajectory_id)`.
    AddSensorDataToTrajectory(i32),
}

/// Fake `MapBuilderContextInterface`.
pub struct MockMapBuilderContext {
    map_builder: Mutex<BoxedMapBuilder>,
    sensor_data_queue: BlockingQueue<Box<Data>>,
    local_trajectory_uploader: Mutex<Option<Arc<MockLocalTrajectoryUploader>>>,
    check_client_id_result: AtomicBool,
    calls: Mutex<Vec<ContextCall>>,
    enqueued_sensor_data: Mutex<Vec<(i32, DispatchedData)>>,
    enqueued_local_slam_result_data: Mutex<Vec<(i32, String, mapping_proto::LocalSlamResultData)>>,
    local_slam_subscriptions: Mutex<BTreeMap<i32, (i32, LocalSlamSubscriptionCallback)>>,
    global_slam_subscriptions: Mutex<BTreeMap<i32, GlobalSlamOptimizationCallback>>,
    next_subscription_index: Mutex<i32>,
}

impl MockMapBuilderContext {
    /// Creates the fake around `map_builder`. `CheckClientIdForTrajectory`
    /// returns true and there is no uploader until configured otherwise.
    pub fn new(map_builder: BoxedMapBuilder) -> Self {
        Self {
            map_builder: Mutex::new(map_builder),
            sensor_data_queue: BlockingQueue::new(),
            local_trajectory_uploader: Mutex::new(None),
            check_client_id_result: AtomicBool::new(true),
            calls: Mutex::new(Vec::new()),
            enqueued_sensor_data: Mutex::new(Vec::new()),
            enqueued_local_slam_result_data: Mutex::new(Vec::new()),
            local_slam_subscriptions: Mutex::new(BTreeMap::new()),
            global_slam_subscriptions: Mutex::new(BTreeMap::new()),
            next_subscription_index: Mutex::new(0),
        }
    }

    /// C++ `SetNoLocalTrajectoryUploader` / `SetMockLocalTrajectoryUploader`.
    pub fn set_local_trajectory_uploader(
        &self,
        uploader: Option<Arc<MockLocalTrajectoryUploader>>,
    ) {
        *lock(&self.local_trajectory_uploader) = uploader;
    }

    /// Sets the return value of `CheckClientIdForTrajectory`.
    pub fn set_check_client_id_result(&self, result: bool) {
        self.check_client_id_result.store(result, Ordering::SeqCst);
    }

    /// Copy of the recorded calls.
    pub fn calls(&self) -> Vec<ContextCall> {
        lock(&self.calls).clone()
    }

    /// The sensor data passed to `EnqueueSensorData` (C++
    /// `DoEnqueueSensorData`), dispatched.
    pub fn enqueued_sensor_data(&self) -> Vec<(i32, DispatchedData)> {
        lock(&self.enqueued_sensor_data).clone()
    }

    /// The arguments of `EnqueueLocalSlamResultData`.
    pub fn enqueued_local_slam_result_data(
        &self,
    ) -> Vec<(i32, String, mapping_proto::LocalSlamResultData)> {
        lock(&self.enqueued_local_slam_result_data).clone()
    }

    /// Calls every local SLAM subscription of `trajectory_id` with `result`
    /// and returns the callbacks' return values.
    pub fn run_local_slam_subscriptions(
        &self,
        trajectory_id: i32,
        result: Option<Box<crate::internal::map_builder_context_interface::LocalSlamResult>>,
    ) -> Vec<bool> {
        let callbacks: Vec<LocalSlamSubscriptionCallback> = lock(&self.local_slam_subscriptions)
            .values()
            .filter(|(subscribed_trajectory_id, _)| *subscribed_trajectory_id == trajectory_id)
            .map(|(_, callback)| Arc::clone(callback))
            .collect();
        callbacks
            .into_iter()
            .map(|callback| callback(result.clone()))
            .collect()
    }

    /// Number of active local and global SLAM subscriptions.
    pub fn num_subscriptions(&self) -> (usize, usize) {
        (
            lock(&self.local_slam_subscriptions).len(),
            lock(&self.global_slam_subscriptions).len(),
        )
    }

    /// Calls every global SLAM subscription and returns their return values.
    pub fn run_global_slam_subscriptions(
        &self,
        last_optimized_submap_ids: &BTreeMap<i32, cartographer::mapping::id::SubmapId>,
        last_optimized_node_ids: &BTreeMap<i32, cartographer::mapping::id::NodeId>,
    ) -> Vec<bool> {
        let callbacks: Vec<GlobalSlamOptimizationCallback> = lock(&self.global_slam_subscriptions)
            .values()
            .cloned()
            .collect();
        callbacks
            .into_iter()
            .map(|callback| callback(last_optimized_submap_ids, last_optimized_node_ids))
            .collect()
    }

    fn record(&self, call: ContextCall) {
        lock(&self.calls).push(call);
    }
}

impl MapBuilderContextInterface for MockMapBuilderContext {
    fn map_builder(&self) -> MutexGuard<'_, BoxedMapBuilder> {
        lock(&self.map_builder)
    }

    fn sensor_data_queue(&self) -> &BlockingQueue<Box<Data>> {
        &self.sensor_data_queue
    }

    /// Like the C++ test (`WillOnce(Return(nullptr))`), returns no callback.
    fn get_local_slam_result_callback_for_subscriptions(&self) -> Option<LocalSlamResultCallback> {
        self.record(ContextCall::GetLocalSlamResultCallbackForSubscriptions);
        None
    }

    fn add_sensor_data_to_trajectory(&self, sensor_data: Box<Data>) {
        self.record(ContextCall::AddSensorDataToTrajectory(
            sensor_data.trajectory_id,
        ));
    }

    fn subscribe_local_slam_results(
        &self,
        trajectory_id: i32,
        callback: LocalSlamSubscriptionCallback,
    ) -> LocalSlamSubscriptionId {
        let mut next_subscription_index = lock(&self.next_subscription_index);
        let subscription_index = *next_subscription_index;
        *next_subscription_index += 1;
        lock(&self.local_slam_subscriptions).insert(subscription_index, (trajectory_id, callback));
        LocalSlamSubscriptionId {
            trajectory_id,
            subscription_index,
        }
    }

    fn unsubscribe_local_slam_results(&self, subscription_id: &LocalSlamSubscriptionId) {
        lock(&self.local_slam_subscriptions).remove(&subscription_id.subscription_index);
    }

    fn subscribe_global_slam_optimizations(&self, callback: GlobalSlamOptimizationCallback) -> i32 {
        let mut next_subscription_index = lock(&self.next_subscription_index);
        let subscription_index = *next_subscription_index;
        *next_subscription_index += 1;
        lock(&self.global_slam_subscriptions).insert(subscription_index, callback);
        subscription_index
    }

    fn unsubscribe_global_slam_optimizations(&self, subscription_index: i32) {
        lock(&self.global_slam_subscriptions).remove(&subscription_index);
    }

    fn notify_finish_trajectory(&self, trajectory_id: i32) {
        self.record(ContextCall::NotifyFinishTrajectory(trajectory_id));
    }

    fn local_trajectory_uploader(&self) -> Option<Arc<dyn LocalTrajectoryUploaderInterface>> {
        self.record(ContextCall::LocalTrajectoryUploader);
        lock(&self.local_trajectory_uploader)
            .clone()
            .map(|uploader| uploader as Arc<dyn LocalTrajectoryUploaderInterface>)
    }

    fn enqueue_sensor_data(&self, trajectory_id: i32, data: Box<dyn SensorData>) {
        lock(&self.enqueued_sensor_data).push((trajectory_id, dispatch(data)));
    }

    fn enqueue_local_slam_result_data(
        &self,
        trajectory_id: i32,
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
    ) {
        lock(&self.enqueued_local_slam_result_data).push((
            trajectory_id,
            sensor_id.to_string(),
            local_slam_result_data.clone(),
        ));
    }

    fn register_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32) {
        self.record(ContextCall::RegisterClientIdForTrajectory(
            client_id.to_string(),
            trajectory_id,
        ));
    }

    fn check_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32) -> bool {
        self.record(ContextCall::CheckClientIdForTrajectory(
            client_id.to_string(),
            trajectory_id,
        ));
        self.check_client_id_result.load(Ordering::SeqCst)
    }
}
