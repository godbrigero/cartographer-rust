//! Port of `cartographer/cloud/internal/map_builder_context_impl` (.h, .cc)
//! (the `MapBuilderContext<SubmapType>` template declared in
//! `map_builder_server.h`).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use cartographer::common::internal::blocking_queue::BlockingQueue;
use cartographer::mapping::internal::submap_controller::{ControllableSubmap, SubmapController};
use cartographer::mapping::internal::three_d::local_slam_result_3d::LocalSlamResult3D;
use cartographer::mapping::internal::two_d::local_slam_result_2d::LocalSlamResult2D;
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::three_d::submap_3d::Submap3D;
use cartographer::mapping::trajectory_builder_interface::LocalSlamResultCallback;
use cartographer::mapping::two_d::submap_2d::Submap2D;
use cartographer::sensor::data::Data as SensorData;

use crate::internal::local_trajectory_uploader::LocalTrajectoryUploaderInterface;
use crate::internal::map_builder_context_interface::{
    BoxedMapBuilder, Data, GlobalSlamOptimizationCallback, LocalSlamSubscriptionCallback,
    LocalSlamSubscriptionId, MapBuilderContextInterface,
};
use crate::internal::map_builder_server::{ServerState, lock};

/// Submap types the server can reassemble from uploaded local SLAM results.
///
/// RUST PORT NOTE: replaces the explicit specializations of
/// `MapBuilderContext<Submap2D/3D>::EnqueueLocalSlamResultData`.
pub trait ContextSubmap: ControllableSubmap + 'static {
    /// Creates the `LocalSlamResult2D`/`LocalSlamResult3D` packet.
    fn create_local_slam_result(
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
        submap_controller: &Arc<Mutex<SubmapController<Self>>>,
    ) -> Box<dyn SensorData>;
}

impl ContextSubmap for Submap2D {
    fn create_local_slam_result(
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
        submap_controller: &Arc<Mutex<SubmapController<Self>>>,
    ) -> Box<dyn SensorData> {
        Box::new(LocalSlamResult2D::new(
            sensor_id,
            local_slam_result_data.clone(),
            Arc::clone(submap_controller),
        ))
    }
}

impl ContextSubmap for Submap3D {
    fn create_local_slam_result(
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
        submap_controller: &Arc<Mutex<SubmapController<Self>>>,
    ) -> Box<dyn SensorData> {
        Box::new(LocalSlamResult3D::new(
            sensor_id,
            local_slam_result_data.clone(),
            Arc::clone(submap_controller),
        ))
    }
}

/// The execution context of the [`MapBuilderServer`](super::map_builder_server::MapBuilderServer)
/// for 2D (`Submap2D`) or 3D (`Submap3D`) SLAM.
pub struct MapBuilderContext<S: ContextSubmap> {
    map_builder_server: Arc<ServerState>,
    submap_controller: Arc<Mutex<SubmapController<S>>>,
    client_ids: Mutex<BTreeMap<i32 /* trajectory_id */, String /* client_id */>>,
    self_weak: Weak<Self>,
}

impl<S: ContextSubmap> MapBuilderContext<S> {
    /// Creates the context for the server state.
    pub fn new(map_builder_server: Arc<ServerState>) -> Arc<Self> {
        Arc::new_cyclic(|self_weak| Self {
            map_builder_server,
            submap_controller: Arc::new(Mutex::new(SubmapController::new())),
            client_ids: Mutex::new(BTreeMap::new()),
            self_weak: self_weak.clone(),
        })
    }
}

impl<S: ContextSubmap> MapBuilderContextInterface for MapBuilderContext<S> {
    fn map_builder(&self) -> MutexGuard<'_, BoxedMapBuilder> {
        lock(&self.map_builder_server.map_builder)
    }

    fn sensor_data_queue(&self) -> &BlockingQueue<Box<Data>> {
        &self.map_builder_server.incoming_data_queue
    }

    /// RUST PORT NOTE: the callback holds a weak reference to the context
    /// (the C++ captures `this`), because the map builder that stores the
    /// callback is owned by the context's server state.
    fn get_local_slam_result_callback_for_subscriptions(&self) -> Option<LocalSlamResultCallback> {
        let context = self.self_weak.clone();
        Some(Arc::new(
            move |trajectory_id, time, local_pose, range_data, insertion_result| {
                let Some(context) = context.upgrade() else {
                    return;
                };
                let client_id = lock(&context.client_ids).get(&trajectory_id).cloned();
                let Some(client_id) = client_id else {
                    log::error!("Unknown trajectory_id {trajectory_id}. Ignoring.");
                    return;
                };
                context.map_builder_server.on_local_slam_result(
                    trajectory_id,
                    &client_id,
                    time,
                    local_pose,
                    range_data,
                    insertion_result,
                );
            },
        ))
    }

    /// RUST PORT NOTE: the C++ dereferences the (possibly null) trajectory
    /// builder; here data for a trajectory without builder is dropped with
    /// an error message.
    fn add_sensor_data_to_trajectory(&self, sensor_data: Box<Data>) {
        let Data {
            trajectory_id,
            data,
        } = *sensor_data;
        let mut map_builder = self.map_builder();
        match map_builder.get_trajectory_builder(trajectory_id) {
            Some(trajectory_builder) => data.add_to_trajectory_builder(trajectory_builder),
            None => log::error!(
                "No trajectory builder for trajectory_id {trajectory_id}; dropping data of \
                 sensor '{}'.",
                data.get_sensor_id()
            ),
        }
    }

    fn subscribe_local_slam_results(
        &self,
        trajectory_id: i32,
        callback: LocalSlamSubscriptionCallback,
    ) -> LocalSlamSubscriptionId {
        self.map_builder_server
            .subscribe_local_slam_results(trajectory_id, callback)
    }

    fn unsubscribe_local_slam_results(&self, subscription_id: &LocalSlamSubscriptionId) {
        self.map_builder_server
            .unsubscribe_local_slam_results(subscription_id);
    }

    fn subscribe_global_slam_optimizations(&self, callback: GlobalSlamOptimizationCallback) -> i32 {
        self.map_builder_server
            .subscribe_global_slam_optimizations(callback)
    }

    fn unsubscribe_global_slam_optimizations(&self, subscription_index: i32) {
        self.map_builder_server
            .unsubscribe_global_slam_optimizations(subscription_index);
    }

    fn notify_finish_trajectory(&self, trajectory_id: i32) {
        self.map_builder_server
            .notify_finish_trajectory(trajectory_id);
    }

    fn local_trajectory_uploader(&self) -> Option<Arc<dyn LocalTrajectoryUploaderInterface>> {
        self.map_builder_server.local_trajectory_uploader()
    }

    fn enqueue_sensor_data(&self, trajectory_id: i32, data: Box<dyn SensorData>) {
        self.map_builder_server
            .incoming_data_queue
            .push(Box::new(Data {
                trajectory_id,
                data,
            }));
    }

    fn enqueue_local_slam_result_data(
        &self,
        trajectory_id: i32,
        sensor_id: &str,
        local_slam_result_data: &mapping_proto::LocalSlamResultData,
    ) {
        self.map_builder_server
            .incoming_data_queue
            .push(Box::new(Data {
                trajectory_id,
                data: S::create_local_slam_result(
                    sensor_id,
                    local_slam_result_data,
                    &self.submap_controller,
                ),
            }));
    }

    fn register_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32) {
        let mut client_ids = lock(&self.client_ids);
        assert!(
            !client_ids.contains_key(&trajectory_id),
            "Check failed: client_ids_.count(trajectory_id) == 0 (trajectory_id {trajectory_id})"
        );
        log::info!("Registering trajectory_id {trajectory_id} to client_id {client_id}");
        client_ids.insert(trajectory_id, client_id.to_string());
    }

    fn check_client_id_for_trajectory(&self, client_id: &str, trajectory_id: i32) -> bool {
        lock(&self.client_ids)
            .get(&trajectory_id)
            .is_some_and(|registered| registered == client_id)
    }
}
