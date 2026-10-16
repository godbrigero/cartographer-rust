//! Port of `cartographer/mapping/map_builder` (.h, .cc).
//!
//! `CreateMapBuilderOptions` is defined in `map_builder_interface.cc` in C++
//! (and declared in `map_builder_interface.h`, which `map_builder.h`
//! includes); it is re-exported here for convenience.
//!
//! # Ownership
//!
//! RUST PORT NOTE: the C++ `MapBuilder` owns a `common::ThreadPool`, a
//! `std::unique_ptr<PoseGraph>` (statically cast to `PoseGraph2D*` /
//! `PoseGraph3D*` when creating trajectory builders), a
//! `std::unique_ptr<CollatorInterface>` and the trajectory builders, which
//! hold raw pointers to the pose graph and the collator. Here:
//!
//! * the thread pool is an `Arc<ThreadPool>` shared with the pose graph;
//! * the pose graph is a [`MapBuilderPoseGraph`] enum over
//!   `Arc<PoseGraph2D>` / `Arc<PoseGraph3D>` (the C++ `static_cast`s become a
//!   `match`), shared with the global trajectory builders;
//! * the collator is a [`SharedCollator`] (`Arc<Mutex<dyn
//!   CollatorInterface>>`), shared with the collated trajectory builders,
//!   whose collator callbacks own the wrapped global trajectory builders.
//!
//! Fields are declared (and therefore dropped) in the order trajectory
//! builders, collator, pose graph, thread pool: once the builders and the
//! collator callbacks are gone, the map builder holds the last pose graph
//! handle (unless the user kept one from [`MapBuilder::pose_graph`]), whose
//! drop waits for all background computations; the thread pool is dropped
//! after it (C++ declares `thread_pool_` before `pose_graph_`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, PoisonError};

use crate::common::thread_pool::{ThreadPool, ThreadPoolInterface};
use crate::common::time::from_universal;
use crate::io::internal::mapping_state_serialization::{
    FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS, write_pb_stream,
};
use crate::io::proto_stream::{ProtoStreamReader, ProtoStreamWriter};
use crate::io::proto_stream_deserializer::ProtoStreamDeserializer;
use crate::io::proto_stream_interface::{ProtoStreamReaderInterface, ProtoStreamWriterInterface};
use crate::io::serialization_format_migration::migrate_submap_format_version1_to_version2;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::internal::collated_trajectory_builder::{
    CollatedTrajectoryBuilder, SharedCollator,
};
use crate::mapping::internal::global_trajectory_builder::{
    create_global_trajectory_builder_2d, create_global_trajectory_builder_3d,
};
use crate::mapping::internal::motion_filter::MotionFilter;
use crate::mapping::internal::optimization::optimization_problem_2d::OptimizationProblem2D;
use crate::mapping::internal::optimization::optimization_problem_3d::OptimizationProblem3D;
use crate::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
use crate::mapping::internal::three_d::pose_graph_3d::PoseGraph3D;
use crate::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use crate::mapping::internal::two_d::pose_graph_2d::PoseGraph2D;
use crate::mapping::map_builder_interface::MapBuilderInterface;
pub use crate::mapping::map_builder_interface::create_map_builder_options;
use crate::mapping::pose_graph::{self, PoseGraph};
use crate::mapping::pose_graph_interface::PoseGraphInterface;
use crate::mapping::pose_graph_trimmer::PureLocalizationTrimmer;
use crate::mapping::proto::{self, SerializedData, serialized_data::Data};
use crate::mapping::trajectory_builder_interface::{
    self, LocalSlamResultCallback, SensorId, SensorType, TrajectoryBuilderInterface,
};
use crate::sensor;
use crate::sensor::internal::collator::Collator;
use crate::sensor::internal::trajectory_collator::TrajectoryCollator;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::to_rigid3;

fn select_range_sensor_ids(expected_sensor_ids: &BTreeSet<SensorId>) -> Vec<String> {
    expected_sensor_ids
        .iter()
        .filter(|sensor_id| sensor_id.r#type == SensorType::Range)
        .map(|sensor_id| sensor_id.id.clone())
        .collect()
}

fn maybe_add_pure_localization_trimmer(
    trajectory_id: i32,
    trajectory_options: &proto::TrajectoryBuilderOptions,
    pose_graph: &dyn PoseGraph,
) {
    // The deprecated field is still honoured, like in C++.
    #[allow(deprecated)]
    let pure_localization = trajectory_options.pure_localization;
    if pure_localization {
        log::warn!(
            "'TrajectoryBuilderOptions::pure_localization' field is deprecated. Use \
             'TrajectoryBuilderOptions::pure_localization_trimmer' instead."
        );
        pose_graph.add_trimmer(Box::new(PureLocalizationTrimmer::new(
            trajectory_id,
            3, /* max_submaps_to_keep */
        )));
        return;
    }
    if let Some(pure_localization_trimmer) = &trajectory_options.pure_localization_trimmer {
        pose_graph.add_trimmer(Box::new(PureLocalizationTrimmer::new(
            trajectory_id,
            pure_localization_trimmer.max_submaps_to_keep,
        )));
    }
}

/// The pose graph of a [`MapBuilder`]: 2D or 3D, as selected by the
/// `MapBuilderOptions` (replaces the C++ `std::unique_ptr<PoseGraph>` plus
/// `static_cast`s to the concrete type).
#[derive(Clone, Debug)]
pub enum MapBuilderPoseGraph {
    /// `use_trajectory_builder_2d`.
    TwoD(Arc<PoseGraph2D>),
    /// `use_trajectory_builder_3d`.
    ThreeD(Arc<PoseGraph3D>),
}

impl MapBuilderPoseGraph {
    /// The pose graph's write API (C++ `PoseGraph*`).
    pub fn as_pose_graph(&self) -> &dyn PoseGraph {
        match self {
            MapBuilderPoseGraph::TwoD(pose_graph) => pose_graph.as_ref(),
            MapBuilderPoseGraph::ThreeD(pose_graph) => pose_graph.as_ref(),
        }
    }

    /// A shared handle to the pose graph's write API.
    pub fn to_pose_graph(&self) -> Arc<dyn PoseGraph> {
        match self {
            MapBuilderPoseGraph::TwoD(pose_graph) => Arc::clone(pose_graph) as Arc<dyn PoseGraph>,
            MapBuilderPoseGraph::ThreeD(pose_graph) => Arc::clone(pose_graph) as Arc<dyn PoseGraph>,
        }
    }

    /// A shared handle to the pose graph's read API.
    pub fn to_pose_graph_interface(&self) -> Arc<dyn PoseGraphInterface> {
        match self {
            MapBuilderPoseGraph::TwoD(pose_graph) => {
                Arc::clone(pose_graph) as Arc<dyn PoseGraphInterface>
            }
            MapBuilderPoseGraph::ThreeD(pose_graph) => {
                Arc::clone(pose_graph) as Arc<dyn PoseGraphInterface>
            }
        }
    }
}

/// Wires up the complete SLAM stack with `TrajectoryBuilder`s (for local
/// submaps) and a `PoseGraph` for loop closure.
///
/// See the module documentation for the ownership model. `MapBuilder` is
/// not internally synchronized (like C++): it is driven through `&mut self`
/// from one thread at a time, while the pose graph handle returned by
/// [`MapBuilder::pose_graph`] may be queried from any thread.
pub struct MapBuilder {
    options: proto::MapBuilderOptions,
    // Dropped first: they hold handles to the collator and the pose graph.
    trajectory_builders: Vec<Option<Box<dyn TrajectoryBuilderInterface>>>,
    // Its per-trajectory callbacks own the wrapped global trajectory
    // builders.
    sensor_collator: SharedCollator,
    pose_graph: MapBuilderPoseGraph,
    // Dropped last (C++ declares it first).
    thread_pool: Arc<ThreadPool>,
    all_trajectory_builder_options: Vec<proto::TrajectoryBuilderOptionsWithSensorIds>,
}

impl std::fmt::Debug for MapBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapBuilder")
            .field("options", &self.options)
            .field("num_trajectory_builders", &self.trajectory_builders.len())
            .field("pose_graph", &self.pose_graph)
            .finish_non_exhaustive()
    }
}

impl MapBuilder {
    /// Creates the thread pool, the 2D or 3D pose graph and the sensor
    /// collator. Panics unless exactly one of `use_trajectory_builder_2d` and
    /// `use_trajectory_builder_3d` is set.
    pub fn new(options: &proto::MapBuilderOptions) -> Self {
        assert!(
            options.use_trajectory_builder_2d ^ options.use_trajectory_builder_3d,
            "Check failed: options.use_trajectory_builder_2d() ^ \
             options.use_trajectory_builder_3d()"
        );
        let thread_pool = Arc::new(ThreadPool::new(options.num_background_threads));
        let pose_graph_options = options.pose_graph_options.clone().unwrap_or_default();
        let optimization_problem_options = pose_graph_options
            .optimization_problem_options
            .unwrap_or_default();
        let pool = Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>;
        let pose_graph = if options.use_trajectory_builder_2d {
            MapBuilderPoseGraph::TwoD(Arc::new(PoseGraph2D::new(
                &pose_graph_options,
                Box::new(OptimizationProblem2D::new(&optimization_problem_options)),
                pool,
            )))
        } else {
            MapBuilderPoseGraph::ThreeD(Arc::new(PoseGraph3D::new(
                &pose_graph_options,
                Box::new(OptimizationProblem3D::new(&optimization_problem_options)),
                pool,
            )))
        };
        let sensor_collator: SharedCollator = if options.collate_by_trajectory {
            Arc::new(Mutex::new(TrajectoryCollator::new()))
        } else {
            Arc::new(Mutex::new(Collator::new()))
        };
        MapBuilder {
            options: options.clone(),
            trajectory_builders: Vec::new(),
            sensor_collator,
            pose_graph,
            thread_pool,
            all_trajectory_builder_options: Vec::new(),
        }
    }

    /// The concrete (2D or 3D) pose graph, with its full write API.
    pub fn pose_graph_handle(&self) -> &MapBuilderPoseGraph {
        &self.pose_graph
    }

    /// The options this map builder was created with.
    pub fn options(&self) -> &proto::MapBuilderOptions {
        &self.options
    }

    /// The thread pool running the pose graph's background work.
    pub fn thread_pool(&self) -> &Arc<ThreadPool> {
        &self.thread_pool
    }
}

impl MapBuilderInterface for MapBuilder {
    fn add_trajectory_builder(
        &mut self,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &proto::TrajectoryBuilderOptions,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> i32 {
        let trajectory_id = self.num_trajectory_builders();

        let mut pose_graph_odometry_motion_filter = None;
        if let Some(motion_filter_options) = &trajectory_options.pose_graph_odometry_motion_filter {
            log::info!("Using a motion filter for adding odometry to the pose graph.");
            pose_graph_odometry_motion_filter = Some(MotionFilter::new(motion_filter_options));
        }

        let global_trajectory_builder = match &self.pose_graph {
            MapBuilderPoseGraph::ThreeD(pose_graph) => {
                let local_trajectory_builder = trajectory_options
                    .trajectory_builder_3d_options
                    .as_ref()
                    .map(|options| {
                        LocalTrajectoryBuilder3D::new(
                            options,
                            &select_range_sensor_ids(expected_sensor_ids),
                        )
                    });
                create_global_trajectory_builder_3d(
                    local_trajectory_builder,
                    trajectory_id,
                    Arc::clone(pose_graph),
                    local_slam_result_callback,
                    pose_graph_odometry_motion_filter,
                )
            }
            MapBuilderPoseGraph::TwoD(pose_graph) => {
                let local_trajectory_builder = trajectory_options
                    .trajectory_builder_2d_options
                    .as_ref()
                    .map(|options| {
                        LocalTrajectoryBuilder2D::new(
                            options,
                            &select_range_sensor_ids(expected_sensor_ids),
                        )
                    });
                create_global_trajectory_builder_2d(
                    local_trajectory_builder,
                    trajectory_id,
                    Arc::clone(pose_graph),
                    local_slam_result_callback,
                    pose_graph_odometry_motion_filter,
                )
            }
        };
        self.trajectory_builders
            .push(Some(Box::new(CollatedTrajectoryBuilder::new(
                trajectory_options,
                Arc::clone(&self.sensor_collator),
                trajectory_id,
                expected_sensor_ids,
                global_trajectory_builder,
            ))));
        let pose_graph = self.pose_graph.as_pose_graph();
        maybe_add_pure_localization_trimmer(trajectory_id, trajectory_options, pose_graph);

        if let Some(initial_trajectory_pose) = &trajectory_options.initial_trajectory_pose {
            pose_graph.set_initial_trajectory_pose(
                trajectory_id,
                initial_trajectory_pose.to_trajectory_id,
                &to_rigid3(&initial_trajectory_pose.relative_pose.unwrap_or_default()),
                from_universal(initial_trajectory_pose.timestamp),
            );
        }
        let options_with_sensor_ids_proto = proto::TrajectoryBuilderOptionsWithSensorIds {
            sensor_id: expected_sensor_ids
                .iter()
                .map(trajectory_builder_interface::to_proto)
                .collect(),
            trajectory_builder_options: Some(trajectory_options.clone()),
        };
        self.all_trajectory_builder_options
            .push(options_with_sensor_ids_proto);
        assert_eq!(
            self.trajectory_builders.len(),
            self.all_trajectory_builder_options.len(),
            "Check failed: trajectory_builders_.size() == all_trajectory_builder_options_.size()"
        );
        trajectory_id
    }

    fn add_trajectory_for_deserialization(
        &mut self,
        options_with_sensor_ids_proto: &proto::TrajectoryBuilderOptionsWithSensorIds,
    ) -> i32 {
        let trajectory_id = self.num_trajectory_builders();
        self.trajectory_builders.push(None);
        self.all_trajectory_builder_options
            .push(options_with_sensor_ids_proto.clone());
        assert_eq!(
            self.trajectory_builders.len(),
            self.all_trajectory_builder_options.len(),
            "Check failed: trajectory_builders_.size() == all_trajectory_builder_options_.size()"
        );
        trajectory_id
    }

    /// RUST PORT NOTE: like the C++ `std::vector::at`, panics if
    /// `trajectory_id` is out of range; returns `None` for trajectories
    /// created by deserialization.
    fn get_trajectory_builder(
        &mut self,
        trajectory_id: i32,
    ) -> Option<&mut dyn TrajectoryBuilderInterface> {
        let num_trajectory_builders = self.trajectory_builders.len();
        let builder = usize::try_from(trajectory_id)
            .ok()
            .and_then(|index| self.trajectory_builders.get_mut(index))
            .unwrap_or_else(|| {
                panic!(
                    "trajectory_builders_.at({trajectory_id}): out of range (size \
                     {num_trajectory_builders})"
                )
            });
        match builder {
            Some(builder) => Some(builder.as_mut()),
            None => None,
        }
    }

    fn finish_trajectory(&mut self, trajectory_id: i32) {
        self.sensor_collator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .finish_trajectory(trajectory_id);
        self.pose_graph
            .as_pose_graph()
            .finish_trajectory(trajectory_id);
    }

    fn submap_to_proto(
        &mut self,
        submap_id: &SubmapId,
        response: &mut proto::submap_query::Response,
    ) -> String {
        if submap_id.trajectory_id < 0 || submap_id.trajectory_id >= self.num_trajectory_builders()
        {
            return format!(
                "Requested submap from trajectory {} but there are only {} trajectories.",
                submap_id.trajectory_id,
                self.num_trajectory_builders()
            );
        }

        let submap_data = self.pose_graph.as_pose_graph().get_submap_data(submap_id);
        let Some(submap) = submap_data.submap else {
            return format!(
                "Requested submap {} from trajectory {} but it does not exist: maybe it has \
                 been trimmed.",
                submap_id.submap_index, submap_id.trajectory_id
            );
        };
        submap
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .to_response_proto(&submap_data.pose, response);
        String::new()
    }

    fn serialize_state(
        &mut self,
        include_unfinished_submaps: bool,
        writer: &mut dyn ProtoStreamWriterInterface,
    ) {
        write_pb_stream(
            self.pose_graph.as_pose_graph(),
            &self.all_trajectory_builder_options,
            writer,
            include_unfinished_submaps,
        );
    }

    fn serialize_state_to_file(
        &mut self,
        include_unfinished_submaps: bool,
        filename: &str,
    ) -> bool {
        let mut writer = ProtoStreamWriter::new(filename);
        write_pb_stream(
            self.pose_graph.as_pose_graph(),
            &self.all_trajectory_builder_options,
            &mut writer,
            include_unfinished_submaps,
        );
        writer.close()
    }

    /// RUST PORT NOTE: for 3D, C++ `CHECK`s that the stream is not of format
    /// version 1 (submaps without rotational scan matcher histograms) and
    /// asks the user to run `pbstream migrate`. Here such streams are
    /// migrated on the fly instead: their submaps are collected and, once
    /// all nodes are read, completed with
    /// `io::migrate_submap_format_version1_to_version2` (what `pbstream
    /// migrate` does) before being added to the pose graph.
    fn load_state(
        &mut self,
        reader: &mut dyn ProtoStreamReaderInterface,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        let mut trajectory_remapping: BTreeMap<i32, i32> = BTreeMap::new();
        {
            let mut deserializer = ProtoStreamDeserializer::new(&mut *reader);

            // Create a copy of the pose_graph_proto, such that we can
            // re-write the trajectory ids.
            let mut pose_graph_proto = deserializer.pose_graph().clone();
            let all_builder_options_proto = deserializer.all_trajectory_builder_options().clone();
            let format_version = deserializer.header().format_version;

            for (i, trajectory_proto) in pose_graph_proto.trajectory.iter_mut().enumerate() {
                let options_with_sensor_ids_proto =
                    &all_builder_options_proto.options_with_sensor_ids[i];
                let new_trajectory_id =
                    self.add_trajectory_for_deserialization(options_with_sensor_ids_proto);
                let previous =
                    trajectory_remapping.insert(trajectory_proto.trajectory_id, new_trajectory_id);
                assert!(
                    previous.is_none(),
                    "Duplicate trajectory ID: {}",
                    trajectory_proto.trajectory_id
                );
                trajectory_proto.trajectory_id = new_trajectory_id;
                if load_frozen_state {
                    self.pose_graph
                        .as_pose_graph()
                        .freeze_trajectory(new_trajectory_id);
                }
            }
            let remap = |trajectory_id: i32| -> i32 {
                *trajectory_remapping
                    .get(&trajectory_id)
                    .unwrap_or_else(|| panic!("map::at: unknown trajectory id {trajectory_id}"))
            };

            // Apply the calculated remapping to constraints in the pose graph
            // proto.
            for constraint_proto in &mut pose_graph_proto.constraint {
                let submap_id = constraint_proto
                    .submap_id
                    .get_or_insert_with(Default::default);
                submap_id.trajectory_id = remap(submap_id.trajectory_id);
                let node_id = constraint_proto
                    .node_id
                    .get_or_insert_with(Default::default);
                node_id.trajectory_id = remap(node_id.trajectory_id);
            }

            let mut submap_poses: MapById<SubmapId, Rigid3d> = MapById::new();
            for trajectory_proto in &pose_graph_proto.trajectory {
                for submap_proto in &trajectory_proto.submap {
                    submap_poses.insert(
                        &SubmapId::new(trajectory_proto.trajectory_id, submap_proto.submap_index),
                        to_rigid3(&submap_proto.pose.unwrap_or_default()),
                    );
                }
            }

            let mut node_poses: MapById<NodeId, Rigid3d> = MapById::new();
            for trajectory_proto in &pose_graph_proto.trajectory {
                for node_proto in &trajectory_proto.node {
                    node_poses.insert(
                        &NodeId::new(trajectory_proto.trajectory_id, node_proto.node_index),
                        to_rigid3(&node_proto.pose.unwrap_or_default()),
                    );
                }
            }

            let pose_graph = self.pose_graph.as_pose_graph();
            // Set global poses of landmarks.
            for landmark in &pose_graph_proto.landmark_poses {
                pose_graph.set_landmark_pose(
                    &landmark.landmark_id,
                    &to_rigid3(&landmark.global_pose.unwrap_or_default()),
                    true,
                );
            }

            let migrate_submaps = self.options.use_trajectory_builder_3d
                && format_version == FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS;
            if migrate_submaps {
                log::warn!(
                    "The pbstream file contains submaps without rotational histograms. They are \
                     migrated while loading; use the 'pbstream migrate' tool to convert the \
                     file permanently."
                );
            }
            let mut submaps_to_migrate: MapById<SubmapId, proto::Submap> = MapById::new();
            let mut nodes_for_migration: MapById<NodeId, proto::Node> = MapById::new();

            let mut proto = SerializedData::default();
            while deserializer.read_next_serialized_data(&mut proto) {
                match &mut proto.data {
                    Some(Data::PoseGraph(_)) => {
                        log::error!(
                            "Found multiple serialized `PoseGraph`. Serialized stream likely \
                             corrupt!."
                        );
                    }
                    Some(Data::AllTrajectoryBuilderOptions(_)) => {
                        log::error!(
                            "Found multiple serialized `AllTrajectoryBuilderOptions`. Serialized \
                             stream likely corrupt!."
                        );
                    }
                    Some(Data::Submap(submap)) => {
                        let submap_id_proto = submap.submap_id.get_or_insert_with(Default::default);
                        submap_id_proto.trajectory_id = remap(submap_id_proto.trajectory_id);
                        let submap_id = SubmapId::new(
                            submap_id_proto.trajectory_id,
                            submap_id_proto.submap_index,
                        );
                        if migrate_submaps {
                            submaps_to_migrate.insert(&submap_id, submap.clone());
                        } else {
                            pose_graph.add_submap_from_proto(submap_poses.at(&submap_id), submap);
                        }
                    }
                    Some(Data::Node(node)) => {
                        let node_id_proto = node.node_id.get_or_insert_with(Default::default);
                        node_id_proto.trajectory_id = remap(node_id_proto.trajectory_id);
                        let node_id =
                            NodeId::new(node_id_proto.trajectory_id, node_id_proto.node_index);
                        let node_pose = node_poses.at(&node_id);
                        pose_graph.add_node_from_proto(node_pose, node);
                        if migrate_submaps {
                            nodes_for_migration.insert(&node_id, node.clone());
                        }
                    }
                    Some(Data::TrajectoryData(trajectory_data)) => {
                        trajectory_data.trajectory_id = remap(trajectory_data.trajectory_id);
                        pose_graph.set_trajectory_data_from_proto(trajectory_data);
                    }
                    Some(Data::ImuData(imu_data)) => {
                        if load_frozen_state {
                            continue;
                        }
                        pose_graph.add_imu_data(
                            remap(imu_data.trajectory_id),
                            &sensor::imu_data::from_proto(&imu_data.imu_data.unwrap_or_default()),
                        );
                    }
                    Some(Data::OdometryData(odometry_data)) => {
                        if load_frozen_state {
                            continue;
                        }
                        pose_graph.add_odometry_data(
                            remap(odometry_data.trajectory_id),
                            &sensor::odometry_data::from_proto(
                                &odometry_data.odometry_data.unwrap_or_default(),
                            ),
                        );
                    }
                    Some(Data::FixedFramePoseData(fixed_frame_pose_data)) => {
                        if load_frozen_state {
                            continue;
                        }
                        pose_graph.add_fixed_frame_pose_data(
                            remap(fixed_frame_pose_data.trajectory_id),
                            &sensor::fixed_frame_pose_data::from_proto(
                                &fixed_frame_pose_data
                                    .fixed_frame_pose_data
                                    .unwrap_or_default(),
                            ),
                        );
                    }
                    Some(Data::LandmarkData(landmark_data)) => {
                        if load_frozen_state {
                            continue;
                        }
                        pose_graph.add_landmark_data(
                            remap(landmark_data.trajectory_id),
                            &sensor::landmark_data::from_proto(
                                &landmark_data.landmark_data.clone().unwrap_or_default(),
                            ),
                        );
                    }
                    None => {
                        log::warn!(
                            "Skipping unknown message type in stream: \
                             cartographer.mapping.proto.SerializedData"
                        );
                    }
                }
            }

            if migrate_submaps {
                let migrated_submaps = migrate_submap_format_version1_to_version2(
                    &submaps_to_migrate,
                    &nodes_for_migration,
                    &pose_graph_proto,
                );
                for submap_id_data in &migrated_submaps {
                    pose_graph.add_submap_from_proto(
                        submap_poses.at(&submap_id_data.id),
                        submap_id_data.data,
                    );
                }
            }

            if load_frozen_state {
                // Add information about which nodes belong to which submap.
                // This is required, even without constraints.
                for constraint_proto in &pose_graph_proto.constraint {
                    if constraint_proto.tag
                        != proto::pose_graph::constraint::Tag::IntraSubmap as i32
                    {
                        continue;
                    }
                    let node_id = constraint_proto.node_id.unwrap_or_default();
                    let submap_id = constraint_proto.submap_id.unwrap_or_default();
                    pose_graph.add_node_to_submap(
                        &NodeId::new(node_id.trajectory_id, node_id.node_index),
                        &SubmapId::new(submap_id.trajectory_id, submap_id.submap_index),
                    );
                }
            } else {
                // When loading unfrozen trajectories, 'AddSerializedConstraints'
                // will take care of adding information about which nodes
                // belong to which submap.
                pose_graph.add_serialized_constraints(&pose_graph::from_proto(
                    &pose_graph_proto.constraint,
                ));
            }
        }
        assert!(reader.eof(), "Check failed: reader->eof()");
        trajectory_remapping
    }

    fn load_state_from_file(
        &mut self,
        filename: &str,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        const SUFFIX: &str = ".pbstream";
        if !filename.ends_with(SUFFIX) {
            log::warn!("The file containing the state should be a .pbstream file.");
        }
        log::info!("Loading saved state '{filename}'...");
        let mut stream = ProtoStreamReader::new(filename);
        self.load_state(&mut stream, load_frozen_state)
    }

    fn num_trajectory_builders(&self) -> i32 {
        i32::try_from(self.trajectory_builders.len()).expect("too many trajectory builders")
    }

    fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
        self.pose_graph.to_pose_graph_interface()
    }

    fn get_all_trajectory_builder_options(
        &self,
    ) -> &[proto::TrajectoryBuilderOptionsWithSensorIds] {
        &self.all_trajectory_builder_options
    }
}

/// Creates a [`MapBuilder`] behind the [`MapBuilderInterface`] (C++
/// `CreateMapBuilder`).
pub fn create_map_builder(options: &proto::MapBuilderOptions) -> Box<dyn MapBuilderInterface> {
    Box::new(MapBuilder::new(options))
}

#[cfg(test)]
mod tests {
    //! Port of `map_builder_test.cc`. The C++ value-parameterized suites
    //! (`MapBuilderTestByGridType`, `MapBuilderTestByGridTypeAndDimensions`)
    //! become one `#[test]` per parameter value; additional tests cover
    //! `collate_by_trajectory = true` and API not exercised by the C++ tests.

    use super::*;
    use crate::common::time::{from_seconds, to_universal};
    use crate::io::internal::in_memory_proto_stream::{
        ForwardingProtoStreamWriter, InMemoryProtoStreamReader,
    };
    use crate::io::proto_stream_interface::ProtoMessage;
    use crate::mapping::internal::testing::test_helpers::{
        generate_fake_range_measurements, generate_fake_range_measurements_with_transform,
        resolve_lua_parameters,
    };
    use crate::mapping::pose_graph_interface::Tag;
    use crate::mapping::proto::grid_options2_d::GridType;
    use crate::mapping::proto::range_data_inserter_options::RangeDataInserterType;
    use crate::mapping::trajectory_builder_interface::create_trajectory_builder_options;
    use crate::sensor::imu_data::ImuData;
    use crate::transform::rigid_transform::AngleAxis;
    use nalgebra::{Quaternion, Vector3};

    fn range_sensor_id() -> SensorId {
        SensorId {
            r#type: SensorType::Range,
            id: "range".to_string(),
        }
    }

    fn imu_sensor_id() -> SensorId {
        SensorId {
            r#type: SensorType::Imu,
            id: "imu".to_string(),
        }
    }

    const DURATION: f64 = 4.; // Seconds.
    const TIME_STEP: f64 = 0.1; // Seconds.
    const TRAVEL_DISTANCE: f64 = 1.2; // Meters.

    fn temp_filename(name: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "cartographer_map_builder_{}_{name}.pbstream",
                std::process::id()
            ))
            .to_str()
            .unwrap()
            .to_string()
    }

    /// C++ `MapBuilderTestBase`.
    struct MapBuilderTestBase {
        map_builder: Option<Box<dyn MapBuilderInterface>>,
        map_builder_options: proto::MapBuilderOptions,
        trajectory_builder_options: proto::TrajectoryBuilderOptions,
        local_slam_result_poses: Arc<Mutex<Vec<Rigid3d>>>,
    }

    impl MapBuilderTestBase {
        /// C++ `SetUp`.
        fn set_up() -> Self {
            // Global SLAM optimization is not executed.
            let map_builder_options = create_map_builder_options(&resolve_lua_parameters(
                r#"
                include "map_builder.lua"
                MAP_BUILDER.use_trajectory_builder_2d = true
                MAP_BUILDER.pose_graph.optimize_every_n_nodes = 0
                MAP_BUILDER.pose_graph.global_sampling_ratio = 0.05
                MAP_BUILDER.pose_graph.global_constraint_search_after_n_seconds = 0
                return MAP_BUILDER"#,
            ));
            // Multiple submaps are created because of a small
            // 'num_range_data'.
            let trajectory_builder_options =
                create_trajectory_builder_options(&resolve_lua_parameters(
                    r#"
                    include "trajectory_builder.lua"
                    TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false
                    TRAJECTORY_BUILDER.trajectory_builder_2d.submaps.num_range_data = 4
                    TRAJECTORY_BUILDER.trajectory_builder_3d.submaps.num_range_data = 4
                    return TRAJECTORY_BUILDER"#,
                ));
            MapBuilderTestBase {
                map_builder: None,
                map_builder_options,
                trajectory_builder_options,
                local_slam_result_poses: Arc::default(),
            }
        }

        fn build_map_builder(&mut self) {
            // Drop the previous map builder (and wait for its computations)
            // first.
            self.map_builder = None;
            self.map_builder = Some(create_map_builder(&self.map_builder_options));
        }

        fn map_builder(&mut self) -> &mut dyn MapBuilderInterface {
            self.map_builder.as_deref_mut().unwrap()
        }

        fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
            self.map_builder.as_ref().unwrap().pose_graph()
        }

        fn set_options_to_3d(&mut self) {
            self.map_builder_options.use_trajectory_builder_2d = false;
            self.map_builder_options.use_trajectory_builder_3d = true;
        }

        fn set_options_to_tsdf_2d(&mut self) {
            let options_2d = self
                .trajectory_builder_options
                .trajectory_builder_2d_options
                .as_mut()
                .unwrap();
            let submaps_options = options_2d.submaps_options.as_mut().unwrap();
            submaps_options
                .range_data_inserter_options
                .as_mut()
                .unwrap()
                .range_data_inserter_type = RangeDataInserterType::TsdfInserter2d as i32;
            submaps_options.grid_options_2d.as_mut().unwrap().grid_type = GridType::Tsdf as i32;
            options_2d
                .ceres_scan_matcher_options
                .as_mut()
                .unwrap()
                .occupied_space_weight = 10.0;
            self.map_builder_options
                .pose_graph_options
                .as_mut()
                .unwrap()
                .constraint_builder_options
                .as_mut()
                .unwrap()
                .ceres_scan_matcher_options
                .as_mut()
                .unwrap()
                .occupied_space_weight = 50.0;
        }

        fn set_options_enable_global_optimization(&mut self) {
            self.map_builder_options
                .pose_graph_options
                .as_mut()
                .unwrap()
                .optimize_every_n_nodes = 3;
            self.trajectory_builder_options
                .trajectory_builder_2d_options
                .as_mut()
                .unwrap()
                .motion_filter_options
                .as_mut()
                .unwrap()
                .max_distance_meters = 0.;
        }

        fn get_local_slam_result_callback(&self) -> LocalSlamResultCallback {
            let local_slam_result_poses = Arc::clone(&self.local_slam_result_poses);
            Arc::new(move |_, _, local_pose, _, _| {
                local_slam_result_poses.lock().unwrap().push(local_pose);
            })
        }

        fn local_slam_result_poses(&self) -> Vec<Rigid3d> {
            self.local_slam_result_poses.lock().unwrap().clone()
        }

        fn add_trajectory(&mut self, sensor_ids: &[SensorId]) -> i32 {
            let callback = self.get_local_slam_result_callback();
            let options = self.trajectory_builder_options.clone();
            self.map_builder().add_trajectory_builder(
                &sensor_ids.iter().cloned().collect(),
                &options,
                Some(callback),
            )
        }

        /// C++ `CreateTrajectoryWithFakeData`.
        fn create_trajectory_with_fake_data(&mut self, timestamps_add_duration: f64) -> i32 {
            let trajectory_id = self.add_trajectory(&[range_sensor_id()]);
            let trajectory_builder = self
                .map_builder()
                .get_trajectory_builder(trajectory_id)
                .unwrap();
            let measurements =
                generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP);
            for mut measurement in measurements {
                measurement.time += from_seconds(timestamps_add_duration);
                trajectory_builder.add_timed_point_cloud_data(&range_sensor_id().id, &measurement);
            }
            self.map_builder().finish_trajectory(trajectory_id);
            trajectory_id
        }

        /// Feeds the fake range data (and, if `with_imu`, IMU data
        /// `(0, 0, 9.8)` at each measurement time) to `trajectory_id`.
        /// Returns the number of range measurements.
        fn feed_fake_data(&mut self, trajectory_id: i32, with_imu: bool) -> usize {
            let trajectory_builder = self
                .map_builder()
                .get_trajectory_builder(trajectory_id)
                .unwrap();
            let measurements =
                generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP);
            for measurement in &measurements {
                trajectory_builder.add_timed_point_cloud_data(&range_sensor_id().id, measurement);
                if with_imu {
                    trajectory_builder.add_imu_data(
                        &imu_sensor_id().id,
                        &ImuData {
                            time: measurement.time,
                            linear_acceleration: Vector3::new(0., 0., 9.8),
                            angular_velocity: Vector3::zeros(),
                        },
                    );
                }
            }
            measurements.len()
        }

        /// `EXPECT_NEAR(kTravelDistance, |last - first|, factor *
        /// kTravelDistance)` on the local SLAM result poses.
        fn assert_travel_distance(&self, tolerance_factor: f64) {
            let poses = self.local_slam_result_poses();
            let distance =
                (poses.last().unwrap().translation() - poses.first().unwrap().translation()).norm();
            assert!(
                (distance - TRAVEL_DISTANCE).abs() <= tolerance_factor * TRAVEL_DISTANCE,
                "travelled {distance}"
            );
        }
    }

    fn num_inter_submap_constraints(pose_graph: &dyn PoseGraphInterface) -> usize {
        pose_graph
            .constraints()
            .iter()
            .filter(|constraint| constraint.tag == Tag::InterSubmap)
            .count()
    }

    fn assert_final_distance(
        pose_graph: &dyn PoseGraphInterface,
        trajectory_id: i32,
        test: &MapBuilderTestBase,
    ) {
        let final_pose = pose_graph.get_local_to_global_transform(trajectory_id)
            * *test.local_slam_result_poses().last().unwrap();
        let distance = final_pose.translation().norm();
        assert!(
            (distance - TRAVEL_DISTANCE).abs() <= 0.1 * TRAVEL_DISTANCE,
            "final distance {distance}"
        );
    }

    // TEST_P(MapBuilderTestByGridTypeAndDimensions, TrajectoryAddFinish)
    fn trajectory_add_finish(grid_type: GridType, dimensions: i32, collate_by_trajectory: bool) {
        let mut test = MapBuilderTestBase::set_up();
        if dimensions == 3 {
            test.set_options_to_3d();
        }
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.map_builder_options.collate_by_trajectory = collate_by_trajectory;
        test.build_map_builder();
        let options = test.trajectory_builder_options.clone();
        let trajectory_id = test.map_builder().add_trajectory_builder(
            &[range_sensor_id()].into_iter().collect(),
            &options,
            None, /* GetLocalSlamResultCallbackForSubscriptions */
        );
        assert_eq!(1, test.map_builder().num_trajectory_builders());
        assert!(
            test.map_builder()
                .get_trajectory_builder(trajectory_id)
                .is_some()
        );
        let pose_graph = test.pose_graph();
        test.map_builder().finish_trajectory(trajectory_id);
        pose_graph.run_final_optimization();
        assert!(pose_graph.is_trajectory_finished(trajectory_id));
    }

    #[test]
    fn trajectory_add_finish_probability_grid_2d() {
        trajectory_add_finish(GridType::ProbabilityGrid, 2, false);
    }

    #[test]
    fn trajectory_add_finish_probability_grid_3d() {
        trajectory_add_finish(GridType::ProbabilityGrid, 3, false);
    }

    #[test]
    fn trajectory_add_finish_tsdf_2d() {
        trajectory_add_finish(GridType::Tsdf, 2, false);
    }

    #[test]
    fn trajectory_add_finish_probability_grid_2d_collate_by_trajectory() {
        trajectory_add_finish(GridType::ProbabilityGrid, 2, true);
    }

    #[test]
    fn trajectory_add_finish_probability_grid_3d_collate_by_trajectory() {
        trajectory_add_finish(GridType::ProbabilityGrid, 3, true);
    }

    // TEST_P(MapBuilderTestByGridType, LocalSlam2D)
    fn local_slam_2d(grid_type: GridType, collate_by_trajectory: bool) {
        let mut test = MapBuilderTestBase::set_up();
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.map_builder_options.collate_by_trajectory = collate_by_trajectory;
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id()]);
        let num_measurements = test.feed_fake_data(trajectory_id, false);
        test.map_builder().finish_trajectory(trajectory_id);
        test.pose_graph().run_final_optimization();
        assert_eq!(test.local_slam_result_poses().len(), num_measurements);
        test.assert_travel_distance(0.1);
    }

    #[test]
    fn local_slam_2d_probability_grid() {
        local_slam_2d(GridType::ProbabilityGrid, false);
    }

    #[test]
    fn local_slam_2d_tsdf() {
        local_slam_2d(GridType::Tsdf, false);
    }

    #[test]
    fn local_slam_2d_probability_grid_collate_by_trajectory() {
        local_slam_2d(GridType::ProbabilityGrid, true);
    }

    #[test]
    fn local_slam_2d_tsdf_collate_by_trajectory() {
        local_slam_2d(GridType::Tsdf, true);
    }

    // TEST_F(MapBuilderTest, LocalSlam3D)
    fn local_slam_3d(collate_by_trajectory: bool) {
        let mut test = MapBuilderTestBase::set_up();
        test.set_options_to_3d();
        test.map_builder_options.collate_by_trajectory = collate_by_trajectory;
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id(), imu_sensor_id()]);
        let num_measurements = test.feed_fake_data(trajectory_id, true);
        test.map_builder().finish_trajectory(trajectory_id);
        test.pose_graph().run_final_optimization();
        assert_eq!(test.local_slam_result_poses().len(), num_measurements);
        test.assert_travel_distance(0.1);
    }

    #[test]
    fn local_slam_3d_default_collator() {
        local_slam_3d(false);
    }

    #[test]
    fn local_slam_3d_collate_by_trajectory() {
        local_slam_3d(true);
    }

    // TEST_P(MapBuilderTestByGridType, GlobalSlam2D)
    fn global_slam_2d(grid_type: GridType) {
        let mut test = MapBuilderTestBase::set_up();
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.set_options_enable_global_optimization();
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id()]);
        let num_measurements = test.feed_fake_data(trajectory_id, false);
        test.map_builder().finish_trajectory(trajectory_id);
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        assert_eq!(test.local_slam_result_poses().len(), num_measurements);
        test.assert_travel_distance(0.1);
        assert!(pose_graph.constraints().len() >= 50);
        assert!(num_inter_submap_constraints(pose_graph.as_ref()) > 0);
        assert!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 20
        );
        assert!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 5
        );
        assert_final_distance(pose_graph.as_ref(), trajectory_id, &test);
    }

    #[test]
    fn global_slam_2d_probability_grid() {
        global_slam_2d(GridType::ProbabilityGrid);
    }

    #[test]
    fn global_slam_2d_tsdf() {
        global_slam_2d(GridType::Tsdf);
    }

    // TEST_F(MapBuilderTest, GlobalSlam3D)
    #[test]
    fn global_slam_3d() {
        let mut test = MapBuilderTestBase::set_up();
        test.set_options_to_3d();
        test.set_options_enable_global_optimization();
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id(), imu_sensor_id()]);
        let num_measurements = test.feed_fake_data(trajectory_id, true);
        test.map_builder().finish_trajectory(trajectory_id);
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        assert_eq!(test.local_slam_result_poses().len(), num_measurements);
        test.assert_travel_distance(0.1);
        assert!(pose_graph.constraints().len() >= 10);
        assert!(num_inter_submap_constraints(pose_graph.as_ref()) > 0);
        assert!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 5
        );
        assert!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 2
        );
        assert_final_distance(pose_graph.as_ref(), trajectory_id, &test);
    }

    // TEST_P(MapBuilderTestByGridType, DeleteFinishedTrajectory2D)
    fn delete_finished_trajectory_2d(grid_type: GridType) {
        let mut test = MapBuilderTestBase::set_up();
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.set_options_enable_global_optimization();
        test.build_map_builder();
        let pose_graph = test.pose_graph();
        let trajectory_id = test.create_trajectory_with_fake_data(0.);
        pose_graph.run_final_optimization();
        assert!(pose_graph.is_trajectory_finished(trajectory_id));
        assert!(pose_graph.constraints().len() >= 50);
        assert!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 20
        );
        assert!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 5
        );
        pose_graph.delete_trajectory(trajectory_id);
        let another_trajectory_id = test.create_trajectory_with_fake_data(100.);
        pose_graph.run_final_optimization();
        assert!(pose_graph.is_trajectory_finished(another_trajectory_id));
        assert_eq!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(trajectory_id),
            0
        );
        assert_eq!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(trajectory_id),
            0
        );
        pose_graph.delete_trajectory(another_trajectory_id);
        pose_graph.run_final_optimization();
        assert_eq!(pose_graph.constraints().len(), 0);
        assert_eq!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(another_trajectory_id),
            0
        );
        assert_eq!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(another_trajectory_id),
            0
        );
    }

    #[test]
    fn delete_finished_trajectory_2d_probability_grid() {
        delete_finished_trajectory_2d(GridType::ProbabilityGrid);
    }

    #[test]
    fn delete_finished_trajectory_2d_tsdf() {
        delete_finished_trajectory_2d(GridType::Tsdf);
    }

    // TEST_P(MapBuilderTestByGridTypeAndDimensions, SaveLoadState)
    fn save_load_state(grid_type: GridType, dimensions: i32) {
        let mut test = MapBuilderTestBase::set_up();
        if dimensions == 3 {
            test.set_options_to_3d();
        }
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.trajectory_builder_options
            .trajectory_builder_2d_options
            .as_mut()
            .unwrap()
            .use_imu_data = true;
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id(), imu_sensor_id()]);
        test.feed_fake_data(trajectory_id, true);
        test.map_builder().finish_trajectory(trajectory_id);
        test.pose_graph().run_final_optimization();
        let num_constraints = test.pose_graph().constraints().len();
        let num_nodes = test
            .pose_graph()
            .get_trajectory_nodes()
            .size_of_trajectory_or_zero(trajectory_id);
        assert!(num_constraints > 0);
        assert!(num_nodes > 0);
        // TODO(gaschler): Consider using in-memory to avoid side effects.
        let filename = temp_filename(&format!("SaveLoadState_{grid_type:?}_{dimensions}"));
        let mut writer = ProtoStreamWriter::new(&filename);
        test.map_builder()
            .serialize_state(/*include_unfinished_submaps=*/ true, &mut writer);
        assert!(writer.close());

        // Reset 'map_builder_'.
        test.build_map_builder();
        let mut reader = ProtoStreamReader::new(&filename);
        let trajectory_remapping = test
            .map_builder()
            .load_state(&mut reader, false /* load_frozen_state */);
        test.pose_graph().run_final_optimization();
        assert_eq!(num_constraints, test.pose_graph().constraints().len());
        assert_eq!(trajectory_remapping.len(), 1);
        let new_trajectory_id = *trajectory_remapping.values().next().unwrap();
        assert_eq!(
            num_nodes,
            test.pose_graph()
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(new_trajectory_id)
        );
        std::fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn save_load_state_probability_grid_2d() {
        save_load_state(GridType::ProbabilityGrid, 2);
    }

    #[test]
    fn save_load_state_probability_grid_3d() {
        save_load_state(GridType::ProbabilityGrid, 3);
    }

    #[test]
    fn save_load_state_tsdf_2d() {
        save_load_state(GridType::Tsdf, 2);
    }

    // TEST_P(MapBuilderTestByGridType, LocalizationOnFrozenTrajectory2D)
    fn localization_on_frozen_trajectory_2d(grid_type: GridType) {
        let mut test = MapBuilderTestBase::set_up();
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test.build_map_builder();
        let temp_trajectory_id = test.create_trajectory_with_fake_data(0.);
        test.pose_graph().run_final_optimization();
        assert!(!test.pose_graph().constraints().is_empty());
        assert!(
            test.pose_graph()
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(temp_trajectory_id)
                > 0
        );
        let filename = temp_filename(&format!("LocalizationOnFrozenTrajectory2D_{grid_type:?}"));
        let mut writer = ProtoStreamWriter::new(&filename);
        test.map_builder()
            .serialize_state(/*include_unfinished_submaps=*/ true, &mut writer);
        assert!(writer.close());

        // Reset 'map_builder_'.
        test.local_slam_result_poses.lock().unwrap().clear();
        test.set_options_enable_global_optimization();
        test.build_map_builder();
        let mut reader = ProtoStreamReader::new(&filename);
        test.map_builder()
            .load_state(&mut reader, true /* load_frozen_state */);
        std::fs::remove_file(&filename).unwrap();
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        let trajectory_id = test.add_trajectory(&[range_sensor_id()]);
        let frozen_trajectory_to_global = Rigid3d::new(
            Vector3::new(0.5, 0.4, 0.),
            Quaternion::from(AngleAxis::new(1.2, Vector3::z())),
        );
        let travel_translation: Vector3<f64> =
            Vector3::new(2., 1., 0.).normalize() * TRAVEL_DISTANCE;
        let measurements = generate_fake_range_measurements_with_transform(
            &travel_translation.cast::<f32>(),
            DURATION,
            TIME_STEP,
            &frozen_trajectory_to_global.cast::<f32>(),
        );
        let trajectory_builder = test
            .map_builder()
            .get_trajectory_builder(trajectory_id)
            .unwrap();
        for mut measurement in measurements.iter().cloned() {
            measurement.time += from_seconds(100.);
            trajectory_builder.add_timed_point_cloud_data(&range_sensor_id().id, &measurement);
        }
        test.map_builder().finish_trajectory(trajectory_id);
        pose_graph.run_final_optimization();
        assert_eq!(test.local_slam_result_poses().len(), measurements.len());
        test.assert_travel_distance(0.15);
        let constraints = pose_graph.constraints();
        assert!(constraints.len() >= 50, "{}", constraints.len());
        let num_cross_trajectory_constraints = constraints
            .iter()
            .filter(|constraint| {
                constraint.node_id.trajectory_id != constraint.submap_id.trajectory_id
            })
            .count();
        assert!(num_cross_trajectory_constraints >= 3);
        // TODO(gaschler): Subscribe global slam callback, verify that all
        // nodes are optimized.
        assert!(num_inter_submap_constraints(pose_graph.as_ref()) > 0);
        assert!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 20
        );
        assert!(
            pose_graph
                .get_all_submap_data()
                .size_of_trajectory_or_zero(trajectory_id)
                >= 5
        );
        let local_to_global = pose_graph.get_local_to_global_transform(trajectory_id);
        let global_pose = local_to_global * *test.local_slam_result_poses().last().unwrap();
        assert!(
            (frozen_trajectory_to_global.translation().norm()
                - local_to_global.translation().norm())
            .abs()
                <= 0.1
        );
        let expected_global_pose =
            frozen_trajectory_to_global * Rigid3d::from_translation(travel_translation);
        let error = (global_pose.translation() - expected_global_pose.translation()).norm();
        assert!(
            error <= 0.3,
            "global_pose: {global_pose:?} expected_global_pose: {expected_global_pose:?}"
        );
    }

    #[test]
    fn localization_on_frozen_trajectory_2d_probability_grid() {
        localization_on_frozen_trajectory_2d(GridType::ProbabilityGrid);
    }

    #[test]
    fn localization_on_frozen_trajectory_2d_tsdf() {
        localization_on_frozen_trajectory_2d(GridType::Tsdf);
    }

    // ---- Additional tests (no C++ counterpart).

    #[test]
    #[should_panic(expected = "use_trajectory_builder_2d() ^")]
    fn requires_exactly_one_dimension() {
        let mut test = MapBuilderTestBase::set_up();
        test.map_builder_options.use_trajectory_builder_3d = true;
        let _ = MapBuilder::new(&test.map_builder_options);
    }

    #[test]
    fn pose_graph_type_follows_options() {
        let mut test = MapBuilderTestBase::set_up();
        let map_builder = MapBuilder::new(&test.map_builder_options);
        assert!(matches!(
            map_builder.pose_graph_handle(),
            MapBuilderPoseGraph::TwoD(_)
        ));
        test.set_options_to_3d();
        let map_builder = MapBuilder::new(&test.map_builder_options);
        assert!(matches!(
            map_builder.pose_graph_handle(),
            MapBuilderPoseGraph::ThreeD(_)
        ));
    }

    #[test]
    fn trajectory_options_and_deserialization_builders() {
        let mut test = MapBuilderTestBase::set_up();
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id(), imu_sensor_id()]);
        assert_eq!(trajectory_id, 0);
        let options = test.map_builder().get_all_trajectory_builder_options()[0].clone();
        assert_eq!(
            options.sensor_id,
            vec![
                trajectory_builder_interface::to_proto(&range_sensor_id()),
                trajectory_builder_interface::to_proto(&imu_sensor_id()),
            ]
        );
        assert_eq!(
            options.trajectory_builder_options.as_ref(),
            Some(&test.trajectory_builder_options)
        );
        test.map_builder().finish_trajectory(trajectory_id);
        let deserialized_id = test
            .map_builder()
            .add_trajectory_for_deserialization(&options);
        assert_eq!(deserialized_id, 1);
        assert_eq!(test.map_builder().num_trajectory_builders(), 2);
        assert!(test.map_builder().get_trajectory_builder(1).is_none());
        assert_eq!(
            test.map_builder()
                .get_all_trajectory_builder_options()
                .len(),
            2
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn get_trajectory_builder_out_of_range_panics() {
        let mut test = MapBuilderTestBase::set_up();
        test.build_map_builder();
        let _ = test.map_builder().get_trajectory_builder(0);
    }

    #[test]
    fn submap_to_proto_reports_errors_and_fills_response() {
        let mut test = MapBuilderTestBase::set_up();
        test.build_map_builder();
        let mut response = proto::submap_query::Response::default();
        assert_eq!(
            test.map_builder()
                .submap_to_proto(&SubmapId::new(0, 0), &mut response),
            "Requested submap from trajectory 0 but there are only 0 trajectories."
        );
        let trajectory_id = test.create_trajectory_with_fake_data(0.);
        test.pose_graph().run_final_optimization();
        assert_eq!(
            test.map_builder()
                .submap_to_proto(&SubmapId::new(trajectory_id, 1000), &mut response),
            "Requested submap 1000 from trajectory 0 but it does not exist: maybe it has been \
             trimmed."
        );
        assert_eq!(
            test.map_builder()
                .submap_to_proto(&SubmapId::new(trajectory_id, 0), &mut response),
            ""
        );
        assert!(!response.textures.is_empty());
    }

    #[test]
    fn pure_localization_trimmer_keeps_few_submaps() {
        let mut test = MapBuilderTestBase::set_up();
        test.set_options_enable_global_optimization();
        test.trajectory_builder_options.pure_localization_trimmer = Some(
            proto::trajectory_builder_options::PureLocalizationTrimmerOptions {
                max_submaps_to_keep: 3,
            },
        );
        test.build_map_builder();
        let trajectory_id = test.add_trajectory(&[range_sensor_id()]);
        test.feed_fake_data(trajectory_id, false);
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        let num_submaps = pose_graph
            .get_all_submap_data()
            .size_of_trajectory_or_zero(trajectory_id);
        assert!(num_submaps <= 3, "{num_submaps}");
        test.map_builder().finish_trajectory(trajectory_id);
        pose_graph.run_final_optimization();
    }

    #[test]
    fn initial_trajectory_pose_is_applied() {
        let mut test = MapBuilderTestBase::set_up();
        test.build_map_builder();
        let first = test.create_trajectory_with_fake_data(0.);
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        let last_node_pose = pose_graph
            .get_trajectory_node_poses()
            .trajectory(first)
            .next_back()
            .unwrap()
            .data
            .global_pose;
        let relative_pose = Rigid3d::from_translation(Vector3::new(10., -3., 0.));
        // After the end of the first trajectory: its last node pose is used.
        test.trajectory_builder_options.initial_trajectory_pose =
            Some(proto::InitialTrajectoryPose {
                relative_pose: Some(crate::transform::transform::to_proto(&relative_pose)),
                to_trajectory_id: first,
                timestamp: to_universal(
                    generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP)[0].time
                        + from_seconds(100.),
                ),
            });
        let second = test.add_trajectory(&[range_sensor_id()]);
        // Without nodes, the local to global transform is the initial pose.
        let local_to_global = pose_graph.get_local_to_global_transform(second);
        let expected = last_node_pose * relative_pose;
        assert!(
            (local_to_global.translation() - expected.translation()).norm() < 1e-9,
            "{local_to_global:?} vs. {expected:?}"
        );
        test.map_builder().finish_trajectory(second);
        pose_graph.run_final_optimization();
    }

    #[test]
    fn load_state_frozen_remaps_trajectory_ids_in_memory() {
        let mut test = MapBuilderTestBase::set_up();
        test.build_map_builder();
        let trajectory_id = test.create_trajectory_with_fake_data(0.);
        test.pose_graph().run_final_optimization();
        let num_nodes = test
            .pose_graph()
            .get_trajectory_nodes()
            .size_of_trajectory_or_zero(trajectory_id);
        // Serialize into memory.
        let chunks: Arc<Mutex<Vec<Box<dyn ProtoMessage>>>> = Arc::default();
        {
            let chunks = Arc::clone(&chunks);
            let mut writer = ForwardingProtoStreamWriter::new(Box::new(move |proto| {
                if let Some(proto) = proto {
                    chunks.lock().unwrap().push(proto.clone_message());
                }
                true
            }));
            test.map_builder().serialize_state(true, &mut writer);
            assert!(writer.close());
        }
        // A map builder which already has a trajectory: the loaded
        // trajectory gets id 1.
        test.build_map_builder();
        let first_trajectory_id = test.add_trajectory(&[range_sensor_id()]);
        test.map_builder().finish_trajectory(first_trajectory_id);
        let mut reader =
            InMemoryProtoStreamReader::new(std::mem::take(&mut *chunks.lock().unwrap()).into());
        let remapping = test.map_builder().load_state(&mut reader, true);
        assert_eq!(remapping, BTreeMap::from([(trajectory_id, 1)]));
        let pose_graph = test.pose_graph();
        pose_graph.run_final_optimization();
        assert!(pose_graph.is_trajectory_frozen(1));
        assert_eq!(
            pose_graph
                .get_trajectory_nodes()
                .size_of_trajectory_or_zero(1),
            num_nodes
        );
        // Frozen state loads no constraints.
        assert!(
            pose_graph
                .constraints()
                .iter()
                .all(|constraint| constraint.node_id.trajectory_id != 1)
        );
        assert!(test.map_builder().get_trajectory_builder(1).is_none());
    }
}
