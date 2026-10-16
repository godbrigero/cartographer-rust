//! End-to-end tests of `MapBuilder` serialization on simulated data (no C++
//! counterpart):
//!
//! * 2D and 3D `MapBuilder` runs on `cartographer-sim` datasets, fed through
//!   `get_trajectory_builder(id).add_*` (2D: LiDAR, IMU, odometry, fixed
//!   frame poses and landmarks; 3D: LiDAR, IMU and odometry), are saved with
//!   `serialize_state_to_file` and loaded with `load_state_from_file` into
//!   fresh map builders: frozen loads reproduce node and submap poses, unfrozen
//!   loads reproduce the constraints (and, after re-optimizing, the poses);
//!   `pbstream info` (library and binary) summarizes the file.
//! * A pbstream written by Rust is re-read with `ProtoStreamDeserializer` and
//!   compared message by message (all proto fields) with the protos the map
//!   builder serialized into memory.
//! * `pbstream migrate` (library and binary) and the on-the-fly migration in
//!   `load_state` handle a 3D stream of format version 1.
//! * The `.proto` schema files are byte-identical to the C++ ones.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::io::internal::in_memory_proto_stream::ForwardingProtoStreamWriter;
use cartographer::io::internal::pbstream_info;
use cartographer::io::proto_stream::{ProtoStreamReader, ProtoStreamWriter};
use cartographer::io::proto_stream_deserializer::ProtoStreamDeserializer;
use cartographer::io::proto_stream_interface::{
    ProtoMessage, ProtoStreamReaderInterface, ProtoStreamWriterInterface,
};
use cartographer::io::serialization_format_migration::migrate_stream_version1_to_version2;
use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::map_builder::MapBuilder;
use cartographer::mapping::map_builder_interface::{
    MapBuilderInterface, create_map_builder_options,
};
use cartographer::mapping::pose_graph_interface::{PoseGraphInterface, Tag};
use cartographer::mapping::proto::{self, SerializedData, serialized_data::Data};
use cartographer::mapping::trajectory_builder_interface::{
    SensorId, SensorType, create_trajectory_builder_options,
};
use cartographer::sensor::fixed_frame_pose_data::FixedFramePoseData;
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::landmark_data::{LandmarkData, LandmarkObservation};
use cartographer::sensor::odometry_data::OdometryData;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::{
    DatasetBuilder, FixedFramePoseConfig, ImuConfig, LandmarkConfig, Lidar2dConfig, Lidar3dConfig,
    OdometryConfig, SensorData, Trajectory, World2d, World3d,
};
use nalgebra::{Isometry3, Point3, Quaternion};

const PBSTREAM: &str = env!("CARGO_BIN_EXE_cartographer_pbstream");

/// Offset of the simulated time axis (simulated times start at 0 s).
fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

fn to_rigid3d(pose: &Isometry3<f64>) -> Rigid3d {
    let q = pose.rotation.quaternion();
    Rigid3d::new(pose.translation.vector, Quaternion::new(q.w, q.i, q.j, q.k))
}

fn dictionary(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::new(
        code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    )
}

fn temp_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "cartographer_map_builder_sim_{}_{name}",
            std::process::id()
        ))
        .to_str()
        .unwrap()
        .to_string()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Dims {
    Two,
    Three,
}

fn map_builder_options(dims: Dims) -> proto::MapBuilderOptions {
    let dimension = match dims {
        Dims::Two => "2d",
        Dims::Three => "3d",
    };
    create_map_builder_options(&dictionary(&format!(
        "include \"map_builder.lua\"\n\
         MAP_BUILDER.use_trajectory_builder_{dimension} = true\n\
         MAP_BUILDER.pose_graph.optimize_every_n_nodes = 20\n\
         return MAP_BUILDER"
    )))
}

fn trajectory_builder_options() -> proto::TrajectoryBuilderOptions {
    create_trajectory_builder_options(&dictionary(
        "include \"trajectory_builder.lua\"\n\
         TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = true\n\
         TRAJECTORY_BUILDER.trajectory_builder_2d.submaps.num_range_data = 20\n\
         TRAJECTORY_BUILDER.trajectory_builder_3d.submaps.num_range_data = 20\n\
         return TRAJECTORY_BUILDER",
    ))
}

fn sensor_id(r#type: SensorType, id: &str) -> SensorId {
    SensorId {
        r#type,
        id: id.to_string(),
    }
}

/// Builds a map of a simulated run with a fresh `MapBuilder`, finishes the
/// trajectory and runs the final optimization.
fn build_map(dims: Dims) -> MapBuilder {
    let (dataset, range_id, mut sensor_ids) = match dims {
        Dims::Two => (
            DatasetBuilder::new(
                Trajectory::corridor_loops(1.0, 1.5).with_start_from_rest(1.0),
                World2d::loop_corridor(),
            )
            .seed(7)
            .time_range(0.0, 20.0)
            .lidar_2d("scan", Lidar2dConfig::default())
            .imu("imu", ImuConfig::default())
            .odometry("odom", OdometryConfig::default())
            .fixed_frame_pose("gps", FixedFramePoseConfig::default())
            .landmarks("landmarks", LandmarkConfig::default())
            .build(),
            "scan",
            vec![
                sensor_id(SensorType::FixedFramePose, "gps"),
                sensor_id(SensorType::Landmark, "landmarks"),
            ],
        ),
        Dims::Three => (
            DatasetBuilder::new(
                Trajectory::corridor_loops_3d(1.0, 1.2).with_start_from_rest(1.0),
                World3d::loop_corridor(),
            )
            .seed(7)
            .time_range(0.0, 8.0)
            .lidar_3d("points", Lidar3dConfig::default())
            .imu("imu", ImuConfig::default())
            .odometry("odom", OdometryConfig::default())
            .build(),
            "points",
            Vec::new(),
        ),
    };
    sensor_ids.extend([
        sensor_id(SensorType::Range, range_id),
        sensor_id(SensorType::Imu, "imu"),
        sensor_id(SensorType::Odometry, "odom"),
    ]);
    let sensor_ids: BTreeSet<SensorId> = sensor_ids.into_iter().collect();

    let mut map_builder = MapBuilder::new(&map_builder_options(dims));
    let num_local_slam_results = Arc::new(Mutex::new(0_usize));
    let callback = {
        let num_local_slam_results = Arc::clone(&num_local_slam_results);
        Arc::new(move |_, _, _, _, _| *num_local_slam_results.lock().unwrap() += 1)
    };
    let trajectory_id = map_builder.add_trajectory_builder(
        &sensor_ids,
        &trajectory_builder_options(),
        Some(callback),
    );
    assert_eq!(trajectory_id, 0);
    let mut num_range_messages = 0;
    {
        let builder = map_builder.get_trajectory_builder(trajectory_id).unwrap();
        for message in &dataset.messages {
            let extrinsic = dataset
                .sensor(&message.sensor_id)
                .expect("known sensor")
                .extrinsic;
            let time = to_time(message.time);
            match &message.data {
                SensorData::PointCloud(cloud) => {
                    num_range_messages += 1;
                    let extrinsic_f32 = extrinsic.cast::<f32>();
                    let transform = |p| (extrinsic_f32 * Point3::from(p)).coords;
                    builder.add_timed_point_cloud_data(
                        &message.sensor_id,
                        &TimedPointCloudData {
                            time,
                            origin: transform(cloud.origin),
                            ranges: cloud
                                .points
                                .iter()
                                .map(|p| TimedRangefinderPoint {
                                    position: transform(p.position),
                                    time: p.time,
                                })
                                .collect(),
                            intensities: cloud.intensities.clone(),
                        },
                    );
                }
                SensorData::Imu(imu) => builder.add_imu_data(
                    &message.sensor_id,
                    &ImuData {
                        time,
                        linear_acceleration: imu.linear_acceleration,
                        angular_velocity: imu.angular_velocity,
                    },
                ),
                SensorData::Odometry(odometry) => builder.add_odometry_data(
                    &message.sensor_id,
                    &OdometryData {
                        time,
                        pose: to_rigid3d(&(odometry.pose * extrinsic.inverse())),
                    },
                ),
                SensorData::FixedFramePose(fixed_frame_pose) => builder.add_fixed_frame_pose_data(
                    &message.sensor_id,
                    &FixedFramePoseData {
                        time,
                        pose: fixed_frame_pose
                            .pose
                            .map(|pose| to_rigid3d(&(pose * extrinsic.inverse()))),
                    },
                ),
                SensorData::Landmark(landmarks) => builder.add_landmark_data(
                    &message.sensor_id,
                    &LandmarkData {
                        time,
                        landmark_observations: landmarks
                            .observations
                            .iter()
                            .map(|observation| LandmarkObservation {
                                id: observation.id.clone(),
                                landmark_to_tracking_transform: to_rigid3d(
                                    &(extrinsic * observation.landmark_to_sensor),
                                ),
                                translation_weight: observation.translation_weight,
                                rotation_weight: observation.rotation_weight,
                            })
                            .collect(),
                    },
                ),
            }
        }
    }
    map_builder.finish_trajectory(trajectory_id);
    let pose_graph = map_builder.pose_graph();
    pose_graph.run_final_optimization();
    assert!(pose_graph.is_trajectory_finished(trajectory_id));
    assert!(*num_local_slam_results.lock().unwrap() > 0);
    assert!(*num_local_slam_results.lock().unwrap() <= num_range_messages);
    map_builder
}

/// The poses and counts compared between saved and loaded pose graphs.
#[derive(Debug)]
struct Snapshot {
    node_poses: BTreeMap<NodeId, Rigid3d>,
    submap_poses: BTreeMap<SubmapId, Rigid3d>,
    num_constraints: usize,
    num_intra_submap_constraints: usize,
}

fn snapshot(pose_graph: &dyn PoseGraphInterface) -> Snapshot {
    let constraints = pose_graph.constraints();
    Snapshot {
        node_poses: pose_graph
            .get_trajectory_node_poses()
            .iter()
            .map(|node| (node.id, node.data.global_pose))
            .collect(),
        submap_poses: pose_graph
            .get_all_submap_poses()
            .iter()
            .map(|submap| (submap.id, submap.data.pose))
            .collect(),
        num_constraints: constraints.len(),
        num_intra_submap_constraints: constraints
            .iter()
            .filter(|constraint| constraint.tag == Tag::IntraSubmap)
            .count(),
    }
}

/// Largest translation (m) and rotation (rad) difference between poses with
/// the same ids; panics if the id sets differ.
fn max_pose_difference<I: Ord + Copy + std::fmt::Debug>(
    a: &BTreeMap<I, Rigid3d>,
    b: &BTreeMap<I, Rigid3d>,
) -> (f64, f64) {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "different ids"
    );
    let mut max_translation: f64 = 0.;
    let mut max_rotation: f64 = 0.;
    for (id, pose_a) in a {
        let pose_b = &b[id];
        max_translation = max_translation.max((pose_a.translation() - pose_b.translation()).norm());
        let delta = pose_a.inverse() * *pose_b;
        let w = delta.rotation().w.abs().min(1.);
        max_rotation = max_rotation.max(2. * w.acos());
    }
    (max_translation, max_rotation)
}

fn load_into_fresh_map_builder(dims: Dims, filename: &str, frozen: bool) -> MapBuilder {
    let mut map_builder = MapBuilder::new(&map_builder_options(dims));
    let remapping = map_builder.load_state_from_file(filename, frozen);
    assert_eq!(remapping, BTreeMap::from([(0, 0)]));
    assert_eq!(map_builder.num_trajectory_builders(), 1);
    assert!(map_builder.get_trajectory_builder(0).is_none());
    map_builder.pose_graph().run_final_optimization();
    map_builder
}

fn save_and_load(dims: Dims) {
    let mut map_builder = build_map(dims);
    let pose_graph = map_builder.pose_graph();
    let original = snapshot(pose_graph.as_ref());
    println!(
        "{dims:?}: {} nodes, {} submaps, {} constraints",
        original.node_poses.len(),
        original.submap_poses.len(),
        original.num_constraints
    );
    assert!(original.node_poses.len() >= 10);
    assert!(original.submap_poses.len() >= 2);
    assert!(original.num_constraints > original.num_intra_submap_constraints);

    let filename = temp_path(&format!("{dims:?}.pbstream"));
    assert!(map_builder.serialize_state_to_file(/*include_unfinished_submaps=*/ true, &filename));

    // `pbstream info` (library).
    let counts = pbstream_info::run(&filename, false);
    assert_eq!(counts["node"] as usize, original.node_poses.len());
    assert_eq!(counts["submap"] as usize, original.submap_poses.len());
    assert_eq!(counts["trajectory_data"], 1);
    assert!(counts["odometry_data"] > 0);
    match dims {
        Dims::Two => {
            assert_eq!(counts["submap_2d"] as usize, original.submap_poses.len());
            assert_eq!(
                counts["submap_2d_grid"] as usize,
                original.submap_poses.len()
            );
            assert_eq!(counts["submap_3d"], 0);
            // The 2D optimization problem drops IMU data (like C++).
            assert!(!counts.contains_key("imu_data"));
            assert!(counts["fixed_frame_pose_data"] > 0);
            assert!(counts["landmark_data"] > 0);
        }
        Dims::Three => {
            assert_eq!(counts["submap_3d"] as usize, original.submap_poses.len());
            assert_eq!(
                counts["submap_3d_high_resolution_hybrid_grid"] as usize,
                original.submap_poses.len()
            );
            assert_eq!(counts["submap_2d"], 0);
            assert!(counts["imu_data"] > 0);
        }
    }

    // `pbstream info` (binary).
    let output = Command::new(PBSTREAM)
        .args(["info", &filename])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!(
            "SerializedData package contains #node: {}",
            original.node_poses.len()
        )),
        "{stderr}"
    );
    assert!(stderr.contains(&format!(
        "Trajectory id: 0 has #nodes {} has #submaps {}",
        original.node_poses.len(),
        original.submap_poses.len()
    )));

    // Frozen: the poses are kept exactly (up to float round-off of the 2D
    // gravity alignment), and no constraints are loaded.
    {
        let loaded = load_into_fresh_map_builder(dims, &filename, true);
        let pose_graph = loaded.pose_graph();
        assert!(pose_graph.is_trajectory_frozen(0));
        let frozen = snapshot(pose_graph.as_ref());
        let (node_translation, node_rotation) =
            max_pose_difference(&original.node_poses, &frozen.node_poses);
        let (submap_translation, submap_rotation) =
            max_pose_difference(&original.submap_poses, &frozen.submap_poses);
        println!(
            "{dims:?} frozen: nodes {node_translation:e} m / {node_rotation:e} rad, submaps \
             {submap_translation:e} m / {submap_rotation:e} rad"
        );
        assert!(node_translation < 1e-9 && node_rotation < 1e-6);
        assert!(submap_translation < 1e-9 && submap_rotation < 1e-6);
        assert_eq!(frozen.num_constraints, 0);
    }

    // Not frozen: the same constraints; re-optimizing from the loaded
    // optimum keeps the poses (observed: bit-identical; the bound allows for
    // solver round-off).
    {
        let loaded = load_into_fresh_map_builder(dims, &filename, false);
        let pose_graph = loaded.pose_graph();
        assert!(!pose_graph.is_trajectory_frozen(0));
        let unfrozen = snapshot(pose_graph.as_ref());
        assert_eq!(unfrozen.num_constraints, original.num_constraints);
        assert_eq!(
            unfrozen.num_intra_submap_constraints,
            original.num_intra_submap_constraints
        );
        let (node_translation, node_rotation) =
            max_pose_difference(&original.node_poses, &unfrozen.node_poses);
        let (submap_translation, submap_rotation) =
            max_pose_difference(&original.submap_poses, &unfrozen.submap_poses);
        println!(
            "{dims:?} unfrozen: nodes {node_translation:e} m / {node_rotation:e} rad, submaps \
             {submap_translation:e} m / {submap_rotation:e} rad"
        );
        assert!(node_translation < 1e-6 && node_rotation < 1e-6);
        assert!(submap_translation < 1e-6 && submap_rotation < 1e-6);
        // The loaded state serializes to the same pose graph structure.
        let reloaded = pose_graph.to_proto(true);
        let saved = pose_graph_of_file(&filename);
        assert_eq!(reloaded.constraint.len(), saved.constraint.len());
        assert_eq!(reloaded.trajectory.len(), saved.trajectory.len());
        assert_eq!(
            reloaded.trajectory[0].node.len(),
            saved.trajectory[0].node.len()
        );
        assert_eq!(
            reloaded.trajectory[0].submap.len(),
            saved.trajectory[0].submap.len()
        );
    }
    std::fs::remove_file(&filename).unwrap();
}

fn pose_graph_of_file(filename: &str) -> proto::PoseGraph {
    let mut reader = ProtoStreamReader::new(filename);
    let deserializer = ProtoStreamDeserializer::new(&mut reader);
    deserializer.pose_graph().clone()
}

#[test]
fn save_load_2d() {
    save_and_load(Dims::Two);
}

#[test]
fn save_load_3d() {
    save_and_load(Dims::Three);
}

/// Serializes `map_builder` into memory.
fn serialize_in_memory(map_builder: &mut MapBuilder) -> Vec<Box<dyn ProtoMessage>> {
    let chunks: Arc<Mutex<Vec<Box<dyn ProtoMessage>>>> = Arc::default();
    {
        let chunks = Arc::clone(&chunks);
        let mut writer = ForwardingProtoStreamWriter::new(Box::new(move |proto| {
            if let Some(proto) = proto {
                chunks.lock().unwrap().push(proto.clone_message());
            }
            true
        }));
        map_builder.serialize_state(true, &mut writer);
        assert!(writer.close());
    }
    std::mem::take(&mut *chunks.lock().unwrap())
}

fn data_name(data: &SerializedData) -> &'static str {
    match &data.data {
        Some(Data::PoseGraph(_)) => "pose_graph",
        Some(Data::AllTrajectoryBuilderOptions(_)) => "all_trajectory_builder_options",
        Some(Data::Submap(_)) => "submap",
        Some(Data::Node(_)) => "node",
        Some(Data::TrajectoryData(_)) => "trajectory_data",
        Some(Data::ImuData(_)) => "imu_data",
        Some(Data::OdometryData(_)) => "odometry_data",
        Some(Data::FixedFramePoseData(_)) => "fixed_frame_pose_data",
        Some(Data::LandmarkData(_)) => "landmark_data",
        None => "none",
    }
}

fn written_pbstream_keeps_every_proto_field(dims: Dims) {
    let expected_kinds: &[&str] = match dims {
        // The 2D optimization problem drops IMU data (like C++).
        Dims::Two => &[
            "submap",
            "node",
            "trajectory_data",
            "odometry_data",
            "fixed_frame_pose_data",
            "landmark_data",
        ],
        Dims::Three => &[
            "submap",
            "node",
            "trajectory_data",
            "imu_data",
            "odometry_data",
        ],
    };
    let mut map_builder = build_map(dims);
    let in_memory = serialize_in_memory(&mut map_builder);
    let filename = temp_path(&format!("fields_{dims:?}.pbstream"));
    assert!(map_builder.serialize_state_to_file(true, &filename));

    let header = in_memory[0]
        .as_any()
        .downcast_ref::<proto::SerializationHeader>()
        .expect("first message is the header");
    assert_eq!(header.format_version, 2);
    let in_memory_data: Vec<&SerializedData> = in_memory[1..]
        .iter()
        .map(|message| {
            message
                .as_any()
                .downcast_ref::<SerializedData>()
                .expect("SerializedData")
        })
        .collect();

    let mut reader = ProtoStreamReader::new(&filename);
    {
        let mut deserializer = ProtoStreamDeserializer::new(&mut reader);
        assert_eq!(deserializer.header(), header);
        assert_eq!(
            Some(&Data::PoseGraph(deserializer.pose_graph().clone())),
            in_memory_data[0].data.as_ref()
        );
        assert_eq!(
            Some(&Data::AllTrajectoryBuilderOptions(
                deserializer.all_trajectory_builder_options().clone()
            )),
            in_memory_data[1].data.as_ref()
        );
        // The pose graph and options are the map builder's.
        assert_eq!(
            *deserializer.pose_graph(),
            map_builder.pose_graph().to_proto(true)
        );
        assert_eq!(
            deserializer
                .all_trajectory_builder_options()
                .options_with_sensor_ids,
            map_builder.get_all_trajectory_builder_options()
        );

        let mut read = Vec::new();
        let mut proto = SerializedData::default();
        while deserializer.read_next_serialized_data(&mut proto) {
            read.push(std::mem::take(&mut proto));
        }
        assert_eq!(read.len(), in_memory_data.len() - 2);
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for (read, expected) in read.iter().zip(&in_memory_data[2..]) {
            assert_eq!(read, *expected);
            *counts.entry(data_name(read)).or_default() += 1;
        }
        println!("{counts:?}");
        assert_eq!(
            counts.keys().copied().collect::<BTreeSet<_>>(),
            expected_kinds.iter().copied().collect()
        );
        // The data appears in the C++ `WritePbStream` order.
        let order: Vec<&str> = read.iter().map(data_name).collect();
        let mut deduplicated = order.clone();
        deduplicated.dedup();
        assert_eq!(deduplicated, expected_kinds);
        // Spot checks that nested fields made it through.
        let node = read
            .iter()
            .find_map(|data| match &data.data {
                Some(Data::Node(node)) => Some(node),
                _ => None,
            })
            .unwrap();
        let node_data = node.node_data.as_ref().unwrap();
        assert!(node_data.timestamp > 0);
        assert!(node_data.local_pose.is_some());
        assert!(node_data.gravity_alignment.is_some());
        assert!(node_data.filtered_gravity_aligned_point_cloud.is_some());
        let submap = read
            .iter()
            .find_map(|data| match &data.data {
                Some(Data::Submap(submap)) => Some(submap),
                _ => None,
            })
            .unwrap();
        match dims {
            Dims::Two => {
                let submap_2d = submap.submap_2d.as_ref().unwrap();
                assert!(submap_2d.num_range_data > 0);
                assert!(submap_2d.grid.as_ref().unwrap().cells.len() > 1);
            }
            Dims::Three => {
                let submap_3d = submap.submap_3d.as_ref().unwrap();
                assert!(submap_3d.num_range_data > 0);
                assert!(!submap_3d.rotational_scan_matcher_histogram.is_empty());
                assert!(submap_3d.high_resolution_hybrid_grid.is_some());
                assert!(submap_3d.low_resolution_hybrid_grid.is_some());
            }
        }
    }
    assert!(reader.eof());
    drop(reader);
    std::fs::remove_file(&filename).unwrap();
}

#[test]
fn written_pbstream_keeps_every_proto_field_2d() {
    written_pbstream_keeps_every_proto_field(Dims::Two);
}

#[test]
fn written_pbstream_keeps_every_proto_field_3d() {
    written_pbstream_keeps_every_proto_field(Dims::Three);
}

/// Rewrites the 3D stream `input` as format version 1: header version 1 and
/// submaps without rotational scan matcher histograms.
fn downgrade_to_version_1(input: &str, output: &str) -> usize {
    let mut reader = ProtoStreamReader::new(input);
    let mut deserializer = ProtoStreamDeserializer::new(&mut reader);
    let mut writer = ProtoStreamWriter::new(output);
    writer.write_proto(&proto::SerializationHeader { format_version: 1 });
    writer.write_proto(&SerializedData {
        data: Some(Data::PoseGraph(deserializer.pose_graph().clone())),
    });
    writer.write_proto(&SerializedData {
        data: Some(Data::AllTrajectoryBuilderOptions(
            deserializer.all_trajectory_builder_options().clone(),
        )),
    });
    let mut num_submaps = 0;
    let mut proto = SerializedData::default();
    while deserializer.read_next_serialized_data(&mut proto) {
        if let Some(Data::Submap(submap)) = &mut proto.data {
            let submap_3d = submap.submap_3d.as_mut().unwrap();
            assert!(!submap_3d.rotational_scan_matcher_histogram.is_empty());
            submap_3d.rotational_scan_matcher_histogram.clear();
            num_submaps += 1;
        }
        writer.write_proto(&proto);
    }
    assert!(writer.close());
    num_submaps
}

fn submap_histograms(filename: &str) -> (u32, Vec<Vec<f32>>) {
    let mut reader = ProtoStreamReader::new(filename);
    let mut deserializer = ProtoStreamDeserializer::new(&mut reader);
    let format_version = deserializer.header().format_version;
    let mut histograms = Vec::new();
    let mut proto = SerializedData::default();
    while deserializer.read_next_serialized_data(&mut proto) {
        if let Some(Data::Submap(submap)) = &proto.data {
            histograms.push(
                submap
                    .submap_3d
                    .as_ref()
                    .unwrap()
                    .rotational_scan_matcher_histogram
                    .clone(),
            );
        }
    }
    (format_version, histograms)
}

#[test]
fn migrate_3d_version_1() {
    let mut map_builder = build_map(Dims::Three);
    let version_2 = temp_path("migrate_v2.pbstream");
    assert!(map_builder.serialize_state_to_file(true, &version_2));
    let num_nodes = map_builder.pose_graph().get_trajectory_nodes().size();
    drop(map_builder);
    let version_1 = temp_path("migrate_v1.pbstream");
    let num_submaps = downgrade_to_version_1(&version_2, &version_1);
    assert!(num_submaps > 0);

    // Library.
    let migrated = temp_path("migrated_lib.pbstream");
    {
        let mut input = ProtoStreamReader::new(&version_1);
        let mut output = ProtoStreamWriter::new(&migrated);
        migrate_stream_version1_to_version2(&mut input, &mut output, true);
        assert!(output.close());
    }
    let (format_version, histograms) = submap_histograms(&migrated);
    assert_eq!(format_version, 2);
    assert_eq!(histograms.len(), num_submaps);
    assert!(histograms.iter().all(|histogram| !histogram.is_empty()));
    assert!(
        histograms
            .iter()
            .all(|histogram| histogram.iter().any(|&value| value > 0.))
    );

    // Binary.
    let migrated_bin = temp_path("migrated_bin.pbstream");
    let output = Command::new(PBSTREAM)
        .args(["migrate", &version_1, &migrated_bin])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(submap_histograms(&migrated_bin), (2, histograms.clone()));

    // `load_state` migrates version 1 3D streams on the fly.
    let mut loaded = MapBuilder::new(&map_builder_options(Dims::Three));
    loaded.load_state_from_file(&version_1, true);
    let pose_graph = loaded.pose_graph();
    pose_graph.run_final_optimization();
    assert_eq!(pose_graph.get_trajectory_nodes().size(), num_nodes);
    let submaps = pose_graph.get_all_submap_data();
    assert_eq!(submaps.size(), num_submaps);
    let migrated_in_memory = loaded.pose_graph().to_proto(true);
    assert_eq!(migrated_in_memory.trajectory[0].submap.len(), num_submaps);

    for file in [&version_2, &version_1, &migrated, &migrated_bin] {
        std::fs::remove_file(file).unwrap();
    }
}

#[test]
fn pbstream_binary_usage() {
    let output = Command::new(PBSTREAM).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Swiss Army knife for pbstreams."));
    let output = Command::new(PBSTREAM).arg("unknown").output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Unknown subtool: \"unknown"));
    let output = Command::new(PBSTREAM).arg("info").output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("<pbstream_filename>"));
    let output = Command::new(PBSTREAM)
        .args(["migrate", "only_input.pbstream"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("<input_filename> <output_filename>"));
}

/// Finds the C++ source tree (`<ancestor>/cartographer/cartographer`) next to
/// one of the ancestors of this crate (e.g. `../cartographer/cartographer`
/// relative to the workspace root).
fn cpp_source_directory() -> Option<PathBuf> {
    if let Ok(directory) = std::env::var("CARTOGRAPHER_CPP_SOURCE_DIR") {
        return Some(PathBuf::from(directory));
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|ancestor| ancestor.join("cartographer").join("cartographer"))
        .find(|candidate| candidate.join("mapping").join("proto").is_dir())
}

fn proto_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            proto_files(&path, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "proto")
        {
            files.push(path);
        }
    }
}

#[test]
fn proto_schemas_are_byte_identical_to_cpp() {
    let Some(cpp_directory) = cpp_source_directory() else {
        println!(
            "NOTE: C++ source tree (../cartographer/cartographer) not found; skipping the proto \
             byte-identity check."
        );
        return;
    };
    // `crates/cartographer/proto/cartographer/<path>` <-> `<cpp>/<path>`.
    let rust_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("proto")
        .join("cartographer");
    let mut files = Vec::new();
    proto_files(&rust_root, &mut files);
    assert!(files.len() >= 40, "{} proto files", files.len());
    for file in &files {
        let relative = file.strip_prefix(&rust_root).unwrap();
        let cpp_file = cpp_directory.join(relative);
        let rust_bytes = std::fs::read(file).unwrap();
        let cpp_bytes = std::fs::read(&cpp_file)
            .unwrap_or_else(|e| panic!("{} has no C++ counterpart: {e}", file.display()));
        assert!(
            rust_bytes == cpp_bytes,
            "{} differs from {}",
            file.display(),
            cpp_file.display()
        );
    }
    println!(
        "{} proto files are byte-identical to {}",
        files.len(),
        cpp_directory.display()
    );
}
