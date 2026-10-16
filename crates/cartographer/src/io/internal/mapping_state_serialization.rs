//! Port of `cartographer/io/internal/mapping_state_serialization` (.h, .cc).

use std::collections::BTreeMap;

use nalgebra::Quaternion;

use crate::common::time::to_universal;
use crate::io::proto_stream_interface::ProtoStreamWriterInterface;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::pose_graph::PoseGraph;
use crate::mapping::pose_graph_interface::{
    LandmarkNode, SubmapData, TrajectoryData, TrajectoryState,
};
use crate::mapping::proto::{self, SerializedData, serialized_data::Data};
use crate::mapping::trajectory_node::{self, TrajectoryNode};
use crate::sensor;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::transform::to_proto;

/// The current serialization format version.
pub const MAPPING_STATE_SERIALIZATION_FORMAT_VERSION: u32 = 2;
/// The format version of streams whose 3D submaps lack rotational scan
/// matcher histograms.
pub const FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS: u32 = 1;

fn serialized(data: Data) -> SerializedData {
    SerializedData { data: Some(data) }
}

fn create_all_trajectory_builder_options_proto(
    all_options_with_sensor_ids: &[proto::TrajectoryBuilderOptionsWithSensorIds],
    trajectory_ids_to_serialize: &[i32],
) -> proto::AllTrajectoryBuilderOptions {
    proto::AllTrajectoryBuilderOptions {
        options_with_sensor_ids: trajectory_ids_to_serialize
            .iter()
            .map(|&id| {
                // C++ `operator[]` on the vector (indexed by trajectory id).
                all_options_with_sensor_ids[usize::try_from(id).expect("negative trajectory id")]
                    .clone()
            })
            .collect(),
    }
}

/// Will return all trajectory ids, that have `state != DELETED`.
fn get_valid_trajectory_ids(trajectory_states: &BTreeMap<i32, TrajectoryState>) -> Vec<i32> {
    trajectory_states
        .iter()
        .filter(|(_, state)| **state != TrajectoryState::Deleted)
        .map(|(id, _)| *id)
        .collect()
}

fn create_header() -> proto::SerializationHeader {
    proto::SerializationHeader {
        format_version: MAPPING_STATE_SERIALIZATION_FORMAT_VERSION,
    }
}

fn serialize_pose_graph(
    pose_graph: &dyn PoseGraph,
    include_unfinished_submaps: bool,
) -> SerializedData {
    serialized(Data::PoseGraph(
        pose_graph.to_proto(include_unfinished_submaps),
    ))
}

fn serialize_trajectory_builder_options(
    trajectory_builder_options: &[proto::TrajectoryBuilderOptionsWithSensorIds],
    trajectory_ids_to_serialize: &[i32],
) -> SerializedData {
    serialized(Data::AllTrajectoryBuilderOptions(
        create_all_trajectory_builder_options_proto(
            trajectory_builder_options,
            trajectory_ids_to_serialize,
        ),
    ))
}

fn serialize_submaps(
    submap_data: &MapById<SubmapId, SubmapData>,
    include_unfinished_submaps: bool,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    // Next serialize all submaps.
    for submap_id_data in submap_data {
        let submap = submap_id_data
            .data
            .submap
            .as_ref()
            .expect("Check failed: submap != nullptr")
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !include_unfinished_submaps && !submap.insertion_finished() {
            continue;
        }
        let mut submap_proto = submap.to_proto(/*include_probability_grid_data=*/ true);
        submap_proto.submap_id = Some(proto::SubmapId {
            trajectory_id: submap_id_data.id.trajectory_id,
            submap_index: submap_id_data.id.submap_index,
        });
        writer.write_proto(&serialized(Data::Submap(submap_proto)));
    }
}

fn serialize_trajectory_nodes(
    trajectory_nodes: &MapById<NodeId, TrajectoryNode>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for node_id_data in trajectory_nodes {
        let constant_data = node_id_data
            .data
            .constant_data
            .as_ref()
            .expect("Check failed: constant_data != nullptr");
        let node_proto = proto::Node {
            node_id: Some(proto::NodeId {
                trajectory_id: node_id_data.id.trajectory_id,
                node_index: node_id_data.id.node_index,
            }),
            node_data: Some(trajectory_node::to_proto(constant_data)),
        };
        writer.write_proto(&serialized(Data::Node(node_proto)));
    }
}

fn serialize_trajectory_data(
    all_trajectory_data: &BTreeMap<i32, TrajectoryData>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for (&trajectory_id, trajectory_data) in all_trajectory_data {
        let c = &trajectory_data.imu_calibration;
        // `imu_calibration` is stored as (w, x, y, z), like the C++
        // `Eigen::Quaterniond(c[0], c[1], c[2], c[3])`.
        let trajectory_data_proto = proto::TrajectoryData {
            trajectory_id,
            gravity_constant: trajectory_data.gravity_constant,
            imu_calibration: Some(to_proto(&Quaternion::new(c[0], c[1], c[2], c[3]))),
            fixed_frame_origin_in_map: trajectory_data
                .fixed_frame_origin_in_map
                .as_ref()
                .map(to_proto),
        };
        writer.write_proto(&serialized(Data::TrajectoryData(trajectory_data_proto)));
    }
}

fn serialize_imu_data(
    all_imu_data: &MapByTime<ImuData>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for trajectory_id in all_imu_data.trajectory_ids() {
        for imu_data in all_imu_data.trajectory(trajectory_id) {
            writer.write_proto(&serialized(Data::ImuData(proto::ImuData {
                trajectory_id,
                imu_data: Some(sensor::imu_data::to_proto(imu_data)),
            })));
        }
    }
}

fn serialize_odometry_data(
    all_odometry_data: &MapByTime<OdometryData>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for trajectory_id in all_odometry_data.trajectory_ids() {
        for odometry_data in all_odometry_data.trajectory(trajectory_id) {
            writer.write_proto(&serialized(Data::OdometryData(proto::OdometryData {
                trajectory_id,
                odometry_data: Some(sensor::odometry_data::to_proto(odometry_data)),
            })));
        }
    }
}

fn serialize_fixed_frame_pose_data(
    all_fixed_frame_pose_data: &MapByTime<FixedFramePoseData>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for trajectory_id in all_fixed_frame_pose_data.trajectory_ids() {
        for fixed_frame_pose_data in all_fixed_frame_pose_data.trajectory(trajectory_id) {
            writer.write_proto(&serialized(Data::FixedFramePoseData(
                proto::FixedFramePoseData {
                    trajectory_id,
                    fixed_frame_pose_data: Some(sensor::fixed_frame_pose_data::to_proto(
                        fixed_frame_pose_data,
                    )),
                },
            )));
        }
    }
}

fn serialize_landmark_nodes(
    all_landmark_nodes: &BTreeMap<String, LandmarkNode>,
    writer: &mut dyn ProtoStreamWriterInterface,
) {
    for (landmark_id, node) in all_landmark_nodes {
        for observation in &node.landmark_observations {
            let landmark_data = sensor::proto::LandmarkData {
                timestamp: to_universal(observation.time),
                landmark_observations: vec![sensor::proto::landmark_data::LandmarkObservation {
                    id: landmark_id.as_bytes().to_vec(),
                    landmark_to_tracking_transform: Some(to_proto(
                        &observation.landmark_to_tracking_transform,
                    )),
                    translation_weight: observation.translation_weight,
                    rotation_weight: observation.rotation_weight,
                }],
            };
            writer.write_proto(&serialized(Data::LandmarkData(proto::LandmarkData {
                trajectory_id: observation.trajectory_id,
                landmark_data: Some(landmark_data),
            })));
        }
    }
}

/// Serializes the state of `pose_graph` (and the options of its
/// trajectories, indexed by trajectory id) to `writer` in the pbstream
/// format version 2: header, pose graph, trajectory builder options of all
/// non-deleted trajectories, submaps, nodes, trajectory data, IMU, odometry,
/// fixed frame pose and landmark data.
pub fn write_pb_stream(
    pose_graph: &dyn PoseGraph,
    trajectory_builder_options: &[proto::TrajectoryBuilderOptionsWithSensorIds],
    writer: &mut dyn ProtoStreamWriterInterface,
    include_unfinished_submaps: bool,
) {
    writer.write_proto(&create_header());
    writer.write_proto(&serialize_pose_graph(
        pose_graph,
        include_unfinished_submaps,
    ));
    writer.write_proto(&serialize_trajectory_builder_options(
        trajectory_builder_options,
        &get_valid_trajectory_ids(&pose_graph.get_trajectory_states()),
    ));

    serialize_submaps(
        &pose_graph.get_all_submap_data(),
        include_unfinished_submaps,
        writer,
    );
    serialize_trajectory_nodes(&pose_graph.get_trajectory_nodes(), writer);
    serialize_trajectory_data(&pose_graph.get_trajectory_data(), writer);
    serialize_imu_data(&pose_graph.get_imu_data(), writer);
    serialize_odometry_data(&pose_graph.get_odometry_data(), writer);
    serialize_fixed_frame_pose_data(&pose_graph.get_fixed_frame_pose_data(), writer);
    serialize_landmark_nodes(&pose_graph.get_landmark_nodes(), writer);
}
