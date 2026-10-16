//! Port of `cartographer/mapping/pose_graph` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ abstract class `PoseGraph` becomes the trait
//! [`PoseGraph`] (with [`PoseGraphInterface`] as supertrait). The C++
//! non-virtual `PoseGraph::ToProto(bool)` (implemented in the base class on
//! top of the virtual getters) is the free function [`pose_graph_to_proto`],
//! which implementations call from their `to_proto`. The nested
//! `PoseGraph::InitialTrajectoryPose` is the free type
//! [`InitialTrajectoryPose`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::PoisonError;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::{Time, to_universal};
use crate::mapping::constraints::create_constraint_builder_options;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::optimization::create_optimization_problem_options;
use crate::mapping::pose_graph_interface::{
    Constraint, LandmarkNode, Pose, PoseGraphInterface, Tag,
};
use crate::mapping::pose_graph_trimmer::PoseGraphTrimmer;
use crate::mapping::proto;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{self as transform_fns, to_rigid3};

use proto::pose_graph::constraint::Tag as ProtoTag;

/// `PoseGraph::InitialTrajectoryPose`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitialTrajectoryPose {
    /// The trajectory the pose is relative to.
    pub to_trajectory_id: i32,
    /// The relative pose.
    pub relative_pose: Rigid3d,
    /// The time at which `relative_pose` holds.
    pub time: Time,
}

/// The write API of the pose graph (C++ `PoseGraph`).
pub trait PoseGraph: PoseGraphInterface {
    /// Inserts an IMU measurement.
    fn add_imu_data(&self, trajectory_id: i32, imu_data: &ImuData);

    /// Inserts an odometry measurement.
    fn add_odometry_data(&self, trajectory_id: i32, odometry_data: &OdometryData);

    /// Inserts a fixed frame pose measurement.
    fn add_fixed_frame_pose_data(
        &self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    );

    /// Inserts landmarks observations.
    fn add_landmark_data(&self, trajectory_id: i32, landmark_data: &LandmarkData);

    /// Finishes the given trajectory.
    fn finish_trajectory(&self, trajectory_id: i32);

    /// Freezes a trajectory. Poses in this trajectory will not be optimized.
    fn freeze_trajectory(&self, trajectory_id: i32);

    /// Adds a `submap` from a proto with the given `global_pose` to the
    /// appropriate trajectory.
    fn add_submap_from_proto(&self, global_pose: &Rigid3d, submap: &proto::Submap);

    /// Adds a `node` from a proto with the given `global_pose` to the
    /// appropriate trajectory.
    fn add_node_from_proto(&self, global_pose: &Rigid3d, node: &proto::Node);

    /// Sets the trajectory data from a proto.
    fn set_trajectory_data_from_proto(&self, data: &proto::TrajectoryData);

    /// Adds information that `node_id` was inserted into `submap_id`. The
    /// submap has to be deserialized first.
    fn add_node_to_submap(&self, node_id: &NodeId, submap_id: &SubmapId);

    /// Adds serialized constraints. The corresponding trajectory nodes and
    /// submaps have to be deserialized before calling this function.
    fn add_serialized_constraints(&self, constraints: &[Constraint]);

    /// Adds a `trimmer`. It will be used after all data added before it has
    /// been included in the pose graph.
    fn add_trimmer(&self, trimmer: Box<dyn PoseGraphTrimmer>);

    /// Gets the current trajectory clusters.
    fn get_connected_trajectories(&self) -> Vec<Vec<i32>>;

    /// Returns the IMU data.
    fn get_imu_data(&self) -> MapByTime<ImuData>;

    /// Returns the odometry data.
    fn get_odometry_data(&self) -> MapByTime<OdometryData>;

    /// Returns the fixed frame pose data.
    fn get_fixed_frame_pose_data(&self) -> MapByTime<FixedFramePoseData>;

    /// Returns the landmark data.
    fn get_landmark_nodes(&self) -> BTreeMap<String /* landmark ID */, LandmarkNode>;

    /// Sets a relative initial pose `relative_pose` for `from_trajectory_id`
    /// with respect to `to_trajectory_id` at time `time`.
    fn set_initial_trajectory_pose(
        &self,
        from_trajectory_id: i32,
        to_trajectory_id: i32,
        pose: &Rigid3d,
        time: Time,
    );
}

/// Reads the optional `overlapping_submaps_trimmer_2d` dictionary into
/// `pose_graph_options` (C++ `PopulateOverlappingSubmapsTrimmerOptions2D`).
pub fn populate_overlapping_submaps_trimmer_options_2d(
    pose_graph_options: &mut proto::PoseGraphOptions,
    parameter_dictionary: &LuaParameterDictionary,
) {
    const DICTIONARY_KEY: &str = "overlapping_submaps_trimmer_2d";
    if !parameter_dictionary.has_key(DICTIONARY_KEY) {
        return;
    }
    let options_dictionary = parameter_dictionary.get_dictionary(DICTIONARY_KEY);
    pose_graph_options.overlapping_submaps_trimmer_2d = Some(
        proto::pose_graph_options::OverlappingSubmapsTrimmerOptions2D {
            fresh_submaps_count: options_dictionary.get_int("fresh_submaps_count"),
            min_covered_area: options_dictionary.get_double("min_covered_area"),
            min_added_submaps_count: options_dictionary.get_int("min_added_submaps_count"),
        },
    );
}

/// Reads `PoseGraphOptions` from `parameter_dictionary` (C++
/// `CreatePoseGraphOptions`).
pub fn create_pose_graph_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::PoseGraphOptions {
    let mut options = proto::PoseGraphOptions {
        optimize_every_n_nodes: parameter_dictionary.get_int("optimize_every_n_nodes"),
        ..Default::default()
    };
    options.constraint_builder_options = Some(create_constraint_builder_options(
        &parameter_dictionary.get_dictionary("constraint_builder"),
    ));
    options.matcher_translation_weight =
        parameter_dictionary.get_double("matcher_translation_weight");
    options.matcher_rotation_weight = parameter_dictionary.get_double("matcher_rotation_weight");
    options.optimization_problem_options = Some(create_optimization_problem_options(
        &parameter_dictionary.get_dictionary("optimization_problem"),
    ));
    options.max_num_final_iterations =
        parameter_dictionary.get_non_negative_int("max_num_final_iterations");
    assert!(
        options.max_num_final_iterations > 0,
        "Check failed: options.max_num_final_iterations() > 0 ({} vs. 0)",
        options.max_num_final_iterations
    );
    options.global_sampling_ratio = parameter_dictionary.get_double("global_sampling_ratio");
    options.log_residual_histograms = parameter_dictionary.get_bool("log_residual_histograms");
    options.global_constraint_search_after_n_seconds =
        parameter_dictionary.get_double("global_constraint_search_after_n_seconds");
    populate_overlapping_submaps_trimmer_options_2d(&mut options, parameter_dictionary);
    options
}

/// Converts a constraint tag to its proto enum (C++ `ToProto(const Tag&)`).
pub fn tag_to_proto(tag: Tag) -> ProtoTag {
    match tag {
        Tag::IntraSubmap => ProtoTag::IntraSubmap,
        Tag::InterSubmap => ProtoTag::InterSubmap,
    }
}

/// Converts a proto enum value to a constraint tag (C++
/// `FromProto(const proto::PoseGraph::Constraint::Tag&)`). Panics on
/// unsupported values, like the C++ `LOG(FATAL)`.
pub fn tag_from_proto(proto: i32) -> Tag {
    match ProtoTag::try_from(proto) {
        Ok(ProtoTag::IntraSubmap) => Tag::IntraSubmap,
        Ok(ProtoTag::InterSubmap) => Tag::InterSubmap,
        Err(_) => panic!("Unsupported tag."),
    }
}

/// Converts serialized constraints (C++
/// `FromProto(const RepeatedPtrField<proto::PoseGraph::Constraint>&)`).
pub fn from_proto(constraint_protos: &[proto::pose_graph::Constraint]) -> Vec<Constraint> {
    constraint_protos
        .iter()
        .map(|constraint_proto| {
            let submap_id_proto = constraint_proto.submap_id.unwrap_or_default();
            let node_id_proto = constraint_proto.node_id.unwrap_or_default();
            let submap_id =
                SubmapId::new(submap_id_proto.trajectory_id, submap_id_proto.submap_index);
            let node_id = NodeId::new(node_id_proto.trajectory_id, node_id_proto.node_index);
            let pose = Pose {
                zbar_ij: to_rigid3(&constraint_proto.relative_pose.unwrap_or_default()),
                translation_weight: constraint_proto.translation_weight,
                rotation_weight: constraint_proto.rotation_weight,
            };
            let tag = tag_from_proto(constraint_proto.tag);
            Constraint {
                submap_id,
                node_id,
                pose,
                tag,
            }
        })
        .collect()
}

/// Serializes a constraint (C++ `ToProto(const PoseGraph::Constraint&)`).
pub fn to_proto(constraint: &Constraint) -> proto::pose_graph::Constraint {
    proto::pose_graph::Constraint {
        submap_id: Some(proto::SubmapId {
            trajectory_id: constraint.submap_id.trajectory_id,
            submap_index: constraint.submap_id.submap_index,
        }),
        node_id: Some(proto::NodeId {
            trajectory_id: constraint.node_id.trajectory_id,
            node_index: constraint.node_id.node_index,
        }),
        relative_pose: Some(transform_fns::to_proto(&constraint.pose.zbar_ij)),
        translation_weight: constraint.pose.translation_weight,
        rotation_weight: constraint.pose.rotation_weight,
        tag: tag_to_proto(constraint.tag) as i32,
    }
}

/// Serializes the constraints and trajectories of `pose_graph` (the C++
/// base-class implementation `PoseGraph::ToProto(bool)`). If
/// `include_unfinished_submaps` is `false`, submaps that have not yet
/// received all rangefinder data insertions and their constraints are left
/// out.
pub fn pose_graph_to_proto<P: PoseGraphInterface + ?Sized>(
    pose_graph: &P,
    include_unfinished_submaps: bool,
) -> proto::PoseGraph {
    let mut proto = proto::PoseGraph::default();
    // Trajectory protos are created lazily, in order of first appearance.
    let mut trajectory_protos: BTreeMap<i32, usize> = BTreeMap::new();
    let mut trajectory = |proto: &mut proto::PoseGraph, trajectory_id: i32| -> usize {
        *trajectory_protos.entry(trajectory_id).or_insert_with(|| {
            proto.trajectory.push(proto::Trajectory {
                trajectory_id,
                ..Default::default()
            });
            proto.trajectory.len() - 1
        })
    };

    let mut unfinished_submaps: BTreeSet<SubmapId> = BTreeSet::new();
    for submap_id_data in &pose_graph.get_all_submap_data() {
        let trajectory_index = trajectory(&mut proto, submap_id_data.id.trajectory_id);
        let submap = submap_id_data
            .data
            .submap
            .as_ref()
            .expect("Check failed: submap_id_data.data.submap != nullptr");
        let insertion_finished = submap
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .insertion_finished();
        if !include_unfinished_submaps && !insertion_finished {
            // Collect IDs of all unfinished submaps and skip them.
            unfinished_submaps.insert(submap_id_data.id);
            continue;
        }
        proto.trajectory[trajectory_index]
            .submap
            .push(proto::trajectory::Submap {
                submap_index: submap_id_data.id.submap_index,
                pose: Some(transform_fns::to_proto(&submap_id_data.data.pose)),
            });
    }

    // RUST PORT NOTE: the C++ additionally collects the nodes of the skipped
    // constraints as "orphaned" but never uses them (all nodes are
    // serialized below); that dead code is not ported.
    let constraints_copy = pose_graph.constraints();
    proto.constraint.reserve(constraints_copy.len());
    for constraint in &constraints_copy {
        if !include_unfinished_submaps && unfinished_submaps.contains(&constraint.submap_id) {
            // Skip all those constraints that refer to unfinished submaps.
            continue;
        }
        proto.constraint.push(to_proto(constraint));
    }

    for node_id_data in &pose_graph.get_trajectory_nodes() {
        let trajectory_index = trajectory(&mut proto, node_id_data.id.trajectory_id);
        let constant_data = node_id_data
            .data
            .constant_data
            .as_ref()
            .expect("Check failed: node_id_data.data.constant_data != nullptr");
        proto.trajectory[trajectory_index]
            .node
            .push(proto::trajectory::Node {
                node_index: node_id_data.id.node_index,
                timestamp: to_universal(constant_data.time),
                pose: Some(transform_fns::to_proto(&node_id_data.data.global_pose)),
            });
    }

    let landmarks_copy = pose_graph.get_landmark_poses();
    proto.landmark_poses.reserve(landmarks_copy.len());
    for (landmark_id, global_pose) in &landmarks_copy {
        proto.landmark_poses.push(proto::pose_graph::LandmarkPose {
            landmark_id: landmark_id.clone(),
            global_pose: Some(transform_fns::to_proto(global_pose)),
        });
    }
    proto
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::internal::testing::test_helpers::{
        create_fake_constraint, create_fake_node, create_fake_submap_3d,
    };

    mod pose_graph {
        use super::*;

        #[test]
        fn serialize_constraint() {
            let expected_constraint =
                create_fake_constraint(&create_fake_node(1, 2), &create_fake_submap_3d(2, 3, true));
            let constraint_protos = vec![expected_constraint];
            let constraint = from_proto(&constraint_protos)[0];
            assert_eq!(expected_constraint, to_proto(&constraint));
        }
    }

    // Additional Rust tests (no C++ counterpart).
    mod create_pose_graph_options_test {
        use super::*;
        use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;

        #[test]
        fn reads_shipped_defaults() {
            let parameter_dictionary =
                resolve_lua_parameters("include \"pose_graph.lua\"\nreturn POSE_GRAPH");
            let options = create_pose_graph_options(&parameter_dictionary);
            assert_eq!(options.optimize_every_n_nodes, 90);
            assert_eq!(options.matcher_translation_weight, 5e2);
            assert_eq!(options.matcher_rotation_weight, 1.6e3);
            assert_eq!(options.max_num_final_iterations, 200);
            assert_eq!(options.global_sampling_ratio, 0.003);
            assert!(options.log_residual_histograms);
            assert_eq!(options.global_constraint_search_after_n_seconds, 10.);
            assert!(options.overlapping_submaps_trimmer_2d.is_none());
            let constraint_builder_options = options.constraint_builder_options.unwrap();
            assert_eq!(constraint_builder_options.sampling_ratio, 0.3);
            assert_eq!(
                options
                    .optimization_problem_options
                    .unwrap()
                    .ceres_solver_options
                    .unwrap()
                    .max_num_iterations,
                50
            );
        }

        #[test]
        fn reads_overlapping_submaps_trimmer_options() {
            let parameter_dictionary = resolve_lua_parameters(
                "include \"pose_graph.lua\"\n\
                 POSE_GRAPH.overlapping_submaps_trimmer_2d = {\n\
                   fresh_submaps_count = 1,\n\
                   min_covered_area = 2,\n\
                   min_added_submaps_count = 5,\n\
                 }\n\
                 return POSE_GRAPH",
            );
            let options = create_pose_graph_options(&parameter_dictionary);
            let trimmer_options = options.overlapping_submaps_trimmer_2d.unwrap();
            assert_eq!(trimmer_options.fresh_submaps_count, 1);
            assert_eq!(trimmer_options.min_covered_area, 2.);
            assert_eq!(trimmer_options.min_added_submaps_count, 5);
        }
    }

    #[test]
    fn tag_round_trip() {
        for tag in [Tag::IntraSubmap, Tag::InterSubmap] {
            assert_eq!(tag_from_proto(tag_to_proto(tag) as i32), tag);
        }
    }

    #[test]
    #[should_panic(expected = "Unsupported tag.")]
    fn unsupported_tag_panics() {
        tag_from_proto(7);
    }
}
