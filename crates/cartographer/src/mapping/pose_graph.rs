//! Port of `cartographer/mapping/pose_graph` (.h, .cc).
//!
//! Ported so far: `PoseGraph::InitialTrajectoryPose` and the constraint proto
//! conversions (`ToProto(Constraint)`, `FromProto(constraints)` and the tag
//! conversions).
//!
//! RUST PORT NOTE: not yet ported (they need types of other modules):
//! the `PoseGraph` trait itself and `PoseGraph::ToProto(bool)` (needs
//! `Submap`, `TrajectoryNode` and the `PoseGraphInterface` trait),
//! `CreatePoseGraphOptions` and its helper
//! `PopulateOverlappingSubmapsTrimmerOptions2D` (need
//! `constraints::CreateConstraintBuilderOptions` and
//! `optimization::CreateOptimizationProblemOptions`).

use crate::common::time::Time;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::pose_graph_interface::{Constraint, Pose, Tag};
use crate::mapping::proto;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::AngleAxis;
    use nalgebra::Vector3;

    /// `testing::CreateFakeConstraint(CreateFakeNode(1, 2),
    /// CreateFakeSubmap3D(2, 3))` (inlined until `internal/testing/test_helpers`
    /// is ported).
    fn create_fake_constraint() -> proto::pose_graph::Constraint {
        let pose = Rigid3d::with_angle_axis(
            Vector3::new(2., 3., 4.),
            AngleAxis::new(std::f64::consts::PI / 8., Vector3::x()),
        );
        proto::pose_graph::Constraint {
            submap_id: Some(proto::SubmapId {
                trajectory_id: 2,
                submap_index: 3,
            }),
            node_id: Some(proto::NodeId {
                trajectory_id: 1,
                node_index: 2,
            }),
            relative_pose: Some(transform_fns::to_proto(&pose)),
            translation_weight: f64::from(0.2f32),
            rotation_weight: f64::from(0.1f32),
            tag: ProtoTag::InterSubmap as i32,
        }
    }

    mod pose_graph {
        use super::*;

        #[test]
        fn serialize_constraint() {
            let expected_constraint = create_fake_constraint();
            let constraint_protos = vec![expected_constraint];
            let constraint = from_proto(&constraint_protos)[0];
            assert_eq!(expected_constraint, to_proto(&constraint));
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
