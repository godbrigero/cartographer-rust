//! Port of `cartographer/io/serialization_format_migration` (.h, .cc).
//!
//! Only [`migrate_submap_format_version1_to_version2`] is ported so far.
//! `MigrateStreamVersion1ToVersion2` rebuilds a `PoseGraph3D` from the input
//! stream and writes it with `WritePbStream`; it is ported together with
//! `mapping::PoseGraph3D` and `io::internal::mapping_state_serialization`.

use nalgebra::{DVector, Quaternion};

use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
use crate::mapping::proto;
use crate::mapping::trajectory_node;
use crate::transform::rigid_transform::{
    quaternion_conjugate, quaternion_mul, quaternion_squared_norm,
};
use crate::transform::transform::{get_yaw_quaternion, to_rigid3};

/// `Eigen::Quaternion::inverse()`: `conjugate / squaredNorm`, or the zero
/// quaternion if the norm is zero (no NaNs).
fn eigen_quaternion_inverse(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = quaternion_squared_norm(q);
    if n2 > 0. {
        quaternion_conjugate(q) / n2
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

/// Adds rotational scan matcher histograms to 3D submaps of format version 1
/// streams, computed from the histograms of the nodes inserted into them
/// (`INTRA_SUBMAP` constraints). 2D submaps (judged by the first submap) are
/// returned unchanged.
///
/// RUST PORT NOTE: C++ takes `node_id_to_node` by non-const reference but
/// never modifies it; it is borrowed immutably here.
pub fn migrate_submap_format_version1_to_version2(
    submap_id_to_submap: &MapById<SubmapId, proto::Submap>,
    node_id_to_node: &MapById<NodeId, proto::Node>,
    pose_graph_proto: &proto::PoseGraph,
) -> MapById<SubmapId, proto::Submap> {
    let first_is_2d = submap_id_to_submap
        .iter()
        .next()
        .is_some_and(|first| first.data.submap_2d.is_some());
    if submap_id_to_submap.empty() || first_is_2d {
        return submap_id_to_submap.clone();
    }

    let mut migrated_submaps = submap_id_to_submap.clone();
    for constraint_proto in &pose_graph_proto.constraint {
        if constraint_proto.tag == proto::pose_graph::constraint::Tag::IntraSubmap as i32 {
            let node_id_proto = constraint_proto.node_id.unwrap_or_default();
            let node_id = NodeId::new(node_id_proto.trajectory_id, node_id_proto.node_index);
            assert!(
                node_id_to_node.contains(&node_id),
                "Check failed: node_id_to_node.Contains(node_id)"
            );
            let node_data = trajectory_node::from_proto(
                &node_id_to_node
                    .at(&node_id)
                    .node_data
                    .clone()
                    .unwrap_or_default(),
            );
            let rotational_scan_matcher_histogram_in_gravity =
                &node_data.rotational_scan_matcher_histogram;

            let submap_id_proto = constraint_proto.submap_id.unwrap_or_default();
            let submap_id =
                SubmapId::new(submap_id_proto.trajectory_id, submap_id_proto.submap_index);
            assert!(
                migrated_submaps.contains(&submap_id),
                "Check failed: migrated_submaps.Contains(submap_id)"
            );
            let migrated_submap_proto = migrated_submaps.at_mut(&submap_id);
            let submap_3d_proto = migrated_submap_proto
                .submap_3d
                .as_mut()
                .expect("Check failed: migrated_submap_proto.has_submap_3d()");

            // Computed with raw (unnormalized) quaternion math like Eigen: an
            // unset gravity alignment is the zero quaternion, whose inverse
            // is zero, which leaves UnitX unrotated (yaw 0).
            let submap_local_pose = to_rigid3(&submap_3d_proto.local_pose.unwrap_or_default());
            let rotation = quaternion_mul(
                &quaternion_mul(
                    submap_local_pose.inverse().rotation(),
                    node_data.local_pose.rotation(),
                ),
                &eigen_quaternion_inverse(&node_data.gravity_alignment),
            );
            let submap_yaw_from_gravity: f64 = get_yaw_quaternion(&rotation);
            let rotational_scan_matcher_histogram_in_submap: DVector<f32> =
                RotationalScanMatcher::rotate_histogram(
                    rotational_scan_matcher_histogram_in_gravity,
                    submap_yaw_from_gravity as f32,
                );

            if submap_3d_proto.rotational_scan_matcher_histogram.is_empty() {
                submap_3d_proto
                    .rotational_scan_matcher_histogram
                    .extend(rotational_scan_matcher_histogram_in_submap.iter());
            } else {
                // Like C++ `Mutable(i)`, assumes equal histogram sizes.
                let submap_histogram = &mut submap_3d_proto.rotational_scan_matcher_histogram;
                for (i, value) in rotational_scan_matcher_histogram_in_submap
                    .iter()
                    .enumerate()
                {
                    submap_histogram[i] += value;
                }
            }
        }
    }
    migrated_submaps
}

#[cfg(test)]
mod tests {
    mod submap_histogram_migration_test {
        use crate::io::serialization_format_migration::*;
        use crate::mapping::id::{MapById, NodeId, SubmapId};
        use crate::mapping::proto;
        use crate::transform::rigid_transform::Rigid3d;
        use crate::transform::transform::to_proto;

        // C++ `mapping::testing::CreateFakeSubmap3D()` (defaults: trajectory
        // 1, submap 1, finished).
        fn create_fake_submap_3d(
            trajectory_id: i32,
            submap_index: i32,
            finished: bool,
        ) -> proto::Submap {
            proto::Submap {
                submap_id: Some(proto::SubmapId {
                    trajectory_id,
                    submap_index,
                }),
                submap_3d: Some(proto::Submap3D {
                    num_range_data: 1,
                    local_pose: Some(to_proto(&Rigid3d::identity())),
                    finished,
                    ..Default::default()
                }),
                ..Default::default()
            }
        }

        // C++ `mapping::testing::CreateFakeNode()` (defaults: trajectory 1,
        // node 1). Note: `gravity_alignment` is left unset.
        fn create_fake_node(trajectory_id: i32, node_index: i32) -> proto::Node {
            proto::Node {
                node_id: Some(proto::NodeId {
                    trajectory_id,
                    node_index,
                }),
                node_data: Some(proto::TrajectoryNodeData {
                    timestamp: 42,
                    local_pose: Some(to_proto(&Rigid3d::identity())),
                    ..Default::default()
                }),
            }
        }

        struct Fixture {
            pose_graph: proto::PoseGraph,
            submap: proto::Submap,
            node: proto::Node,
        }

        fn set_up() -> Fixture {
            // CreateSubmap()
            let mut submap = create_fake_submap_3d(1, 1, true);
            submap
                .submap_3d
                .as_mut()
                .unwrap()
                .rotational_scan_matcher_histogram
                .clear();
            // CreateNode()
            let mut node = create_fake_node(1, 1);
            const HISTOGRAM_SIZE: usize = 10;
            for _ in 0..HISTOGRAM_SIZE {
                node.node_data
                    .as_mut()
                    .unwrap()
                    .rotational_scan_matcher_histogram
                    .push(1.0);
            }
            // CreatePoseGraphWithNodeToSubmapConstraint()
            let mut pose_graph = proto::PoseGraph::default();
            pose_graph.constraint.push(proto::pose_graph::Constraint {
                tag: proto::pose_graph::constraint::Tag::IntraSubmap as i32,
                node_id: node.node_id,
                submap_id: submap.submap_id,
                relative_pose: Some(to_proto(&Rigid3d::identity())),
                ..Default::default()
            });
            Fixture {
                pose_graph,
                submap,
                node,
            }
        }

        #[test]
        fn submap_histogram_generation_from_trajectory_nodes() {
            let f = set_up();
            let mut submap_id_to_submap = MapById::<SubmapId, proto::Submap>::new();
            let submap_id_proto = f.submap.submap_id.unwrap();
            let submap_id =
                SubmapId::new(submap_id_proto.trajectory_id, submap_id_proto.submap_index);
            submap_id_to_submap.insert(&submap_id, f.submap.clone());

            let mut node_id_to_node = MapById::<NodeId, proto::Node>::new();
            let node_id_proto = f.node.node_id.unwrap();
            let node_id = NodeId::new(node_id_proto.trajectory_id, node_id_proto.node_index);
            node_id_to_node.insert(&node_id, f.node.clone());

            let submap_id_to_submap_migrated = migrate_submap_format_version1_to_version2(
                &submap_id_to_submap,
                &node_id_to_node,
                &f.pose_graph,
            );

            assert_eq!(
                submap_id_to_submap_migrated.size(),
                submap_id_to_submap.size()
            );
            let migrated_submap = submap_id_to_submap_migrated.at(&submap_id);
            assert!(migrated_submap.submap_2d.is_none());
            assert!(migrated_submap.submap_3d.is_some());
            let migrated_submap_3d = migrated_submap.submap_3d.as_ref().unwrap();
            let node_data = f.node.node_data.as_ref().unwrap();
            assert_eq!(
                migrated_submap_3d.rotational_scan_matcher_histogram.len(),
                node_data.rotational_scan_matcher_histogram.len()
            );
            for i in 0..node_data.rotational_scan_matcher_histogram.len() {
                assert_eq!(
                    migrated_submap_3d.rotational_scan_matcher_histogram[i],
                    node_data.rotational_scan_matcher_histogram[i]
                );
            }
        }

        #[test]
        fn existing_histograms_are_accumulated() {
            let mut f = set_up();
            f.submap
                .submap_3d
                .as_mut()
                .unwrap()
                .rotational_scan_matcher_histogram = vec![0.5; 10];
            let mut submaps = MapById::<SubmapId, proto::Submap>::new();
            let submap_id = SubmapId::new(1, 1);
            submaps.insert(&submap_id, f.submap.clone());
            let mut nodes = MapById::<NodeId, proto::Node>::new();
            nodes.insert(&NodeId::new(1, 1), f.node.clone());
            let migrated =
                migrate_submap_format_version1_to_version2(&submaps, &nodes, &f.pose_graph);
            assert_eq!(
                migrated
                    .at(&submap_id)
                    .submap_3d
                    .as_ref()
                    .unwrap()
                    .rotational_scan_matcher_histogram,
                vec![1.5; 10]
            );
        }

        #[test]
        fn yaw_rotates_histogram() {
            // A node yawed by pi/2 relative to the submap rotates its
            // histogram by pi/2 (5 of the 10 buckets covering pi).
            let mut f = set_up();
            let histogram: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let node_data = f.node.node_data.as_mut().unwrap();
            node_data.rotational_scan_matcher_histogram = histogram.clone();
            node_data.gravity_alignment = Some(to_proto(&Rigid3d::identity()).rotation.unwrap());
            node_data.local_pose = Some(to_proto(&Rigid3d::from_rotation(
                crate::transform::rigid_transform::AngleAxis::new(
                    std::f64::consts::FRAC_PI_2,
                    nalgebra::Vector3::z(),
                )
                .to_quaternion(),
            )));
            let mut submaps = MapById::<SubmapId, proto::Submap>::new();
            submaps.insert(&SubmapId::new(1, 1), f.submap.clone());
            let mut nodes = MapById::<NodeId, proto::Node>::new();
            nodes.insert(&NodeId::new(1, 1), f.node.clone());
            let migrated =
                migrate_submap_format_version1_to_version2(&submaps, &nodes, &f.pose_graph);
            let result = &migrated
                .at(&SubmapId::new(1, 1))
                .submap_3d
                .as_ref()
                .unwrap()
                .rotational_scan_matcher_histogram;
            let expected = RotationalScanMatcher::rotate_histogram(
                &nalgebra::DVector::from_vec(histogram),
                std::f64::consts::FRAC_PI_2 as f32,
            );
            assert_eq!(result.len(), 10);
            for (a, b) in result.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }

        #[test]
        fn submaps_2d_are_returned_unchanged() {
            let f = set_up();
            let mut submaps = MapById::<SubmapId, proto::Submap>::new();
            let submap = proto::Submap {
                submap_id: Some(proto::SubmapId {
                    trajectory_id: 1,
                    submap_index: 1,
                }),
                submap_2d: Some(proto::Submap2D::default()),
                ..Default::default()
            };
            submaps.insert(&SubmapId::new(1, 1), submap.clone());
            let migrated = migrate_submap_format_version1_to_version2(
                &submaps,
                &MapById::new(),
                &f.pose_graph,
            );
            assert_eq!(migrated.at(&SubmapId::new(1, 1)), &submap);
        }
    }
}
