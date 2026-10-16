//! Port of `cartographer/ground_truth/autogenerate_ground_truth` (.h, .cc).

use super::proto;
use crate::mapping::proto::pose_graph::constraint::Tag;
use crate::mapping::proto::{PoseGraph, Trajectory};
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{get_angle, to_proto, to_rigid3};

fn node_pose(trajectory: &Trajectory, index: usize) -> Rigid3d {
    to_rigid3(&trajectory.node[index].pose.unwrap_or_default())
}

fn compute_covered_distance(trajectory: &Trajectory) -> Vec<f64> {
    let mut covered_distance = vec![0.];
    assert!(
        !trajectory.node.is_empty(),
        "Check failed: trajectory.node_size() > 0 Trajectory does not contain any nodes."
    );
    for i in 1..trajectory.node.len() {
        let last_pose = node_pose(trajectory, i - 1);
        let this_pose = node_pose(trajectory, i);
        let last = *covered_distance.last().unwrap();
        covered_distance.push(last + (last_pose.inverse() * this_pose).translation().norm());
    }
    covered_distance
}

/// We pick the representative node in the middle of the submap.
///
/// TODO(whess): Should we consider all nodes inserted into the submap and
/// exclude, e.g. based on large relative linear or angular distance?
fn compute_submap_representative_node(pose_graph: &PoseGraph) -> Vec<i32> {
    let mut submap_to_node_index = Vec::new();
    for constraint in &pose_graph.constraint {
        if constraint.tag != Tag::IntraSubmap as i32 {
            continue;
        }
        let submap_id = constraint.submap_id.unwrap_or_default();
        let node_id = constraint.node_id.unwrap_or_default();
        assert_eq!(
            submap_id.trajectory_id, 0,
            "Check failed: constraint.submap_id().trajectory_id() == 0"
        );
        assert_eq!(
            node_id.trajectory_id, 0,
            "Check failed: constraint.node_id().trajectory_id() == 0"
        );

        let next_submap_index = submap_to_node_index.len() as i32;
        let submap_index = submap_id.submap_index;
        if submap_index <= next_submap_index {
            continue;
        }

        assert_eq!(
            submap_index,
            next_submap_index + 1,
            "Check failed: submap_index == next_submap_index + 1"
        );
        submap_to_node_index.push(node_id.node_index);
    }
    submap_to_node_index
}

/// Generates GroundTruth proto from the given pose graph using the specified
/// criteria parameters. See
/// 'https://google-cartographer.readthedocs.io/en/latest/evaluation.html' for
/// more details.
///
/// Only trajectory 0 is used; all constraints must refer to trajectory 0 and
/// node/submap indices are positions in the trajectory's node/submap lists
/// (panics otherwise, like the C++ `CHECK`s / `at()`).
pub fn generate_ground_truth(
    pose_graph: &PoseGraph,
    min_covered_distance: f64,
    outlier_threshold_meters: f64,
    outlier_threshold_radians: f64,
) -> proto::GroundTruth {
    let trajectory = &pose_graph.trajectory[0];
    let covered_distance = compute_covered_distance(trajectory);

    let submap_to_node_index = compute_submap_representative_node(pose_graph);

    let mut num_outliers = 0;
    let mut ground_truth = proto::GroundTruth::default();
    for constraint in &pose_graph.constraint {
        // We're only interested in loop closure constraints.
        if constraint.tag == Tag::IntraSubmap as i32 {
            continue;
        }

        // For some submaps at the very end, we have not chosen a
        // representative node, but those should not be part of loop closure
        // anyway.
        let submap_id = constraint.submap_id.unwrap_or_default();
        let node_id = constraint.node_id.unwrap_or_default();
        assert_eq!(
            submap_id.trajectory_id, 0,
            "Check failed: constraint.submap_id().trajectory_id() == 0"
        );
        assert_eq!(
            node_id.trajectory_id, 0,
            "Check failed: constraint.node_id().trajectory_id() == 0"
        );
        if submap_id.submap_index >= submap_to_node_index.len() as i32 {
            continue;
        }
        let submap_index = usize::try_from(submap_id.submap_index).expect("negative submap_index");
        let matched_node = usize::try_from(node_id.node_index).expect("negative node_index");
        let representative_node = usize::try_from(submap_to_node_index[submap_index])
            .expect("negative representative node_index");

        // Covered distance between the two should not be too small.
        let covered_distance_in_constraint =
            (covered_distance[matched_node] - covered_distance[representative_node]).abs();
        if covered_distance_in_constraint < min_covered_distance {
            continue;
        }

        // Compute the transform between the nodes according to the solution
        // and the constraint.
        let solution_pose1 = node_pose(trajectory, representative_node);
        let solution_pose2 = node_pose(trajectory, matched_node);
        let solution = solution_pose1.inverse() * solution_pose2;

        let submap_solution = to_rigid3(&trajectory.submap[submap_index].pose.unwrap_or_default());
        let submap_solution_to_node_solution = solution_pose1.inverse() * submap_solution;
        let node_to_submap_constraint = to_rigid3(&constraint.relative_pose.unwrap_or_default());
        let expected = submap_solution_to_node_solution * node_to_submap_constraint;

        let error = solution * expected.inverse();

        if error.translation().norm() > outlier_threshold_meters
            || get_angle(&error) > outlier_threshold_radians
        {
            num_outliers += 1;
            continue;
        }
        ground_truth.relation.push(proto::Relation {
            timestamp1: trajectory.node[representative_node].timestamp,
            timestamp2: trajectory.node[matched_node].timestamp,
            expected: Some(to_proto(&expected)),
            covered_distance: covered_distance_in_constraint,
        });
    }
    log::info!(
        "Generated {} relations and ignored {} outliers.",
        ground_truth.relation.len(),
        num_outliers
    );
    ground_truth
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::mapping::proto::pose_graph::Constraint;
    use crate::mapping::proto::trajectory::{Node, Submap};
    use crate::mapping::proto::{NodeId, SubmapId};
    use crate::transform::rigid_transform::roll_pitch_yaw;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;

    const NUM_NODES: usize = 21;
    const NODES_PER_SUBMAP_HALF: usize = 2;

    fn node_global_pose(i: usize) -> Rigid3d {
        // Straight path along x with 1 m spacing.
        Rigid3d::from_translation(Vector3::new(i as f64, 0., 0.))
    }

    fn submap_global_pose(s: usize) -> Rigid3d {
        Rigid3d::new(
            Vector3::new((s * NODES_PER_SUBMAP_HALF) as f64, 0.5, 0.),
            roll_pitch_yaw(0., 0., 0.1),
        )
    }

    fn constraint(submap: usize, node: usize, tag: Tag, relative_pose: &Rigid3d) -> Constraint {
        Constraint {
            submap_id: Some(SubmapId {
                trajectory_id: 0,
                submap_index: submap as i32,
            }),
            node_id: Some(NodeId {
                trajectory_id: 0,
                node_index: node as i32,
            }),
            relative_pose: Some(to_proto(relative_pose)),
            translation_weight: 1.,
            rotation_weight: 1.,
            tag: tag as i32,
        }
    }

    /// Exact node-in-submap pose according to the global solution.
    fn consistent_relative_pose(submap: usize, node: usize) -> Rigid3d {
        submap_global_pose(submap).inverse() * node_global_pose(node)
    }

    /// A straight trajectory of 21 nodes (1 m apart), submaps starting every
    /// 2 nodes and spanning 4 nodes (so consecutive submaps overlap by half).
    /// Loop closures (INTER_SUBMAP) are added by the caller.
    fn straight_pose_graph() -> PoseGraph {
        let num_submaps = (NUM_NODES - 1) / NODES_PER_SUBMAP_HALF;
        let trajectory = Trajectory {
            trajectory_id: 0,
            node: (0..NUM_NODES)
                .map(|i| Node {
                    node_index: i as i32,
                    timestamp: 1000 + 10 * i as i64,
                    pose: Some(to_proto(&node_global_pose(i))),
                })
                .collect(),
            submap: (0..num_submaps)
                .map(|s| Submap {
                    submap_index: s as i32,
                    pose: Some(to_proto(&submap_global_pose(s))),
                })
                .collect(),
        };
        let mut constraints = Vec::new();
        for node in 0..NUM_NODES {
            // Node i is inserted into submaps (i / 2 - 1) and (i / 2).
            let newest = (node / NODES_PER_SUBMAP_HALF).min(num_submaps - 1);
            let oldest = newest.saturating_sub(1);
            for submap in oldest..=newest {
                constraints.push(constraint(
                    submap,
                    node,
                    Tag::IntraSubmap,
                    &consistent_relative_pose(submap, node),
                ));
            }
        }
        PoseGraph {
            constraint: constraints,
            trajectory: vec![trajectory],
            landmark_poses: vec![],
        }
    }

    #[test]
    fn covered_distance_along_straight_path() {
        let pose_graph = straight_pose_graph();
        let covered = compute_covered_distance(&pose_graph.trajectory[0]);
        assert_eq!(covered.len(), NUM_NODES);
        for (i, d) in covered.iter().enumerate() {
            assert!((d - i as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn representative_nodes() {
        let pose_graph = straight_pose_graph();
        // Submap i+1 is first seen (as intra constraint) with node 2 * (i+1).
        let representative = compute_submap_representative_node(&pose_graph);
        assert_eq!(representative, vec![2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    #[should_panic(expected = "trajectory.node_size() > 0")]
    fn empty_trajectory_panics() {
        let mut pose_graph = straight_pose_graph();
        pose_graph.trajectory[0].node.clear();
        generate_ground_truth(&pose_graph, 0., 1., 1.);
    }

    #[test]
    #[should_panic(expected = "submap_index == next_submap_index + 1")]
    fn submap_gap_panics() {
        let mut pose_graph = straight_pose_graph();
        pose_graph
            .constraint
            .insert(0, constraint(2, 0, Tag::IntraSubmap, &Rigid3d::identity()));
        generate_ground_truth(&pose_graph, 0., 1., 1.);
    }

    #[test]
    fn loop_closure_and_outlier_rejection() {
        let mut pose_graph = straight_pose_graph();
        // 0: Consistent loop closure: node 20 against submap 1 (rep node 4).
        pose_graph.constraint.push(constraint(
            1,
            20,
            Tag::InterSubmap,
            &consistent_relative_pose(1, 20),
        ));
        // 1: Consistent but too short: node 7 against submap 2 (rep node 6).
        pose_graph.constraint.push(constraint(
            2,
            7,
            Tag::InterSubmap,
            &consistent_relative_pose(2, 7),
        ));
        // 2: Translational outlier (0.2 m off): node 19 against submap 0.
        pose_graph.constraint.push(constraint(
            0,
            19,
            Tag::InterSubmap,
            &(consistent_relative_pose(0, 19)
                * Rigid3d::from_translation(Vector3::new(0., 0.2, 0.))),
        ));
        // 3: Rotational outlier (0.05 rad off): node 18 against submap 3.
        pose_graph.constraint.push(constraint(
            3,
            18,
            Tag::InterSubmap,
            &(consistent_relative_pose(3, 18)
                * Rigid3d::from_rotation(roll_pitch_yaw(0.05, 0., 0.))),
        ));
        // 4: Small error within thresholds: node 17 against submap 0 (rep 2).
        // Rotation about x (parallel to the lever arm) so it adds no translation error.
        let small_error = Rigid3d::new(Vector3::new(0.1, 0., 0.), roll_pitch_yaw(0.01, 0., 0.));
        pose_graph.constraint.push(constraint(
            0,
            17,
            Tag::InterSubmap,
            &(consistent_relative_pose(0, 17) * small_error),
        ));
        // 5: Submap without representative node (the last one): ignored.
        pose_graph.constraint.push(constraint(
            9,
            0,
            Tag::InterSubmap,
            &consistent_relative_pose(9, 0),
        ));

        let ground_truth = generate_ground_truth(&pose_graph, 10., 0.15, 0.02);
        assert_eq!(ground_truth.relation.len(), 2);

        let r0 = &ground_truth.relation[0];
        assert_eq!(r0.timestamp1, 1000 + 10 * 4);
        assert_eq!(r0.timestamp2, 1000 + 10 * 20);
        assert!((r0.covered_distance - 16.).abs() < 1e-12);
        assert_is_nearly(
            &to_rigid3(r0.expected.as_ref().unwrap()),
            &Rigid3d::from_translation(Vector3::new(16., 0., 0.)),
            1e-9,
        );

        let r1 = &ground_truth.relation[1];
        assert_eq!(r1.timestamp1, 1000 + 10 * 2);
        assert_eq!(r1.timestamp2, 1000 + 10 * 17);
        assert!((r1.covered_distance - 15.).abs() < 1e-12);
        assert_is_nearly(
            &to_rigid3(r1.expected.as_ref().unwrap()),
            &(Rigid3d::from_translation(Vector3::new(15., 0., 0.)) * small_error),
            1e-9,
        );

        // Loosening the thresholds accepts the outliers too.
        let ground_truth = generate_ground_truth(&pose_graph, 10., 1., 1.);
        assert_eq!(ground_truth.relation.len(), 4);
        // Lowering min_covered_distance accepts the short constraint.
        let ground_truth = generate_ground_truth(&pose_graph, 0., 1., 1.);
        assert_eq!(ground_truth.relation.len(), 5);
        assert!((ground_truth.relation[1].covered_distance - 1.).abs() < 1e-12);
    }
}
