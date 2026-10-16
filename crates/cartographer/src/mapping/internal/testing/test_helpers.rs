//! Port of `cartographer/mapping/internal/testing/test_helpers` (.h, .cc).

use nalgebra::Vector3;

use crate::common::config::SOURCE_DIRECTORY;
use crate::common::configuration_file_resolver::ConfigurationFileResolver;
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::{Time, from_seconds, from_universal};
use crate::mapping::proto;
use crate::sensor::point_cloud::{TimedPointCloud, transform_timed_point_cloud};
use crate::sensor::rangefinder_point::TimedRangefinderPoint;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::{AngleAxis, Rigid3d, Rigid3f};
use crate::transform::transform::to_proto;

/// Creates a dictionary from `lua_code`, resolving `include`s against the
/// `configuration_files` directory of the source tree.
pub fn resolve_lua_parameters(lua_code: &str) -> LuaParameterDictionary {
    let file_resolver =
        ConfigurationFileResolver::new(vec![format!("{SOURCE_DIRECTORY}/configuration_files")]);
    LuaParameterDictionary::new(lua_code, Box::new(file_resolver))
}

/// Generates range measurements of a half circle of radius 5 while moving
/// `travel_distance` along `(2, 1, 0)` in `duration` seconds.
pub fn generate_fake_range_measurements(
    travel_distance: f64,
    duration: f64,
    time_step: f64,
) -> Vec<TimedPointCloudData> {
    let direction: Vector3<f32> = Vector3::new(2., 1., 0.).normalize();
    generate_fake_range_measurements_with_transform(
        &(direction * travel_distance as f32),
        duration,
        time_step,
        &Rigid3f::identity(),
    )
}

/// Generates range measurements of a half circle of radius 5 while moving by
/// `translation` in `duration` seconds, with the given `local_to_global`
/// transform (C++ overload of `GenerateFakeRangeMeasurements`).
pub fn generate_fake_range_measurements_with_transform(
    translation: &Vector3<f32>,
    duration: f64,
    time_step: f64,
    local_to_global: &Rigid3f,
) -> Vec<TimedPointCloudData> {
    let mut measurements = Vec::new();
    let mut point_cloud: TimedPointCloud = Vec::new();
    let mut angle = 0.0f64;
    while angle < std::f64::consts::PI {
        for height in [-0.4, -0.2, 0.0, 0.2, 0.4] {
            const RADIUS: f64 = 5.;
            point_cloud.push(TimedRangefinderPoint {
                position: Vector3::new(RADIUS * angle.cos(), RADIUS * angle.sin(), height)
                    .cast::<f32>(),
                time: 0.,
            });
        }
        angle += 0.01;
    }
    // Eigen converts the double scalars to float.
    let velocity: Vector3<f32> = translation / duration as f32;
    let mut elapsed_time = 0.0f64;
    while elapsed_time < duration {
        let time: Time = from_universal(123) + from_seconds(elapsed_time);
        let global_pose =
            *local_to_global * Rigid3f::from_translation(velocity * elapsed_time as f32);
        let ranges = transform_timed_point_cloud(&point_cloud, &global_pose.inverse());
        measurements.push(TimedPointCloudData {
            time,
            origin: Vector3::zeros(),
            ranges,
            intensities: Vec::new(),
        });
        elapsed_time += time_step;
    }
    measurements
}

/// A fake serialized 3D submap with identity local pose and one range data.
pub fn create_fake_submap_3d(
    trajectory_id: i32,
    submap_index: i32,
    finished: bool,
) -> proto::Submap {
    proto::Submap {
        submap_id: Some(proto::SubmapId {
            trajectory_id,
            submap_index,
        }),
        submap_2d: None,
        submap_3d: Some(proto::Submap3D {
            num_range_data: 1,
            local_pose: Some(to_proto(&Rigid3d::identity())),
            finished,
            ..Default::default()
        }),
    }
}

/// [`create_fake_submap_3d`] with the C++ default arguments `(1, 1, true)`.
pub fn create_fake_submap_3d_default() -> proto::Submap {
    create_fake_submap_3d(1, 1, true)
}

/// A fake serialized node with timestamp 42 and identity local pose.
pub fn create_fake_node(trajectory_id: i32, node_index: i32) -> proto::Node {
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

/// [`create_fake_node`] with the C++ default arguments `(1, 1)`.
pub fn create_fake_node_default() -> proto::Node {
    create_fake_node(1, 1)
}

/// A fake INTER_SUBMAP constraint between `node` and `submap` with relative
/// pose `t = (2, 3, 4)`, `R = AngleAxis(pi / 8, X)` and weights `0.2f` /
/// `0.1f`.
pub fn create_fake_constraint(
    node: &proto::Node,
    submap: &proto::Submap,
) -> proto::pose_graph::Constraint {
    let submap_id = submap.submap_id.unwrap_or_default();
    let node_id = node.node_id.unwrap_or_default();
    let pose = Rigid3d::with_angle_axis(
        Vector3::new(2., 3., 4.),
        AngleAxis::new(std::f64::consts::PI / 8., Vector3::x()),
    );
    proto::pose_graph::Constraint {
        submap_id: Some(proto::SubmapId {
            trajectory_id: submap_id.trajectory_id,
            submap_index: submap_id.submap_index,
        }),
        node_id: Some(proto::NodeId {
            trajectory_id: node_id.trajectory_id,
            node_index: node_id.node_index,
        }),
        relative_pose: Some(to_proto(&pose)),
        translation_weight: f64::from(0.2f32),
        rotation_weight: f64::from(0.1f32),
        tag: proto::pose_graph::constraint::Tag::InterSubmap as i32,
    }
}

/// Returns the trajectory with `trajectory_id` of `pose_graph`, adding it if
/// needed.
pub fn create_trajectory_if_needed(
    trajectory_id: i32,
    pose_graph: &mut proto::PoseGraph,
) -> &mut proto::Trajectory {
    let position = pose_graph
        .trajectory
        .iter()
        .position(|trajectory| trajectory.trajectory_id == trajectory_id);
    let index = match position {
        Some(index) => index,
        None => {
            pose_graph.trajectory.push(proto::Trajectory {
                trajectory_id,
                ..Default::default()
            });
            pose_graph.trajectory.len() - 1
        }
    };
    &mut pose_graph.trajectory[index]
}

/// A fake landmark pose.
pub fn create_fake_landmark(
    landmark_id: &str,
    global_pose: &Rigid3d,
) -> proto::pose_graph::LandmarkPose {
    proto::pose_graph::LandmarkPose {
        landmark_id: landmark_id.to_string(),
        global_pose: Some(to_proto(global_pose)),
    }
}

/// Adds `node_data` to its trajectory in `pose_graph` (C++
/// `AddToProtoGraph(const proto::Node&, ...)`).
pub fn add_node_to_proto_graph(node_data: &proto::Node, pose_graph: &mut proto::PoseGraph) {
    let node_id = node_data.node_id.unwrap_or_default();
    let data = node_data.node_data.clone().unwrap_or_default();
    let trajectory = create_trajectory_if_needed(node_id.trajectory_id, pose_graph);
    trajectory.node.push(proto::trajectory::Node {
        timestamp: data.timestamp,
        node_index: node_id.node_index,
        // C++ assigns `mutable_pose()`, so the field is always present.
        pose: Some(data.local_pose.unwrap_or_default()),
    });
}

/// Adds `submap_data` to its trajectory in `pose_graph` (C++
/// `AddToProtoGraph(const proto::Submap&, ...)`).
pub fn add_submap_to_proto_graph(submap_data: &proto::Submap, pose_graph: &mut proto::PoseGraph) {
    let submap_id = submap_data.submap_id.unwrap_or_default();
    let pose = match &submap_data.submap_2d {
        Some(submap_2d) => submap_2d.local_pose,
        None => submap_data
            .submap_3d
            .as_ref()
            .and_then(|submap_3d| submap_3d.local_pose),
    };
    let trajectory = create_trajectory_if_needed(submap_id.trajectory_id, pose_graph);
    trajectory.submap.push(proto::trajectory::Submap {
        submap_index: submap_id.submap_index,
        // C++ assigns `mutable_pose()`, so the field is always present.
        pose: Some(pose.unwrap_or_default()),
    });
}

/// Adds `constraint` to `pose_graph` (C++ `AddToProtoGraph(const
/// proto::PoseGraph::Constraint&, ...)`).
pub fn add_constraint_to_proto_graph(
    constraint: &proto::pose_graph::Constraint,
    pose_graph: &mut proto::PoseGraph,
) {
    pose_graph.constraint.push(constraint.clone());
}

/// Adds `landmark` to `pose_graph` (C++ `AddToProtoGraph(const
/// proto::PoseGraph::LandmarkPose&, ...)`).
pub fn add_landmark_to_proto_graph(
    landmark: &proto::pose_graph::LandmarkPose,
    pose_graph: &mut proto::PoseGraph,
) {
    pose_graph.landmark_poses.push(landmark.clone());
}

/// Overloaded C++ `AddToProtoGraph`: implemented for every type that can be
/// added to a `proto::PoseGraph`.
pub trait AddToProtoGraph {
    /// Adds `self` to `pose_graph`.
    fn add_to_proto_graph(&self, pose_graph: &mut proto::PoseGraph);
}

impl AddToProtoGraph for proto::Node {
    fn add_to_proto_graph(&self, pose_graph: &mut proto::PoseGraph) {
        add_node_to_proto_graph(self, pose_graph);
    }
}

impl AddToProtoGraph for proto::Submap {
    fn add_to_proto_graph(&self, pose_graph: &mut proto::PoseGraph) {
        add_submap_to_proto_graph(self, pose_graph);
    }
}

impl AddToProtoGraph for proto::pose_graph::Constraint {
    fn add_to_proto_graph(&self, pose_graph: &mut proto::PoseGraph) {
        add_constraint_to_proto_graph(self, pose_graph);
    }
}

impl AddToProtoGraph for proto::pose_graph::LandmarkPose {
    fn add_to_proto_graph(&self, pose_graph: &mut proto::PoseGraph) {
        add_landmark_to_proto_graph(self, pose_graph);
    }
}

/// C++ `AddToProtoGraph(data, pose_graph)` for any supported `data`.
pub fn add_to_proto_graph<T: AddToProtoGraph + ?Sized>(
    data: &T,
    pose_graph: &mut proto::PoseGraph,
) {
    data.add_to_proto_graph(pose_graph);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_lua_parameters_reads_pose_graph_lua() {
        let dictionary = resolve_lua_parameters(
            r#"
            include "pose_graph.lua"
            return { sampling_ratio = POSE_GRAPH.constraint_builder.sampling_ratio }"#,
        );
        assert_eq!(dictionary.get_double("sampling_ratio"), 0.3);
    }

    #[test]
    fn fake_range_measurements() {
        let measurements = generate_fake_range_measurements(1., 2., 0.5);
        assert_eq!(measurements.len(), 4);
        // ceil(pi / 0.01) = 315 angles, 5 heights each.
        assert_eq!(measurements[0].ranges.len(), 315 * 5);
        assert_eq!(measurements[0].time, from_universal(123));
        assert_eq!(
            measurements[1].time,
            from_universal(123) + from_seconds(0.5)
        );
        // The first measurement is taken at the origin.
        let p = measurements[0].ranges[0].position;
        assert!((p - Vector3::new(5., 0., -0.4)).norm() < 1e-6);
        // Later measurements are shifted against the direction of motion.
        let direction = Vector3::new(2.0f32, 1., 0.).normalize();
        let q = measurements[2].ranges[0].position;
        assert!((q - (Vector3::new(5., 0., -0.4) - direction * 0.5)).norm() < 1e-5);
    }

    #[test]
    fn proto_graph_helpers() {
        let mut pose_graph = proto::PoseGraph::default();
        let node = create_fake_node(1, 2);
        let submap = create_fake_submap_3d(2, 3, true);
        add_to_proto_graph(&node, &mut pose_graph);
        add_to_proto_graph(&submap, &mut pose_graph);
        add_to_proto_graph(&create_fake_constraint(&node, &submap), &mut pose_graph);
        add_to_proto_graph(
            &create_fake_landmark("landmark", &Rigid3d::identity()),
            &mut pose_graph,
        );
        assert_eq!(pose_graph.trajectory.len(), 2);
        assert_eq!(pose_graph.trajectory[0].trajectory_id, 1);
        assert_eq!(pose_graph.trajectory[0].node[0].node_index, 2);
        assert_eq!(pose_graph.trajectory[0].node[0].timestamp, 42);
        assert_eq!(pose_graph.trajectory[1].trajectory_id, 2);
        assert_eq!(pose_graph.trajectory[1].submap[0].submap_index, 3);
        assert_eq!(pose_graph.constraint.len(), 1);
        assert_eq!(pose_graph.landmark_poses[0].landmark_id, "landmark");
        // Adding to an existing trajectory does not create a new one.
        add_to_proto_graph(&create_fake_node(1, 3), &mut pose_graph);
        assert_eq!(pose_graph.trajectory.len(), 2);
        assert_eq!(pose_graph.trajectory[0].node.len(), 2);
    }
}
