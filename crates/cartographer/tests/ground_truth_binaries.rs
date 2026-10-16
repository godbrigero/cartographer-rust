//! Smoke and end-to-end tests of the `cartographer_autogenerate_ground_truth`
//! and `cartographer_compute_relations_metrics` binaries.

use std::path::PathBuf;
use std::process::Command;

use cartographer::ground_truth::proto::GroundTruth;
use cartographer::io::proto_stream::ProtoStreamWriter;
use cartographer::io::proto_stream_interface::ProtoStreamWriterInterface;
use cartographer::mapping::proto::pose_graph::Constraint;
use cartographer::mapping::proto::pose_graph::constraint::Tag;
use cartographer::mapping::proto::serialized_data::Data;
use cartographer::mapping::proto::trajectory::{Node, Submap};
use cartographer::mapping::proto::{
    AllTrajectoryBuilderOptions, NodeId, PoseGraph, SerializationHeader, SerializedData, SubmapId,
    Trajectory,
};
use cartographer::transform::rigid_transform::{Rigid3d, roll_pitch_yaw};
use cartographer::transform::transform::to_proto;
use nalgebra::Vector3;
use prost::Message;

const AUTOGENERATE: &str = env!("CARGO_BIN_EXE_cartographer_autogenerate_ground_truth");
const COMPUTE_METRICS: &str = env!("CARGO_BIN_EXE_cartographer_compute_relations_metrics");

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "cartographer_ground_truth_binaries_{}_{name}",
        std::process::id()
    ))
}

#[test]
fn autogenerate_ground_truth_help() {
    let output = Command::new(AUTOGENERATE).arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--pose_graph_filename",
        "--output_filename",
        "--min_covered_distance",
        "--outlier_threshold_meters",
        "--outlier_threshold_radians",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in:\n{stdout}");
    }
}

#[test]
fn compute_relations_metrics_help() {
    let output = Command::new(COMPUTE_METRICS)
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--pose_graph_filename",
        "--relations_filename",
        "--read_text_file_with_unix_timestamps",
        "--write_relation_metrics",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in:\n{stdout}");
    }
}

#[test]
fn missing_filenames_fail() {
    let output = Command::new(AUTOGENERATE).output().unwrap();
    assert!(!output.status.success());
    let output = Command::new(COMPUTE_METRICS)
        .arg("--pose_graph_filename=foo")
        .output()
        .unwrap();
    assert!(!output.status.success());
}

/// Straight trajectory along x, 1 m per node; submap `s` starts at node
/// `2 s`; two loop closures with small errors.
fn pose_graph() -> PoseGraph {
    let num_nodes: usize = 21;
    let num_submaps: usize = 10;
    let node_pose = |i: usize| Rigid3d::from_translation(Vector3::new(i as f64, 0., 0.));
    let submap_pose = |s: usize| Rigid3d::from_translation(Vector3::new(2. * s as f64, 0., 0.));
    let constraint = |submap: usize, node: usize, tag: Tag, relative_pose: Rigid3d| Constraint {
        submap_id: Some(SubmapId {
            trajectory_id: 0,
            submap_index: submap as i32,
        }),
        node_id: Some(NodeId {
            trajectory_id: 0,
            node_index: node as i32,
        }),
        relative_pose: Some(to_proto(&relative_pose)),
        translation_weight: 1.,
        rotation_weight: 1.,
        tag: tag as i32,
    };
    let mut constraints = Vec::new();
    for node in 0..num_nodes {
        let newest = (node / 2).min(num_submaps - 1);
        for submap in newest.saturating_sub(1)..=newest {
            constraints.push(constraint(
                submap,
                node,
                Tag::IntraSubmap,
                submap_pose(submap).inverse() * node_pose(node),
            ));
        }
    }
    constraints.push(constraint(
        0,
        20,
        Tag::InterSubmap,
        submap_pose(0).inverse()
            * node_pose(20)
            * Rigid3d::new(Vector3::new(0.05, 0., 0.), roll_pitch_yaw(0.01, 0., 0.)),
    ));
    constraints.push(constraint(
        1,
        19,
        Tag::InterSubmap,
        submap_pose(1).inverse()
            * node_pose(19)
            * Rigid3d::from_translation(Vector3::new(0., 0., 0.1)),
    ));
    PoseGraph {
        constraint: constraints,
        trajectory: vec![Trajectory {
            trajectory_id: 0,
            node: (0..num_nodes)
                .map(|i| Node {
                    node_index: i as i32,
                    timestamp: 1_000_000 + 100 * i as i64,
                    pose: Some(to_proto(&node_pose(i))),
                })
                .collect(),
            submap: (0..num_submaps)
                .map(|s| Submap {
                    submap_index: s as i32,
                    pose: Some(to_proto(&submap_pose(s))),
                })
                .collect(),
        }],
        landmark_poses: vec![],
    }
}

fn write_pbstream(filename: &str, pose_graph: PoseGraph) {
    let mut writer = ProtoStreamWriter::new(filename);
    writer.write_proto(&SerializationHeader { format_version: 2 });
    writer.write_proto(&SerializedData {
        data: Some(Data::PoseGraph(pose_graph)),
    });
    writer.write_proto(&SerializedData {
        data: Some(Data::AllTrajectoryBuilderOptions(
            AllTrajectoryBuilderOptions {
                options_with_sensor_ids: vec![Default::default()],
            },
        )),
    });
    assert!(writer.close());
}

#[test]
fn end_to_end() {
    let pbstream = temp_path("pose_graph.pbstream");
    let ground_truth_path = temp_path("ground_truth.pb");
    let csv_path = PathBuf::from(format!("{}.relation_metrics.csv", pbstream.display()));
    let pbstream_str = pbstream.to_str().unwrap();
    write_pbstream(pbstream_str, pose_graph());

    let output = Command::new(AUTOGENERATE)
        .arg(format!("--pose_graph_filename={pbstream_str}"))
        .arg(format!("--output_filename={}", ground_truth_path.display()))
        .args(["--min_covered_distance", "10"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("Generated 2 relations and ignored 0 outliers."),
        "{stderr}"
    );
    let ground_truth =
        GroundTruth::decode(std::fs::read(&ground_truth_path).unwrap().as_slice()).unwrap();
    assert_eq!(ground_truth.relation.len(), 2);

    let output = Command::new(COMPUTE_METRICS)
        .arg(format!("--pose_graph_filename={pbstream_str}"))
        .arg(format!(
            "--relations_filename={}",
            ground_truth_path.display()
        ))
        .arg("--write_relation_metrics")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    // The generated relations reproduce the constraints, so the pose graph
    // solution differs from them by the injected errors: 0.05 m / 0.01 rad
    // and 0.1 m / 0 rad.
    assert!(
        stderr.contains("Abs translational error 0.07500 +/- 0.03536 m\n"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Abs rotational error 0.28648 +/- 0.40514 deg\n"),
        "{stderr}"
    );
    let csv = std::fs::read_to_string(&csv_path).unwrap();
    assert_eq!(csv.lines().count(), 3);
    assert!(csv.starts_with("translational_error,squared_translational_error,"));

    std::fs::remove_file(&pbstream).unwrap();
    std::fs::remove_file(&ground_truth_path).unwrap();
    std::fs::remove_file(&csv_path).unwrap();
}
