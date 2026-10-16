//! Library part of `cartographer/ground_truth/compute_relations_metrics_main.cc`.
//!
//! RUST PORT NOTE: this file has no C++ counterpart; the C++ keeps all of
//! this in an anonymous namespace of the `main` file. It is moved into the
//! library so it can be unit tested and the
//! `cartographer_compute_relations_metrics` binary stays thin.
//!
//! Computes the relation based metric described in:
//! R. Kuemmerle, B. Steder, C. Dornhege, M. Ruhnke, G. Grisetti,
//! C. Stachniss, and A. Kleiner, "On measuring the accuracy of SLAM
//! algorithms," Autonomous Robots, vol. 27, no. 4, pp. 387–407, 2009.

use prost::Message;

use super::proto;
use super::relations_text_file::read_relations_text_file;
use crate::common::math::{pow2, rad_to_deg};
use crate::common::printf::{format_f_precision, format_g};
use crate::common::time::{Time, from_universal};
use crate::io::proto_stream_deserializer::deserialize_pose_graph_from_file;
use crate::mapping::proto::Trajectory;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{get_angle, to_rigid3};
use crate::transform::transform_interpolation_buffer::TransformInterpolationBuffer;

/// Squared translational and rotational error of one relation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Error {
    /// Squared norm of the translation error, in m^2.
    pub translational_squared: f64,
    /// Squared rotation angle of the error, in rad^2.
    pub rotational_squared: f64,
}

/// Computes the error of the relative transform `pose1^-1 * pose2` with
/// respect to `expected`.
///
/// TODO(whess): This gives different results for the translational error if
/// 'pose1' and 'pose2' are swapped and 'expected' is inverted. Consider a
/// different way to compute translational error. Maybe just look at the
/// absolute difference in translation norms of each relative transform as a
/// lower bound of the translational error.
pub fn compute_error(pose1: &Rigid3d, pose2: &Rigid3d, expected: &Rigid3d) -> Error {
    let error = (pose1.inverse() * *pose2) * expected.inverse();
    Error {
        translational_squared: error.translation().norm_squared(),
        rotational_squared: pow2(get_angle(&error)),
    }
}

/// Returns `"<mean> +/- <sample standard deviation>"` with 5 fixed decimals.
/// Panics if fewer than 2 values are given (C++ `CHECK_GE`).
pub fn mean_and_std_dev_string(values: &[f64]) -> String {
    assert!(
        values.len() >= 2,
        "Check failed: values.size() >= 2 ({} vs. 2)",
        values.len()
    );
    let mean = values.iter().fold(0., |sum, value| sum + value) / values.len() as f64;
    let mut sum_of_squared_differences = 0.;
    for value in values {
        sum_of_squared_differences += pow2(value - mean);
    }
    let standard_deviation = (sum_of_squared_differences / (values.len() - 1) as f64).sqrt();
    format!(
        "{} +/- {}",
        format_f_precision(mean, 5),
        format_f_precision(standard_deviation, 5)
    )
}

/// Formats the statistics of `errors` (absolute and squared translational
/// and rotational errors) as printed by the C++ tool.
pub fn statistics_string(errors: &[Error]) -> String {
    let mut translational_errors = Vec::with_capacity(errors.len());
    let mut squared_translational_errors = Vec::with_capacity(errors.len());
    let mut rotational_errors_degrees = Vec::with_capacity(errors.len());
    let mut squared_rotational_errors_degrees = Vec::with_capacity(errors.len());
    for error in errors {
        translational_errors.push(error.translational_squared.sqrt());
        squared_translational_errors.push(error.translational_squared);
        let rotational_error_degrees = rad_to_deg(error.rotational_squared.sqrt());
        rotational_errors_degrees.push(rotational_error_degrees);
        squared_rotational_errors_degrees.push(pow2(rotational_error_degrees));
    }
    format!(
        "Abs translational error {} m\n\
         Sqr translational error {} m^2\n\
         Abs rotational error {} deg\n\
         Sqr rotational error {} deg^2\n",
        mean_and_std_dev_string(&translational_errors),
        mean_and_std_dev_string(&squared_translational_errors),
        mean_and_std_dev_string(&rotational_errors_degrees),
        mean_and_std_dev_string(&squared_rotational_errors_degrees),
    )
}

/// Header line of the relation metrics CSV file.
pub const RELATION_METRICS_CSV_HEADER: &str = "translational_error,squared_translational_error,\
rotational_errors_degree,squared_rotational_errors_degree,expected_translation_x,\
expected_translation_y,expected_translation_z,expected_rotation_w,expected_rotation_x,\
expected_rotation_y,expected_rotation_z,covered_distance\n";

/// Returns the contents of the relation metrics CSV file: a header and one
/// row per relation, numbers formatted like a default C++ `std::ostream`
/// (`%g`, 6 significant digits). `errors[i]` belongs to
/// `ground_truth.relation[i]`.
pub fn relation_metrics_csv(errors: &[Error], ground_truth: &proto::GroundTruth) -> String {
    let mut out = String::from(RELATION_METRICS_CSV_HEADER);
    for (relation_index, relation) in ground_truth.relation.iter().enumerate() {
        let error = &errors[relation_index];
        let translational_error = error.translational_squared.sqrt();
        let squared_translational_error = error.translational_squared;
        let rotational_errors_degree = rad_to_deg(error.rotational_squared.sqrt());
        let squared_rotational_errors_degree = pow2(rotational_errors_degree);
        let expected = relation.expected.unwrap_or_default();
        let translation = expected.translation.unwrap_or_default();
        let rotation = expected.rotation.unwrap_or_default();
        let row = [
            translational_error,
            squared_translational_error,
            rotational_errors_degree,
            squared_rotational_errors_degree,
            translation.x,
            translation.y,
            translation.z,
            rotation.w,
            rotation.x,
            rotation.y,
            rotation.z,
            relation.covered_distance,
        ]
        .iter()
        .map(|&value| format_g(value, 6))
        .collect::<Vec<_>>()
        .join(",");
        out.push_str(&row);
        out.push('\n');
    }
    out
}

/// Writes [`relation_metrics_csv`] to `relation_metrics_filename`.
///
/// RUST PORT NOTE: the C++ ignores write failures; here they are returned.
pub fn write_relation_metrics_to_file(
    errors: &[Error],
    ground_truth: &proto::GroundTruth,
    relation_metrics_filename: &str,
) -> std::io::Result<()> {
    log::info!("Writing relation metrics to '{relation_metrics_filename}'...");
    std::fs::write(
        relation_metrics_filename,
        relation_metrics_csv(errors, ground_truth),
    )
}

/// Looks up the transform at `time`, clamping to the earliest/latest
/// transform outside of the buffer's time range.
pub fn lookup_transform(
    transform_interpolation_buffer: &TransformInterpolationBuffer,
    time: Time,
) -> Rigid3d {
    let earliest_time = transform_interpolation_buffer.earliest_time();
    if transform_interpolation_buffer.has(time) {
        transform_interpolation_buffer.lookup(time)
    } else if time < earliest_time {
        transform_interpolation_buffer.lookup(earliest_time)
    } else {
        transform_interpolation_buffer.lookup(transform_interpolation_buffer.latest_time())
    }
}

/// Computes the error of every relation of `ground_truth` against the
/// (interpolated) node poses of `trajectory`.
pub fn compute_relation_errors(
    trajectory: &Trajectory,
    ground_truth: &proto::GroundTruth,
) -> Vec<Error> {
    let transform_interpolation_buffer = TransformInterpolationBuffer::from_trajectory(trajectory);
    ground_truth
        .relation
        .iter()
        .map(|relation| {
            let pose1 = lookup_transform(
                &transform_interpolation_buffer,
                from_universal(relation.timestamp1),
            );
            let pose2 = lookup_transform(
                &transform_interpolation_buffer,
                from_universal(relation.timestamp2),
            );
            let expected = to_rigid3(&relation.expected.unwrap_or_default());
            compute_error(&pose1, &pose2, &expected)
        })
        .collect()
}

/// Reads the ground truth either from a relations text file with Unix
/// timestamps (see [`read_relations_text_file`]) or from a serialized
/// `GroundTruth` proto. Panics on failure (C++ `CHECK`).
pub fn read_ground_truth(
    relations_filename: &str,
    read_text_file_with_unix_timestamps: bool,
) -> proto::GroundTruth {
    if read_text_file_with_unix_timestamps {
        log::info!("Reading relations from '{relations_filename}'...");
        read_relations_text_file(relations_filename)
    } else {
        log::info!("Reading ground truth from '{relations_filename}'...");
        let bytes = std::fs::read(relations_filename).unwrap_or_else(|e| {
            panic!(
                "Check failed: ground_truth.ParseFromIstream(&ground_truth_stream) \
                 (could not read '{relations_filename}': {e})"
            )
        });
        proto::GroundTruth::decode(bytes.as_slice()).unwrap_or_else(|e| {
            panic!(
                "Check failed: ground_truth.ParseFromIstream(&ground_truth_stream) \
                 ('{relations_filename}': {e})"
            )
        })
    }
}

/// Runs the whole `compute_relations_metrics` tool: reads the pose graph
/// from the pbstream `pose_graph_filename` and the ground truth from
/// `relations_filename`, optionally writes
/// `<pose_graph_filename>.relation_metrics.csv`, and logs and returns the
/// [`statistics_string`].
pub fn run(
    pose_graph_filename: &str,
    relations_filename: &str,
    read_text_file_with_unix_timestamps: bool,
    write_relation_metrics: bool,
) -> String {
    log::info!("Reading pose graph from '{pose_graph_filename}'...");
    let pose_graph = deserialize_pose_graph_from_file(pose_graph_filename);
    let trajectory = &pose_graph.trajectory[0];

    let ground_truth = read_ground_truth(relations_filename, read_text_file_with_unix_timestamps);

    let errors = compute_relation_errors(trajectory, &ground_truth);

    let relation_metrics_filename = format!("{pose_graph_filename}.relation_metrics.csv");
    if write_relation_metrics
        && let Err(e) =
            write_relation_metrics_to_file(&errors, &ground_truth, &relation_metrics_filename)
    {
        log::error!("Could not write '{relation_metrics_filename}': {e}");
    }

    let result = statistics_string(&errors);
    log::info!("Result:\n{result}");
    result
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::mapping::proto::trajectory::Node;
    use crate::transform::rigid_transform::roll_pitch_yaw;
    use crate::transform::transform::to_proto;

    fn translation(x: f64, y: f64, z: f64) -> Rigid3d {
        Rigid3d::from_translation(Vector3::new(x, y, z))
    }

    #[test]
    fn compute_error_known_transforms() {
        let error = compute_error(
            &Rigid3d::identity(),
            &translation(1., 0., 0.),
            &translation(1., 0., 0.),
        );
        assert!(error.translational_squared.abs() < 1e-24);
        assert!(error.rotational_squared.abs() < 1e-24);

        let error = compute_error(
            &translation(1., 1., 0.),
            &translation(4., 5., 0.),
            &translation(3., 4., 0.5),
        );
        assert!((error.translational_squared - 0.25).abs() < 1e-12);
        assert!(error.rotational_squared.abs() < 1e-24);

        let error = compute_error(
            &Rigid3d::identity(),
            &Rigid3d::from_rotation(roll_pitch_yaw(0., 0., 0.3)),
            &Rigid3d::from_rotation(roll_pitch_yaw(0., 0., 0.1)),
        );
        assert!(error.translational_squared.abs() < 1e-24);
        assert!((error.rotational_squared - 0.04).abs() < 1e-12);

        // Asymmetry noted in the C++ TODO: the rotation error rotates the
        // translation, error = (p1^-1 p2) * expected^-1.
        let error = compute_error(
            &Rigid3d::identity(),
            &Rigid3d::new(Vector3::new(1., 0., 0.), roll_pitch_yaw(0., 0., 0.1)),
            &translation(1., 0., 0.),
        );
        let expected_translation: Vector3<f64> = Vector3::new(1., 0., 0.)
            - nalgebra::UnitQuaternion::from_euler_angles(0., 0., 0.1) * Vector3::new(1., 0., 0.);
        assert!((error.translational_squared - expected_translation.norm_squared()).abs() < 1e-12);
        assert!((error.rotational_squared - 0.01).abs() < 1e-12);
    }

    #[test]
    fn mean_and_std_dev() {
        assert_eq!(
            mean_and_std_dev_string(&[1., 2., 3., 4.]),
            "2.50000 +/- 1.29099"
        );
        assert_eq!(mean_and_std_dev_string(&[0., 0.]), "0.00000 +/- 0.00000");
    }

    #[test]
    #[should_panic(expected = "values.size() >= 2")]
    fn mean_and_std_dev_needs_two_values() {
        mean_and_std_dev_string(&[1.]);
    }

    #[test]
    fn statistics() {
        let errors = [
            Error {
                translational_squared: 1.,
                rotational_squared: pow2(1f64.to_radians()),
            },
            Error {
                translational_squared: 9.,
                rotational_squared: pow2(3f64.to_radians()),
            },
        ];
        assert_eq!(
            statistics_string(&errors),
            "Abs translational error 2.00000 +/- 1.41421 m\n\
             Sqr translational error 5.00000 +/- 5.65685 m^2\n\
             Abs rotational error 2.00000 +/- 1.41421 deg\n\
             Sqr rotational error 5.00000 +/- 5.65685 deg^2\n"
        );
    }

    #[test]
    fn csv() {
        let ground_truth = proto::GroundTruth {
            relation: vec![
                proto::Relation {
                    timestamp1: 0,
                    timestamp2: 1,
                    expected: Some(to_proto(&Rigid3d::new(
                        Vector3::new(1., 0.1234567, 1e-7),
                        roll_pitch_yaw(0., 0., 0.),
                    ))),
                    covered_distance: 123.5,
                },
                proto::Relation::default(),
            ],
        };
        let errors = [
            Error {
                translational_squared: 4.,
                rotational_squared: 0.,
            },
            Error {
                translational_squared: 2.,
                rotational_squared: pow2(std::f64::consts::PI),
            },
        ];
        let csv = relation_metrics_csv(&errors, &ground_truth);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[0],
            "translational_error,squared_translational_error,rotational_errors_degree,\
             squared_rotational_errors_degree,expected_translation_x,expected_translation_y,\
             expected_translation_z,expected_rotation_w,expected_rotation_x,expected_rotation_y,\
             expected_rotation_z,covered_distance"
        );
        assert_eq!(lines[1], "2,4,0,0,1,0.123457,1e-07,1,0,0,0,123.5");
        assert_eq!(lines[2], "1.41421,2,180,32400,0,0,0,0,0,0,0,0");
        assert!(csv.ends_with('\n'));
    }

    fn trajectory() -> Trajectory {
        // Node at t = 100 at the origin, at t = 200 at (10, 0, 0) yawed 0.2.
        Trajectory {
            trajectory_id: 0,
            node: vec![
                Node {
                    node_index: 0,
                    timestamp: 100,
                    pose: Some(to_proto(&Rigid3d::identity())),
                },
                Node {
                    node_index: 1,
                    timestamp: 200,
                    pose: Some(to_proto(&Rigid3d::new(
                        Vector3::new(10., 0., 0.),
                        roll_pitch_yaw(0., 0., 0.2),
                    ))),
                },
            ],
            submap: vec![],
        }
    }

    #[test]
    fn lookup_transform_clamps_and_interpolates() {
        let buffer = TransformInterpolationBuffer::from_trajectory(&trajectory());
        let before = lookup_transform(&buffer, from_universal(50));
        assert!(before.translation().norm() < 1e-12);
        let after = lookup_transform(&buffer, from_universal(1000));
        assert!((after.translation() - Vector3::new(10., 0., 0.)).norm() < 1e-12);
        let middle = lookup_transform(&buffer, from_universal(150));
        assert!((middle.translation() - Vector3::new(5., 0., 0.)).norm() < 1e-12);
        assert!((get_angle(&middle) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn relation_errors() {
        let relation = |t1: i64, t2: i64, expected: Rigid3d| proto::Relation {
            timestamp1: t1,
            timestamp2: t2,
            expected: Some(to_proto(&expected)),
            covered_distance: 0.,
        };
        let ground_truth = proto::GroundTruth {
            relation: vec![
                // Exact.
                relation(
                    100,
                    200,
                    Rigid3d::new(Vector3::new(10., 0., 0.), roll_pitch_yaw(0., 0., 0.2)),
                ),
                // Interpolated and clamped (t < earliest) poses; 1 m off.
                relation(0, 150, translation(4., 0., 0.)),
            ],
        };
        let errors = compute_relation_errors(&trajectory(), &ground_truth);
        assert_eq!(errors.len(), 2);
        assert!(errors[0].translational_squared < 1e-20);
        assert!(errors[0].rotational_squared < 1e-20);
        // error = T(5, 0, 0) R(0.1) * T(-4, 0, 0).
        let e: Vector3<f64> = Vector3::new(5., 0., 0.)
            - nalgebra::UnitQuaternion::from_euler_angles(0., 0., 0.1) * Vector3::new(4., 0., 0.);
        assert!((errors[1].translational_squared - e.norm_squared()).abs() < 1e-9);
        assert!((errors[1].rotational_squared - 0.01).abs() < 1e-12);
    }

    #[test]
    fn read_ground_truth_both_modes() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let text_path = dir.join(format!("cartographer_crm_test_{pid}.txt"));
        let proto_path = dir.join(format!("cartographer_crm_test_{pid}.pb"));
        std::fs::write(&text_path, "1 2 3 4 5 0 0 0\n").unwrap();
        let from_text = read_ground_truth(text_path.to_str().unwrap(), true);
        std::fs::write(&proto_path, from_text.encode_to_vec()).unwrap();
        let from_proto = read_ground_truth(proto_path.to_str().unwrap(), false);
        std::fs::remove_file(&text_path).unwrap();
        std::fs::remove_file(&proto_path).unwrap();
        assert_eq!(from_text.relation.len(), 1);
        assert_eq!(from_text, from_proto);
    }
}
