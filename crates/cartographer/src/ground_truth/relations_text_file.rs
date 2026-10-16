//! Port of `cartographer/ground_truth/relations_text_file` (.h, .cc).

use nalgebra::Vector3;

use super::proto;
use crate::common::time::{
    TICKS_PER_SECOND, Time, UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS, from_seconds,
    from_universal, to_universal,
};
use crate::transform::rigid_transform::{Rigid3d, roll_pitch_yaw};
use crate::transform::transform::to_proto;

fn unix_to_common_time(unix_time: f64) -> Time {
    // C++ uses a local `kUtsTicksPerSecond = 10000000`.
    from_universal(UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * TICKS_PER_SECOND)
        + from_seconds(unix_time)
}

/// Parses the contents of a relations text file (see
/// [`read_relations_text_file`]).
///
/// Values are whitespace-separated; 8 values form one relation. Like the C++
/// `std::ifstream >> double` loop, a trailing incomplete record at the end of
/// the input is silently dropped, while a token that is not a number panics
/// (C++ `CHECK(relations_stream.eof())`).
///
/// RUST PORT NOTE: tokens are parsed with `str::parse::<f64>`, which also
/// accepts e.g. `inf`/`nan` (the C++ stream extraction would fail on those),
/// and a token like `1.5abc` is rejected as a whole instead of being split.
pub fn parse_relations_text(contents: &str) -> proto::GroundTruth {
    let mut ground_truth = proto::GroundTruth::default();
    let mut values = [0.; 8];
    let mut num_values = 0;
    for token in contents.split_whitespace() {
        let value: f64 = token.parse().unwrap_or_else(|_| {
            panic!("Check failed: relations_stream.eof() (could not parse '{token}' as a number)")
        });
        values[num_values] = value;
        num_values += 1;
        if num_values < values.len() {
            continue;
        }
        num_values = 0;
        let [unix_time_1, unix_time_2, x, y, z, roll, pitch, yaw] = values;
        let common_time_1 = unix_to_common_time(unix_time_1);
        let common_time_2 = unix_to_common_time(unix_time_2);
        let expected = Rigid3d::new(Vector3::new(x, y, z), roll_pitch_yaw(roll, pitch, yaw));
        ground_truth.relation.push(proto::Relation {
            timestamp1: to_universal(common_time_1),
            timestamp2: to_universal(common_time_2),
            expected: Some(to_proto(&expected)),
            ..Default::default()
        });
    }
    ground_truth
}

/// Reads a text file and converts it to a GroundTruth proto. Each line
/// contains:
/// time1 time2 x y z roll pitch yaw
/// using Unix epoch timestamps.
///
/// This is the format used in the relations files provided for:
/// R. Kuemmerle, B. Steder, C. Dornhege, M. Ruhnke, G. Grisetti, C. Stachniss,
/// and A. Kleiner, "On measuring the accuracy of SLAM algorithms," Autonomous
/// Robots, vol. 27, no. 4, pp. 387–407, 2009.
///
/// Panics if the file cannot be read or contains non-numeric data (C++
/// `CHECK`).
pub fn read_relations_text_file(relations_filename: &str) -> proto::GroundTruth {
    let contents = std::fs::read_to_string(relations_filename).unwrap_or_else(|e| {
        panic!("Check failed: relations_stream.eof() (could not read '{relations_filename}': {e})")
    });
    parse_relations_text(&contents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use crate::transform::transform::to_rigid3;

    const UNIX_EPOCH_UNIVERSAL: i64 = 62_135_596_800 * 10_000_000;

    #[test]
    fn unix_to_common_time_offsets_epoch() {
        assert_eq!(to_universal(unix_to_common_time(0.)), UNIX_EPOCH_UNIVERSAL);
        assert_eq!(
            to_universal(unix_to_common_time(1.5)),
            UNIX_EPOCH_UNIVERSAL + 15_000_000
        );
    }

    #[test]
    fn parses_relations() {
        let text = "1000.5 1001.25 1 2 3 0 0 1.5707963267948966\n\
                    \t2000 2000.0000001 -1 0.5 0 0.1 -0.2 0.3\n";
        let ground_truth = parse_relations_text(text);
        assert_eq!(ground_truth.relation.len(), 2);

        let r0 = &ground_truth.relation[0];
        assert_eq!(r0.timestamp1, UNIX_EPOCH_UNIVERSAL + 10_005_000_000);
        assert_eq!(r0.timestamp2, UNIX_EPOCH_UNIVERSAL + 10_012_500_000);
        assert_eq!(r0.covered_distance, 0.);
        let expected = Rigid3d::new(
            Vector3::new(1., 2., 3.),
            roll_pitch_yaw(0., 0., std::f64::consts::FRAC_PI_2),
        );
        assert_is_nearly(&to_rigid3(r0.expected.as_ref().unwrap()), &expected, 1e-12);
        // Yaw of 90 degrees maps x onto y.
        let p = to_rigid3(r0.expected.as_ref().unwrap()) * Vector3::new(1., 0., 0.);
        assert!((p - Vector3::new(1., 3., 3.)).norm() < 1e-12);

        let r1 = &ground_truth.relation[1];
        assert_eq!(r1.timestamp1, UNIX_EPOCH_UNIVERSAL + 20_000_000_000);
        // 1e-7 s = 1 tick; truncation toward zero of the double product.
        assert_eq!(
            r1.timestamp2,
            UNIX_EPOCH_UNIVERSAL + (2_000.000_000_1_f64 * 1e7) as i64
        );
        assert_is_nearly(
            &to_rigid3(r1.expected.as_ref().unwrap()),
            &Rigid3d::new(Vector3::new(-1., 0.5, 0.), roll_pitch_yaw(0.1, -0.2, 0.3)),
            1e-12,
        );
    }

    #[test]
    fn trailing_partial_record_is_dropped() {
        let ground_truth = parse_relations_text("1 2 0 0 0 0 0 0\n3 4 5");
        assert_eq!(ground_truth.relation.len(), 1);
        assert!(parse_relations_text("").relation.is_empty());
    }

    #[test]
    #[should_panic(expected = "relations_stream.eof()")]
    fn garbage_panics() {
        parse_relations_text("1 2 0 0 0 0 0 0\nfoo 1 2 3 4 5 6 7\n");
    }

    #[test]
    #[should_panic(expected = "relations_stream.eof()")]
    fn missing_file_panics() {
        read_relations_text_file("/nonexistent/cartographer/relations.txt");
    }

    #[test]
    fn reads_file() {
        let path = std::env::temp_dir().join(format!(
            "cartographer_relations_text_file_test_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "10 20 1 0 0 0 0 0\n").unwrap();
        let ground_truth = read_relations_text_file(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        assert_eq!(ground_truth.relation.len(), 1);
        assert_eq!(
            ground_truth.relation[0].timestamp2 - ground_truth.relation[0].timestamp1,
            100_000_000
        );
    }
}
