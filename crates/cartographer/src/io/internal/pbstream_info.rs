//! Port of `cartographer/io/internal/pbstream_info` (.h, .cc): the `info`
//! subtool of the `cartographer_pbstream` swiss army knife.
//!
//! RUST PORT NOTE: the gflags flag `--all_debug_strings` is passed
//! explicitly (or parsed from the arguments by [`pbstream_info`]);
//! protobuf `DebugString()` output is replaced by Rust's pretty `Debug`
//! formatting.

use std::collections::BTreeMap;

use crate::io::proto_stream::ProtoStreamReader;
use crate::io::proto_stream_deserializer::ProtoStreamDeserializer;
use crate::mapping::proto::{SerializedData, serialized_data::Data};

fn data_case_name(data: &SerializedData) -> Option<&'static str> {
    match &data.data {
        Some(Data::Submap(_)) => Some("submap"),
        Some(Data::Node(_)) => Some("node"),
        Some(Data::TrajectoryData(_)) => Some("trajectory_data"),
        Some(Data::ImuData(_)) => Some("imu_data"),
        Some(Data::OdometryData(_)) => Some("odometry_data"),
        Some(Data::FixedFramePoseData(_)) => Some("fixed_frame_pose_data"),
        Some(Data::LandmarkData(_)) => Some("landmark_data"),
        _ => None,
    }
}

/// Reads `pbstream_filename`, logs a summary of its contents and returns the
/// counts of the `SerializedData` packages by kind (in the alphabetical order
/// in which they are logged).
pub fn run(pbstream_filename: &str, all_debug_strings: bool) -> BTreeMap<String, i32> {
    log::info!("Reading pbstream file from '{pbstream_filename}'...");
    let mut reader = ProtoStreamReader::new(pbstream_filename);
    let mut deserializer = ProtoStreamDeserializer::new(&mut reader);
    let header = *deserializer.header();
    log::info!("Header: {header:#?}");
    for trajectory_options in &deserializer
        .all_trajectory_builder_options()
        .options_with_sensor_ids
    {
        log::info!("Trajectory options: {trajectory_options:#?}");
    }
    let pose_graph = deserializer.pose_graph().clone();
    for trajectory in &pose_graph.trajectory {
        log::info!(
            "Trajectory id: {} has #nodes {} has #submaps {}",
            trajectory.trajectory_id,
            trajectory.node.len(),
            trajectory.submap.len()
        );
    }
    if all_debug_strings {
        log::info!("Pose graph: {pose_graph:#?}");
    }

    // Initialize so zero counts of these are also reported.
    let mut data_counts: BTreeMap<String, i32> = [
        "submap_2d",
        "submap_2d_grid",
        "submap_3d",
        "submap_3d_high_resolution_hybrid_grid",
    ]
    .into_iter()
    .map(|name| (name.to_string(), 0))
    .collect();
    let mut proto = SerializedData::default();
    while deserializer.read_next_serialized_data(&mut proto) {
        if all_debug_strings {
            log::info!("Serialized data: {proto:#?}");
        }
        let Some(data_name) = data_case_name(&proto) else {
            // RUST PORT NOTE: C++ dereferences the end iterator here (UB);
            // we skip the message.
            log::warn!("Skipping unknown message type in stream: SerializedData");
            continue;
        };
        *data_counts.entry(data_name.to_string()).or_insert(0) += 1;
        if let Some(Data::Submap(submap)) = &proto.data {
            if let Some(submap_2d) = &submap.submap_2d {
                *data_counts.get_mut("submap_2d").unwrap() += 1;
                if submap_2d.grid.is_some() {
                    *data_counts.get_mut("submap_2d_grid").unwrap() += 1;
                }
            }
            if let Some(submap_3d) = &submap.submap_3d {
                *data_counts.get_mut("submap_3d").unwrap() += 1;
                if submap_3d.high_resolution_hybrid_grid.is_some() {
                    *data_counts
                        .get_mut("submap_3d_high_resolution_hybrid_grid")
                        .unwrap() += 1;
                }
            }
        }
    }

    for (name, count) in &data_counts {
        log::info!("SerializedData package contains #{name}: {count}");
    }
    data_counts
}

/// The `info` subtool. `args` are the command line arguments including the
/// program name and the subcommand (`<program> info <pbstream_filename>
/// [--all_debug_strings]`). Returns the process exit code.
pub fn pbstream_info(args: &[String]) -> i32 {
    let mut all_debug_strings = false;
    let mut positional = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--all_debug_strings" | "-all_debug_strings" | "--all_debug_strings=true" => {
                all_debug_strings = true
            }
            "--noall_debug_strings" | "--all_debug_strings=false" => all_debug_strings = false,
            _ => positional.push(arg.as_str()),
        }
    }
    if positional.len() < 3 {
        let program = positional
            .first()
            .copied()
            .unwrap_or("cartographer_pbstream");
        let subcommand = positional.get(1).copied().unwrap_or("info");
        eprintln!(
            "\n\nReads a pbstream file and summarizes its contents.\n\n\
             Usage: {program} {subcommand} <pbstream_filename> [flags]\n\n  \
             --all_debug_strings (Print debug strings of all serialized data.) \
             type: bool default: false"
        );
        return 1;
    }
    run(positional[2], all_debug_strings);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::proto_stream::ProtoStreamWriter;
    use crate::io::proto_stream_interface::ProtoStreamWriterInterface;
    use crate::mapping::proto;

    #[test]
    fn counts_serialized_data() {
        let path = std::env::temp_dir().join(format!(
            "cartographer_{}_pbstream_info.pbstream",
            std::process::id()
        ));
        let filename = path.to_str().unwrap();
        {
            let mut writer = ProtoStreamWriter::new(filename);
            writer.write_proto(&proto::SerializationHeader { format_version: 2 });
            writer.write_proto(&SerializedData {
                data: Some(Data::PoseGraph(proto::PoseGraph {
                    trajectory: vec![proto::Trajectory::default()],
                    ..Default::default()
                })),
            });
            writer.write_proto(&SerializedData {
                data: Some(Data::AllTrajectoryBuilderOptions(
                    proto::AllTrajectoryBuilderOptions {
                        options_with_sensor_ids: vec![Default::default()],
                    },
                )),
            });
            for submap in [
                proto::Submap {
                    submap_2d: Some(proto::Submap2D {
                        grid: Some(Default::default()),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                proto::Submap {
                    submap_2d: Some(proto::Submap2D::default()),
                    ..Default::default()
                },
                proto::Submap {
                    submap_3d: Some(proto::Submap3D::default()),
                    ..Default::default()
                },
            ] {
                writer.write_proto(&SerializedData {
                    data: Some(Data::Submap(submap)),
                });
            }
            writer.write_proto(&SerializedData {
                data: Some(Data::Node(proto::Node::default())),
            });
            // Unknown (unset) data is skipped.
            writer.write_proto(&SerializedData::default());
            writer.write_proto(&SerializedData {
                data: Some(Data::ImuData(proto::ImuData::default())),
            });
            assert!(writer.close());
        }
        let counts = run(filename, true);
        let expected: BTreeMap<String, i32> = [
            ("imu_data", 1),
            ("node", 1),
            ("submap", 3),
            ("submap_2d", 2),
            ("submap_2d_grid", 1),
            ("submap_3d", 1),
            ("submap_3d_high_resolution_hybrid_grid", 0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(counts, expected);

        let args: Vec<String> = ["prog", "info", filename, "--all_debug_strings"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pbstream_info(&args), 0);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn usage_without_filename() {
        let args: Vec<String> = ["prog", "info"].iter().map(|s| s.to_string()).collect();
        assert_eq!(pbstream_info(&args), 1);
    }
}
