//! Port of `cartographer/ground_truth/compute_relations_metrics_main.cc`.
//!
//! The computation lives in
//! `cartographer::ground_truth::compute_relations_metrics`.

use std::process::ExitCode;

use cartographer::ground_truth::compute_relations_metrics::run;
use clap::{ArgAction, CommandFactory, Parser};

/// This program computes the relation based metric described in:
/// R. Kuemmerle, B. Steder, C. Dornhege, M. Ruhnke, G. Grisetti,
/// C. Stachniss, and A. Kleiner, "On measuring the accuracy of SLAM
/// algorithms," Autonomous Robots, vol. 27, no. 4, pp. 387–407, 2009.
#[derive(Parser, Debug)]
#[command(name = "cartographer_compute_relations_metrics", verbatim_doc_comment)]
struct Flags {
    /// Proto stream file containing the pose graph used to assess quality.
    #[arg(long = "pose_graph_filename", default_value = "")]
    pose_graph_filename: String,

    /// Relations file containing the ground truth.
    #[arg(long = "relations_filename", default_value = "")]
    relations_filename: String,

    /// Enable support for the relations text files as in the paper. Default
    /// is to read from a GroundTruth proto file.
    // gflags accepts both `--flag` and `--flag=true|false`.
    #[arg(
        long = "read_text_file_with_unix_timestamps",
        num_args = 0..=1,
        default_value_t = false,
        default_missing_value = "true",
        action = ArgAction::Set
    )]
    read_text_file_with_unix_timestamps: bool,

    /// Enable exporting relation metrics as comma-separated values to
    /// [pose_graph_filename].relation_metrics.csv
    #[arg(
        long = "write_relation_metrics",
        num_args = 0..=1,
        default_value_t = false,
        default_missing_value = "true",
        action = ArgAction::Set
    )]
    write_relation_metrics: bool,
}

/// Minimal logger printing to stderr (C++ `FLAGS_logtostderr = true`).
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let level = record.level().as_str().chars().next().unwrap_or('I');
            eprintln!("{level} {}] {}", record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

fn main() -> ExitCode {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    let flags = Flags::parse();

    if flags.pose_graph_filename.is_empty() || flags.relations_filename.is_empty() {
        let _ = Flags::command().print_help();
        return ExitCode::FAILURE;
    }

    run(
        &flags.pose_graph_filename,
        &flags.relations_filename,
        flags.read_text_file_with_unix_timestamps,
        flags.write_relation_metrics,
    );
    ExitCode::SUCCESS
}
