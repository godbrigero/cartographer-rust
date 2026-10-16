//! Port of `cartographer/io/pbstream_main.cc`: the swiss army knife for
//! pbstreams.
//!
//! Usage (like C++):
//!
//! ```text
//! cartographer_pbstream info <pbstream_filename> [--all_debug_strings]
//! cartographer_pbstream migrate <input_filename> <output_filename> [--include_unfinished_submaps=true]
//! ```
//!
//! RUST PORT NOTE: gflags removes flags from `argv` before the subcommand is
//! looked up; here the subcommand is the first argument not starting with
//! `-`, and the subtools parse their own flags from the full argument list.
//! Log output goes to stderr (C++ `FLAGS_logtostderr = true`).

use std::process::ExitCode;

use cartographer::io::internal::pbstream_info::pbstream_info;
use cartographer::io::internal::pbstream_migrate::pbstream_migrate;

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

const USAGE_MESSAGE: &str = "Swiss Army knife for pbstreams.\n\n\
                             Currently supported subcommands are:\n\
                             \tinfo    - Prints summary of pbstream.\n\
                             \tmigrate - Migrates pbstream to the new submap format.";

fn show_usage(program: &str) {
    eprintln!("{program}: {USAGE_MESSAGE}");
}

fn exit_code(code: i32) -> ExitCode {
    if code == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
    let args: Vec<String> = std::env::args().collect();
    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("cartographer_pbstream");
    let subcommand = args.iter().skip(1).find(|arg| !arg.starts_with('-'));
    match subcommand.map(String::as_str) {
        None => {
            show_usage(program);
            ExitCode::FAILURE
        }
        Some("info") => exit_code(pbstream_info(&args)),
        Some("migrate") => exit_code(pbstream_migrate(&args)),
        Some(unknown) => {
            log::info!("Unknown subtool: \"{unknown}");
            show_usage(program);
            ExitCode::FAILURE
        }
    }
}
