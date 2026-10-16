//! Port of `cartographer/cloud/map_builder_server_main.cc`
//! (`cartographer_grpc_server`): offers a MapBuilder service via a gRPC
//! interface.

use std::process::ExitCode;

use cartographer_cloud::map_builder_server_options::load_map_builder_server_options;
use cartographer_cloud::proto;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "This program offers a MapBuilder service via a gRPC interface.")]
struct Flags {
    /// First directory in which configuration files are searched, second is
    /// always the Cartographer installation to allow including files from
    /// there.
    #[arg(long = "configuration_directory", default_value = "")]
    configuration_directory: String,
    /// Basename, i.e. not containing any directory prefix, of the
    /// configuration file.
    #[arg(long = "configuration_basename", default_value = "")]
    configuration_basename: String,
}

/// Creates the concrete map builder (C++ `mapping::CreateMapBuilder`).
fn create_map_builder(
    options: &cartographer::mapping::proto::MapBuilderOptions,
) -> Result<cartographer_cloud::internal::map_builder_context_interface::BoxedMapBuilder, String> {
    Ok(Box::new(cartographer::mapping::MapBuilder::new(options)))
}

fn run(configuration_directory: &str, configuration_basename: &str) -> Result<(), String> {
    #[cfg(feature = "prometheus")]
    let _exposer = {
        use cartographer_cloud::metrics::prometheus::{Exposer, FamilyFactory};
        let mut registry = FamilyFactory::new();
        cartographer::metrics::register::register_all_metrics(&mut registry);
        cartographer_cloud::register_map_builder_server_metrics(&mut registry);
        let exposer = Exposer::new("0.0.0.0:9100", registry.registry())
            .map_err(|e| format!("Failed to start the metrics exposer: {e}"))?;
        log::info!("Exposing metrics at http://localhost:9100/metrics");
        exposer
    };

    let map_builder_server_options: proto::MapBuilderServerOptions =
        load_map_builder_server_options(configuration_directory, configuration_basename);
    let map_builder = create_map_builder(
        &map_builder_server_options
            .map_builder_options
            .clone()
            .unwrap_or_default(),
    )?;
    let map_builder_server =
        cartographer_cloud::create_map_builder_server(&map_builder_server_options, map_builder);
    map_builder_server.start();
    map_builder_server.wait_for_shutdown();
    Ok(())
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
    if flags.configuration_directory.is_empty() || flags.configuration_basename.is_empty() {
        use clap::CommandFactory;
        let _ = Flags::command().print_help();
        return ExitCode::FAILURE;
    }
    match run(
        &flags.configuration_directory,
        &flags.configuration_basename,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
