//! Port of `cartographer/common/print_configuration_main.cc`.
//!
//! Resolves and compiles a Lua configuration and prints it to stdout.

use std::process::ExitCode;

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::{FileResolver, LuaParameterDictionary};
use clap::{CommandFactory, Parser};

/// Resolves and compiles a Lua configuration and prints it to stdout.
/// The output can be restricted to a subdictionary using the optional
/// '--subdictionary' parameter, which can be given in Lua syntax.
#[derive(Parser, Debug)]
#[command(name = "cartographer_print_configuration", verbatim_doc_comment)]
struct Flags {
    /// Comma separated list of directories in which configuration files are
    /// searched, the last is always the Cartographer installation to allow
    /// including files from there.
    #[arg(
        long = "configuration_directories",
        alias = "configuration-directories",
        default_value = ""
    )]
    configuration_directories: String,

    /// Basename, i.e. not containing any directory prefix, of the
    /// configuration file.
    #[arg(
        long = "configuration_basename",
        alias = "configuration-basename",
        default_value = ""
    )]
    configuration_basename: String,

    /// Only print a subdictionary referenced by its Lua ID, e.g.:
    /// '--subdictionary trajectory_builder.trajectory_builder_3d'
    #[arg(long = "subdictionary", default_value = "")]
    subdictionary: String,
}

fn load_lua_dictionary(
    configuration_directories: &[String],
    configuration_basename: &str,
) -> LuaParameterDictionary {
    let file_resolver = ConfigurationFileResolver::new(configuration_directories.iter().cloned());
    let code = file_resolver.get_file_content_or_die(configuration_basename);
    // We use no reference count because we just want to print the
    // configuration.
    LuaParameterDictionary::non_reference_counted(&code, Box::new(file_resolver))
}

fn print_subdictionary_by_id(lua_dictionary: &LuaParameterDictionary, subdictionary_id: &str) {
    let subdictionary_keys: Vec<&str> = subdictionary_id
        .split('.')
        .filter(|key| !key.is_empty())
        .collect();
    assert!(
        !subdictionary_keys.is_empty(),
        "Failed to parse 'subdictionary_id'."
    );
    let mut dictionary = lua_dictionary.get_dictionary(subdictionary_keys[0]);
    for key in &subdictionary_keys[1..] {
        dictionary = dictionary.get_dictionary(key);
    }
    println!("{subdictionary_id} = {dictionary}");
}

fn main() -> ExitCode {
    let flags = Flags::parse();

    if flags.configuration_directories.is_empty() || flags.configuration_basename.is_empty() {
        let _ = Flags::command().print_help();
        return ExitCode::FAILURE;
    }

    let configuration_directories: Vec<String> = flags
        .configuration_directories
        .split(',')
        .filter(|d| !d.is_empty())
        .map(str::to_string)
        .collect();

    let lua_dictionary =
        load_lua_dictionary(&configuration_directories, &flags.configuration_basename);

    if flags.subdictionary.is_empty() {
        println!("return {lua_dictionary}");
        return ExitCode::SUCCESS;
    }
    print_subdictionary_by_id(&lua_dictionary, &flags.subdictionary);
    ExitCode::SUCCESS
}
