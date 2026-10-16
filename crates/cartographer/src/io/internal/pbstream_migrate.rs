//! Port of `cartographer/io/internal/pbstream_migrate` (.h, .cc): the
//! `migrate` subtool of the `cartographer_pbstream` swiss army knife.
//!
//! RUST PORT NOTE: the gflags flag `--include_unfinished_submaps` (default
//! `true`) is parsed from the arguments by [`pbstream_migrate`], accepting
//! the gflags spellings `--flag`, `--noflag`, `--flag=true|false` (also with
//! a single dash).

use crate::io::proto_stream::{ProtoStreamReader, ProtoStreamWriter};
use crate::io::proto_stream_interface::ProtoStreamWriterInterface;
use crate::io::serialization_format_migration::migrate_stream_version1_to_version2;

/// Parses a gflags-style boolean flag `name`. Returns `None` if `arg` is not
/// that flag.
fn parse_bool_flag(arg: &str, name: &str) -> Option<bool> {
    let flag = arg.strip_prefix("--").or_else(|| arg.strip_prefix('-'))?;
    if flag == name {
        return Some(true);
    }
    if flag.strip_prefix("no") == Some(name) {
        return Some(false);
    }
    match flag.strip_prefix(name)?.strip_prefix('=')? {
        "true" | "1" | "yes" | "t" | "y" => Some(true),
        "false" | "0" | "no" | "f" | "n" => Some(false),
        _ => None,
    }
}

/// Migrates the pbstream `input_filename` (format version 1) to
/// `output_filename` (format version 2). Returns whether the output could be
/// written.
pub fn run(input_filename: &str, output_filename: &str, include_unfinished_submaps: bool) -> bool {
    let mut input = ProtoStreamReader::new(input_filename);
    let mut output = ProtoStreamWriter::new(output_filename);
    log::info!(
        "Migrating serialization format 1 in \"{input_filename}\" to serialization format 2 in \
         \"{output_filename}\""
    );
    migrate_stream_version1_to_version2(&mut input, &mut output, include_unfinished_submaps);
    output.close()
}

/// The `migrate` subtool. `args` are the command line arguments including
/// the program name and the subcommand (`<program> migrate <input_filename>
/// <output_filename> [--include_unfinished_submaps=true]`). Returns the
/// process exit code.
pub fn pbstream_migrate(args: &[String]) -> i32 {
    let mut include_unfinished_submaps = true;
    let mut positional = Vec::new();
    for arg in args {
        match parse_bool_flag(arg, "include_unfinished_submaps") {
            Some(value) => include_unfinished_submaps = value,
            None => positional.push(arg.as_str()),
        }
    }
    let program = positional
        .first()
        .copied()
        .unwrap_or("cartographer_pbstream");
    let subcommand = positional.get(1).copied().unwrap_or("migrate");
    if positional.len() < 4 {
        eprintln!(
            "\n\nTool for migrating files that use submaps without histograms to the new submap \
             format, which includes a histogram. You can set --include_unfinished_submaps to \
             false if you want to exclude unfinished submaps in the output.\n\
             Usage: {program} {subcommand} <input_filename> <output_filename> [flags]\n\n  \
             --include_unfinished_submaps (Whether to include unfinished submaps in the \
             output.) type: bool default: true"
        );
        return 1;
    }
    let output_filename = positional[3];
    assert!(
        run(positional[2], output_filename, include_unfinished_submaps),
        "Could not write migrated pbstream file to: {output_filename}"
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gflags_booleans() {
        let name = "include_unfinished_submaps";
        assert_eq!(
            parse_bool_flag("--include_unfinished_submaps", name),
            Some(true)
        );
        assert_eq!(
            parse_bool_flag("-include_unfinished_submaps", name),
            Some(true)
        );
        assert_eq!(
            parse_bool_flag("--noinclude_unfinished_submaps", name),
            Some(false)
        );
        assert_eq!(
            parse_bool_flag("--include_unfinished_submaps=false", name),
            Some(false)
        );
        assert_eq!(
            parse_bool_flag("--include_unfinished_submaps=true", name),
            Some(true)
        );
        assert_eq!(parse_bool_flag("input.pbstream", name), None);
        assert_eq!(parse_bool_flag("--other", name), None);
    }

    #[test]
    fn usage_without_filenames() {
        let args: Vec<String> = ["prog", "migrate", "in.pbstream"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pbstream_migrate(&args), 1);
    }
}
