//! Port of `cartographer/common/config.h.cmake`.
//!
//! RUST PORT NOTE: the C++ values are filled in by CMake. Here the
//! configuration files shipped with this crate (copied byte-for-byte from the
//! C++ `configuration_files/`) live in `<crate>/configuration_files`, which
//! plays the role of both the install directory and the source directory.

/// Directory searched last by
/// [`ConfigurationFileResolver`](super::configuration_file_resolver::ConfigurationFileResolver).
pub const CONFIGURATION_FILES_DIRECTORY: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/configuration_files");

/// Source directory of the crate; `SOURCE_DIRECTORY + "/configuration_files"`
/// is [`CONFIGURATION_FILES_DIRECTORY`] like in the C++ source tree.
pub const SOURCE_DIRECTORY: &str = env!("CARGO_MANIFEST_DIR");
