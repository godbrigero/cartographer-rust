//! Port of `cartographer/io/internal/pbstream_migrate` (.h, .cc).
//!
//! Not ported yet: the `migrate` subtool only calls
//! `io::MigrateStreamVersion1ToVersion2`, which needs `mapping::PoseGraph3D`
//! and `WritePbStream` (`io::internal::mapping_state_serialization`). It is
//! ported together with them.
