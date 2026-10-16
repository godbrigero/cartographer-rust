//! Port of `cartographer/io/internal/mapping_state_serialization` (.h, .cc).
//!
//! Only the format version constants are ported so far; `WritePbStream`
//! needs `mapping::PoseGraph` and is ported together with it.

/// The current serialization format version.
pub const MAPPING_STATE_SERIALIZATION_FORMAT_VERSION: u32 = 2;
/// The format version of streams whose 3D submaps lack rotational scan
/// matcher histograms.
pub const FORMAT_VERSION_WITHOUT_SUBMAP_HISTOGRAMS: u32 = 1;
