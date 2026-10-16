//! Port of `cartographer/mapping/map_builder` (.h, .cc).
//!
//! `CreateMapBuilderOptions` is defined in `map_builder_interface.cc` in C++
//! (and declared in `map_builder_interface.h`, which `map_builder.h`
//! includes); it is re-exported here for convenience.

pub use crate::mapping::map_builder_interface::create_map_builder_options;

// MapBuilder struct added by the map builder port
