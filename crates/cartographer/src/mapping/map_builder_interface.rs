//! Port of `cartographer/mapping/map_builder_interface` (.h, .cc).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::io::proto_stream_interface::{ProtoStreamReaderInterface, ProtoStreamWriterInterface};
use crate::mapping::id::SubmapId;
use crate::mapping::pose_graph::create_pose_graph_options;
use crate::mapping::pose_graph_interface::PoseGraphInterface;
use crate::mapping::proto;
use crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
pub use crate::mapping::trajectory_builder_interface::{LocalSlamResultCallback, SensorId};

/// Reads `MapBuilderOptions` from `parameter_dictionary` (C++
/// `CreateMapBuilderOptions`). Exactly one of `use_trajectory_builder_2d`
/// and `use_trajectory_builder_3d` must be set.
pub fn create_map_builder_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::MapBuilderOptions {
    let mut options = proto::MapBuilderOptions {
        use_trajectory_builder_2d: parameter_dictionary.get_bool("use_trajectory_builder_2d"),
        ..Default::default()
    };
    options.use_trajectory_builder_3d = parameter_dictionary.get_bool("use_trajectory_builder_3d");
    options.num_background_threads =
        parameter_dictionary.get_non_negative_int("num_background_threads");
    options.collate_by_trajectory = parameter_dictionary.get_bool("collate_by_trajectory");
    options.pose_graph_options = Some(create_pose_graph_options(
        &parameter_dictionary.get_dictionary("pose_graph"),
    ));
    assert_ne!(
        options.use_trajectory_builder_2d, options.use_trajectory_builder_3d,
        "Check failed: options.use_trajectory_builder_2d() != options.use_trajectory_builder_3d()"
    );
    options
}

/// This interface is used for both library and RPC implementations.
/// Implementations wire up the complete SLAM stack.
///
/// RUST PORT NOTE: the C++ nested aliases `MapBuilderInterface::SensorId`
/// and `MapBuilderInterface::LocalSlamResultCallback` are re-exported from
/// this module. Ownership-driven signature changes are documented per method.
pub trait MapBuilderInterface {
    /// Creates a new trajectory builder and returns its index.
    ///
    /// RUST PORT NOTE: the possibly empty C++ `std::function` is an
    /// `Option<LocalSlamResultCallback>`.
    fn add_trajectory_builder(
        &mut self,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &proto::TrajectoryBuilderOptions,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> i32;

    /// Creates a new trajectory and returns its index. Querying the
    /// trajectory builder for it will return `None`.
    fn add_trajectory_for_deserialization(
        &mut self,
        options_with_sensor_ids_proto: &proto::TrajectoryBuilderOptionsWithSensorIds,
    ) -> i32;

    /// Returns the `TrajectoryBuilderInterface` corresponding to the
    /// specified `trajectory_id` or `None` if the trajectory has no
    /// corresponding builder.
    ///
    /// RUST PORT NOTE: the C++ method is `const` but returns a mutable
    /// pointer; here it takes `&mut self` to hand out `&mut`.
    fn get_trajectory_builder(
        &mut self,
        trajectory_id: i32,
    ) -> Option<&mut dyn TrajectoryBuilderInterface>;

    /// Marks the TrajectoryBuilder corresponding to `trajectory_id` as
    /// finished, i.e. no further sensor data is expected.
    fn finish_trajectory(&mut self, trajectory_id: i32);

    /// Fills the `SubmapQuery::Response` corresponding to `submap_id`.
    /// Returns an error string on failure, or an empty string on success.
    fn submap_to_proto(
        &mut self,
        submap_id: &SubmapId,
        response: &mut proto::submap_query::Response,
    ) -> String;

    /// Serializes the current state to a proto stream. If
    /// `include_unfinished_submaps` is set to true, unfinished submaps, i.e.
    /// submaps that have not yet received all rangefinder data insertions,
    /// will be included in the serialized state.
    fn serialize_state(
        &mut self,
        include_unfinished_submaps: bool,
        writer: &mut dyn ProtoStreamWriterInterface,
    );

    /// Serializes the current state to a proto stream file on the host
    /// system. If `include_unfinished_submaps` is set to true, unfinished
    /// submaps, i.e. submaps that have not yet received all rangefinder data
    /// insertions, will be included in the serialized state.
    /// Returns true if the file was successfully written.
    fn serialize_state_to_file(&mut self, include_unfinished_submaps: bool, filename: &str)
    -> bool;

    /// Loads the SLAM state from a proto stream. Returns the remapping of new
    /// trajectory_ids (trajectory id in proto -> trajectory id).
    fn load_state(
        &mut self,
        reader: &mut dyn ProtoStreamReaderInterface,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32>;

    /// Loads the SLAM state from a pbstream file. Returns the remapping of
    /// new trajectory_ids (trajectory id in proto -> trajectory id).
    fn load_state_from_file(
        &mut self,
        filename: &str,
        load_frozen_state: bool,
    ) -> BTreeMap<i32, i32>;

    /// The number of trajectory builders (including deserialized
    /// trajectories without a builder).
    fn num_trajectory_builders(&self) -> i32;

    /// The pose graph.
    ///
    /// RUST PORT NOTE: the C++ returns a non-owning `PoseGraphInterface*`;
    /// here a shared handle, so that callers can query the (internally
    /// synchronized) pose graph while still feeding data through
    /// `&mut self`.
    fn pose_graph(&self) -> Arc<dyn PoseGraphInterface>;

    /// The options of all trajectories, indexed by trajectory id.
    fn get_all_trajectory_builder_options(&self)
    -> &[proto::TrajectoryBuilderOptionsWithSensorIds];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;

    #[test]
    fn create_map_builder_options_reads_all_fields() {
        let options = create_map_builder_options(&resolve_lua_parameters(
            r#"
            include "map_builder.lua"
            MAP_BUILDER.use_trajectory_builder_3d = true
            MAP_BUILDER.num_background_threads = 2
            MAP_BUILDER.collate_by_trajectory = true
            return MAP_BUILDER"#,
        ));
        assert!(!options.use_trajectory_builder_2d);
        assert!(options.use_trajectory_builder_3d);
        assert_eq!(options.num_background_threads, 2);
        assert!(options.collate_by_trajectory);
        assert_eq!(
            options.pose_graph_options.unwrap().optimize_every_n_nodes,
            90
        );
    }

    #[test]
    #[should_panic(expected = "use_trajectory_builder_2d() != options.use_trajectory_builder_3d()")]
    fn create_map_builder_options_requires_exactly_one_dimension() {
        let _ = create_map_builder_options(&resolve_lua_parameters(
            r#"include "map_builder.lua" return MAP_BUILDER"#,
        ));
    }
}
