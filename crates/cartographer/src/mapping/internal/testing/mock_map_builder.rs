//! Port of `cartographer/mapping/internal/testing/mock_map_builder` (.h).
//!
//! RUST PORT NOTE: the gmock mock becomes a hand-written fake that records
//! calls (by C++ method name) and hands out [`MockTrajectoryBuilder`]s and a
//! [`MockPoseGraph`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::io::proto_stream_interface::{ProtoStreamReaderInterface, ProtoStreamWriterInterface};
use crate::mapping::id::SubmapId;
use crate::mapping::internal::testing::mock_pose_graph::MockPoseGraph;
use crate::mapping::internal::testing::mock_trajectory_builder::MockTrajectoryBuilder;
use crate::mapping::map_builder_interface::{
    LocalSlamResultCallback, MapBuilderInterface, SensorId,
};
use crate::mapping::pose_graph_interface::PoseGraphInterface;
use crate::mapping::proto;
use crate::mapping::trajectory_builder_interface::{self, TrajectoryBuilderInterface};

/// Fake `MapBuilderInterface`.
#[derive(Debug, Default)]
pub struct MockMapBuilder {
    /// Called methods (C++ names) in call order.
    pub calls: Vec<&'static str>,
    /// One entry per trajectory; `None` for deserialized trajectories.
    pub trajectory_builders: Vec<Option<MockTrajectoryBuilder>>,
    /// The options of all trajectories.
    pub all_trajectory_builder_options: Vec<proto::TrajectoryBuilderOptionsWithSensorIds>,
    /// The fake pose graph returned by `pose_graph()`.
    pub pose_graph: Arc<MockPoseGraph>,
}

impl MockMapBuilder {
    /// Creates an empty fake.
    pub fn new() -> Self {
        Self::default()
    }
}

impl MapBuilderInterface for MockMapBuilder {
    fn add_trajectory_builder(
        &mut self,
        expected_sensor_ids: &BTreeSet<SensorId>,
        trajectory_options: &proto::TrajectoryBuilderOptions,
        _local_slam_result_callback: Option<LocalSlamResultCallback>,
    ) -> i32 {
        self.calls.push("AddTrajectoryBuilder");
        self.trajectory_builders
            .push(Some(MockTrajectoryBuilder::new()));
        self.all_trajectory_builder_options
            .push(proto::TrajectoryBuilderOptionsWithSensorIds {
                sensor_id: expected_sensor_ids
                    .iter()
                    .map(trajectory_builder_interface::to_proto)
                    .collect(),
                trajectory_builder_options: Some(trajectory_options.clone()),
            });
        self.trajectory_builders.len() as i32 - 1
    }

    fn add_trajectory_for_deserialization(
        &mut self,
        options_with_sensor_ids_proto: &proto::TrajectoryBuilderOptionsWithSensorIds,
    ) -> i32 {
        self.calls.push("AddTrajectoryForDeserialization");
        self.trajectory_builders.push(None);
        self.all_trajectory_builder_options
            .push(options_with_sensor_ids_proto.clone());
        self.trajectory_builders.len() as i32 - 1
    }

    fn get_trajectory_builder(
        &mut self,
        trajectory_id: i32,
    ) -> Option<&mut dyn TrajectoryBuilderInterface> {
        self.calls.push("GetTrajectoryBuilder");
        self.trajectory_builders
            .get_mut(usize::try_from(trajectory_id).ok()?)?
            .as_mut()
            .map(|builder| builder as &mut dyn TrajectoryBuilderInterface)
    }

    fn finish_trajectory(&mut self, _trajectory_id: i32) {
        self.calls.push("FinishTrajectory");
    }

    fn submap_to_proto(
        &mut self,
        _submap_id: &SubmapId,
        _response: &mut proto::submap_query::Response,
    ) -> String {
        self.calls.push("SubmapToProto");
        String::new()
    }

    fn serialize_state(
        &mut self,
        _include_unfinished_submaps: bool,
        _writer: &mut dyn ProtoStreamWriterInterface,
    ) {
        self.calls.push("SerializeState");
    }

    fn serialize_state_to_file(
        &mut self,
        _include_unfinished_submaps: bool,
        _filename: &str,
    ) -> bool {
        self.calls.push("SerializeStateToFile");
        true
    }

    fn load_state(
        &mut self,
        _reader: &mut dyn ProtoStreamReaderInterface,
        _load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        self.calls.push("LoadState");
        BTreeMap::new()
    }

    fn load_state_from_file(
        &mut self,
        _filename: &str,
        _load_frozen_state: bool,
    ) -> BTreeMap<i32, i32> {
        self.calls.push("LoadStateFromFile");
        BTreeMap::new()
    }

    fn num_trajectory_builders(&self) -> i32 {
        self.trajectory_builders.len() as i32
    }

    fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
        Arc::clone(&self.pose_graph) as Arc<dyn PoseGraphInterface>
    }

    fn get_all_trajectory_builder_options(
        &self,
    ) -> &[proto::TrajectoryBuilderOptionsWithSensorIds] {
        &self.all_trajectory_builder_options
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_universal;
    use crate::mapping::trajectory_builder_interface::SensorType;
    use crate::sensor::imu_data::ImuData;

    #[test]
    fn fake_is_usable_as_trait_object() {
        let mut fake = MockMapBuilder::new();
        let map_builder: &mut dyn MapBuilderInterface = &mut fake;
        let sensor_ids: BTreeSet<SensorId> = [SensorId {
            r#type: SensorType::Imu,
            id: "imu".to_string(),
        }]
        .into_iter()
        .collect();
        let trajectory_id = map_builder.add_trajectory_builder(
            &sensor_ids,
            &proto::TrajectoryBuilderOptions::default(),
            None,
        );
        assert_eq!(trajectory_id, 0);
        let deserialized = map_builder.add_trajectory_for_deserialization(
            &proto::TrajectoryBuilderOptionsWithSensorIds::default(),
        );
        assert_eq!(deserialized, 1);
        assert!(map_builder.get_trajectory_builder(deserialized).is_none());
        map_builder
            .get_trajectory_builder(trajectory_id)
            .unwrap()
            .add_imu_data(
                "imu",
                &ImuData {
                    time: from_universal(1),
                    ..Default::default()
                },
            );
        assert_eq!(map_builder.num_trajectory_builders(), 2);
        assert_eq!(map_builder.get_all_trajectory_builder_options().len(), 2);
        assert_eq!(
            map_builder.get_all_trajectory_builder_options()[0].sensor_id[0].id,
            "imu"
        );
        assert!(map_builder.pose_graph().get_trajectory_nodes().empty());
        assert_eq!(
            fake.trajectory_builders[0]
                .as_ref()
                .unwrap()
                .recorded_calls()
                .len(),
            1
        );
    }
}
