//! Port of `cartographer/mapping/trajectory_builder_interface` (.h, .cc).

use std::fmt;
use std::sync::{Arc, RwLock};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::Time;
use crate::mapping::id::NodeId;
use crate::mapping::internal::local_slam_result_data::LocalSlamResultData;
use crate::mapping::internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
use crate::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
use crate::mapping::proto;
use crate::mapping::submaps::Submap;
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::range_data::RangeData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::Rigid3d;

/// Reads the optional `pure_localization_trimmer` dictionary into
/// `trajectory_builder_options` (C++ anonymous-namespace
/// `PopulatePureLocalizationTrimmerOptions`).
pub fn populate_pure_localization_trimmer_options(
    trajectory_builder_options: &mut proto::TrajectoryBuilderOptions,
    parameter_dictionary: &LuaParameterDictionary,
) {
    const DICTIONARY_KEY: &str = "pure_localization_trimmer";
    if !parameter_dictionary.has_key(DICTIONARY_KEY) {
        return;
    }
    let options_dictionary = parameter_dictionary.get_dictionary(DICTIONARY_KEY);
    trajectory_builder_options.pure_localization_trimmer = Some(
        proto::trajectory_builder_options::PureLocalizationTrimmerOptions {
            max_submaps_to_keep: options_dictionary.get_int("max_submaps_to_keep"),
        },
    );
}

/// Reads the optional `pose_graph_odometry_motion_filter` dictionary into
/// `trajectory_builder_options` (C++ anonymous-namespace
/// `PopulatePoseGraphOdometryMotionFilterOptions`).
pub fn populate_pose_graph_odometry_motion_filter_options(
    trajectory_builder_options: &mut proto::TrajectoryBuilderOptions,
    parameter_dictionary: &LuaParameterDictionary,
) {
    const DICTIONARY_KEY: &str = "pose_graph_odometry_motion_filter";
    if !parameter_dictionary.has_key(DICTIONARY_KEY) {
        return;
    }
    let options_dictionary = parameter_dictionary.get_dictionary(DICTIONARY_KEY);
    trajectory_builder_options.pose_graph_odometry_motion_filter =
        Some(proto::MotionFilterOptions {
            max_time_seconds: options_dictionary.get_double("max_time_seconds"),
            max_distance_meters: options_dictionary.get_double("max_distance_meters"),
            max_angle_radians: options_dictionary.get_double("max_angle_radians"),
        });
}

/// Reads `TrajectoryBuilderOptions` from `parameter_dictionary` (C++
/// `CreateTrajectoryBuilderOptions`).
///
/// Like in C++, `initial_trajectory_pose` is not read from Lua (it is only
/// set programmatically, e.g. by cartographer_ros), so it stays unset.
pub fn create_trajectory_builder_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::TrajectoryBuilderOptions {
    let mut options = proto::TrajectoryBuilderOptions {
        trajectory_builder_2d_options: Some(create_local_trajectory_builder_options_2d(
            &parameter_dictionary.get_dictionary("trajectory_builder_2d"),
        )),
        trajectory_builder_3d_options: Some(create_local_trajectory_builder_options_3d(
            &parameter_dictionary.get_dictionary("trajectory_builder_3d"),
        )),
        ..Default::default()
    };
    options.collate_fixed_frame = parameter_dictionary.get_bool("collate_fixed_frame");
    options.collate_landmarks = parameter_dictionary.get_bool("collate_landmarks");
    populate_pure_localization_trimmer_options(&mut options, parameter_dictionary);
    populate_pose_graph_odometry_motion_filter_options(&mut options, parameter_dictionary);
    options
}

/// The result of inserting a node into the pose graph (C++
/// `TrajectoryBuilderInterface::InsertionResult`).
#[derive(Clone)]
pub struct InsertionResult {
    /// The id the pose graph assigned to the node.
    pub node_id: NodeId,
    /// The constant data of the node.
    pub constant_data: Arc<TrajectoryNodeData>,
    /// The submaps the node was inserted into (C++
    /// `std::vector<std::shared_ptr<const Submap>>`).
    pub insertion_submaps: Vec<Arc<RwLock<dyn Submap>>>,
}

impl fmt::Debug for InsertionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InsertionResult")
            .field("node_id", &self.node_id)
            .field("constant_data", &self.constant_data)
            .field("num_insertion_submaps", &self.insertion_submaps.len())
            .finish()
    }
}

/// A callback which is called after local SLAM processes an accumulated
/// `sensor::RangeData`. If the data was inserted into a submap, reports the
/// assigned `NodeId`, otherwise `None` if the data was filtered out.
///
/// Arguments: `(trajectory_id, time, local_pose_estimate,
/// range_data_in_local_frame, insertion_result)`.
///
/// RUST PORT NOTE: the C++ `std::function` (copyable, possibly empty)
/// becomes a shareable `Arc<dyn Fn>`; "empty" is expressed as
/// `Option<LocalSlamResultCallback>`. The `std::unique_ptr<const
/// InsertionResult>` argument becomes `Option<InsertionResult>`.
pub type LocalSlamResultCallback =
    Arc<dyn Fn(i32, Time, Rigid3d, RangeData, Option<InsertionResult>) + Send + Sync>;

/// Sensor type of a [`SensorId`] (C++ `SensorId::SensorType`). The
/// declaration order equals the C++ enum values and defines the ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SensorType {
    /// C++ `RANGE`.
    Range = 0,
    /// C++ `IMU`.
    Imu,
    /// C++ `ODOMETRY`.
    Odometry,
    /// C++ `FIXED_FRAME_POSE`.
    FixedFramePose,
    /// C++ `LANDMARK`.
    Landmark,
    /// C++ `LOCAL_SLAM_RESULT`.
    LocalSlamResult,
}

/// Identifies a sensor of a trajectory (C++
/// `TrajectoryBuilderInterface::SensorId`). Ordered by `(type, id)` like in
/// C++.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId {
    /// The sensor type.
    pub r#type: SensorType,
    /// The sensor id (topic name).
    pub id: String,
}

/// This interface is used for both 2D and 3D SLAM. Implementations wire up a
/// global SLAM stack, i.e. local SLAM for initial pose estimates, scan matching
/// to detect loop closure, and a sparse pose graph optimization to compute
/// optimized pose estimates.
///
/// RUST PORT NOTE: Rust has no overloading, so the C++ `AddSensorData`
/// overloads are disambiguated by a type suffix:
///
/// | C++ | Rust |
/// |---|---|
/// | `AddSensorData(id, const TimedPointCloudData&)` | [`add_timed_point_cloud_data`](Self::add_timed_point_cloud_data) |
/// | `AddSensorData(id, const ImuData&)` | [`add_imu_data`](Self::add_imu_data) |
/// | `AddSensorData(id, const OdometryData&)` | [`add_odometry_data`](Self::add_odometry_data) |
/// | `AddSensorData(id, const FixedFramePoseData&)` | [`add_fixed_frame_pose_data`](Self::add_fixed_frame_pose_data) |
/// | `AddSensorData(id, const LandmarkData&)` | [`add_landmark_data`](Self::add_landmark_data) |
///
/// Implementations must be `Send` so that a map builder owning them can be
/// moved between threads.
pub trait TrajectoryBuilderInterface: Send {
    /// C++ `AddSensorData(sensor_id, const sensor::TimedPointCloudData&)`.
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    );
    /// C++ `AddSensorData(sensor_id, const sensor::ImuData&)`.
    fn add_imu_data(&mut self, sensor_id: &str, imu_data: &ImuData);
    /// C++ `AddSensorData(sensor_id, const sensor::OdometryData&)`.
    fn add_odometry_data(&mut self, sensor_id: &str, odometry_data: &OdometryData);
    /// C++ `AddSensorData(sensor_id, const sensor::FixedFramePoseData&)`.
    fn add_fixed_frame_pose_data(&mut self, sensor_id: &str, fixed_frame_pose: &FixedFramePoseData);
    /// C++ `AddSensorData(sensor_id, const sensor::LandmarkData&)`.
    fn add_landmark_data(&mut self, sensor_id: &str, landmark_data: &LandmarkData);

    /// Allows to directly add local SLAM results to the `PoseGraph`. Note
    /// that it is invalid to add local SLAM results for a trajectory that has
    /// a `LocalTrajectoryBuilder2D/3D`.
    fn add_local_slam_result_data(&mut self, local_slam_result_data: Box<dyn LocalSlamResultData>);
}

/// Converts `sensor_id` to a `proto::SensorId`.
pub fn to_proto(sensor_id: &SensorId) -> proto::SensorId {
    use proto::sensor_id::SensorType as P;
    let r#type = match sensor_id.r#type {
        SensorType::Range => P::Range,
        SensorType::Imu => P::Imu,
        SensorType::Odometry => P::Odometry,
        SensorType::FixedFramePose => P::FixedFramePose,
        SensorType::Landmark => P::Landmark,
        SensorType::LocalSlamResult => P::LocalSlamResult,
    };
    proto::SensorId {
        r#type: r#type as i32,
        id: sensor_id.id.clone(),
    }
}

/// Converts `sensor_id_proto` to a `SensorId`. Panics on an unknown sensor
/// type (C++ `LOG(FATAL)`).
pub fn from_proto(sensor_id_proto: &proto::SensorId) -> SensorId {
    use proto::sensor_id::SensorType as P;
    let r#type = match P::try_from(sensor_id_proto.r#type) {
        Ok(P::Range) => SensorType::Range,
        Ok(P::Imu) => SensorType::Imu,
        Ok(P::Odometry) => SensorType::Odometry,
        Ok(P::FixedFramePose) => SensorType::FixedFramePose,
        Ok(P::Landmark) => SensorType::Landmark,
        Ok(P::LocalSlamResult) => SensorType::LocalSlamResult,
        Err(_) => panic!("Unsupported sensor type."),
    };
    SensorId {
        r#type,
        id: sensor_id_proto.id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    #[test]
    fn sensor_id_proto_round_trip() {
        for r#type in [
            SensorType::Range,
            SensorType::Imu,
            SensorType::Odometry,
            SensorType::FixedFramePose,
            SensorType::Landmark,
            SensorType::LocalSlamResult,
        ] {
            let sensor_id = SensorId {
                r#type,
                id: "some_sensor".to_string(),
            };
            let proto = to_proto(&sensor_id);
            assert_eq!(proto.r#type, r#type as i32);
            assert_eq!(from_proto(&proto), sensor_id);
        }
    }

    #[test]
    fn sensor_id_ordering_is_type_then_id() {
        let a = SensorId {
            r#type: SensorType::Range,
            id: "z".to_string(),
        };
        let b = SensorId {
            r#type: SensorType::Imu,
            id: "a".to_string(),
        };
        assert!(a < b);
    }

    #[test]
    #[should_panic(expected = "Unsupported sensor type.")]
    fn unknown_sensor_type_panics() {
        let _ = from_proto(&proto::SensorId {
            r#type: 42,
            id: String::new(),
        });
    }

    #[test]
    fn optional_dictionaries_are_read_when_present() {
        let mut options = proto::TrajectoryBuilderOptions::default();
        let dictionary = make_dictionary(
            r#"return {
              pure_localization_trimmer = { max_submaps_to_keep = 3 },
              pose_graph_odometry_motion_filter = {
                max_time_seconds = 1.5,
                max_distance_meters = 0.25,
                max_angle_radians = 0.125,
              },
            }"#,
        );
        populate_pure_localization_trimmer_options(&mut options, &dictionary);
        populate_pose_graph_odometry_motion_filter_options(&mut options, &dictionary);
        assert_eq!(
            options
                .pure_localization_trimmer
                .unwrap()
                .max_submaps_to_keep,
            3
        );
        let filter = options.pose_graph_odometry_motion_filter.unwrap();
        assert_eq!(filter.max_time_seconds, 1.5);
        assert_eq!(filter.max_distance_meters, 0.25);
        assert_eq!(filter.max_angle_radians, 0.125);
    }

    #[test]
    fn optional_dictionaries_are_skipped_when_absent() {
        let mut options = proto::TrajectoryBuilderOptions::default();
        let dictionary = make_dictionary("return {}");
        populate_pure_localization_trimmer_options(&mut options, &dictionary);
        populate_pose_graph_odometry_motion_filter_options(&mut options, &dictionary);
        assert!(options.pure_localization_trimmer.is_none());
        assert!(options.pose_graph_odometry_motion_filter.is_none());
    }
}
