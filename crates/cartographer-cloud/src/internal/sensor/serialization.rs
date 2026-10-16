//! Port of `cartographer/cloud/internal/sensor/serialization` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ `Create*Request(..., proto::X* proto)` functions
//! fill an output parameter; here they return the message.

use cartographer::common::time::{Time, to_universal};
use cartographer::mapping::trajectory_builder_interface::{InsertionResult, SensorId, SensorType};
use cartographer::mapping::{proto as mapping_proto, trajectory_node};
use cartographer::sensor::proto as sensor_proto;

use crate::proto;

/// Creates the `SensorMetadata` of an `Add*DataRequest`.
pub fn create_sensor_metadata(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
) -> proto::SensorMetadata {
    proto::SensorMetadata {
        trajectory_id,
        sensor_id: sensor_id.to_string(),
        client_id: client_id.to_string(),
    }
}

/// C++ `CreateAddFixedFramePoseDataRequest`.
pub fn create_add_fixed_frame_pose_data_request(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    fixed_frame_pose_data: &sensor_proto::FixedFramePoseData,
) -> proto::AddFixedFramePoseDataRequest {
    proto::AddFixedFramePoseDataRequest {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        fixed_frame_pose_data: Some(*fixed_frame_pose_data),
    }
}

/// C++ `CreateAddImuDataRequest`.
pub fn create_add_imu_data_request(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    imu_data: &sensor_proto::ImuData,
) -> proto::AddImuDataRequest {
    proto::AddImuDataRequest {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        imu_data: Some(*imu_data),
    }
}

/// C++ `CreateAddOdometryDataRequest`.
pub fn create_add_odometry_data_request(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    odometry_data: &sensor_proto::OdometryData,
) -> proto::AddOdometryDataRequest {
    proto::AddOdometryDataRequest {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        odometry_data: Some(*odometry_data),
    }
}

/// C++ `CreateAddRangeFinderDataRequest`.
pub fn create_add_range_finder_data_request(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    timed_point_cloud_data: &sensor_proto::TimedPointCloudData,
) -> proto::AddRangefinderDataRequest {
    proto::AddRangefinderDataRequest {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        timed_point_cloud_data: Some(timed_point_cloud_data.clone()),
    }
}

/// C++ `CreateAddLandmarkDataRequest`.
pub fn create_add_landmark_data_request(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    landmark_data: &sensor_proto::LandmarkData,
) -> proto::AddLandmarkDataRequest {
    proto::AddLandmarkDataRequest {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        landmark_data: Some(landmark_data.clone()),
    }
}

/// Serializes a local SLAM result (node plus insertion submaps) for upload.
/// The submaps get consecutive indices starting at `starting_submap_index`;
/// a submap's grid is only included if the submap is finished.
pub fn create_sensor_data_for_local_slam_result(
    sensor_id: &str,
    trajectory_id: i32,
    client_id: &str,
    time: Time,
    mut starting_submap_index: i32,
    insertion_result: &InsertionResult,
) -> proto::SensorData {
    let mut local_slam_result_data = mapping_proto::LocalSlamResultData {
        timestamp: to_universal(time),
        node_data: Some(trajectory_node::to_proto(&insertion_result.constant_data)),
        submaps: Vec::new(),
    };
    for insertion_submap in &insertion_result.insertion_submaps {
        // We only send the probability grid up if the submap is finished.
        let submap = insertion_submap
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut submap_proto = submap.to_proto(submap.insertion_finished());
        submap_proto.submap_id = Some(mapping_proto::SubmapId {
            trajectory_id,
            submap_index: starting_submap_index,
        });
        local_slam_result_data.submaps.push(submap_proto);
        starting_submap_index += 1;
    }
    proto::SensorData {
        sensor_metadata: Some(create_sensor_metadata(sensor_id, trajectory_id, client_id)),
        sensor_data: Some(proto::sensor_data::SensorData::LocalSlamResultData(
            local_slam_result_data,
        )),
    }
}

/// Converts a `SensorId` to its cloud proto.
pub fn to_proto(sensor_id: &SensorId) -> proto::SensorId {
    let r#type = match sensor_id.r#type {
        SensorType::Range => proto::SensorType::Range,
        SensorType::Imu => proto::SensorType::Imu,
        SensorType::Odometry => proto::SensorType::Odometry,
        SensorType::FixedFramePose => proto::SensorType::FixedFramePose,
        SensorType::Landmark => proto::SensorType::Landmark,
        SensorType::LocalSlamResult => proto::SensorType::LocalSlamResult,
    };
    proto::SensorId {
        id: sensor_id.id.clone(),
        r#type: r#type as i32,
    }
}

/// Converts a cloud proto to a `SensorId`. Panics on an unknown type (C++
/// `LOG(FATAL) << "unknown proto::SensorType"`).
pub fn from_proto(proto: &proto::SensorId) -> SensorId {
    let r#type = match proto::SensorType::try_from(proto.r#type) {
        Ok(proto::SensorType::Range) => SensorType::Range,
        Ok(proto::SensorType::Imu) => SensorType::Imu,
        Ok(proto::SensorType::Odometry) => SensorType::Odometry,
        Ok(proto::SensorType::FixedFramePose) => SensorType::FixedFramePose,
        Ok(proto::SensorType::Landmark) => SensorType::Landmark,
        Ok(proto::SensorType::LocalSlamResult) => SensorType::LocalSlamResult,
        Err(_) => panic!("unknown proto::SensorType"),
    };
    SensorId {
        r#type,
        id: proto.id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, RwLock};

    use cartographer::common::time::from_universal;
    use cartographer::mapping::id::NodeId;
    use cartographer::mapping::internal::testing::test_helpers::resolve_lua_parameters;
    use cartographer::mapping::submaps::Submap;
    use cartographer::mapping::trajectory_node::TrajectoryNodeData;
    use cartographer::mapping::two_d::submap_2d::{ActiveSubmaps2D, create_submaps_options_2d};
    use cartographer::sensor::point_cloud::PointCloud;
    use cartographer::sensor::range_data::RangeData;
    use cartographer::sensor::rangefinder_point::RangefinderPoint;
    use cartographer::transform::rigid_transform::Rigid3d;
    use nalgebra::{DVector, Quaternion, Vector3};

    const ALL_TYPES: [SensorType; 6] = [
        SensorType::Range,
        SensorType::Imu,
        SensorType::Odometry,
        SensorType::FixedFramePose,
        SensorType::Landmark,
        SensorType::LocalSlamResult,
    ];

    #[test]
    fn sensor_id_round_trip() {
        for r#type in ALL_TYPES {
            let sensor_id = SensorId {
                r#type,
                id: "sensor".to_string(),
            };
            let proto = to_proto(&sensor_id);
            assert_eq!(proto.r#type, r#type as i32);
            assert_eq!(from_proto(&proto), sensor_id);
        }
    }

    #[test]
    #[should_panic(expected = "unknown proto::SensorType")]
    fn unknown_sensor_type_panics() {
        let _ = from_proto(&proto::SensorId {
            id: String::new(),
            r#type: 17,
        });
    }

    #[test]
    fn add_requests_carry_metadata_and_payload() {
        let imu = sensor_proto::ImuData {
            timestamp: 2,
            ..Default::default()
        };
        let request = create_add_imu_data_request("imu", 3, "CLIENT", &imu);
        assert_eq!(
            request.sensor_metadata,
            Some(proto::SensorMetadata {
                trajectory_id: 3,
                sensor_id: "imu".to_string(),
                client_id: "CLIENT".to_string(),
            })
        );
        assert_eq!(request.imu_data, Some(imu));
        let odometry = sensor_proto::OdometryData {
            timestamp: 4,
            ..Default::default()
        };
        let request = create_add_odometry_data_request("odom", 1, "C", &odometry);
        assert_eq!(request.odometry_data, Some(odometry));
        assert_eq!(request.sensor_metadata.unwrap().sensor_id, "odom");
        let fixed = sensor_proto::FixedFramePoseData {
            timestamp: 5,
            ..Default::default()
        };
        let request = create_add_fixed_frame_pose_data_request("gps", 1, "C", &fixed);
        assert_eq!(request.fixed_frame_pose_data, Some(fixed));
        let landmark = sensor_proto::LandmarkData {
            timestamp: 6,
            ..Default::default()
        };
        let request = create_add_landmark_data_request("lm", 1, "C", &landmark);
        assert_eq!(request.landmark_data, Some(landmark));
        let cloud = sensor_proto::TimedPointCloudData {
            timestamp: 7,
            ..Default::default()
        };
        let request = create_add_range_finder_data_request("range", 1, "C", &cloud);
        assert_eq!(request.timed_point_cloud_data, Some(cloud));
    }

    #[test]
    fn local_slam_result_numbers_submaps_and_includes_finished_grids_only() {
        let mut active_submaps =
            ActiveSubmaps2D::new(&create_submaps_options_2d(&resolve_lua_parameters(
                r#"include "trajectory_builder_2d.lua"
                TRAJECTORY_BUILDER_2D.submaps.num_range_data = 1
                return TRAJECTORY_BUILDER_2D.submaps"#,
            )));
        let range_data = RangeData {
            origin: Vector3::zeros(),
            returns: PointCloud::from_points(vec![RangefinderPoint {
                position: Vector3::new(1., 0., 0.),
            }]),
            misses: PointCloud::default(),
        };
        active_submaps.insert_range_data(&range_data);
        let submaps = active_submaps.insert_range_data(&range_data);
        assert_eq!(submaps.len(), 2);
        assert!(submaps[0].read().unwrap().insertion_finished());
        assert!(!submaps[1].read().unwrap().insertion_finished());
        let insertion_result = InsertionResult {
            node_id: NodeId::new(0, 1),
            constant_data: Arc::new(TrajectoryNodeData {
                time: from_universal(42),
                gravity_alignment: Quaternion::identity(),
                filtered_gravity_aligned_point_cloud: PointCloud::default(),
                high_resolution_point_cloud: PointCloud::default(),
                low_resolution_point_cloud: PointCloud::default(),
                rotational_scan_matcher_histogram: DVector::zeros(0),
                local_pose: Rigid3d::identity(),
            }),
            insertion_submaps: submaps
                .iter()
                .map(|s| Arc::clone(s) as Arc<RwLock<dyn Submap>>)
                .collect(),
        };
        let sensor_data = create_sensor_data_for_local_slam_result(
            "local_slam_result_3",
            3,
            "CLIENT",
            from_universal(42),
            7,
            &insertion_result,
        );
        let metadata = sensor_data.sensor_metadata.unwrap();
        assert_eq!(metadata.trajectory_id, 3);
        assert_eq!(metadata.sensor_id, "local_slam_result_3");
        assert_eq!(metadata.client_id, "CLIENT");
        let Some(proto::sensor_data::SensorData::LocalSlamResultData(data)) =
            sensor_data.sensor_data
        else {
            panic!("expected local_slam_result_data");
        };
        assert_eq!(data.timestamp, 42);
        assert_eq!(data.node_data.unwrap().timestamp, 42);
        assert_eq!(data.submaps.len(), 2);
        for (i, submap) in data.submaps.iter().enumerate() {
            assert_eq!(
                submap.submap_id,
                Some(mapping_proto::SubmapId {
                    trajectory_id: 3,
                    submap_index: 7 + i as i32,
                })
            );
        }
        assert!(data.submaps[0].submap_2d.as_ref().unwrap().grid.is_some());
        assert!(data.submaps[1].submap_2d.as_ref().unwrap().grid.is_none());
    }
}
