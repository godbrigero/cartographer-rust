//! Port of `cartographer/mapping/trajectory_node` (.h, .cc).

use std::sync::Arc;

use nalgebra::{DVector, Quaternion};

use crate::common::time::{Time, from_universal, to_universal};
use crate::mapping::proto;
use crate::sensor::compressed_point_cloud::CompressedPointCloud;
use crate::sensor::data::HasTime;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{ToEigen, ToProto, to_rigid3};

/// Constant data of a [`TrajectoryNodePose`]
/// (C++ `TrajectoryNodePose::ConstantPoseData`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantPoseData {
    /// Time of the node.
    pub time: Time,
    /// The node pose in the local SLAM frame.
    pub local_pose: Rigid3d,
}

/// A trajectory node pose without the (heavy) sensor data.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrajectoryNodePose {
    /// The node pose in the global SLAM frame.
    pub global_pose: Rigid3d,
    /// Constant data, if known.
    pub constant_pose_data: Option<ConstantPoseData>,
}

impl TrajectoryNodePose {
    /// Creates a node pose (C++ aggregate initialization).
    pub fn new(global_pose: Rigid3d, constant_pose_data: Option<ConstantPoseData>) -> Self {
        TrajectoryNodePose {
            global_pose,
            constant_pose_data,
        }
    }
}

/// The constant data of a [`TrajectoryNode`] (C++ `TrajectoryNode::Data`).
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryNodeData {
    /// Time of the node.
    pub time: Time,
    /// Transform to approximately gravity align the tracking frame as
    /// determined by local SLAM.
    pub gravity_alignment: Quaternion<f64>,
    /// Used for loop closure in 2D: voxel filtered returns in the
    /// `gravity_alignment` frame.
    pub filtered_gravity_aligned_point_cloud: PointCloud,
    /// Used for loop closure in 3D.
    pub high_resolution_point_cloud: PointCloud,
    /// Used for loop closure in 3D.
    pub low_resolution_point_cloud: PointCloud,
    /// Used for loop closure in 3D.
    pub rotational_scan_matcher_histogram: DVector<f32>,
    /// The node pose in the local SLAM frame.
    pub local_pose: Rigid3d,
}

/// A node of a trajectory: shared constant data plus the global pose.
#[derive(Clone, Debug, Default)]
pub struct TrajectoryNode {
    /// This must be shared (`Arc`). If the data is used for visualization
    /// while the node is being trimmed, it must survive until all use
    /// finishes.
    ///
    /// RUST PORT NOTE: `Option` models the C++ `shared_ptr` that may be null
    /// (e.g. a default-constructed node).
    pub constant_data: Option<Arc<TrajectoryNodeData>>,
    /// The node pose in the global SLAM frame.
    pub global_pose: Rigid3d,
}

impl TrajectoryNode {
    /// Creates a node from its constant data and global pose.
    pub fn new(constant_data: Arc<TrajectoryNodeData>, global_pose: Rigid3d) -> Self {
        TrajectoryNode {
            constant_data: Some(constant_data),
            global_pose,
        }
    }

    /// The time of the node (C++ `TrajectoryNode::time()`). Panics if
    /// `constant_data` is unset (a null dereference in C++).
    pub fn time(&self) -> Time {
        self.constant_data
            .as_ref()
            .expect("TrajectoryNode::time() requires constant_data")
            .time
    }
}

impl HasTime for TrajectoryNode {
    fn time(&self) -> Time {
        TrajectoryNode::time(self)
    }
}

/// Converts the constant data of a node to its proto (C++ `ToProto`).
pub fn to_proto(constant_data: &TrajectoryNodeData) -> proto::TrajectoryNodeData {
    proto::TrajectoryNodeData {
        timestamp: to_universal(constant_data.time),
        gravity_alignment: Some(constant_data.gravity_alignment.to_proto()),
        filtered_gravity_aligned_point_cloud: Some(
            CompressedPointCloud::from_point_cloud(
                &constant_data.filtered_gravity_aligned_point_cloud,
            )
            .to_proto(),
        ),
        high_resolution_point_cloud: Some(
            CompressedPointCloud::from_point_cloud(&constant_data.high_resolution_point_cloud)
                .to_proto(),
        ),
        low_resolution_point_cloud: Some(
            CompressedPointCloud::from_point_cloud(&constant_data.low_resolution_point_cloud)
                .to_proto(),
        ),
        rotational_scan_matcher_histogram: constant_data
            .rotational_scan_matcher_histogram
            .iter()
            .copied()
            .collect(),
        local_pose: Some(constant_data.local_pose.to_proto()),
    }
}

/// Converts a proto to the constant data of a node (C++ `FromProto`).
/// Missing sub-messages are treated like their proto3 defaults.
pub fn from_proto(proto: &proto::TrajectoryNodeData) -> TrajectoryNodeData {
    let decompress = |cloud: &Option<crate::sensor::proto::CompressedPointCloud>| {
        CompressedPointCloud::from_proto(&cloud.clone().unwrap_or_default()).decompress()
    };
    TrajectoryNodeData {
        time: from_universal(proto.timestamp),
        gravity_alignment: proto.gravity_alignment.unwrap_or_default().to_eigen(),
        filtered_gravity_aligned_point_cloud: decompress(
            &proto.filtered_gravity_aligned_point_cloud,
        ),
        high_resolution_point_cloud: decompress(&proto.high_resolution_point_cloud),
        low_resolution_point_cloud: decompress(&proto.low_resolution_point_cloud),
        rotational_scan_matcher_histogram: DVector::from_vec(
            proto.rotational_scan_matcher_histogram.clone(),
        ),
        local_pose: to_rigid3(&proto.local_pose.unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::quaternion_normalized;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use nalgebra::Vector3;
    use proptest::prelude::*;

    fn cloud(points: &[[f32; 3]]) -> PointCloud {
        PointCloud::from_points(
            points
                .iter()
                .map(|p| RangefinderPoint {
                    position: Vector3::new(p[0], p[1], p[2]),
                })
                .collect(),
        )
    }

    fn compressed(points: &[[f32; 3]]) -> PointCloud {
        CompressedPointCloud::from_point_cloud(&cloud(points)).decompress()
    }

    /// Eigen's `isApprox` for quaternions (coefficient vectors, default
    /// double precision 1e-12).
    fn quaternion_is_approx(a: &Quaternion<f64>, b: &Quaternion<f64>) -> bool {
        let diff = (a.coords - b.coords).norm_squared();
        diff <= 1e-24 * a.coords.norm_squared().min(b.coords.norm_squared())
    }

    mod trajectory_node_test {
        use super::*;

        #[test]
        fn to_and_from_proto() {
            let mut histogram = DVector::<f32>::zeros(20);
            histogram[4] = 1.;
            let expected = TrajectoryNodeData {
                time: from_universal(42),
                gravity_alignment: Quaternion::new(1., 2., -3., -4.),
                filtered_gravity_aligned_point_cloud: compressed(&[[1., 2., 0.], [0., 0., 1.]]),
                high_resolution_point_cloud: compressed(&[[2., 3., 4.]]),
                low_resolution_point_cloud: compressed(&[[-1., 2., 0.]]),
                rotational_scan_matcher_histogram: histogram,
                local_pose: Rigid3d::new(
                    Vector3::new(1., 2., 3.),
                    quaternion_normalized(&Quaternion::new(4., 5., -6., -7.)),
                ),
            };
            let proto = to_proto(&expected);
            let actual = from_proto(&proto);
            assert_eq!(expected.time, actual.time);
            assert!(quaternion_is_approx(
                &actual.gravity_alignment,
                &expected.gravity_alignment
            ));
            assert_eq!(
                expected.filtered_gravity_aligned_point_cloud.points(),
                actual.filtered_gravity_aligned_point_cloud.points()
            );
            assert_eq!(
                expected.high_resolution_point_cloud.points(),
                actual.high_resolution_point_cloud.points()
            );
            assert_eq!(
                expected.low_resolution_point_cloud.points(),
                actual.low_resolution_point_cloud.points()
            );
            assert_eq!(
                expected.rotational_scan_matcher_histogram,
                actual.rotational_scan_matcher_histogram
            );
            assert_is_nearly(&actual.local_pose, &expected.local_pose, 1e-9);
        }
    }

    #[test]
    fn trajectory_node_time_and_has_time() {
        let data = from_proto(&proto::TrajectoryNodeData {
            timestamp: 1234,
            ..Default::default()
        });
        assert!(data.filtered_gravity_aligned_point_cloud.is_empty());
        assert_eq!(data.rotational_scan_matcher_histogram.len(), 0);
        let node = TrajectoryNode::new(Arc::new(data), Rigid3d::identity());
        assert_eq!(node.time(), from_universal(1234));
        assert_eq!(HasTime::time(&node), from_universal(1234));
    }

    proptest! {
        #[test]
        fn proto_round_trip_is_stable(
            timestamp in any::<i64>(),
            w in -1.0..1.0f64, x in -1.0..1.0f64, y in -1.0..1.0f64, z in -1.0..1.0f64,
            tx in -100.0..100.0f64, ty in -100.0..100.0f64, tz in -100.0..100.0f64,
            histogram in proptest::collection::vec(-10.0..10.0f32, 0..40),
            points in proptest::collection::vec(
                (-50.0..50.0f32, -50.0..50.0f32, -50.0..50.0f32), 0..20),
        ) {
            let points: Vec<[f32; 3]> = points.into_iter().map(|(a, b, c)| [a, b, c]).collect();
            let data = TrajectoryNodeData {
                time: from_universal(timestamp),
                gravity_alignment: Quaternion::new(w, x, y, z),
                filtered_gravity_aligned_point_cloud: compressed(&points),
                high_resolution_point_cloud: compressed(&points),
                low_resolution_point_cloud: PointCloud::new(),
                rotational_scan_matcher_histogram: DVector::from_vec(histogram),
                local_pose: Rigid3d::new(
                    Vector3::new(tx, ty, tz),
                    Quaternion::new(w, x, y, z),
                ),
            };
            // Compressed clouds are a fixed point of compression, so the
            // round trip is exact.
            let actual = from_proto(&to_proto(&data));
            prop_assert_eq!(actual, data);
        }
    }
}
