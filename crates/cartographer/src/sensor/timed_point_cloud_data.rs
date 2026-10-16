//! Port of `cartographer/sensor/timed_point_cloud_data` (.h, .cc).

use nalgebra::Vector3;

use crate::common::time::{Time, from_universal, to_universal};
use crate::sensor::data::HasTime;
use crate::sensor::point_cloud::TimedPointCloud;
use crate::sensor::proto;
use crate::sensor::rangefinder_point::{TimedRangefinderPoint, timed_from_proto, timed_to_proto};
use crate::transform::transform::{ToEigen, ToProto};

/// A timed point cloud as received from a range sensor. `time` is the time
/// of the last point; point times are relative to it (<= 0).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimedPointCloudData {
    /// Time of the last point.
    pub time: Time,
    /// Sensor origin.
    pub origin: Vector3<f32>,
    /// The points.
    pub ranges: TimedPointCloud,
    /// `intensities` has to be same size as `ranges`, or empty.
    pub intensities: Vec<f32>,
}

impl HasTime for TimedPointCloudData {
    fn time(&self) -> Time {
        self.time
    }
}

/// A range measurement with the index of its origin (C++
/// `TimedPointCloudOriginData::RangeMeasurement`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RangeMeasurement {
    /// Point and its relative time.
    pub point_time: TimedRangefinderPoint,
    /// Intensity.
    pub intensity: f32,
    /// Index into `TimedPointCloudOriginData::origins`.
    pub origin_index: usize,
}

/// Merged range data of several origins (produced by the
/// `RangeDataCollator`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimedPointCloudOriginData {
    /// Reference time; point times are relative to it.
    pub time: Time,
    /// The sensor origins.
    pub origins: Vec<Vector3<f32>>,
    /// The measurements.
    pub ranges: Vec<RangeMeasurement>,
}

impl HasTime for TimedPointCloudOriginData {
    fn time(&self) -> Time {
        self.time
    }
}

/// Converts `timed_point_cloud_data` to a `proto::TimedPointCloudData`
/// (the legacy field is never written).
pub fn to_proto(timed_point_cloud_data: &TimedPointCloudData) -> proto::TimedPointCloudData {
    proto::TimedPointCloudData {
        timestamp: to_universal(timed_point_cloud_data.time),
        origin: Some(timed_point_cloud_data.origin.to_proto()),
        point_data: timed_point_cloud_data
            .ranges
            .iter()
            .map(timed_to_proto)
            .collect(),
        intensities: timed_point_cloud_data.intensities.clone(),
        ..Default::default()
    }
}

/// Converts `proto` to `TimedPointCloudData`. Falls back to the legacy
/// `point_data_legacy` field if `point_data` is empty.
///
/// Like in C++, panics unless `intensities` is empty or has the size of the
/// (new) `point_data` field - so legacy data with intensities is rejected.
pub fn from_proto(proto: &proto::TimedPointCloudData) -> TimedPointCloudData {
    assert!(
        proto.intensities.is_empty() || proto.intensities.len() == proto.point_data.len(),
        "Check failed: proto.intensities().size() == 0 || proto.intensities().size() == \
         proto.point_data().size()"
    );
    let timed_point_cloud: TimedPointCloud = if !proto.point_data.is_empty() {
        proto.point_data.iter().map(timed_from_proto).collect()
    } else {
        proto
            .point_data_legacy
            .iter()
            .map(|timed_point_proto| {
                let timed_point = timed_point_proto.to_eigen();
                TimedRangefinderPoint {
                    position: timed_point.xyz(),
                    time: timed_point[3],
                }
            })
            .collect()
    };
    TimedPointCloudData {
        time: from_universal(proto.timestamp),
        origin: proto.origin.unwrap_or_default().to_eigen(),
        ranges: timed_point_cloud,
        intensities: proto.intensities.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proto_round_trip(
            t in any::<i64>(),
            origin in prop::array::uniform3(-1e3f32..1e3),
            points in prop::collection::vec((prop::array::uniform3(-1e3f32..1e3), -1f32..0.), 0..20),
            with_intensities in any::<bool>(),
        ) {
            let ranges: TimedPointCloud = points.iter().map(|(p, time)| TimedRangefinderPoint {
                position: Vector3::from(*p), time: *time }).collect();
            let intensities = if with_intensities {
                (0..ranges.len()).map(|i| i as f32).collect()
            } else {
                Vec::new()
            };
            let data = TimedPointCloudData {
                time: from_universal(t),
                origin: Vector3::from(origin),
                ranges,
                intensities,
            };
            let proto = to_proto(&data);
            prop_assert!(proto.point_data_legacy.is_empty());
            prop_assert_eq!(from_proto(&proto), data);
        }
    }

    #[test]
    fn from_legacy_proto() {
        let proto = proto::TimedPointCloudData {
            timestamp: 42,
            origin: Some(Vector3::new(1f32, 2., 3.).to_proto()),
            point_data_legacy: vec![
                Vector4::new(1f32, 2., 3., -0.5).to_proto(),
                Vector4::new(4f32, 5., 6., 0.).to_proto(),
            ],
            ..Default::default()
        };
        let data = from_proto(&proto);
        assert_eq!(data.time, from_universal(42));
        assert_eq!(data.origin, Vector3::new(1., 2., 3.));
        assert_eq!(
            data.ranges,
            vec![
                TimedRangefinderPoint {
                    position: Vector3::new(1., 2., 3.),
                    time: -0.5
                },
                TimedRangefinderPoint {
                    position: Vector3::new(4., 5., 6.),
                    time: 0.
                },
            ]
        );
        assert!(data.intensities.is_empty());
    }

    #[test]
    #[should_panic]
    fn legacy_proto_with_intensities_panics() {
        let proto = proto::TimedPointCloudData {
            point_data_legacy: vec![Vector4::new(1f32, 2., 3., 0.).to_proto()],
            intensities: vec![1.],
            ..Default::default()
        };
        let _ = from_proto(&proto);
    }
}
