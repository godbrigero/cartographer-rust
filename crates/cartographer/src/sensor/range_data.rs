//! Port of `cartographer/sensor/range_data` (.h, .cc).

use nalgebra::Vector3;

use crate::sensor::point_cloud::{PointCloud, crop_point_cloud, transform_point_cloud};
use crate::sensor::proto;
use crate::sensor::rangefinder_point::{self, RangefinderPoint};
use crate::transform::rigid_transform::Rigid3f;
use crate::transform::transform::{ToEigen, ToProto};

/// Rays begin at `origin`. `returns` are the points where obstructions were
/// detected. `misses` are points in the direction of rays for which no return
/// was detected, and were inserted at a configured distance. It is assumed that
/// between the `origin` and `misses` is free space.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RangeData {
    /// Origin of the rays.
    pub origin: Vector3<f32>,
    /// Points where obstructions were detected.
    pub returns: PointCloud,
    /// Points along rays without a return.
    pub misses: PointCloud,
}

/// Transforms origin, returns and misses of `range_data` by `transform`.
pub fn transform_range_data(range_data: &RangeData, transform: &Rigid3f) -> RangeData {
    RangeData {
        origin: transform * &range_data.origin,
        returns: transform_point_cloud(&range_data.returns, transform),
        misses: transform_point_cloud(&range_data.misses, transform),
    }
}

/// Crops `range_data` according to the region defined by `min_z` and `max_z`
/// (the origin is kept).
pub fn crop_range_data(range_data: &RangeData, min_z: f32, max_z: f32) -> RangeData {
    RangeData {
        origin: range_data.origin,
        returns: crop_point_cloud(&range_data.returns, min_z, max_z),
        misses: crop_point_cloud(&range_data.misses, min_z, max_z),
    }
}

/// Converts `range_data` to a `proto::RangeData`. Intensities are dropped and
/// the legacy fields are never written.
pub fn to_proto(range_data: &RangeData) -> proto::RangeData {
    proto::RangeData {
        origin: Some(range_data.origin.to_proto()),
        returns: range_data
            .returns
            .iter()
            .map(rangefinder_point::to_proto)
            .collect(),
        misses: range_data
            .misses
            .iter()
            .map(rangefinder_point::to_proto)
            .collect(),
        ..Default::default()
    }
}

fn points_from_proto(
    points: &[proto::RangefinderPoint],
    legacy: &[crate::transform::proto::Vector3f],
) -> Vec<RangefinderPoint> {
    if !points.is_empty() {
        points.iter().map(rangefinder_point::from_proto).collect()
    } else {
        legacy
            .iter()
            .map(|point_proto| RangefinderPoint {
                position: point_proto.to_eigen(),
            })
            .collect()
    }
}

/// Converts `proto` to `RangeData`. For returns and misses independently, the
/// legacy field is read if the new one is empty.
pub fn from_proto(proto: &proto::RangeData) -> RangeData {
    let returns = points_from_proto(&proto.returns, &proto.returns_legacy);
    let misses = points_from_proto(&proto.misses, &proto.misses_legacy);
    RangeData {
        origin: proto.origin.unwrap_or_default().to_eigen(),
        returns: PointCloud::from_points(returns),
        misses: PointCloud::from_points(misses),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // C++ `range_data_test.cc` only contains the fixture `RangeDataTest`
    // (origin (1,1,1), returns (0,1,2),(4,5,6),(0,1,2), misses (7,8,9)) and no
    // TEST_F; it is used here for round-trip tests.
    fn fixture() -> RangeData {
        let p = |x: f32, y: f32, z: f32| RangefinderPoint {
            position: Vector3::new(x, y, z),
        };
        RangeData {
            origin: Vector3::new(1., 1., 1.),
            returns: PointCloud::from_points(vec![p(0., 1., 2.), p(4., 5., 6.), p(0., 1., 2.)]),
            misses: PointCloud::from_points(vec![p(7., 8., 9.)]),
        }
    }

    #[test]
    fn to_and_from_proto() {
        let range_data = fixture();
        let proto = to_proto(&range_data);
        assert!(proto.returns_legacy.is_empty());
        assert!(proto.misses_legacy.is_empty());
        assert_eq!(from_proto(&proto), range_data);
    }

    #[test]
    fn from_legacy_proto() {
        let range_data = fixture();
        let v = |p: &RangefinderPoint| p.position.to_proto();
        let proto = proto::RangeData {
            origin: Some(range_data.origin.to_proto()),
            returns_legacy: range_data.returns.iter().map(v).collect(),
            misses_legacy: range_data.misses.iter().map(v).collect(),
            ..Default::default()
        };
        assert_eq!(from_proto(&proto), range_data);
    }

    #[test]
    fn intensities_are_dropped() {
        let mut range_data = fixture();
        range_data.returns = PointCloud::with_intensities(
            range_data.returns.points().to_vec(),
            vec![1., 2., 3.],
        );
        let decoded = from_proto(&to_proto(&range_data));
        assert!(decoded.returns.intensities().is_empty());
        assert_eq!(decoded.returns.points(), range_data.returns.points());
    }

    #[test]
    fn transform_and_crop() {
        let range_data = fixture();
        let transformed =
            transform_range_data(&range_data, &Rigid3f::from_translation(Vector3::new(0., 0., 1.)));
        assert_eq!(transformed.origin, Vector3::new(1., 1., 2.));
        assert_eq!(transformed.returns[1].position, Vector3::new(4., 5., 7.));
        let cropped = crop_range_data(&range_data, 0., 6.);
        assert_eq!(cropped.origin, range_data.origin);
        assert_eq!(cropped.returns.size(), 3);
        assert_eq!(cropped.misses.size(), 0);
    }

    proptest! {
        #[test]
        fn proto_round_trip(
            origin in prop::array::uniform3(-1e3f32..1e3),
            returns in prop::collection::vec(prop::array::uniform3(-1e3f32..1e3), 0..20),
            misses in prop::collection::vec(prop::array::uniform3(-1e3f32..1e3), 0..20),
        ) {
            let to_cloud = |v: &[[f32; 3]]| PointCloud::from_points(
                v.iter().map(|a| RangefinderPoint { position: Vector3::from(*a) }).collect());
            let range_data = RangeData {
                origin: Vector3::from(origin),
                returns: to_cloud(&returns),
                misses: to_cloud(&misses),
            };
            prop_assert_eq!(from_proto(&to_proto(&range_data)), range_data);
        }
    }
}
