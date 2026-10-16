//! Port of `cartographer/sensor/rangefinder_point` (.h).

use std::ops::Mul;

use nalgebra::Vector3;

use crate::sensor::proto;
use crate::transform::rigid_transform::Rigid3f;
use crate::transform::transform::{ToEigen, ToProto};

/// Stores 3D position of a point observed by a rangefinder sensor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RangefinderPoint {
    /// Position of the point.
    pub position: Vector3<f32>,
}

/// Stores 3D position of a point with its relative measurement time.
/// See `point_cloud` for more details.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimedRangefinderPoint {
    /// Position of the point.
    pub position: Vector3<f32>,
    /// Measurement time in seconds, relative to the last point of the cloud.
    pub time: f32,
}

// RUST PORT NOTE: the C++ `operator*` is templated over `Rigid3<T>`, but the
// position is an `Eigen::Vector3f`, so only `T = float` is ever used.
impl Mul<RangefinderPoint> for Rigid3f {
    type Output = RangefinderPoint;
    fn mul(self, rhs: RangefinderPoint) -> RangefinderPoint {
        RangefinderPoint {
            position: self * rhs.position,
        }
    }
}

impl Mul<&RangefinderPoint> for &Rigid3f {
    type Output = RangefinderPoint;
    fn mul(self, rhs: &RangefinderPoint) -> RangefinderPoint {
        *self * *rhs
    }
}

impl Mul<TimedRangefinderPoint> for Rigid3f {
    type Output = TimedRangefinderPoint;
    fn mul(self, rhs: TimedRangefinderPoint) -> TimedRangefinderPoint {
        TimedRangefinderPoint {
            position: self * rhs.position,
            time: rhs.time,
        }
    }
}

impl Mul<&TimedRangefinderPoint> for &Rigid3f {
    type Output = TimedRangefinderPoint;
    fn mul(self, rhs: &TimedRangefinderPoint) -> TimedRangefinderPoint {
        *self * *rhs
    }
}

/// Converts a `proto::RangefinderPoint` (C++ `FromProto`). A missing
/// position is the zero vector, like the proto3 default instance in C++.
pub fn from_proto(rangefinder_point_proto: &proto::RangefinderPoint) -> RangefinderPoint {
    RangefinderPoint {
        position: rangefinder_point_proto
            .position
            .unwrap_or_default()
            .to_eigen(),
    }
}

/// Converts to a `proto::RangefinderPoint` (C++ `ToProto`).
pub fn to_proto(rangefinder_point: &RangefinderPoint) -> proto::RangefinderPoint {
    proto::RangefinderPoint {
        position: Some(rangefinder_point.position.to_proto()),
    }
}

/// Converts a `proto::TimedRangefinderPoint` (C++ `FromProto` overload).
pub fn timed_from_proto(
    timed_rangefinder_point_proto: &proto::TimedRangefinderPoint,
) -> TimedRangefinderPoint {
    TimedRangefinderPoint {
        position: timed_rangefinder_point_proto
            .position
            .unwrap_or_default()
            .to_eigen(),
        time: timed_rangefinder_point_proto.time,
    }
}

/// Converts to a `proto::TimedRangefinderPoint` (C++ `ToProto` overload).
pub fn timed_to_proto(
    timed_rangefinder_point: &TimedRangefinderPoint,
) -> proto::TimedRangefinderPoint {
    proto::TimedRangefinderPoint {
        position: Some(timed_rangefinder_point.position.to_proto()),
        time: timed_rangefinder_point.time,
    }
}

/// Drops the time of a timed point (C++ `ToRangefinderPoint`).
pub fn to_rangefinder_point(timed_rangefinder_point: &TimedRangefinderPoint) -> RangefinderPoint {
    RangefinderPoint {
        position: timed_rangefinder_point.position,
    }
}

/// Adds a time to a point (C++ `ToTimedRangefinderPoint`).
pub fn to_timed_rangefinder_point(
    rangefinder_point: &RangefinderPoint,
    time: f32,
) -> TimedRangefinderPoint {
    TimedRangefinderPoint {
        position: rangefinder_point.position,
        time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn finite() -> impl Strategy<Value = f32> {
        -1e6f32..1e6f32
    }

    proptest! {
        #[test]
        fn rangefinder_point_proto_round_trip(x in finite(), y in finite(), z in finite()) {
            let point = RangefinderPoint { position: Vector3::new(x, y, z) };
            prop_assert_eq!(from_proto(&to_proto(&point)), point);
        }

        #[test]
        fn timed_rangefinder_point_proto_round_trip(
            x in finite(), y in finite(), z in finite(), t in -1.0f32..0.0f32
        ) {
            let point = TimedRangefinderPoint { position: Vector3::new(x, y, z), time: t };
            prop_assert_eq!(timed_from_proto(&timed_to_proto(&point)), point);
            prop_assert_eq!(
                to_timed_rangefinder_point(&to_rangefinder_point(&point), t),
                point
            );
        }
    }

    #[test]
    fn transform_keeps_time() {
        let transform = Rigid3f::from_translation(Vector3::new(1., 2., 3.));
        let point = TimedRangefinderPoint {
            position: Vector3::new(1., 1., 1.),
            time: -0.5,
        };
        let result = transform * point;
        assert_eq!(Mul::mul(&transform, &point), result);
        assert_eq!(result.position, Vector3::new(2., 3., 4.));
        assert_eq!(result.time, -0.5);
    }

    #[test]
    fn missing_position_is_zero() {
        assert_eq!(
            from_proto(&proto::RangefinderPoint::default()),
            RangefinderPoint::default()
        );
    }
}
