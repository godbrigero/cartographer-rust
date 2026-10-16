//! Port of `cartographer/sensor/landmark_data` (.h, .cc).

use crate::common::time::{Time, from_universal, to_universal};
use crate::sensor::data::HasTime;
use crate::sensor::proto;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{ToProto, to_rigid3};

/// A single landmark observation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkObservation {
    /// Landmark id.
    ///
    /// RUST PORT NOTE: the proto field is `bytes` (C++ `std::string`); here it
    /// is a `String` and non-UTF-8 ids are converted lossily in
    /// [`from_proto`].
    pub id: String,
    /// Pose of the landmark in the tracking frame.
    pub landmark_to_tracking_transform: Rigid3d,
    /// Weight of the translational part of the observation.
    pub translation_weight: f64,
    /// Weight of the rotational part of the observation.
    pub rotation_weight: f64,
}

/// Landmark observations at a point in time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LandmarkData {
    /// Measurement time.
    pub time: Time,
    /// The observations.
    pub landmark_observations: Vec<LandmarkObservation>,
}

impl HasTime for LandmarkData {
    fn time(&self) -> Time {
        self.time
    }
}

/// Converts `landmark_data` to a `proto::LandmarkData`.
pub fn to_proto(landmark_data: &LandmarkData) -> proto::LandmarkData {
    proto::LandmarkData {
        timestamp: to_universal(landmark_data.time),
        landmark_observations: landmark_data
            .landmark_observations
            .iter()
            .map(|observation| proto::landmark_data::LandmarkObservation {
                id: observation.id.as_bytes().to_vec(),
                landmark_to_tracking_transform: Some(
                    observation.landmark_to_tracking_transform.to_proto(),
                ),
                translation_weight: observation.translation_weight,
                rotation_weight: observation.rotation_weight,
            })
            .collect(),
    }
}

/// Converts `proto` to `LandmarkData`.
pub fn from_proto(proto: &proto::LandmarkData) -> LandmarkData {
    LandmarkData {
        time: from_universal(proto.timestamp),
        landmark_observations: proto
            .landmark_observations
            .iter()
            .map(|item| LandmarkObservation {
                id: String::from_utf8_lossy(&item.id).into_owned(),
                landmark_to_tracking_transform: to_rigid3(
                    &item.landmark_to_tracking_transform.unwrap_or_default(),
                ),
                translation_weight: item.translation_weight,
                rotation_weight: item.rotation_weight,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform_test_helpers::is_nearly;
    use nalgebra::{Quaternion, Vector3};
    use proptest::prelude::*;

    fn equals_landmark(actual: &LandmarkObservation, expected: &LandmarkObservation) {
        assert_eq!(actual.id, expected.id);
        assert!(is_nearly(
            &actual.landmark_to_tracking_transform,
            &expected.landmark_to_tracking_transform,
            1e-2
        ));
        assert!((actual.translation_weight - expected.translation_weight).abs() <= 0.01);
        assert!((actual.rotation_weight - expected.rotation_weight).abs() <= 0.01);
    }

    mod landmark_data_test {
        use super::*;

        fn observations() -> Vec<LandmarkObservation> {
            vec![
                LandmarkObservation {
                    id: "ID1".to_string(),
                    landmark_to_tracking_transform: Rigid3d::new(
                        Vector3::new(1., 1., 1.),
                        Quaternion::new(1., 1., -1., -1.),
                    ),
                    translation_weight: 1.,
                    rotation_weight: 3.,
                },
                LandmarkObservation {
                    id: "ID2".to_string(),
                    landmark_to_tracking_transform: Rigid3d::new(
                        Vector3::new(2., 2., 2.),
                        Quaternion::new(2., 2., -2., -2.),
                    ),
                    translation_weight: 2.,
                    rotation_weight: 4.,
                },
            ]
        }

        #[test]
        fn landmark_data_to_and_from_proto() {
            let expected = LandmarkData {
                time: from_universal(50),
                landmark_observations: observations(),
            };
            let actual = from_proto(&to_proto(&expected));
            assert_eq!(expected.time, actual.time);
            assert_eq!(actual.landmark_observations.len(), 2);
            equals_landmark(
                &actual.landmark_observations[0],
                &expected.landmark_observations[0],
            );
            equals_landmark(
                &actual.landmark_observations[1],
                &expected.landmark_observations[1],
            );
        }
    }

    proptest! {
        #[test]
        fn proto_round_trip(
            t in any::<i64>(),
            observations in prop::collection::vec(
                ("[a-zA-Z0-9_]{0,8}", prop::array::uniform3(-1e3f64..1e3),
                 prop::array::uniform4(-1f64..1.), 0f64..10., 0f64..10.),
                0..5),
        ) {
            let data = LandmarkData {
                time: from_universal(t),
                landmark_observations: observations.into_iter().map(|(id, tr, q, tw, rw)| {
                    LandmarkObservation {
                        id,
                        landmark_to_tracking_transform: Rigid3d::new(
                            Vector3::from(tr), Quaternion::new(q[0], q[1], q[2], q[3])),
                        translation_weight: tw,
                        rotation_weight: rw,
                    }
                }).collect(),
            };
            prop_assert_eq!(from_proto(&to_proto(&data)), data);
        }
    }
}
