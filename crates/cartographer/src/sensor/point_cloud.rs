//! Port of `cartographer/sensor/point_cloud` (.h, .cc).

use std::ops::Index;

use crate::sensor::rangefinder_point::{RangefinderPoint, TimedRangefinderPoint};
use crate::transform::rigid_transform::Rigid3f;

/// Stores 3D positions of points together with some additional data, e.g.
/// intensities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
    // For 2D points, the third entry is 0.f.
    points: Vec<RangefinderPoint>,
    // Intensities are optional. If non-empty, they must have the same size as
    // points.
    intensities: Vec<f32>,
}

/// C++ `PointCloud::PointType`.
pub type PointType = RangefinderPoint;

impl PointCloud {
    /// Creates an empty point cloud.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a point cloud without intensities.
    pub fn from_points(points: Vec<RangefinderPoint>) -> Self {
        PointCloud {
            points,
            intensities: Vec::new(),
        }
    }

    /// Creates a point cloud with intensities, which must be empty or have the
    /// same size as `points`.
    pub fn with_intensities(points: Vec<RangefinderPoint>, intensities: Vec<f32>) -> Self {
        if !intensities.is_empty() {
            assert_eq!(
                points.len(),
                intensities.len(),
                "Check failed: points_.size() == intensities_.size()"
            );
        }
        PointCloud {
            points,
            intensities,
        }
    }

    /// Returns the number of points in the point cloud.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Same as [`PointCloud::size`].
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Checks whether there are any points in the point cloud.
    pub fn empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Same as [`PointCloud::empty`].
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The points.
    pub fn points(&self) -> &[RangefinderPoint] {
        &self.points
    }

    /// The (optional) intensities: empty or of the same size as the points.
    pub fn intensities(&self) -> &[f32] {
        &self.intensities
    }

    /// Iterator over the points in the point cloud.
    pub fn iter(&self) -> std::slice::Iter<'_, RangefinderPoint> {
        self.points.iter()
    }

    /// Appends a point. Like in C++, this does not touch the intensities, so
    /// it must only be used on clouds without intensities to keep the size
    /// invariant (C++ `push_back`).
    pub fn push_back(&mut self, value: RangefinderPoint) {
        // RUST PORT NOTE: the C++ silently breaks the size invariant here;
        // we flag it in debug builds.
        debug_assert!(
            self.intensities.is_empty(),
            "push_back on a PointCloud with intensities"
        );
        self.points.push(value);
    }

    /// Creates a `PointCloud` consisting of all the points for which
    /// `predicate` returns true, together with the corresponding intensities.
    pub fn copy_if<P: FnMut(&RangefinderPoint) -> bool>(&self, mut predicate: P) -> PointCloud {
        let mut points = Vec::new();
        let mut intensities = Vec::new();
        // Note: benchmarks show that it is better to have this conditional
        // outside the loop.
        if self.intensities.is_empty() {
            for point in &self.points {
                if predicate(point) {
                    points.push(*point);
                }
            }
        } else {
            for (point, intensity) in self.points.iter().zip(&self.intensities) {
                if predicate(point) {
                    points.push(*point);
                    intensities.push(*intensity);
                }
            }
        }
        PointCloud::with_intensities(points, intensities)
    }

    /// Consumes the cloud, returning `(points, intensities)`.
    pub fn into_parts(self) -> (Vec<RangefinderPoint>, Vec<f32>) {
        (self.points, self.intensities)
    }
}

impl From<Vec<RangefinderPoint>> for PointCloud {
    fn from(points: Vec<RangefinderPoint>) -> Self {
        PointCloud::from_points(points)
    }
}

impl Index<usize> for PointCloud {
    type Output = RangefinderPoint;
    fn index(&self, index: usize) -> &RangefinderPoint {
        &self.points[index]
    }
}

impl<'a> IntoIterator for &'a PointCloud {
    type Item = &'a RangefinderPoint;
    type IntoIter = std::slice::Iter<'a, RangefinderPoint>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

/// Stores 3D positions of points with their relative measurement time in the
/// fourth entry. Time is in seconds, increasing and relative to the moment when
/// the last point was acquired. So, the fourth entry for the last point is 0.f.
/// If timing is not available, all fourth entries are 0.f. For 2D points, the
/// third entry is 0.f (and the fourth entry is time).
pub type TimedPointCloud = Vec<TimedRangefinderPoint>;

/// Retained for cartographer_ros (C++ TODO: to be removed once it is no
/// longer used there).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloudWithIntensities {
    /// The points.
    pub points: TimedPointCloud,
    /// The intensities.
    pub intensities: Vec<f32>,
}

/// Transforms `point_cloud` according to `transform` (keeps intensities).
pub fn transform_point_cloud(point_cloud: &PointCloud, transform: &Rigid3f) -> PointCloud {
    let points = point_cloud.points().iter().map(|p| transform * p).collect();
    PointCloud::with_intensities(points, point_cloud.intensities().to_vec())
}

/// Transforms `point_cloud` according to `transform` (keeps times).
pub fn transform_timed_point_cloud(
    point_cloud: &[TimedRangefinderPoint],
    transform: &Rigid3f,
) -> TimedPointCloud {
    point_cloud.iter().map(|p| transform * p).collect()
}

/// Returns a new point cloud without points that fall outside the region
/// defined by `min_z` and `max_z` (both inclusive).
pub fn crop_point_cloud(point_cloud: &PointCloud, min_z: f32, max_z: f32) -> PointCloud {
    point_cloud.copy_if(|point| min_z <= point.position.z && point.position.z <= max_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::Rigid2f;
    use crate::transform::transform::embed_3d;
    use nalgebra::Vector3;

    fn p(x: f32, y: f32, z: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, y, z),
        }
    }

    mod point_cloud_test {
        use super::*;

        #[test]
        fn transform_point_cloud() {
            let mut point_cloud = PointCloud::new();
            point_cloud.push_back(p(0.5, 0.5, 1.));
            point_cloud.push_back(p(3.5, 0.5, 42.));
            let transformed_point_cloud = super::super::transform_point_cloud(
                &point_cloud,
                &embed_3d(&Rigid2f::from_rotation(std::f32::consts::FRAC_PI_2)),
            );
            assert!((-0.5 - transformed_point_cloud[0].position.x).abs() <= 1e-6);
            assert!((0.5 - transformed_point_cloud[0].position.y).abs() <= 1e-6);
            assert!((-0.5 - transformed_point_cloud[1].position.x).abs() <= 1e-6);
            assert!((3.5 - transformed_point_cloud[1].position.y).abs() <= 1e-6);
        }

        #[test]
        fn transform_timed_point_cloud() {
            let point_cloud: TimedPointCloud = vec![
                TimedRangefinderPoint {
                    position: Vector3::new(0.5, 0.5, 1.),
                    time: 0.,
                },
                TimedRangefinderPoint {
                    position: Vector3::new(3.5, 0.5, 42.),
                    time: 0.,
                },
            ];
            let transformed_point_cloud = super::super::transform_timed_point_cloud(
                &point_cloud,
                &embed_3d(&Rigid2f::from_rotation(std::f32::consts::FRAC_PI_2)),
            );
            assert!((-0.5 - transformed_point_cloud[0].position.x).abs() <= 1e-6);
            assert!((0.5 - transformed_point_cloud[0].position.y).abs() <= 1e-6);
            assert!((-0.5 - transformed_point_cloud[1].position.x).abs() <= 1e-6);
            assert!((3.5 - transformed_point_cloud[1].position.y).abs() <= 1e-6);
        }

        #[test]
        fn copy_if() {
            let points = vec![p(0., 0., 0.), p(1., 1., 1.), p(2., 2., 2.)];
            let point_cloud = PointCloud::from_points(points);
            let copied_point_cloud = point_cloud.copy_if(|point| point.position.x > 0.1);
            assert_eq!(copied_point_cloud.size(), 2);
            assert!(copied_point_cloud.intensities().is_empty());
            assert_eq!(copied_point_cloud.points(), &[p(1., 1., 1.), p(2., 2., 2.)]);
        }

        #[test]
        fn copy_if_with_intensities() {
            let points = vec![p(0., 0., 0.), p(1., 1., 1.), p(2., 2., 2.)];
            let intensities = vec![0., 1., 2.];
            let point_cloud = PointCloud::with_intensities(points, intensities);
            let copied_point_cloud = point_cloud.copy_if(|point| point.position.x > 0.1);
            assert_eq!(copied_point_cloud.size(), 2);
            assert_eq!(copied_point_cloud.intensities().len(), 2);
            assert_eq!(copied_point_cloud.points(), &[p(1., 1., 1.), p(2., 2., 2.)]);
            assert!((copied_point_cloud.intensities()[0] - 1.).abs() <= 1e-6);
            assert!((copied_point_cloud.intensities()[1] - 2.).abs() <= 1e-6);
        }
    }

    #[test]
    fn crop_point_cloud_is_inclusive_and_keeps_intensities() {
        let cloud = PointCloud::with_intensities(
            vec![p(0., 0., -1.), p(0., 0., 0.), p(0., 0., 1.), p(0., 0., 2.)],
            vec![10., 11., 12., 13.],
        );
        let cropped = crop_point_cloud(&cloud, 0., 1.);
        assert_eq!(cropped.points(), &[p(0., 0., 0.), p(0., 0., 1.)]);
        assert_eq!(cropped.intensities(), &[11., 12.]);
    }

    #[test]
    #[should_panic]
    fn mismatched_intensities_panic() {
        let _ = PointCloud::with_intensities(vec![p(0., 0., 0.)], vec![1., 2.]);
    }
}
