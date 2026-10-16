//! Port of `cartographer/io/points_batch` (.h, .cc).

use std::collections::HashSet;

use nalgebra::Vector3;

use crate::common::time::Time;
use crate::io::color::FloatColor;
use crate::sensor::rangefinder_point::RangefinderPoint;

/// A number of points, captured around the same `time` and by a sensor at the
/// same `origin`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointsBatch {
    /// Time at which the first point of this batch has been acquired.
    pub start_time: Time,
    /// Origin of the data, i.e. the location of the sensor in the world at
    /// `time`.
    pub origin: Vector3<f32>,
    /// Sensor that generated this data's `frame_id` or empty if this
    /// information is unknown.
    pub frame_id: String,
    /// Trajectory ID that produced this point.
    pub trajectory_id: i32,
    /// Geometry of the points in the map frame.
    pub points: Vec<RangefinderPoint>,
    /// Intensities are optional and may be unspecified. The meaning of these
    /// intensity values varies by device. For example, the VLP16 provides
    /// values in the range [0, 100] for non-specular return values and values
    /// up to 255 for specular returns. On the other hand, Hokuyo lasers
    /// provide a 16-bit value that rarely peaks above 4096.
    pub intensities: Vec<f32>,
    /// Colors are optional. If set, they are RGB values.
    pub colors: Vec<FloatColor>,
}

/// Removes the indices in `to_remove` from `batch`.
///
/// `intensities` and `colors` are filtered as well if they are non-empty (in
/// which case they must have as many entries as `points`).
///
/// RUST PORT NOTE: the C++ set holds `int`s; indices are `usize` here.
pub fn remove_points(to_remove: &HashSet<usize>, batch: &mut PointsBatch) {
    let new_num_points = batch.points.len().saturating_sub(to_remove.len());
    let mut points = Vec::with_capacity(new_num_points);
    let mut intensities = Vec::new();
    if !batch.intensities.is_empty() {
        intensities.reserve(new_num_points);
    }
    let mut colors = Vec::new();
    if !batch.colors.is_empty() {
        colors.reserve(new_num_points);
    }
    for i in 0..batch.points.len() {
        if to_remove.contains(&i) {
            continue;
        }
        points.push(batch.points[i]);
        if !batch.colors.is_empty() {
            colors.push(batch.colors[i]);
        }
        if !batch.intensities.is_empty() {
            intensities.push(batch.intensities[i]);
        }
    }
    batch.points = points;
    batch.intensities = intensities;
    batch.colors = colors;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, 0.0, 0.0),
        }
    }

    #[test]
    fn remove_points_keeps_order_and_parallel_arrays() {
        let mut batch = PointsBatch {
            points: (0..5).map(|i| point(i as f32)).collect(),
            intensities: (0..5).map(|i| i as f32 * 10.0).collect(),
            colors: (0..5).map(|i| [i as f32, 0.0, 0.0]).collect(),
            ..Default::default()
        };
        remove_points(&[1, 3].into_iter().collect(), &mut batch);
        assert_eq!(batch.points, vec![point(0.0), point(2.0), point(4.0)]);
        assert_eq!(batch.intensities, vec![0.0, 20.0, 40.0]);
        assert_eq!(
            batch.colors,
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn remove_points_without_optional_data() {
        let mut batch = PointsBatch {
            points: (0..3).map(|i| point(i as f32)).collect(),
            ..Default::default()
        };
        remove_points(&[0].into_iter().collect(), &mut batch);
        assert_eq!(batch.points, vec![point(1.0), point(2.0)]);
        assert!(batch.intensities.is_empty());
        assert!(batch.colors.is_empty());
    }
}
