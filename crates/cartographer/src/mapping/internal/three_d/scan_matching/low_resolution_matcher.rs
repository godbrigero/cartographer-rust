//! Port of `cartographer/mapping/internal/3d/scan_matching/low_resolution_matcher` (.h, .cc).

use crate::mapping::three_d::hybrid_grid::HybridGrid;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid3f;

/// Used to compute scores between 0 and 1 how well the given pose matches
/// (C++ `std::function<float(const transform::Rigid3f&)>`).
pub type MatchingFunction<'a> = dyn Fn(&Rigid3f) -> f32 + Send + Sync + 'a;

/// Returns a function scoring a pose by the mean probability of the
/// `low_resolution_grid` voxels hit by `points` (nearest voxel, unknown is
/// `MIN_PROBABILITY`). An empty point cloud scores NaN, as in C++.
///
/// RUST PORT NOTE: C++ captures raw pointers; the closure borrows the grid
/// and points instead. Points are transformed on the fly (C++ materializes
/// `TransformPointCloud`), which gives identical results without allocating.
pub fn create_low_resolution_matcher<'a>(
    low_resolution_grid: &'a HybridGrid,
    points: &'a PointCloud,
) -> Box<MatchingFunction<'a>> {
    Box::new(move |pose: &Rigid3f| {
        let mut score = 0.0f32;
        for point in points {
            // TODO(zhengj, whess): Interpolate the Grid to get better score.
            let world = pose * &point.position;
            score +=
                low_resolution_grid.get_probability(&low_resolution_grid.get_cell_index(&world));
        }
        score / points.size() as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::three_d::hybrid_grid::Array3i;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use nalgebra::Vector3;

    #[test]
    fn mean_probability_of_hit_voxels() {
        let mut grid = HybridGrid::new(0.5);
        grid.set_probability(&Array3i::new(2, 0, 0), 0.9);
        let points = PointCloud::from_points(vec![
            RangefinderPoint {
                position: Vector3::new(0., 0., 0.),
            },
            RangefinderPoint {
                position: Vector3::new(0., 1., 0.),
            },
        ]);
        let matcher = create_low_resolution_matcher(&grid, &points);
        let identity_score = matcher(&Rigid3f::identity());
        assert!((identity_score - 0.1).abs() < 1e-6);
        let score = matcher(&Rigid3f::from_translation(Vector3::new(1., 0., 0.)));
        assert!((score - 0.5).abs() < 1e-4, "{score}");
        let empty = PointCloud::new();
        assert!(create_low_resolution_matcher(&grid, &empty)(&Rigid3f::identity()).is_nan());
    }
}
