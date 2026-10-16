//! Port of `cartographer/mapping/internal/3d/scan_matching/occupied_space_cost_function_3d` (.h).

use ceres::{AutoDiffCostFunction, CostFunctor, Scalar};
use nalgebra::{Quaternion, Vector3};

use crate::mapping::internal::three_d::scan_matching::interpolated_grid::InterpolatedProbabilityGrid;
use crate::mapping::three_d::hybrid_grid::HybridGrid;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid3;

/// Computes a cost for matching the `point_cloud` to the `hybrid_grid` with a
/// `translation` (3 parameters) and `rotation` (4 parameters, `(w, x, y, z)`):
/// one residual `scaling_factor * (1 - p)` per point, with `p` the
/// interpolated occupancy probability at the transformed point.
pub struct OccupiedSpaceCostFunction3D<'a> {
    scaling_factor: f64,
    point_cloud: &'a PointCloud,
    interpolated_grid: InterpolatedProbabilityGrid<'a>,
}

impl<'a> OccupiedSpaceCostFunction3D<'a> {
    /// Creates the autodiff cost function (residuals: `point_cloud.size()`,
    /// parameter blocks: translation (3), rotation (4)).
    ///
    /// RUST PORT NOTE: the cost function borrows `point_cloud` and
    /// `hybrid_grid` (C++ holds references that must outlive the problem).
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        point_cloud: &'a PointCloud,
        hybrid_grid: &'a HybridGrid,
    ) -> AutoDiffCostFunction<Self, 7> {
        AutoDiffCostFunction::new(
            Self {
                scaling_factor,
                point_cloud,
                interpolated_grid: InterpolatedProbabilityGrid::new(hybrid_grid),
            },
            point_cloud.size(),
            &[
                3, /* translation variables */
                4, /* rotation variables */
            ],
        )
    }

    fn evaluate_transform<T: Scalar>(&self, transform: &Rigid3<T>, residual: &mut [T]) -> bool {
        let scaling_factor = T::constant(self.scaling_factor);
        for (point, residual) in self.point_cloud.points().iter().zip(residual.iter_mut()) {
            let world = *transform * point.position.map(|c| T::constant(f64::from(c)));
            let probability = self
                .interpolated_grid
                .get_interpolated_value(world[0], world[1], world[2]);
            *residual = scaling_factor * (T::constant(1.) - probability);
        }
        true
    }
}

/// Builds the C++ `Rigid3<T>(Map<translation>, Quaternion<T>(r0, r1, r2, r3))`
/// (the quaternion is not normalized).
#[inline]
pub(crate) fn transform_from_parameters<T: Scalar>(translation: &[T], rotation: &[T]) -> Rigid3<T> {
    Rigid3::new(
        Vector3::new(translation[0], translation[1], translation[2]),
        Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]),
    )
}

impl CostFunctor for OccupiedSpaceCostFunction3D<'_> {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residuals: &mut [T]) -> bool {
        let transform = transform_from_parameters(parameters[0], parameters[1]);
        self.evaluate_transform(&transform, residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::three_d::hybrid_grid::Array3i;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use ceres::CostFunction;

    #[test]
    fn residuals_and_jacobian_shape() {
        let mut grid = HybridGrid::new(0.5);
        grid.set_probability(&Array3i::new(2, 0, 0), 0.9);
        let point_cloud = PointCloud::from_points(vec![
            RangefinderPoint {
                position: Vector3::new(1., 0., 0.),
            },
            RangefinderPoint {
                position: Vector3::new(0., 3., 0.),
            },
        ]);
        let cost =
            OccupiedSpaceCostFunction3D::create_auto_diff_cost_function(2., &point_cloud, &grid);
        assert_eq!(cost.num_residuals(), 2);
        assert_eq!(cost.parameter_block_sizes(), &[3, 4]);
        let mut residuals = [0.; 2];
        assert!(cost.evaluate(&[&[0., 0., 0.], &[1., 0., 0., 0.]], &mut residuals, None));
        assert!(
            (residuals[0] - 2. * (1. - 0.9)).abs() < 1e-6,
            "{residuals:?}"
        );
        assert!(
            (residuals[1] - 2. * (1. - 0.1)).abs() < 1e-6,
            "{residuals:?}"
        );
    }
}
