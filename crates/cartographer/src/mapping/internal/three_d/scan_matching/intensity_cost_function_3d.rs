//! Port of `cartographer/mapping/internal/3d/scan_matching/intensity_cost_function_3d` (.h, .cc).

use ceres::{AutoDiffCostFunction, CostFunctor, Scalar};

use crate::mapping::internal::three_d::scan_matching::interpolated_grid::InterpolatedIntensityGrid;
use crate::mapping::internal::three_d::scan_matching::occupied_space_cost_function_3d::transform_from_parameters;
use crate::mapping::three_d::hybrid_grid::IntensityHybridGrid;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::Rigid3;

/// Computes the cost of the intensities of `point_cloud` (transformed by a
/// `translation` (3) and `rotation` (4, `(w, x, y, z)`)) against the
/// interpolated intensities of an [`IntensityHybridGrid`]. Returns with an
/// intensity above `intensity_threshold` get a zero residual.
pub struct IntensityCostFunction3D<'a> {
    scaling_factor: f64,
    // We will ignore returns with intensity above this threshold.
    intensity_threshold: f32,
    point_cloud: &'a PointCloud,
    interpolated_grid: InterpolatedIntensityGrid<'a>,
}

impl<'a> IntensityCostFunction3D<'a> {
    /// Creates the autodiff cost function (residuals: `point_cloud.size()`,
    /// parameter blocks: translation (3), rotation (4)). Panics if the
    /// point cloud has no intensities.
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        intensity_threshold: f32,
        point_cloud: &'a PointCloud,
        hybrid_grid: &'a IntensityHybridGrid,
    ) -> AutoDiffCostFunction<Self, 7> {
        assert!(
            !point_cloud.intensities().is_empty(),
            "Check failed: !point_cloud.intensities().empty()"
        );
        AutoDiffCostFunction::new(
            Self {
                scaling_factor,
                intensity_threshold,
                point_cloud,
                interpolated_grid: InterpolatedIntensityGrid::new(hybrid_grid),
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
        let intensities = self.point_cloud.intensities();
        for (i, point) in self.point_cloud.points().iter().enumerate() {
            if intensities[i] > self.intensity_threshold {
                residual[i] = T::constant(0.);
            } else {
                let point = point.position.map(|c| T::constant(f64::from(c)));
                let intensity = T::constant(f64::from(intensities[i]));
                let world = *transform * point;
                let interpolated_intensity = self
                    .interpolated_grid
                    .get_interpolated_value(world[0], world[1], world[2]);
                residual[i] = scaling_factor * (interpolated_intensity - intensity);
            }
        }
        true
    }
}

impl CostFunctor for IntensityCostFunction3D<'_> {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residuals: &mut [T]) -> bool {
        let transform = transform_from_parameters(parameters[0], parameters[1]);
        self.evaluate_transform(&transform, residuals)
    }
}

#[cfg(test)]
mod tests {
    mod intensity_cost_function_3d_test {
        use ceres::CostFunction;
        use nalgebra::Vector3;

        use crate::mapping::internal::three_d::scan_matching::intensity_cost_function_3d::IntensityCostFunction3D;
        use crate::mapping::three_d::hybrid_grid::IntensityHybridGrid;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        #[test]
        fn smoke_test() {
            let point = |x: f32, y: f32, z: f32| RangefinderPoint {
                position: Vector3::new(x, y, z),
            };
            let point_cloud = PointCloud::with_intensities(
                vec![point(0., 0., 0.), point(1., 1., 1.), point(2., 2., 2.)],
                vec![50., 100., 150.],
            );
            let mut hybrid_grid = IntensityHybridGrid::new(0.3);
            let index = hybrid_grid.get_cell_index(&Vector3::new(0., 0., 0.));
            hybrid_grid.add_intensity(&index, 50.);

            let cost_function = IntensityCostFunction3D::create_auto_diff_cost_function(
                /*scaling_factor=*/ 1.0,
                /*intensity_threshold=*/ 100.,
                &point_cloud,
                &hybrid_grid,
            );

            let translation = [0., 0., 0.];
            let rotation = [1., 0., 0., 0.];
            let mut residuals = [0.; 3];
            assert!(cost_function.evaluate(&[&translation, &rotation], &mut residuals, None));
            for (residual, expected) in residuals.iter().zip([0., -100., 0.]) {
                assert!((residual - expected).abs() <= 1e-9, "{residuals:?}");
            }
        }
    }
}
