//! Port of `cartographer/mapping/internal/2d/scan_matching/occupied_space_cost_function_2d` (.h, .cc).

use ceres::{AutoDiffCostFunction, BiCubicInterpolator, CostFunction, CostFunctor, Grid2, Scalar};

use crate::mapping::probability_values::MAX_CORRESPONDENCE_COST;
use crate::mapping::two_d::grid_2d::{Grid2D, Grid2DTrait};
use crate::mapping::two_d::xy_index::Array2i;
use crate::sensor::point_cloud::PointCloud;

/// Padding of the interpolated grid so that the arguments of `floor()` in
/// the bicubic interpolator stay positive (C++ `INT_MAX / 4`).
const PADDING: i32 = i32::MAX / 4;

/// Exposes a [`Grid2D`]'s correspondence costs, padded by [`PADDING`] cells
/// of `MAX_CORRESPONDENCE_COST` on every side, to the Ceres bicubic
/// interpolator (rows = cell y index, columns = cell x index).
struct GridArrayAdapter<'a> {
    grid: &'a Grid2D,
}

impl GridArrayAdapter<'_> {
    fn num_rows(&self) -> i32 {
        self.grid.limits().cell_limits().num_y_cells + 2 * PADDING
    }

    fn num_cols(&self) -> i32 {
        self.grid.limits().cell_limits().num_x_cells + 2 * PADDING
    }
}

impl Grid2<1> for GridArrayAdapter<'_> {
    #[inline]
    fn get_value(&self, row: i32, column: i32, value: &mut [f64; 1]) {
        if row < PADDING
            || column < PADDING
            || row >= self.num_rows() - PADDING
            || column >= self.num_cols() - PADDING
        {
            value[0] = f64::from(MAX_CORRESPONDENCE_COST);
        } else {
            value[0] = f64::from(
                self.grid
                    .get_correspondence_cost(&Array2i::new(column - PADDING, row - PADDING)),
            );
        }
    }
}

/// Computes a cost for matching the `point_cloud` to the `grid` with a
/// `pose`. The cost increases with poorer correspondence of the grid and the
/// point observation (e.g. points falling into less occupied space).
struct OccupiedSpaceCostFunction2D<'a> {
    scaling_factor: f64,
    point_cloud: &'a PointCloud,
    grid: &'a Grid2D,
}

impl CostFunctor for OccupiedSpaceCostFunction2D<'_> {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let pose = parameters[0];
        let (sin, cos) = (pose[2].sin(), pose[2].cos());
        let interpolator = BiCubicInterpolator::new(GridArrayAdapter { grid: self.grid });
        let limits = self.grid.limits();
        let max_x = T::constant(limits.max().x);
        let max_y = T::constant(limits.max().y);
        let resolution = T::constant(limits.resolution());
        let half = T::constant(0.5);
        let padding = T::constant(f64::from(PADDING));
        let scaling_factor = T::constant(self.scaling_factor);

        for (point, r) in self.point_cloud.iter().zip(residual.iter_mut()) {
            // Note that this is a 2D point. The third (homogeneous)
            // component is 1.
            let x = T::constant(f64::from(point.position.x));
            let y = T::constant(f64::from(point.position.y));
            let world_x = cos * x + -sin * y + pose[0];
            let world_y = sin * x + cos * y + pose[1];
            let mut value = [T::zero()];
            interpolator.evaluate(
                &((max_x - world_x) / resolution - half + padding),
                &((max_y - world_y) / resolution - half + padding),
                &mut value,
            );
            *r = scaling_factor * value[0];
        }
        true
    }
}

/// Creates a cost function for matching the `point_cloud` to the `grid`
/// with a pose `[x, y, theta]`; one residual per point, the interpolated
/// correspondence cost times `scaling_factor`. The cost function borrows
/// `point_cloud` and `grid`.
pub fn create_occupied_space_cost_function_2d<'a>(
    scaling_factor: f64,
    point_cloud: &'a PointCloud,
    grid: &'a dyn Grid2DTrait,
) -> Box<dyn CostFunction + 'a> {
    Box::new(AutoDiffCostFunction::<_, 3>::new(
        OccupiedSpaceCostFunction2D {
            scaling_factor,
            point_cloud,
            grid: grid.grid_2d(),
        },
        point_cloud.size(), // residuals
        &[3],               // pose variables
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::{Vector2, Vector3};

    use super::*;
    use crate::mapping::probability_values::MAX_PROBABILITY;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    mod occupied_space_cost_function_2d_test {
        use super::*;

        #[test]
        fn smoke_test() {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                &conversion_tables,
            );
            let point_cloud = PointCloud::from_points(vec![RangefinderPoint {
                position: Vector3::new(0.0, 0.0, 0.0),
            }]);
            let cost_function = create_occupied_space_cost_function_2d(1.0, &point_cloud, &grid);

            let pose_estimate = [0.0, 0.0, 0.0];
            let mut residuals = [0.0; 1];
            let mut jacobians = [0.0; 3];
            cost_function.evaluate(
                &[&pose_estimate],
                &mut residuals,
                Some(&mut [Some(&mut jacobians[..])]),
            );

            assert_eq!(residuals, [f64::from(MAX_PROBABILITY)]);
        }

        // Not in C++: the interpolated cost is low at an occupied cell and
        // its gradient points away from it.
        #[test]
        fn cost_and_gradient_near_occupied_cell() {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(10.0, 10.0), CellLimits::new(20, 20)),
                &conversion_tables,
            );
            let cell = grid.limits().get_cell_index(&Vector2::new(-3.5, 2.5));
            grid.set_probability(&cell, MAX_PROBABILITY);
            let point_cloud = PointCloud::from_points(vec![RangefinderPoint {
                position: Vector3::new(-3.0, 2.0, 0.0),
            }]);
            let cost_function = create_occupied_space_cost_function_2d(1.0, &point_cloud, &grid);
            let mut residual = [0.0; 1];
            let mut jacobian = [0.0; 3];
            assert!(cost_function.evaluate(
                &[&[-0.5, 0.5, 0.0]],
                &mut residual,
                Some(&mut [Some(&mut jacobian[..])]),
            ));
            assert!((residual[0] - f64::from(1.0 - MAX_PROBABILITY)).abs() < 1e-6);
            assert!(jacobian[0].abs() < 1e-9 && jacobian[1].abs() < 1e-9);
            assert!(cost_function.evaluate(
                &[&[-0.2, 0.5, 0.0]],
                &mut residual,
                Some(&mut [Some(&mut jacobian[..])]),
            ));
            // Moving further away (+x) increases the cost.
            assert!(jacobian[0] > 0.0);
        }
    }
}
