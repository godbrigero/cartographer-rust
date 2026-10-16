//! Port of `cartographer/mapping/internal/2d/scan_matching/rotation_delta_cost_functor_2d` (.h).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};

/// Computes the cost of rotating `pose` to `target_angle`. Cost increases
/// with the solution's distance from `target_angle` (no angle wrapping).
#[derive(Clone, Copy, Debug)]
pub struct RotationDeltaCostFunctor2D {
    scaling_factor: f64,
    angle: f64,
}

impl RotationDeltaCostFunctor2D {
    /// Creates the autodiff cost function (1 residual, one 3-dimensional
    /// pose block `[x, y, theta]`).
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        target_angle: f64,
    ) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 3>::new(
            Self {
                scaling_factor,
                angle: target_angle,
            },
            1,    // residuals
            &[3], // pose variables
        ))
    }
}

impl CostFunctor for RotationDeltaCostFunctor2D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let pose = parameters[0];
        residual[0] = T::constant(self.scaling_factor) * (pose[2] - T::constant(self.angle));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Not in C++ (no test there): residual and Jacobian.
    #[test]
    fn residual_and_jacobian() {
        let cost_function = RotationDeltaCostFunctor2D::create_auto_diff_cost_function(4.0, 0.25);
        let mut residuals = [0.0; 1];
        let mut jacobian = [0.0; 3];
        assert!(cost_function.evaluate(
            &[&[1.0, 2.0, 0.75]],
            &mut residuals,
            Some(&mut [Some(&mut jacobian[..])]),
        ));
        assert_eq!(residuals, [2.0]);
        assert_eq!(jacobian, [0.0, 0.0, 4.0]);
    }
}
