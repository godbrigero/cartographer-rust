//! Port of `cartographer/mapping/internal/2d/scan_matching/translation_delta_cost_functor_2d` (.h).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
use nalgebra::Vector2;

/// Computes the cost of translating `pose` to `target_translation`.
/// Cost increases with the solution's distance from `target_translation`.
#[derive(Clone, Copy, Debug)]
pub struct TranslationDeltaCostFunctor2D {
    scaling_factor: f64,
    x: f64,
    y: f64,
}

impl TranslationDeltaCostFunctor2D {
    /// Creates the autodiff cost function (2 residuals, one 3-dimensional
    /// pose block `[x, y, theta]`).
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        target_translation: &Vector2<f64>,
    ) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 3>::new(
            Self::new(scaling_factor, target_translation),
            2,    // residuals
            &[3], // pose variables
        ))
    }

    // Constructs a new TranslationDeltaCostFunctor2D from the given
    // 'target_translation' (x, y).
    fn new(scaling_factor: f64, target_translation: &Vector2<f64>) -> Self {
        Self {
            scaling_factor,
            x: target_translation.x,
            y: target_translation.y,
        }
    }
}

impl CostFunctor for TranslationDeltaCostFunctor2D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let pose = parameters[0];
        residual[0] = T::constant(self.scaling_factor) * (pose[0] - T::constant(self.x));
        residual[1] = T::constant(self.scaling_factor) * (pose[1] - T::constant(self.y));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Not in C++ (no test there): residuals and Jacobian.
    #[test]
    fn residuals_and_jacobian() {
        let cost_function = TranslationDeltaCostFunctor2D::create_auto_diff_cost_function(
            2.0,
            &Vector2::new(1.0, -1.0),
        );
        let mut residuals = [0.0; 2];
        let mut jacobian = [0.0; 6];
        assert!(cost_function.evaluate(
            &[&[1.5, -2.0, 0.3]],
            &mut residuals,
            Some(&mut [Some(&mut jacobian[..])]),
        ));
        assert_eq!(residuals, [1.0, -2.0]);
        assert_eq!(jacobian, [2.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }
}
