//! Port of `cartographer/mapping/internal/3d/scan_matching/translation_delta_cost_functor_3d` (.h).

use ceres::{AutoDiffCostFunction, CostFunctor, Scalar};
use nalgebra::Vector3;

/// Computes the cost of translating `translation` to `target_translation`.
/// Cost increases with the solution's distance from `target_translation`.
pub struct TranslationDeltaCostFunctor3D {
    scaling_factor: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl TranslationDeltaCostFunctor3D {
    /// Creates the autodiff cost function (3 residuals, one translation
    /// parameter block of size 3).
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        target_translation: &Vector3<f64>,
    ) -> AutoDiffCostFunction<Self, 3> {
        AutoDiffCostFunction::new(
            Self {
                scaling_factor,
                x: target_translation.x,
                y: target_translation.y,
                z: target_translation.z,
            },
            3, /* residuals */
            &[3 /* translation variables */],
        )
    }
}

impl CostFunctor for TranslationDeltaCostFunctor3D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let translation = parameters[0];
        let scaling_factor = T::constant(self.scaling_factor);
        residual[0] = scaling_factor * (translation[0] - T::constant(self.x));
        residual[1] = scaling_factor * (translation[1] - T::constant(self.y));
        residual[2] = scaling_factor * (translation[2] - T::constant(self.z));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ceres::CostFunction;

    #[test]
    fn computes_scaled_difference_and_jacobian() {
        let cost = TranslationDeltaCostFunctor3D::create_auto_diff_cost_function(
            2.,
            &Vector3::new(1., -2., 3.),
        );
        let mut residuals = [0.; 3];
        let mut jacobian = [0.; 9];
        {
            let mut jacobians = [Some(&mut jacobian[..])];
            assert!(cost.evaluate(&[&[2., 0., 3.5]], &mut residuals, Some(&mut jacobians)));
        }
        assert_eq!(residuals, [2., 4., 1.]);
        assert_eq!(jacobian, [2., 0., 0., 0., 2., 0., 0., 0., 2.]);
    }
}
