//! Port of `cartographer/mapping/internal/3d/scan_matching/rotation_delta_cost_functor_3d` (.h).

use ceres::{AutoDiffCostFunction, CostFunctor, Scalar};
use nalgebra::Quaternion;

use crate::common::math::quaternion_product;

/// Computes the cost of rotating `rotation_quaternion` to `target_rotation`.
/// Cost increases with the solution's distance from `target_rotation`.
pub struct RotationDeltaCostFunctor3D {
    scaling_factor: f64,
    // The conjugate (not a normalized inverse) of the target, `(w, x, y, z)`.
    target_rotation_inverse: [f64; 4],
}

impl RotationDeltaCostFunctor3D {
    /// Creates the autodiff cost function (3 residuals, one rotation
    /// parameter block of size 4, `(w, x, y, z)`).
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        target_rotation: &Quaternion<f64>,
    ) -> AutoDiffCostFunction<Self, 4> {
        AutoDiffCostFunction::new(
            Self {
                scaling_factor,
                target_rotation_inverse: [
                    target_rotation.w,
                    -target_rotation.i,
                    -target_rotation.j,
                    -target_rotation.k,
                ],
            },
            3, /* residuals */
            &[4 /* rotation variables */],
        )
    }
}

impl CostFunctor for RotationDeltaCostFunctor3D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let q = parameters[0];
        let rotation_quaternion = [q[0], q[1], q[2], q[3]];
        let mut delta = [T::constant(0.); 4];
        quaternion_product(
            &self.target_rotation_inverse,
            &rotation_quaternion,
            &mut delta,
        );
        // Will compute the squared norm of the imaginary component of the
        // delta quaternion which is sin(phi/2)^2.
        let scaling_factor = T::constant(self.scaling_factor);
        residual[0] = scaling_factor * delta[1];
        residual[1] = scaling_factor * delta[2];
        residual[2] = scaling_factor * delta[3];
        true
    }
}

#[cfg(test)]
mod tests {
    mod rotation_delta_cost_functor_3d_test {
        use ceres::CostFunction;
        use nalgebra::{Quaternion, Vector3};

        use crate::mapping::internal::three_d::scan_matching::rotation_delta_cost_functor_3d::RotationDeltaCostFunctor3D;
        use crate::transform::rigid_transform::{AngleAxis, quaternion_identity, quaternion_mul};

        const PRECISION: f64 = 1e-8;

        fn compute_rotation_delta_squared_cost(
            rotation: &Quaternion<f64>,
            scaling_factor: f64,
            target_rotation: &Quaternion<f64>,
        ) -> f64 {
            let cost_function = RotationDeltaCostFunctor3D::create_auto_diff_cost_function(
                scaling_factor,
                target_rotation,
            );
            let parameter_quaternion = [rotation.w, rotation.i, rotation.j, rotation.k];
            let mut residuals = vec![0.; cost_function.num_residuals()];
            assert!(cost_function.evaluate(&[&parameter_quaternion], &mut residuals, None));
            residuals.iter().map(|r| r * r).sum()
        }

        fn near(a: f64, b: f64) {
            assert!((a - b).abs() <= PRECISION, "{a} vs {b}");
        }

        #[test]
        fn same_rotation_gives_zero_cost() {
            near(
                0.,
                compute_rotation_delta_squared_cost(
                    &quaternion_identity(),
                    1.0,
                    &quaternion_identity(),
                ),
            );
            let rotation =
                AngleAxis::new(0.9, Vector3::new(0.2, 0.1, 0.3).normalize()).to_quaternion();
            near(
                0.,
                compute_rotation_delta_squared_cost(&rotation, 1.0, &rotation),
            );
        }

        #[test]
        fn computes_correct_cost() {
            let scaling_factor = 1.2;
            let angle: f64 = 0.8;
            let rotation =
                AngleAxis::new(angle, Vector3::new(0.2, 0.1, 0.8).normalize()).to_quaternion();
            let target_rotation =
                AngleAxis::new(0.2, Vector3::new(-0.5, 0.3, 0.4).normalize()).to_quaternion();
            let expected_cost = (scaling_factor * (angle / 2.0).sin()).powi(2);
            near(
                expected_cost,
                compute_rotation_delta_squared_cost(
                    &rotation,
                    scaling_factor,
                    &quaternion_identity(),
                ),
            );
            near(
                expected_cost,
                compute_rotation_delta_squared_cost(
                    &quaternion_mul(&target_rotation, &rotation),
                    scaling_factor,
                    &target_rotation,
                ),
            );
            near(
                expected_cost,
                compute_rotation_delta_squared_cost(
                    &quaternion_mul(&rotation, &target_rotation),
                    scaling_factor,
                    &target_rotation,
                ),
            );
        }
    }
}
