//! Port of `cartographer/mapping/internal/optimization/cost_functions/rotation_cost_function_3d` (.h).
//!
//! Note: in C++ this class lives in namespace `cartographer::mapping` (not
//! `mapping::optimization`).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
use nalgebra::Quaternion;

use crate::transform::rigid_transform::{quaternion_cast, quaternion_conjugate, quaternion_mul};

/// Penalizes differences between IMU data and optimized orientations.
///
/// 3 residuals; parameter blocks: start rotation, end rotation and IMU
/// calibration, all `[w, x, y, z]`.
pub struct RotationCostFunction3D {
    scaling_factor: f64,
    delta_rotation_imu_frame: Quaternion<f64>,
}

impl RotationCostFunction3D {
    /// Creates the autodiff cost function.
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        delta_rotation_imu_frame: &Quaternion<f64>,
    ) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 12>::new(
            RotationCostFunction3D {
                scaling_factor,
                delta_rotation_imu_frame: *delta_rotation_imu_frame,
            },
            3, /* residuals */
            &[
                4, /* rotation variables */
                4, /* rotation variables */
                4, /* rotation variables */
            ],
        ))
    }
}

impl CostFunctor for RotationCostFunction3D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let (start_rotation, end_rotation, imu_calibration) =
            (parameters[0], parameters[1], parameters[2]);
        let start = Quaternion::new(
            start_rotation[0],
            start_rotation[1],
            start_rotation[2],
            start_rotation[3],
        );
        let end = Quaternion::new(
            end_rotation[0],
            end_rotation[1],
            end_rotation[2],
            end_rotation[3],
        );
        let eigen_imu_calibration = Quaternion::new(
            imu_calibration[0],
            imu_calibration[1],
            imu_calibration[2],
            imu_calibration[3],
        );
        // end.conjugate() * start * eigen_imu_calibration *
        // delta_rotation_imu_frame_.cast<T>() * eigen_imu_calibration.conjugate()
        let error = quaternion_mul(
            &quaternion_mul(
                &quaternion_mul(
                    &quaternion_mul(&quaternion_conjugate(&end), &start),
                    &eigen_imu_calibration,
                ),
                &quaternion_cast::<f64, T>(&self.delta_rotation_imu_frame),
            ),
            &quaternion_conjugate(&eigen_imu_calibration),
        );
        let scaling_factor = T::constant(self.scaling_factor);
        residual[0] = scaling_factor * error.i;
        residual[1] = scaling_factor * error.j;
        residual[2] = scaling_factor * error.k;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{check_jacobians_numerically, evaluate, random_unit_quaternion};
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn zero_residual_for_consistent_rotation() {
        // start * delta == end with identity calibration.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let delta = Quaternion::new(s, 0., 0., s); // 90 degrees about z.
        let cost_function = RotationCostFunction3D::create_auto_diff_cost_function(3., &delta);
        let (residuals, _) = evaluate(
            cost_function.as_ref(),
            &[&[1., 0., 0., 0.], &[s, 0., 0., s], &[1., 0., 0., 0.]],
        );
        for r in residuals {
            assert!(r.abs() < 1e-15, "{r}");
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..20 {
            let q = random_unit_quaternion(&mut rng);
            let cost_function = RotationCostFunction3D::create_auto_diff_cost_function(
                rng.random_range(0.5..5.0),
                &Quaternion::new(q[0], q[1], q[2], q[3]),
            );
            let parameters = vec![
                random_unit_quaternion(&mut rng),
                random_unit_quaternion(&mut rng),
                random_unit_quaternion(&mut rng),
            ];
            check_jacobians_numerically(cost_function.as_ref(), &parameters, 1e-5);
        }
    }
}
