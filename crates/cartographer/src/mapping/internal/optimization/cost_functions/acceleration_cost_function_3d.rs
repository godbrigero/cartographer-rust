//! Port of `cartographer/mapping/internal/optimization/cost_functions/acceleration_cost_function_3d` (.h).
//!
//! Note: in C++ this class lives in namespace `cartographer::mapping` (not
//! `mapping::optimization`).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
use nalgebra::{Quaternion, Vector3};

use crate::transform::rigid_transform::{quaternion_mul, quaternion_rotate};

/// Penalizes differences between IMU data and optimized accelerations.
///
/// 3 residuals; parameter blocks: middle rotation `[w, x, y, z]`, start
/// position, middle position, end position, gravity constant (1), IMU
/// calibration `[w, x, y, z]`.
pub struct AccelerationCostFunction3D {
    scaling_factor: f64,
    delta_velocity_imu_frame: Vector3<f64>,
    first_delta_time_seconds: f64,
    second_delta_time_seconds: f64,
}

impl AccelerationCostFunction3D {
    /// Creates the autodiff cost function.
    pub fn create_auto_diff_cost_function(
        scaling_factor: f64,
        delta_velocity_imu_frame: &Vector3<f64>,
        first_delta_time_seconds: f64,
        second_delta_time_seconds: f64,
    ) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 18>::new(
            AccelerationCostFunction3D {
                scaling_factor,
                delta_velocity_imu_frame: *delta_velocity_imu_frame,
                first_delta_time_seconds,
                second_delta_time_seconds,
            },
            3, /* residuals */
            &[
                4, /* rotation variables */
                3, /* position variables */
                3, /* position variables */
                3, /* position variables */
                1, /* gravity variables */
                4, /* rotation variables */
            ],
        ))
    }
}

/// `Eigen::Quaternion<T>(q[0], q[1], q[2], q[3])` (w, x, y, z).
fn to_eigen<T: Scalar>(quaternion: &[T]) -> Quaternion<T> {
    Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
}

fn to_vector<T: Scalar>(v: &[T]) -> Vector3<T> {
    Vector3::new(v[0], v[1], v[2])
}

impl CostFunctor for AccelerationCostFunction3D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residual: &mut [T]) -> bool {
        let (
            middle_rotation,
            start_position,
            middle_position,
            end_position,
            gravity_constant,
            imu_calibration,
        ) = (
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
            parameters[5],
        );
        let eigen_imu_calibration = to_eigen(imu_calibration);
        let rotated_delta_velocity = quaternion_rotate(
            &quaternion_mul(&to_eigen(middle_rotation), &eigen_imu_calibration),
            &self.delta_velocity_imu_frame.map(T::constant),
        );
        let gravity_delta: Vector3<T> = (Vector3::z()
            * (0.5 * (self.first_delta_time_seconds + self.second_delta_time_seconds)))
            .map(T::constant);
        let imu_delta_velocity =
            rotated_delta_velocity - gravity_delta.map(|x| gravity_constant[0] * x);
        let start_velocity = (to_vector(middle_position) - to_vector(start_position))
            / T::constant(self.first_delta_time_seconds);
        let end_velocity = (to_vector(end_position) - to_vector(middle_position))
            / T::constant(self.second_delta_time_seconds);
        let delta_velocity = end_velocity - start_velocity;

        let result = (imu_delta_velocity - delta_velocity) * T::constant(self.scaling_factor);
        residual[..3].copy_from_slice(result.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{
        check_jacobians_numerically, evaluate, random_unit_quaternion, random_vector,
    };
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn zero_residual_for_consistent_motion() {
        // Constant acceleration a = (1, 0, 0) plus gravity, identity rotations:
        // positions 0, 0.5, 2 at t = 0, 1, 2 => delta velocity (1, 0, 0) over
        // 0.5 * (1 + 1) = 1 s. The IMU measures a + g * z.
        let cost_function = AccelerationCostFunction3D::create_auto_diff_cost_function(
            2.,
            &Vector3::new(1., 0., 9.8),
            1.,
            1.,
        );
        let (residuals, _) = evaluate(
            cost_function.as_ref(),
            &[
                &[1., 0., 0., 0.],
                &[0., 0., 0.],
                &[0.5, 0., 0.],
                &[2., 0., 0.],
                &[9.8],
                &[1., 0., 0., 0.],
            ],
        );
        for r in residuals {
            assert!(r.abs() < 1e-12, "{r}");
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..20 {
            let cost_function = AccelerationCostFunction3D::create_auto_diff_cost_function(
                rng.random_range(0.5..5.0),
                &Vector3::from_vec(random_vector(&mut rng, 3, 10.)),
                rng.random_range(0.1..2.0),
                rng.random_range(0.1..2.0),
            );
            let parameters = vec![
                random_unit_quaternion(&mut rng),
                random_vector(&mut rng, 3, 5.),
                random_vector(&mut rng, 3, 5.),
                random_vector(&mut rng, 3, 5.),
                vec![rng.random_range(9.0..10.5)],
                random_unit_quaternion(&mut rng),
            ];
            check_jacobians_numerically(cost_function.as_ref(), &parameters, 1e-5);
        }
    }
}
