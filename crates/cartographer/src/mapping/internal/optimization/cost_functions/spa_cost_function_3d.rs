//! Port of `cartographer/mapping/internal/optimization/cost_functions/spa_cost_function_3d` (.h).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};

use super::cost_helpers::{compute_unscaled_error_3d, scale_error_3d};
use crate::mapping::pose_graph_interface::Pose;

/// The 3D SPA cost: 6 residuals; parameter blocks `c_i` rotation
/// `[w, x, y, z]`, `c_i` translation, `c_j` rotation, `c_j` translation.
pub struct SpaCostFunction3D {
    pose: Pose,
}

impl SpaCostFunction3D {
    /// Creates the autodiff cost function for the constraint `pose`.
    pub fn create_auto_diff_cost_function(pose: &Pose) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 14>::new(
            SpaCostFunction3D { pose: *pose },
            6, /* residuals */
            &[
                4, /* rotation variables */
                3, /* translation variables */
                4, /* rotation variables */
                3, /* translation variables */
            ],
        ))
    }
}

impl CostFunctor for SpaCostFunction3D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], e: &mut [T]) -> bool {
        let (c_i_rotation, c_i_translation, c_j_rotation, c_j_translation) =
            (parameters[0], parameters[1], parameters[2], parameters[3]);
        let error = scale_error_3d(
            &compute_unscaled_error_3d(
                &self.pose.zbar_ij,
                c_i_rotation,
                c_i_translation,
                c_j_rotation,
                c_j_translation,
            ),
            self.pose.translation_weight,
            self.pose.rotation_weight,
        );
        e[..6].copy_from_slice(&error);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{
        check_jacobians_numerically, evaluate, random_unit_quaternion, random_vector,
    };
    use super::*;
    use crate::transform::rigid_transform::Rigid3d;
    use nalgebra::{Quaternion, Vector3};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn zero_residual_at_constraint() {
        let pose = Pose {
            zbar_ij: Rigid3d::new(
                Vector3::new(1., 2., 3.),
                Quaternion::new(0.8, 0.0, 0.6, 0.0),
            ),
            translation_weight: 2.,
            rotation_weight: 3.,
        };
        let cost = SpaCostFunction3D::create_auto_diff_cost_function(&pose);
        let (residuals, _) = evaluate(
            cost.as_ref(),
            &[
                &[1., 0., 0., 0.],
                &[0., 0., 0.],
                &[0.8, 0.0, 0.6, 0.0],
                &[1., 2., 3.],
            ],
        );
        for r in residuals {
            assert!(r.abs() < 1e-12, "{r}");
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            let q = random_unit_quaternion(&mut rng);
            let pose = Pose {
                zbar_ij: Rigid3d::new(
                    Vector3::new(
                        rng.random_range(-3.0..3.0),
                        rng.random_range(-3.0..3.0),
                        rng.random_range(-3.0..3.0),
                    ),
                    Quaternion::new(q[0], q[1], q[2], q[3]),
                ),
                translation_weight: rng.random_range(0.5..10.0),
                rotation_weight: rng.random_range(0.5..10.0),
            };
            let cost = SpaCostFunction3D::create_auto_diff_cost_function(&pose);
            let parameters = vec![
                random_unit_quaternion(&mut rng),
                random_vector(&mut rng, 3, 5.),
                random_unit_quaternion(&mut rng),
                random_vector(&mut rng, 3, 5.),
            ];
            check_jacobians_numerically(cost.as_ref(), &parameters, 1e-5);
        }
    }
}
