//! Port of `cartographer/mapping/internal/optimization/cost_functions/spa_cost_function_2d` (.h, .cc).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar, SizedCostFunction};

use super::cost_helpers::{compute_unscaled_error_2d, scale_error_2d};
use crate::common::math::normalize_angle_difference;
use crate::mapping::pose_graph_interface::Pose;
use crate::transform::rigid_transform::Rigid2d;
use crate::transform::transform::project_2d;

/// Autodiff functor of the 2D SPA cost (C++ `SpaCostFunction2D`, file-local).
struct SpaCostFunction2D {
    observed_relative_pose: Pose,
}

impl CostFunctor for SpaCostFunction2D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], e: &mut [T]) -> bool {
        let (start_pose, end_pose) = (parameters[0], parameters[1]);
        let error = scale_error_2d(
            &compute_unscaled_error_2d(
                &project_2d(&self.observed_relative_pose.zbar_ij),
                start_pose,
                end_pose,
            ),
            self.observed_relative_pose.translation_weight,
            self.observed_relative_pose.rotation_weight,
        );
        e[..3].copy_from_slice(&error);
        true
    }
}

/// 2D SPA cost with an analytical Jacobian (C++
/// `AnalyticalSpaCostFunction2D`, file-local): 3 residuals, parameter blocks
/// start pose `[x, y, theta]` and end pose `[x, y, theta]`.
struct AnalyticalSpaCostFunction2D {
    observed_relative_pose: Rigid2d,
    translation_weight: f64,
    rotation_weight: f64,
}

impl AnalyticalSpaCostFunction2D {
    fn new(constraint_pose: &Pose) -> Self {
        AnalyticalSpaCostFunction2D {
            observed_relative_pose: project_2d(&constraint_pose.zbar_ij),
            translation_weight: constraint_pose.translation_weight,
            rotation_weight: constraint_pose.rotation_weight,
        }
    }
}

impl SizedCostFunction for AnalyticalSpaCostFunction2D {
    const NUM_RESIDUALS: usize = 3;
    const PARAMETER_BLOCK_SIZES: &'static [usize] =
        &[3 /* start pose */, 3 /* end pose */];

    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let start = parameters[0];
        let end = parameters[1];

        let cos_start_rotation = start[2].cos();
        let sin_start_rotation = start[2].sin();
        let delta_x = end[0] - start[0];
        let delta_y = end[1] - start[1];

        residuals[0] = self.translation_weight
            * (self.observed_relative_pose.translation().x
                - (cos_start_rotation * delta_x + sin_start_rotation * delta_y));
        residuals[1] = self.translation_weight
            * (self.observed_relative_pose.translation().y
                - (-sin_start_rotation * delta_x + cos_start_rotation * delta_y));
        residuals[2] = self.rotation_weight
            * normalize_angle_difference(
                self.observed_relative_pose.rotation().angle() - (end[2] - start[2]),
            );
        let Some(jacobians) = jacobians else {
            return true;
        };

        let weighted_cos_start_rotation = self.translation_weight * cos_start_rotation;
        let weighted_sin_start_rotation = self.translation_weight * sin_start_rotation;

        // Jacobians in Ceres are ordered by the parameter blocks:
        // jacobian[i] = [(dr_0 / dx_i)^T, ..., (dr_n / dx_i)^T].
        if let Some(jacobian) = jacobians[0].as_deref_mut() {
            jacobian[0] = weighted_cos_start_rotation;
            jacobian[1] = weighted_sin_start_rotation;
            jacobian[2] =
                weighted_sin_start_rotation * delta_x - weighted_cos_start_rotation * delta_y;
            jacobian[3] = -weighted_sin_start_rotation;
            jacobian[4] = weighted_cos_start_rotation;
            jacobian[5] =
                weighted_cos_start_rotation * delta_x + weighted_sin_start_rotation * delta_y;
            jacobian[6] = 0.;
            jacobian[7] = 0.;
            jacobian[8] = self.rotation_weight;
        }
        if let Some(jacobian) = jacobians[1].as_deref_mut() {
            jacobian[0] = -weighted_cos_start_rotation;
            jacobian[1] = -weighted_sin_start_rotation;
            jacobian[2] = 0.;
            jacobian[3] = weighted_sin_start_rotation;
            jacobian[4] = -weighted_cos_start_rotation;
            jacobian[5] = 0.;
            jacobian[6] = 0.;
            jacobian[7] = 0.;
            jacobian[8] = -self.rotation_weight;
        }
        true
    }
}

/// Creates the 2D SPA cost function with automatic differentiation:
/// 3 residuals, parameter blocks start pose and end pose `[x, y, theta]`.
pub fn create_auto_diff_spa_cost_function(observed_relative_pose: &Pose) -> Box<dyn CostFunction> {
    Box::new(AutoDiffCostFunction::<_, 6>::new(
        SpaCostFunction2D {
            observed_relative_pose: *observed_relative_pose,
        },
        3, /* residuals */
        &[
            3, /* start pose variables */
            3, /* end pose variables */
        ],
    ))
}

/// Creates the 2D SPA cost function with an analytical Jacobian (same
/// residuals and parameter blocks as [`create_auto_diff_spa_cost_function`]).
pub fn create_analytical_spa_cost_function(observed_relative_pose: &Pose) -> Box<dyn CostFunction> {
    Box::new(AnalyticalSpaCostFunction2D::new(observed_relative_pose))
}

#[cfg(test)]
mod tests {
    use super::super::testing::evaluate;
    use super::*;
    use crate::transform::rigid_transform::Rigid3d;
    use nalgebra::{Quaternion, Vector3};
    use proptest::prelude::*;

    const RESIDUALS_COUNT: usize = 3;
    const PARAMETER_BLOCKS_COUNT: usize = 2;
    const JACOBIAN_COL_DIMENSION: usize = 9;

    fn near(actual: f64, expected: f64) {
        const PRECISION: f64 = 1e-05;
        assert!(
            (actual - expected).abs() <= PRECISION,
            "{actual} is not near {expected}"
        );
    }

    /// The `SpaCostFunction2DTest` fixture.
    struct Fixture {
        auto_diff_cost: Box<dyn CostFunction>,
        analytical_cost: Box<dyn CostFunction>,
    }

    impl Fixture {
        fn new() -> Self {
            let constraint = Pose {
                zbar_ij: Rigid3d::new(Vector3::new(1., 1., 1.), Quaternion::new(1., 1., -1., -1.)),
                translation_weight: 1.,
                rotation_weight: 10.,
            };
            Fixture {
                auto_diff_cost: create_auto_diff_spa_cost_function(&constraint),
                analytical_cost: create_analytical_spa_cost_function(&constraint),
            }
        }
    }

    const START_POSE: [f64; 3] = [1., 1., 1.];
    const END_POSE: [f64; 3] = [10., 1., 100.];

    fn check_expected_values(residuals: &[f64], jacobians: &[Vec<f64>]) {
        let expected_residuals = [-3.86272, 8.57324, -6.83333];
        let expected_jacobians = [
            [
                0.540302, 0.841471, 7.57324, -0.841471, 0.540302, 4.86272, 0., 0., 10.,
            ],
            [
                -0.540302, -0.841471, 0., 0.841471, -0.540302, 0., 0., 0., -10.,
            ],
        ];
        assert_eq!(residuals.len(), RESIDUALS_COUNT);
        for (&actual, &expected) in residuals.iter().zip(&expected_residuals) {
            near(actual, expected);
        }
        assert_eq!(jacobians.len(), PARAMETER_BLOCKS_COUNT);
        for (jacobian, expected) in jacobians.iter().zip(&expected_jacobians) {
            assert_eq!(jacobian.len(), JACOBIAN_COL_DIMENSION);
            for (&actual, &expected) in jacobian.iter().zip(expected) {
                near(actual, expected);
            }
        }
    }

    mod spa_cost_function_2d_test {
        use super::*;

        #[test]
        fn compare_auto_diff_and_analytical() {
            let fixture = Fixture::new();
            let parameter_blocks: [&[f64]; 2] = [&START_POSE, &END_POSE];
            let (auto_diff_residual, auto_diff_jacobian) =
                evaluate(fixture.auto_diff_cost.as_ref(), &parameter_blocks);
            let (analytical_residual, analytical_jacobian) =
                evaluate(fixture.analytical_cost.as_ref(), &parameter_blocks);
            for i in 0..RESIDUALS_COUNT {
                near(auto_diff_residual[i], analytical_residual[i]);
            }
            for i in 0..PARAMETER_BLOCKS_COUNT {
                for j in 0..JACOBIAN_COL_DIMENSION {
                    near(auto_diff_jacobian[i][j], analytical_jacobian[i][j]);
                }
            }
        }

        #[test]
        fn evaluate_analytical_cost() {
            let fixture = Fixture::new();
            let (residuals, jacobians) =
                evaluate(fixture.analytical_cost.as_ref(), &[&START_POSE, &END_POSE]);
            check_expected_values(&residuals, &jacobians);
        }

        #[test]
        fn evaluate_auto_diff_cost() {
            let fixture = Fixture::new();
            let (residuals, jacobians) =
                evaluate(fixture.auto_diff_cost.as_ref(), &[&START_POSE, &END_POSE]);
            check_expected_values(&residuals, &jacobians);
        }
    }

    proptest! {
        /// The autodiff and analytical cost functions agree on random poses
        /// and constraints (residuals and Jacobians).
        #[test]
        fn auto_diff_and_analytical_agree(
            start in prop::array::uniform3(-10.0f64..10.0),
            end in prop::array::uniform3(-10.0f64..10.0),
            translation in prop::array::uniform3(-5.0f64..5.0),
            rotation in prop::array::uniform4(-1.0f64..1.0),
            translation_weight in 0.1f64..100.0,
            rotation_weight in 0.1f64..100.0,
        ) {
            let norm = rotation.iter().map(|x| x * x).sum::<f64>().sqrt();
            prop_assume!(norm > 0.1);
            let constraint = Pose {
                zbar_ij: Rigid3d::new(
                    Vector3::from(translation),
                    Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]) / norm,
                ),
                translation_weight,
                rotation_weight,
            };
            let auto_diff = create_auto_diff_spa_cost_function(&constraint);
            let analytical = create_analytical_spa_cost_function(&constraint);
            let blocks: [&[f64]; 2] = [&start, &end];
            let (r_auto, j_auto) = evaluate(auto_diff.as_ref(), &blocks);
            let (r_analytical, j_analytical) = evaluate(analytical.as_ref(), &blocks);
            for (a, b) in r_auto.iter().zip(&r_analytical) {
                prop_assert!((a - b).abs() <= 1e-9 * a.abs().max(1.), "{} vs {}", a, b);
            }
            for (ja, jb) in j_auto.iter().zip(&j_analytical) {
                for (a, b) in ja.iter().zip(jb) {
                    prop_assert!((a - b).abs() <= 1e-9 * a.abs().max(1.), "{} vs {}", a, b);
                }
            }
        }
    }
}
