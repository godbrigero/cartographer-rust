//! Port of `cartographer/mapping/internal/optimization/cost_functions/landmark_cost_function_2d` (.h).

use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
use nalgebra::Quaternion;

use super::cost_helpers::{compute_unscaled_error_3d, interpolate_nodes_2d, scale_error_3d};
use crate::common::time::to_seconds;
use crate::mapping::internal::optimization::optimization_problem_2d::NodeSpec2D;
use crate::mapping::pose_graph_interface::LandmarkObservation;
use crate::transform::rigid_transform::Rigid3d;

/// Cost function measuring the weighted error between the observed pose
/// given by the landmark measurement and the linearly interpolated pose of
/// embedded in 3D space node poses.
///
/// 6 residuals; parameter blocks: previous node pose `[x, y, theta]`, next
/// node pose `[x, y, theta]`, landmark rotation `[w, x, y, z]`, landmark
/// translation `[x, y, z]`.
pub struct LandmarkCostFunction2D {
    landmark_to_tracking_transform: Rigid3d,
    prev_node_gravity_alignment: Quaternion<f64>,
    next_node_gravity_alignment: Quaternion<f64>,
    translation_weight: f64,
    rotation_weight: f64,
    interpolation_parameter: f64,
}

impl LandmarkCostFunction2D {
    /// Creates the autodiff cost function for `observation` between the
    /// nodes `prev_node` and `next_node`.
    pub fn create_auto_diff_cost_function(
        observation: &LandmarkObservation,
        prev_node: &NodeSpec2D,
        next_node: &NodeSpec2D,
    ) -> Box<dyn CostFunction> {
        Box::new(AutoDiffCostFunction::<_, 13>::new(
            LandmarkCostFunction2D::new(observation, prev_node, next_node),
            6, /* residuals */
            &[
                3, /* previous node pose variables */
                3, /* next node pose variables */
                4, /* landmark rotation variables */
                3, /* landmark translation variables */
            ],
        ))
    }

    fn new(
        observation: &LandmarkObservation,
        prev_node: &NodeSpec2D,
        next_node: &NodeSpec2D,
    ) -> Self {
        LandmarkCostFunction2D {
            landmark_to_tracking_transform: observation.landmark_to_tracking_transform,
            prev_node_gravity_alignment: prev_node.gravity_alignment,
            next_node_gravity_alignment: next_node.gravity_alignment,
            translation_weight: observation.translation_weight,
            rotation_weight: observation.rotation_weight,
            interpolation_parameter: to_seconds(observation.time - prev_node.time)
                / to_seconds(next_node.time - prev_node.time),
        }
    }
}

impl CostFunctor for LandmarkCostFunction2D {
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], e: &mut [T]) -> bool {
        let (prev_node_pose, next_node_pose, landmark_rotation, landmark_translation) =
            (parameters[0], parameters[1], parameters[2], parameters[3]);
        let (interpolated_rotation, interpolated_translation) = interpolate_nodes_2d(
            prev_node_pose,
            &self.prev_node_gravity_alignment,
            next_node_pose,
            &self.next_node_gravity_alignment,
            self.interpolation_parameter,
        );
        let error = scale_error_3d(
            &compute_unscaled_error_3d(
                &self.landmark_to_tracking_transform,
                &interpolated_rotation,
                &interpolated_translation,
                landmark_rotation,
                landmark_translation,
            ),
            self.translation_weight,
            self.rotation_weight,
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
    use crate::common::time::{Time, from_universal};
    use crate::transform::rigid_transform::Rigid2d;
    use approx::assert_ulps_eq;
    use nalgebra::Vector3;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn node(time: Time, gravity_alignment: Quaternion<f64>) -> NodeSpec2D {
        NodeSpec2D {
            time,
            local_pose_2d: Rigid2d::identity(),
            global_pose_2d: Rigid2d::identity(),
            gravity_alignment,
        }
    }

    mod landmark_cost_function_test {
        use super::*;

        #[test]
        fn smoke_test() {
            let prev_node = node(from_universal(0), Quaternion::identity());
            let next_node = node(from_universal(10), Quaternion::identity());

            let cost_function = LandmarkCostFunction2D::create_auto_diff_cost_function(
                &LandmarkObservation {
                    trajectory_id: 0,
                    time: from_universal(5),
                    landmark_to_tracking_transform: Rigid3d::from_translation(Vector3::new(
                        1., 1., 1.,
                    )),
                    translation_weight: 1.,
                    rotation_weight: 2.,
                },
                &prev_node,
                &next_node,
            );

            let prev_node_pose = [2., 0., 0.];
            let next_node_pose = [0., 2., 0.];
            let landmark_rotation = [1., 0., 0., 0.];
            let landmark_translation = [1., 2., 1.];
            let (residuals, _jacobians) = evaluate(
                cost_function.as_ref(),
                &[
                    &prev_node_pose,
                    &next_node_pose,
                    &landmark_rotation,
                    &landmark_translation,
                ],
            );
            let expected = [1., 0., 0., 0., 0., 0.];
            for (&actual, &expected) in residuals.iter().zip(&expected) {
                // gmock DoubleEq: within 4 ULPs.
                assert_ulps_eq!(actual, expected, max_ulps = 4);
            }
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let gravity = |rng: &mut StdRng| {
                let q = random_unit_quaternion(rng);
                Quaternion::new(q[0], q[1], q[2], q[3])
            };
            let prev_node = node(from_universal(0), gravity(&mut rng));
            let next_node = node(from_universal(100), gravity(&mut rng));
            let q = random_unit_quaternion(&mut rng);
            let observation = LandmarkObservation {
                trajectory_id: 0,
                time: from_universal(rng.random_range(0..=100)),
                landmark_to_tracking_transform: Rigid3d::new(
                    Vector3::from_vec(random_vector(&mut rng, 3, 2.)),
                    Quaternion::new(q[0], q[1], q[2], q[3]),
                ),
                translation_weight: rng.random_range(0.5..5.0),
                rotation_weight: rng.random_range(0.5..5.0),
            };
            let cost_function = LandmarkCostFunction2D::create_auto_diff_cost_function(
                &observation,
                &prev_node,
                &next_node,
            );
            let parameters = vec![
                random_vector(&mut rng, 3, 3.),
                random_vector(&mut rng, 3, 3.),
                random_unit_quaternion(&mut rng),
                random_vector(&mut rng, 3, 3.),
            ];
            check_jacobians_numerically(cost_function.as_ref(), &parameters, 1e-5);
        }
    }
}
