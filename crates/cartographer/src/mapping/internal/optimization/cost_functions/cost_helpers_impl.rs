//! Port of `cartographer/mapping/internal/optimization/cost_functions/cost_helpers_impl` (.h).
//!
//! Implementations of the helpers declared in
//! [`cost_helpers`](super::cost_helpers). All functions are generic over
//! `T: ceres::Scalar` so they run on `f64` and on `ceres::Jet`s, like the C++
//! templates. Eigen quaternion arithmetic is reproduced with the helpers of
//! [`crate::transform::rigid_transform`] (same formulas and operation order
//! as Eigen).

use ceres::Scalar;
use nalgebra::{Quaternion, Vector3};

use crate::common::math::normalize_angle_difference;
use crate::transform::rigid_transform::{
    AngleAxis, Rigid2d, Rigid3d, quaternion_cast, quaternion_mul, quaternion_normalized,
    quaternion_rotate,
};
use crate::transform::transform::rotation_quaternion_to_angle_axis_vector;

/// Computes the error between the given relative pose and the difference of
/// poses `start` and `end` which are both in an arbitrary common frame.
///
/// `start` and `end` poses have the format `[x, y, rotation]`.
pub fn compute_unscaled_error_2d<T: Scalar>(
    relative_pose: &Rigid2d,
    start: &[T],
    end: &[T],
) -> [T; 3] {
    let cos_theta_i = start[2].cos();
    let sin_theta_i = start[2].sin();
    let delta_x = end[0] - start[0];
    let delta_y = end[1] - start[1];
    let h = [
        cos_theta_i * delta_x + sin_theta_i * delta_y,
        -sin_theta_i * delta_x + cos_theta_i * delta_y,
        end[2] - start[2],
    ];
    [
        T::constant(relative_pose.translation().x) - h[0],
        T::constant(relative_pose.translation().y) - h[1],
        normalize_angle_difference(T::constant(relative_pose.rotation().angle()) - h[2]),
    ]
}

/// Scales a 2D error `[x, y, rotation]` by the translation and rotation
/// weights.
pub fn scale_error_2d<T: Scalar>(
    error: &[T; 3],
    translation_weight: f64,
    rotation_weight: f64,
) -> [T; 3] {
    [
        error[0] * T::constant(translation_weight),
        error[1] * T::constant(translation_weight),
        error[2] * T::constant(rotation_weight),
    ]
}

/// Computes the error between the given relative pose and the difference of
/// poses `start` and `end` which are both in an arbitrary common frame.
///
/// `start` and `end` translation has the format `[x, y, z]`.
/// `start` and `end` rotation are quaternions in the format
/// `[w, n_1, n_2, n_3]`.
pub fn compute_unscaled_error_3d<T: Scalar>(
    relative_pose: &Rigid3d,
    start_rotation: &[T],
    start_translation: &[T],
    end_rotation: &[T],
    end_translation: &[T],
) -> [T; 6] {
    let r_i_inverse = Quaternion::new(
        start_rotation[0],
        -start_rotation[1],
        -start_rotation[2],
        -start_rotation[3],
    );

    let delta = Vector3::new(
        end_translation[0] - start_translation[0],
        end_translation[1] - start_translation[1],
        end_translation[2] - start_translation[2],
    );
    let h_translation = quaternion_rotate(&r_i_inverse, &delta);

    let h_rotation_inverse = quaternion_mul(
        &Quaternion::new(
            end_rotation[0],
            -end_rotation[1],
            -end_rotation[2],
            -end_rotation[3],
        ),
        &Quaternion::new(
            start_rotation[0],
            start_rotation[1],
            start_rotation[2],
            start_rotation[3],
        ),
    );

    let angle_axis_difference = rotation_quaternion_to_angle_axis_vector(&quaternion_mul(
        &h_rotation_inverse,
        &quaternion_cast::<f64, T>(relative_pose.rotation()),
    ));

    [
        T::constant(relative_pose.translation().x) - h_translation[0],
        T::constant(relative_pose.translation().y) - h_translation[1],
        T::constant(relative_pose.translation().z) - h_translation[2],
        angle_axis_difference[0],
        angle_axis_difference[1],
        angle_axis_difference[2],
    ]
}

/// Scales a 3D error `[x, y, z, rx, ry, rz]` by the translation and rotation
/// weights.
pub fn scale_error_3d<T: Scalar>(
    error: &[T; 6],
    translation_weight: f64,
    rotation_weight: f64,
) -> [T; 6] {
    [
        error[0] * T::constant(translation_weight),
        error[1] * T::constant(translation_weight),
        error[2] * T::constant(translation_weight),
        error[3] * T::constant(rotation_weight),
        error[4] * T::constant(rotation_weight),
        error[5] * T::constant(rotation_weight),
    ]
}

/// Computes spherical linear interpolation of unit quaternions.
///
/// `start` and `end` are quaternions in the format `[w, n_1, n_2, n_3]`.
///
/// Eigen implementation of slerp is not compatible with Ceres on all
/// supported platforms. Our own implementation is used instead.
pub fn slerp_quaternions<T: Scalar>(start: &[T], end: &[T], factor: f64) -> [T; 4] {
    // Angle 'theta' is the half-angle "between" quaternions. It can be
    // computed as the arccosine of their dot product.
    let cos_theta = start[0] * end[0] + start[1] * end[1] + start[2] * end[2] + start[3] * end[3];
    let abs_cos_theta = cos_theta.abs();
    // If numerical error brings 'cos_theta' outside [-1 + epsilon, 1 - epsilon]
    // interval, then the quaternions are likely to be collinear.
    let mut prev_scale = T::constant(1. - factor);
    let mut next_scale = T::constant(factor);
    if abs_cos_theta < T::constant(1. - 1e-5) {
        let theta = abs_cos_theta.acos();
        let sin_theta = theta.sin();
        prev_scale = (T::constant(1. - factor) * theta).sin() / sin_theta;
        next_scale = (T::constant(factor) * theta).sin() / sin_theta;
    }
    if cos_theta < T::constant(0.) {
        next_scale = -next_scale;
    }
    [
        prev_scale * start[0] + next_scale * end[0],
        prev_scale * start[1] + next_scale * end[1],
        prev_scale * start[2] + next_scale * end[2],
        prev_scale * start[3] + next_scale * end[3],
    ]
}

/// Interpolates 3D poses. Linear interpolation is performed for translation
/// and spherical-linear one for rotation. Returns `(rotation [w, x, y, z],
/// translation [x, y, z])`.
pub fn interpolate_nodes_3d<T: Scalar>(
    prev_node_rotation: &[T],
    prev_node_translation: &[T],
    next_node_rotation: &[T],
    next_node_translation: &[T],
    interpolation_parameter: f64,
) -> ([T; 4], [T; 3]) {
    let p = T::constant(interpolation_parameter);
    (
        slerp_quaternions(
            prev_node_rotation,
            next_node_rotation,
            interpolation_parameter,
        ),
        [
            prev_node_translation[0] + p * (next_node_translation[0] - prev_node_translation[0]),
            prev_node_translation[1] + p * (next_node_translation[1] - prev_node_translation[1]),
            prev_node_translation[2] + p * (next_node_translation[2] - prev_node_translation[2]),
        ],
    )
}

/// `(AngleAxis(pose[2], UnitZ) * gravity_alignment.cast<T>()).normalized()`
/// as `[w, x, y, z]`.
fn embedded_rotation<T: Scalar>(node_pose: &[T], gravity_alignment: &Quaternion<f64>) -> [T; 4] {
    let q = quaternion_normalized(&quaternion_mul(
        &AngleAxis::new(node_pose[2], Vector3::z()).to_quaternion(),
        &quaternion_cast::<f64, T>(gravity_alignment),
    ));
    [q.w, q.i, q.j, q.k]
}

/// Embeds 2D poses into 3D and interpolates them. Linear interpolation is
/// performed for translation and spherical-linear one for rotation. Returns
/// `(rotation [w, x, y, z], translation [x, y, z])`.
pub fn interpolate_nodes_2d<T: Scalar>(
    prev_node_pose: &[T],
    prev_node_gravity_alignment: &Quaternion<f64>,
    next_node_pose: &[T],
    next_node_gravity_alignment: &Quaternion<f64>,
    interpolation_parameter: f64,
) -> ([T; 4], [T; 3]) {
    // The following is equivalent to (Embed3D(prev_node_pose) *
    // Rigid3d::Rotation(prev_node_gravity_alignment)).rotation().
    let prev_node_rotation = embedded_rotation(prev_node_pose, prev_node_gravity_alignment);

    // The following is equivalent to (Embed3D(next_node_pose) *
    // Rigid3d::Rotation(next_node_gravity_alignment)).rotation().
    let next_node_rotation = embedded_rotation(next_node_pose, next_node_gravity_alignment);

    let p = T::constant(interpolation_parameter);
    (
        slerp_quaternions(
            &prev_node_rotation,
            &next_node_rotation,
            interpolation_parameter,
        ),
        [
            prev_node_pose[0] + p * (next_node_pose[0] - prev_node_pose[0]),
            prev_node_pose[1] + p * (next_node_pose[1] - prev_node_pose[1]),
            T::constant(0.),
        ],
    )
}
