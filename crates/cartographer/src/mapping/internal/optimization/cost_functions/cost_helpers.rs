//! Port of `cartographer/mapping/internal/optimization/cost_functions/cost_helpers` (.h).
//!
//! Helpers shared by the pose graph cost functions; the implementations live
//! in [`cost_helpers_impl`](super::cost_helpers_impl), like in C++.
//!
//! RUST PORT NOTE: the C++ overloads `ComputeUnscaledError` (2D / 3D) and
//! `ScaleError` (`std::array<T, 3>` / `std::array<T, 6>`) get `_2d` / `_3d`
//! suffixes. Pointer parameters `const T*` become slices; `std::array` return
//! values become Rust arrays and `std::tuple` becomes a tuple.

pub use super::cost_helpers_impl::{
    compute_unscaled_error_2d, compute_unscaled_error_3d, interpolate_nodes_2d,
    interpolate_nodes_3d, scale_error_2d, scale_error_3d, slerp_quaternions,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::{Rigid2d, Rigid3d, quaternion_slerp};
    use crate::transform::transform::embed_3d;
    use approx::assert_abs_diff_eq;
    use ceres::Jet;
    use nalgebra::{Quaternion, Vector2, Vector3};
    use proptest::prelude::*;

    fn unit_quaternion(w: f64, x: f64, y: f64, z: f64) -> [f64; 4] {
        let n = (w * w + x * x + y * y + z * z).sqrt();
        [w / n, x / n, y / n, z / n]
    }

    #[test]
    fn compute_unscaled_error_2d_identity_is_zero() {
        let relative = Rigid2d::new(Vector2::new(1., 2.), 0.5);
        // start at origin, end at relative pose => zero error.
        let error = compute_unscaled_error_2d(&relative, &[0., 0., 0.], &[1., 2., 0.5]);
        for e in error {
            assert_abs_diff_eq!(e, 0., epsilon = 1e-12);
        }
        let scaled = scale_error_2d(&[1., 2., 3.], 10., 100.);
        assert_eq!(scaled, [10., 20., 300.]);
        let scaled = scale_error_3d(&[1., 2., 3., 4., 5., 6.], 10., 100.);
        assert_eq!(scaled, [10., 20., 30., 400., 500., 600.]);
    }

    #[test]
    fn compute_unscaled_error_3d_matches_rigid_arithmetic() {
        let start = Rigid3d::new(
            Vector3::new(1., -2., 0.5),
            quat(unit_quaternion(0.9, 0.1, -0.3, 0.2)),
        );
        let end = Rigid3d::new(
            Vector3::new(-0.5, 3., 1.5),
            quat(unit_quaternion(0.7, -0.2, 0.4, 0.1)),
        );
        let relative = start.inverse() * end;
        let s = start.rotation();
        let e = end.rotation();
        let error = compute_unscaled_error_3d(
            &relative,
            &[s.w, s.i, s.j, s.k],
            start.translation().as_slice(),
            &[e.w, e.i, e.j, e.k],
            end.translation().as_slice(),
        );
        for v in error {
            assert_abs_diff_eq!(v, 0., epsilon = 1e-9);
        }
    }

    fn quat(q: [f64; 4]) -> Quaternion<f64> {
        Quaternion::new(q[0], q[1], q[2], q[3])
    }

    #[test]
    fn slerp_quaternions_runs_on_jets() {
        let start = [
            Jet::<1>::variable(1., 0),
            Jet::constant(0.),
            Jet::constant(0.),
            Jet::constant(0.),
        ];
        let end = unit_quaternion(0.5, 0.5, 0.5, 0.5).map(Jet::<1>::constant);
        let result = slerp_quaternions(&start, &end, 0.25);
        assert!(result.iter().all(|j| j.a.is_finite() && j.v[0].is_finite()));
    }

    #[test]
    fn interpolate_nodes_2d_matches_embed_3d() {
        let gravity = Quaternion::new(1., 0., 0., 0.);
        let pose = [1., 2., 0.3];
        let (rotation, translation) = interpolate_nodes_2d(&pose, &gravity, &pose, &gravity, 0.4);
        let expected = embed_3d(&Rigid2d::new(Vector2::new(1., 2.), 0.3));
        let q = expected.rotation();
        assert_abs_diff_eq!(rotation[0], q.w, epsilon = 1e-12);
        assert_abs_diff_eq!(rotation[1], q.i, epsilon = 1e-12);
        assert_abs_diff_eq!(rotation[2], q.j, epsilon = 1e-12);
        assert_abs_diff_eq!(rotation[3], q.k, epsilon = 1e-12);
        assert_abs_diff_eq!(translation[0], 1., epsilon = 1e-12);
        assert_abs_diff_eq!(translation[1], 2., epsilon = 1e-12);
        assert_eq!(translation[2], 0.);
    }

    #[test]
    fn interpolate_nodes_3d_translation_is_linear() {
        let identity = [1., 0., 0., 0.];
        let (rotation, translation) =
            interpolate_nodes_3d(&identity, &[0., 0., 0.], &identity, &[2., 4., -6.], 0.25);
        assert_eq!(rotation, identity);
        assert_eq!(translation, [0.5, 1., -1.5]);
    }

    proptest! {
        /// Our slerp agrees with Eigen's slerp (away from the collinear
        /// threshold, where the two linearize at different epsilons).
        #[test]
        fn slerp_quaternions_matches_eigen_slerp(
            a in prop::array::uniform4(-1.0f64..1.0),
            b in prop::array::uniform4(-1.0f64..1.0),
            factor in 0.0f64..1.0,
        ) {
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            prop_assume!(na > 0.1 && nb > 0.1);
            let a = a.map(|x| x / na);
            let b = b.map(|x| x / nb);
            let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            prop_assume!(dot.abs() < 1. - 1e-4);
            let ours = slerp_quaternions(&a, &b, factor);
            let eigen = quaternion_slerp(
                &Quaternion::new(a[0], a[1], a[2], a[3]),
                factor,
                &Quaternion::new(b[0], b[1], b[2], b[3]),
            );
            prop_assert!((ours[0] - eigen.w).abs() < 1e-9);
            prop_assert!((ours[1] - eigen.i).abs() < 1e-9);
            prop_assert!((ours[2] - eigen.j).abs() < 1e-9);
            prop_assert!((ours[3] - eigen.k).abs() < 1e-9);
        }
    }
}
