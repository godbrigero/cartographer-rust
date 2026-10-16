//! Port of `cartographer/mapping/internal/eigen_quaterniond_from_two_vectors` (.h, .cc).

use nalgebra::{Matrix3, Quaternion, Vector3};

/// Returns a quaternion representing the rotation that maps `a` onto `b`
/// (C++ `FromTwoVectors`, a wrapper around Eigen's
/// `Quaterniond::FromTwoVectors`).
///
/// Follows Eigen's `QuaternionBase::setFromTwoVectors` formulas and operation
/// order. The result is a unit quaternion.
pub fn from_two_vectors(a: &Vector3<f64>, b: &Vector3<f64>) -> Quaternion<f64> {
    // `NumTraits<double>::dummy_precision()`.
    const DUMMY_PRECISION: f64 = 1e-12;
    let v0 = a.normalize();
    let v1 = b.normalize();
    let mut c = v1.dot(&v0);

    // If the dot product is close to -1, the vectors are nearly opposite:
    // the rotation axis is any vector orthogonal to both, and it is
    // computed as the singular vector of the smallest singular value of the
    // matrix whose rows are 'v0' and 'v1'.
    if c < -1. + DUMMY_PRECISION {
        c = c.max(-1.);
        // RUST PORT NOTE: Eigen runs a JacobiSVD on the 2x3 matrix
        // [v0^T; v1^T] with a full V. nalgebra only computes thin SVDs of
        // non-square matrices, so the equivalent square matrix
        // [v0^T; v1^T; 0] is decomposed instead (same right singular
        // vectors). The sign of the axis may differ from Eigen's, which
        // represents the same 180 degree rotation up to the sign of the axis.
        let m = Matrix3::from_rows(&[v0.transpose(), v1.transpose(), Vector3::zeros().transpose()]);
        let svd = m.svd(false, true);
        let v_t = svd.v_t.expect("V^T was requested");
        let (min_index, _) = svd.singular_values.iter().enumerate().fold(
            (0, f64::INFINITY),
            |(best_i, best), (i, &s)| {
                if s < best { (i, s) } else { (best_i, best) }
            },
        );
        let axis: Vector3<f64> = v_t.row(min_index).transpose();
        let w2 = (1. + c) * 0.5;
        let vec = axis * (1. - w2).sqrt();
        return Quaternion::new(w2.sqrt(), vec.x, vec.y, vec.z);
    }
    let axis = v0.cross(&v1);
    let s = ((1. + c) * 2.).sqrt();
    let invs = 1. / s;
    let vec = axis * invs;
    Quaternion::new(s * 0.5, vec.x, vec.y, vec.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::rigid_transform::{quaternion_norm, quaternion_rotate};
    use proptest::prelude::*;

    fn assert_maps(a: Vector3<f64>, b: Vector3<f64>) {
        let q = from_two_vectors(&a, &b);
        assert!((quaternion_norm(&q) - 1.).abs() < 1e-12);
        let rotated = quaternion_rotate(&q, &a.normalize());
        assert!(
            (rotated - b.normalize()).norm() < 1e-9,
            "{rotated:?} vs {:?}",
            b.normalize()
        );
    }

    #[test]
    fn identical_vectors_give_identity() {
        let q = from_two_vectors(&Vector3::new(1., 2., 3.), &Vector3::new(2., 4., 6.));
        assert!((q.w - 1.).abs() < 1e-15);
        assert!(q.imag().norm() < 1e-15);
    }

    #[test]
    fn opposite_vectors() {
        assert_maps(Vector3::new(0., 0., 1.), Vector3::new(0., 0., -1.));
        assert_maps(Vector3::new(1., 2., 3.), Vector3::new(-1., -2., -3.));
    }

    proptest! {
        #[test]
        fn maps_a_onto_b(
            ax in -10.0..10.0f64, ay in -10.0..10.0f64, az in -10.0..10.0f64,
            bx in -10.0..10.0f64, by in -10.0..10.0f64, bz in -10.0..10.0f64,
        ) {
            let a = Vector3::new(ax, ay, az);
            let b = Vector3::new(bx, by, bz);
            prop_assume!(a.norm() > 1e-3 && b.norm() > 1e-3);
            assert_maps(a, b);
        }
    }
}
