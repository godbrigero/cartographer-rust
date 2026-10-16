//! Least-squares point-set alignment (Horn / Umeyama).

use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3,
};

/// A similarity transform `x -> scale * (rotation * x) + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityTransform {
    /// Rigid part (`rotation`, `translation`).
    pub isometry: Isometry3<f64>,
    /// Uniform scale (1 for rigid alignment).
    pub scale: f64,
}

impl SimilarityTransform {
    /// The identity.
    pub fn identity() -> Self {
        Self {
            isometry: Isometry3::identity(),
            scale: 1.0,
        }
    }

    /// Applies the transform to a point.
    pub fn transform_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.isometry.rotation * p * self.scale + self.isometry.translation.vector
    }

    /// Applies the transform to a pose (rotation composed, position mapped).
    pub fn transform_pose(&self, pose: &Isometry3<f64>) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::from(self.transform_point(&pose.translation.vector)),
            self.isometry.rotation * pose.rotation,
        )
    }
}

/// Umeyama's closed-form least-squares alignment in `d` dimensions:
/// finds `R` (proper rotation), `t` and optionally `s` minimising
/// `sum |dst_i - (s R src_i + t)|^2`. Columns of `src`/`dst` are points.
/// Returns `(R, t, s)`; degenerate inputs (fewer than 2 points or zero
/// spread) give the identity rotation with the centroid translation.
pub fn umeyama(
    src: &DMatrix<f64>,
    dst: &DMatrix<f64>,
    with_scale: bool,
) -> (DMatrix<f64>, DVector<f64>, f64) {
    assert_eq!(src.shape(), dst.shape(), "point set shape mismatch");
    let (d, n) = src.shape();
    if n == 0 {
        return (DMatrix::identity(d, d), DVector::zeros(d), 1.0);
    }
    let mu_s = src.column_mean();
    let mu_d = dst.column_mean();
    let mut xs = src.clone();
    let mut xd = dst.clone();
    for mut c in xs.column_iter_mut() {
        c -= &mu_s;
    }
    for mut c in xd.column_iter_mut() {
        c -= &mu_d;
    }
    let var_s = xs.norm_squared() / n as f64;
    if n < 2 || var_s < 1e-18 {
        return (DMatrix::identity(d, d), &mu_d - &mu_s, 1.0);
    }
    let sigma = &xd * xs.transpose() / n as f64;
    let svd = sigma.clone().svd(true, true);
    let u = svd.u.expect("svd u");
    let v_t = svd.v_t.expect("svd v_t");
    let mut s_diag = DVector::from_element(d, 1.0);
    if u.determinant() * v_t.determinant() < 0.0 {
        s_diag[d - 1] = -1.0;
    }
    let r = &u * DMatrix::from_diagonal(&s_diag) * &v_t;
    let scale = if with_scale {
        svd.singular_values.dot(&s_diag) / var_s
    } else {
        1.0
    };
    let t = &mu_d - &r * &mu_s * scale;
    (r, t, scale)
}

/// 3D alignment of `src` onto `dst` (rigid, or similarity if `with_scale`).
pub fn align_3d(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    with_scale: bool,
) -> SimilarityTransform {
    let to_mat = |v: &[Vector3<f64>]| DMatrix::from_fn(3, v.len(), |r, c| v[c][r]);
    let (r, t, s) = umeyama(&to_mat(src), &to_mat(dst), with_scale);
    let rot = Rotation3::from_matrix_unchecked(Matrix3::from_fn(|i, j| r[(i, j)]));
    SimilarityTransform {
        isometry: Isometry3::from_parts(
            Translation3::new(t[0], t[1], t[2]),
            UnitQuaternion::from_rotation_matrix(&rot),
        ),
        scale: s,
    }
}

/// Planar alignment: rotation about z plus x-y translation (z ignored),
/// optionally with scale applied to x and y.
pub fn align_2d(
    src: &[Vector3<f64>],
    dst: &[Vector3<f64>],
    with_scale: bool,
) -> SimilarityTransform {
    let to_mat = |v: &[Vector3<f64>]| DMatrix::from_fn(2, v.len(), |r, c| v[c][r]);
    let (r, t, s) = umeyama(&to_mat(src), &to_mat(dst), with_scale);
    let yaw = r[(1, 0)].atan2(r[(0, 0)]);
    SimilarityTransform {
        isometry: Isometry3::from_parts(
            Translation3::new(t[0], t[1], 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
        ),
        scale: s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> Vec<Vector3<f64>> {
        (0..20)
            .map(|i| {
                let f = i as f64;
                Vector3::new(f.sin() * 3.0, (0.7 * f).cos() * 2.0, 0.1 * f)
            })
            .collect()
    }

    #[test]
    fn recovers_rigid_and_similarity_3d() {
        let truth = SimilarityTransform {
            isometry: Isometry3::from_parts(
                Translation3::new(1.0, -2.0, 0.5),
                UnitQuaternion::from_euler_angles(0.3, -0.4, 2.5),
            ),
            scale: 1.7,
        };
        let src = points();
        let dst: Vec<_> = src.iter().map(|p| truth.transform_point(p)).collect();
        let est = align_3d(&src, &dst, true);
        assert!((est.scale - 1.7).abs() < 1e-9);
        assert!(est.isometry.rotation.angle_to(&truth.isometry.rotation) < 1e-9);
        assert!(
            (est.isometry.translation.vector - truth.isometry.translation.vector).norm() < 1e-9
        );
        // Rigid alignment of a rigidly transformed copy.
        let rigid = SimilarityTransform {
            scale: 1.0,
            ..truth
        };
        let dst: Vec<_> = src.iter().map(|p| rigid.transform_point(p)).collect();
        let est = align_3d(&src, &dst, false);
        for (s, d) in src.iter().zip(&dst) {
            assert!((est.transform_point(s) - d).norm() < 1e-9);
        }
    }

    #[test]
    fn recovers_planar() {
        let truth = SimilarityTransform {
            isometry: Isometry3::from_parts(
                Translation3::new(3.0, 4.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, -2.0),
            ),
            scale: 1.0,
        };
        let src = points();
        let dst: Vec<_> = src.iter().map(|p| truth.transform_point(p)).collect();
        let est = align_2d(&src, &dst, false);
        assert!(est.isometry.rotation.angle_to(&truth.isometry.rotation) < 1e-9);
        assert!(
            (est.isometry.translation.vector - truth.isometry.translation.vector).norm() < 1e-9
        );
    }

    #[test]
    fn no_reflection() {
        // Planar points mirrored: best proper rotation, never a reflection.
        let src = points();
        let dst: Vec<_> = src.iter().map(|p| Vector3::new(p.x, -p.y, p.z)).collect();
        let est = align_3d(&src, &dst, false);
        let m = est.isometry.rotation.to_rotation_matrix();
        assert!((m.matrix().determinant() - 1.0).abs() < 1e-9);
    }
}
