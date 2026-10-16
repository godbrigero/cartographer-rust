// Ported from Ceres Solver 2.0.0: internal/ceres/corrector.h and
// internal/ceres/corrector.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/corrector.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Class definition for the object that is responsible for applying a second
//! order correction to the Gauss-Newton based on the "Bundle Adjustment: A
//! Modern Synthesis" (BANS) paper by Triggs et al.
//!
//! Given a loss function `rho(s)`, the corrector rescales the residuals and
//! the Jacobian so that the Gauss-Newton Hessian of the corrected problem
//! approximates the Hessian of `1/2 rho(|f|^2)` (Equation 10 and 11 in BANS).

/// Applies the robust loss correction to residuals and jacobians.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Corrector {
    sqrt_rho1: f64,
    residual_scaling: f64,
    alpha_sq_norm: f64,
}

impl Corrector {
    /// The constructor takes the squared norm, the value, the first and
    /// second derivatives of the LossFunction. It precalculates some of the
    /// constants that are needed to apply the correction. The correction
    /// constant alpha is computed as the smaller of the two roots of
    /// `0.5 * alpha^2 - alpha - rho'' / rho' * z'z = 0`.
    pub(crate) fn new(sq_norm: f64, rho: &[f64; 3]) -> Self {
        assert!(sq_norm >= 0.0, "Check failed: sq_norm >= 0.0");
        let sqrt_rho1 = rho[1].sqrt();

        // If sq_norm = 0.0, the correction becomes trivial, the residual and
        // the jacobian are scaled by the square root of the derivative of
        // rho. Handling this case explicitly avoids the divide by zero error
        // that would occur below.
        //
        // The case where rho'' < 0 also gets special handling. Technically it
        // shouldn't, and the computation of the scaling should proceed as
        // below, however we found in experiments that applying the curvature
        // correction when rho'' < 0, which is the case when we are in the
        // outlier region slows down the convergence of the algorithm
        // significantly.
        //
        // Thus, we have divided the action of the robustifier into two parts.
        // In the inliner region, we do the full second order correction which
        // re-wights the gradient of the function by the square root of the
        // derivative of rho, and the Gauss-Newton Hessian gets both the
        // scaling and the rank-1 curvature correction. Normally, alpha is
        // upper bounded by one, but with this change, alpha is bounded above
        // by zero.
        //
        // Empirically we have observed that the full Triggs correction and
        // the clamped correction both start out as very good approximations
        // to the loss function when we are in the convex part of the
        // function, but as the function starts transitioning from convex to
        // concave, the Triggs approximation diverges more and more and
        // ultimately becomes linear. The clamped Triggs model however remains
        // quadratic.
        //
        // The reason why the Triggs approximation becomes so poor is because
        // the curvature correction that it applies to the gauss newton
        // hessian goes from being a full rank correction to a rank deficient
        // correction making the inversion of the Hessian fraught with all
        // sorts of misery and suffering.
        //
        // The clamped correction retains its quadratic nature and inverting
        // it is always well formed.
        if sq_norm == 0.0 || rho[2] <= 0.0 {
            return Self {
                sqrt_rho1,
                residual_scaling: sqrt_rho1,
                alpha_sq_norm: 0.0,
            };
        }

        // We now require that the first derivative of the loss function be
        // positive only if the second derivative is positive. This is because
        // when the second derivative is non-positive, we do not use the
        // second order correction suggested by BANS and instead use a simpler
        // first order strategy which does not use a division by the gradient
        // of the loss function.
        assert!(rho[1] > 0.0, "Check failed: rho[1] > 0.0");

        // Calculate the smaller of the two solutions to the equation
        //
        // 0.5 *  alpha^2 - alpha - rho'' / rho' *  z'z = 0.
        //
        // Start by calculating the discriminant D.
        let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

        // Since both rho[1] and rho[2] are guaranteed to be positive at this
        // point, we know that D > 1.0.

        let alpha = 1.0 - d.sqrt();

        // Calculate the constants needed by the correction routines.
        Self {
            sqrt_rho1,
            residual_scaling: sqrt_rho1 / (1.0 - alpha),
            alpha_sq_norm: alpha / sq_norm,
        }
    }

    /// Residual correction (Equation 11 in BANS).
    pub(crate) fn correct_residuals(&self, residuals: &mut [f64]) {
        for r in residuals {
            *r *= self.residual_scaling;
        }
    }

    /// Jacobian correction (Equation 11 in BANS):
    /// `J = sqrt(rho) * (J - alpha^2 r * r' J)`.
    ///
    /// The jacobian is `num_rows x num_cols`, row-major. The residuals must
    /// be the uncorrected residuals.
    pub(crate) fn correct_jacobian(
        &self,
        num_rows: usize,
        num_cols: usize,
        residuals: &[f64],
        jacobian: &mut [f64],
    ) {
        // The common case (rho[2] <= 0).
        if self.alpha_sq_norm == 0.0 {
            for j in &mut jacobian[..num_rows * num_cols] {
                *j *= self.sqrt_rho1;
            }
            return;
        }

        // In days gone by this loop used to be a single Eigen expression of
        // the form
        //
        //  J = sqrt_rho1_ * (J - alpha_sq_norm_ * r* (r.transpose() * J));
        //
        // Which turns out to about 17x slower on bal problems. The reason is
        // that Eigen is unable to figure out that this expression can be
        // evaluated columnwise and ends up creating a temporary.
        for c in 0..num_cols {
            let mut r_transpose_j = 0.0;
            for r in 0..num_rows {
                r_transpose_j += jacobian[r * num_cols + c] * residuals[r];
            }

            for r in 0..num_rows {
                jacobian[r * num_cols + c] = self.sqrt_rho1
                    * (jacobian[r * num_cols + c]
                        - self.alpha_sq_norm * residuals[r] * r_transpose_j);
            }
        }
    }
}

// Ported from Ceres Solver 2.0.0: internal/ceres/corrector_test.cc.
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // If rho[1] is zero, the Corrector constructor should crash.
    #[test]
    #[should_panic]
    fn zero_gradient_death_test() {
        let k_rho = [0.0, 0.0, 1.0];
        let _ = Corrector::new(1.0, &k_rho);
    }

    // If rho[1] is negative, the Corrector constructor should crash.
    #[test]
    #[should_panic]
    fn negative_gradient_death_test() {
        let k_rho = [0.0, -0.1, 1.0];
        let _ = Corrector::new(1.0, &k_rho);
    }

    #[test]
    fn scalar_correction() {
        let mut residuals = 3.0f64.sqrt();
        let mut jacobian = 10.0;
        let sq_norm = residuals * residuals;

        let k_rho = [sq_norm, 0.1, -0.01];

        // In light of the rho'' < 0 clamping now implemented in corrector.cc,
        // alpha = 0 whenever rho'' < 0.
        let k_alpha = 0.0;

        // Thus the expected value of the residual is
        // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
        let k_expected_residual = residuals * k_rho[1].sqrt() / (1.0 - k_alpha);

        // The jacobian in this case will be
        // sqrt(kRho[1]) * (1 - kAlpha) * jacobian.
        let k_expected_jacobian = k_rho[1].sqrt() * (1.0 - k_alpha) * jacobian;

        let c = Corrector::new(sq_norm, &k_rho);
        c.correct_jacobian(1, 1, &[residuals], std::slice::from_mut(&mut jacobian));
        c.correct_residuals(std::slice::from_mut(&mut residuals));

        assert!((residuals - k_expected_residual).abs() <= 1e-6);
        assert!((k_expected_jacobian - jacobian).abs() <= 1e-6);
    }

    #[test]
    fn scalar_correction_zero_residual() {
        let mut residuals = 0.0f64;
        let mut jacobian = 10.0;
        let sq_norm = residuals * residuals;

        let k_rho = [0.0f64, 0.1, -0.01];
        let c = Corrector::new(sq_norm, &k_rho);

        // The alpha equation is
        // 1/2 alpha^2 - alpha + 0.0 = 0.
        // i.e. alpha = 1.0 - sqrt(1.0).
        //      alpha = 0.0.
        // i.e. alpha = 0.0
        let k_expected_residual = residuals * k_rho[1].sqrt();

        // The jacobian in this case will be
        // sqrt(kRho[1]) * jacobian.
        let k_expected_jacobian = k_rho[1].sqrt() * jacobian;

        c.correct_jacobian(1, 1, &[residuals], std::slice::from_mut(&mut jacobian));
        c.correct_residuals(std::slice::from_mut(&mut residuals));

        assert!((residuals - k_expected_residual).abs() <= 1e-6);
        assert!((k_expected_jacobian - jacobian).abs() <= 1e-6);
    }

    // Scaling behaviour for one dimensional functions.
    #[test]
    fn scalar_correction_alpha_clamped() {
        let mut residuals = 3.0f64.sqrt();
        let mut jacobian = 10.0;
        let sq_norm = residuals * residuals;

        let k_rho = [3.0f64, 0.1, -0.1];

        // rho[2] < 0 -> alpha = 0.0
        let k_alpha = 0.0;

        // Thus the expected value of the residual is
        // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
        let k_expected_residual = residuals * k_rho[1].sqrt() / (1.0 - k_alpha);

        // The jacobian in this case will be scaled by
        // sqrt(rho[1]) * (1 - alpha) * J.
        let k_expected_jacobian = k_rho[1].sqrt() * (1.0 - k_alpha) * jacobian;

        let c = Corrector::new(sq_norm, &k_rho);
        c.correct_jacobian(1, 1, &[residuals], std::slice::from_mut(&mut jacobian));
        c.correct_residuals(std::slice::from_mut(&mut residuals));

        assert!((residuals - k_expected_residual).abs() <= 1e-6);
        assert!((k_expected_jacobian - jacobian).abs() <= 1e-6);
    }

    // Row-major 3x2 jacobian (C++ MatrixRef).
    fn jac_matrix(j: &[f64; 6]) -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 2, j)
    }

    // Test that the corrected multidimensional residual and jacobians match
    // the expected values and the resulting modified normal equations match
    // the robustified gauss newton approximation.
    #[test]
    fn multidimensional_gauss_newton_approximation() {
        let mut residuals = [0.0; 3];
        let mut jacobian = [0.0; 6];
        let mut rho = [0.0; 3];

        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..10000 {
            // Initialize the jacobian and residual.
            for j in &mut jacobian {
                *j = rng.random::<f64>();
            }
            for r in &mut residuals {
                *r = rng.random::<f64>();
            }
            let jac = jac_matrix(&jacobian);
            let res = DVector::from_column_slice(&residuals);

            let sq_norm = res.dot(&res);

            rho[0] = sq_norm;
            rho[1] = rng.random::<f64>();
            rho[2] = 2.0 * rng.random::<f64>() - 1.0;

            // If rho[2] > 0, then the curvature correction to the correction
            // and the gauss newton approximation will match. Otherwise, we
            // expect alpha to be 0.
            let k_d = 1.0 + 2.0 * rho[2] / rho[1] * sq_norm;
            let k_alpha = if rho[2] > 0.0 { 1.0 - k_d.sqrt() } else { 0.0 };

            // Ground truth values.
            let g_res = (rho[1].sqrt() / (1.0 - k_alpha)) * &res;
            let g_jac =
                rho[1].sqrt() * (&jac - (k_alpha / sq_norm) * &res * res.transpose() * &jac);
            let g_grad = rho[1] * jac.transpose() * &res;

            let c = Corrector::new(sq_norm, &rho);
            c.correct_jacobian(3, 2, &residuals, &mut jacobian);
            c.correct_residuals(&mut residuals);

            let jac = jac_matrix(&jacobian);
            let res = DVector::from_column_slice(&residuals);

            // Corrected gradient.
            let c_grad = jac.transpose() * &res;

            assert!((g_res - &res).norm() <= 1e-10);
            assert!((g_jac - &jac).norm() <= 1e-10);
            assert!((g_grad - c_grad).norm() <= 1e-10);
        }
    }

    #[test]
    fn multidimensional_gauss_newton_approximation_zero_residual() {
        let mut jacobian = [0.0; 6];
        let mut rho = [0.0; 3];

        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..10000 {
            // Initialize the jacobian.
            for j in &mut jacobian {
                *j = rng.random::<f64>();
            }
            // Zero residuals
            let mut residuals = [0.0; 3];
            let jac = jac_matrix(&jacobian);
            let res = DVector::from_column_slice(&residuals);

            let sq_norm = res.dot(&res);

            rho[0] = sq_norm;
            rho[1] = rng.random::<f64>();
            rho[2] = 2.0 * rng.random::<f64>() - 1.0;

            // Ground truth values.
            let g_res = rho[1].sqrt() * &res;
            let g_jac = rho[1].sqrt() * &jac;
            let g_grad = rho[1] * jac.transpose() * &res;
            let g_hess = rho[1] * jac.transpose() * &jac
                + 2.0 * rho[2] * jac.transpose() * &res * res.transpose() * &jac;

            let c = Corrector::new(sq_norm, &rho);
            c.correct_jacobian(3, 2, &residuals, &mut jacobian);
            c.correct_residuals(&mut residuals);

            let jac = jac_matrix(&jacobian);
            let res = DVector::from_column_slice(&residuals);

            // Corrected gradient and hessian.
            let c_grad = jac.transpose() * &res;
            let c_hess = jac.transpose() * &jac;

            assert!((g_res - &res).norm() <= 1e-10);
            assert!((g_jac - &jac).norm() <= 1e-10);
            assert!((g_grad - c_grad).norm() <= 1e-10);
            assert!((g_hess - c_hess).norm() <= 1e-10);
        }
    }
}
