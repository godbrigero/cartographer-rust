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
