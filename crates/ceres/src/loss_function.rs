// Ported from Ceres Solver 2.0.0: include/ceres/loss_function.h and
// internal/ceres/loss_function.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/loss_function.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Robust loss functions.
//!
//! The term "loss function" refers to a function `rho(s)` applied to the
//! squared norm of a residual block: the contribution of a term to the cost
//! is `1/2 rho(||f(x)||^2)`. For least squares problems where the
//! minimization may encounter input terms that contain outliers, that is,
//! completely bogus measurements, it is important to use a loss function
//! that reduces their associated penalty. Passing `None` as loss function
//! to [`Problem::add_residual_block`](crate::Problem::add_residual_block)
//! gives the standard squared loss.
//!
//! Note: in the region of interest (i.e. `s < 3`) we have:
//! `TrivialLoss >= HuberLoss >= SoftLOneLoss >= CauchyLoss`.

use std::sync::{Arc, RwLock};

/// A robustifier `rho(s)`.
pub trait LossFunction: Send + Sync {
    /// For a residual vector with squared 2-norm `sq_norm`, this method is
    /// required to fill in the value and derivatives of the loss function
    /// (rho in this example):
    ///
    /// ```text
    ///   out[0] = rho(sq_norm),
    ///   out[1] = rho'(sq_norm),
    ///   out[2] = rho''(sq_norm),
    /// ```
    ///
    /// Here the convention is that the contribution of a term to the cost
    /// function is given by `1/2 rho(s)`, where `s = ||residuals||^2`.
    ///
    /// Calling the method with a negative value of `s` is an error and the
    /// implementations are not required to handle that case.
    ///
    /// Most sane choices of rho() satisfy:
    ///
    /// ```text
    ///   rho(0) = 0,
    ///   rho'(0) = 1,
    ///   rho'(s) < 1 in outlier region,
    ///   rho''(s) < 0 in outlier region,
    /// ```
    ///
    /// so that they mimic the least squares cost for small residuals.
    fn evaluate(&self, sq_norm: f64, out: &mut [f64; 3]);
}

impl<L: LossFunction + ?Sized> LossFunction for Arc<L> {
    fn evaluate(&self, sq_norm: f64, out: &mut [f64; 3]) {
        (**self).evaluate(sq_norm, out);
    }
}

impl<L: LossFunction + ?Sized> LossFunction for Box<L> {
    fn evaluate(&self, sq_norm: f64, out: &mut [f64; 3]) {
        (**self).evaluate(sq_norm, out);
    }
}

/// This corresponds to no robustification.
///
/// ```text
///   rho(s) = s
/// ```
///
/// At s = 0: rho = [0, 1, 0].
///
/// It is not normally necessary to use this, as passing `None` for the loss
/// function when building the problem accomplishes the same thing.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrivialLoss;

impl LossFunction for TrivialLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        rho[0] = s;
        rho[1] = 1.0;
        rho[2] = 0.0;
    }
}

/// Scaling
/// -------
/// Given one robustifier
///
/// ```text
///   s -> rho(s)
/// ```
///
/// one can change the length scale at which robustification takes place, by
/// adding a scale factor 'a' as follows:
///
/// ```text
///   s -> a^2 rho(s / a^2).
/// ```
///
/// The first and second derivatives are:
///
/// ```text
///   s -> rho'(s / a^2),
///   s -> (1 / a^2) rho''(s / a^2),
/// ```
///
/// but the behaviour near s = 0 is the same as the original function, i.e.
///
/// ```text
///   rho(s) = s + higher order terms,
///   a^2 rho(s / a^2) = s + higher order terms.
/// ```
///
/// The scalar 'a' should be positive.
///
/// The reason for the appearance of squaring is that 'a' is in the units of
/// the residual vector norm whereas 's' is a squared norm. For applications
/// it is more convenient to specify 'a' than its square. The commonly used
/// robustifiers below are described in un-scaled format (a = 1) but their
/// implementations work for any non-zero value of 'a'.
///
/// Huber.
///
/// ```text
///   rho(s) = s               for s <= 1,
///   rho(s) = 2 sqrt(s) - 1   for s >= 1.
/// ```
///
/// At s = 0: rho = [0, 1, 0].
///
/// The scaling parameter 'a' corresponds to 'delta' on this page:
/// <http://en.wikipedia.org/wiki/Huber_Loss_Function>
#[derive(Clone, Copy, Debug)]
pub struct HuberLoss {
    a: f64,
    // b = a^2.
    b: f64,
}

impl HuberLoss {
    /// Creates a Huber loss with scale `a`.
    pub fn new(a: f64) -> Self {
        Self { a, b: a * a }
    }
}

impl LossFunction for HuberLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        if s > self.b {
            // Outlier region.
            // 'r' is always positive.
            let r = s.sqrt();
            rho[0] = 2.0 * self.a * r - self.b;
            rho[1] = f64::MIN_POSITIVE.max(self.a / r);
            rho[2] = -rho[1] / (2.0 * s);
        } else {
            // Inlier region.
            rho[0] = s;
            rho[1] = 1.0;
            rho[2] = 0.0;
        }
    }
}

/// Soft L1, similar to Huber but smooth.
///
/// ```text
///   rho(s) = 2 (sqrt(1 + s) - 1).
/// ```
///
/// At s = 0: rho = [0, 1, -1 / (2 * a^2)].
#[derive(Clone, Copy, Debug)]
pub struct SoftLOneLoss {
    // b = a^2.
    b: f64,
    // c = 1 / a^2.
    c: f64,
}

impl SoftLOneLoss {
    /// Creates a soft L1 loss with scale `a`.
    pub fn new(a: f64) -> Self {
        let b = a * a;
        Self { b, c: 1.0 / b }
    }
}

impl LossFunction for SoftLOneLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * self.c;
        let tmp = sum.sqrt();
        // 'sum' and 'tmp' are always positive, assuming that 's' is.
        rho[0] = 2.0 * self.b * (tmp - 1.0);
        rho[1] = f64::MIN_POSITIVE.max(1.0 / tmp);
        rho[2] = -(self.c * rho[1]) / (2.0 * sum);
    }
}

/// Inspired by the Cauchy distribution
///
/// ```text
///   rho(s) = log(1 + s).
/// ```
///
/// At s = 0: rho = [0, 1, -1 / a^2].
#[derive(Clone, Copy, Debug)]
pub struct CauchyLoss {
    // b = a^2.
    b: f64,
    // c = 1 / a^2.
    c: f64,
}

impl CauchyLoss {
    /// Creates a Cauchy loss with scale `a`.
    pub fn new(a: f64) -> Self {
        let b = a * a;
        Self { b, c: 1.0 / b }
    }
}

impl LossFunction for CauchyLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * self.c;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive, assuming that 's' is.
        rho[0] = self.b * sum.ln();
        rho[1] = f64::MIN_POSITIVE.max(inv);
        rho[2] = -self.c * (inv * inv);
    }
}

/// Loss that is capped beyond a certain level using the arc-tangent
/// function. The scaling parameter 'a' determines the level where falloff
/// occurs. For costs much smaller than 'a', the loss function is linear and
/// behaves like TrivialLoss, and for values much larger than 'a' the value
/// asymptotically approaches the constant value of a * PI / 2.
///
/// ```text
///   rho(s) = a atan(s / a).
/// ```
///
/// At s = 0: rho = [0, 1, 0].
#[derive(Clone, Copy, Debug)]
pub struct ArctanLoss {
    a: f64,
    // b = 1 / a^2.
    b: f64,
}

impl ArctanLoss {
    /// Creates an arctan loss with scale `a`.
    pub fn new(a: f64) -> Self {
        Self {
            a,
            b: 1.0 / (a * a),
        }
    }
}

impl LossFunction for ArctanLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * s * self.b;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive.
        rho[0] = self.a * s.atan2(self.a);
        rho[1] = f64::MIN_POSITIVE.max(inv);
        rho[2] = -2.0 * s * self.b * (inv * inv);
    }
}

/// Loss function that maps to approximately zero cost in a range around the
/// origin, and reverts to linear in error (quadratic in cost) beyond this
/// range. The tolerance parameter 'a' sets the nominal point at which the
/// transition occurs, and the transition size parameter 'b' sets the nominal
/// distance over which most of the transition occurs. Both a and b must be
/// greater than zero, and typically b will be set to a fraction of a. The
/// slope rho'[s] varies smoothly from about 0 at s <= a - b to about 1 at
/// s >= a + b.
///
/// The term is computed as:
///
/// ```text
///   rho(s) = b log(1 + exp((s - a) / b)) - c0.
/// ```
///
/// where c0 is chosen so that rho(0) == 0
///
/// ```text
///   c0 = b log(1 + exp(-a / b)
/// ```
///
/// This has the following useful properties:
///
/// ```text
///   rho(s) == 0               for s = 0
///   rho'(s) ~= 0              for s << a - b
///   rho'(s) ~= 1              for s >> a + b
///   rho''(s) > 0              for all s
/// ```
///
/// In addition, all derivatives are continuous, and the curvature is
/// concentrated in the range a - b to a + b.
///
/// At s = 0: rho = [0, ~0, ~0].
#[derive(Clone, Copy, Debug)]
pub struct TolerantLoss {
    a: f64,
    b: f64,
    c: f64,
}

impl TolerantLoss {
    /// Creates a tolerant loss; panics unless `a >= 0` and `b > 0`.
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a >= 0.0, "Check failed: a >= 0.0");
        assert!(b > 0.0, "Check failed: b > 0.0");
        Self {
            a,
            b,
            c: b * (1.0 + (-a / b).exp()).ln(),
        }
    }
}

impl LossFunction for TolerantLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let x = (s - self.a) / self.b;
        // The basic equation is rho[0] = b ln(1 + e^x). However, if e^x is
        // too large, it will overflow. Since numerically 1 + e^x == e^x when
        // the x is greater than about ln(2^53) for doubles, beyond this
        // threshold we substitute x for ln(1 + e^x) as a numerically
        // equivalent approximation.
        const K_LOG2_POW53: f64 = 36.7; // ln(MathLimits<double>::kEpsilon).
        if x > K_LOG2_POW53 {
            rho[0] = s - self.a - self.c;
            rho[1] = 1.0;
            rho[2] = 0.0;
        } else {
            let e_x = x.exp();
            rho[0] = self.b * (1.0 + e_x).ln() - self.c;
            rho[1] = f64::MIN_POSITIVE.max(e_x / (1.0 + e_x));
            rho[2] = 0.5 / (self.b * (1.0 + x.cosh()));
        }
    }
}

/// This is the Tukey biweight loss function which aggressively attempts to
/// suppress large errors.
///
/// The term is computed as follows where the equations are scaled by a
/// factor of 2 because the cost function is given by 1/2 rho(s):
///
/// ```text
///   rho(s) = a^2 / 3 * (1 - (1 - s / a^2)^3 )   for s <= a^2,
///   rho(s) = a^2 / 3                            for s >  a^2.
/// ```
///
/// At s = 0: rho = [0, 1, -2 / a^2]
#[derive(Clone, Copy, Debug)]
pub struct TukeyLoss {
    a_squared: f64,
}

impl TukeyLoss {
    /// Creates a Tukey loss with scale `a`.
    pub fn new(a: f64) -> Self {
        Self { a_squared: a * a }
    }
}

impl LossFunction for TukeyLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        if s <= self.a_squared {
            // Inlier region.
            let value = 1.0 - s / self.a_squared;
            let value_sq = value * value;
            rho[0] = self.a_squared / 3.0 * (1.0 - value_sq * value);
            rho[1] = value_sq;
            rho[2] = -2.0 / self.a_squared * value;
        } else {
            // Outlier region.
            rho[0] = self.a_squared / 3.0;
            rho[1] = 0.0;
            rho[2] = 0.0;
        }
    }
}

/// Composition of two loss functions. The error is the result of first
/// evaluating g followed by f to yield the composition f(g(s)).
///
/// RUST PORT NOTE: C++ `Ownership` flags are replaced by ownership of the
/// boxed loss functions (wrap them in `Arc` to share).
pub struct ComposedLoss {
    f: Box<dyn LossFunction>,
    g: Box<dyn LossFunction>,
}

impl ComposedLoss {
    /// Creates `f(g(s))`.
    pub fn new(f: Box<dyn LossFunction>, g: Box<dyn LossFunction>) -> Self {
        Self { f, g }
    }
}

impl LossFunction for ComposedLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let mut rho_f = [0.0; 3];
        let mut rho_g = [0.0; 3];
        self.g.evaluate(s, &mut rho_g);
        self.f.evaluate(rho_g[0], &mut rho_f);
        rho[0] = rho_f[0];
        // f'(g(s)) * g'(s).
        rho[1] = rho_f[1] * rho_g[1];
        // f''(g(s)) * g'(s) * g'(s) + f'(g(s)) * g''(s).
        rho[2] = rho_f[2] * rho_g[1] * rho_g[1] + rho_f[1] * rho_g[2];
    }
}

/// If rho is the wrapped robustifier, then this simply outputs
/// `s -> a * rho(s)`.
///
/// The first and second derivatives are, not surprisingly
///
/// ```text
///   s -> a * rho'(s)
///   s -> a * rho''(s)
/// ```
///
/// Since we treat a `None` loss function as the identity loss function,
/// `rho = None` is a valid input and will result in the input being scaled
/// by a. This provides a simple way of implementing a scaled ResidualBlock.
pub struct ScaledLoss {
    rho: Option<Box<dyn LossFunction>>,
    a: f64,
}

impl ScaledLoss {
    /// Creates `a * rho(s)`.
    pub fn new(rho: Option<Box<dyn LossFunction>>, a: f64) -> Self {
        Self { rho, a }
    }
}

impl LossFunction for ScaledLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        match &self.rho {
            None => {
                rho[0] = self.a * s;
                rho[1] = self.a;
                rho[2] = 0.0;
            }
            Some(inner) => {
                inner.evaluate(s, rho);
                rho[0] *= self.a;
                rho[1] *= self.a;
                rho[2] *= self.a;
            }
        }
    }
}

/// Sometimes after the optimization problem has been constructed, we wish
/// to mutate the scale of the loss function. For example, when performing
/// estimation from data which has substantial outliers, convergence can be
/// improved by starting out with a large scale, optimizing the problem and
/// then reducing the scale. This can have better convergence behaviour than
/// just using a loss function with a small scale.
///
/// This type allows the user to implement a loss function whose scale can be
/// mutated after an optimization problem has been constructed: share it via
/// `Arc<LossFunctionWrapper>` and call [`LossFunctionWrapper::reset`].
///
/// Since we treat a `None` loss function as the identity loss function,
/// `rho = None` is a valid input.
pub struct LossFunctionWrapper {
    rho: RwLock<Option<Box<dyn LossFunction>>>,
}

impl LossFunctionWrapper {
    /// Wraps `rho`.
    pub fn new(rho: Option<Box<dyn LossFunction>>) -> Self {
        Self {
            rho: RwLock::new(rho),
        }
    }

    /// Replaces the wrapped loss function.
    pub fn reset(&self, rho: Option<Box<dyn LossFunction>>) {
        *self
            .rho
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = rho;
    }
}

impl LossFunction for LossFunctionWrapper {
    fn evaluate(&self, sq_norm: f64, out: &mut [f64; 3]) {
        let guard = self
            .rho
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match guard.as_ref() {
            None => {
                out[0] = sq_norm;
                out[1] = 1.0;
                out[2] = 0.0;
            }
            Some(rho) => rho.evaluate(sq_norm, out),
        }
    }
}
