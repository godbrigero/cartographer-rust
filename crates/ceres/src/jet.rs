// Ported from Ceres Solver 2.0.0: include/ceres/jet.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/jet.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original author: keir@google.com (Keir Mierle)

//! Dual numbers ("jets") for forward-mode automatic differentiation.
//!
//! A [`Jet<N>`] is `a + v` where `a` is the scalar (value) part and `v` is an
//! `N`-dimensional infinitesimal (derivative) part with `t_i * t_j = 0`.
//! Evaluating a function on jets seeded with `v = e_i` yields the function
//! value in `a` and the partial derivatives in `v`.
//!
//! Like `ceres::Jet`, comparisons (`==`, `<`, ...) only look at the scalar
//! part, so that code follows the same branches as with plain `f64`.
//!
//! Besides the arithmetic operators and the jet.h math functions (available
//! both as inherent methods and as free functions, e.g. [`sqrt`]), `Jet<N>`
//! implements the `num-traits` and `simba`/`nalgebra` numeric traits
//! (`nalgebra::RealField`, `ComplexField`, ...), so `nalgebra::Vector3<Jet<N>>`,
//! `Matrix3<Jet<N>>`, `UnitQuaternion<Jet<N>>` and any code generic over
//! [`crate::Scalar`] work with jets exactly like Eigen works with
//! `ceres::Jet`.

use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::SimdValue;

/// A jet: scalar part `a` plus an `N`-dimensional infinitesimal part `v`.
#[derive(Clone, Copy)]
pub struct Jet<const N: usize> {
    /// The scalar part.
    pub a: f64,
    /// The infinitesimal part.
    pub v: [f64; N],
}

impl<const N: usize> Jet<N> {
    /// Number of infinitesimal dimensions (`Jet::DIMENSION` in C++).
    pub const DIMENSION: usize = N;

    /// Constructor from scalar: `a + 0`.
    #[inline]
    pub const fn constant(a: f64) -> Self {
        Self { a, v: [0.0; N] }
    }

    /// Constructor from scalar plus variable: `a + t_k`.
    ///
    /// Panics if `k >= N`.
    #[inline]
    pub fn variable(a: f64, k: usize) -> Self {
        let mut v = [0.0; N];
        v[k] = 1.0;
        Self { a, v }
    }

    /// Constructor from scalar and vector part.
    #[inline]
    pub const fn new(a: f64, v: [f64; N]) -> Self {
        Self { a, v }
    }

    /// Returns `(a, s * v)`; helper for the chain rule.
    #[inline(always)]
    fn chain(a: f64, s: f64, v: &[f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = s * v[i];
        }
        Self { a, v: out }
    }

    /// `x.a / tmp * x.v + y.a / tmp * y.v`-style linear combination of two
    /// infinitesimal parts.
    #[inline(always)]
    fn combine(a: f64, s: f64, x: &[f64; N], t: f64, y: &[f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = s * x[i] + t * y[i];
        }
        Self { a, v: out }
    }

    // ---------------------------------------------------------------------
    // jet.h math functions (inherent so that they resolve unambiguously on
    // concrete jets; the same functions are exposed through the num-traits /
    // simba traits and as free functions below).
    // ---------------------------------------------------------------------

    /// `abs(a + h) ~= abs(a) + sgn(a) h`
    #[inline]
    pub fn abs(self) -> Self {
        if self.a < 0.0 { -self } else { self }
    }

    /// `log(a + h) ~= log(a) + h / a`
    #[inline]
    pub fn ln(self) -> Self {
        let a_inverse = 1.0 / self.a;
        Self::chain(self.a.ln(), a_inverse, &self.v)
    }

    /// Alias of [`Jet::ln`] (C++ `log`).
    #[inline]
    pub fn log(self) -> Self {
        self.ln()
    }

    /// `exp(a + h) ~= exp(a) + exp(a) h`
    #[inline]
    pub fn exp(self) -> Self {
        let tmp = self.a.exp();
        Self::chain(tmp, tmp, &self.v)
    }

    /// `sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))`
    #[inline]
    pub fn sqrt(self) -> Self {
        let tmp = self.a.sqrt();
        let two_a_inverse = 1.0 / (2.0 * tmp);
        Self::chain(tmp, two_a_inverse, &self.v)
    }

    /// `cos(a + h) ~= cos(a) - sin(a) h`
    #[inline]
    pub fn cos(self) -> Self {
        Self::chain(self.a.cos(), -self.a.sin(), &self.v)
    }

    /// `acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h`
    #[inline]
    pub fn acos(self) -> Self {
        let tmp = -1.0 / (1.0 - self.a * self.a).sqrt();
        Self::chain(self.a.acos(), tmp, &self.v)
    }

    /// `sin(a + h) ~= sin(a) + cos(a) h`
    #[inline]
    pub fn sin(self) -> Self {
        Self::chain(self.a.sin(), self.a.cos(), &self.v)
    }

    /// `asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h`
    #[inline]
    pub fn asin(self) -> Self {
        let tmp = 1.0 / (1.0 - self.a * self.a).sqrt();
        Self::chain(self.a.asin(), tmp, &self.v)
    }

    /// `tan(a + h) ~= tan(a) + (1 + tan(a)^2) h`
    #[inline]
    pub fn tan(self) -> Self {
        let tan_a = self.a.tan();
        let tmp = 1.0 + tan_a * tan_a;
        Self::chain(tan_a, tmp, &self.v)
    }

    /// `atan(a + h) ~= atan(a) + 1 / (1 + a^2) h`
    #[inline]
    pub fn atan(self) -> Self {
        let tmp = 1.0 / (1.0 + self.a * self.a);
        Self::chain(self.a.atan(), tmp, &self.v)
    }

    /// `sinh(a + h) ~= sinh(a) + cosh(a) h`
    #[inline]
    pub fn sinh(self) -> Self {
        Self::chain(self.a.sinh(), self.a.cosh(), &self.v)
    }

    /// `cosh(a + h) ~= cosh(a) + sinh(a) h`
    #[inline]
    pub fn cosh(self) -> Self {
        Self::chain(self.a.cosh(), self.a.sinh(), &self.v)
    }

    /// `tanh(a + h) ~= tanh(a) + (1 - tanh(a)^2) h`
    #[inline]
    pub fn tanh(self) -> Self {
        let tanh_a = self.a.tanh();
        let tmp = 1.0 - tanh_a * tanh_a;
        Self::chain(tanh_a, tmp, &self.v)
    }

    /// The floor function should be used with extreme care as this operation
    /// will result in a zero derivative which provides no information to the
    /// solver.
    ///
    /// `floor(a + h) ~= floor(a) + 0`
    #[inline]
    pub fn floor(self) -> Self {
        Self::constant(self.a.floor())
    }

    /// `ceil(a + h) ~= ceil(a) + 0` (same caveat as [`Jet::floor`]).
    #[inline]
    pub fn ceil(self) -> Self {
        Self::constant(self.a.ceil())
    }

    /// `round(a + h) ~= round(a) + 0` (not in jet.h; same caveat as floor).
    #[inline]
    pub fn round(self) -> Self {
        Self::constant(self.a.round())
    }

    /// `trunc(a + h) ~= trunc(a) + 0` (not in jet.h; same caveat as floor).
    #[inline]
    pub fn trunc(self) -> Self {
        Self::constant(self.a.trunc())
    }

    /// `cbrt(a + h) ~= cbrt(a) + h / (3 a ^ (2/3))`
    #[inline]
    pub fn cbrt(self) -> Self {
        let derivative = 1.0 / (3.0 * (self.a * self.a).cbrt());
        Self::chain(self.a.cbrt(), derivative, &self.v)
    }

    /// `exp2(x + h) = 2^(x+h) ~= 2^x + h*2^x*log(2)`
    #[inline]
    pub fn exp2(self) -> Self {
        let tmp = self.a.exp2();
        let derivative = tmp * 2.0f64.ln();
        Self::chain(tmp, derivative, &self.v)
    }

    /// `log2(x + h) ~= log2(x) + h / (x * log(2))`
    #[inline]
    pub fn log2(self) -> Self {
        let derivative = 1.0 / (self.a * 2.0f64.ln());
        Self::chain(self.a.log2(), derivative, &self.v)
    }

    /// `log10(x + h) ~= log10(x) + h / (x * log(10))` (not in jet.h 2.0.0).
    #[inline]
    pub fn log10(self) -> Self {
        let derivative = 1.0 / (self.a * 10.0f64.ln());
        Self::chain(self.a.log10(), derivative, &self.v)
    }

    /// `hypot(x, y) = sqrt(x^2 + y^2)` with
    /// `d/dx = x / hypot(x, y)`, `d/dy = y / hypot(x, y)`.
    #[inline]
    pub fn hypot(self, y: Self) -> Self {
        let tmp = self.a.hypot(y.a);
        Self::combine(tmp, self.a / tmp, &self.v, y.a / tmp, &y.v)
    }

    /// `fmax(x, y)`: `x < y ? y : x`.
    #[inline]
    pub fn fmax(self, y: Self) -> Self {
        if self < y { y } else { self }
    }

    /// `fmin(x, y)`: `y < x ? y : x`.
    #[inline]
    pub fn fmin(self, y: Self) -> Self {
        if y < self { y } else { self }
    }

    /// `erf` is defined as an integral that cannot be expressed analytically,
    /// however the derivative is trivial to compute:
    /// `erf'(x) = 2/sqrt(pi) exp(-x^2)`.
    #[inline]
    pub fn erf(self) -> Self {
        let d = std::f64::consts::FRAC_2_SQRT_PI * (-self.a * self.a).exp();
        Self::chain(libm::erf(self.a), d, &self.v)
    }

    /// `erfc(x) = 1 - erf(x)` ⇒ `erfc'(x) = -2/sqrt(pi) exp(-x^2)`.
    #[inline]
    pub fn erfc(self) -> Self {
        let d = std::f64::consts::FRAC_2_SQRT_PI * (-self.a * self.a).exp();
        // C++: -x.v * M_2_SQRTPI * exp(-x.a * x.a).
        let mut out = [0.0; N];
        for (o, &vi) in out.iter_mut().zip(self.v.iter()) {
            *o = -vi * d;
        }
        Self::new(libm::erfc(self.a), out)
    }

    /// Bessel function of the first kind of order 0, `J0'(x) = -J1(x)`.
    #[inline]
    pub fn bessel_j0(self) -> Self {
        Self::chain(libm::j0(self.a), -libm::j1(self.a), &self.v)
    }

    /// Bessel function of the first kind of order 1,
    /// `J1'(x) = (J0(x) - J2(x)) / 2`.
    #[inline]
    pub fn bessel_j1(self) -> Self {
        Self::chain(
            libm::j1(self.a),
            0.5 * (libm::j0(self.a) - libm::jn(2, self.a)),
            &self.v,
        )
    }

    /// Bessel function of the first kind of order `n`,
    /// `Jn'(x) = (J_{n-1}(x) - J_{n+1}(x)) / 2`.
    #[inline]
    pub fn bessel_jn(self, n: i32) -> Self {
        Self::chain(
            libm::jn(n, self.a),
            0.5 * (libm::jn(n - 1, self.a) - libm::jn(n + 1, self.a)),
            &self.v,
        )
    }

    /// `atan2(b + db, a + da) ~= atan2(b, a) + (- b da + a db) / (a^2 + b^2)`
    ///
    /// Note the order of the arguments: `self` is `y` (`g` in C++), `x` is
    /// `x` (`f` in C++), like `f64::atan2`.
    #[inline]
    pub fn atan2(self, x: Self) -> Self {
        let g = self;
        let f = x;
        let tmp = 1.0 / (f.a * f.a + g.a * g.a);
        // tmp * (-g.a * f.v + f.a * g.v)
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = tmp * (-g.a * f.v[i] + f.a * g.v[i]);
        }
        Self::new(g.a.atan2(f.a), out)
    }

    /// `pow(f, g)` with a scalar exponent:
    /// `pow(a + h, g) ~= pow(a, g) + g pow(a, g - 1) h`.
    #[inline]
    pub fn powf_scalar(self, g: f64) -> Self {
        let tmp = g * self.a.powf(g - 1.0);
        Self::chain(self.a.powf(g), tmp, &self.v)
    }

    /// `pow(f, g)` for a scalar base and jet exponent:
    /// `pow(a, b + h) ~= pow(a, b) + pow(a, b) log(a) h`, with the special
    /// cases of jet.h (`a == 0, b > 0` ⇒ 0; `a < 0, b integer` ⇒ NaN
    /// derivative where `h != 0`).
    #[inline]
    pub fn scalar_powf(f: f64, g: Self) -> Self {
        if f == 0.0 && g.a > 0.0 {
            // Handle case 2.
            Self::constant(0.0)
        } else if f < 0.0 && g.a == g.a.floor() {
            // Handle case 3.
            let mut result = Self::constant(f.powf(g.a));
            for i in 0..N {
                if g.v[i] != 0.0 {
                    // Return a NaN when g.v != 0.
                    result.v[i] = f64::NAN;
                }
            }
            result
        } else {
            // Handle case 1.
            let tmp = f.powf(g.a);
            Self::chain(tmp, f.ln() * tmp, &g.v)
        }
    }

    /// `pow(f, g)` for jet base and exponent:
    /// `pow(a + h, b + k) ~= pow(a, b) + b pow(a, b-1) h + pow(a, b) log(a) k`
    /// with the special cases of jet.h.
    #[inline]
    pub fn powj(self, g: Self) -> Self {
        let f = self;
        if f.a == 0.0 && g.a >= 1.0 {
            // Handle cases 2 and 3.
            if g.a > 1.0 { Self::constant(0.0) } else { f }
        } else if f.a < 0.0 && g.a == g.a.floor() {
            // Handle cases 7 and 8.
            let tmp = g.a * f.a.powf(g.a - 1.0);
            let mut result = Self::chain(f.a.powf(g.a), tmp, &f.v);
            for i in 0..N {
                if g.v[i] != 0.0 {
                    // Return a NaN when g.v != 0.
                    result.v[i] = f64::NAN;
                }
            }
            result
        } else {
            // Handle the remaining cases. For cases 4,5,6,9 we allow the log()
            // function to generate -HUGE_VAL or NaN, since those cases result
            // in a nonfinite derivative.
            let tmp1 = f.a.powf(g.a);
            let tmp2 = g.a * f.a.powf(g.a - 1.0);
            let tmp3 = tmp1 * f.a.ln();
            Self::combine(tmp1, tmp2, &f.v, tmp3, &g.v)
        }
    }

    /// Integer power, implemented as `pow(f, n as f64)`.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf_scalar(f64::from(n))
    }

    /// True iff the scalar part and all derivatives are finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.a.is_finite() && self.v.iter().all(|x| x.is_finite())
    }

    /// True iff the scalar part or any derivative is infinite.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.a.is_infinite() || self.v.iter().any(|x| x.is_infinite())
    }

    /// True iff the scalar part or any derivative is NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.a.is_nan() || self.v.iter().any(|x| x.is_nan())
    }

    /// True iff the scalar part and all derivatives are normal.
    #[inline]
    pub fn is_normal(&self) -> bool {
        self.a.is_normal() && self.v.iter().all(|x| x.is_normal())
    }
}

impl<const N: usize> Default for Jet<N> {
    #[inline]
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> From<f64> for Jet<N> {
    #[inline]
    fn from(a: f64) -> Self {
        Self::constant(a)
    }
}

// ---------------------------------------------------------------------------
// Free functions mirroring the C++ `ceres::sqrt(...)` style API.
// ---------------------------------------------------------------------------

macro_rules! free_unary {
    ($($name:ident => $method:ident),* $(,)?) => {$(
        #[doc = concat!("Free-function form of [`Jet::", stringify!($method), "`].")]
        #[inline]
        pub fn $name<const N: usize>(f: Jet<N>) -> Jet<N> {
            f.$method()
        }
    )*};
}

free_unary!(
    abs => abs, log => ln, exp => exp, sqrt => sqrt, cos => cos, acos => acos,
    sin => sin, asin => asin, tan => tan, atan => atan, sinh => sinh,
    cosh => cosh, tanh => tanh, floor => floor, ceil => ceil, cbrt => cbrt,
    exp2 => exp2, log2 => log2, erf => erf, erfc => erfc,
    bessel_j0 => bessel_j0, bessel_j1 => bessel_j1,
);

/// Free-function form of [`Jet::bessel_jn`].
#[inline]
pub fn bessel_jn<const N: usize>(n: i32, f: Jet<N>) -> Jet<N> {
    f.bessel_jn(n)
}

/// Free-function form of [`Jet::hypot`].
#[inline]
pub fn hypot<const N: usize>(x: Jet<N>, y: Jet<N>) -> Jet<N> {
    x.hypot(y)
}

/// Free-function form of [`Jet::fmax`].
#[inline]
pub fn fmax<const N: usize>(x: Jet<N>, y: Jet<N>) -> Jet<N> {
    x.fmax(y)
}

/// Free-function form of [`Jet::fmin`].
#[inline]
pub fn fmin<const N: usize>(x: Jet<N>, y: Jet<N>) -> Jet<N> {
    x.fmin(y)
}

/// `atan2(g, f)` (note the order of arguments: `y` first, as in C++).
#[inline]
pub fn atan2<const N: usize>(g: Jet<N>, f: Jet<N>) -> Jet<N> {
    g.atan2(f)
}

/// `pow(f, g)` for a jet base and scalar exponent.
#[inline]
pub fn pow_jet_scalar<const N: usize>(f: Jet<N>, g: f64) -> Jet<N> {
    f.powf_scalar(g)
}

/// `pow(f, g)` for a scalar base and jet exponent.
#[inline]
pub fn pow_scalar_jet<const N: usize>(f: f64, g: Jet<N>) -> Jet<N> {
    Jet::scalar_powf(f, g)
}

/// `pow(f, g)` for jet base and exponent.
#[inline]
pub fn pow<const N: usize>(f: Jet<N>, g: Jet<N>) -> Jet<N> {
    f.powj(g)
}

/// `IsFinite(f)`: all parts finite.
#[inline]
pub fn is_finite<const N: usize>(f: &Jet<N>) -> bool {
    f.is_finite()
}

/// `IsInfinite(f)`: any part infinite.
#[inline]
pub fn is_infinite<const N: usize>(f: &Jet<N>) -> bool {
    f.is_infinite()
}

/// `IsNaN(f)`: any part NaN.
#[inline]
pub fn is_nan<const N: usize>(f: &Jet<N>) -> bool {
    f.is_nan()
}

/// `IsNormal(f)`: all parts normal.
#[inline]
pub fn is_normal<const N: usize>(f: &Jet<N>) -> bool {
    f.is_normal()
}

// ---------------------------------------------------------------------------
// Arithmetic.
// ---------------------------------------------------------------------------

impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    /// Unary -
    #[inline]
    fn neg(self) -> Self {
        let mut v = self.v;
        for x in &mut v {
            *x = -*x;
        }
        Self::new(-self.a, v)
    }
}

impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    /// Binary +
    #[inline]
    fn add(self, g: Self) -> Self {
        let mut v = self.v;
        for i in 0..N {
            v[i] += g.v[i];
        }
        Self::new(self.a + g.a, v)
    }
}

impl<const N: usize> Add<f64> for Jet<N> {
    type Output = Self;
    /// Binary + with a scalar: x + s
    #[inline]
    fn add(self, s: f64) -> Self {
        Self::new(self.a + s, self.v)
    }
}

impl<const N: usize> Add<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// Binary + with a scalar: s + x
    #[inline]
    fn add(self, f: Jet<N>) -> Jet<N> {
        Jet::new(f.a + self, f.v)
    }
}

impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    /// Binary -
    #[inline]
    fn sub(self, g: Self) -> Self {
        let mut v = self.v;
        for i in 0..N {
            v[i] -= g.v[i];
        }
        Self::new(self.a - g.a, v)
    }
}

impl<const N: usize> Sub<f64> for Jet<N> {
    type Output = Self;
    /// Binary - with a scalar: x - s
    #[inline]
    fn sub(self, s: f64) -> Self {
        Self::new(self.a - s, self.v)
    }
}

impl<const N: usize> Sub<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// Binary - with a scalar: s - x
    #[inline]
    fn sub(self, f: Jet<N>) -> Jet<N> {
        let mut v = f.v;
        for x in &mut v {
            *x = -*x;
        }
        Jet::new(self - f.a, v)
    }
}

impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    /// Binary *
    #[inline]
    fn mul(self, g: Self) -> Self {
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = self.a * g.v[i] + self.v[i] * g.a;
        }
        Self::new(self.a * g.a, v)
    }
}

impl<const N: usize> Mul<f64> for Jet<N> {
    type Output = Self;
    /// Binary * with a scalar: x * s
    #[inline]
    fn mul(self, s: f64) -> Self {
        let mut v = self.v;
        for x in &mut v {
            *x *= s;
        }
        Self::new(self.a * s, v)
    }
}

impl<const N: usize> Mul<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// Binary * with a scalar: s * x
    #[inline]
    fn mul(self, f: Jet<N>) -> Jet<N> {
        f * self
    }
}

impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    /// Binary /
    ///
    /// This uses:
    ///
    /// ```text
    ///   a + u   (a + u)(b - v)   (a + u)(b - v)
    ///   ----- = -------------- = --------------
    ///   b + v   (b + v)(b - v)        b^2
    /// ```
    ///
    /// which holds because `v*v = 0`.
    #[inline]
    fn div(self, g: Self) -> Self {
        let g_a_inverse = 1.0 / g.a;
        let f_a_by_g_a = self.a * g_a_inverse;
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = (self.v[i] - f_a_by_g_a * g.v[i]) * g_a_inverse;
        }
        Self::new(f_a_by_g_a, v)
    }
}

impl<const N: usize> Div<f64> for Jet<N> {
    type Output = Self;
    /// Binary / by a scalar: x / s
    #[inline]
    fn div(self, s: f64) -> Self {
        let s_inverse = 1.0 / s;
        self * s_inverse
    }
}

impl<const N: usize> Div<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// Binary / by a jet: s / x
    #[inline]
    fn div(self, g: Jet<N>) -> Jet<N> {
        let minus_s_g_a_inverse2 = -self / (g.a * g.a);
        Jet::chain(self / g.a, minus_s_g_a_inverse2, &g.v)
    }
}

impl<const N: usize> Rem for Jet<N> {
    type Output = Self;
    /// Floating point remainder `f - trunc(f / g) * g` (not in jet.h; needed
    /// by `num_traits::Num`). The derivative treats `trunc(f / g)` as
    /// locally constant.
    #[inline]
    fn rem(self, g: Self) -> Self {
        let q = (self.a / g.a).trunc();
        Self::combine(self.a % g.a, 1.0, &self.v, -q, &g.v)
    }
}

impl<const N: usize> Rem<f64> for Jet<N> {
    type Output = Self;
    #[inline]
    fn rem(self, s: f64) -> Self {
        Self::new(self.a % s, self.v)
    }
}

macro_rules! assign_ops {
    ($($tr:ident $m:ident $op:tt),*) => {$(
        impl<const N: usize> $tr for Jet<N> {
            #[inline]
            fn $m(&mut self, y: Self) {
                *self = *self $op y;
            }
        }
        impl<const N: usize> $tr<f64> for Jet<N> {
            #[inline]
            fn $m(&mut self, s: f64) {
                *self = *self $op s;
            }
        }
    )*};
}

assign_ops!(AddAssign add_assign +, SubAssign sub_assign -, MulAssign mul_assign *,
            DivAssign div_assign /, RemAssign rem_assign %);

// Reference forms for convenience (`&a + &b`).
macro_rules! ref_binops {
    ($($tr:ident $m:ident),*) => {$(
        impl<const N: usize> $tr<&Jet<N>> for &Jet<N> {
            type Output = Jet<N>;
            #[inline]
            fn $m(self, g: &Jet<N>) -> Jet<N> {
                $tr::$m(*self, *g)
            }
        }
        impl<const N: usize> $tr<Jet<N>> for &Jet<N> {
            type Output = Jet<N>;
            #[inline]
            fn $m(self, g: Jet<N>) -> Jet<N> {
                $tr::$m(*self, g)
            }
        }
        impl<const N: usize> $tr<&Jet<N>> for Jet<N> {
            type Output = Jet<N>;
            #[inline]
            fn $m(self, g: &Jet<N>) -> Jet<N> {
                $tr::$m(self, *g)
            }
        }
    )*};
}

ref_binops!(Add add, Sub sub, Mul mul, Div div);

impl<const N: usize> Neg for &Jet<N> {
    type Output = Jet<N>;
    #[inline]
    fn neg(self) -> Jet<N> {
        -*self
    }
}

impl<const N: usize> std::iter::Sum for Jet<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(0.0), |acc, x| acc + x)
    }
}

impl<const N: usize> std::iter::Product for Jet<N> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::constant(1.0), |acc, x| acc * x)
    }
}

// ---------------------------------------------------------------------------
// Comparisons: only the scalar part is compared (as in jet.h).
// ---------------------------------------------------------------------------

impl<const N: usize> PartialEq for Jet<N> {
    #[inline]
    fn eq(&self, g: &Self) -> bool {
        self.a == g.a
    }
}

impl<const N: usize> PartialEq<f64> for Jet<N> {
    #[inline]
    fn eq(&self, s: &f64) -> bool {
        self.a == *s
    }
}

impl<const N: usize> PartialEq<Jet<N>> for f64 {
    #[inline]
    fn eq(&self, g: &Jet<N>) -> bool {
        *self == g.a
    }
}

impl<const N: usize> PartialOrd for Jet<N> {
    #[inline]
    fn partial_cmp(&self, g: &Self) -> Option<std::cmp::Ordering> {
        self.a.partial_cmp(&g.a)
    }
}

impl<const N: usize> PartialOrd<f64> for Jet<N> {
    #[inline]
    fn partial_cmp(&self, s: &f64) -> Option<std::cmp::Ordering> {
        self.a.partial_cmp(s)
    }
}

impl<const N: usize> PartialOrd<Jet<N>> for f64 {
    #[inline]
    fn partial_cmp(&self, g: &Jet<N>) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&g.a)
    }
}

// ---------------------------------------------------------------------------
// Formatting: "[a ; v0, v1, ...]" like operator<< in jet.h.
// ---------------------------------------------------------------------------

impl<const N: usize> fmt::Display for Jet<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ; ", self.a)?;
        for (i, x) in self.v.iter().enumerate() {
            write!(f, "{x}")?;
            if i + 1 != N {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

impl<const N: usize> fmt::Debug for Jet<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Jet[{:?} ; {:?}]", self.a, self.v)
    }
}

// ---------------------------------------------------------------------------
// num-traits.
// ---------------------------------------------------------------------------

impl<const N: usize> Zero for Jet<N> {
    #[inline]
    fn zero() -> Self {
        Self::constant(0.0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.a == 0.0
    }
}

impl<const N: usize> One for Jet<N> {
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl<const N: usize> Num for Jet<N> {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(s, radix).map(Self::constant)
    }
}

impl<const N: usize> Signed for Jet<N> {
    #[inline]
    fn abs(&self) -> Self {
        Jet::abs(*self)
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            Self::zero()
        } else {
            *self - *other
        }
    }
    #[inline]
    fn signum(&self) -> Self {
        Self::constant(Signed::signum(&self.a))
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.a.is_sign_positive() && self.a != 0.0
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.a.is_sign_negative() && self.a != 0.0
    }
}

impl<const N: usize> Bounded for Jet<N> {
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }
    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
}

impl<const N: usize> FromPrimitive for Jet<N> {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        f64::from_i64(n).map(Self::constant)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        f64::from_u64(n).map(Self::constant)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::constant(n))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Some(Self::constant(f64::from(n)))
    }
}

// ---------------------------------------------------------------------------
// approx: like the comparison operators, only the scalar part is compared.
// ---------------------------------------------------------------------------

impl<const N: usize> AbsDiffEq for Jet<N> {
    type Epsilon = Self;
    #[inline]
    fn default_epsilon() -> Self {
        Self::constant(f64::default_epsilon())
    }
    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self) -> bool {
        self.a.abs_diff_eq(&other.a, epsilon.a)
    }
}

impl<const N: usize> RelativeEq for Jet<N> {
    #[inline]
    fn default_max_relative() -> Self {
        Self::constant(f64::default_max_relative())
    }
    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: Self, max_relative: Self) -> bool {
        self.a.relative_eq(&other.a, epsilon.a, max_relative.a)
    }
}

impl<const N: usize> UlpsEq for Jet<N> {
    #[inline]
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self, max_ulps: u32) -> bool {
        self.a.ulps_eq(&other.a, epsilon.a, max_ulps)
    }
}

// ---------------------------------------------------------------------------
// simba / nalgebra.
// ---------------------------------------------------------------------------

impl<const N: usize> SimdValue for Jet<N> {
    const LANES: usize = 1;
    type Element = Self;
    type SimdBool = bool;

    #[inline(always)]
    fn splat(val: Self) -> Self {
        val
    }
    #[inline(always)]
    fn extract(&self, _: usize) -> Self {
        *self
    }
    // RUST PORT NOTE: `SimdValue` (required by `nalgebra::RealField`) declares
    // these two methods as `unsafe fn`. Their bodies perform no unsafe
    // operation (a jet has exactly one lane), but implementing an `unsafe fn`
    // trips the workspace-wide `unsafe_code` lint, hence the narrowly scoped
    // allow. This is the only `unsafe` token in the crate.
    #[allow(unsafe_code)]
    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self {
        *self
    }
    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self) {
        *self = val;
    }
    #[allow(unsafe_code)]
    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self) {
        *self = val;
    }
    #[inline(always)]
    fn select(self, cond: bool, other: Self) -> Self {
        if cond { self } else { other }
    }
}

impl<const N: usize> Field for Jet<N> {}

impl<const N: usize> SubsetOf<Jet<N>> for Jet<N> {
    #[inline(always)]
    fn to_superset(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn from_superset_unchecked(element: &Self) -> Self {
        *element
    }
    #[inline(always)]
    fn is_in_subset(_: &Self) -> bool {
        true
    }
}

impl<const N: usize> SubsetOf<Jet<N>> for f64 {
    #[inline(always)]
    fn to_superset(&self) -> Jet<N> {
        Jet::constant(*self)
    }
    /// Takes the scalar part.
    #[inline(always)]
    fn from_superset_unchecked(element: &Jet<N>) -> f64 {
        element.a
    }
    #[inline(always)]
    fn is_in_subset(_: &Jet<N>) -> bool {
        true
    }
}

impl<const N: usize> SubsetOf<Jet<N>> for f32 {
    #[inline(always)]
    fn to_superset(&self) -> Jet<N> {
        Jet::constant(f64::from(*self))
    }
    /// Takes the scalar part.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn from_superset_unchecked(element: &Jet<N>) -> f32 {
        element.a as f32
    }
    #[inline(always)]
    fn is_in_subset(_: &Jet<N>) -> bool {
        true
    }
}

impl<const N: usize> ComplexField for Jet<N> {
    type RealField = Self;

    #[inline]
    fn from_real(re: Self) -> Self {
        re
    }
    #[inline]
    fn real(self) -> Self {
        self
    }
    #[inline]
    fn imaginary(self) -> Self {
        Self::zero()
    }
    #[inline]
    fn modulus(self) -> Self {
        Jet::abs(self)
    }
    #[inline]
    fn modulus_squared(self) -> Self {
        self * self
    }
    #[inline]
    fn argument(self) -> Self {
        if self.a >= 0.0 {
            Self::zero()
        } else {
            Self::constant(std::f64::consts::PI)
        }
    }
    #[inline]
    fn norm1(self) -> Self {
        Jet::abs(self)
    }
    #[inline]
    fn scale(self, factor: Self) -> Self {
        self * factor
    }
    #[inline]
    fn unscale(self, factor: Self) -> Self {
        self / factor
    }
    #[inline]
    fn to_exp(self) -> (Self, Self) {
        if self.a >= 0.0 {
            (self, Self::one())
        } else {
            (-self, -Self::one())
        }
    }
    #[inline]
    fn signum(self) -> Self {
        Signed::signum(&self)
    }
    #[inline]
    fn floor(self) -> Self {
        Jet::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        Jet::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        Jet::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        Jet::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        Self::new(self.a.fract(), self.v)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    fn abs(self) -> Self {
        Jet::abs(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        Jet::hypot(self, other)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
    #[inline]
    fn sin(self) -> Self {
        Jet::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Jet::cos(self)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (Jet::sin(self), Jet::cos(self))
    }
    #[inline]
    fn tan(self) -> Self {
        Jet::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        Jet::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        Jet::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        Jet::atan(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        Jet::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        Jet::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        Jet::tanh(self)
    }
    /// `asinh'(x) = 1 / sqrt(x^2 + 1)` (not in jet.h).
    #[inline]
    fn asinh(self) -> Self {
        Self::chain(
            self.a.asinh(),
            1.0 / (self.a * self.a + 1.0).sqrt(),
            &self.v,
        )
    }
    /// `acosh'(x) = 1 / sqrt(x^2 - 1)` (not in jet.h).
    #[inline]
    fn acosh(self) -> Self {
        Self::chain(
            self.a.acosh(),
            1.0 / (self.a * self.a - 1.0).sqrt(),
            &self.v,
        )
    }
    /// `atanh'(x) = 1 / (1 - x^2)` (not in jet.h).
    #[inline]
    fn atanh(self) -> Self {
        Self::chain(self.a.atanh(), 1.0 / (1.0 - self.a * self.a), &self.v)
    }
    #[inline]
    fn log(self, base: Self) -> Self {
        Jet::ln(self) / Jet::ln(base)
    }
    #[inline]
    fn log2(self) -> Self {
        Jet::log2(self)
    }
    #[inline]
    fn log10(self) -> Self {
        Jet::log10(self)
    }
    #[inline]
    fn ln(self) -> Self {
        Jet::ln(self)
    }
    /// `ln_1p'(x) = 1 / (1 + x)` (not in jet.h).
    #[inline]
    fn ln_1p(self) -> Self {
        Self::chain(self.a.ln_1p(), 1.0 / (1.0 + self.a), &self.v)
    }
    #[inline]
    fn sqrt(self) -> Self {
        Jet::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        Jet::exp(self)
    }
    #[inline]
    fn exp2(self) -> Self {
        Jet::exp2(self)
    }
    /// `exp_m1'(x) = exp(x)` (not in jet.h).
    #[inline]
    fn exp_m1(self) -> Self {
        Self::chain(self.a.exp_m1(), self.a.exp(), &self.v)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        Jet::powi(self, n)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        Jet::powj(self, n)
    }
    #[inline]
    fn powc(self, n: Self) -> Self {
        Jet::powj(self, n)
    }
    #[inline]
    fn cbrt(self) -> Self {
        Jet::cbrt(self)
    }
    #[inline]
    fn is_finite(&self) -> bool {
        Jet::is_finite(self)
    }
    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        if self.a >= 0.0 {
            Some(Jet::sqrt(self))
        } else {
            None
        }
    }
}

impl<const N: usize> RealField for Jet<N> {
    #[inline]
    fn is_sign_positive(&self) -> bool {
        self.a.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(&self) -> bool {
        self.a.is_sign_negative()
    }
    /// `|self|` with the sign of `sign`; the derivative follows `self`.
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if self.a.is_sign_negative() == sign.a.is_sign_negative() {
            self
        } else {
            -self
        }
    }
    /// `fmax` semantics.
    #[inline]
    fn max(self, other: Self) -> Self {
        Jet::fmax(self, other)
    }
    /// `fmin` semantics.
    #[inline]
    fn min(self, other: Self) -> Self {
        Jet::fmin(self, other)
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Jet::atan2(self, other)
    }
    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }
    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::constant(f64::MAX))
    }
    #[inline]
    fn pi() -> Self {
        Self::constant(std::f64::consts::PI)
    }
    #[inline]
    fn two_pi() -> Self {
        Self::constant(std::f64::consts::PI + std::f64::consts::PI)
    }
    #[inline]
    fn frac_pi_2() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_2)
    }
    #[inline]
    fn frac_pi_3() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_3)
    }
    #[inline]
    fn frac_pi_4() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_4)
    }
    #[inline]
    fn frac_pi_6() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_6)
    }
    #[inline]
    fn frac_pi_8() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_8)
    }
    #[inline]
    fn frac_1_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_1_PI)
    }
    #[inline]
    fn frac_2_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_PI)
    }
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_SQRT_PI)
    }
    #[inline]
    fn e() -> Self {
        Self::constant(std::f64::consts::E)
    }
    #[inline]
    fn log2_e() -> Self {
        Self::constant(std::f64::consts::LOG2_E)
    }
    #[inline]
    fn log10_e() -> Self {
        Self::constant(std::f64::consts::LOG10_E)
    }
    #[inline]
    fn ln_2() -> Self {
        Self::constant(std::f64::consts::LN_2)
    }
    #[inline]
    fn ln_10() -> Self {
        Self::constant(std::f64::consts::LN_10)
    }
}
