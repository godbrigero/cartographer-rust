//! C2 cubic interpolating splines (natural or periodic) with analytic first
//! and second derivatives.

use nalgebra::{DMatrix, DVector};

/// Boundary condition of a [`CubicSpline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplineKind {
    /// Zero second derivative at both ends; evaluation is clamped to the knot
    /// range.
    Natural,
    /// Periodic: value, slope and curvature match across the wrap; the last
    /// knot value must equal the first. Evaluation wraps modulo the period.
    Periodic,
}

/// A scalar C2 cubic spline `y(t)` through knots `(t_i, y_i)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CubicSpline {
    t: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>, // second derivatives at knots
    kind: SplineKind,
}

impl CubicSpline {
    /// Fits the spline. `t` must be strictly increasing with at least 2
    /// knots (3 for periodic). For [`SplineKind::Periodic`] `y` must satisfy
    /// `y[last] == y[0]` (it is overwritten to be exact).
    ///
    /// # Panics
    /// On invalid input (length mismatch, too few knots, non-increasing `t`).
    pub fn new(t: &[f64], y: &[f64], kind: SplineKind) -> Self {
        assert_eq!(t.len(), y.len(), "knot/value length mismatch");
        let n = t.len();
        assert!(n >= 2, "need at least 2 knots");
        assert!(t.windows(2).all(|w| w[1] > w[0]), "knots must increase");
        let mut y = y.to_vec();
        let h: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        let slope = |i: usize, y: &[f64]| (y[i + 1] - y[i]) / h[i];
        let m = match kind {
            SplineKind::Natural => {
                let mut a = DMatrix::<f64>::zeros(n, n);
                let mut b = DVector::<f64>::zeros(n);
                a[(0, 0)] = 1.0;
                a[(n - 1, n - 1)] = 1.0;
                for i in 1..n - 1 {
                    a[(i, i - 1)] = h[i - 1];
                    a[(i, i)] = 2.0 * (h[i - 1] + h[i]);
                    a[(i, i + 1)] = h[i];
                    b[i] = 6.0 * (slope(i, &y) - slope(i - 1, &y));
                }
                a.lu()
                    .solve(&b)
                    .expect("spline system singular")
                    .as_slice()
                    .to_vec()
            }
            SplineKind::Periodic => {
                assert!(n >= 3, "periodic spline needs at least 3 knots");
                y[n - 1] = y[0];
                let k = n - 1; // unknowns M_0..M_{k-1}, M_k = M_0
                let mut a = DMatrix::<f64>::zeros(k, k);
                let mut b = DVector::<f64>::zeros(k);
                for i in 0..k {
                    let im1 = (i + k - 1) % k;
                    let hp = h[im1];
                    let hn = h[i];
                    a[(i, im1)] += hp;
                    a[(i, i)] += 2.0 * (hp + hn);
                    a[(i, (i + 1) % k)] += hn;
                    b[i] = 6.0 * (slope(i, &y) - slope(im1, &y));
                }
                let mut m = a
                    .lu()
                    .solve(&b)
                    .expect("spline system singular")
                    .as_slice()
                    .to_vec();
                m.push(m[0]);
                m
            }
        };
        Self {
            t: t.to_vec(),
            y,
            m,
            kind,
        }
    }

    /// First knot time.
    pub fn t_min(&self) -> f64 {
        self.t[0]
    }

    /// Last knot time.
    pub fn t_max(&self) -> f64 {
        *self.t.last().unwrap()
    }

    /// Value, first and second derivative at `t`.
    pub fn eval(&self, t: f64) -> (f64, f64, f64) {
        let (t0, t1) = (self.t_min(), self.t_max());
        let t = match self.kind {
            SplineKind::Natural => t.clamp(t0, t1),
            SplineKind::Periodic => t0 + (t - t0).rem_euclid(t1 - t0),
        };
        let i = match self.t.partition_point(|&k| k <= t) {
            0 => 0,
            p => (p - 1).min(self.t.len() - 2),
        };
        let (a, b) = (self.t[i], self.t[i + 1]);
        let h = b - a;
        let (mi, mj) = (self.m[i], self.m[i + 1]);
        let (yi, yj) = (self.y[i], self.y[i + 1]);
        let (u, v) = (b - t, t - a);
        let ci = yi / h - mi * h / 6.0;
        let cj = yj / h - mj * h / 6.0;
        let val = mi * u * u * u / (6.0 * h) + mj * v * v * v / (6.0 * h) + ci * u + cj * v;
        let d1 = -mi * u * u / (2.0 * h) + mj * v * v / (2.0 * h) - ci + cj;
        let d2 = mi * u / h + mj * v / h;
        (val, d1, d2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_knots_and_is_c2() {
        let t = [0.0, 1.0, 2.5, 3.0, 4.0];
        let y = [0.0, 2.0, -1.0, 0.5, 1.0];
        for kind in [SplineKind::Natural, SplineKind::Periodic] {
            let mut yy = y;
            if kind == SplineKind::Periodic {
                yy[4] = 0.0;
            }
            let s = CubicSpline::new(&t, &yy, kind);
            for (ti, yi) in t.iter().zip(yy) {
                assert!((s.eval(*ti).0 - yi).abs() < 1e-12);
            }
            for &k in &t[1..4] {
                let (l, r) = (s.eval(k - 1e-9), s.eval(k + 1e-9));
                assert!((l.1 - r.1).abs() < 1e-6);
                assert!((l.2 - r.2).abs() < 1e-6);
            }
            // Derivatives agree with finite differences.
            let e = 1e-6;
            for &x in &[0.3, 1.7, 3.3] {
                let fd1 = (s.eval(x + e).0 - s.eval(x - e).0) / (2.0 * e);
                let fd2 = (s.eval(x + e).1 - s.eval(x - e).1) / (2.0 * e);
                assert!((fd1 - s.eval(x).1).abs() < 1e-6);
                assert!((fd2 - s.eval(x).2).abs() < 1e-6);
            }
            if kind == SplineKind::Periodic {
                let (a, b) = (s.eval(1e-9), s.eval(4.0 - 1e-9));
                assert!((a.1 - b.1).abs() < 1e-6 && (a.2 - b.2).abs() < 1e-6);
                assert!((s.eval(5.0).0 - s.eval(1.0).0).abs() < 1e-12);
            }
        }
    }
}
