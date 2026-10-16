// Ported from Ceres Solver 2.0.0: include/ceres/cubic_interpolation.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/cubic_interpolation.h).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Cubic Hermite spline (Catmull-Rom / Keys) interpolation of 1D and 2D grids,
//! usable from autodiff functors.
//!
//! RUST PORT NOTE: C++ overloads `Evaluate(double...)` / `Evaluate(Jet...)`
//! become a single generic [`CubicInterpolator::evaluate`] /
//! [`BiCubicInterpolator::evaluate`] over [`Scalar`]: the value and
//! derivatives are computed at the scalar part, and the result is assembled
//! as `f + dfdr * (r - r.a) + dfdc * (c - c.a)`, which for jets yields exactly
//! `Jet(f, dfdr * r.v + dfdc * c.v)` (as in C++) and for `f64` yields `f`.
//! The C++ `enum { DATA_DIMENSION }` of grids becomes the const generic `D`.

use num_traits::AsPrimitive;

use crate::scalar::Scalar;

/// Given samples from a function sampled at four equally spaced points,
///
/// ```text
///   p0 = f(-1)
///   p1 = f(0)
///   p2 = f(1)
///   p3 = f(2)
/// ```
///
/// Evaluate the cubic Hermite spline (also known as the Catmull-Rom spline)
/// at a point x that lies in the interval [0, 1].
///
/// This is also the interpolation kernel (for the case of a = 0.5) as
/// proposed by R. Keys, in: "Cubic convolution interpolation for digital
/// image processing". IEEE Transactions on Acoustics, Speech, and Signal
/// Processing 29 (6): 1153-1160.
///
/// `f` if not `None` will contain the interpolated function values. `dfdx`
/// if not `None` will contain the interpolated derivative values.
#[inline]
pub fn cubic_hermite_spline<const D: usize>(
    p0: &[f64; D],
    p1: &[f64; D],
    p2: &[f64; D],
    p3: &[f64; D],
    x: f64,
    f: Option<&mut [f64; D]>,
    dfdx: Option<&mut [f64; D]>,
) {
    let mut a = [0.0; D];
    let mut b = [0.0; D];
    let mut c = [0.0; D];
    for i in 0..D {
        // Same association order as the Eigen expressions in C++.
        a[i] = 0.5 * (((-p0[i]) + 3.0 * p1[i]) - 3.0 * p2[i] + p3[i]);
        b[i] = 0.5 * (((2.0 * p0[i]) - 5.0 * p1[i]) + 4.0 * p2[i] - p3[i]);
        c[i] = 0.5 * ((-p0[i]) + p2[i]);
    }
    let d = p1;

    // Use Horner's rule to evaluate the function value and its derivative.

    // f = ax^3 + bx^2 + cx + d
    if let Some(f) = f {
        for i in 0..D {
            f[i] = d[i] + x * (c[i] + x * (b[i] + x * a[i]));
        }
    }

    // dfdx = 3ax^2 + 2bx + c
    if let Some(dfdx) = dfdx {
        for i in 0..D {
            dfdx[i] = c[i] + x * (2.0 * b[i] + 3.0 * a[i] * x);
        }
    }
}

/// An infinite one dimensional grid, as needed by [`CubicInterpolator`].
///
/// `get_value` gives the value of a function f (possibly vector valued, of
/// dimension `D`) for any integer `n`.
pub trait Grid1<const D: usize> {
    /// Writes `f(n)` into `f`.
    fn get_value(&self, n: i32, f: &mut [f64; D]);
}

/// An infinite two dimensional grid, as needed by [`BiCubicInterpolator`].
///
/// `get_value` gives the value of a function f (possibly vector valued, of
/// dimension `D`) for any pair of integers `(row, col)`.
pub trait Grid2<const D: usize> {
    /// Writes `f(row, col)` into `f`.
    fn get_value(&self, row: i32, col: i32, f: &mut [f64; D]);
}

impl<const D: usize, G: Grid1<D> + ?Sized> Grid1<D> for &G {
    fn get_value(&self, n: i32, f: &mut [f64; D]) {
        (**self).get_value(n, f);
    }
}

impl<const D: usize, G: Grid2<D> + ?Sized> Grid2<D> for &G {
    fn get_value(&self, row: i32, col: i32, f: &mut [f64; D]) {
        (**self).get_value(row, col, f);
    }
}

/// `CubicInterpolator` uses cubic Hermite splines to produce a smooth
/// approximation to a [`Grid1`] that can be used to evaluate `f(x)` and
/// `f'(x)` at any point on the real number line.
///
/// ```
/// use ceres::cubic_interpolation::{CubicInterpolator, Grid1D};
/// let data = [1.0, 2.0, 5.0, 6.0];
/// let grid = Grid1D::<f64, 1>::new(&data, 0, 4);
/// let interpolator = CubicInterpolator::new(grid);
/// let (mut f, mut dfdx) = ([0.0], [0.0]);
/// interpolator.evaluate_with_derivative(1.5, &mut f, &mut dfdx);
/// ```
pub struct CubicInterpolator<G, const D: usize> {
    grid: G,
}

impl<G: Grid1<D>, const D: usize> CubicInterpolator<G, D> {
    /// Creates an interpolator over `grid` (pass `&grid` to borrow).
    pub fn new(grid: G) -> Self {
        assert!(D >= 1, "DATA_DIMENSION must be >= 1");
        Self { grid }
    }

    /// Evaluates the interpolated function value and its derivative.
    pub fn evaluate_with_derivative(&self, x: f64, f: &mut [f64; D], dfdx: &mut [f64; D]) {
        self.evaluate_impl(x, Some(f), Some(dfdx));
    }

    /// Evaluates the interpolated function value (and optionally its
    /// derivative), C++ `Evaluate(double x, double* f, double* dfdx)`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn evaluate_impl(&self, x: f64, f: Option<&mut [f64; D]>, dfdx: Option<&mut [f64; D]>) {
        let n = x.floor() as i32;
        let mut p0 = [0.0; D];
        let mut p1 = [0.0; D];
        let mut p2 = [0.0; D];
        let mut p3 = [0.0; D];
        self.grid.get_value(n - 1, &mut p0);
        self.grid.get_value(n, &mut p1);
        self.grid.get_value(n + 1, &mut p2);
        self.grid.get_value(n + 2, &mut p3);
        cubic_hermite_spline(&p0, &p1, &p2, &p3, x - f64::from(n), f, dfdx);
    }

    /// Evaluation for `f64` and jets (autodiff), C++ `Evaluate(const T& x,
    /// T* f)`.
    pub fn evaluate<T: Scalar>(&self, x: &T, f: &mut [T]) {
        let xa = x.scalar_part();
        let mut fx = [0.0; D];
        let mut dfdx = [0.0; D];
        self.evaluate_impl(xa, Some(&mut fx), Some(&mut dfdx));
        let dx = *x - T::constant(xa);
        for i in 0..D {
            f[i] = T::constant(fx[i]) + T::constant(dfdx[i]) * dx;
        }
    }
}

/// `BiCubicInterpolator` uses the cubic convolution interpolation algorithm
/// of R. Keys, to produce a smooth approximation to a [`Grid2`] that can be
/// used to evaluate `f(r,c)`, `df(r,c)/dr` and `df(r,c)/dc` at any point in
/// the real plane.
///
/// For more details on the algorithm used here see: "Cubic convolution
/// interpolation for digital image processing". Robert G. Keys, IEEE Trans.
/// on Acoustics, Speech, and Signal Processing 29 (6): 1153-1160, 1981.
///
/// ```
/// use ceres::cubic_interpolation::{BiCubicInterpolator, Grid2D};
/// let data = [1.0, 3.0, -1.0, 4.0,
///             3.6, 2.1,  4.2, 2.0,
///             2.0, 1.0,  3.1, 5.2];
/// let grid = Grid2D::<f64, 1>::new(&data, 0, 3, 0, 4);
/// let interpolator = BiCubicInterpolator::new(&grid);
/// let (mut f, mut dfdr, mut dfdc) = ([0.0], [0.0], [0.0]);
/// interpolator.evaluate_impl(1.2, 2.5, &mut f, Some(&mut dfdr), Some(&mut dfdc));
/// ```
pub struct BiCubicInterpolator<G, const D: usize> {
    grid: G,
}

impl<G: Grid2<D>, const D: usize> BiCubicInterpolator<G, D> {
    /// Creates an interpolator over `grid` (pass `&grid` to borrow).
    pub fn new(grid: G) -> Self {
        assert!(D >= 1, "DATA_DIMENSION must be >= 1");
        Self { grid }
    }

    /// Evaluate the interpolated function value and/or its derivative. Uses
    /// the nearest point on the grid boundary if r or c is out of bounds.
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    pub fn evaluate_impl(
        &self,
        r: f64,
        c: f64,
        f: &mut [f64; D],
        dfdr: Option<&mut [f64; D]>,
        dfdc: Option<&mut [f64; D]>,
    ) {
        // BiCubic interpolation requires 16 values around the point being
        // evaluated. We will use pij, to indicate the elements of the 4x4
        // grid of values.
        //
        //          col
        //      p00 p01 p02 p03
        // row  p10 p11 p12 p13
        //      p20 p21 p22 p23
        //      p30 p31 p32 p33
        //
        // The point (r,c) being evaluated is assumed to lie in the square
        // defined by p11, p12, p22 and p21.

        let row = r.floor() as i32;
        let col = c.floor() as i32;

        let mut p0 = [0.0; D];
        let mut p1 = [0.0; D];
        let mut p2 = [0.0; D];
        let mut p3 = [0.0; D];

        // Interpolate along each of the four rows, evaluating the function
        // value and the horizontal derivative in each row.
        let mut fr = [[0.0; D]; 4];
        let mut dfdc_r = [[0.0; D]; 4];
        let cc = c - f64::from(col);
        for k in 0..4 {
            let rr = row - 1 + k as i32;
            self.grid.get_value(rr, col - 1, &mut p0);
            self.grid.get_value(rr, col, &mut p1);
            self.grid.get_value(rr, col + 1, &mut p2);
            self.grid.get_value(rr, col + 2, &mut p3);
            let (fk, dk) = (&mut fr[k], &mut dfdc_r[k]);
            cubic_hermite_spline(&p0, &p1, &p2, &p3, cc, Some(fk), Some(dk));
        }

        // Interpolate vertically the interpolated value from each row and
        // compute the derivative along the columns.
        let rr = r - f64::from(row);
        cubic_hermite_spline(&fr[0], &fr[1], &fr[2], &fr[3], rr, Some(f), dfdr);
        if let Some(dfdc) = dfdc {
            // Interpolate vertically the derivative along the columns.
            cubic_hermite_spline(
                &dfdc_r[0],
                &dfdc_r[1],
                &dfdc_r[2],
                &dfdc_r[3],
                rr,
                Some(dfdc),
                None,
            );
        }
    }

    /// Evaluation for `f64` and jets (autodiff), C++
    /// `Evaluate(const T& r, const T& c, T* f)`.
    pub fn evaluate<T: Scalar>(&self, r: &T, c: &T, f: &mut [T]) {
        let ra = r.scalar_part();
        let ca = c.scalar_part();
        let mut frc = [0.0; D];
        let mut dfdr = [0.0; D];
        let mut dfdc = [0.0; D];
        self.evaluate_impl(ra, ca, &mut frc, Some(&mut dfdr), Some(&mut dfdc));
        let dr = *r - T::constant(ra);
        let dc = *c - T::constant(ca);
        for i in 0..D {
            f[i] = T::constant(frc[i]) + T::constant(dfdr[i]) * dr + T::constant(dfdc[i]) * dc;
        }
    }
}

/// An object that implements an infinite one dimensional grid needed by the
/// [`CubicInterpolator`] where the source of the function values is an array
/// of type `T` on the interval `[begin, ..., end - 1]`.
///
/// Since the input array is finite and the grid is infinite, values outside
/// this interval needs to be computed. `Grid1D` uses the value from the
/// nearest edge.
///
/// The function being provided can be vector valued, in which case `D > 1`.
/// The dimensional slices of the function maybe interleaved, or they maybe
/// stacked, i.e, if the function has `D = 2`, if `INTERLEAVED = true`, then
/// it is stored as `f01, f02, f11, f12 ....` and if `INTERLEAVED = false`,
/// then it is stored as `f01, f11, .. fn1, f02, f12, .. , fn2`.
pub struct Grid1D<'a, T, const D: usize = 1, const INTERLEAVED: bool = true> {
    data: &'a [T],
    begin: i32,
    end: i32,
    num_values: i32,
}

impl<'a, T, const D: usize, const INTERLEAVED: bool> Grid1D<'a, T, D, INTERLEAVED> {
    /// Creates the grid over `data` for the interval `[begin, end)`.
    pub fn new(data: &'a [T], begin: i32, end: i32) -> Self {
        assert!(begin < end, "Check failed: begin < end");
        Self {
            data,
            begin,
            end,
            num_values: end - begin,
        }
    }
}

impl<T: AsPrimitive<f64>, const D: usize, const INTERLEAVED: bool> Grid1<D>
    for Grid1D<'_, T, D, INTERLEAVED>
{
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn get_value(&self, n: i32, f: &mut [f64; D]) {
        let idx = (n.max(self.begin).min(self.end - 1) - self.begin) as usize;
        if INTERLEAVED {
            for i in 0..D {
                f[i] = self.data[D * idx + i].as_();
            }
        } else {
            for i in 0..D {
                f[i] = self.data[i * self.num_values as usize + idx].as_();
            }
        }
    }
}

/// An object that implements an infinite two dimensional grid needed by the
/// [`BiCubicInterpolator`] where the source of the function values is an
/// grid of type `T` on the grid
///
/// ```text
///   [(row_start,   col_start), ..., (row_start,   col_end - 1)]
///   [                          ...                            ]
///   [(row_end - 1, col_start), ..., (row_end - 1, col_end - 1)]
/// ```
///
/// Since the input grid is finite and the grid is infinite, values outside
/// this interval needs to be computed. `Grid2D` uses the value from the
/// nearest edge.
///
/// The function being provided can be vector valued, in which case `D > 1`.
/// The data maybe stored in row or column major format and the various
/// dimensional slices of the function maybe interleaved, or they maybe
/// stacked, i.e, if the function has `D = 2`, is stored in row-major format
/// and if `INTERLEAVED = true`, then it is stored as
/// `f001, f002, f011, f012, ...`. If `INTERLEAVED = false`, then it is
/// stored as `f001, f011, ..., fnm1, f002, f012, ...`.
pub struct Grid2D<
    'a,
    T,
    const D: usize = 1,
    const ROW_MAJOR: bool = true,
    const INTERLEAVED: bool = true,
> {
    data: &'a [T],
    row_begin: i32,
    row_end: i32,
    col_begin: i32,
    col_end: i32,
    num_rows: i32,
    num_cols: i32,
    num_values: i32,
}

impl<'a, T, const D: usize, const ROW_MAJOR: bool, const INTERLEAVED: bool>
    Grid2D<'a, T, D, ROW_MAJOR, INTERLEAVED>
{
    /// Creates the grid over `data` for rows `[row_begin, row_end)` and
    /// columns `[col_begin, col_end)`.
    pub fn new(data: &'a [T], row_begin: i32, row_end: i32, col_begin: i32, col_end: i32) -> Self {
        assert!(D >= 1, "Check failed: kDataDimension >= 1");
        assert!(row_begin < row_end, "Check failed: row_begin < row_end");
        assert!(col_begin < col_end, "Check failed: col_begin < col_end");
        let num_rows = row_end - row_begin;
        let num_cols = col_end - col_begin;
        Self {
            data,
            row_begin,
            row_end,
            col_begin,
            col_end,
            num_rows,
            num_cols,
            num_values: num_rows * num_cols,
        }
    }
}

impl<T: AsPrimitive<f64>, const D: usize, const ROW_MAJOR: bool, const INTERLEAVED: bool> Grid2<D>
    for Grid2D<'_, T, D, ROW_MAJOR, INTERLEAVED>
{
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn get_value(&self, r: i32, c: i32, f: &mut [f64; D]) {
        let row_idx = r.max(self.row_begin).min(self.row_end - 1) - self.row_begin;
        let col_idx = c.max(self.col_begin).min(self.col_end - 1) - self.col_begin;

        let n = if ROW_MAJOR {
            self.num_cols * row_idx + col_idx
        } else {
            self.num_rows * col_idx + row_idx
        } as usize;

        if INTERLEAVED {
            for i in 0..D {
                f[i] = self.data[D * n + i].as_();
            }
        } else {
            for i in 0..D {
                f[i] = self.data[i * self.num_values as usize + n].as_();
            }
        }
    }
}
