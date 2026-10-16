// Ported from Ceres Solver 2.0.0: internal/ceres/polynomial.{h,cc} and
// internal/ceres/function_sample.{h,cc}
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/polynomial.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original authors: moll.markus@arcor.de (Markus Moll),
// sameeragarwal@google.com (Sameer Agarwal)

//! Polynomial root finding, minimization and interpolation used by the line
//! search, and the `FunctionSample` it operates on.
//!
//! All polynomials are assumed to be the form
//!
//! ```text
//!   sum_{i=0}^N polynomial(i) x^{N-i}.
//! ```
//!
//! and are given by a vector of coefficients of size N + 1.
//!
//! RUST PORT NOTE: Eigen's `EigenSolver` is replaced by nalgebra's
//! (real Schur based) `complex_eigenvalues`, and `FullPivLU` with a zero
//! threshold by a direct port of Eigen's full pivoting LU (see
//! [`full_piv_lu_solve`]), so rank deficient interpolation systems are
//! solved exactly like Eigen does (the free unknowns are set to zero).

use nalgebra::{DMatrix, DVector};

use crate::stringprintf::fmt_e;

/// FunctionSample is used by the line search routines to store and
/// communicate the value and (optionally) the gradient of the function
/// being minimized (C++ `FunctionSample`).
///
/// Since line search as the name implies happens along a certain
/// line/direction. FunctionSample contains the information in two ways.
/// Information in the ambient space and information along the direction of
/// search.
#[derive(Clone, Debug, Default)]
pub(crate) struct FunctionSample {
    /// x is the location of the sample along the search direction.
    pub x: f64,
    /// Let p be a point and d be the search direction then
    ///
    /// vector_x = p + x * d;
    pub vector_x: Vec<f64>,
    /// True if vector_x has been assigned a valid value.
    pub vector_x_is_valid: bool,
    /// value = f(vector_x)
    pub value: f64,
    /// True of the evaluation was successful and value is a finite number.
    pub value_is_valid: bool,
    /// vector_gradient = Df(vector_position);
    ///
    /// D is the derivative operator.
    pub vector_gradient: Vec<f64>,
    /// True if the vector gradient was evaluated and the evaluation was
    /// successful (the value is a finite number).
    pub vector_gradient_is_valid: bool,
    /// gradient = d.transpose() * vector_gradient
    ///
    /// where d is the search direction.
    pub gradient: f64,
    /// True if the evaluation of the gradient was sucessful and the value is
    /// a finite number.
    pub gradient_is_valid: bool,
}

impl FunctionSample {
    /// C++ `FunctionSample(double x, double value)`.
    pub(crate) fn with_value(x: f64, value: f64) -> Self {
        Self {
            x,
            value,
            value_is_valid: true,
            ..Self::default()
        }
    }

    /// C++ `FunctionSample(double x, double value, double gradient)`.
    pub(crate) fn with_value_and_gradient(x: f64, value: f64, gradient: f64) -> Self {
        Self {
            x,
            value,
            value_is_valid: true,
            gradient,
            gradient_is_valid: true,
            ..Self::default()
        }
    }

    /// C++ `FunctionSample::ToDebugString`.
    pub(crate) fn to_debug_string(&self) -> String {
        format!(
            "[x: {}, value: {}, gradient: {}, value_is_valid: {}, gradient_is_valid: {}]",
            fmt_e(self.x, 0, 8, false),
            fmt_e(self.value, 0, 8, false),
            fmt_e(self.gradient, 0, 8, false),
            i32::from(self.value_is_valid),
            i32::from(self.gradient_is_valid)
        )
    }
}

/// Evaluate the polynomial at x using the Horner scheme.
pub(crate) fn evaluate_polynomial(polynomial: &DVector<f64>, x: f64) -> f64 {
    let mut v = 0.0;
    for &c in polynomial.iter() {
        v = v * x + c;
    }
    v
}

/// Balancing function as described by B. N. Parlett and C. Reinsch,
/// "Balancing a Matrix for Calculation of Eigenvalues and Eigenvectors".
/// In: Numerische Mathematik, Volume 13, Number 4 (1969), 293-304,
/// Springer Berlin / Heidelberg. DOI: 10.1007/BF02165404
fn balance_companion_matrix(companion_matrix: &mut DMatrix<f64>) {
    let mut offdiagonal = companion_matrix.clone();
    offdiagonal.fill_diagonal(0.0);

    let degree = companion_matrix.nrows();

    // gamma <= 1 controls how much a change in the scaling has to lower the
    // 1-norm of the companion matrix to be accepted.
    //
    // gamma = 1 seems to lead to cycles (numerical issues?), so we set it
    // slightly lower.
    let gamma = 0.9;

    // Greedily scale row/column pairs until there is no change.
    loop {
        let mut scaling_has_changed = false;

        for i in 0..degree {
            let row_norm: f64 = offdiagonal.row(i).iter().map(|v| v.abs()).sum();
            let col_norm: f64 = offdiagonal.column(i).iter().map(|v| v.abs()).sum();

            // Decompose row_norm/col_norm into mantissa * 2^exponent, where
            // 0.5 <= mantissa < 1. Discard mantissa (return value of frexp),
            // as only the exponent is needed.
            let exponent = frexp_exponent(row_norm / col_norm) / 2;

            if exponent != 0 {
                let scaled_col_norm = ldexp(col_norm, exponent);
                let scaled_row_norm = ldexp(row_norm, -exponent);
                if scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm) {
                    // Accept the new scaling. (Multiplication by powers of 2
                    // should not introduce rounding errors (ignoring
                    // non-normalized numbers and over- or underflow))
                    scaling_has_changed = true;
                    let row_scale = ldexp(1.0, -exponent);
                    let col_scale = ldexp(1.0, exponent);
                    offdiagonal.row_mut(i).scale_mut(row_scale);
                    offdiagonal.column_mut(i).scale_mut(col_scale);
                }
            }
        }
        if !scaling_has_changed {
            break;
        }
    }

    let diagonal = companion_matrix.diagonal();
    offdiagonal.set_diagonal(&diagonal);
    *companion_matrix = offdiagonal;
    log::trace!("Balanced companion matrix is\n{companion_matrix}");
}

/// The exponent of C `frexp` (0 for zero, NaN and infinities, like glibc).
fn frexp_exponent(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    libm::frexp(x).1
}

/// C `ldexp`.
fn ldexp(x: f64, exponent: i32) -> f64 {
    libm::ldexp(x, exponent)
}

fn build_companion_matrix(polynomial: &DVector<f64>) -> DMatrix<f64> {
    let degree = polynomial.len() - 1;
    let mut companion_matrix = DMatrix::zeros(degree, degree);
    for i in 1..degree {
        companion_matrix[(i, i - 1)] = 1.0;
    }
    // col(degree - 1) = -polynomial.reverse().head(degree)
    for i in 0..degree {
        companion_matrix[(i, degree - 1)] = -polynomial[degree - i];
    }
    companion_matrix
}

/// Remove leading terms with zero coefficients.
fn remove_leading_zeros(polynomial_in: &DVector<f64>) -> DVector<f64> {
    let mut i = 0;
    while i + 1 < polynomial_in.len() && polynomial_in[i] == 0.0 {
        i += 1;
    }
    polynomial_in.rows(i, polynomial_in.len() - i).into_owned()
}

fn find_linear_polynomial_roots(
    polynomial: &DVector<f64>,
    real: Option<&mut DVector<f64>>,
    imaginary: Option<&mut DVector<f64>>,
) {
    assert_eq!(polynomial.len(), 2);
    if let Some(real) = real {
        *real = DVector::from_element(1, -polynomial[1] / polynomial[0]);
    }
    if let Some(imaginary) = imaginary {
        *imaginary = DVector::zeros(1);
    }
}

fn find_quadratic_polynomial_roots(
    polynomial: &DVector<f64>,
    real: Option<&mut DVector<f64>>,
    imaginary: Option<&mut DVector<f64>>,
) {
    assert_eq!(polynomial.len(), 3);
    let a = polynomial[0];
    let b = polynomial[1];
    let c = polynomial[2];
    let d = b * b - 4.0 * a * c;
    let sqrt_d = d.abs().sqrt();
    let mut real = real.map(|r| {
        *r = DVector::zeros(2);
        r
    });
    let imaginary = imaginary.map(|i| {
        *i = DVector::zeros(2);
        i
    });

    // Real roots.
    if d >= 0.0 {
        if let Some(real) = real.as_deref_mut() {
            // Stable quadratic roots according to BKP Horn.
            // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
            if b >= 0.0 {
                real[0] = (-b - sqrt_d) / (2.0 * a);
                real[1] = (2.0 * c) / (-b - sqrt_d);
            } else {
                real[0] = (2.0 * c) / (-b + sqrt_d);
                real[1] = (-b + sqrt_d) / (2.0 * a);
            }
        }
        return;
    }

    // Use the normal quadratic formula for the complex case.
    if let Some(real) = real {
        real[0] = -b / (2.0 * a);
        real[1] = -b / (2.0 * a);
    }
    if let Some(imaginary) = imaginary {
        imaginary[0] = sqrt_d / (2.0 * a);
        imaginary[1] = -sqrt_d / (2.0 * a);
    }
}

/// Use the companion matrix eigenvalues to determine the roots of the
/// polynomial.
///
/// This function returns true on success, false otherwise. Failure
/// indicates that the polynomial is invalid (of size 0) or that the
/// eigenvalues of the companion matrix could not be computed. On failure, a
/// more detailed message will be written to the error log. If real is not
/// `None`, the real parts of the roots will be returned in it. Likewise, if
/// imaginary is not `None`, imaginary parts will be returned in it.
pub(crate) fn find_polynomial_roots(
    polynomial_in: &DVector<f64>,
    real: Option<&mut DVector<f64>>,
    imaginary: Option<&mut DVector<f64>>,
) -> bool {
    if polynomial_in.is_empty() {
        log::error!("Invalid polynomial of size 0 passed to FindPolynomialRoots");
        return false;
    }

    let mut polynomial = remove_leading_zeros(polynomial_in);
    let degree = polynomial.len() - 1;

    log::trace!("Input polynomial: {}", polynomial_in.transpose());
    if polynomial.len() != polynomial_in.len() {
        log::trace!("Trimmed polynomial: {}", polynomial.transpose());
    }

    // Is the polynomial constant?
    if degree == 0 {
        log::warn!("Trying to extract roots from a constant polynomial in FindPolynomialRoots");
        // We return true with no roots, not false, as if the polynomial is
        // constant it is correct that there are no roots. It is not the case
        // that they were there, but that we have failed to extract them.
        //
        // RUST PORT NOTE: the C++ code leaves the outputs untouched; callers
        // pass empty vectors, so they are cleared here.
        if let Some(real) = real {
            *real = DVector::zeros(0);
        }
        if let Some(imaginary) = imaginary {
            *imaginary = DVector::zeros(0);
        }
        return true;
    }

    // Linear
    if degree == 1 {
        find_linear_polynomial_roots(&polynomial, real, imaginary);
        return true;
    }

    // Quadratic
    if degree == 2 {
        find_quadratic_polynomial_roots(&polynomial, real, imaginary);
        return true;
    }

    // The degree is now known to be at least 3. For cubic or higher roots we
    // use the method of companion matrices.

    // RUST PORT NOTE: nalgebra's real Schur iteration (replacing Eigen's
    // EigenSolver) does not resolve the nilpotent companion matrix of a
    // multiple root at zero exactly (it returns roots of magnitude
    // ~sqrt(eps) where Eigen returns 0). Roots at zero are therefore
    // deflated exactly first: trailing zero coefficients are x^k factors.
    let num_zero_roots = polynomial.iter().rev().take_while(|&&c| c == 0.0).count();
    if num_zero_roots > 0 {
        let reduced = polynomial
            .rows(0, polynomial.len() - num_zero_roots)
            .into_owned();
        let mut reduced_real = DVector::zeros(0);
        let mut reduced_imaginary = DVector::zeros(0);
        if reduced.len() > 1
            && !find_polynomial_roots(
                &reduced,
                Some(&mut reduced_real),
                Some(&mut reduced_imaginary),
            )
        {
            return false;
        }
        let append_zeros = |v: &DVector<f64>| {
            DVector::from_iterator(
                v.len() + num_zero_roots,
                v.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, num_zero_roots)),
            )
        };
        if let Some(real) = real {
            *real = append_zeros(&reduced_real);
        }
        if let Some(imaginary) = imaginary {
            *imaginary = append_zeros(&reduced_imaginary);
        }
        return true;
    }

    // Divide by leading term
    let leading_term = polynomial[0];
    polynomial /= leading_term;

    // Build and balance the companion matrix to the polynomial.
    let mut companion_matrix = build_companion_matrix(&polynomial);
    balance_companion_matrix(&mut companion_matrix);

    // Find its (complex) eigenvalues.
    let Some(schur) = nalgebra::linalg::Schur::try_new(companion_matrix, f64::EPSILON, 0) else {
        log::error!("Failed to extract eigenvalues from companion matrix.");
        return false;
    };
    let eigenvalues = schur.complex_eigenvalues();

    // Output roots
    match real {
        Some(real) => *real = eigenvalues.map(|e| e.re),
        None => log::warn!(
            "NULL pointer passed as real argument to FindPolynomialRoots. Real parts of the \
             roots will not be returned."
        ),
    }
    if let Some(imaginary) = imaginary {
        *imaginary = eigenvalues.map(|e| e.im);
    }
    true
}

/// Return the derivative of the given polynomial. It is assumed that the
/// input polynomial is at least of degree zero.
pub(crate) fn differentiate_polynomial(polynomial: &DVector<f64>) -> DVector<f64> {
    assert!(!polynomial.is_empty());
    let degree = polynomial.len() - 1;

    // Degree zero polynomials are constants, and their derivative does not
    // result in a smaller degree polynomial, just a degree zero polynomial
    // with value zero.
    if degree == 0 {
        return DVector::zeros(1);
    }

    DVector::from_fn(degree, |i, _| (degree - i) as f64 * polynomial[i])
}

/// Find the minimum value of the polynomial in the interval [x_min, x_max].
/// The minimum is obtained by computing all the roots of the derivative of
/// the input polynomial. All real roots within the interval [x_min, x_max]
/// are considered as well as the end points x_min and x_max. Since
/// polynomials are differentiable functions, this ensures that the true
/// minimum is found.
///
/// Returns `(optimal_x, optimal_value)`.
pub(crate) fn minimize_polynomial(polynomial: &DVector<f64>, x_min: f64, x_max: f64) -> (f64, f64) {
    // Find the minimum of the polynomial at the two ends.
    //
    // We start by inspecting the middle of the interval. Technically this is
    // not needed, but we do this to make this code as close to the minFunc
    // package as possible.
    let mut optimal_x = (x_min + x_max) / 2.0;
    let mut optimal_value = evaluate_polynomial(polynomial, optimal_x);

    let x_min_value = evaluate_polynomial(polynomial, x_min);
    if x_min_value < optimal_value {
        optimal_value = x_min_value;
        optimal_x = x_min;
    }

    let x_max_value = evaluate_polynomial(polynomial, x_max);
    if x_max_value < optimal_value {
        optimal_value = x_max_value;
        optimal_x = x_max;
    }

    // If the polynomial is linear or constant, we are done.
    if polynomial.len() <= 2 {
        return (optimal_x, optimal_value);
    }

    let derivative = differentiate_polynomial(polynomial);
    let mut roots_real = DVector::zeros(0);
    if !find_polynomial_roots(&derivative, Some(&mut roots_real), None) {
        log::warn!("Unable to find the critical points of the interpolating polynomial.");
        return (optimal_x, optimal_value);
    }

    // This is a bit of an overkill, as some of the roots may actually have a
    // complex part, but its simpler to just check these values.
    for &root in roots_real.iter() {
        if root < x_min || root > x_max {
            continue;
        }

        let value = evaluate_polynomial(polynomial, root);
        if value < optimal_value {
            optimal_value = value;
            optimal_x = root;
        }
    }
    (optimal_x, optimal_value)
}

/// Given a set of function value and/or gradient samples, find a polynomial
/// whose value and gradients are exactly equal to the ones in samples.
///
/// Generally speaking,
///
/// degree = # values + # gradients - 1
///
/// Of course its possible to sample a polynomial any number of times, in
/// which case, generally speaking the spurious higher order coefficients
/// will be zero.
pub(crate) fn find_interpolating_polynomial(samples: &[FunctionSample]) -> DVector<f64> {
    let num_constraints: usize = samples
        .iter()
        .map(|s| usize::from(s.value_is_valid) + usize::from(s.gradient_is_valid))
        .sum();

    // RUST PORT NOTE: C++ computes `degree = num_constraints - 1` as an int;
    // with no constraints Eigen solves a 0x0 system and returns an empty
    // vector, which is what this returns too.
    if num_constraints == 0 {
        return DVector::zeros(0);
    }
    let degree = num_constraints - 1;

    let mut lhs = DMatrix::zeros(num_constraints, num_constraints);
    let mut rhs = DVector::zeros(num_constraints);

    let mut row = 0;
    for sample in samples {
        if sample.value_is_valid {
            for j in 0..=degree {
                lhs[(row, j)] = sample.x.powi((degree - j) as i32);
            }
            rhs[row] = sample.value;
            row += 1;
        }

        if sample.gradient_is_valid {
            for j in 0..degree {
                lhs[(row, j)] = (degree - j) as f64 * sample.x.powi((degree - j - 1) as i32);
            }
            rhs[row] = sample.gradient;
            row += 1;
        }
    }

    // TODO(sameeragarwal): This is a hack.
    // https://github.com/ceres-solver/ceres-solver/issues/248
    full_piv_lu_solve(lhs, &rhs)
}

/// Interpolate the function described by samples with a polynomial, and
/// minimize it on the interval [x_min, x_max]. Depending on the input
/// samples, it is possible that the interpolation or the root finding
/// algorithms may fail due to numerical difficulties. But the function is
/// guaranteed to return its best guess of an answer, by considering the
/// samples and the end points as possible solutions.
///
/// Returns `(optimal_x, optimal_value)`.
pub(crate) fn minimize_interpolating_polynomial(
    samples: &[FunctionSample],
    x_min: f64,
    x_max: f64,
) -> (f64, f64) {
    let polynomial = find_interpolating_polynomial(samples);
    let (mut optimal_x, mut optimal_value) = minimize_polynomial(&polynomial, x_min, x_max);
    for sample in samples {
        if sample.x < x_min || sample.x > x_max {
            continue;
        }

        let value = evaluate_polynomial(&polynomial, sample.x);
        if value < optimal_value {
            optimal_x = sample.x;
            optimal_value = value;
        }
    }
    (optimal_x, optimal_value)
}

/// Solves the square system `a x = b` like Eigen's
/// `FullPivLU<MatrixXd>(a).setThreshold(0.0).solve(b)`: LU decomposition
/// with complete pivoting; the rank is the number of non-zero pivots, and
/// the unknowns beyond the rank are set to zero.
fn full_piv_lu_solve(mut lu: DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    let n = lu.nrows();
    debug_assert_eq!(n, lu.ncols());
    // Row transpositions and column transpositions, as in Eigen.
    let mut row_transpositions = vec![0usize; n];
    let mut col_transpositions = vec![0usize; n];
    let mut nonzero_pivots = n;
    for k in 0..n {
        // Find the biggest coefficient in the remaining bottom-right corner
        // (Eigen scans column-major and keeps the first maximum).
        let mut biggest = 0.0;
        let mut row_of_biggest = k;
        let mut col_of_biggest = k;
        for c in k..n {
            for r in k..n {
                let v = lu[(r, c)].abs();
                if v > biggest {
                    biggest = v;
                    row_of_biggest = r;
                    col_of_biggest = c;
                }
            }
        }
        if biggest == 0.0 {
            // The remaining corner is zero: the rank is k.
            nonzero_pivots = k;
            for i in k..n {
                row_transpositions[i] = i;
                col_transpositions[i] = i;
            }
            break;
        }
        row_transpositions[k] = row_of_biggest;
        col_transpositions[k] = col_of_biggest;
        if k != row_of_biggest {
            lu.swap_rows(k, row_of_biggest);
        }
        if k != col_of_biggest {
            lu.swap_columns(k, col_of_biggest);
        }
        let pivot = lu[(k, k)];
        for r in k + 1..n {
            lu[(r, k)] /= pivot;
        }
        for c in k + 1..n {
            let u = lu[(k, c)];
            for r in k + 1..n {
                let l = lu[(r, k)];
                lu[(r, c)] -= l * u;
            }
        }
    }

    // Permutations P (rows) and Q (columns) from the transpositions, in the
    // order they were applied: row i of the factored matrix is row p[i] of
    // the input, column i is column q[i].
    let mut p: Vec<usize> = (0..n).collect();
    for k in 0..n {
        p.swap(k, row_transpositions[k]);
    }
    let mut q: Vec<usize> = (0..n).collect();
    for k in 0..n {
        q.swap(k, col_transpositions[k]);
    }

    // With threshold 0, the rank is the number of pivots with |p| > 0.
    let rank = (0..nonzero_pivots)
        .filter(|&i| lu[(i, i)].abs() > 0.0)
        .count();
    let mut x = DVector::zeros(n);
    if rank == 0 {
        return x;
    }
    // c = P b.
    let mut c: Vec<f64> = (0..n).map(|i| b[p[i]]).collect();
    // Solve L (unit lower triangular).
    for i in 0..n {
        for j in 0..i.min(n) {
            c[i] -= lu[(i, j)] * c[j];
        }
    }
    // Solve U on the top-left rank x rank block.
    for i in (0..rank).rev() {
        let mut s = c[i];
        for j in i + 1..rank {
            s -= lu[(i, j)] * c[j];
        }
        c[i] = s / lu[(i, i)];
    }
    for i in 0..rank {
        x[q[i]] = c[i];
    }
    x
}

#[cfg(test)]
mod tests {
    //! Ported from Ceres Solver 2.0.0 internal/ceres/polynomial_test.cc.
    use super::*;

    // For IEEE-754 doubles, machine precision is about 2e-16.
    const EPSILON: f64 = 1e-13;
    const EPSILON_LOOSE: f64 = 1e-9;

    // Return the constant polynomial p(x) = 1.23.
    fn constant_polynomial(value: f64) -> DVector<f64> {
        DVector::from_element(1, value)
    }

    // Return the polynomial p(x) = poly(x) * (x - root).
    fn add_real_root(poly: &DVector<f64>, root: f64) -> DVector<f64> {
        let n = poly.len();
        let mut poly2 = DVector::zeros(n + 1);
        for i in 0..n {
            poly2[i] += poly[i];
            poly2[i + 1] -= root * poly[i];
        }
        poly2
    }

    // Return the polynomial
    // p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
    fn add_complex_root_pair(poly: &DVector<f64>, real: f64, imag: f64) -> DVector<f64> {
        let n = poly.len();
        let mut poly2 = DVector::zeros(n + 2);
        // Multiply poly by x^2 - 2real + abs(real,imag)^2
        for i in 0..n {
            poly2[i] += poly[i];
            poly2[i + 1] -= 2.0 * real * poly[i];
            poly2[i + 2] += (real * real + imag * imag) * poly[i];
        }
        poly2
    }

    // Sort the entries in a vector. Needed because the roots are not
    // returned in sorted order.
    fn sort_vector(v: &DVector<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = v.iter().copied().collect();
        out.sort_by(f64::total_cmp);
        out
    }

    // C++ test_util ExpectClose: relative error check.
    fn expect_close(x: f64, y: f64, max_abs_relative_difference: f64) {
        let absolute_difference = (x - y).abs();
        let mut relative_difference = absolute_difference / x.abs().max(y.abs());
        if x == 0.0 || y == 0.0 {
            // If x or y is exactly zero, then relative difference doesn't
            // have any meaning. Take the absolute difference instead.
            relative_difference = absolute_difference;
        }
        assert!(
            relative_difference <= max_abs_relative_difference,
            "x={x} y={y} abs diff={absolute_difference} rel diff={relative_difference}"
        );
    }

    fn expect_arrays_close(x: &[f64], y: &[f64], epsilon: f64) {
        assert_eq!(x.len(), y.len());
        for (a, b) in x.iter().zip(y) {
            expect_close(*a, *b, epsilon);
        }
    }

    // Run a test with the polynomial defined by the N real roots in
    // roots_real. If use_real is false, None is passed as the real argument
    // to find_polynomial_roots. If use_imaginary is false, None is passed as
    // the imaginary argument.
    fn run_polynomial_test_real_roots(
        real_roots: &[f64],
        use_real: bool,
        use_imaginary: bool,
        epsilon: f64,
    ) {
        let mut real = DVector::zeros(0);
        let mut imaginary = DVector::zeros(0);
        let mut poly = constant_polynomial(1.23);
        for &r in real_roots {
            poly = add_real_root(&poly, r);
        }
        let success = find_polynomial_roots(
            &poly,
            use_real.then_some(&mut real),
            use_imaginary.then_some(&mut imaginary),
        );

        assert!(success);
        let n = real_roots.len();
        if use_real {
            assert_eq!(real.len(), n);
            expect_arrays_close(&sort_vector(&real), real_roots, epsilon);
        }
        if use_imaginary {
            assert_eq!(imaginary.len(), n);
            expect_arrays_close(imaginary.as_slice(), &vec![0.0; n], epsilon);
        }
    }

    #[test]
    fn invalid_polynomial_of_zero_length_is_rejected() {
        let poly = DVector::zeros(0);
        let mut real = DVector::zeros(0);
        let mut imag = DVector::zeros(0);
        assert!(!find_polynomial_roots(
            &poly,
            Some(&mut real),
            Some(&mut imag)
        ));
    }

    #[test]
    fn constant_polynomial_returns_no_roots() {
        let poly = constant_polynomial(1.23);
        let mut real = DVector::zeros(0);
        let mut imag = DVector::zeros(0);
        assert!(find_polynomial_roots(
            &poly,
            Some(&mut real),
            Some(&mut imag)
        ));
        assert_eq!(real.len(), 0);
        assert_eq!(imag.len(), 0);
    }

    #[test]
    fn linear_polynomial_with_positive_root_works() {
        run_polynomial_test_real_roots(&[42.42], true, true, EPSILON);
    }

    #[test]
    fn linear_polynomial_with_negative_root_works() {
        run_polynomial_test_real_roots(&[-42.42], true, true, EPSILON);
    }

    #[test]
    fn quadratic_polynomial_with_positive_roots_works() {
        run_polynomial_test_real_roots(&[1.0, 42.42], true, true, EPSILON);
    }

    #[test]
    fn quadratic_polynomial_with_one_negative_root_works() {
        run_polynomial_test_real_roots(&[-42.42, 1.0], true, true, EPSILON);
    }

    #[test]
    fn quadratic_polynomial_with_two_negative_roots_works() {
        run_polynomial_test_real_roots(&[-42.42, -1.0], true, true, EPSILON);
    }

    #[test]
    fn quadratic_polynomial_with_close_roots_works() {
        run_polynomial_test_real_roots(&[42.42, 42.43], true, false, EPSILON_LOOSE);
    }

    #[test]
    fn quadratic_polynomial_with_complex_roots_works() {
        let mut real = DVector::zeros(0);
        let mut imag = DVector::zeros(0);
        let poly = add_complex_root_pair(&constant_polynomial(1.23), 42.42, 4.2);
        assert!(find_polynomial_roots(
            &poly,
            Some(&mut real),
            Some(&mut imag)
        ));
        assert_eq!(real.len(), 2);
        assert_eq!(imag.len(), 2);
        expect_close(real[0], 42.42, EPSILON);
        expect_close(real[1], 42.42, EPSILON);
        expect_close(imag[0].abs(), 4.2, EPSILON);
        expect_close(imag[1].abs(), 4.2, EPSILON);
        expect_close((imag[0] + imag[1]).abs(), 0.0, EPSILON);
    }

    #[test]
    fn quartic_polynomial_works() {
        run_polynomial_test_real_roots(&[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], true, true, EPSILON);
    }

    #[test]
    fn quartic_polynomial_with_two_clusters_of_close_roots_works() {
        run_polynomial_test_real_roots(
            &[1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5],
            true,
            true,
            EPSILON_LOOSE,
        );
    }

    #[test]
    fn quartic_polynomial_with_two_zero_roots_works() {
        run_polynomial_test_real_roots(&[-42.42, 0.0, 0.0, 42.42], true, true, 2.0 * EPSILON_LOOSE);
    }

    #[test]
    fn quartic_monomial_works() {
        run_polynomial_test_real_roots(&[0.0, 0.0, 0.0, 0.0], true, true, EPSILON);
    }

    #[test]
    fn null_pointer_as_imaginary_part_works() {
        run_polynomial_test_real_roots(&[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], true, false, EPSILON);
    }

    #[test]
    fn null_pointer_as_real_part_works() {
        run_polynomial_test_real_roots(&[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], false, true, EPSILON);
    }

    #[test]
    fn both_output_arguments_null_works() {
        run_polynomial_test_real_roots(
            &[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5],
            false,
            false,
            EPSILON,
        );
    }

    #[test]
    fn differentiate_constant_polynomial() {
        // p(x) = 1;
        let derivative = differentiate_polynomial(&DVector::from_vec(vec![1.0]));
        assert_eq!(derivative.len(), 1);
        assert_eq!(derivative[0], 0.0);
    }

    #[test]
    fn differentiate_quadratic_polynomial() {
        // p(x) = x^2 + 2x + 3;
        let derivative = differentiate_polynomial(&DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(derivative.len(), 2);
        assert_eq!(derivative[0], 2.0);
        assert_eq!(derivative[1], 2.0);
    }

    #[test]
    fn minimize_constant_polynomial() {
        // p(x) = 1;
        let polynomial = DVector::from_vec(vec![1.0]);
        let (min_x, max_x) = (0.0, 1.0);
        let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, min_x, max_x);
        assert_eq!(optimal_value, 1.0);
        assert!(optimal_x <= max_x);
        assert!(optimal_x >= min_x);
    }

    #[test]
    fn minimize_linear_polynomial() {
        // p(x) = x - 2
        let polynomial = DVector::from_vec(vec![1.0, 2.0]);
        let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, 0.0, 1.0);
        assert_eq!(optimal_x, 0.0);
        assert_eq!(optimal_value, 2.0);
    }

    #[test]
    fn minimize_quadratic_polynomial() {
        // p(x) = x^2 - 3 x + 2
        // min_x = 3/2
        // min_value = -1/4;
        let polynomial = DVector::from_vec(vec![1.0, -3.0, 2.0]);
        let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, -2.0, 2.0);
        assert_eq!(optimal_x, 3.0 / 2.0);
        assert_eq!(optimal_value, -1.0 / 4.0);

        let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, -2.0, 1.0);
        assert_eq!(optimal_x, 1.0);
        assert_eq!(optimal_value, 0.0);

        let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, 2.0, 3.0);
        assert_eq!(optimal_x, 2.0);
        assert_eq!(optimal_value, 0.0);
    }

    fn sample(x: f64, value: Option<f64>, gradient: Option<f64>) -> FunctionSample {
        FunctionSample {
            x,
            value: value.unwrap_or(0.0),
            value_is_valid: value.is_some(),
            gradient: gradient.unwrap_or(0.0),
            gradient_is_valid: gradient.is_some(),
            ..FunctionSample::default()
        }
    }

    fn assert_polynomial_near(true_polynomial: &[f64], polynomial: &DVector<f64>, tol: f64) {
        let diff = DVector::from_vec(true_polynomial.to_vec()) - polynomial;
        assert!(diff.norm() <= tol, "{polynomial} vs {true_polynomial:?}");
    }

    #[test]
    fn constant_interpolating_polynomial() {
        // p(x) = 1.0
        let samples = vec![sample(1.0, Some(1.0), None)];
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(&[1.0], &polynomial, 1e-15);
    }

    #[test]
    fn linear_interpolating_polynomial() {
        // p(x) = 2x - 1
        let samples = vec![sample(1.0, Some(1.0), Some(2.0))];
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(&[2.0, -1.0], &polynomial, 1e-15);
    }

    #[test]
    fn quadratic_interpolating_polynomial() {
        // p(x) = 2x^2 + 3x + 2
        let samples = vec![
            sample(1.0, Some(7.0), Some(7.0)),
            sample(-3.0, Some(11.0), None),
        ];
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(&[2.0, 3.0, 2.0], &polynomial, 1e-15);
    }

    #[test]
    fn deficient_cubic_interpolating_polynomial() {
        // p(x) = 2x^2 + 3x + 2
        let samples = vec![
            sample(1.0, Some(7.0), Some(7.0)),
            sample(-3.0, Some(11.0), Some(-9.0)),
        ];
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(&[0.0, 2.0, 3.0, 2.0], &polynomial, 1e-14);
    }

    #[test]
    fn cubic_interpolating_polynomial_from_values() {
        // p(x) = x^3 + 2x^2 + 3x + 2
        let true_polynomial = DVector::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
        let samples: Vec<_> = [1.0, -3.0, 2.0, 0.0]
            .iter()
            .map(|&x| sample(x, Some(evaluate_polynomial(&true_polynomial, x)), None))
            .collect();
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(true_polynomial.as_slice(), &polynomial, 1e-14);
    }

    #[test]
    fn cubic_interpolating_polynomial_from_values_and_one_gradient() {
        // p(x) = x^3 + 2x^2 + 3x + 2
        let true_polynomial = DVector::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
        let true_gradient_polynomial = differentiate_polynomial(&true_polynomial);
        let samples = vec![
            sample(1.0, Some(evaluate_polynomial(&true_polynomial, 1.0)), None),
            sample(
                -3.0,
                Some(evaluate_polynomial(&true_polynomial, -3.0)),
                None,
            ),
            sample(
                2.0,
                Some(evaluate_polynomial(&true_polynomial, 2.0)),
                Some(evaluate_polynomial(&true_gradient_polynomial, 2.0)),
            ),
        ];
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(true_polynomial.as_slice(), &polynomial, 1e-14);
    }

    #[test]
    fn cubic_interpolating_polynomial_from_values_and_gradients() {
        // p(x) = x^3 + 2x^2 + 3x + 2
        let true_polynomial = DVector::from_vec(vec![1.0, 2.0, 3.0, 2.0]);
        let true_gradient_polynomial = differentiate_polynomial(&true_polynomial);
        let samples: Vec<_> = [-3.0, 2.0]
            .iter()
            .map(|&x| {
                sample(
                    x,
                    Some(evaluate_polynomial(&true_polynomial, x)),
                    Some(evaluate_polynomial(&true_gradient_polynomial, x)),
                )
            })
            .collect();
        let polynomial = find_interpolating_polynomial(&samples);
        assert_polynomial_near(true_polynomial.as_slice(), &polynomial, 1e-14);
    }

    // RUST PORT NOTE: not in polynomial_test.cc. Checks the Eigen
    // FullPivLU(threshold 0) emulation on a rank deficient system: the free
    // unknown is set to zero.
    #[test]
    fn full_piv_lu_rank_deficient_system() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let b = DVector::from_vec(vec![2.0, 4.0]);
        let x = full_piv_lu_solve(a.clone(), &b);
        assert!((&a * &x - &b).norm() < 1e-14);
        // The largest pivot is a(1,1) = 4, so x[1] is the basic unknown.
        assert_eq!(x[0], 0.0);
        assert_eq!(x[1], 1.0);
        assert_eq!(
            full_piv_lu_solve(DMatrix::zeros(2, 2), &b),
            DVector::zeros(2)
        );
    }

    #[test]
    fn minimize_interpolating_polynomial_cubic() {
        // f(x) = (x - 1)^2 sampled with value and gradient at 0 and 3: the
        // interpolant is exact and its minimum is at 1.
        let samples = vec![
            FunctionSample::with_value_and_gradient(0.0, 1.0, -2.0),
            FunctionSample::with_value_and_gradient(3.0, 4.0, 4.0),
        ];
        let (x, v) = minimize_interpolating_polynomial(&samples, 0.0, 3.0);
        assert!((x - 1.0).abs() < 1e-12);
        assert!(v.abs() < 1e-12);
        // Restricted interval: minimum at the end point closest to 1.
        let (x, _) = minimize_interpolating_polynomial(&samples, 2.0, 3.0);
        assert_eq!(x, 2.0);
        let _ = FunctionSample::with_value(0.0, 0.0).to_debug_string();
    }
}
