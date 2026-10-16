// Ported from Ceres Solver 2.0.0: internal/ceres/cubic_interpolation_test.cc
//
// Tests for `Grid1D`, `Grid2D`, `CubicInterpolator` and `BiCubicInterpolator`.

#![allow(clippy::float_cmp)]

use ceres::{BiCubicInterpolator, CubicInterpolator, Grid1, Grid1D, Grid2, Grid2D, Jet};

const K_TOLERANCE: f64 = 1e-12;

#[track_caller]
fn expect_near(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} to be within {tol} of {expected} (diff {})",
        (actual - expected).abs()
    );
}

fn norm<const N: usize>(v: &[f64; N]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn grid1d_one_data_dimension() {
    let x: [i32; 3] = [1, 2, 3];
    let grid = Grid1D::<i32, 1>::new(&x, 0, 3);
    for i in 0..3 {
        let mut value = [0.0];
        grid.get_value(i, &mut value);
        assert_eq!(value[0], f64::from(i + 1));
    }
}

#[test]
fn grid1d_one_data_dimension_out_of_bounds() {
    let x: [i32; 3] = [1, 2, 3];
    let grid = Grid1D::<i32, 1>::new(&x, 0, 3);
    let mut value = [0.0];
    grid.get_value(-1, &mut value);
    assert_eq!(value[0], f64::from(x[0]));
    grid.get_value(-2, &mut value);
    assert_eq!(value[0], f64::from(x[0]));
    grid.get_value(3, &mut value);
    assert_eq!(value[0], f64::from(x[2]));
    grid.get_value(4, &mut value);
    assert_eq!(value[0], f64::from(x[2]));
}

#[test]
fn grid1d_two_data_dimension_integer_data_interleaved() {
    #[rustfmt::skip]
    let x: [i32; 6] = [1, 5,
                       2, 6,
                       3, 7];

    let grid = Grid1D::<i32, 2, true>::new(&x, 0, 3);
    for i in 0..3 {
        let mut value = [0.0; 2];
        grid.get_value(i, &mut value);
        assert_eq!(value[0], f64::from(i + 1));
        assert_eq!(value[1], f64::from(i + 5));
    }
}

#[test]
fn grid1d_two_data_dimension_integer_data_stacked() {
    #[rustfmt::skip]
    let x: [i32; 6] = [1, 2, 3,
                       5, 6, 7];

    let grid = Grid1D::<i32, 2, false>::new(&x, 0, 3);
    for i in 0..3 {
        let mut value = [0.0; 2];
        grid.get_value(i, &mut value);
        assert_eq!(value[0], f64::from(i + 1));
        assert_eq!(value[1], f64::from(i + 5));
    }
}

#[test]
fn grid2d_one_data_dimension_row_major() {
    #[rustfmt::skip]
    let x: [i32; 6] = [1, 2, 3,
                       2, 3, 4];
    let grid = Grid2D::<i32, 1, true, true>::new(&x, 0, 2, 0, 3);
    for r in 0..2 {
        for c in 0..3 {
            let mut value = [0.0];
            grid.get_value(r, c, &mut value);
            assert_eq!(value[0], f64::from(r + c + 1));
        }
    }
}

#[test]
fn grid2d_one_data_dimension_row_major_out_of_bounds() {
    #[rustfmt::skip]
    let x: [i32; 6] = [1, 2, 3,
                       2, 3, 4];
    let grid = Grid2D::<i32, 1, true, true>::new(&x, 0, 2, 0, 3);
    let mut value = [0.0];
    let cases: [(i32, i32, usize); 14] = [
        (-1, -1, 0),
        (-1, 0, 0),
        (-1, 1, 1),
        (-1, 2, 2),
        (-1, 3, 2),
        (0, 3, 2),
        (1, 3, 5),
        (2, 3, 5),
        (2, 2, 5),
        (2, 1, 4),
        (2, 0, 3),
        (2, -1, 3),
        (1, -1, 3),
        (0, -1, 0),
    ];
    for (r, c, idx) in cases {
        grid.get_value(r, c, &mut value);
        assert_eq!(value[0], f64::from(x[idx]), "r={r} c={c}");
    }
}

#[test]
fn grid2d_two_data_dimension_row_major_interleaved() {
    #[rustfmt::skip]
    let x: [i32; 12] = [1, 4, 2, 8, 3, 12,
                        2, 8, 3, 12, 4, 16];
    let grid = Grid2D::<i32, 2, true, true>::new(&x, 0, 2, 0, 3);
    for r in 0..2 {
        for c in 0..3 {
            let mut value = [0.0; 2];
            grid.get_value(r, c, &mut value);
            assert_eq!(value[0], f64::from(r + c + 1));
            assert_eq!(value[1], f64::from(4 * (r + c + 1)));
        }
    }
}

#[test]
fn grid2d_two_data_dimension_row_major_stacked() {
    #[rustfmt::skip]
    let x: [i32; 12] = [1,  2,  3,
                        2,  3,  4,
                        4,  8, 12,
                        8, 12, 16];
    let grid = Grid2D::<i32, 2, true, false>::new(&x, 0, 2, 0, 3);
    for r in 0..2 {
        for c in 0..3 {
            let mut value = [0.0; 2];
            grid.get_value(r, c, &mut value);
            assert_eq!(value[0], f64::from(r + c + 1));
            assert_eq!(value[1], f64::from(4 * (r + c + 1)));
        }
    }
}

#[test]
fn grid2d_two_data_dimension_col_major_interleaved() {
    #[rustfmt::skip]
    let x: [i32; 12] = [ 1,  4, 2,  8,
                         2,  8, 3, 12,
                         3, 12, 4, 16];
    let grid = Grid2D::<i32, 2, false, true>::new(&x, 0, 2, 0, 3);
    for r in 0..2 {
        for c in 0..3 {
            let mut value = [0.0; 2];
            grid.get_value(r, c, &mut value);
            assert_eq!(value[0], f64::from(r + c + 1));
            assert_eq!(value[1], f64::from(4 * (r + c + 1)));
        }
    }
}

#[test]
fn grid2d_two_data_dimension_col_major_stacked() {
    #[rustfmt::skip]
    let x: [i32; 12] = [1,   2,
                        2,   3,
                        3,   4,
                        4,   8,
                        8,  12,
                        12, 16];
    let grid = Grid2D::<i32, 2, false, false>::new(&x, 0, 2, 0, 3);
    for r in 0..2 {
        for c in 0..3 {
            let mut value = [0.0; 2];
            grid.get_value(r, c, &mut value);
            assert_eq!(value[0], f64::from(r + c + 1));
            assert_eq!(value[1], f64::from(4 * (r + c + 1)));
        }
    }
}

// CubicInterpolatorTest fixture.
const K_NUM_SAMPLES: i32 = 10;
const K_NUM_TEST_SAMPLES: i32 = 100;

#[allow(clippy::many_single_char_names, clippy::cast_sign_loss)]
fn run_cubic_polynomial_interpolation_test<const D: usize>(a: f64, b: f64, c: f64, d: f64) {
    let mut values = vec![0.0_f64; D * K_NUM_SAMPLES as usize];

    for xi in 0..K_NUM_SAMPLES {
        let x = f64::from(xi);
        for dim in 0..D {
            let dimf = dim as f64;
            values[xi as usize * D + dim] =
                (dimf * dimf + 1.0) * (a * x * x * x + b * x * x + c * x + d);
        }
    }

    let grid = Grid1D::<f64, D>::new(&values, 0, K_NUM_SAMPLES);
    let interpolator = CubicInterpolator::new(grid);

    // Check values in the all the cells but the first and the last ones. In
    // these cells, the interpolated function values should match exactly the
    // values of the function being interpolated.
    //
    // On the boundary, we extrapolate the values of the function on the basis
    // of its first derivative, so we do not expect the function values and
    // its derivatives not to match.
    for j in 0..K_NUM_TEST_SAMPLES {
        let x = 1.0 + 7.0 / f64::from(K_NUM_TEST_SAMPLES - 1) * f64::from(j);
        let mut expected_f = [0.0; D];
        let mut expected_dfdx = [0.0; D];
        let mut f = [0.0; D];
        let mut dfdx = [0.0; D];

        for dim in 0..D {
            let dimf = dim as f64;
            expected_f[dim] = (dimf * dimf + 1.0) * (a * x * x * x + b * x * x + c * x + d);
            expected_dfdx[dim] = (dimf * dimf + 1.0) * (3.0 * a * x * x + 2.0 * b * x + c);
        }

        interpolator.evaluate_impl(x, Some(&mut f), Some(&mut dfdx));
        for dim in 0..D {
            assert!(
                (f[dim] - expected_f[dim]).abs() <= K_TOLERANCE,
                "x: {x} dim: {dim} actual f(x): {} estimated f(x): {}",
                expected_f[dim],
                f[dim]
            );
            assert!(
                (dfdx[dim] - expected_dfdx[dim]).abs() <= K_TOLERANCE,
                "x: {x} dim: {dim} actual df(x)/dx: {} estimated df(x)/dx: {}",
                expected_dfdx[dim],
                dfdx[dim]
            );
        }
    }
}

macro_rules! cubic_polynomial_test {
    ($name:ident, $a:expr, $b:expr, $c:expr, $d:expr) => {
        #[test]
        fn $name() {
            run_cubic_polynomial_interpolation_test::<1>($a, $b, $c, $d);
            run_cubic_polynomial_interpolation_test::<2>($a, $b, $c, $d);
            run_cubic_polynomial_interpolation_test::<3>($a, $b, $c, $d);
        }
    };
}

cubic_polynomial_test!(cubic_interpolator_constant_function, 0.0, 0.0, 0.0, 0.5);
cubic_polynomial_test!(cubic_interpolator_linear_function, 0.0, 0.0, 1.0, 0.5);
cubic_polynomial_test!(cubic_interpolator_quadratic_function, 0.0, 0.4, 1.0, 0.5);

#[test]
fn cubic_interpolator_jet_evaluation() {
    let values: [f64; 8] = [1.0, 2.0, 2.0, 5.0, 3.0, 9.0, 2.0, 7.0];

    let grid = Grid1D::<f64, 2, true>::new(&values, 0, 4);
    let interpolator = CubicInterpolator::new(grid);

    let mut f = [0.0; 2];
    let mut dfdx = [0.0; 2];
    let x = 2.5;
    interpolator.evaluate_impl(x, Some(&mut f), Some(&mut dfdx));

    // Create a Jet with the same scalar part as x, so that the output Jet
    // will be evaluated at x.
    let x_jet = Jet::<4>::new(x, [1.0, 1.1, 1.2, 1.3]);

    let mut f_jets = [Jet::<4>::constant(0.0); 2];
    interpolator.evaluate(&x_jet, &mut f_jets);

    // Check that the scalar part of the Jet is f(x).
    assert_eq!(f_jets[0].a, f[0]);
    assert_eq!(f_jets[1].a, f[1]);

    // Check that the derivative part of the Jet is dfdx * x_jet.v by the
    // chain rule.
    for k in 0..2 {
        let diff: [f64; 4] = std::array::from_fn(|i| f_jets[k].v[i] - dfdx[k] * x_jet.v[i]);
        expect_near(norm(&diff), 0.0, K_TOLERANCE);
    }
}

// BiCubicInterpolatorTest fixture.
const K_NUM_ROWS: i32 = 10;
const K_NUM_COLS: i32 = 10;
const K_NUM_ROW_SAMPLES: i32 = 100;
const K_NUM_COL_SAMPLES: i32 = 100;

type Matrix3 = [[f64; 3]; 3];

/// x^T * coeff * x with x = (r, c, 1).
fn evaluate_f(coeff: &Matrix3, r: f64, c: f64) -> f64 {
    let x = [r, c, 1.0];
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += x[i] * coeff[i][j] * x[j];
        }
    }
    s
}

/// (coeff.row(0) + coeff.col(0)^T) * x.
fn evaluate_dfdr(coeff: &Matrix3, r: f64, c: f64) -> f64 {
    let x = [r, c, 1.0];
    (0..3).map(|j| (coeff[0][j] + coeff[j][0]) * x[j]).sum()
}

/// (coeff.row(1) + coeff.col(1)^T) * x.
fn evaluate_dfdc(coeff: &Matrix3, r: f64, c: f64) -> f64 {
    let x = [r, c, 1.0];
    (0..3).map(|j| (coeff[1][j] + coeff[j][1]) * x[j]).sum()
}

fn run_bicubic_polynomial_interpolation_test<const D: usize>(coeff: &Matrix3) {
    let mut values = Vec::with_capacity((K_NUM_ROWS * K_NUM_COLS) as usize * D);
    for r in 0..K_NUM_ROWS {
        for c in 0..K_NUM_COLS {
            for dim in 0..D {
                let dimf = dim as f64;
                values.push((dimf * dimf + 1.0) * evaluate_f(coeff, f64::from(r), f64::from(c)));
            }
        }
    }

    let grid = Grid2D::<f64, D>::new(&values, 0, K_NUM_ROWS, 0, K_NUM_COLS);
    let interpolator = BiCubicInterpolator::new(grid);

    for j in 0..K_NUM_ROW_SAMPLES {
        let r = 1.0 + 7.0 / f64::from(K_NUM_ROW_SAMPLES - 1) * f64::from(j);
        for k in 0..K_NUM_COL_SAMPLES {
            let c = 1.0 + 7.0 / f64::from(K_NUM_COL_SAMPLES - 1) * f64::from(k);
            let mut f = [0.0; D];
            let mut dfdr = [0.0; D];
            let mut dfdc = [0.0; D];
            interpolator.evaluate_impl(r, c, &mut f, Some(&mut dfdr), Some(&mut dfdc));
            for dim in 0..D {
                let scale = (dim * dim + 1) as f64;
                expect_near(f[dim], scale * evaluate_f(coeff, r, c), K_TOLERANCE);
                expect_near(dfdr[dim], scale * evaluate_dfdr(coeff, r, c), K_TOLERANCE);
                expect_near(dfdc[dim], scale * evaluate_dfdc(coeff, r, c), K_TOLERANCE);
            }
        }
    }
}

macro_rules! bicubic_polynomial_test {
    ($name:ident, |$coeff:ident| $body:block) => {
        #[test]
        fn $name() {
            let mut $coeff: Matrix3 = [[0.0; 3]; 3];
            $body
            run_bicubic_polynomial_interpolation_test::<1>(&$coeff);
            run_bicubic_polynomial_interpolation_test::<2>(&$coeff);
            run_bicubic_polynomial_interpolation_test::<3>(&$coeff);
        }
    };
}

bicubic_polynomial_test!(bicubic_interpolator_zero_function, |coeff| {
    let _ = &mut coeff;
});

bicubic_polynomial_test!(bicubic_interpolator_degree00_function, |coeff| {
    coeff[2][2] = 1.0;
});

bicubic_polynomial_test!(bicubic_interpolator_degree01_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][2] = 0.1;
    coeff[2][0] = 0.1;
});

bicubic_polynomial_test!(bicubic_interpolator_degree10_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][1] = 0.1;
    coeff[1][0] = 0.1;
});

bicubic_polynomial_test!(bicubic_interpolator_degree11_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][1] = 0.1;
    coeff[1][0] = 0.1;
    coeff[0][2] = 0.2;
    coeff[2][0] = 0.2;
});

bicubic_polynomial_test!(bicubic_interpolator_degree12_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][1] = 0.1;
    coeff[1][0] = 0.1;
    coeff[0][2] = 0.2;
    coeff[2][0] = 0.2;
    coeff[1][1] = 0.3;
});

bicubic_polynomial_test!(bicubic_interpolator_degree21_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][1] = 0.1;
    coeff[1][0] = 0.1;
    coeff[0][2] = 0.2;
    coeff[2][0] = 0.2;
    coeff[0][0] = 0.3;
});

bicubic_polynomial_test!(bicubic_interpolator_degree22_function, |coeff| {
    coeff[2][2] = 1.0;
    coeff[0][1] = 0.1;
    coeff[1][0] = 0.1;
    coeff[0][2] = 0.2;
    coeff[2][0] = 0.2;
    coeff[0][0] = 0.3;
    coeff[0][1] = -0.4;
    coeff[1][0] = -0.4;
});

#[test]
fn bicubic_interpolator_jet_evaluation() {
    #[rustfmt::skip]
    let values: [f64; 16] = [1.0, 5.0, 2.0, 10.0, 2.0, 6.0, 3.0, 5.0,
                             1.0, 2.0, 2.0,  2.0, 2.0, 2.0, 3.0, 1.0];

    let grid = Grid2D::<f64, 2>::new(&values, 0, 2, 0, 4);
    let interpolator = BiCubicInterpolator::new(grid);

    let mut f = [0.0; 2];
    let mut dfdr = [0.0; 2];
    let mut dfdc = [0.0; 2];
    let r = 0.5;
    let c = 2.5;
    interpolator.evaluate_impl(r, c, &mut f, Some(&mut dfdr), Some(&mut dfdc));

    // Create a Jet with the same scalar part as x, so that the output Jet
    // will be evaluated at x.
    let r_jet = Jet::<4>::new(r, [1.0, 1.1, 1.2, 1.3]);
    let c_jet = Jet::<4>::new(c, [2.0, 3.1, 4.2, 5.3]);

    let mut f_jets = [Jet::<4>::constant(0.0); 2];
    interpolator.evaluate(&r_jet, &c_jet, &mut f_jets);
    assert_eq!(f_jets[0].a, f[0]);
    assert_eq!(f_jets[1].a, f[1]);
    for k in 0..2 {
        let diff: [f64; 4] =
            std::array::from_fn(|i| f_jets[k].v[i] - dfdr[k] * r_jet.v[i] - dfdc[k] * c_jet.v[i]);
        expect_near(norm(&diff), 0.0, K_TOLERANCE);
    }
}
