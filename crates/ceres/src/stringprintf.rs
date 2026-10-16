// RUST PORT NOTE: minimal replacement for internal/ceres/stringprintf.cc
// (Ceres Solver 2.0.0): just the printf conversions used by the solver
// reports (`%e`, `%f`, `%d`, `%s` with the ' ' flag, width and precision).

//! printf-style number formatting helpers for solver reports.

/// `%[ ][width].[precision]e`, e.g. `% 3.2e` is `fmt_e(x, 3, 2, true)`.
pub(crate) fn fmt_e(x: f64, width: usize, precision: usize, space: bool) -> String {
    let body = if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    } else {
        let s = format!("{x:.precision$e}");
        let (mantissa, exponent) = s.split_once('e').expect("exponent");
        let exponent: i32 = exponent.parse().expect("exponent");
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    };
    pad(body, x, width, space)
}

/// `%[ ][width].[precision]f`.
pub(crate) fn fmt_f(x: f64, width: usize, precision: usize, space: bool) -> String {
    let body = if x.is_nan() {
        "nan".to_string()
    } else {
        format!("{x:.precision$}")
    };
    pad(body, x, width, space)
}

/// `%[ ][width]d`.
pub(crate) fn fmt_d(x: i64, width: usize, space: bool) -> String {
    #[allow(clippy::cast_precision_loss)]
    pad(x.to_string(), x as f64, width, space)
}

fn pad(mut body: String, x: f64, width: usize, space: bool) -> String {
    if space && !(x.is_sign_negative() && !x.is_nan() && body.starts_with('-')) {
        body.insert(0, ' ');
    }
    format!("{body:>width$}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_printf() {
        assert_eq!(fmt_e(1234.5, 0, 6, false), "1.234500e+03");
        assert_eq!(fmt_e(1234.5, 0, 2, true), " 1.23e+03");
        assert_eq!(fmt_e(-0.00012, 0, 2, true), "-1.20e-04");
        assert_eq!(fmt_e(0.0, 12, 6, true), " 0.000000e+00");
        assert_eq!(fmt_e(1e-100, 0, 1, false), "1.0e-100");
        assert_eq!(fmt_d(5, 4, true), "   5");
        assert_eq!(fmt_d(-5, 4, true), "  -5");
        assert_eq!(fmt_f(0.5, 10, 6, false), "  0.500000");
    }
}
