//! C `printf`-compatible number formatting.
//!
//! RUST PORT NOTE: this file has no C++ counterpart. The C++ code relies on
//! the C library (`std::to_string(double)` = `%f`, Lua's `%.14g`,
//! `std::scientific`) and on absl (`absl::StrCat`/`absl::Substitute` print
//! floating point numbers like `%.6g`). Rust's formatting machinery differs
//! (e.g. `1e-1` vs `1.00e-01`, `NaN` vs `nan`), so the exact C behavior is
//! reimplemented here on top of Rust's correctly rounded formatting (which
//! yields the same digits as glibc/macOS libc for these conversions).

/// Flags of a C conversion specification (`%-+ #0`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `-`: left-justify within the field width.
    pub left: bool,
    /// `+`: always print a sign for signed conversions.
    pub plus: bool,
    /// ` `: print a space if no sign is printed.
    pub space: bool,
    /// `#`: alternate form.
    pub alt: bool,
    /// `0`: pad with zeros instead of spaces.
    pub zero: bool,
}

fn special(x: f64, upper: bool) -> Option<String> {
    let s = if x.is_nan() {
        "nan"
    } else if x.is_infinite() {
        "inf"
    } else {
        return None;
    };
    let s = if upper {
        s.to_uppercase()
    } else {
        s.to_string()
    };
    Some(s)
}

/// Digits of `|x|` in `%.<precision>f` style (no sign).
fn fixed_digits(x: f64, precision: usize, alt: bool) -> String {
    let mut s = format!("{:.*}", precision, x.abs());
    if alt && precision == 0 {
        s.push('.');
    }
    s
}

/// Digits of `|x|` in `%.<precision>e` style (no sign).
fn exp_digits(x: f64, precision: usize, upper: bool, alt: bool) -> String {
    let s = format!("{:.*e}", precision, x.abs());
    let (mantissa, exponent) = s.split_once('e').expect("exponent");
    let exponent: i32 = exponent.parse().expect("exponent");
    let mut mantissa = mantissa.to_string();
    if alt && precision == 0 {
        mantissa.push('.');
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    format!(
        "{}{}{}{:02}",
        mantissa,
        if upper { 'E' } else { 'e' },
        sign,
        exponent.unsigned_abs()
    )
}

fn strip_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let s = s.trim_end_matches('0');
    s.strip_suffix('.').unwrap_or(s).to_string()
}

/// Digits of `|x|` in `%.<precision>g` style (no sign).
fn general_digits(x: f64, precision: usize, upper: bool, alt: bool) -> String {
    let p = if precision == 0 { 1 } else { precision };
    let exponent: i32 = if x == 0.0 {
        0
    } else {
        let s = format!("{:.*e}", p - 1, x.abs());
        s.split_once('e').expect("exponent").1.parse().expect("exp")
    };
    if exponent < -4 || exponent >= p as i32 {
        let s = exp_digits(x, p - 1, upper, alt);
        if alt {
            return s;
        }
        let (mantissa, rest) = s.split_at(s.find(['e', 'E']).expect("exponent"));
        format!("{}{}", strip_trailing_zeros(mantissa), rest)
    } else {
        let decimals = (p as i32 - 1 - exponent) as usize;
        let s = fixed_digits(x, decimals, alt);
        if alt { s } else { strip_trailing_zeros(&s) }
    }
}

fn sign_prefix(negative: bool, flags: Flags) -> &'static str {
    if negative {
        "-"
    } else if flags.plus {
        "+"
    } else if flags.space {
        " "
    } else {
        ""
    }
}

/// Applies the field width to `prefix` (sign etc.) + `body`.
fn pad(prefix: &str, body: &str, width: usize, flags: Flags, zero_allowed: bool) -> String {
    let len = prefix.len() + body.len();
    if len >= width {
        return format!("{prefix}{body}");
    }
    let fill = width - len;
    if flags.left {
        format!("{prefix}{body}{}", " ".repeat(fill))
    } else if flags.zero && zero_allowed {
        format!("{prefix}{}{body}", "0".repeat(fill))
    } else {
        format!("{}{prefix}{body}", " ".repeat(fill))
    }
}

/// Formats `x` like C `printf` with conversion `conv` in `"eEfFgG"`.
/// `precision` defaults to 6 if `None`.
pub fn format_float(
    x: f64,
    conv: char,
    flags: Flags,
    width: usize,
    precision: Option<usize>,
) -> String {
    let upper = conv.is_ascii_uppercase();
    let precision = precision.unwrap_or(6);
    let negative = x.is_sign_negative();
    if let Some(s) = special(x, upper) {
        return pad(sign_prefix(negative, flags), &s, width, flags, false);
    }
    let body = match conv {
        'f' | 'F' => fixed_digits(x, precision, flags.alt),
        'e' | 'E' => exp_digits(x, precision, upper, flags.alt),
        'g' | 'G' => general_digits(x, precision, upper, flags.alt),
        _ => panic!("invalid float conversion '{conv}'"),
    };
    pad(sign_prefix(negative, flags), &body, width, flags, true)
}

/// Formats an integer like C `printf` with conversion `conv` in `"diouxXc"`
/// (`c` is not handled here).
pub fn format_int(
    value: i128,
    conv: char,
    flags: Flags,
    width: usize,
    precision: Option<usize>,
) -> String {
    let negative = value < 0;
    let magnitude = value.unsigned_abs();
    let mut digits = match conv {
        'd' | 'i' | 'u' => magnitude.to_string(),
        'o' => format!("{magnitude:o}"),
        'x' => format!("{magnitude:x}"),
        'X' => format!("{magnitude:X}"),
        _ => panic!("invalid integer conversion '{conv}'"),
    };
    if let Some(p) = precision {
        if p == 0 && magnitude == 0 {
            digits.clear();
        }
        if digits.len() < p {
            digits = format!("{}{digits}", "0".repeat(p - digits.len()));
        }
    }
    let mut prefix = String::new();
    match conv {
        'd' | 'i' => prefix.push_str(sign_prefix(negative, flags)),
        'o' if flags.alt && !digits.starts_with('0') => digits.insert(0, '0'),
        'x' if flags.alt && magnitude != 0 => prefix.push_str("0x"),
        'X' if flags.alt && magnitude != 0 => prefix.push_str("0X"),
        _ => {}
    }
    pad(&prefix, &digits, width, flags, precision.is_none())
}

/// Pads a string like C `printf("%<flags><width>s")`.
pub fn format_str(s: &[u8], flags: Flags, width: usize, precision: Option<usize>) -> Vec<u8> {
    let s = match precision {
        Some(p) if p < s.len() => &s[..p],
        _ => s,
    };
    let mut out = Vec::with_capacity(s.len().max(width));
    let fill = width.saturating_sub(s.len());
    if !flags.left {
        out.extend(std::iter::repeat_n(b' ', fill));
    }
    out.extend_from_slice(s);
    if flags.left {
        out.extend(std::iter::repeat_n(b' ', fill));
    }
    out
}

/// C `%f` (what `std::to_string(double)` produces).
pub fn format_f(x: f64) -> String {
    format_float(x, 'f', Flags::default(), 0, None)
}

/// C `%.<precision>f`.
pub fn format_f_precision(x: f64, precision: usize) -> String {
    format_float(x, 'f', Flags::default(), 0, Some(precision))
}

/// C `%.<precision>e` (what `std::scientific << std::setprecision(p)` produces).
pub fn format_e(x: f64, precision: usize) -> String {
    format_float(x, 'e', Flags::default(), 0, Some(precision))
}

/// C `%.<precision>g`.
pub fn format_g(x: f64, precision: usize) -> String {
    format_float(x, 'g', Flags::default(), 0, Some(precision))
}

/// The way absl (`absl::StrCat`, `absl::Substitute`) prints floating point
/// numbers: `%.6g`.
pub fn format_absl(x: f64) -> String {
    format_g(x, 6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed() {
        assert_eq!(format_f(123.0), "123.000000");
        assert_eq!(format_f(3.1), "3.100000");
        assert_eq!(format_f(1234.456786), "1234.456786");
        assert_eq!(format_f(-0.0), "-0.000000");
        assert_eq!(format_f(1e-7), "0.000000");
        assert_eq!(format_f(f64::INFINITY), "inf");
        assert_eq!(format_f(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_f(f64::NAN), "nan");
        assert_eq!(format_f(-f64::NAN), "-nan");
        assert_eq!(format_f_precision(2.5, 0), "2");
        assert_eq!(format_f_precision(0.125, 2), "0.12");
    }

    #[test]
    fn exponential() {
        assert_eq!(format_e(0.1, 2), "1.00e-01");
        assert_eq!(format_e(0.0, 2), "0.00e+00");
        assert_eq!(format_e(12345.0, 3), "1.234e+04");
        assert_eq!(format_e(1e100, 1), "1.0e+100");
        assert_eq!(format_e(-1e-300, 0), "-1e-300");
    }

    #[test]
    fn general() {
        assert_eq!(format_g(10.0, 14), "10");
        assert_eq!(format_g(0.1, 14), "0.1");
        assert_eq!(format_g(1e100, 14), "1e+100");
        assert_eq!(format_g(1e15, 14), "1e+15");
        assert_eq!(format_g(123456789012345.0, 14), "1.2345678901234e+14");
        assert_eq!(format_g(12345678901234.0, 14), "12345678901234");
        assert_eq!(format_g(0.0001, 14), "0.0001");
        assert_eq!(format_g(0.00001, 14), "1e-05");
        assert_eq!(format_g(-2.5, 14), "-2.5");
        assert_eq!(format_g(std::f64::consts::PI, 14), "3.1415926535898");
        assert_eq!(format_g(0.0, 14), "0");
        assert_eq!(format_g(-0.0, 14), "-0");
        assert_eq!(format_g(1.0 / 3.0, 6), "0.333333");
        assert_eq!(format_g(100000.0, 6), "100000");
        assert_eq!(format_g(1000000.0, 6), "1e+06");
        assert_eq!(format_g(999999.5, 6), "1e+06");
        assert_eq!(
            format_float(
                1.5,
                'g',
                Flags {
                    alt: true,
                    ..Flags::default()
                },
                0,
                None
            ),
            "1.50000"
        );
    }

    #[test]
    fn integers_and_padding() {
        let zero = Flags {
            zero: true,
            ..Flags::default()
        };
        let left = Flags {
            left: true,
            ..Flags::default()
        };
        let plus = Flags {
            plus: true,
            ..Flags::default()
        };
        assert_eq!(format_int(-42, 'd', zero, 6, None), "-00042");
        assert_eq!(format_int(42, 'd', left, 5, None), "42   ");
        assert_eq!(format_int(42, 'd', plus, 0, None), "+42");
        assert_eq!(format_int(42, 'd', Flags::default(), 0, Some(4)), "0042");
        assert_eq!(
            format_int(
                255,
                'x',
                Flags {
                    alt: true,
                    ..Flags::default()
                },
                0,
                None
            ),
            "0xff"
        );
        assert_eq!(format_int(255, 'X', Flags::default(), 0, None), "FF");
        assert_eq!(
            format_int(
                8,
                'o',
                Flags {
                    alt: true,
                    ..Flags::default()
                },
                0,
                None
            ),
            "010"
        );
        assert_eq!(format_float(3.5, 'f', zero, 8, Some(2)), "00003.50");
        assert_eq!(
            format_float(-3.5, 'f', Flags::default(), 8, Some(2)),
            "   -3.50"
        );
        assert_eq!(format_str(b"ab", Flags::default(), 4, None), b"  ab");
        assert_eq!(format_str(b"abcdef", left, 4, Some(3)), b"abc ");
    }
}
