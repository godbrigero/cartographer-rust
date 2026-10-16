//! Number <-> string conversions with Lua 5.2 semantics (`luaO_str2d`,
//! `lua_number2str` = `"%.14g"`).

use crate::common::printf::format_g;

/// Converts a number to a string like Lua 5.2 (`LUAI_NUMFFORMAT "%.14g"`).
pub fn number_to_string(n: f64) -> String {
    format_g(n, 14)
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&c| !is_space(c)).unwrap_or(s.len());
    let end = s
        .iter()
        .rposition(|&c| !is_space(c))
        .map_or(start, |p| p + 1);
    &s[start..end]
}

fn hex_value(c: u8) -> Option<u32> {
    (c as char).to_digit(16)
}

/// Lua 5.2 `lua_strx2number`: hexadecimal numerals with optional fraction
/// and binary exponent (`0x1.8p3`).
fn hex_to_number(s: &[u8]) -> Option<f64> {
    let mut i = 0;
    let mut negative = false;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        negative = s[i] == b'-';
        i += 1;
    }
    if !(i + 1 < s.len() && s[i] == b'0' && (s[i + 1] == b'x' || s[i + 1] == b'X')) {
        return None;
    }
    i += 2;
    let mut r = 0f64;
    let mut exponent: i64 = 0;
    let mut any_digit = false;
    while i < s.len() {
        if let Some(d) = hex_value(s[i]) {
            r = r * 16. + f64::from(d);
            any_digit = true;
            i += 1;
        } else {
            break;
        }
    }
    if i < s.len() && s[i] == b'.' {
        i += 1;
        while i < s.len() {
            if let Some(d) = hex_value(s[i]) {
                r = r * 16. + f64::from(d);
                exponent -= 4;
                any_digit = true;
                i += 1;
            } else {
                break;
            }
        }
    }
    if !any_digit {
        return None;
    }
    if i < s.len() && (s[i] == b'p' || s[i] == b'P') {
        i += 1;
        let mut exp_negative = false;
        if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
            exp_negative = s[i] == b'-';
            i += 1;
        }
        let start = i;
        let mut e: i64 = 0;
        while i < s.len() && s[i].is_ascii_digit() {
            e = e.saturating_mul(10).saturating_add(i64::from(s[i] - b'0'));
            i += 1;
        }
        if start == i {
            return None;
        }
        exponent += if exp_negative { -e } else { e };
    }
    if i != s.len() {
        return None;
    }
    let r = r * 2f64.powi(exponent.clamp(-10_000, 10_000) as i32);
    Some(if negative { -r } else { r })
}

/// Converts a string to a number like Lua 5.2 `luaO_str2d` (used for
/// numerals in source code, `tonumber` and string->number coercion):
/// leading/trailing whitespace allowed, decimal (via `strtod`) or
/// hexadecimal, `inf`/`nan` rejected.
pub fn str_to_number(s: &[u8]) -> Option<f64> {
    if s.iter().any(|&c| c == b'n' || c == b'N') {
        // Reject 'inf' and 'nan'.
        return None;
    }
    let t = trim(s);
    if t.is_empty() {
        return None;
    }
    if t.iter().any(|&c| c == b'x' || c == b'X') {
        return hex_to_number(t);
    }
    // strtod accepts: [sign] digits [. digits] [(e|E) [sign] digits], where
    // at least one digit must appear in the mantissa.
    let valid = {
        let mut i = 0;
        if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
            i += 1;
        }
        let mut digits = 0;
        while i < t.len() && t[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
        if i < t.len() && t[i] == b'.' {
            i += 1;
            while i < t.len() && t[i].is_ascii_digit() {
                i += 1;
                digits += 1;
            }
        }
        let mut ok = digits > 0;
        if ok && i < t.len() && (t[i] == b'e' || t[i] == b'E') {
            i += 1;
            if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
                i += 1;
            }
            let start = i;
            while i < t.len() && t[i].is_ascii_digit() {
                i += 1;
            }
            ok = i > start;
        }
        ok && i == t.len()
    };
    if !valid {
        return None;
    }
    std::str::from_utf8(t).ok()?.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_to_number_matches_lua() {
        assert_eq!(str_to_number(b"10"), Some(10.));
        assert_eq!(str_to_number(b"  10  "), Some(10.));
        assert_eq!(str_to_number(b"\t-1.5e3\n"), Some(-1500.));
        assert_eq!(str_to_number(b".5"), Some(0.5));
        assert_eq!(str_to_number(b"5."), Some(5.));
        assert_eq!(str_to_number(b"+5"), Some(5.));
        assert_eq!(str_to_number(b"1e"), None);
        assert_eq!(str_to_number(b"."), None);
        assert_eq!(str_to_number(b""), None);
        assert_eq!(str_to_number(b"   "), None);
        assert_eq!(str_to_number(b"inf"), None);
        assert_eq!(str_to_number(b"nan"), None);
        assert_eq!(str_to_number(b"1 2"), None);
        assert_eq!(str_to_number(b"0x10"), Some(16.));
        assert_eq!(str_to_number(b"0XfF"), Some(255.));
        assert_eq!(str_to_number(b"-0x10"), Some(-16.));
        assert_eq!(str_to_number(b"0x1.8p1"), Some(3.));
        assert_eq!(str_to_number(b"0x.8"), Some(0.5));
        assert_eq!(str_to_number(b"0x"), None);
        assert_eq!(str_to_number(b"0x1p"), None);
        assert_eq!(str_to_number(b"1e400"), Some(f64::INFINITY));
    }

    #[test]
    fn number_to_string_matches_lua() {
        assert_eq!(number_to_string(10.), "10");
        assert_eq!(number_to_string(-0.5), "-0.5");
        assert_eq!(number_to_string(1e15), "1e+15");
        assert_eq!(number_to_string(1e14), "1e+14");
        assert_eq!(number_to_string(123456789012.), "123456789012");
        assert_eq!(number_to_string(0.1 + 0.2), "0.3");
        assert_eq!(number_to_string(f64::INFINITY), "inf");
        assert_eq!(number_to_string(f64::NEG_INFINITY), "-inf");
        assert_eq!(number_to_string(2f64.powi(53)), "9.007199254741e+15");
    }
}
