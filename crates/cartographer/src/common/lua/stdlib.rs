//! The subset of the Lua 5.2 standard library provided by the interpreter:
//! base functions, `math`, `string` (without patterns), `table` and a few
//! `os` functions.

use std::io::Write;

use super::interpreter::{Interpreter, LuaError};
use super::number::{number_to_string, str_to_number};
use super::value::{FunctionRef, LuaString, TableRef, Value};
use crate::common::printf::{Flags, format_float, format_int, format_str};

type NativeResult = Result<Vec<Value>, LuaError>;

fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).cloned().unwrap_or_default()
}

fn type_name_or_no_value(args: &[Value], i: usize) -> &'static str {
    match args.get(i) {
        None => "no value",
        Some(v) => v.type_name(),
    }
}

/// `luaL_argerror`.
pub fn arg_error(interp: &Interpreter, i: usize, fname: &str, extra: &str) -> LuaError {
    interp.error(format!(
        "bad argument #{} to '{}' ({})",
        i + 1,
        fname,
        extra
    ))
}

fn type_error(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
    expected: &str,
) -> LuaError {
    arg_error(
        interp,
        i,
        fname,
        &format!(
            "{} expected, got {}",
            expected,
            type_name_or_no_value(args, i)
        ),
    )
}

/// `luaL_checknumber`.
pub fn check_number(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<f64, LuaError> {
    arg(args, i)
        .to_number()
        .ok_or_else(|| type_error(interp, args, i, fname, "number"))
}

/// `luaL_optnumber`.
fn opt_number(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
    default: f64,
) -> Result<f64, LuaError> {
    if arg(args, i).is_nil() {
        Ok(default)
    } else {
        check_number(interp, args, i, fname)
    }
}

/// `luaL_checkinteger` (Lua 5.2: C cast, i.e. truncation).
fn check_integer(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<i64, LuaError> {
    Ok(check_number(interp, args, i, fname)? as i64)
}

fn opt_integer(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
    default: i64,
) -> Result<i64, LuaError> {
    if arg(args, i).is_nil() {
        Ok(default)
    } else {
        check_integer(interp, args, i, fname)
    }
}

/// `luaL_checklstring` (numbers are converted).
pub fn check_string(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<LuaString, LuaError> {
    arg(args, i)
        .to_lua_string()
        .ok_or_else(|| type_error(interp, args, i, fname, "string"))
}

fn check_table(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<TableRef, LuaError> {
    match arg(args, i) {
        Value::Table(t) => Ok(t),
        _ => Err(type_error(interp, args, i, fname, "table")),
    }
}

fn check_any(
    interp: &Interpreter,
    args: &[Value],
    i: usize,
    fname: &str,
) -> Result<Value, LuaError> {
    match args.get(i) {
        Some(v) => Ok(v.clone()),
        None => Err(arg_error(interp, i, fname, "value expected")),
    }
}

fn num(n: f64) -> Value {
    Value::Number(n)
}

fn string_value(bytes: Vec<u8>) -> Value {
    Value::String(LuaString::from(bytes))
}

fn set_fn(
    table: &TableRef,
    name: &str,
    f: impl Fn(&mut Interpreter, Vec<Value>) -> NativeResult + 'static,
) {
    table.set_str(name, Value::Function(FunctionRef::native(name, f)));
}

/// Opens all libraries into `interp` (`luaL_openlibs`).
pub fn open_libs(interp: &mut Interpreter) {
    open_base(interp);
    open_math(interp);
    open_string(interp);
    open_table(interp);
    open_os(interp);
}

fn open_base(interp: &mut Interpreter) {
    let g = interp.globals().clone();
    g.set_str("_G", Value::Table(g.clone()));
    g.set_str("_VERSION", Value::str("Lua 5.2"));

    set_fn(&g, "assert", |interp, args| {
        if arg(&args, 0).truthy() {
            return Ok(args);
        }
        check_any(interp, &args, 0, "assert")?;
        let message = match args.get(1) {
            None | Some(Value::Nil) => LuaString::from("assertion failed!"),
            Some(_) => check_string(interp, &args, 1, "assert")?,
        };
        Err(interp.error(message.to_str_lossy()))
    });
    set_fn(&g, "error", |interp, args| {
        let level = opt_integer(interp, &args, 1, "error", 1)?;
        let mut value = arg(&args, 0);
        if let Value::String(s) = &value
            && level > 0
        {
            let mut bytes = interp.where_(level as usize).into_bytes();
            bytes.extend_from_slice(s.as_bytes());
            value = string_value(bytes);
        }
        Err(LuaError { value })
    });
    set_fn(&g, "ipairs", |interp, args| {
        let t = check_any(interp, &args, 0, "ipairs")?;
        let iterator = Value::Function(FunctionRef::native("ipairs_iterator", |interp, args| {
            let i = check_integer(interp, &args, 1, "ipairs_iterator")? + 1;
            let t = check_table(interp, &args, 0, "ipairs_iterator")?;
            let v = t.get_int(i);
            if v.is_nil() {
                Ok(vec![Value::Nil])
            } else {
                Ok(vec![num(i as f64), v])
            }
        }));
        Ok(vec![iterator, t, num(0.)])
    });
    let next = FunctionRef::native("next", |interp, args| {
        let t = check_table(interp, &args, 0, "next")?;
        match t.next(&arg(&args, 1)) {
            Ok(Some((k, v))) => Ok(vec![k, v]),
            Ok(None) => Ok(vec![Value::Nil]),
            Err(_) => Err(interp.error("invalid key to 'next'")),
        }
    });
    g.set_str("next", Value::Function(next.clone()));
    set_fn(&g, "pairs", move |interp, args| {
        let t = check_table(interp, &args, 0, "pairs")?;
        Ok(vec![
            Value::Function(next.clone()),
            Value::Table(t),
            Value::Nil,
        ])
    });
    set_fn(&g, "pcall", |interp, mut args| {
        check_any(interp, &args, 0, "pcall")?;
        let f = args.remove(0);
        match interp.call(&f, args) {
            Ok(mut results) => {
                results.insert(0, Value::Boolean(true));
                Ok(results)
            }
            Err(e) => Ok(vec![Value::Boolean(false), e.value]),
        }
    });
    set_fn(&g, "print", |_, args| {
        let line: Vec<u8> = args
            .iter()
            .map(|v| v.to_display_string().as_bytes().to_vec())
            .collect::<Vec<_>>()
            .join(&b'\t');
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(&line);
        let _ = stdout.write_all(b"\n");
        Ok(Vec::new())
    });
    set_fn(&g, "rawequal", |interp, args| {
        check_any(interp, &args, 0, "rawequal")?;
        check_any(interp, &args, 1, "rawequal")?;
        Ok(vec![Value::Boolean(args[0].raw_equals(&args[1]))])
    });
    set_fn(&g, "rawlen", |interp, args| match arg(&args, 0) {
        Value::Table(t) => Ok(vec![num(t.len() as f64)]),
        Value::String(s) => Ok(vec![num(s.len() as f64)]),
        _ => Err(arg_error(interp, 0, "rawlen", "table or string expected")),
    });
    set_fn(&g, "rawget", |interp, args| {
        let t = check_table(interp, &args, 0, "rawget")?;
        check_any(interp, &args, 1, "rawget")?;
        Ok(vec![t.get(&args[1])])
    });
    set_fn(&g, "rawset", |interp, args| {
        let t = check_table(interp, &args, 0, "rawset")?;
        check_any(interp, &args, 1, "rawset")?;
        check_any(interp, &args, 2, "rawset")?;
        t.set(args[1].clone(), args[2].clone())
            .map_err(|e| interp.error(e.to_string()))?;
        Ok(vec![Value::Table(t)])
    });
    set_fn(&g, "select", |interp, args| {
        let n = args.len() as i64 - 1;
        if let Value::String(s) = arg(&args, 0)
            && s.as_bytes().first() == Some(&b'#')
        {
            return Ok(vec![num(n as f64)]);
        }
        let top = args.len() as i64;
        let mut i = check_integer(interp, &args, 0, "select")?;
        if i < 0 {
            i += top;
        } else if i > top {
            i = top;
        }
        if i < 1 {
            return Err(arg_error(interp, 0, "select", "index out of range"));
        }
        Ok(args[(i as usize).min(args.len())..].to_vec())
    });
    set_fn(&g, "tonumber", |interp, args| {
        if arg(&args, 1).is_nil() {
            check_any(interp, &args, 0, "tonumber")?;
            return Ok(vec![match &args[0] {
                Value::Number(n) => num(*n),
                Value::String(s) => str_to_number(s.as_bytes()).map_or(Value::Nil, num),
                _ => Value::Nil,
            }]);
        }
        let base = check_integer(interp, &args, 1, "tonumber")?;
        let s = check_string(interp, &args, 0, "tonumber")?;
        if !(2..=36).contains(&base) {
            return Err(arg_error(interp, 1, "tonumber", "base out of range"));
        }
        Ok(vec![
            parse_with_base(s.as_bytes(), base as u32).map_or(Value::Nil, num),
        ])
    });
    set_fn(&g, "tostring", |interp, args| {
        let v = check_any(interp, &args, 0, "tostring")?;
        Ok(vec![Value::String(v.to_display_string())])
    });
    set_fn(&g, "type", |interp, args| {
        let v = check_any(interp, &args, 0, "type")?;
        Ok(vec![Value::str(v.type_name())])
    });
    let unpack = FunctionRef::native("unpack", |interp, args| {
        let t = check_table(interp, &args, 0, "unpack")?;
        let i = opt_integer(interp, &args, 1, "unpack", 1)?;
        let j = if arg(&args, 2).is_nil() {
            t.len() as i64
        } else {
            check_integer(interp, &args, 2, "unpack")?
        };
        if i > j {
            return Ok(Vec::new());
        }
        if j - i >= 1_000_000 {
            return Err(interp.error("too many results to unpack"));
        }
        Ok((i..=j).map(|k| t.get_int(k)).collect())
    });
    g.set_str("unpack", Value::Function(unpack));
}

/// `tonumber(s, base)`.
fn parse_with_base(s: &[u8], base: u32) -> Option<f64> {
    let s = std::str::from_utf8(s).ok()?;
    let s = s.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'));
    let (negative, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut n = 0f64;
    for c in digits.chars() {
        let d = c.to_digit(36)?;
        if d >= base {
            return None;
        }
        n = n * f64::from(base) + f64::from(d);
    }
    Some(if negative { -n } else { n })
}

fn open_math(interp: &mut Interpreter) {
    let m = TableRef::new();
    macro_rules! unary {
        ($name:literal, $f:expr) => {
            set_fn(&m, $name, |interp, args| {
                let x = check_number(interp, &args, 0, $name)?;
                let f: fn(f64) -> f64 = $f;
                Ok(vec![num(f(x))])
            });
        };
    }
    unary!("abs", f64::abs);
    unary!("ceil", f64::ceil);
    unary!("floor", f64::floor);
    unary!("sqrt", f64::sqrt);
    unary!("sin", f64::sin);
    unary!("cos", f64::cos);
    unary!("tan", f64::tan);
    unary!("asin", f64::asin);
    unary!("acos", f64::acos);
    unary!("atan", f64::atan);
    unary!("exp", f64::exp);
    unary!("log10", f64::log10);
    unary!("sinh", f64::sinh);
    unary!("cosh", f64::cosh);
    unary!("tanh", f64::tanh);
    unary!("rad", |x| x * (std::f64::consts::PI / 180.0));
    unary!("deg", |x| x / (std::f64::consts::PI / 180.0));
    set_fn(&m, "atan2", |interp, args| {
        let y = check_number(interp, &args, 0, "atan2")?;
        let x = opt_number(interp, &args, 1, "atan2", 1.)?;
        Ok(vec![num(y.atan2(x))])
    });
    set_fn(&m, "fmod", |interp, args| {
        let a = check_number(interp, &args, 0, "fmod")?;
        let b = check_number(interp, &args, 1, "fmod")?;
        Ok(vec![num(a % b)])
    });
    set_fn(&m, "modf", |interp, args| {
        let x = check_number(interp, &args, 0, "modf")?;
        let int_part = x.trunc();
        let frac = if x.is_infinite() { 0. } else { x - int_part };
        Ok(vec![num(int_part), num(frac)])
    });
    set_fn(&m, "pow", |interp, args| {
        let a = check_number(interp, &args, 0, "pow")?;
        let b = check_number(interp, &args, 1, "pow")?;
        Ok(vec![num(a.powf(b))])
    });
    set_fn(&m, "log", |interp, args| {
        let x = check_number(interp, &args, 0, "log")?;
        if arg(&args, 1).is_nil() {
            return Ok(vec![num(x.ln())]);
        }
        let base = check_number(interp, &args, 1, "log")?;
        Ok(vec![num(if base == 10. {
            x.log10()
        } else {
            x.ln() / base.ln()
        })])
    });
    set_fn(&m, "frexp", |interp, args| {
        let x = check_number(interp, &args, 0, "frexp")?;
        let (mantissa, exponent) = frexp(x);
        Ok(vec![num(mantissa), num(f64::from(exponent))])
    });
    set_fn(&m, "ldexp", |interp, args| {
        let x = check_number(interp, &args, 0, "ldexp")?;
        let e = check_integer(interp, &args, 1, "ldexp")?;
        Ok(vec![num(x * 2f64.powi(e.clamp(-10_000, 10_000) as i32))])
    });
    set_fn(&m, "min", |interp, args| {
        let mut result = check_number(interp, &args, 0, "min")?;
        for i in 1..args.len() {
            let x = check_number(interp, &args, i, "min")?;
            if x < result {
                result = x;
            }
        }
        Ok(vec![num(result)])
    });
    set_fn(&m, "max", |interp, args| {
        let mut result = check_number(interp, &args, 0, "max")?;
        for i in 1..args.len() {
            let x = check_number(interp, &args, i, "max")?;
            if x > result {
                result = x;
            }
        }
        Ok(vec![num(result)])
    });
    m.set_str("pi", num(std::f64::consts::PI));
    m.set_str("huge", num(f64::INFINITY));
    interp.globals().set_str("math", Value::Table(m));
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0. || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let mut exponent = x.abs().log2().floor() as i32 + 1;
    let mut mantissa = x / 2f64.powi(exponent);
    // Correct rounding issues at the boundaries.
    while mantissa.abs() >= 1. {
        mantissa /= 2.;
        exponent += 1;
    }
    while mantissa.abs() < 0.5 {
        mantissa *= 2.;
        exponent -= 1;
    }
    (mantissa, exponent)
}

/// `posrelat`: translates a relative string position (negative means back
/// from the end).
fn posrelat(pos: i64, len: usize) -> i64 {
    if pos >= 0 {
        pos
    } else if pos.unsigned_abs() as usize > len {
        0
    } else {
        len as i64 + pos + 1
    }
}

/// Lua 5.2 `addquoted` (`string.format("%q", s)`).
pub fn quote_string(s: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    for (i, &c) in s.iter().enumerate() {
        if c == b'"' || c == b'\\' || c == b'\n' {
            out.push(b'\\');
            out.push(c);
        } else if c == 0 || c < 0x20 || c == 0x7f {
            let next_is_digit = s.get(i + 1).is_some_and(|d| d.is_ascii_digit());
            if next_is_digit {
                out.extend_from_slice(format!("\\{c:03}").as_bytes());
            } else {
                out.extend_from_slice(format!("\\{c}").as_bytes());
            }
        } else {
            out.push(c);
        }
    }
    out.push(b'"');
    out
}

/// `string.format`.
fn string_format(interp: &Interpreter, args: &[Value]) -> NativeResult {
    let fmt = check_string(interp, args, 0, "format")?;
    let fmt = fmt.as_bytes();
    let mut out = Vec::new();
    let mut arg_index = 0;
    let mut i = 0;
    while i < fmt.len() {
        let c = fmt[i];
        i += 1;
        if c != b'%' {
            out.push(c);
            continue;
        }
        if i < fmt.len() && fmt[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        // Parse the specification: flags, width, precision.
        let spec_start = i;
        let mut flags = Flags::default();
        while i < fmt.len() && b"-+ #0".contains(&fmt[i]) {
            match fmt[i] {
                b'-' => flags.left = true,
                b'+' => flags.plus = true,
                b' ' => flags.space = true,
                b'#' => flags.alt = true,
                _ => flags.zero = true,
            }
            i += 1;
        }
        if i - spec_start > 5 {
            return Err(interp.error("invalid format (repeated flags)"));
        }
        let mut width = 0usize;
        let width_start = i;
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            width = width * 10 + usize::from(fmt[i] - b'0');
            i += 1;
        }
        let mut too_long = i - width_start > 2;
        let mut precision = None;
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            let precision_start = i;
            let mut p = 0usize;
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                p = p * 10 + usize::from(fmt[i] - b'0');
                i += 1;
            }
            too_long |= i - precision_start > 2;
            precision = Some(p);
        }
        if too_long {
            return Err(interp.error("invalid format (width or precision too long)"));
        }
        let Some(&conversion) = fmt.get(i) else {
            return Err(interp.error("invalid option '%' to 'format'"));
        };
        i += 1;
        arg_index += 1;
        if arg_index >= args.len() && conversion != b'%' {
            return Err(arg_error(interp, arg_index, "format", "no value"));
        }
        match conversion {
            b'c' => {
                let n = check_integer(interp, args, arg_index, "format")?;
                out.extend(format_str(&[n as u8], flags, width, None));
            }
            b'd' | b'i' => {
                let n = check_number(interp, args, arg_index, "format")?;
                let ni = n as i64;
                let diff = n - ni as f64;
                if !(-1. < diff && diff < 1.) {
                    return Err(arg_error(
                        interp,
                        arg_index,
                        "format",
                        "not a number in proper range",
                    ));
                }
                out.extend_from_slice(
                    format_int(i128::from(ni), 'd', flags, width, precision).as_bytes(),
                );
            }
            b'o' | b'u' | b'x' | b'X' => {
                let n = check_number(interp, args, arg_index, "format")?;
                let ni = n as u64;
                let diff = n - ni as f64;
                if !(-1. < diff && diff < 1.) {
                    return Err(arg_error(
                        interp,
                        arg_index,
                        "format",
                        "not a non-negative number in proper range",
                    ));
                }
                out.extend_from_slice(
                    format_int(i128::from(ni), conversion as char, flags, width, precision)
                        .as_bytes(),
                );
            }
            b'e' | b'E' | b'f' | b'g' | b'G' => {
                let n = check_number(interp, args, arg_index, "format")?;
                out.extend_from_slice(
                    format_float(n, conversion as char, flags, width, precision).as_bytes(),
                );
            }
            b'q' => {
                let s = check_string(interp, args, arg_index, "format")?;
                out.extend(quote_string(s.as_bytes()));
            }
            b's' => {
                let s = args[arg_index].to_display_string();
                out.extend(format_str(s.as_bytes(), flags, width, precision));
            }
            other => {
                return Err(
                    interp.error(format!("invalid option '%{}' to 'format'", other as char))
                );
            }
        }
    }
    Ok(vec![string_value(out)])
}

fn open_string(interp: &mut Interpreter) {
    let s = TableRef::new();
    set_fn(&s, "len", |interp, args| {
        let s = check_string(interp, &args, 0, "len")?;
        Ok(vec![num(s.len() as f64)])
    });
    set_fn(&s, "sub", |interp, args| {
        let s = check_string(interp, &args, 0, "sub")?;
        let len = s.len();
        let mut start = posrelat(check_integer(interp, &args, 1, "sub")?, len);
        let mut end = posrelat(opt_integer(interp, &args, 2, "sub", -1)?, len);
        if start < 1 {
            start = 1;
        }
        if end > len as i64 {
            end = len as i64;
        }
        if start <= end {
            Ok(vec![string_value(
                s.as_bytes()[start as usize - 1..end as usize].to_vec(),
            )])
        } else {
            Ok(vec![Value::str("")])
        }
    });
    set_fn(&s, "upper", |interp, args| {
        let s = check_string(interp, &args, 0, "upper")?;
        Ok(vec![string_value(s.as_bytes().to_ascii_uppercase())])
    });
    set_fn(&s, "lower", |interp, args| {
        let s = check_string(interp, &args, 0, "lower")?;
        Ok(vec![string_value(s.as_bytes().to_ascii_lowercase())])
    });
    set_fn(&s, "rep", |interp, args| {
        let s = check_string(interp, &args, 0, "rep")?;
        let n = check_integer(interp, &args, 1, "rep")?;
        let sep = if arg(&args, 2).is_nil() {
            LuaString::from("")
        } else {
            check_string(interp, &args, 2, "rep")?
        };
        if n <= 0 {
            return Ok(vec![Value::str("")]);
        }
        let total = (s.len() + sep.len()).saturating_mul(n as usize);
        if total >= i32::MAX as usize {
            return Err(interp.error("resulting string too large"));
        }
        let mut out = Vec::with_capacity(total);
        for k in 0..n {
            if k > 0 {
                out.extend_from_slice(sep.as_bytes());
            }
            out.extend_from_slice(s.as_bytes());
        }
        Ok(vec![string_value(out)])
    });
    set_fn(&s, "reverse", |interp, args| {
        let s = check_string(interp, &args, 0, "reverse")?;
        let mut bytes = s.as_bytes().to_vec();
        bytes.reverse();
        Ok(vec![string_value(bytes)])
    });
    set_fn(&s, "byte", |interp, args| {
        let s = check_string(interp, &args, 0, "byte")?;
        let len = s.len();
        let start = posrelat(opt_integer(interp, &args, 1, "byte", 1)?, len);
        let end = if arg(&args, 2).is_nil() {
            start
        } else {
            posrelat(check_integer(interp, &args, 2, "byte")?, len)
        };
        let start = start.max(1);
        let end = end.min(len as i64);
        if start > end {
            return Ok(Vec::new());
        }
        Ok(s.as_bytes()[start as usize - 1..end as usize]
            .iter()
            .map(|&b| num(f64::from(b)))
            .collect())
    });
    set_fn(&s, "char", |interp, args| {
        let mut out = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            let c = check_integer(interp, &args, i, "char")?;
            if !(0..=255).contains(&c) {
                return Err(arg_error(interp, i, "char", "value out of range"));
            }
            out.push(c as u8);
        }
        Ok(vec![string_value(out)])
    });
    set_fn(&s, "format", |interp, args| string_format(interp, &args));
    set_fn(&s, "find", |interp, args| {
        let s = check_string(interp, &args, 0, "find")?;
        let pattern = check_string(interp, &args, 1, "find")?;
        let len = s.len();
        let init = posrelat(opt_integer(interp, &args, 2, "find", 1)?, len);
        let init = if init < 1 { 1 } else { init } as usize;
        if init > len + 1 {
            return Ok(vec![Value::Nil]);
        }
        let plain = arg(&args, 3).truthy();
        let has_specials = pattern.as_bytes().iter().any(|c| b"^$*+?.([%-".contains(c));
        if !plain && has_specials {
            return Err(interp.error("string.find: patterns are not supported (use plain = true)"));
        }
        let haystack = &s.as_bytes()[init - 1..];
        let needle = pattern.as_bytes();
        let found = if needle.is_empty() {
            Some(0)
        } else {
            haystack.windows(needle.len()).position(|w| w == needle)
        };
        Ok(match found {
            Some(p) => vec![
                num((init + p) as f64),
                num((init + p + needle.len() - 1) as f64),
            ],
            None => vec![Value::Nil],
        })
    });
    interp.globals().set_str("string", Value::Table(s.clone()));
    interp.set_string_library(s);
}

fn open_table(interp: &mut Interpreter) {
    let t = TableRef::new();
    set_fn(&t, "insert", |interp, args| {
        let table = check_table(interp, &args, 0, "insert")?;
        let e = table.len() as i64 + 1;
        match args.len() {
            2 => {
                table
                    .set(num(e as f64), args[1].clone())
                    .map_err(|err| interp.error(err.to_string()))?;
            }
            3 => {
                let pos = check_integer(interp, &args, 1, "insert")?;
                if !(1 <= pos && pos <= e) {
                    return Err(arg_error(interp, 1, "insert", "position out of bounds"));
                }
                let mut i = e;
                while i > pos {
                    let v = table.get_int(i - 1);
                    table.set(num(i as f64), v).expect("integer key");
                    i -= 1;
                }
                table
                    .set(num(pos as f64), args[2].clone())
                    .expect("integer key");
            }
            _ => return Err(interp.error("wrong number of arguments to 'insert'")),
        }
        Ok(Vec::new())
    });
    set_fn(&t, "remove", |interp, args| {
        let table = check_table(interp, &args, 0, "remove")?;
        let size = table.len() as i64;
        let mut pos = opt_integer(interp, &args, 1, "remove", size)?;
        if pos != size && !(1 <= pos && pos <= size + 1) {
            return Err(arg_error(interp, 0, "remove", "position out of bounds"));
        }
        let removed = table.get_int(pos);
        while pos < size {
            let v = table.get_int(pos + 1);
            table.set(num(pos as f64), v).expect("integer key");
            pos += 1;
        }
        table.set(num(pos as f64), Value::Nil).expect("integer key");
        Ok(vec![removed])
    });
    set_fn(&t, "concat", |interp, args| {
        let table = check_table(interp, &args, 0, "concat")?;
        let sep = if arg(&args, 1).is_nil() {
            LuaString::from("")
        } else {
            check_string(interp, &args, 1, "concat")?
        };
        let i = opt_integer(interp, &args, 2, "concat", 1)?;
        let j = if arg(&args, 3).is_nil() {
            table.len() as i64
        } else {
            check_integer(interp, &args, 3, "concat")?
        };
        let mut out = Vec::new();
        let mut k = i;
        while k <= j {
            let v = table.get_int(k);
            let s = match &v {
                Value::String(_) | Value::Number(_) => v.to_lua_string().expect("string"),
                other => {
                    return Err(interp.error(format!(
                        "invalid value (at index {k}) in table for 'concat'; got {}",
                        other.type_name()
                    )));
                }
            };
            out.extend_from_slice(s.as_bytes());
            if k != j {
                out.extend_from_slice(sep.as_bytes());
            }
            k += 1;
        }
        Ok(vec![string_value(out)])
    });
    set_fn(&t, "sort", |interp, args| {
        let table = check_table(interp, &args, 0, "sort")?;
        let comparator = arg(&args, 1);
        if !comparator.is_nil() && !matches!(comparator, Value::Function(_)) {
            return Err(type_error(interp, &args, 1, "sort", "function"));
        }
        let n = table.len() as i64;
        let mut values: Vec<Value> = (1..=n).map(|k| table.get_int(k)).collect();
        // Merge sort with fallible comparisons.
        let less = |interp: &mut Interpreter, a: &Value, b: &Value| -> Result<bool, LuaError> {
            if comparator.is_nil() {
                interp
                    .less_than(a, b)
                    .map(|v| v.truthy())
                    .map_err(|m| interp.error(m))
            } else {
                Ok(interp
                    .call(&comparator, vec![a.clone(), b.clone()])?
                    .first()
                    .is_some_and(Value::truthy))
            }
        };
        let mut width = 1;
        while width < values.len() {
            let mut merged = Vec::with_capacity(values.len());
            for chunk_start in (0..values.len()).step_by(2 * width) {
                let mid = (chunk_start + width).min(values.len());
                let end = (chunk_start + 2 * width).min(values.len());
                let (mut a, mut b) = (chunk_start, mid);
                while a < mid && b < end {
                    if less(interp, &values[b], &values[a])? {
                        merged.push(values[b].clone());
                        b += 1;
                    } else {
                        merged.push(values[a].clone());
                        a += 1;
                    }
                }
                merged.extend_from_slice(&values[a..mid]);
                merged.extend_from_slice(&values[b..end]);
            }
            values = merged;
            width *= 2;
        }
        for (k, v) in values.into_iter().enumerate() {
            table.set(num((k + 1) as f64), v).expect("integer key");
        }
        Ok(Vec::new())
    });
    let unpack = interp.globals().get_str("unpack");
    t.set_str("unpack", unpack);
    interp.globals().set_str("table", Value::Table(t));
}

fn open_os(interp: &mut Interpreter) {
    let os = TableRef::new();
    set_fn(&os, "getenv", |interp, args| {
        let name = check_string(interp, &args, 0, "getenv")?;
        Ok(vec![match std::env::var_os(name.to_str_lossy()) {
            Some(v) => Value::String(LuaString::from(v.to_string_lossy().into_owned())),
            None => Value::Nil,
        }])
    });
    set_fn(&os, "time", |interp, args| {
        if !arg(&args, 0).is_nil() {
            return Err(interp.error("os.time with a table argument is not supported"));
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Ok(vec![num(now as f64)])
    });
    set_fn(&os, "clock", |_, _| {
        let t = cpu_time::ProcessTime::try_now().map_or(0., |t| t.as_duration().as_secs_f64());
        Ok(vec![num(t)])
    });
    interp.globals().set_str("os", Value::Table(os));
}

/// `tostring` of a number, exposed for users formatting numbers like Lua.
pub fn lua_number_to_string(n: f64) -> String {
    number_to_string(n)
}
