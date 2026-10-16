//! Tests of the Lua interpreter (language semantics, library, errors).

use super::*;

fn run(src: &str) -> Vec<Value> {
    let mut interp = Interpreter::new();
    match interp.exec(src.as_bytes(), "=test") {
        Ok(v) => v,
        Err(e) => panic!("unexpected Lua error: {e}"),
    }
}

/// Runs `src` and returns the first result converted with `tostring`.
fn s(src: &str) -> String {
    run(src)
        .first()
        .cloned()
        .unwrap_or_default()
        .to_display_string()
        .to_str_lossy()
}

fn n(src: &str) -> f64 {
    match run(src).first() {
        Some(Value::Number(x)) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn err(src: &str) -> String {
    let mut interp = Interpreter::new();
    match interp.exec(src.as_bytes(), "=test") {
        Ok(v) => panic!("expected an error, got {v:?}"),
        Err(e) => e.message(),
    }
}

#[test]
fn arithmetic_and_precedence() {
    assert_eq!(n("return 1 + 2 * 3"), 7.);
    assert_eq!(n("return (1 + 2) * 3"), 9.);
    assert_eq!(n("return 2 ^ 3 ^ 2"), 512.);
    assert_eq!(n("return -2 ^ 2"), -4.);
    assert_eq!(n("return 7 % 3"), 1.);
    assert_eq!(n("return -7 % 3"), 2.);
    assert_eq!(n("return 7 % -3"), -2.);
    assert_eq!(n("return 5.5 % 2"), 1.5);
    assert_eq!(n("return 7 / 2"), 3.5);
    assert_eq!(n("return 1 - 2 - 3"), -4.);
    assert_eq!(n("return 2 * 3 / 4"), 1.5);
    assert_eq!(n("return - - 3"), 3.);
    assert_eq!(n("return '10' + 1"), 11.);
    assert_eq!(n("return '0x10' * 2"), 32.);
    assert_eq!(n("return #'abc' + #{1, 2}"), 5.);
    assert_eq!(s("return 1 / 0"), "inf");
    assert_eq!(s("return -1 / 0"), "-inf");
    assert_eq!(s("return -math.pi / 2."), "-1.5707963267949");
    assert_eq!(s("return 1 .. 2"), "12");
    assert_eq!(s("return 'a' .. 'b' .. 'c' == 'abc'"), "true");
    assert_eq!(s("return 1 .. '' == '1'"), "true");
    assert_eq!(s("return 1 + 2 .. ''"), "3");
    assert_eq!(s("return 0.1 .. ''"), "0.1");
    assert_eq!(s("return 1e100 .. ''"), "1e+100");
    assert_eq!(s("return 10 / 2 .. ''"), "5");
}

#[test]
fn comparison_and_logic() {
    assert_eq!(s("return 1 < 2, 2 <= 2, 3 > 4, 4 >= 4"), "true");
    assert_eq!(
        run("return 1 < 2, 2 <= 2, 3 > 4, 4 >= 4")
            .iter()
            .map(|v| v.truthy())
            .collect::<Vec<_>>(),
        vec![true, true, false, true]
    );
    assert_eq!(s("return 'a' < 'b'"), "true");
    assert_eq!(s("return 'Z' < 'a'"), "true");
    assert_eq!(s("return 'abc' < 'abd'"), "true");
    assert_eq!(s("return 1 == 1.0"), "true");
    assert_eq!(s("return '1' == 1"), "false");
    assert_eq!(s("return {} == {}"), "false");
    assert_eq!(s("local t = {} return t == t"), "true");
    assert_eq!(s("return 0/0 ~= 0/0"), "true");
    assert_eq!(s("return nil or false"), "false");
    assert_eq!(s("return false or nil"), "nil");
    assert_eq!(s("return 1 and 2"), "2");
    assert_eq!(s("return nil and error('x')"), "nil");
    assert_eq!(s("return 1 or error('x')"), "1");
    assert_eq!(s("return not nil, not 0"), "true");
    assert_eq!(s("return 1 < 2 == true"), "true");
    assert_eq!(s("return not 1 == 2"), "false");
    assert_eq!(s("return 1 == 1 and 2 == 2 or false"), "true");
}

#[test]
fn locals_globals_and_assignment() {
    assert_eq!(n("x = 1 local x = 2 return x"), 2.);
    assert_eq!(n("x = 1 do local x = 2 end return x"), 1.);
    assert_eq!(
        n("local a, b, c = 1, 2 return c == nil and a + b or -1"),
        3.
    );
    assert_eq!(n("local a, b = 1 return b == nil and a or -1"), 1.);
    assert_eq!(n("local a = 1, 2 return a"), 1.);
    assert_eq!(n("local a, b = 1, 2 a, b = b, a return a * 10 + b"), 21.);
    assert_eq!(n("local x = 1 local x = x + 1 return x"), 2.);
    assert_eq!(n("a, a = 1, 2 return a"), 1.);
    assert_eq!(
        n("A = { b = { c = 1 } } A.b.c = 5 A['b'].d = 2 return A.b.c + A.b.d"),
        7.
    );
    assert_eq!(n("local t = {} t[1], t[2] = 3, 4 return t[1] + t[2]"), 7.);
    assert_eq!(s("x = nil return x"), "nil");
    assert_eq!(s("return undefined_global"), "nil");
    assert_eq!(s("return _G._G == _G"), "true");
    assert_eq!(n("_G.y = 3 return y"), 3.);
    assert_eq!(s("return _VERSION"), "Lua 5.2");
}

#[test]
fn tables() {
    assert_eq!(
        n("local t = { 1, 2, 3; x = 1, ['y'] = 2, [10] = 3 } return #t"),
        3.
    );
    assert_eq!(n("local t = { [1] = 'a', 'b' } return #t"), 1.);
    assert_eq!(s("local t = { [1] = 'a', 'b' } return t[1]"), "b");
    assert_eq!(s("local t = { 'b', [1] = 'a' } return t[1]"), "b");
    assert_eq!(
        n("local function f() return 1, 2, 3 end local t = { f() } return #t"),
        3.
    );
    assert_eq!(
        n("local function f() return 1, 2, 3 end local t = { f(), 10 } return #t"),
        2.
    );
    assert_eq!(
        n("local function f() return 1, 2, 3 end local t = { (f()) } return #t"),
        1.
    );
    assert_eq!(
        n("local t = {} for i = 1, 120 do t[#t + 1] = i end return #t"),
        120.
    );
    let big: Vec<String> = (1..=120).map(|i| i.to_string()).collect();
    assert_eq!(n(&format!("return #{{ {} }}", big.join(", "))), 120.);
    assert_eq!(s("local t = {x = 1} t.x = nil return next(t)"), "nil");
    assert_eq!(n("local t = { [1.0] = 4 } return t[1]"), 4.);
    // Tables are references.
    assert_eq!(n("A = { v = 1 } B = { a = A } A.v = 2 return B.a.v"), 2.);
    assert_eq!(err("local t = {} t[nil] = 1"), "test:1: table index is nil");
    assert_eq!(err("local t = {} t[0/0] = 1"), "test:1: table index is NaN");
    assert_eq!(err("return { [nil] = 1 }"), "test:1: table index is nil");
    assert_eq!(
        err("local t = {} t[nil] = nil"),
        "test:1: table index is nil"
    );
}

#[test]
fn control_flow() {
    assert_eq!(
        n("local s = 0 for i = 1, 10 do s = s + i end return s"),
        55.
    );
    assert_eq!(
        n("local s = 0 for i = 10, 1, -2 do s = s + i end return s"),
        30.
    );
    assert_eq!(n("local s = 0 for i = 1, 0 do s = s + 1 end return s"), 0.);
    assert_eq!(
        n("local s = 0 for i = 0, 1, 0.25 do s = s + 1 end return s"),
        5.
    );
    assert_eq!(
        n("local s = 0 for i = '1', '3' do s = s + i end return s"),
        6.
    );
    assert_eq!(
        n("local s = 0 for i = 1, 10 do if i > 3 then break end s = s + i end return s"),
        6.
    );
    assert_eq!(n("local i = 0 while i < 5 do i = i + 1 end return i"), 5.);
    assert_eq!(
        n("local i = 0 while true do i = i + 1 if i == 7 then break end end return i"),
        7.
    );
    assert_eq!(
        n("local i = 0 repeat local j = i i = i + 1 until j >= 3 return i"),
        4.
    );
    assert_eq!(
        n("local i = 0 repeat i = i + 1 if i == 2 then break end until false return i"),
        2.
    );
    assert_eq!(
        s(
            "local function f(x) if x < 0 then return 'neg' elseif x == 0 then return 'zero' else return 'pos' end end return f(-1) .. f(0) .. f(1)"
        ),
        "negzeropos"
    );
    assert_eq!(
        n("for i = 1, 3 do if i == 2 then return i * 100 end end return 0"),
        200.
    );
    assert_eq!(
        n("local i = 0 while true do i = i + 1 if i > 2 then return i end end"),
        3.
    );
    assert_eq!(
        err("for i = 1, 'x' do end"),
        "test:1: 'for' limit must be a number"
    );
    assert_eq!(
        err("for i = nil, 2 do end"),
        "test:1: 'for' initial value must be a number"
    );
    assert_eq!(
        err("for i = 1, 2, {} do end"),
        "test:1: 'for' step must be a number"
    );
}

#[test]
fn generic_for_pairs_ipairs() {
    assert_eq!(
        n("local s = 0 for i, v in ipairs({ 10, 20, 30, nil, 50 }) do s = s + i * v end return s"),
        140.
    );
    assert_eq!(
        n("local s = 0 for k, v in pairs({ a = 1, b = 2, 3 }) do s = s + v end return s"),
        6.
    );
    assert_eq!(
        s(
            "local keys = {} for k in pairs({ x = 1, y = 2, z = 3 }) do keys[#keys + 1] = k end table.sort(keys) return table.concat(keys, ',')"
        ),
        "x,y,z"
    );
    // Clearing fields during traversal is allowed.
    assert_eq!(
        s("local t = { a = 1, b = 2, c = 3 } for k in pairs(t) do t[k] = nil end return next(t)"),
        "nil"
    );
    // Custom stateless iterator.
    assert_eq!(
        n(
            "local function iter(max, i) if i < max then return i + 1 end end local s = 0 for i in iter, 4, 0 do s = s + i end return s"
        ),
        10.
    );
    assert_eq!(
        err("for x in 1 do end"),
        "test:1: attempt to call a number value"
    );
    assert_eq!(err("next({}, 'nope')"), "test:1: invalid key to 'next'");
    assert_eq!(
        err("pairs(nil)"),
        "test:1: bad argument #1 to 'pairs' (table expected, got nil)"
    );
}

#[test]
fn functions_closures_varargs() {
    assert_eq!(
        n("local function f(a, b) return a + b end return f(1, 2)"),
        3.
    );
    assert_eq!(
        n(
            "local function fact(n) if n <= 1 then return 1 end return n * fact(n - 1) end return fact(10)"
        ),
        3_628_800.
    );
    assert_eq!(
        n(
            "local function counter() local c = 0 return function() c = c + 1 return c end end local a, b = counter(), counter() a() a() b() return a() * 10 + b()"
        ),
        32.
    );
    // Each loop iteration has its own variable.
    assert_eq!(
        n(
            "local fs = {} for i = 1, 3 do fs[i] = function() return i end end return fs[1]() + fs[2]() * 10 + fs[3]() * 100"
        ),
        321.
    );
    assert_eq!(
        n("local function f(...) return select('#', ...) end return f(1, nil, 3, nil)"),
        4.
    );
    assert_eq!(
        n("local function f(...) local a, b = ... return a + b end return f(3, 4, 5)"),
        7.
    );
    assert_eq!(
        n("local function f(...) return ... end return select(2, f(1, 2, 3))"),
        2.
    );
    assert_eq!(n("return select(-1, 1, 2, 3)"), 3.);
    assert_eq!(
        n("local function f(...) local t = { ... } return #t end return f(1, 2, 3)"),
        3.
    );
    assert_eq!(s("local function f(a, ...) return a end return f()"), "nil");
    assert_eq!(run("local function f() end return f()").len(), 0);
    assert_eq!(
        run("local function f() return 1, 2, 3 end return f()").len(),
        3
    );
    assert_eq!(
        run("local function f() return 1, 2, 3 end return (f())").len(),
        1
    );
    assert_eq!(
        run("local function f() return 1, 2, 3 end return f(), 10").len(),
        2
    );
    assert_eq!(run("return ...").len(), 0);
    // Function definition sugar.
    assert_eq!(
        n(
            "local obj = { v = 5 } function obj.get(self) return self.v end function obj:get2() return self.v * 2 end return obj:get() + obj.get2(obj)"
        ),
        15.
    );
    assert_eq!(
        n("a = { b = {} } function a.b.f() return 42 end return a.b.f()"),
        42.
    );
    assert_eq!(n("function g() return 1 end return g()"), 1.);
    // Call sugar with string and table literals.
    assert_eq!(
        n("local function f(t) return t.x end return f{ x = 9 }"),
        9.
    );
    assert_eq!(
        s("local function f(s) return s .. '!' end return f'hi'"),
        "hi!"
    );
    assert_eq!(
        s("local function f(s) return s .. '!' end return f[[long]]"),
        "long!"
    );
    // Methods on strings.
    assert_eq!(s("return ('x'):rep(3)"), "xxx");
    assert_eq!(s("local s = 'abc' return s:upper()"), "ABC");
    assert_eq!(n("local s = 'abc' return s:len()"), 3.);
}

#[test]
fn recursion_limit() {
    let message = err("local function f() return f() + 1 end return f()");
    assert!(message.ends_with("stack overflow"), "{message}");
}

#[test]
fn runtime_errors() {
    assert_eq!(
        err("x.y = 1"),
        "test:1: attempt to index a nil value (global 'x')"
    );
    assert_eq!(
        err("local x; x.y = 1"),
        "test:1: attempt to index a nil value (local 'x')"
    );
    assert_eq!(
        err("a = {} return a.b.c"),
        "test:1: attempt to index a nil value (field 'b')"
    );
    assert_eq!(
        err("local u; local function f() return u.x end return f()"),
        "test:1: attempt to index a nil value (upvalue 'u')"
    );
    assert_eq!(
        err("foo()"),
        "test:1: attempt to call a nil value (global 'foo')"
    );
    assert_eq!(
        err("local t = {} t.m()"),
        "test:1: attempt to call a nil value (field 'm')"
    );
    assert_eq!(
        err("local t = {} t:m()"),
        "test:1: attempt to call a nil value (method 'm')"
    );
    assert_eq!(
        err("return 1 + nil"),
        "test:1: attempt to perform arithmetic on a nil value"
    );
    assert_eq!(
        err("return x + 1"),
        "test:1: attempt to perform arithmetic on a nil value (global 'x')"
    );
    assert_eq!(
        err("local t = {} return 1 + t.x"),
        "test:1: attempt to perform arithmetic on a nil value (field 'x')"
    );
    assert_eq!(
        err("return -{}"),
        "test:1: attempt to perform arithmetic on a table value"
    );
    assert_eq!(
        err("return 'a' + 1"),
        "test:1: attempt to perform arithmetic on a string value"
    );
    assert_eq!(
        err("return 'a' .. {}"),
        "test:1: attempt to concatenate a table value"
    );
    assert_eq!(
        err("return nil .. 'a'"),
        "test:1: attempt to concatenate a nil value"
    );
    assert_eq!(
        err("return 1 < 'x'"),
        "test:1: attempt to compare number with string"
    );
    assert_eq!(
        err("return {} < {}"),
        "test:1: attempt to compare two table values"
    );
    assert_eq!(
        err("return nil > 1"),
        "test:1: attempt to compare number with nil"
    );
    assert_eq!(
        err("return #nil"),
        "test:1: attempt to get length of a nil value"
    );
    assert_eq!(
        err("return #y"),
        "test:1: attempt to get length of a nil value (global 'y')"
    );
    assert_eq!(
        err("local x = 1\n\nx()"),
        "test:3: attempt to call a number value (local 'x')"
    );
    assert_eq!(
        err("return (1).x"),
        "test:1: attempt to index a number value"
    );
}

#[test]
fn error_pcall_assert() {
    assert_eq!(err("error('boom')"), "test:1: boom");
    assert_eq!(err("error('boom', 0)"), "boom");
    assert_eq!(err("\nerror('boom', 1)"), "test:2: boom");
    assert_eq!(
        err("local function f() error('deep', 2) end\nf()"),
        "test:2: deep"
    );
    assert_eq!(err("error({})"), "(error object is a table value)");
    assert_eq!(err("error()"), "(error object is a nil value)");
    assert_eq!(err("assert(false)"), "test:1: assertion failed!");
    assert_eq!(err("assert(nil, 'custom')"), "test:1: custom");
    assert_eq!(n("return assert(5)"), 5.);
    assert_eq!(s("return pcall(error, 'x')"), "false");
    // Level 1 is pcall (a native function), so no position is added.
    assert_eq!(s("local ok, e = pcall(error, 'x') return e"), "x");
    assert_eq!(
        s("local ok, e = pcall(1) return e"),
        "attempt to call a number value"
    );
    assert_eq!(s("local ok, e = pcall(error, 'x', 0) return e"), "x");
    assert_eq!(
        s("local ok, e = pcall(function() error('y') end) return e"),
        "test:1: y"
    );
    assert_eq!(
        s("local ok, e = pcall(function() local t = nil; return t.x end) return e"),
        "test:1: attempt to index a nil value (local 't')"
    );
    assert_eq!(
        n("local ok, a, b = pcall(function() return 1, 2 end) return a + b"),
        3.
    );
    assert_eq!(
        s("local ok, e = pcall(error, { code = 7 }) return e.code"),
        "7"
    );
    // The call stack is restored after an error.
    assert_eq!(
        s(
            "local function f() error('z') end for i = 1, 300 do pcall(f) end local ok, e = pcall(f) return e"
        ),
        "test:1: z"
    );
}

#[test]
fn chunk_names_in_errors() {
    let mut interp = Interpreter::new();
    let e = interp.exec(b"x = = 1", "x = = 1").unwrap_err();
    assert_eq!(
        e.message(),
        "[string \"x = = 1\"]:1: unexpected symbol near '='"
    );
    let e = interp
        .exec(b"\nerror('a')", "/path/to/file.lua")
        .unwrap_err();
    assert_eq!(e.message(), "[string \"/path/to/file.lua\"]:2: a");
    let e = interp.exec(b"error('a')", "@file.lua").unwrap_err();
    assert_eq!(e.message(), "file.lua:1: a");
}

#[test]
fn base_library() {
    assert_eq!(s("return type(nil), type(1)"), "nil");
    assert_eq!(
        run("return type(nil), type(true), type(1), type('x'), type({}), type(print)")
            .iter()
            .map(|v| v.to_display_string().to_str_lossy())
            .collect::<Vec<_>>(),
        vec!["nil", "boolean", "number", "string", "table", "function"]
    );
    assert_eq!(
        err("type()"),
        "test:1: bad argument #1 to 'type' (value expected)"
    );
    assert_eq!(s("return tostring(12)"), "12");
    assert_eq!(s("return tostring(1.5)"), "1.5");
    assert_eq!(s("return tostring(nil)"), "nil");
    assert_eq!(s("return tostring(true)"), "true");
    assert!(s("return tostring({})").starts_with("table: 0x"));
    assert!(s("return tostring(print)").starts_with("function: 0x"));
    assert_eq!(n("return tonumber('  12  ')"), 12.);
    assert_eq!(n("return tonumber('0x1F')"), 31.);
    assert_eq!(n("return tonumber('1e2')"), 100.);
    assert_eq!(s("return tonumber('abc')"), "nil");
    assert_eq!(s("return tonumber('')"), "nil");
    assert_eq!(s("return tonumber({})"), "nil");
    assert_eq!(n("return tonumber('ff', 16)"), 255.);
    assert_eq!(n("return tonumber('-101', 2)"), -5.);
    assert_eq!(n("return tonumber('zz', 36)"), 1295.);
    assert_eq!(s("return tonumber('8', 8)"), "nil");
    assert_eq!(
        err("return tonumber('1', 99)"),
        "test:1: bad argument #2 to 'tonumber' (base out of range)"
    );
    assert_eq!(n("return rawlen({1, 2})"), 2.);
    assert_eq!(
        n("local t = {} rawset(t, 'a', 3) return rawget(t, 'a')"),
        3.
    );
    assert_eq!(s("return rawequal({}, {})"), "false");
    assert_eq!(n("return select('#')"), 0.);
    assert_eq!(
        err("return select(0, 1)"),
        "test:1: bad argument #1 to 'select' (index out of range)"
    );
    assert_eq!(n("local a, b, c = unpack({1, 2, 3}) return a + b + c"), 6.);
    assert_eq!(
        n("local a, b = table.unpack({1, 2, 3}, 2) return a + b"),
        5.
    );
    assert_eq!(run("return unpack({1, 2, 3}, 2, 1)").len(), 0);
    assert_eq!(run("print('printing', 1, nil, true)").len(), 0);
}

#[test]
fn math_library() {
    assert_eq!(n("return math.pi"), std::f64::consts::PI);
    assert_eq!(n("return math.huge"), f64::INFINITY);
    assert_eq!(n("return -math.huge"), f64::NEG_INFINITY);
    assert_eq!(
        n("return math.rad(180)"),
        180. * (std::f64::consts::PI / 180.)
    );
    assert_eq!(
        n("return math.rad(30.)"),
        30. * (std::f64::consts::PI / 180.)
    );
    assert_eq!(n("return math.deg(math.pi)"), 180.);
    assert_eq!(n("return math.floor(-1.5)"), -2.);
    assert_eq!(n("return math.ceil(1.2)"), 2.);
    assert_eq!(n("return math.sqrt(16)"), 4.);
    assert_eq!(n("return math.abs(-3)"), 3.);
    assert_eq!(n("return math.min(3, 1, 2)"), 1.);
    assert_eq!(n("return math.max(3, 1, 2)"), 3.);
    assert_eq!(n("return math.max('5', 1)"), 5.);
    assert_eq!(n("return math.fmod(7, 3)"), 1.);
    assert_eq!(n("return math.fmod(-7, 3)"), -1.);
    assert_eq!(n("return math.pow(2, 10)"), 1024.);
    assert_eq!(n("return math.log(8, 2)"), 3.);
    assert_eq!(n("return math.log(100, 10)"), 2.);
    assert_eq!(n("return math.exp(0)"), 1.);
    assert_eq!(n("local i, f = math.modf(3.25) return i + f * 100"), 28.);
    assert_eq!(n("local m, e = math.frexp(8) return m * 100 + e"), 54.);
    assert_eq!(n("return math.ldexp(0.5, 4)"), 8.);
    assert_eq!(n("return math.atan2(1, 1)"), std::f64::consts::FRAC_PI_4);
    assert_eq!(n("return math.cos(0) + math.sin(0) + math.tan(0)"), 1.);
    assert_eq!(
        err("return math.floor(nil)"),
        "test:1: bad argument #1 to 'floor' (number expected, got nil)"
    );
    assert_eq!(
        err("return math.max()"),
        "test:1: bad argument #1 to 'max' (number expected, got no value)"
    );
}

#[test]
fn string_library() {
    assert_eq!(n("return string.len('abc')"), 3.);
    assert_eq!(s("return string.sub('hello', 2, 4)"), "ell");
    assert_eq!(s("return string.sub('hello', -3)"), "llo");
    assert_eq!(s("return string.sub('hello', 4, 2)"), "");
    assert_eq!(s("return string.sub('hello', 0)"), "hello");
    assert_eq!(s("return string.upper('aBc')"), "ABC");
    assert_eq!(s("return string.lower('aBc')"), "abc");
    assert_eq!(s("return string.rep('ab', 3, ',')"), "ab,ab,ab");
    assert_eq!(s("return string.rep('ab', 0)"), "");
    assert_eq!(s("return string.reverse('abc')"), "cba");
    assert_eq!(n("return string.byte('A')"), 65.);
    assert_eq!(n("local a, b = string.byte('AB', 1, 2) return a + b"), 131.);
    assert_eq!(s("return string.char(72, 105)"), "Hi");
    assert_eq!(n("return string.find('hello world', 'o w', 1, true)"), 5.);
    assert_eq!(n("return string.find('hello', 'l')"), 3.);
    assert_eq!(s("return string.find('hello', 'z')"), "nil");
    assert!(err("return string.find('hello', 'l+')").contains("patterns are not supported"));
}

#[test]
fn string_format() {
    assert_eq!(s("return string.format('%d', 3.7)"), "3");
    assert_eq!(
        s("return string.format('%5d|%-5d|%05d', 42, 42, -42)"),
        "   42|42   |-0042"
    );
    assert_eq!(s("return string.format('%+d % d', 5, 5)"), "+5  5");
    assert_eq!(
        s("return string.format('%x %X %#x %o', 255, 255, 255, 8)"),
        "ff FF 0xff 10"
    );
    assert_eq!(s("return string.format('%f', 1.5)"), "1.500000");
    assert_eq!(s("return string.format('%.2f', 2.345)"), "2.35");
    assert_eq!(s("return string.format('%10.3f|', math.pi)"), "     3.142|");
    assert_eq!(s("return string.format('%e', 12345.678)"), "1.234568e+04");
    assert_eq!(
        s("return string.format('%g %g %g', 1e20, 0.0001, 100)"),
        "1e+20 0.0001 100"
    );
    assert_eq!(s("return string.format('%.14g', 0.1)"), "0.1");
    assert_eq!(s("return string.format('%s and %s', 'a', 12)"), "a and 12");
    assert_eq!(
        s("return string.format('%5s|%-5s|%.2s', 'ab', 'ab', 'abc')"),
        "   ab|ab   |ab"
    );
    assert_eq!(s("return string.format('%s', nil)"), "nil");
    assert_eq!(s("return string.format('%c%c', 72, 105)"), "Hi");
    assert_eq!(s("return string.format('100%%')"), "100%");
    assert_eq!(s("return ('%d-%d'):format(1, 2)"), "1-2");
    assert_eq!(
        s(r#"return string.format('%q', 'a"b\\c\nd\re\0f\0012')"#),
        "\"a\\\"b\\\\c\\\nd\\13e\\0f\\0012\""
    );
    assert_eq!(
        err("return string.format('%d')"),
        "test:1: bad argument #2 to 'format' (no value)"
    );
    assert_eq!(
        err("return string.format('%y', 1)"),
        "test:1: invalid option '%y' to 'format'"
    );
    assert_eq!(
        err("return string.format('%d', 'x')"),
        "test:1: bad argument #2 to 'format' (number expected, got string)"
    );
    assert_eq!(
        err("return string.format('%123d', 1)"),
        "test:1: invalid format (width or precision too long)"
    );
}

#[test]
fn quoted_strings_round_trip() {
    let mut interp = Interpreter::new();
    let original: Vec<u8> = (0u8..=255).collect();
    let quoted = stdlib::quote_string(&original);
    let mut code = b"return ".to_vec();
    code.extend_from_slice(&quoted);
    let result = interp.exec(&code, "=q").unwrap();
    match &result[0] {
        Value::String(s) => assert_eq!(s.as_bytes(), &original[..]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn table_library() {
    assert_eq!(
        s("local t = {1, 2} table.insert(t, 3) return table.concat(t, ',')"),
        "1,2,3"
    );
    assert_eq!(
        s("local t = {1, 2} table.insert(t, 1, 0) return table.concat(t, ',')"),
        "0,1,2"
    );
    assert_eq!(
        err("local t = {} table.insert(t, 5, 1)"),
        "test:1: bad argument #2 to 'insert' (position out of bounds)"
    );
    assert_eq!(
        err("table.insert({}, 1, 2, 3)"),
        "test:1: wrong number of arguments to 'insert'"
    );
    assert_eq!(
        s("local t = {1, 2, 3} local r = table.remove(t) return r .. ':' .. table.concat(t, ',')"),
        "3:1,2"
    );
    assert_eq!(
        s(
            "local t = {1, 2, 3} local r = table.remove(t, 1) return r .. ':' .. table.concat(t, ',')"
        ),
        "1:2,3"
    );
    assert_eq!(s("local t = {} return table.remove(t)"), "nil");
    assert_eq!(s("return table.concat({1, 'a', 2.5}, '-', 2)"), "a-2.5");
    assert_eq!(s("return table.concat({})"), "");
    assert!(
        err("return table.concat({ {} })")
            .starts_with("test:1: invalid value (at index 1) in table for 'concat'")
    );
    assert_eq!(
        s("local t = {5, 2, 8, 1} table.sort(t) return table.concat(t, ',')"),
        "1,2,5,8"
    );
    assert_eq!(
        s(
            "local t = {5, 2, 8, 1} table.sort(t, function(a, b) return a > b end) return table.concat(t, ',')"
        ),
        "8,5,2,1"
    );
    assert_eq!(
        s("local t = {'b', 'c', 'a'} table.sort(t) return table.concat(t)"),
        "abc"
    );
    assert_eq!(
        err("table.sort({1, 'x'})"),
        "test:1: attempt to compare string with number"
    );
}

#[test]
fn os_library() {
    assert_eq!(
        s("return os.getenv('CARTOGRAPHER_SURELY_NOT_SET_12345')"),
        "nil"
    );
    assert!(n("return os.time()") > 1e9);
    assert!(n("return os.clock()") >= 0.);
}

#[test]
fn comments_and_long_strings_in_code() {
    let src = "--[[ block\ncomment ]] local x = [[\nline1\nline2]] -- trailing\n--[==[ another ]==]\nreturn x";
    assert_eq!(s(src), "line1\nline2");
    assert_eq!(s("return 'tab\\tnew\\nline'"), "tab\tnew\nline");
}

#[test]
fn configuration_style_code() {
    let src = r#"
        POSE_GRAPH = {
          optimize_every_n_nodes = 90,
          constraint_builder = {
            sampling_ratio = 0.3,
            fast_correlative_scan_matcher = {
              angular_search_window = math.rad(30.),
            },
          },
          solver_options = {
            use_nonmonotonic_steps = false;
            max_num_iterations = 10;
          },
        }
        MAP_BUILDER = { pose_graph = POSE_GRAPH, num_background_threads = 4 }
        POSE_GRAPH.optimize_every_n_nodes = 0
        return MAP_BUILDER
    "#;
    let result = run(src);
    let Value::Table(t) = &result[0] else {
        panic!("expected a table");
    };
    let Value::Table(pose_graph) = t.get_str("pose_graph") else {
        panic!("expected a table");
    };
    assert_eq!(
        pose_graph.get_str("optimize_every_n_nodes").to_number(),
        Some(0.)
    );
    let Value::Table(solver) = pose_graph.get_str("solver_options") else {
        panic!("expected a table");
    };
    assert_eq!(solver.get_str("max_num_iterations").to_number(), Some(10.));
}
