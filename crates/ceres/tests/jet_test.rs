// Ported from Ceres Solver 2.0.0: internal/ceres/jet_test.cc
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: keir@google.com (Keir Mierle)
//
// The C++ `TEST(Jet, Jet)` is one very long test; here it is split into
// several `#[test]`s inside `mod jet` (one per group of checks), keeping all
// inputs and tolerances. Eigen-specific tests are adapted to nalgebra.
// Additionally, proptest-based finite-difference checks of every jet math
// function and tests of nalgebra generic code on jets are appended at the end
// (not part of the C++ test).

// The C++ inputs use literals like 3.14 / 3.14159 (not meant to be pi), and
// index loops mirror the C++ code / multi-array indexing.
#![allow(clippy::approx_constant, clippy::needless_range_loop)]

use ceres::Jet;

const K_E: f64 = std::f64::consts::E;

type J = Jet<2>;

/// Convenient shorthand for making a jet.
fn make_jet(a: f64, v0: f64, v1: f64) -> J {
    Jet::new(a, [v0, v1])
}

/// On a 32-bit optimized build, the mismatch is about 1.4e-14.
const K_TOLERANCE: f64 = 1e-13;

/// Port of `ExpectClose` from ceres/test_util.cc (panics on failure).
#[track_caller]
fn expect_close(x: f64, y: f64, max_abs_relative_difference: f64) {
    if x.is_infinite() && y.is_infinite() {
        assert_eq!(
            x.is_sign_negative(),
            y.is_sign_negative(),
            "x={x} y={y}: infinities of different sign"
        );
        return;
    }
    if x.is_nan() && y.is_nan() {
        return;
    }
    let absolute_difference = (x - y).abs();
    let mut relative_difference = absolute_difference / x.abs().max(y.abs());
    if x == 0.0 || y == 0.0 {
        // If x or y is exactly zero, then relative difference doesn't have
        // any meaning. Take the absolute difference instead.
        relative_difference = absolute_difference;
    }
    // EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
    // (a NaN relative difference fails, like in gtest).
    assert!(
        relative_difference.abs() <= max_abs_relative_difference,
        "x={x:e} y={y:e} abs={absolute_difference:e} rel={relative_difference:e} \
         tol={max_abs_relative_difference:e}"
    );
}

#[track_caller]
fn expect_jets_close(x: J, y: J) {
    expect_close(x.a, y.a, K_TOLERANCE);
    expect_close(x.v[0], y.v[0], K_TOLERANCE);
    expect_close(x.v[1], y.v[1], K_TOLERANCE);
}

const K_STEP: f64 = 1e-8;
/// Numeric derivation is quite inexact.
const K_NUMERICAL_TOLERANCE: f64 = 1e-6;

/// Differentiate using Jet and confirm results with numerical derivation.
#[track_caller]
fn numerical_test(name: &str, f: impl Fn(J) -> J, x: f64) {
    let exact_dx = f(make_jet(x, 1.0, 0.0)).v[0];
    let estimated_dx =
        (f(J::constant(x + K_STEP)).a - f(J::constant(x - K_STEP)).a) / (2.0 * K_STEP);
    println!("{name}({x}), exact dx: {exact_dx}, estimated dx: {estimated_dx}");
    expect_close(exact_dx, estimated_dx, K_NUMERICAL_TOLERANCE);
}

/// Same as `numerical_test`, but given a function taking two arguments.
#[track_caller]
fn numerical_test2(name: &str, f: impl Fn(J, J) -> J, x: f64, y: f64) {
    let exact_delta = f(make_jet(x, 1.0, 0.0), make_jet(y, 0.0, 1.0));
    let exact_dx = exact_delta.v[0];
    let exact_dy = exact_delta.v[1];

    // Sanity check - these should be equivalent:
    assert_eq!(
        exact_dx,
        f(make_jet(x, 1.0, 0.0), make_jet(y, 0.0, 0.0)).v[0]
    );
    assert_eq!(
        exact_dx,
        f(make_jet(x, 0.0, 1.0), make_jet(y, 0.0, 0.0)).v[1]
    );
    assert_eq!(
        exact_dy,
        f(make_jet(x, 0.0, 0.0), make_jet(y, 1.0, 0.0)).v[0]
    );
    assert_eq!(
        exact_dy,
        f(make_jet(x, 0.0, 0.0), make_jet(y, 0.0, 1.0)).v[1]
    );

    let estimated_dx = (f(J::constant(x + K_STEP), J::constant(y)).a
        - f(J::constant(x - K_STEP), J::constant(y)).a)
        / (2.0 * K_STEP);
    let estimated_dy = (f(J::constant(x), J::constant(y + K_STEP)).a
        - f(J::constant(x), J::constant(y - K_STEP)).a)
        / (2.0 * K_STEP);
    println!("{name}({x}, {y}), exact dx: {exact_dx}, estimated dx: {estimated_dx}");
    expect_close(exact_dx, estimated_dx, K_NUMERICAL_TOLERANCE);
    println!("{name}({x}, {y}), exact dy: {exact_dy}, estimated dy: {estimated_dy}");
    expect_close(exact_dy, estimated_dy, K_NUMERICAL_TOLERANCE);
}

/// Pick arbitrary values for x and y (shared by the `TEST(Jet, Jet)` parts).
fn x_and_y() -> (J, J) {
    (make_jet(2.3, -2.7, 1e-3), make_jet(1.7, 0.5, 1e+2))
}

/// `TEST(Jet, Jet)`, split into groups.
mod jet {
    use super::*;
    use ceres::jet::{
        abs, acos, asin, atan, atan2, bessel_j0, bessel_j1, bessel_jn, cbrt, ceil, cos, cosh, erf,
        erfc, exp, exp2, floor, fmax, fmin, hypot, log, log2, pow, pow_jet_scalar, pow_scalar_jet,
        sin, sinh, sqrt, tan, tanh,
    };

    #[test]
    fn log_exp_mul_div_sqrt_identities() {
        let (x, y) = x_and_y();
        println!("x = {x}");
        println!("y = {y}");

        {
            // Check that log(exp(x)) == x.
            let z = exp(x);
            let w = log(z);
            expect_jets_close(w, x);
        }
        {
            // Check that (x * y) / x == y.
            let z = x * y;
            let w = z / x;
            expect_jets_close(w, y);
        }
        {
            // Check that sqrt(x * x) == x.
            let z = x * x;
            let w = sqrt(z);
            expect_jets_close(w, x);
        }
        {
            // Check that sqrt(y) * sqrt(y) == y.
            let z = sqrt(y);
            let w = z * z;
            expect_jets_close(w, y);
        }

        numerical_test("sqrt", sqrt, 0.00001);
        numerical_test("sqrt", sqrt, 1.0);
    }

    #[test]
    fn trigonometric_and_hyperbolic_identities() {
        let (x, y) = x_and_y();
        {
            // Check that cos(2*x) = cos(x)^2 - sin(x)^2
            let z = cos(J::constant(2.0) * x);
            let w = cos(x) * cos(x) - sin(x) * sin(x);
            expect_jets_close(w, z);
        }
        {
            // Check that sin(2*x) = 2*cos(x)*sin(x)
            let z = sin(J::constant(2.0) * x);
            let w = J::constant(2.0) * cos(x) * sin(x);
            expect_jets_close(w, z);
        }
        {
            // Check that cos(x)*cos(x) + sin(x)*sin(x) = 1
            let z = cos(x) * cos(x);
            let w = sin(x) * sin(x);
            expect_jets_close(z + w, J::constant(1.0));
        }
        {
            // Check that atan2(r*sin(t), r*cos(t)) = t.
            let t = make_jet(0.7, -0.3, 1.5);
            let r = make_jet(2.3, 0.13, -2.4);
            let u = atan2(r * sin(t), r * cos(t));
            println!("u = {u}");
            expect_jets_close(u, t);
        }
        {
            // Check that tan(x) = sin(x) / cos(x).
            let z = tan(x);
            let w = sin(x) / cos(x);
            expect_jets_close(z, w);
        }
        {
            // Check that tan(atan(x)) = x.
            let z = tan(atan(x));
            let w = x;
            expect_jets_close(z, w);
        }
        {
            // Check that cosh(x)*cosh(x) - sinh(x)*sinh(x) = 1
            let z = cosh(x) * cosh(x);
            let w = sinh(x) * sinh(x);
            expect_jets_close(z - w, J::constant(1.0));
        }
        {
            // Check that tanh(x + y) = (tanh(x) + tanh(y)) / (1 + tanh(x) tanh(y))
            let z = tanh(x + y);
            let w = (tanh(x) + tanh(y)) / (J::constant(1.0) + tanh(x) * tanh(y));
            expect_jets_close(z, w);
        }
    }

    #[test]
    fn pow_identities() {
        let (x, y) = x_and_y();
        {
            // Check that pow(x, 1) == x.
            let u = pow_jet_scalar(x, 1.0);
            expect_jets_close(x, u);
        }
        {
            // Check that pow(x, 1) == x.
            let y = make_jet(1.0, 0.0, 0.0);
            let u = pow(x, y);
            expect_jets_close(x, u);
        }
        {
            // Check that pow(e, log(x)) == x.
            let logx = log(x);
            let u = pow_scalar_jet(K_E, logx);
            expect_jets_close(x, u);
        }
        {
            // Check that pow(e, log(x)) == x.
            let logx = log(x);
            let e = make_jet(K_E, 0.0, 0.0);
            let u = pow(e, logx);
            expect_jets_close(x, u);
        }
        {
            // Check that pow(e, log(x)) == x. (Duplicated in the C++ test.)
            let logx = log(x);
            let e = make_jet(K_E, 0.0, 0.0);
            let u = pow(e, logx);
            expect_jets_close(x, u);
        }
        {
            // Check that pow(x,y) = exp(y*log(x)).
            let logx = log(x);
            let e = make_jet(K_E, 0.0, 0.0);
            let u = pow(e, y * logx);
            let v = pow(x, y);
            expect_jets_close(v, u);
        }
    }

    #[test]
    fn pow_jet_jet_special_cases() {
        {
            // Check that pow(0, y) == 0 for y > 1, with both arguments Jets.
            // This tests special case handling inside pow().
            let a = make_jet(0.0, 1.0, 2.0);
            let b = make_jet(2.0, 3.0, 4.0);
            let c = pow(a, b);
            println!("a^b = {c}");
            expect_jets_close(c, make_jet(0.0, 0.0, 0.0));
        }
        {
            // Check that pow(0, y) == 0 for y == 1, with both arguments Jets.
            // This tests special case handling inside pow().
            let a = make_jet(0.0, 1.0, 2.0);
            let b = make_jet(1.0, 3.0, 4.0);
            let c = pow(a, b);
            expect_jets_close(c, make_jet(0.0, 1.0, 2.0));
        }
        {
            // Check that pow(0, <1) is not finite, with both arguments Jets.
            for i in 1..10 {
                let a = make_jet(0.0, 1.0, 2.0);
                let b = make_jet(f64::from(i) * 0.1, 3.0, 4.0); // b = 0.1 ... 0.9
                let c = pow(a, b);
                println!("a = {a} b = {b} a^b = {c}");
                assert_eq!(c.a, 0.0);
                assert!(!c.v[0].is_finite());
                assert!(!c.v[1].is_finite());
            }
            for i in -10..0 {
                let a = make_jet(0.0, 1.0, 2.0);
                let b = make_jet(f64::from(i) * 0.1, 3.0, 4.0); // b = -1,-0.9 ... -0.1
                let c = pow(a, b);
                println!("a = {a} b = {b} a^b = {c}");
                assert!(!c.a.is_finite());
                assert!(!c.v[0].is_finite());
                assert!(!c.v[1].is_finite());
            }
            {
                // The special case of 0^0 = 1 defined by the C standard.
                let a = make_jet(0.0, 1.0, 2.0);
                let b = make_jet(0.0, 3.0, 4.0);
                let c = pow(a, b);
                assert_eq!(c.a, 1.0);
                assert!(!c.v[0].is_finite());
                assert!(!c.v[1].is_finite());
            }
        }
        {
            // Check that pow(<0, b) is correct for integer b.
            // This tests special case handling inside pow().
            let a = make_jet(-1.5, 3.0, 4.0);
            // b integer:
            for i in -10..=10 {
                let b = make_jet(f64::from(i), 0.0, 5.0);
                let c = pow(a, b);
                println!("a = {a} b = {b} a^b = {c}");
                expect_close(c.a, (-1.5f64).powi(i), K_TOLERANCE);
                assert!(c.v[0].is_finite());
                assert!(!c.v[1].is_finite());
                expect_close(
                    c.v[0],
                    f64::from(i) * (-1.5f64).powi(i - 1) * 3.0,
                    K_TOLERANCE,
                );
            }
        }
        {
            // Check that pow(<0, b) is correct for noninteger b.
            // This tests special case handling inside pow().
            let a = make_jet(-1.5, 3.0, 4.0);
            let b = make_jet(-2.5, 0.0, 5.0);
            let c = pow(a, b);
            assert!(!c.a.is_finite());
            assert!(!c.v[0].is_finite());
            assert!(!c.v[1].is_finite());
        }
    }

    #[test]
    fn pow_scalar_jet_special_cases() {
        {
            // Check that pow(0,y) == 0 for y == 2, with the second argument a
            // Jet.  This tests special case handling inside pow().
            let a = 0.0;
            let b = make_jet(2.0, 3.0, 4.0);
            let c = pow_scalar_jet(a, b);
            expect_jets_close(c, make_jet(0.0, 0.0, 0.0));
        }
        {
            // Check that pow(<0,y) is correct for integer y. This tests special
            // case handling inside pow().
            let a = -1.5;
            for i in -10..=10 {
                let b = make_jet(f64::from(i), 3.0, 0.0);
                let c = pow_scalar_jet(a, b);
                println!("a = {a} b = {b} a^b = {c}");
                expect_close(c.a, (-1.5f64).powi(i), K_TOLERANCE);
                assert!(!c.v[0].is_finite());
                assert!(c.v[1].is_finite());
                expect_close(c.v[1], 0.0, K_TOLERANCE);
            }
        }
        {
            // Check that pow(<0,y) is correct for noninteger y. This tests
            // special case handling inside pow().
            let a = -1.5;
            let b = make_jet(-3.14, 3.0, 0.0);
            let c = pow_scalar_jet(a, b);
            assert!(!c.a.is_finite());
            assert!(!c.v[0].is_finite());
            assert!(!c.v[1].is_finite());
        }
    }

    #[test]
    fn scalar_arithmetic_and_abs() {
        let (x, y) = x_and_y();
        {
            // Check that 1 + x == x + 1.
            let a = x + 1.0;
            let b = 1.0 + x;
            let mut c = x;
            c += 1.0;
            expect_jets_close(a, b);
            expect_jets_close(a, c);
        }
        {
            // Check that 1 - x == -(x - 1).
            let a = 1.0 - x;
            let b = -(x - 1.0);
            let mut c = x;
            c -= 1.0;
            expect_jets_close(a, b);
            expect_jets_close(a, -c);
        }
        {
            // Check that (x/s)*s == (x*s)/s.
            let a = x / 5.0;
            let b = x * 5.0;
            let mut c = x;
            c /= 5.0;
            let mut d = x;
            d *= 5.0;
            expect_jets_close(5.0 * a, b / 5.0);
            expect_jets_close(a, c);
            expect_jets_close(b, d);
        }
        {
            // Check that x / y == 1 / (y / x).
            let a = x / y;
            let b = 1.0 / (y / x);
            expect_jets_close(a, b);
        }
        {
            // Check that abs(-x * x) == sqrt(x * x).
            expect_jets_close(abs(-x), sqrt(x * x));
        }
    }

    #[test]
    fn inverse_trigonometric() {
        {
            // Check that cos(acos(x)) == x.
            let a = make_jet(0.1, -2.7, 1e-3);
            expect_jets_close(cos(acos(a)), a);
            expect_jets_close(acos(cos(a)), a);

            let b = make_jet(0.6, 0.5, 1e+2);
            expect_jets_close(cos(acos(b)), b);
            expect_jets_close(acos(cos(b)), b);
        }
        {
            // Check that sin(asin(x)) == x.
            let a = make_jet(0.1, -2.7, 1e-3);
            expect_jets_close(sin(asin(a)), a);
            expect_jets_close(asin(sin(a)), a);

            let b = make_jet(0.4, 0.5, 1e+2);
            expect_jets_close(sin(asin(b)), b);
            expect_jets_close(asin(sin(b)), b);
        }
    }

    #[test]
    fn bessel() {
        let zero = J::constant(0.0);

        // Check that J0(0) == 1.
        expect_jets_close(bessel_j0(zero), J::constant(1.0));
        // Check that J1(0) == 0.
        expect_jets_close(bessel_j1(zero), zero);
        // Check that J2(0) == 0.
        expect_jets_close(bessel_jn(2, zero), zero);
        // Check that J3(0) == 0.
        expect_jets_close(bessel_jn(3, zero), zero);

        let z = make_jet(0.1, -2.7, 1e-3);

        // Check that J0(z) == Jn(0,z).
        expect_jets_close(bessel_j0(z), bessel_jn(0, z));
        // Check that J1(z) == Jn(1,z).
        expect_jets_close(bessel_j1(z), bessel_jn(1, z));
        // Check that J0(z)+J2(z) == (2/z)*J1(z).
        // See formula http://dlmf.nist.gov/10.6.E1
        expect_jets_close(bessel_j0(z) + bessel_jn(2, z), (2.0 / z) * bessel_j1(z));
    }

    // Note: like C++ `EXPECT_EQ` on jets, `assert_eq!` only compares the
    // scalar parts (Jet's PartialEq).
    #[test]
    fn floor_and_ceil() {
        for a in [
            // positive, negative, positive.
            make_jet(0.1, -2.7, 1e-3),
            make_jet(-1.1, -2.7, 1e-3),
            make_jet(10.123, -2.7, 1e-3),
        ] {
            // Check that floor works.
            let b = floor(a);
            let expected = make_jet(a.a.floor(), 0.0, 0.0);
            assert_eq!(expected, b);

            // Check that ceil works.
            let b = ceil(a);
            let expected = make_jet(a.a.ceil(), 0.0, 0.0);
            assert_eq!(expected, b);
        }
    }

    #[test]
    fn erf_and_erfc() {
        {
            // Check that erf works.
            let a = make_jet(10.123, -2.7, 1e-3);
            let b = erf(a);
            let expected = make_jet(libm::erf(a.a), 0.0, 0.0);
            assert_eq!(expected, b);
        }
        numerical_test("erf", erf, -1.0);
        numerical_test("erf", erf, 1e-5);
        numerical_test("erf", erf, 0.5);
        numerical_test("erf", erf, 100.0);

        {
            // Check that erfc works.
            let a = make_jet(10.123, -2.7, 1e-3);
            let b = erfc(a);
            let expected = make_jet(libm::erfc(a.a), 0.0, 0.0);
            assert_eq!(expected, b);
        }
        numerical_test("erfc", erfc, -1.0);
        numerical_test("erfc", erfc, 1e-5);
        numerical_test("erfc", erfc, 0.5);
        numerical_test("erfc", erfc, 100.0);
    }

    #[test]
    fn cbrt_tests() {
        let (x, y) = x_and_y();
        {
            // Check that cbrt(x * x * x) == x.
            let z = x * x * x;
            let w = cbrt(z);
            expect_jets_close(w, x);
        }
        {
            // Check that cbrt(y) * cbrt(y) * cbrt(y) == y.
            let z = cbrt(y);
            let w = z * z * z;
            expect_jets_close(w, y);
        }
        {
            // Check that cbrt(x) == pow(x, 1/3).
            let z = cbrt(x);
            let w = pow_jet_scalar(x, 1.0 / 3.0);
            expect_jets_close(z, w);
        }
        numerical_test("cbrt", cbrt, -1.0);
        numerical_test("cbrt", cbrt, -1e-5);
        numerical_test("cbrt", cbrt, 1e-5);
        numerical_test("cbrt", cbrt, 1.0);
    }

    #[test]
    fn exp2_tests() {
        let (x, _) = x_and_y();
        {
            // Check that exp2(x) == exp(x * log(2))
            let z = exp2(x);
            let w = exp(x * 2.0f64.ln());
            expect_jets_close(z, w);
        }
        numerical_test("exp2", exp2, -1.0);
        numerical_test("exp2", exp2, -1e-5);
        numerical_test("exp2", exp2, -1e-200);
        numerical_test("exp2", exp2, 0.0);
        numerical_test("exp2", exp2, 1e-200);
        numerical_test("exp2", exp2, 1e-5);
        numerical_test("exp2", exp2, 1.0);
    }

    #[test]
    fn log2_tests() {
        let (x, _) = x_and_y();
        {
            // Check that log2(x) == log(x) / log(2)
            let z = log2(x);
            let w = log(x) / 2.0f64.ln();
            expect_jets_close(z, w);
        }
        numerical_test("log2", log2, 1e-5);
        numerical_test("log2", log2, 1.0);
        numerical_test("log2", log2, 100.0);
    }

    #[test]
    fn hypot_tests() {
        let (x, y) = x_and_y();
        {
            // Check that hypot(x, y) == sqrt(x^2 + y^2)
            let h = hypot(x, y);
            let s = sqrt(x * x + y * y);
            expect_jets_close(h, s);
        }
        {
            // Check that hypot(x, x) == sqrt(2) * abs(x)
            let h = hypot(x, x);
            let s = 2.0f64.sqrt() * abs(x);
            expect_jets_close(h, s);
        }
        {
            // Check that the derivative is zero tangentially to the circle:
            let h = hypot(make_jet(2.0, 1.0, 1.0), make_jet(2.0, 1.0, -1.0));
            expect_jets_close(h, make_jet(8.0f64.sqrt(), 2.0f64.sqrt(), 0.0));
        }
        {
            // Check that hypot(x, 0) == x
            let zero = make_jet(0.0, 2.0, 3.14);
            let h = hypot(x, zero);
            expect_jets_close(x, h);
        }
        {
            // Check that hypot(0, y) == y
            let zero = make_jet(0.0, 2.0, 3.14);
            let h = hypot(zero, y);
            expect_jets_close(y, h);
        }
        {
            // Check that hypot(x, 0) == sqrt(x * x) == x, even when x * x
            // underflows:
            assert_eq!(f64::MIN_POSITIVE * f64::MIN_POSITIVE, 0.0); // Make sure it underflows
            let huge = make_jet(f64::MIN_POSITIVE, 2.0, 3.14);
            let h = hypot(huge, J::constant(0.0));
            expect_jets_close(h, huge);
        }
        {
            // Check that hypot(x, 0) == sqrt(x * x) == x, even when x * x
            // overflows:
            assert_eq!(f64::MAX * f64::MAX, f64::INFINITY);
            let huge = make_jet(f64::MAX, 2.0, 3.14);
            let h = hypot(huge, J::constant(0.0));
            expect_jets_close(h, huge);
        }

        numerical_test2("hypot", hypot, 0.0, 1e-5);
        numerical_test2("hypot", hypot, -1e-5, 0.0);
        numerical_test2("hypot", hypot, 1e-5, 1e-5);
        numerical_test2("hypot", hypot, 0.0, 1.0);
        numerical_test2("hypot", hypot, 1e-3, 1.0);
        numerical_test2("hypot", hypot, 1e-3, -1.0);
        numerical_test2("hypot", hypot, -1e-3, 1.0);
        numerical_test2("hypot", hypot, -1e-3, -1.0);
        numerical_test2("hypot", hypot, 1.0, 2.0);
    }

    #[test]
    fn fmax_and_fmin() {
        let (x, y) = x_and_y();
        {
            let z = fmax(x, y);
            expect_jets_close(x, z);
        }
        {
            let z = fmin(x, y);
            expect_jets_close(y, z);
        }
    }

    /// `TEST(Jet, JetsInEigenMatrices)`, with nalgebra instead of Eigen.
    #[test]
    fn jets_in_eigen_matrices() {
        use nalgebra::{Matrix2, Vector2};
        let x = make_jet(2.3, -2.7, 1e-3);
        let y = make_jet(1.7, 0.5, 1e+2);
        let z = make_jet(5.3, -4.7, 1e-3);
        let w = make_jet(9.7, 1.5, 10.1);

        // Row-major like Eigen's `M << x, y, z, w;`.
        let m = Matrix2::new(x, y, z, w);
        let v = Vector2::new(x, z);

        // Check that M * v == (v^T * M^T)^T
        let r1 = m * v;
        let r2 = (v.transpose() * m.transpose()).transpose();

        expect_jets_close(r1[0], r2[0]);
        expect_jets_close(r1[1], r2[1]);
    }

    fn decomposition_problem() -> (nalgebra::Matrix3<J>, nalgebra::Vector3<J>) {
        let mut a = nalgebra::Matrix3::<J>::zeros();
        let mut b = nalgebra::Vector3::<J>::zeros();
        for i in 0..3 {
            let fi = i as f64;
            for j in 0..3 {
                let fj = j as f64;
                a[(i, j)] = make_jet(0.0, fi, fj * fj);
            }
            b[i] = make_jet(fi, fi, fi);
            a[(i, i)] = make_jet(1.0, fi, fi * fi);
        }
        (a, b)
    }

    /// `TEST(Jet, FullRankEigenLLTSolve)`: nalgebra `Cholesky` (reads the
    /// lower triangle, like Eigen's LLT).
    #[test]
    fn full_rank_eigen_llt_solve() {
        let (a, b) = decomposition_problem();
        let x = a.cholesky().expect("cholesky failed").solve(&b);
        for i in 0..3 {
            assert_eq!(x[i].a, b[i].a);
        }
    }

    /// `TEST(Jet, FullRankEigenLDLTSolve)`: nalgebra has no LDLT solver, so
    /// this uses another nalgebra decomposition that needs extra Jet traits
    /// (Householder QR: norms, signum, division).
    #[test]
    fn full_rank_eigen_ldlt_solve() {
        let (a, b) = decomposition_problem();
        let x = a.qr().solve(&b).expect("qr solve failed");
        for i in 0..3 {
            assert_eq!(x[i].a, b[i].a);
        }
    }

    /// `TEST(Jet, FullRankEigenLUSolve)`: nalgebra partial-pivoting `LU`.
    #[test]
    fn full_rank_eigen_lu_solve() {
        let (a, b) = decomposition_problem();
        let x = a.lu().solve(&b).expect("lu solve failed");
        for i in 0..3 {
            assert_eq!(x[i].a, b[i].a);
        }
    }

    // TEST(Jet, nested3x) is skipped: the Rust `Jet<N>` has `f64` parts only,
    // so nested jets (`Jet<Jet<Jet<double, 2>, 2>, 2>`) cannot be expressed.
}

/// `TEST(JetTraitsTest, ...)`.
mod jet_traits_test {
    use super::*;
    use ceres::jet::{is_finite, is_infinite, is_nan, is_normal};

    #[test]
    fn classification_mixed() {
        let a = Jet::<3>::new(5.5, [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        assert!(!is_finite(&a));
        assert!(!is_normal(&a));
        assert!(is_infinite(&a));
        assert!(is_nan(&a));
    }

    #[test]
    fn classification_nan() {
        let a = Jet::<3>::new(5.5, [f64::NAN, 0.0, 0.0]);
        assert!(!is_finite(&a));
        assert!(!is_normal(&a));
        assert!(!is_infinite(&a));
        assert!(is_nan(&a));
    }

    #[test]
    fn classification_inf() {
        let a = Jet::<3>::new(5.5, [f64::INFINITY, 0.0, 0.0]);
        assert!(!is_finite(&a));
        assert!(!is_normal(&a));
        assert!(is_infinite(&a));
        assert!(!is_nan(&a));
    }

    #[test]
    fn classification_finite() {
        let a = Jet::<3>::new(5.5, [100.0, 1.0, 3.14159]);
        assert!(is_finite(&a));
        assert!(is_normal(&a));
        assert!(!is_infinite(&a));
        assert!(!is_nan(&a));
    }

    #[test]
    fn matrix_scalar_unary_ops() {
        let x = make_jet(2.3, -2.7, 1e-3);
        let y = make_jet(1.7, 0.5, 1e+2);
        let a = nalgebra::Vector2::new(x, y);

        let sum = a.sum();
        let sum2 = a[0] + a[1];
        expect_jets_close(sum, sum2);
    }

    #[test]
    fn matrix_scalar_binary_ops() {
        use nalgebra::{Matrix2, Vector2};
        let x = make_jet(2.3, -2.7, 1e-3);
        let y = make_jet(1.7, 0.5, 1e+2);
        let z = make_jet(5.3, -4.7, 1e-3);
        let w = make_jet(9.7, 1.5, 10.1);

        let m = Matrix2::new(x, y, z, w);
        let v = Vector2::new(0.6, -2.1);

        // Check that M * v == M * v.cast<J>(). nalgebra has no mixed-scalar
        // matrix product, so `M * v` (Eigen's ScalarBinaryOpTraits) is
        // written out with the Jet * f64 operator.
        let r1 = Vector2::new(
            m[(0, 0)] * v[0] + m[(0, 1)] * v[1],
            m[(1, 0)] * v[0] + m[(1, 1)] * v[1],
        );
        let r2 = m * v.cast::<J>();
        expect_jets_close(r1[0], r2[0]);
        expect_jets_close(r1[1], r2[1]);

        // Check that M * a == M * T(a). (nalgebra: `Matrix<Jet> * f64` is not
        // available, so it is applied per element.)
        let a = 3.1;
        let r3 = m.map(|e| e * a);
        let r4 = m * J::constant(a);
        expect_jets_close(r3[(0, 0)], r4[(0, 0)]);
        expect_jets_close(r3[(1, 0)], r4[(1, 0)]);
        expect_jets_close(r3[(0, 1)], r4[(0, 1)]);
        expect_jets_close(r3[(1, 1)], r4[(1, 1)]);
    }

    /// Eigen `Array` -> nalgebra vector with component-wise ops.
    #[test]
    fn array_scalar_unary_ops() {
        let x = make_jet(2.3, -2.7, 1e-3);
        let y = make_jet(1.7, 0.5, 1e+2);
        let a = nalgebra::Vector2::new(x, y);

        let sum = a.iter().copied().sum::<J>();
        let sum2 = a[0] + a[1];
        expect_jets_close(sum, sum2);
        expect_jets_close(a.sum(), sum2);
    }

    #[test]
    fn array_scalar_binary_ops() {
        use nalgebra::Vector2;
        let x = make_jet(2.3, -2.7, 1e-3);
        let y = make_jet(1.7, 0.5, 1e+2);

        let a = Vector2::new(x, y);
        let b = Vector2::new(0.6, -2.1);

        // Check that a * b == a * b.cast<T>()
        let r1 = Vector2::new(a[0] * b[0], a[1] * b[1]);
        let r2 = a.component_mul(&b.cast::<J>());
        expect_jets_close(r1[0], r2[0]);
        expect_jets_close(r1[1], r2[1]);

        // Check that a * c == a * T(c).
        let c = 3.1;
        let r3 = a.map(|e| e * c);
        let r4 = a * J::constant(c);
        // The C++ test compares r3(0) with itself and r4(1) with itself
        // (a typo); here r3 is compared with r4 as intended.
        expect_jets_close(r3[0], r4[0]);
        expect_jets_close(r3[1], r4[1]);
    }
}

// ---------------------------------------------------------------------------
// Not in jet_test.cc: property-based checks of every jet derivative against
// central finite differences.
// ---------------------------------------------------------------------------
mod finite_differences {
    use ceres::Jet;
    use nalgebra::ComplexField;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    /// Relative step (scaled by `max(1, |x|)`).
    const FD_STEP: f64 = 1e-6;
    /// Tolerance relative to `max(1, |exact|)`.
    const FD_TOL: f64 = 1e-5;

    type J1 = Jet<1>;
    type J2 = Jet<2>;

    fn step(x: f64) -> f64 {
        FD_STEP * x.abs().max(1.0)
    }

    fn check(name: &str, at: &str, exact: f64, est: f64) -> Result<(), TestCaseError> {
        prop_assert!(
            (exact - est).abs() <= FD_TOL * exact.abs().max(1.0),
            "{}({}): jet derivative {:e} vs finite difference {:e}",
            name,
            at,
            exact,
            est
        );
        Ok(())
    }

    /// Checks `d f / dx` of a unary jet function against central differences
    /// and, if given, the value against the reference `f64` function.
    fn unary(
        name: &str,
        f: impl Fn(J1) -> J1,
        reference: Option<fn(f64) -> f64>,
        x: f64,
    ) -> Result<(), TestCaseError> {
        let j = f(J1::variable(x, 0));
        if let Some(r) = reference {
            prop_assert_eq!(j.a.to_bits(), r(x).to_bits(), "{}({}) value", name, x);
        }
        let h = step(x);
        let est = (f(J1::constant(x + h)).a - f(J1::constant(x - h)).a) / (2.0 * h);
        check(name, &format!("{x}"), j.v[0], est)
    }

    /// Checks both partials of a binary jet function.
    fn binary(name: &str, f: impl Fn(J2, J2) -> J2, x: f64, y: f64) -> Result<(), TestCaseError> {
        let j = f(J2::variable(x, 0), J2::variable(y, 1));
        let (hx, hy) = (step(x), step(y));
        let c = J2::constant;
        let dx = (f(c(x + hx), c(y)).a - f(c(x - hx), c(y)).a) / (2.0 * hx);
        let dy = (f(c(x), c(y + hy)).a - f(c(x), c(y - hy)).a) / (2.0 * hy);
        let at = format!("{x}, {y}");
        check(&format!("{name} d/dx"), &at, j.v[0], dx)?;
        check(&format!("{name} d/dy"), &at, j.v[1], dy)
    }

    /// Non-zero values with `|x|` in `[lo, hi]` and random sign.
    fn signed_away_from_zero(lo: f64, hi: f64) -> impl Strategy<Value = f64> {
        (lo..hi, any::<bool>()).prop_map(|(m, neg)| if neg { -m } else { m })
    }

    proptest! {
        #[test]
        fn sqrt(x in 0.01f64..100.0) {
            unary("sqrt", J1::sqrt, Some(f64::sqrt), x)?;
            unary("jet::sqrt", ceres::jet::sqrt, Some(f64::sqrt), x)?;
        }

        #[test]
        fn sin_cos(x in -10.0f64..10.0) {
            unary("sin", J1::sin, Some(f64::sin), x)?;
            unary("cos", J1::cos, Some(f64::cos), x)?;
        }

        #[test]
        fn tan(x in -1.4f64..1.4) {
            unary("tan", J1::tan, Some(f64::tan), x)?;
        }

        #[test]
        fn asin_acos(x in -0.95f64..0.95) {
            unary("asin", J1::asin, Some(f64::asin), x)?;
            unary("acos", J1::acos, Some(f64::acos), x)?;
        }

        #[test]
        fn atan(x in -100.0f64..100.0) {
            unary("atan", J1::atan, Some(f64::atan), x)?;
        }

        #[test]
        fn atan2(y in -10.0f64..10.0, x in -10.0f64..10.0) {
            prop_assume!(x.hypot(y) > 0.1);
            // Avoid the branch cut on the negative x axis.
            prop_assume!(!(x < 0.0 && y.abs() < 1e-3));
            binary("atan2 (method)", J2::atan2, y, x)?;
            binary("jet::atan2", ceres::jet::atan2, y, x)?;
            prop_assert_eq!(J2::constant(y).atan2(J2::constant(x)).a, y.atan2(x));
        }

        #[test]
        fn hyperbolic(x in -10.0f64..10.0) {
            unary("sinh", J1::sinh, Some(f64::sinh), x)?;
            unary("cosh", J1::cosh, Some(f64::cosh), x)?;
            unary("tanh", J1::tanh, Some(f64::tanh), x)?;
        }

        #[test]
        fn exp(x in -20.0f64..20.0) {
            unary("exp", J1::exp, Some(f64::exp), x)?;
        }

        #[test]
        fn exp2(x in -30.0f64..30.0) {
            unary("exp2", J1::exp2, Some(f64::exp2), x)?;
        }

        #[test]
        fn logarithms(x in 1e-3f64..1e3) {
            unary("ln", J1::ln, Some(f64::ln), x)?;
            unary("log", J1::log, Some(f64::ln), x)?;
            unary("jet::log", ceres::jet::log, Some(f64::ln), x)?;
            unary("log2", J1::log2, Some(f64::log2), x)?;
            unary("log10", J1::log10, Some(f64::log10), x)?;
        }

        #[test]
        fn cbrt(x in signed_away_from_zero(0.01, 100.0)) {
            unary("cbrt", J1::cbrt, Some(f64::cbrt), x)?;
        }

        #[test]
        fn abs(x in signed_away_from_zero(0.01, 100.0)) {
            unary("abs", J1::abs, Some(f64::abs), x)?;
        }

        #[test]
        fn hypot(x in -10.0f64..10.0, y in -10.0f64..10.0) {
            prop_assume!(x.hypot(y) > 0.1);
            binary("hypot", J2::hypot, x, y)?;
            prop_assert_eq!(J2::constant(x).hypot(J2::constant(y)).a, x.hypot(y));
        }

        #[test]
        fn powf_scalar(x in 0.1f64..10.0, g in -3.0f64..3.0) {
            unary("powf_scalar", |f| f.powf_scalar(g), None, x)?;
            unary("jet::pow_jet_scalar", |f| ceres::jet::pow_jet_scalar(f, g), None, x)?;
            prop_assert_eq!(J1::constant(x).powf_scalar(g).a, x.powf(g));
        }

        #[test]
        fn scalar_powf(f in 0.1f64..10.0, g in -3.0f64..3.0) {
            unary("scalar_powf", |gj| J1::scalar_powf(f, gj), None, g)?;
            unary("jet::pow_scalar_jet", |gj| ceres::jet::pow_scalar_jet(f, gj), None, g)?;
            prop_assert_eq!(J1::scalar_powf(f, J1::constant(g)).a, f.powf(g));
        }

        #[test]
        fn powj(f in 0.1f64..10.0, g in -3.0f64..3.0) {
            binary("powj", J2::powj, f, g)?;
            binary("jet::pow", ceres::jet::pow, f, g)?;
            prop_assert_eq!(J2::constant(f).powj(J2::constant(g)).a, f.powf(g));
        }

        #[test]
        fn erf_erfc(x in -5.0f64..5.0) {
            unary("erf", J1::erf, Some(libm::erf), x)?;
            unary("erfc", J1::erfc, Some(libm::erfc), x)?;
        }

        #[test]
        fn bessel_j0_j1(x in -20.0f64..20.0) {
            unary("bessel_j0", J1::bessel_j0, Some(libm::j0), x)?;
            unary("bessel_j1", J1::bessel_j1, Some(libm::j1), x)?;
        }

        #[test]
        fn bessel_jn(n in 0i32..6, x in 0.1f64..20.0) {
            unary(&format!("bessel_jn({n})"), |j| j.bessel_jn(n), None, x)?;
            unary(&format!("jet::bessel_jn({n})"), |j| ceres::jet::bessel_jn(n, j), None, x)?;
            prop_assert_eq!(J1::constant(x).bessel_jn(n).a, libm::jn(n, x));
        }

        #[test]
        fn fmax_fmin(x in -10.0f64..10.0, y in -10.0f64..10.0) {
            prop_assume!((x - y).abs() > 1e-3);
            binary("fmax", J2::fmax, x, y)?;
            binary("fmin", J2::fmin, x, y)?;
            binary("jet::fmax", ceres::jet::fmax, x, y)?;
            binary("jet::fmin", ceres::jet::fmin, x, y)?;
            prop_assert_eq!(J2::constant(x).fmax(J2::constant(y)).a, x.max(y));
            prop_assert_eq!(J2::constant(x).fmin(J2::constant(y)).a, x.min(y));
        }

        #[test]
        fn jet_jet_arithmetic(x in -10.0f64..10.0, y in signed_away_from_zero(0.1, 10.0)) {
            binary("+", |a, b| a + b, x, y)?;
            binary("-", |a, b| a - b, x, y)?;
            binary("*", |a, b| a * b, x, y)?;
            binary("/", |a, b| a / b, x, y)?;
            binary("neg", |a, b| -(a * b), x, y)?;
            binary("+=", |mut a, b| { a += b; a }, x, y)?;
            binary("-=", |mut a, b| { a -= b; a }, x, y)?;
            binary("*=", |mut a, b| { a *= b; a }, x, y)?;
            binary("/=", |mut a, b| { a /= b; a }, x, y)?;
        }

        #[test]
        fn jet_scalar_arithmetic(
            x in signed_away_from_zero(0.1, 10.0),
            c in -10.0f64..10.0,
        ) {
            unary("jet + f64", |a| a + c, None, x)?;
            unary("f64 + jet", |a| c + a, None, x)?;
            unary("jet - f64", |a| a - c, None, x)?;
            unary("f64 - jet", |a| c - a, None, x)?;
            unary("jet * f64", |a| a * c, None, x)?;
            unary("f64 * jet", |a| c * a, None, x)?;
            unary("f64 / jet", |a| c / a, None, x)?;
            if c.abs() > 0.1 {
                unary("jet / f64", |a| a / c, None, x)?;
            }
            prop_assert_eq!((J1::constant(x) + c).a, x + c);
            prop_assert_eq!((c - J1::constant(x)).a, c - x);
            prop_assert_eq!((J1::constant(x) * c).a, x * c);
            prop_assert_eq!((c / J1::constant(x)).a, c / x);
        }

        #[test]
        fn complex_field_asinh(x in -10.0f64..10.0) {
            unary("asinh", <J1 as ComplexField>::asinh, Some(f64::asinh), x)?;
        }

        #[test]
        fn complex_field_acosh(x in 1.01f64..100.0) {
            unary("acosh", <J1 as ComplexField>::acosh, Some(f64::acosh), x)?;
        }

        #[test]
        fn complex_field_atanh(x in -0.95f64..0.95) {
            unary("atanh", <J1 as ComplexField>::atanh, Some(f64::atanh), x)?;
        }

        #[test]
        fn complex_field_ln_1p(x in -0.9f64..100.0) {
            unary("ln_1p", <J1 as ComplexField>::ln_1p, Some(f64::ln_1p), x)?;
        }

        #[test]
        fn complex_field_exp_m1(x in -20.0f64..20.0) {
            unary("exp_m1", <J1 as ComplexField>::exp_m1, Some(f64::exp_m1), x)?;
        }

        #[test]
        fn complex_field_recip(x in signed_away_from_zero(0.1, 100.0)) {
            unary("recip", <J1 as ComplexField>::recip, Some(f64::recip), x)?;
        }

        #[test]
        fn complex_field_powi(x in signed_away_from_zero(0.1, 10.0), n in -4i32..6) {
            unary(&format!("powi({n})"), |j| <J1 as ComplexField>::powi(j, n), None, x)?;
            unary(&format!("inherent powi({n})"), |j| j.powi(n), None, x)?;
            let value = <J1 as ComplexField>::powi(J1::constant(x), n).a;
            let expected = x.powi(n);
            prop_assert!((value - expected).abs() <= 1e-14 * expected.abs());
        }

        #[test]
        fn complex_field_math_delegates(x in 0.1f64..0.9) {
            // The ComplexField / RealField versions of the jet.h functions
            // must agree with the inherent ones.
            unary("CF::sqrt", <J1 as ComplexField>::sqrt, Some(f64::sqrt), x)?;
            unary("CF::sin", <J1 as ComplexField>::sin, Some(f64::sin), x)?;
            unary("CF::cos", <J1 as ComplexField>::cos, Some(f64::cos), x)?;
            unary("CF::tan", <J1 as ComplexField>::tan, Some(f64::tan), x)?;
            unary("CF::asin", <J1 as ComplexField>::asin, Some(f64::asin), x)?;
            unary("CF::acos", <J1 as ComplexField>::acos, Some(f64::acos), x)?;
            unary("CF::atan", <J1 as ComplexField>::atan, Some(f64::atan), x)?;
            unary("CF::sinh", <J1 as ComplexField>::sinh, Some(f64::sinh), x)?;
            unary("CF::cosh", <J1 as ComplexField>::cosh, Some(f64::cosh), x)?;
            unary("CF::tanh", <J1 as ComplexField>::tanh, Some(f64::tanh), x)?;
            unary("CF::exp", <J1 as ComplexField>::exp, Some(f64::exp), x)?;
            unary("CF::exp2", <J1 as ComplexField>::exp2, Some(f64::exp2), x)?;
            unary("CF::ln", <J1 as ComplexField>::ln, Some(f64::ln), x)?;
            unary("CF::log2", <J1 as ComplexField>::log2, Some(f64::log2), x)?;
            unary("CF::log10", <J1 as ComplexField>::log10, Some(f64::log10), x)?;
            unary("CF::cbrt", <J1 as ComplexField>::cbrt, Some(f64::cbrt), x)?;
            unary("CF::abs", <J1 as ComplexField>::abs, Some(f64::abs), x)?;
            unary("CF::log(3)", |j| <J1 as ComplexField>::log(j, J1::constant(3.0)), None, x)?;
            binary("CF::powf", <J2 as ComplexField>::powf, x, x + 1.0)?;
            binary("CF::hypot", <J2 as ComplexField>::hypot, x, x - 2.0)?;
            binary("RF::atan2", <J2 as nalgebra::RealField>::atan2, x, x - 2.0)?;
        }
    }
}

// ---------------------------------------------------------------------------
// Not in jet_test.cc: nalgebra generic code on jets.
// ---------------------------------------------------------------------------
mod nalgebra_generic {
    use ceres::{Jet, Scalar};
    use nalgebra::{Matrix3, Point3, RealField, UnitQuaternion, Vector2, Vector3};
    use proptest::prelude::*;

    const FD_H: f64 = 1e-6;
    const FD_TOL: f64 = 1e-6;

    fn fd_close(exact: f64, est: f64) -> bool {
        (exact - est).abs() <= FD_TOL * exact.abs().max(1.0)
    }

    /// Central-difference gradient of an `f64` function of 9 parameters.
    fn fd_gradient9(f: impl Fn(&[f64; 9]) -> f64, p: &[f64; 9]) -> [f64; 9] {
        let mut g = [0.0; 9];
        for k in 0..9 {
            let (mut pp, mut pm) = (*p, *p);
            pp[k] += FD_H;
            pm[k] -= FD_H;
            g[k] = (f(&pp) - f(&pm)) / (2.0 * FD_H);
        }
        g
    }

    /// Matrix whose 9 entries are the jet variables (row-major indexing).
    fn variable_matrix(p: &[f64; 9]) -> Matrix3<Jet<9>> {
        Matrix3::from_fn(|r, c| Jet::variable(p[3 * r + c], 3 * r + c))
    }

    fn f64_matrix(p: &[f64; 9]) -> Matrix3<f64> {
        Matrix3::from_fn(|r, c| p[3 * r + c])
    }

    /// A well-conditioned matrix: identity * 3 plus entries in [-1, 1].
    fn well_conditioned() -> impl Strategy<Value = [f64; 9]> {
        proptest::array::uniform9(-1.0f64..1.0).prop_map(|mut p| {
            p[0] += 3.0;
            p[4] += 3.0;
            p[8] += 3.0;
            p
        })
    }

    #[test]
    fn vector3_norm_gradient() {
        let x = [1.5, -2.0, 0.5];
        let v = Vector3::new(
            Jet::<3>::variable(x[0], 0),
            Jet::variable(x[1], 1),
            Jet::variable(x[2], 2),
        );
        let n = v.norm();
        let expected = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        assert!((n.a - expected).abs() < 1e-15);
        for k in 0..3 {
            assert!(
                (n.v[k] - x[k] / expected).abs() < 1e-15,
                "k={k}: {}",
                n.v[k]
            );
        }
        // normalize() also works and has a unit scalar part.
        let u = v.normalize();
        assert!((u.norm().a - 1.0).abs() < 1e-15);
    }

    #[test]
    fn vector_ops_dot_cross() {
        let a = Vector3::new(
            Jet::<3>::variable(1.0, 0),
            Jet::variable(2.0, 1),
            Jet::variable(3.0, 2),
        );
        let b = Vector3::new(Jet::constant(-1.0), Jet::constant(0.5), Jet::constant(4.0));
        let d = a.dot(&b);
        assert_eq!(d.a, -1.0 + 1.0 + 12.0);
        assert_eq!(d.v, [-1.0, 0.5, 4.0]);
        let c = a.cross(&b);
        // d(a x b)_0 / da = (0, b2, -b1)
        assert_eq!(c[0].v, [0.0, 4.0, -0.5]);
    }

    proptest! {
        #[test]
        fn matrix3_determinant_gradient(p in well_conditioned()) {
            let det = variable_matrix(&p).determinant();
            prop_assert_eq!(det.a, f64_matrix(&p).determinant());
            let g = fd_gradient9(|q| f64_matrix(q).determinant(), &p);
            // Analytic: d det / dA = det * A^{-T}.
            let cof = f64_matrix(&p).try_inverse().unwrap().transpose() * det.a;
            for k in 0..9 {
                prop_assert!(fd_close(det.v[k], g[k]), "k={} jet={} fd={}", k, det.v[k], g[k]);
                prop_assert!((det.v[k] - cof[(k / 3, k % 3)]).abs() < 1e-9 * det.a.abs().max(1.0));
            }
        }

        #[test]
        fn matrix3_inverse_derivative(p in well_conditioned(), r in 0usize..3, c in 0usize..3) {
            let inv = variable_matrix(&p).try_inverse().expect("singular");
            let inv_f = f64_matrix(&p).try_inverse().unwrap();
            prop_assert!((inv[(r, c)].a - inv_f[(r, c)]).abs() < 1e-14);
            let g = fd_gradient9(|q| f64_matrix(q).try_inverse().unwrap()[(r, c)], &p);
            for k in 0..9 {
                prop_assert!(fd_close(inv[(r, c)].v[k], g[k]), "k={} jet={} fd={}", k, inv[(r, c)].v[k], g[k]);
            }
        }

        #[test]
        fn matrix3_product_derivative(
            p in well_conditioned(),
            q in proptest::array::uniform9(-2.0f64..2.0),
            r in 0usize..3,
            c in 0usize..3,
        ) {
            // (A * B * A)[r, c] where A holds the jet variables.
            let a = variable_matrix(&p);
            let b = f64_matrix(&q).cast::<Jet<9>>();
            let prod = a * b * a;
            let f = |x: &[f64; 9]| (f64_matrix(x) * f64_matrix(&q) * f64_matrix(x))[(r, c)];
            prop_assert_eq!(prod[(r, c)].a, f(&p));
            let g = fd_gradient9(f, &p);
            for k in 0..9 {
                prop_assert!(fd_close(prod[(r, c)].v[k], g[k]), "k={} jet={} fd={}", k, prod[(r, c)].v[k], g[k]);
            }
        }

        #[test]
        fn unit_quaternion_rotates_point(
            w in proptest::array::uniform3(-2.0f64..2.0),
            pt in proptest::array::uniform3(-5.0f64..5.0),
        ) {
            prop_assume!(Vector3::from(w).norm() > 0.1);
            let rotate = |w: [f64; 3]| -> Vector3<f64> {
                let q = UnitQuaternion::from_scaled_axis(Vector3::from(w));
                (q * Point3::from(pt)).coords
            };
            let wj = Vector3::new(
                Jet::<3>::variable(w[0], 0),
                Jet::variable(w[1], 1),
                Jet::variable(w[2], 2),
            );
            let q = UnitQuaternion::from_scaled_axis(wj);
            let pj = Point3::from(Vector3::from(pt).cast::<Jet<3>>());
            let rotated = q * pj;
            let expected = rotate(w);
            for i in 0..3 {
                prop_assert!((rotated[i].a - expected[i]).abs() < 1e-12);
                for k in 0..3 {
                    let (mut wp, mut wm) = (w, w);
                    wp[k] += FD_H;
                    wm[k] -= FD_H;
                    let est = (rotate(wp)[i] - rotate(wm)[i]) / (2.0 * FD_H);
                    prop_assert!(fd_close(rotated[i].v[k], est), "i={} k={} jet={} fd={}", i, k, rotated[i].v[k], est);
                }
            }
        }

        #[test]
        fn generic_function_same_values_on_f64_and_jet(
            x in 0.1f64..3.0,
            y in -3.0f64..3.0,
        ) {
            let fx = generic_function(x, y);
            let fj = generic_function(Jet::<2>::variable(x, 0), Jet::variable(y, 1));
            prop_assert_eq!(fx.to_bits(), fj.a.to_bits());
            prop_assert_eq!(fj.scalar_part().to_bits(), fx.to_bits());
            let dx = (generic_function(x + FD_H, y) - generic_function(x - FD_H, y)) / (2.0 * FD_H);
            let dy = (generic_function(x, y + FD_H) - generic_function(x, y - FD_H)) / (2.0 * FD_H);
            prop_assert!(fd_close(fj.v[0], dx), "jet={} fd={}", fj.v[0], dx);
            prop_assert!(fd_close(fj.v[1], dy), "jet={} fd={}", fj.v[1], dy);
        }
    }

    /// Generic code written once for `f64` and jets (uses the
    /// ComplexField/RealField methods and nalgebra containers). `powi` and
    /// `mul_add` are avoided because the `f64` versions are not bitwise
    /// equal to `powf` / `a * b + c`. Jet / Jet division is also avoided:
    /// like C++ jet.h it computes the value as `a * (1 / b)`, which can
    /// differ from `a / b` in the last bit, so `* recip()` is used instead.
    fn generic_function<T: Scalar>(x: T, y: T) -> T {
        let v = Vector2::new(x, y);
        let r = v.norm();
        let half = T::constant(0.5);
        x.sin() * y
            + (x * x + T::one()).ln()
            + y.atan2(x) * half
            + (r + T::constant(1.0)).sqrt()
            + RealField::max(x, y).exp()
            + x.cbrt() * (T::constant(2.0) + y.tanh()).recip()
    }

    #[test]
    fn scalar_constant_and_scalar_part() {
        let j = <Jet<4> as Scalar>::constant(2.5);
        assert_eq!(j.a, 2.5);
        assert_eq!(j.v, [0.0; 4]);
        assert_eq!(Jet::<4>::variable(1.25, 3).scalar_part(), 1.25);
        assert_eq!(<f64 as Scalar>::constant(2.5), 2.5);
        assert_eq!(Jet::<4>::DIMENSION, 4);
    }
}
