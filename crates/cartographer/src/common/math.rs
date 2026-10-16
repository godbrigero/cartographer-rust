//! Port of `cartographer/common/math` (.h).
//!
//! The C++ templates are generic here over `nalgebra::RealField + Copy` (or
//! weaker bounds) so that they also work with autodiff `ceres::Jet`s.

use std::ops::Mul;

use nalgebra::{RealField, Vector2};
use num_traits::One;

pub use super::port::{round_to_int, round_to_int64};

/// Clamps `value` to be in the range [`min`, `max`]. NaN passes through
/// unchanged (unlike `f64::clamp` this never panics).
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value > max {
        return max;
    }
    if value < min {
        return min;
    }
    value
}

/// Calculates `base`^`exponent` by repeated multiplication
/// (`base * (base * (... * 1))`).
///
/// RUST PORT NOTE: a negative exponent recurses forever in C++; here it
/// panics instead.
pub fn power<T: Copy + Mul<Output = T> + One>(base: T, exponent: i32) -> T {
    assert!(exponent >= 0, "power() requires a non-negative exponent");
    if exponent != 0 {
        base * power(base, exponent - 1)
    } else {
        T::one()
    }
}

/// Calculates a^2.
pub fn pow2<T: Copy + Mul<Output = T> + One>(a: T) -> T {
    power(a, 2)
}

/// Converts from degrees to radians (`M_PI * deg / 180.`, evaluated in that
/// order).
pub fn deg_to_rad(deg: f64) -> f64 {
    std::f64::consts::PI * deg / 180.
}

/// Converts form radians to degrees (`180. * rad / M_PI`).
pub fn rad_to_deg(rad: f64) -> f64 {
    180. * rad / std::f64::consts::PI
}

/// Bring the `difference` between two angles into [-pi; pi].
pub fn normalize_angle_difference<T: RealField + Copy>(mut difference: T) -> T {
    let k_pi: T = T::pi();
    let two: T = nalgebra::convert(2.);
    while difference > k_pi {
        difference -= two * k_pi;
    }
    while difference < -k_pi {
        difference += two * k_pi;
    }
    difference
}

/// `atan2(vector.y, vector.x)`.
pub fn atan2<T: RealField + Copy>(vector: &Vector2<T>) -> T {
    vector.y.atan2(vector.x)
}

/// Quaternion product `zw = z * w` with ceres' `[w, x, y, z]` storage order.
pub fn quaternion_product<T: RealField + Copy>(z: &[f64; 4], w: &[T; 4], zw: &mut [T; 4]) {
    let z: [T; 4] = [
        nalgebra::convert(z[0]),
        nalgebra::convert(z[1]),
        nalgebra::convert(z[2]),
        nalgebra::convert(z[3]),
    ];
    zw[0] = z[0] * w[0] - z[1] * w[1] - z[2] * w[2] - z[3] * w[3];
    zw[1] = z[0] * w[1] + z[1] * w[0] + z[2] * w[3] - z[3] * w[2];
    zw[2] = z[0] * w[2] - z[1] * w[3] + z[2] * w[0] + z[3] * w[1];
    zw[3] = z[0] * w[3] + z[1] * w[2] - z[2] * w[1] + z[3] * w[0];
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn expect_near(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "{a} vs {b} (eps {eps})");
    }

    mod math_test {
        use super::*;

        #[test]
        fn test_power() {
            assert_eq!(0., f64::from(power(0, 42)));
            assert_eq!(1., f64::from(power(0, 0)));
            assert_eq!(1., f64::from(power(1, 0)));
            assert_eq!(1., f64::from(power(1, 42)));
            assert_eq!(4., f64::from(power(2, 2)));
        }

        #[test]
        fn test_pow2() {
            assert_eq!(0., f64::from(pow2(0)));
            assert_eq!(1., f64::from(pow2(1)));
            assert_eq!(4., f64::from(pow2(2)));
            assert_eq!(49., f64::from(pow2(7)));
        }

        #[test]
        fn test_deg2rad() {
            expect_near(PI, deg_to_rad(180.), 1e-9);
            expect_near(2. * PI, deg_to_rad(360. - 1e-9), 1e-6);
        }

        #[test]
        fn test_rad2deg() {
            expect_near(180., rad_to_deg(PI), 1e-9);
            expect_near(360., rad_to_deg(2. * PI - 1e-9), 1e-6);
        }

        #[test]
        fn test_normalize_angle_difference() {
            expect_near(0., normalize_angle_difference(0.), 1e-9);
            expect_near(PI, normalize_angle_difference(PI), 1e-9);
            expect_near(-PI, normalize_angle_difference(-PI), 1e-9);
            expect_near(0., normalize_angle_difference(2. * PI), 1e-9);
            expect_near(PI, normalize_angle_difference(5. * PI), 1e-9);
            expect_near(-PI, normalize_angle_difference(-5. * PI), 1e-9);
        }
    }

    #[test]
    fn clamp_matches_cpp() {
        assert_eq!(clamp(5, 0, 3), 3);
        assert_eq!(clamp(-5, 0, 3), 0);
        assert_eq!(clamp(2.5, 0., 3.), 2.5);
        assert!(clamp(f64::NAN, 0., 1.).is_nan());
    }

    #[test]
    fn atan2_and_quaternion_product() {
        expect_near(atan2(&Vector2::new(1., 1.)), PI / 4., 1e-15);
        // i * j = k
        let mut zw = [0.; 4];
        quaternion_product(&[0., 1., 0., 0.], &[0., 0., 1., 0.], &mut zw);
        assert_eq!(zw, [0., 0., 0., 1.]);
        let mut zw = [0.; 4];
        quaternion_product(&[1., 2., 3., 4.], &[5., 6., 7., 8.], &mut zw);
        assert_eq!(zw, [-60., 12., 30., 24.]);
    }

    proptest::proptest! {
        #[test]
        fn normalize_angle_difference_in_range(x in -1000.0f64..1000.0) {
            let n = normalize_angle_difference(x);
            proptest::prop_assert!((-PI..=PI).contains(&n));
            // Same angle modulo 2 pi.
            let k = ((x - n) / (2. * PI)).round();
            proptest::prop_assert!((x - n - k * 2. * PI).abs() < 1e-9);
        }

        #[test]
        fn deg_rad_round_trip(x in -1000.0f64..1000.0) {
            proptest::prop_assert!((rad_to_deg(deg_to_rad(x)) - x).abs() < 1e-9);
        }
    }
}
