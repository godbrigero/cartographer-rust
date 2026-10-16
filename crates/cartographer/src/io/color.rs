//! Port of `cartographer/io/color` (.h, .cc).

use crate::common::math::clamp;
use crate::common::port::round_to_int;

/// An RGB color with 8-bit components.
pub type Uint8Color = [u8; 3];
/// An RGB color with float components in `[0, 1]`.
pub type FloatColor = [f32; 3];

const INITIAL_HUE: f32 = 0.69;
const SATURATION: f32 = 0.85;
const VALUE: f32 = 0.77;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> FloatColor {
    let h_6 = if h == 1.0 { 0.0 } else { 6.0 * h };
    let h_i = h_6.floor() as i32;
    let f = h_6 - h_i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match h_i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        5 => [v, p, q],
        _ => [0.0, 0.0, 0.0],
    }
}

/// A function for on-demand generation of a color palette, with every two
/// direct successors having large contrast.
pub fn get_color(id: i32) -> FloatColor {
    assert!(id >= 0, "Check failed: id >= 0 ({id} vs. 0)");
    // Uniform color sampling using the golden ratio from
    // http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_034;
    // `%` on f32 is C `fmodf`.
    let hue = (INITIAL_HUE + GOLDEN_RATIO_CONJUGATE * id as f32) % 1.0;
    hsv_to_rgb(hue, SATURATION, VALUE)
}

/// Converts a float component in `[0, 1]` (clamped) to `[0, 255]`, rounding
/// half away from zero.
#[inline]
pub fn float_component_to_uint8(c: f32) -> u8 {
    round_to_int(clamp(c, 0.0, 1.0) * 255.0) as u8
}

/// Converts an 8-bit component to a float in `[0, 1]`.
#[inline]
pub fn uint8_component_to_float(c: u8) -> f32 {
    f32::from(c) / 255.0
}

/// Converts a float color to an 8-bit color.
#[inline]
pub fn to_uint8_color(color: &FloatColor) -> Uint8Color {
    [
        float_component_to_uint8(color[0]),
        float_component_to_uint8(color[1]),
        float_component_to_uint8(color[2]),
    ]
}

/// Converts an 8-bit color to a float color.
#[inline]
pub fn to_float_color(color: &Uint8Color) -> FloatColor {
    [
        uint8_component_to_float(color[0]),
        uint8_component_to_float(color[1]),
        uint8_component_to_float(color[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_ratio_constant_matches_cpp_float() {
        // C++ writes the literal 0.6180339887498949f, which rounds to this f32.
        #[allow(clippy::excessive_precision)]
        let cpp = 0.6180339887498949_f32;
        assert_eq!(cpp.to_bits(), 0.618_034_f32.to_bits());
    }

    #[test]
    fn get_color_first_entries() {
        // id 0: hue 0.69 -> h_6 = 4.14 -> sector 4 -> (t, p, v).
        let c = get_color(0);
        let v = VALUE;
        let p = v * (1.0 - SATURATION);
        assert_eq!(c[1], p);
        assert_eq!(c[2], v);
        for id in 0..100 {
            let c = get_color(id);
            for component in c {
                assert!((0.0..=1.0).contains(&component));
            }
        }
    }

    #[test]
    #[should_panic(expected = "id >= 0")]
    fn get_color_negative_id_panics() {
        get_color(-1);
    }

    #[test]
    fn conversions() {
        assert_eq!(float_component_to_uint8(-1.0), 0);
        assert_eq!(float_component_to_uint8(2.0), 255);
        assert_eq!(float_component_to_uint8(0.5), 128);
        for c in 0..=255u8 {
            assert_eq!(float_component_to_uint8(uint8_component_to_float(c)), c);
        }
        assert_eq!(
            to_uint8_color(&to_float_color(&[1, 128, 255])),
            [1, 128, 255]
        );
    }
}
