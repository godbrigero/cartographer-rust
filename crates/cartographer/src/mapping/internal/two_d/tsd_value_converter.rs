//! Port of `cartographer/mapping/internal/2d/tsd_value_converter` (.h, .cc).

use std::fmt;
use std::sync::Arc;

use crate::common::math::clamp;
use crate::common::port::round_to_int;
use crate::mapping::value_conversion_tables::ValueConversionTables;

const MIN_WEIGHT: f32 = 0.0;
const UNKNOWN_TSD_VALUE: u16 = 0;
const UNKNOWN_WEIGHT_VALUE: u16 = 0;
const UPDATE_MARKER: u16 = 1 << 15;

/// Provides conversions between float and `u16` representations for
/// truncated signed distance values and weights.
#[derive(Clone)]
pub struct TSDValueConverter {
    max_tsd: f32,
    min_tsd: f32,
    max_weight: f32,
    tsd_resolution: f32,
    weight_resolution: f32,
    value_to_tsd: Arc<[f32]>,
    value_to_weight: Arc<[f32]>,
}

impl fmt::Debug for TSDValueConverter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TSDValueConverter")
            .field("max_tsd", &self.max_tsd)
            .field("min_tsd", &self.min_tsd)
            .field("max_weight", &self.max_weight)
            .finish_non_exhaustive()
    }
}

impl TSDValueConverter {
    /// Creates a converter for TSD values in `[-max_tsd, max_tsd]` and weights
    /// in `[0, max_weight]`. Unknown TSD decodes to `-max_tsd`, unknown weight
    /// to 0.
    pub fn new(max_tsd: f32, max_weight: f32, conversion_tables: &ValueConversionTables) -> Self {
        let min_tsd = -max_tsd;
        Self {
            max_tsd,
            min_tsd,
            max_weight,
            tsd_resolution: 32766.0_f32 / (max_tsd - min_tsd),
            weight_resolution: 32766.0_f32 / (max_weight - MIN_WEIGHT),
            value_to_tsd: conversion_tables.get_conversion_table(min_tsd, min_tsd, max_tsd),
            value_to_weight: conversion_tables
                .get_conversion_table(MIN_WEIGHT, MIN_WEIGHT, max_weight),
        }
    }

    /// Converts a tsd to a `u16` in the [1, 32767] range.
    #[inline]
    pub fn tsd_to_value(&self, tsd: f32) -> u16 {
        let value = round_to_int((self.clamp_tsd(tsd) - self.min_tsd) * self.tsd_resolution) + 1;
        debug_assert!(value >= 1);
        debug_assert!(value <= 32767);
        value as u16
    }

    /// Converts a weight to a `u16` in the [1, 32767] range.
    #[inline]
    pub fn weight_to_value(&self, weight: f32) -> u16 {
        let value =
            round_to_int((self.clamp_weight(weight) - MIN_WEIGHT) * self.weight_resolution) + 1;
        debug_assert!(value >= 1);
        debug_assert!(value <= 32767);
        value as u16
    }

    /// Converts a `u16` (which may or may not have the update marker set) to
    /// a value in the range [min_tsd, max_tsd].
    #[inline]
    pub fn value_to_tsd(&self, value: u16) -> f32 {
        self.value_to_tsd[value as usize]
    }

    /// Converts a `u16` (which may or may not have the update marker set) to
    /// a value in the range [min_weight, max_weight].
    #[inline]
    pub fn value_to_weight(&self, value: u16) -> f32 {
        self.value_to_weight[value as usize]
    }

    /// Value of an unknown TSD cell (C++ `getUnknownTSDValue`).
    pub fn get_unknown_tsd_value() -> u16 {
        UNKNOWN_TSD_VALUE
    }

    /// Value of an unknown weight cell (C++ `getUnknownWeightValue`).
    pub fn get_unknown_weight_value() -> u16 {
        UNKNOWN_WEIGHT_VALUE
    }

    /// The update marker bit (C++ `getUpdateMarker`).
    pub fn get_update_marker() -> u16 {
        UPDATE_MARKER
    }

    /// Maximum TSD (= truncation distance).
    pub fn get_max_tsd(&self) -> f32 {
        self.max_tsd
    }

    /// Minimum TSD (= -truncation distance).
    pub fn get_min_tsd(&self) -> f32 {
        self.min_tsd
    }

    /// Maximum weight.
    pub fn get_max_weight(&self) -> f32 {
        self.max_weight
    }

    /// Minimum weight (always 0).
    pub fn get_min_weight(&self) -> f32 {
        MIN_WEIGHT
    }

    // Clamps TSD to be in the range [min_tsd, max_tsd].
    #[inline]
    fn clamp_tsd(&self, tsd: f32) -> f32 {
        clamp(tsd, self.min_tsd, self.max_tsd)
    }

    // Clamps weight to be in the range [min_weight, max_weight].
    #[inline]
    fn clamp_weight(&self, weight: f32) -> f32 {
        clamp(weight, MIN_WEIGHT, self.max_weight)
    }
}

#[cfg(test)]
mod tests {
    mod tsd_value_converter_test {
        use crate::mapping::internal::two_d::tsd_value_converter::TSDValueConverter;
        use crate::mapping::value_conversion_tables::ValueConversionTables;

        struct Fixture {
            truncation_distance: f32,
            max_weight: f32,
            tsd_value_converter: TSDValueConverter,
        }

        fn fixture() -> Fixture {
            let conversion_tables = ValueConversionTables::new();
            let truncation_distance = 0.1_f32;
            let max_weight = 10.0_f32;
            Fixture {
                truncation_distance,
                max_weight,
                tsd_value_converter: TSDValueConverter::new(
                    truncation_distance,
                    max_weight,
                    &conversion_tables,
                ),
            }
        }

        fn assert_near(a: f32, b: f32, tolerance: f32) {
            assert!(
                (a - b).abs() <= tolerance,
                "{a} vs {b} (tolerance {tolerance})"
            );
        }

        #[test]
        fn default_values() {
            let f = fixture();
            assert_eq!(TSDValueConverter::get_unknown_weight_value(), 0);
            assert_eq!(TSDValueConverter::get_unknown_tsd_value(), 0);
            assert_eq!(f.tsd_value_converter.get_min_tsd(), -f.truncation_distance);
            assert_eq!(f.tsd_value_converter.get_max_tsd(), f.truncation_distance);
            assert_eq!(f.tsd_value_converter.get_min_weight(), 0.0);
            assert_eq!(f.tsd_value_converter.get_max_weight(), f.max_weight);
        }

        #[test]
        fn value_to_tsd_conversions() {
            let c = fixture().tsd_value_converter;
            for i in 1..32768_u16 {
                assert_eq!(c.tsd_to_value(c.value_to_tsd(i)), i);
            }
        }

        #[test]
        fn value_to_tsd_conversions_with_update_marker() {
            let c = fixture().tsd_value_converter;
            for i in 1..32768_u16 {
                assert_eq!(
                    c.tsd_to_value(c.value_to_tsd(i + TSDValueConverter::get_update_marker())),
                    i
                );
            }
        }

        #[test]
        fn value_to_weight_conversions() {
            let c = fixture().tsd_value_converter;
            for i in 1..32768_u16 {
                assert_eq!(c.weight_to_value(c.value_to_weight(i)), i);
            }
        }

        #[test]
        fn value_to_weight_conversions_with_update_marker() {
            let c = fixture().tsd_value_converter;
            for i in 1..32768_u16 {
                assert_eq!(
                    c.weight_to_value(
                        c.value_to_weight(i + TSDValueConverter::get_update_marker())
                    ),
                    i
                );
            }
        }

        #[test]
        fn tsd_to_value_conversions() {
            let f = fixture();
            let c = &f.tsd_value_converter;
            let num_samples: u16 = 1000;
            let tolerance = f.truncation_distance * 2.0 / 32767.0;
            for i in 0..num_samples {
                let sdf_sample = -f.truncation_distance
                    + f32::from(i) * 2.0 * f.truncation_distance / f32::from(num_samples);
                assert_near(
                    c.value_to_tsd(c.tsd_to_value(sdf_sample)),
                    sdf_sample,
                    tolerance,
                );
            }
        }

        #[test]
        fn weight_to_value_conversions() {
            let f = fixture();
            let c = &f.tsd_value_converter;
            let num_samples: u16 = 1000;
            let tolerance = f.max_weight / 32767.0;
            for i in 0..num_samples {
                let weight_sample = f32::from(i) * f.max_weight / f32::from(num_samples);
                assert_near(
                    c.value_to_weight(c.weight_to_value(weight_sample)),
                    weight_sample,
                    tolerance,
                );
            }
        }

        #[test]
        fn weight_to_value_out_of_range_conversions() {
            let f = fixture();
            let c = &f.tsd_value_converter;
            let tolerance = f.max_weight / 32767.0;
            assert_near(
                c.value_to_weight(c.weight_to_value(2.0 * f.max_weight)),
                f.max_weight,
                tolerance,
            );
            assert_near(
                c.value_to_weight(c.weight_to_value(-f.max_weight)),
                0.0,
                tolerance,
            );
        }

        #[test]
        fn tsd_to_value_out_of_range_conversions() {
            let f = fixture();
            let c = &f.tsd_value_converter;
            let tolerance = f.truncation_distance * 2.0 / 32767.0;
            assert_near(
                c.value_to_tsd(c.tsd_to_value(2.0 * f.truncation_distance)),
                f.truncation_distance,
                tolerance,
            );
            assert_near(
                c.value_to_tsd(c.tsd_to_value(-2.0 * f.truncation_distance)),
                -f.truncation_distance,
                tolerance,
            );
        }
    }

    mod proptests {
        use proptest::prelude::*;

        use crate::mapping::internal::two_d::tsd_value_converter::TSDValueConverter;
        use crate::mapping::value_conversion_tables::ValueConversionTables;

        proptest! {
            #[test]
            fn tsd_and_weight_round_trip(
                truncation in 0.05_f32..2.0, max_weight in 1.0_f32..100.0,
                ft in -1.0_f32..1.0, fw in 0.0_f32..1.0) {
                let c = TSDValueConverter::new(truncation, max_weight, &ValueConversionTables::new());
                let tsd = ft * truncation;
                let weight = fw * max_weight;
                // Half a quantization step plus float slack.
                let tsd_tolerance = 2.0 * truncation / 32766.0;
                let weight_tolerance = max_weight / 32766.0;
                prop_assert!((c.value_to_tsd(c.tsd_to_value(tsd)) - tsd).abs() <= tsd_tolerance);
                prop_assert!(
                    (c.value_to_weight(c.weight_to_value(weight)) - weight).abs() <= weight_tolerance);
                prop_assert_eq!(c.value_to_tsd(0), -truncation);
                prop_assert_eq!(c.value_to_weight(0), 0.0);
            }
        }
    }
}
