//! Port of `cartographer/mapping/probability_values` (.h, .cc).
//!
//! Probabilities and correspondence costs are stored in grids as `u16`
//! values: `0` means unknown, `[1, 32767]` maps linearly to
//! `[lower_bound, upper_bound]` and bit 15 ([`UPDATE_MARKER`]) flags cells
//! updated during the current insertion. All arithmetic is `f32`, like C++.

use std::sync::LazyLock;

use crate::common::math::clamp;
use crate::common::port::round_to_int;

/// Number of distinct values without the update marker (C++ `kValueCount`).
const VALUE_COUNT: usize = 32768;

/// Converts `float_value` (clamped to `[lower_bound, upper_bound]`) to a
/// value in `[1, 32767]` (C++ anonymous-namespace `BoundedFloatToValue`).
#[inline]
fn bounded_float_to_value(float_value: f32, lower_bound: f32, upper_bound: f32) -> u16 {
    let value = round_to_int(
        (clamp(float_value, lower_bound, upper_bound) - lower_bound)
            * (32766.0_f32 / (upper_bound - lower_bound)),
    ) + 1;
    // DCHECK for performance.
    debug_assert!(value >= 1, "Check failed: value >= 1 ({value})");
    debug_assert!(value <= 32767, "Check failed: value <= 32767 ({value})");
    value as u16
}

/// Converts a probability to odds: `p / (1 - p)`.
#[inline]
pub fn odds(probability: f32) -> f32 {
    probability / (1.0 - probability)
}

/// Converts odds to a probability: `o / (o + 1)`.
#[inline]
pub fn probability_from_odds(odds: f32) -> f32 {
    odds / (odds + 1.0)
}

/// Converts a probability to a correspondence cost: `1 - p`.
#[inline]
pub fn probability_to_correspondence_cost(probability: f32) -> f32 {
    1.0 - probability
}

/// Converts a correspondence cost to a probability: `1 - c`.
#[inline]
pub fn correspondence_cost_to_probability(correspondence_cost: f32) -> f32 {
    1.0 - correspondence_cost
}

/// Minimum representable probability (`kMinProbability`).
pub const MIN_PROBABILITY: f32 = 0.1;
/// Maximum representable probability (`kMaxProbability = 1 - kMinProbability`).
pub const MAX_PROBABILITY: f32 = 1.0 - MIN_PROBABILITY;
/// Minimum correspondence cost (`kMinCorrespondenceCost = 1 - kMaxProbability`).
pub const MIN_CORRESPONDENCE_COST: f32 = 1.0 - MAX_PROBABILITY;
/// Maximum correspondence cost (`kMaxCorrespondenceCost = 1 - kMinProbability`).
pub const MAX_CORRESPONDENCE_COST: f32 = 1.0 - MIN_PROBABILITY;

/// Clamps probability to be in the range [`MIN_PROBABILITY`, `MAX_PROBABILITY`].
#[inline]
pub fn clamp_probability(probability: f32) -> f32 {
    clamp(probability, MIN_PROBABILITY, MAX_PROBABILITY)
}

/// Clamps correspondence cost to be in the range
/// [`MIN_CORRESPONDENCE_COST`, `MAX_CORRESPONDENCE_COST`].
#[inline]
pub fn clamp_correspondence_cost(correspondence_cost: f32) -> f32 {
    clamp(
        correspondence_cost,
        MIN_CORRESPONDENCE_COST,
        MAX_CORRESPONDENCE_COST,
    )
}

/// Value of an unknown probability cell (`kUnknownProbabilityValue`).
pub const UNKNOWN_PROBABILITY_VALUE: u16 = 0;
/// Value of an unknown correspondence cost cell (`kUnknownCorrespondenceValue`).
pub const UNKNOWN_CORRESPONDENCE_VALUE: u16 = UNKNOWN_PROBABILITY_VALUE;
/// Bit flagging a cell as updated during the current insertion (`kUpdateMarker`).
pub const UPDATE_MARKER: u16 = 1 << 15;

/// Converts a correspondence cost to a `u16` in the [1, 32767] range.
#[inline]
pub fn correspondence_cost_to_value(correspondence_cost: f32) -> u16 {
    bounded_float_to_value(
        correspondence_cost,
        MIN_CORRESPONDENCE_COST,
        MAX_CORRESPONDENCE_COST,
    )
}

/// Converts a probability to a `u16` in the [1, 32767] range.
#[inline]
pub fn probability_to_value(probability: f32) -> u16 {
    bounded_float_to_value(probability, MIN_PROBABILITY, MAX_PROBABILITY)
}

/// 0 is unknown, [1, 32767] maps to [lower_bound, upper_bound]
/// (C++ `SlowValueToBoundedFloat` in `probability_values.cc`).
fn slow_value_to_bounded_float(
    value: u16,
    unknown_value: u16,
    unknown_result: f32,
    lower_bound: f32,
    upper_bound: f32,
) -> f32 {
    assert!(
        (value as usize) < VALUE_COUNT,
        "Check failed: value < kValueCount ({value} vs. {VALUE_COUNT})"
    );
    if value == unknown_value {
        return unknown_result;
    }
    let scale = (upper_bound - lower_bound) / (VALUE_COUNT as f32 - 2.0);
    f32::from(value) * scale + (lower_bound - scale)
}

/// C++ `PrecomputeValueToBoundedFloat`: repeats the 32768 values two times, so
/// that both values with and without the update marker can be converted.
fn precompute_value_to_bounded_float(
    unknown_value: u16,
    unknown_result: f32,
    lower_bound: f32,
    upper_bound: f32,
) -> Vec<f32> {
    const REPETITION_COUNT: usize = 2;
    let mut result = Vec::with_capacity(REPETITION_COUNT * VALUE_COUNT);
    for _ in 0..REPETITION_COUNT {
        for value in 0..VALUE_COUNT {
            result.push(slow_value_to_bounded_float(
                value as u16,
                unknown_value,
                unknown_result,
                lower_bound,
                upper_bound,
            ));
        }
    }
    result
}

/// Global table (65536 entries) mapping a value (with or without update
/// marker) to a probability; unknown maps to `MIN_PROBABILITY`
/// (C++ `kValueToProbability`).
pub static VALUE_TO_PROBABILITY: LazyLock<Vec<f32>> = LazyLock::new(|| {
    precompute_value_to_bounded_float(
        UNKNOWN_PROBABILITY_VALUE,
        MIN_PROBABILITY,
        MIN_PROBABILITY,
        MAX_PROBABILITY,
    )
});

/// Global table (65536 entries) mapping a value (with or without update
/// marker) to a correspondence cost; unknown maps to
/// `MAX_CORRESPONDENCE_COST` (C++ `kValueToCorrespondenceCost`).
pub static VALUE_TO_CORRESPONDENCE_COST: LazyLock<Vec<f32>> = LazyLock::new(|| {
    precompute_value_to_bounded_float(
        UNKNOWN_CORRESPONDENCE_VALUE,
        MAX_CORRESPONDENCE_COST,
        MIN_CORRESPONDENCE_COST,
        MAX_CORRESPONDENCE_COST,
    )
});

/// Converts a `u16` (which may or may not have the update marker set) to a
/// probability in the range [`MIN_PROBABILITY`, `MAX_PROBABILITY`].
#[inline]
pub fn value_to_probability(value: u16) -> f32 {
    VALUE_TO_PROBABILITY[value as usize]
}

/// Converts a `u16` (which may or may not have the update marker set) to a
/// correspondence cost in the range
/// [`MIN_CORRESPONDENCE_COST`, `MAX_CORRESPONDENCE_COST`].
#[inline]
pub fn value_to_correspondence_cost(value: u16) -> f32 {
    VALUE_TO_CORRESPONDENCE_COST[value as usize]
}

/// Converts a probability value to the corresponding correspondence cost
/// value, preserving the update marker.
#[inline]
pub fn probability_value_to_correspondence_cost_value(mut probability_value: u16) -> u16 {
    if probability_value == UNKNOWN_PROBABILITY_VALUE {
        return UNKNOWN_CORRESPONDENCE_VALUE;
    }
    let mut update_carry = false;
    // Note: `>` (not `>=`), as in C++.
    if probability_value > UPDATE_MARKER {
        probability_value -= UPDATE_MARKER;
        update_carry = true;
    }
    let mut result = correspondence_cost_to_value(probability_to_correspondence_cost(
        value_to_probability(probability_value),
    ));
    if update_carry {
        result += UPDATE_MARKER;
    }
    result
}

/// Converts a correspondence cost value to the corresponding probability
/// value, preserving the update marker.
#[inline]
pub fn correspondence_cost_value_to_probability_value(mut correspondence_cost_value: u16) -> u16 {
    if correspondence_cost_value == UNKNOWN_CORRESPONDENCE_VALUE {
        return UNKNOWN_PROBABILITY_VALUE;
    }
    let mut update_carry = false;
    if correspondence_cost_value > UPDATE_MARKER {
        correspondence_cost_value -= UPDATE_MARKER;
        update_carry = true;
    }
    let mut result = probability_to_value(correspondence_cost_to_probability(
        value_to_correspondence_cost(correspondence_cost_value),
    ));
    if update_carry {
        result += UPDATE_MARKER;
    }
    result
}

/// Computes a 32768-entry table that applies `odds` to a probability value.
/// Every resulting entry carries the update marker.
pub fn compute_lookup_table_to_apply_odds(odds_value: f32) -> Vec<u16> {
    let mut result = Vec::with_capacity(VALUE_COUNT);
    result.push(probability_to_value(probability_from_odds(odds_value)) + UPDATE_MARKER);
    for cell in 1..VALUE_COUNT {
        result.push(
            probability_to_value(probability_from_odds(
                odds_value * odds(VALUE_TO_PROBABILITY[cell]),
            )) + UPDATE_MARKER,
        );
    }
    result
}

/// Computes a 32768-entry table that applies `odds` (of the *probability*) to
/// a correspondence cost value. Every resulting entry carries the update
/// marker. This is the table used by the 2D probability grid inserter.
pub fn compute_lookup_table_to_apply_correspondence_cost_odds(odds_value: f32) -> Vec<u16> {
    let mut result = Vec::with_capacity(VALUE_COUNT);
    result.push(
        correspondence_cost_to_value(probability_to_correspondence_cost(probability_from_odds(
            odds_value,
        ))) + UPDATE_MARKER,
    );
    for cell in 1..VALUE_COUNT {
        result.push(
            correspondence_cost_to_value(probability_to_correspondence_cost(
                probability_from_odds(
                    odds_value
                        * odds(correspondence_cost_to_probability(
                            VALUE_TO_CORRESPONDENCE_COST[cell],
                        )),
                ),
            )) + UPDATE_MARKER,
        );
    }
    result
}

#[cfg(test)]
mod tests {

    fn assert_near(a: f32, b: f32, eps: f64) {
        assert!(
            (f64::from(a) - f64::from(b)).abs() <= eps,
            "{a} vs {b} (tolerance {eps})"
        );
    }

    mod probability_values_test {
        use super::super::*;
        use super::assert_near;

        #[test]
        fn odds_conversions() {
            assert_near(
                probability_from_odds(odds(MIN_PROBABILITY)),
                MIN_PROBABILITY,
                1e-6,
            );
            assert_near(
                probability_from_odds(odds(MAX_PROBABILITY)),
                MAX_PROBABILITY,
                1e-6,
            );
            assert_near(probability_from_odds(odds(0.5)), 0.5, 1e-6);
        }

        #[test]
        fn odds_conversions_correspondence_cost() {
            assert_near(
                probability_to_correspondence_cost(probability_from_odds(odds(
                    correspondence_cost_to_probability(MAX_CORRESPONDENCE_COST),
                ))),
                MAX_CORRESPONDENCE_COST,
                1e-6,
            );
            assert_near(
                probability_to_correspondence_cost(probability_from_odds(odds(
                    correspondence_cost_to_probability(MIN_CORRESPONDENCE_COST),
                ))),
                MIN_CORRESPONDENCE_COST,
                1e-6,
            );
            assert_near(
                probability_to_correspondence_cost(probability_from_odds(odds(
                    correspondence_cost_to_probability(0.5),
                ))),
                0.5,
                1e-6,
            );
        }

        #[test]
        fn probability_value_to_correspondence_cost_value_conversions() {
            for i in 0..32768_u16 {
                assert_eq!(
                    probability_value_to_correspondence_cost_value(
                        correspondence_cost_value_to_probability_value(i)
                    ),
                    i
                );
                assert_eq!(
                    correspondence_cost_value_to_probability_value(
                        probability_value_to_correspondence_cost_value(i)
                    ),
                    i
                );
            }
        }

        #[test]
        fn probability_value_to_correspondence_cost_value_conversions_with_update_marker() {
            for i in 1..32768_u16 {
                assert_eq!(
                    probability_value_to_correspondence_cost_value(
                        correspondence_cost_value_to_probability_value(i + UPDATE_MARKER)
                    ),
                    i + UPDATE_MARKER
                );
                assert_eq!(
                    correspondence_cost_value_to_probability_value(
                        probability_value_to_correspondence_cost_value(i + UPDATE_MARKER)
                    ),
                    i + UPDATE_MARKER
                );
            }
        }

        #[test]
        fn conversion_look_up_table() {
            assert_near(
                value_to_probability(0),
                1.0 - value_to_correspondence_cost(0),
                1e-6,
            );
            for i in 1..32768_u16 {
                assert_near(
                    value_to_probability(i),
                    value_to_correspondence_cost(i),
                    1e-6,
                );
            }
        }

        #[test]
        fn cell_update() {
            let probability_table = compute_lookup_table_to_apply_odds(odds(0.9));
            let correspondence_table =
                compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.9));
            let mut cell_pg_pre_update: u16 = 0;
            let mut cell_cg_pre_update: u16 = 0;
            let mut cell_pg_post_update = probability_table[cell_pg_pre_update as usize];
            let mut cell_cg_post_update = correspondence_table[cell_cg_pre_update as usize];
            let mut p_post = value_to_probability(cell_pg_post_update);
            let mut c_post = value_to_correspondence_cost(cell_cg_post_update);
            assert_near(p_post, 1.0 - c_post, 1e-6);
            let num_evaluations = 5000;
            for i_probability in 0..num_evaluations {
                let p = (i_probability as f32 / num_evaluations as f32)
                    * (MAX_PROBABILITY - MIN_PROBABILITY)
                    + MIN_PROBABILITY;
                cell_pg_pre_update = probability_to_value(p);
                cell_cg_pre_update =
                    correspondence_cost_to_value(probability_to_correspondence_cost(p));
                assert!(
                    (i32::from(cell_pg_pre_update) - (32768 - i32::from(cell_cg_pre_update))).abs()
                        <= 1,
                    "p {p}"
                );
                cell_pg_post_update = probability_table[cell_pg_pre_update as usize];
                cell_cg_post_update = correspondence_table[cell_cg_pre_update as usize];
                p_post = value_to_probability(cell_pg_post_update);
                c_post = value_to_correspondence_cost(cell_cg_post_update);
                assert_near(p_post, 1.0 - c_post, 5e-5);
            }
        }

        #[test]
        fn multiple_cell_update() {
            let probability_table = compute_lookup_table_to_apply_odds(odds(0.55));
            let correspondence_table =
                compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.55));
            let mut cell_pg_post_update = probability_table[0];
            let mut cell_cg_post_update = correspondence_table[0];
            let mut p_post = value_to_probability(cell_pg_post_update);
            let mut c_post = value_to_correspondence_cost(cell_cg_post_update);
            assert_near(p_post, 1.0 - c_post, 1e-6);
            let num_evaluations = 5000;
            for i_probability in 0..num_evaluations {
                let p = (i_probability as f32 / num_evaluations as f32)
                    * (MAX_PROBABILITY - MIN_PROBABILITY)
                    + MIN_PROBABILITY;
                cell_pg_post_update = probability_to_value(p) + UPDATE_MARKER;
                cell_cg_post_update =
                    correspondence_cost_to_value(probability_to_correspondence_cost(p))
                        + UPDATE_MARKER;
                for _ in 0..20 {
                    cell_pg_post_update =
                        probability_table[(cell_pg_post_update - UPDATE_MARKER) as usize];
                    cell_cg_post_update =
                        correspondence_table[(cell_cg_post_update - UPDATE_MARKER) as usize];
                }
                p_post = value_to_probability(cell_pg_post_update);
                c_post = value_to_correspondence_cost(cell_cg_post_update);
                assert_near(p_post, 1.0 - c_post, 5e-5);
            }
        }

        #[test]
        fn equality_lookup_table_to_apply_odds() {
            let probability_table = compute_lookup_table_to_apply_odds(0.3);
            let correspondence_table = compute_lookup_table_to_apply_correspondence_cost_odds(0.3);
            for i in 0..32768_u16 {
                let a = i32::from(probability_table[i as usize]);
                let b = i32::from(correspondence_cost_value_to_probability_value(
                    correspondence_table
                        [probability_value_to_correspondence_cost_value(i) as usize],
                ));
                assert!((a - b).abs() <= 1, "i {i}: {a} vs {b}");
                let a = i32::from(probability_value_to_correspondence_cost_value(
                    probability_table[correspondence_cost_value_to_probability_value(i) as usize],
                ));
                let b = i32::from(correspondence_table[i as usize]);
                assert!((a - b).abs() <= 1, "i {i}: {a} vs {b}");
            }
        }
    }

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn probability_value_round_trip(p in MIN_PROBABILITY..=MAX_PROBABILITY) {
                let value = probability_to_value(p);
                prop_assert!((1..=32767).contains(&value));
                // One quantization step is (0.9 - 0.1) / 32766.
                let step = (MAX_PROBABILITY - MIN_PROBABILITY) / 32766.0;
                prop_assert!((value_to_probability(value) - p).abs() <= step);
                prop_assert_eq!(value_to_probability(value | UPDATE_MARKER),
                                value_to_probability(value));
                prop_assert_eq!(probability_to_value(value_to_probability(value)), value);
            }

            #[test]
            fn correspondence_cost_value_round_trip(c in MIN_CORRESPONDENCE_COST..=MAX_CORRESPONDENCE_COST) {
                let value = correspondence_cost_to_value(c);
                let step = (MAX_CORRESPONDENCE_COST - MIN_CORRESPONDENCE_COST) / 32766.0;
                prop_assert!((value_to_correspondence_cost(value) - c).abs() <= step);
                prop_assert_eq!(
                    correspondence_cost_to_value(value_to_correspondence_cost(value)), value);
            }

            #[test]
            fn odds_round_trip(p in 0.01_f32..0.99) {
                prop_assert!((probability_from_odds(odds(p)) - p).abs() <= 1e-6);
            }

            #[test]
            fn values_out_of_range_are_clamped(p in -10.0_f32..10.0) {
                let value = probability_to_value(p);
                prop_assert_eq!(value, probability_to_value(clamp_probability(p)));
                prop_assert!((1..=32767).contains(&value));
            }

            #[test]
            fn lookup_table_entries_carry_update_marker(p in 0.05_f32..0.95, cell in 0usize..32768) {
                let table = compute_lookup_table_to_apply_correspondence_cost_odds(odds(p));
                prop_assert_eq!(table.len(), 32768);
                prop_assert!(table[cell] > UPDATE_MARKER);
            }
        }
    }
}
