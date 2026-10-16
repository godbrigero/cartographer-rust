//! Port of `cartographer/mapping/value_conversion_tables` (.h, .cc).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const UPDATE_MARKER: u16 = 1 << 15;

/// 0 is unknown, [1, 32767] maps to [lower_bound, upper_bound].
fn slow_value_to_bounded_float(
    value: u16,
    unknown_value: u16,
    unknown_result: f32,
    lower_bound: f32,
    upper_bound: f32,
) -> f32 {
    assert!(value <= 32767, "Check failed: value <= 32767 ({value})");
    if value == unknown_value {
        return unknown_result;
    }
    let scale = (upper_bound - lower_bound) / 32766.0_f32;
    f32::from(value) * scale + (lower_bound - scale)
}

fn precompute_value_to_bounded_float(
    unknown_value: u16,
    unknown_result: f32,
    lower_bound: f32,
    upper_bound: f32,
) -> Arc<[f32]> {
    let num_values = usize::from(u16::MAX) + 1;
    (0..num_values)
        .map(|value| {
            slow_value_to_bounded_float(
                (value as u16) & !UPDATE_MARKER,
                unknown_value,
                unknown_result,
                lower_bound,
                upper_bound,
            )
        })
        .collect()
}

/// Performs lazy computations of lookup tables for mapping from a `u16` value
/// to a float in `[lower_bound, upper_bound]`. The first element of the table
/// is set to `unknown_result`. Tables have 65536 entries so that values with
/// the update marker set decode like the ones without.
///
/// RUST PORT NOTE: C++ hands out raw `const std::vector<float>*` that stay
/// valid as long as the (non thread-safe) `ValueConversionTables` lives, and
/// grids keep a raw `ValueConversionTables*` (owned by `ActiveSubmaps2D` /
/// the pose graph). Here tables are returned as shared `Arc<[f32]>` (so a grid
/// does not borrow the cache and can outlive it or move between threads), and
/// the cache uses an interior `Mutex` so `get_conversion_table` takes `&self`.
/// Grids hold an `Arc<ValueConversionTables>` (needed by
/// `compute_cropped_grid`). The cache is keyed on the `f32` bit patterns
/// (C++ `std::tuple<float>` ordering would treat `-0.0 == 0.0`; irrelevant in
/// practice).
#[derive(Debug, Default)]
pub struct ValueConversionTables {
    bounds_to_lookup_table: Mutex<HashMap<[u32; 3], Arc<[f32]>>>,
}

impl ValueConversionTables {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the (lazily computed, cached) table for the given bounds.
    /// Calling this twice with the same arguments returns the same table
    /// (`Arc::ptr_eq`).
    pub fn get_conversion_table(
        &self,
        unknown_result: f32,
        lower_bound: f32,
        upper_bound: f32,
    ) -> Arc<[f32]> {
        let bounds = [
            unknown_result.to_bits(),
            lower_bound.to_bits(),
            upper_bound.to_bits(),
        ];
        let mut tables = self
            .bounds_to_lookup_table
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(tables.entry(bounds).or_insert_with(|| {
            precompute_value_to_bounded_float(0, unknown_result, lower_bound, upper_bound)
        }))
    }
}

#[cfg(test)]
mod tests {
    mod value_conversion_tables_test {
        use std::sync::Arc;

        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        use crate::mapping::value_conversion_tables::ValueConversionTables;

        #[test]
        fn equal_tables() {
            let value_conversion_tables = ValueConversionTables::new();
            let reference_table = value_conversion_tables.get_conversion_table(0.1, 0.1, 0.5);
            let test_table = value_conversion_tables.get_conversion_table(0.1, 0.1, 0.5);
            assert!(Arc::ptr_eq(&reference_table, &test_table));
        }

        #[test]
        fn inequal_tables() {
            let value_conversion_tables = ValueConversionTables::new();
            let reference_table = value_conversion_tables.get_conversion_table(0.1, 0.1, 0.5);
            let test_table = value_conversion_tables.get_conversion_table(0.1, 0.4, 0.5);
            assert!(!Arc::ptr_eq(&reference_table, &test_table));
        }

        #[test]
        fn value_conversion() {
            let value_conversion_tables = ValueConversionTables::new();
            // C++ uses std::mt19937(42); exact samples are not required.
            let mut rng = StdRng::seed_from_u64(42);
            for _ in 0..100 {
                let bound_sample_0: f32 = rng.random_range(-10.0..10.0);
                let bound_sample_1: f32 = rng.random_range(-10.0..10.0);
                let lower_bound = bound_sample_0.min(bound_sample_1);
                let upper_bound = bound_sample_0.max(bound_sample_1);
                let undefined_value: f32 = rng.random_range(-10.0..10.0);
                let conversion_table = value_conversion_tables.get_conversion_table(
                    undefined_value,
                    lower_bound,
                    upper_bound,
                );
                assert_eq!(conversion_table[0], undefined_value);
                let scale = (upper_bound - lower_bound) / 32766.0_f32;
                for i in 1..32768_u16 {
                    assert_eq!(
                        conversion_table[i as usize],
                        f32::from(i) * scale + (lower_bound - scale)
                    );
                }
            }
        }

        #[test]
        fn update_marker_decodes_like_plain_value() {
            let value_conversion_tables = ValueConversionTables::new();
            let table = value_conversion_tables.get_conversion_table(-1.0, -1.0, 1.0);
            assert_eq!(table.len(), 65536);
            for i in 0..32768_usize {
                assert_eq!(table[i], table[i | 0x8000]);
            }
        }
    }
}
