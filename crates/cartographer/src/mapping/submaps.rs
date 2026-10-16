//! Port of `cartographer/mapping/submaps` (.h).
//!
//! RUST PORT NOTE: only the free log-odds helpers are ported so far (needed
//! by `ProbabilityGrid::draw_to_submap_texture`); the abstract `Submap` class
//! belongs to the submaps port.

use std::sync::LazyLock;

use crate::common::port::round_to_int;
use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY};

/// Converts the given probability to log odds.
#[inline]
pub fn logit(probability: f32) -> f32 {
    (probability / (1.0 - probability)).ln()
}

/// `Logit(kMaxProbability)` (C++ `kMaxLogOdds`, dynamically initialized).
pub static MAX_LOG_ODDS: LazyLock<f32> = LazyLock::new(|| logit(MAX_PROBABILITY));
/// `Logit(kMinProbability)` (C++ `kMinLogOdds`, dynamically initialized).
pub static MIN_LOG_ODDS: LazyLock<f32> = LazyLock::new(|| logit(MIN_PROBABILITY));

/// Converts a probability to a log odds integer. 0 means unknown,
/// [`MIN_LOG_ODDS`, `MAX_LOG_ODDS`] is mapped to [1, 255].
#[inline]
pub fn probability_to_log_odds_integer(probability: f32) -> u8 {
    let value = round_to_int(
        (logit(probability) - *MIN_LOG_ODDS) * 254.0_f32 / (*MAX_LOG_ODDS - *MIN_LOG_ODDS),
    ) + 1;
    assert!(1 <= value, "Check failed: 1 <= value ({value})");
    assert!(255 >= value, "Check failed: 255 >= value ({value})");
    value as u8
}

#[cfg(test)]
mod tests {
    mod submaps_test {
        use crate::mapping::submaps::*;

        // Converts the given log odds to a probability. This function is
        // known to be very slow, because expf is incredibly slow.
        fn expit(log_odds: f32) -> f32 {
            let exp_log_odds = log_odds.exp();
            exp_log_odds / (1.0 + exp_log_odds)
        }

        #[test]
        fn log_odds_conversions() {
            assert!((expit(logit(MIN_PROBABILITY)) - MIN_PROBABILITY).abs() <= 1e-6);
            assert!((expit(logit(MAX_PROBABILITY)) - MAX_PROBABILITY).abs() <= 1e-6);
            assert!((expit(logit(0.5)) - 0.5).abs() <= 1e-6);
        }

        #[test]
        fn log_odds_integer_bounds() {
            assert_eq!(probability_to_log_odds_integer(MIN_PROBABILITY), 1);
            assert_eq!(probability_to_log_odds_integer(MAX_PROBABILITY), 255);
            assert_eq!(probability_to_log_odds_integer(0.5), 128);
        }
    }
}
