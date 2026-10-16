//! Port of `cartographer/common/fixed_ratio_sampler` (.h, .cc).

use crate::common::printf::format_f;

/// Signals when a sample should be taken from a stream of data to select a
/// uniformly distributed fraction of the data.
#[derive(Debug)]
pub struct FixedRatioSampler {
    /// Sampling occurs if the proportion of samples to pulses drops below
    /// this number.
    ratio: f64,
    num_pulses: i64,
    num_samples: i64,
}

impl FixedRatioSampler {
    /// Creates a sampler. Panics unless `0 <= ratio <= 1`.
    pub fn new(ratio: f64) -> Self {
        assert!(ratio >= 0., "Check failed: ratio >= 0. ({ratio} vs. 0)");
        if ratio == 0. {
            log::warn!("FixedRatioSampler is dropping all data.");
        }
        assert!(ratio <= 1., "Check failed: ratio <= 1. ({ratio} vs. 1)");
        FixedRatioSampler {
            ratio,
            num_pulses: 0,
            num_samples: 0,
        }
    }

    /// Returns true if this pulse should result in an sample.
    pub fn pulse(&mut self) -> bool {
        self.num_pulses += 1;
        if (self.num_samples as f64) / (self.num_pulses as f64) < self.ratio {
            self.num_samples += 1;
            return true;
        }
        false
    }

    /// Returns a debug string describing the current ratio of samples to
    /// pulses.
    pub fn debug_string(&self) -> String {
        format!(
            "{} ({}%)",
            self.num_samples,
            format_f(100. * self.num_samples as f64 / self.num_pulses as f64)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fixed_ratio_sampler_test {
        use super::*;

        #[test]
        fn always_true() {
            let mut fixed_ratio_sampler = FixedRatioSampler::new(1.);
            for _ in 0..100 {
                assert!(fixed_ratio_sampler.pulse());
            }
        }

        #[test]
        fn always_false() {
            let mut fixed_ratio_sampler = FixedRatioSampler::new(0.);
            for _ in 0..100 {
                assert!(!fixed_ratio_sampler.pulse());
            }
        }

        #[test]
        #[should_panic(expected = "ratio")]
        fn non_sensical_ratio_too_large() {
            FixedRatioSampler::new(2.);
        }

        #[test]
        #[should_panic(expected = "ratio")]
        fn non_sensical_ratio_negative() {
            FixedRatioSampler::new(-0.1);
        }

        #[test]
        fn sometimes_true() {
            let mut fixed_ratio_sampler = FixedRatioSampler::new(0.5);
            for i in 0..100 {
                assert_eq!(i % 2 == 0, fixed_ratio_sampler.pulse());
            }
        }

        #[test]
        fn first_pulse_is_true() {
            // Choose a very very small positive number for the ratio.
            let mut fixed_ratio_sampler = FixedRatioSampler::new(1e-20);
            assert!(fixed_ratio_sampler.pulse());
            for _ in 0..100 {
                assert!(!fixed_ratio_sampler.pulse());
            }
        }
    }

    #[test]
    fn debug_string() {
        let mut sampler = FixedRatioSampler::new(0.5);
        sampler.pulse();
        sampler.pulse();
        assert_eq!(sampler.debug_string(), "1 (50.000000%)");
    }
}
