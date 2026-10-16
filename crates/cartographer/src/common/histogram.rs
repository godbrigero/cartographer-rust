//! Port of `cartographer/common/histogram` (.h, .cc).
//!
//! A text histogram for debug output (distinct from
//! [`crate::metrics::histogram`]).

use crate::common::printf::{format_absl, format_f};

/// Collects values and renders them as a text histogram.
#[derive(Clone, Debug, Default)]
pub struct Histogram {
    values: Vec<f32>,
}

impl Histogram {
    /// Creates an empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a value.
    pub fn add(&mut self, value: f32) {
        self.values.push(value);
    }

    /// Renders the histogram with `buckets` buckets. Panics if `buckets < 1`.
    pub fn to_string(&self, buckets: i32) -> String {
        assert!(buckets >= 1, "Check failed: buckets >= 1 ({buckets} vs. 1)");
        if self.values.is_empty() {
            return "Count: 0".to_string();
        }
        let n = self.values.len();
        let min = self
            .values
            .iter()
            .copied()
            .fold(self.values[0], |a, b| if b < a { b } else { a });
        let max = self
            .values
            .iter()
            .copied()
            .fold(self.values[0], |a, b| if a < b { b } else { a });
        let mean = self.values.iter().fold(0f32, |acc, &v| acc + v) / n as f32;
        let mut result = format!(
            "Count: {}  Min: {}  Max: {}  Mean: {}",
            n,
            format_absl(f64::from(min)),
            format_absl(f64::from(max)),
            format_absl(f64::from(mean))
        );
        if min == max {
            return result;
        }
        assert!(min < max, "Check failed: min < max ({min} vs. {max})");
        let mut lower_bound = min;
        let mut total_count = 0i32;
        for i in 0..buckets {
            let upper_bound = if i + 1 == buckets {
                max
            } else {
                max * (i + 1) as f32 / buckets as f32
                    + min * (buckets - i - 1) as f32 / buckets as f32
            };
            let mut count = 0i32;
            for &value in &self.values {
                if lower_bound <= value
                    && (if i + 1 == buckets {
                        value <= upper_bound
                    } else {
                        value < upper_bound
                    })
                {
                    count += 1;
                }
            }
            total_count += count;
            result.push_str(&format!(
                "\n[{}, {}{}",
                format_f(f64::from(lower_bound)),
                format_f(f64::from(upper_bound)),
                if i + 1 == buckets { ']' } else { ')' }
            ));
            const MAX_BAR_CHARS: i32 = 20;
            // Integer arithmetic on size_t in C++.
            let bar = ((count as usize * MAX_BAR_CHARS as usize + n / 2) / n) as i32;
            result.push('\t');
            for j in 0..MAX_BAR_CHARS {
                result.push(if j < MAX_BAR_CHARS - bar { ' ' } else { '#' });
            }
            result.push_str(&format!(
                "\tCount: {} ({}%)\tTotal: {} ({}%)",
                count,
                format_absl(f64::from(count as f32 * 1e2f32 / n as f32)),
                total_count,
                format_absl(f64::from(total_count as f32 * 1e2f32 / n as f32))
            ));
            lower_bound = upper_bound;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        assert_eq!(Histogram::new().to_string(3), "Count: 0");
    }

    #[test]
    fn single_value() {
        let mut h = Histogram::new();
        h.add(1.5);
        h.add(1.5);
        assert_eq!(h.to_string(2), "Count: 2  Min: 1.5  Max: 1.5  Mean: 1.5");
    }

    #[test]
    fn buckets() {
        let mut h = Histogram::new();
        for v in [0., 1., 2., 3.] {
            h.add(v);
        }
        let expected = "Count: 4  Min: 0  Max: 3  Mean: 1.5\n\
            [0.000000, 1.500000)\t          ##########\tCount: 2 (50%)\tTotal: 2 (50%)\n\
            [1.500000, 3.000000]\t          ##########\tCount: 2 (50%)\tTotal: 4 (100%)";
        assert_eq!(h.to_string(2), expected);
    }

    #[test]
    #[should_panic(expected = "buckets >= 1")]
    fn zero_buckets() {
        Histogram::new().to_string(0);
    }
}
