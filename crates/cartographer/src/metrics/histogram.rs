//! Port of `cartographer/metrics/histogram` (.h, .cc).

/// Upper bucket boundaries of a histogram (C++
/// `Histogram::BucketBoundaries`).
pub type BucketBoundaries = Vec<f64>;

/// A histogram metric. Implementations must be thread-safe.
pub trait Histogram: Send + Sync {
    /// Records an observation.
    fn observe(&self, value: f64);
}

/// Implementation of histogram that does nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullHistogram;

impl Histogram for NullHistogram {
    fn observe(&self, _value: f64) {}
}

static NULL_HISTOGRAM: NullHistogram = NullHistogram;

impl dyn Histogram {
    /// Histogram instance that does nothing. Safe for use in static
    /// initializers.
    pub fn null() -> &'static dyn Histogram {
        &NULL_HISTOGRAM
    }

    /// See [`fixed_width`].
    pub fn fixed_width(width: f64, num_finite_buckets: i32) -> BucketBoundaries {
        fixed_width(width, num_finite_buckets)
    }

    /// See [`scaled_powers_of`].
    pub fn scaled_powers_of(base: f64, scale_factor: f64, max_value: f64) -> BucketBoundaries {
        scaled_powers_of(base, scale_factor, max_value)
    }
}

/// `[width, 2 * width, ..., num_finite_buckets * width]`, computed by
/// repeated addition like the C++ code.
pub fn fixed_width(width: f64, num_finite_buckets: i32) -> BucketBoundaries {
    let mut result = BucketBoundaries::new();
    let mut boundary = 0.;
    for _ in 0..num_finite_buckets {
        boundary += width;
        result.push(boundary);
    }
    result
}

/// `[scale_factor, scale_factor * base, ...]` for all values `< max_value`.
pub fn scaled_powers_of(base: f64, scale_factor: f64, max_value: f64) -> BucketBoundaries {
    assert!(base > 1., "Check failed: base > 1 ({base} vs. 1)");
    assert!(
        scale_factor > 0.,
        "Check failed: scale_factor > 0 ({scale_factor} vs. 0)"
    );
    let mut result = BucketBoundaries::new();
    let mut boundary = scale_factor;
    while boundary < max_value {
        result.push(boundary);
        boundary *= base;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_boundaries() {
        assert_eq!(fixed_width(0.5, 4), vec![0.5, 1., 1.5, 2.]);
        assert_eq!(<dyn Histogram>::fixed_width(1., 0), Vec::<f64>::new());
        assert_eq!(scaled_powers_of(2., 1., 16.), vec![1., 2., 4., 8.]);
        assert_eq!(
            <dyn Histogram>::scaled_powers_of(10., 0.5, 50.1),
            vec![0.5, 5., 50.]
        );
        <dyn Histogram>::null().observe(1.);
    }

    #[test]
    #[should_panic(expected = "base > 1")]
    fn scaled_powers_of_requires_base_greater_one() {
        scaled_powers_of(1., 1., 10.);
    }
}
