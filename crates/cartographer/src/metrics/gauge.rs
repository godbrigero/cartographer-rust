//! Port of `cartographer/metrics/gauge` (.h, .cc).

/// A value that can go up and down. Implementations must be thread-safe.
pub trait Gauge: Send + Sync {
    /// Increments by one (C++ `Increment()`).
    fn increment(&self);
    /// Increments by `by_value` (C++ `Increment(double)`).
    fn increment_by(&self, by_value: f64);
    /// Decrements by one (C++ `Decrement()`).
    fn decrement(&self);
    /// Decrements by `by_value` (C++ `Decrement(double)`).
    fn decrement_by(&self, by_value: f64);
    /// Sets the value.
    fn set(&self, value: f64);
}

/// Implementation of gauge that does nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullGauge;

impl Gauge for NullGauge {
    fn increment(&self) {}
    fn increment_by(&self, _by_value: f64) {}
    fn decrement(&self) {}
    fn decrement_by(&self, _by_value: f64) {}
    fn set(&self, _value: f64) {}
}

static NULL_GAUGE: NullGauge = NullGauge;

impl dyn Gauge {
    /// Gauge instance that does nothing. Safe for use in static initializers.
    pub fn null() -> &'static dyn Gauge {
        &NULL_GAUGE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_gauge_is_a_no_op() {
        let gauge = <dyn Gauge>::null();
        gauge.increment();
        gauge.increment_by(2.);
        gauge.decrement();
        gauge.decrement_by(2.);
        gauge.set(5.);
    }
}
