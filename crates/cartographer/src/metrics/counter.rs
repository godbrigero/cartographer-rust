//! Port of `cartographer/metrics/counter` (.h, .cc).
//!
//! RUST PORT NOTE: C++ hands out raw pointers owned by a registry that
//! outlives all users; in Rust metrics are `&'static dyn Counter` (a registry
//! implementation can obtain those via `Box::leak`).

/// A monotonically increasing counter. Implementations must be thread-safe.
pub trait Counter: Send + Sync {
    /// Increments by one (C++ `Increment()`).
    fn increment(&self);
    /// Increments by `by_value` (C++ `Increment(double)`).
    fn increment_by(&self, by_value: f64);
}

/// Implementation of counter that does nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullCounter;

impl Counter for NullCounter {
    fn increment(&self) {}
    fn increment_by(&self, _by_value: f64) {}
}

static NULL_COUNTER: NullCounter = NullCounter;

impl dyn Counter {
    /// Counter instance that does nothing. Safe for use in static
    /// initializers.
    pub fn null() -> &'static dyn Counter {
        &NULL_COUNTER
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_counter_is_a_no_op() {
        let counter = <dyn Counter>::null();
        counter.increment();
        counter.increment_by(3.);
    }
}
