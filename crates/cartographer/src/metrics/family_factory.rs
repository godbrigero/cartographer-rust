//! Port of `cartographer/metrics/family_factory` (.h).

use std::collections::BTreeMap;
use std::marker::PhantomData;

use super::counter::Counter;
use super::gauge::Gauge;
use super::histogram::{BucketBoundaries, Histogram};

/// Labels of a metric instance (C++ `std::map<std::string, std::string>`).
pub type Labels = BTreeMap<String, String>;

/// A family of metrics of type `M` (`dyn Counter`, `dyn Gauge` or
/// `dyn Histogram`) distinguished by labels.
pub trait Family<M: ?Sized + 'static>: Send + Sync {
    /// Returns the metric for the given labels.
    fn add(&self, labels: &Labels) -> &'static M;
}

/// Metric types that have a null implementation (C++ `MetricType::Null()`).
pub trait NullMetric: 'static {
    /// The metric instance that does nothing.
    fn null_metric() -> &'static Self;
    /// The family instance that does nothing (C++ `Family<M>::Null()`).
    fn null_family() -> &'static dyn Family<Self>;
}

/// Family whose `add` always returns the null metric.
pub struct NullFamily<M: ?Sized>(PhantomData<fn(&M)>);

impl<M: ?Sized> std::fmt::Debug for NullFamily<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NullFamily")
    }
}

impl<M: ?Sized> NullFamily<M> {
    /// Creates a null family.
    pub const fn new() -> Self {
        NullFamily(PhantomData)
    }
}

impl<M: ?Sized> Default for NullFamily<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: ?Sized + NullMetric> Family<M> for NullFamily<M> {
    fn add(&self, _labels: &Labels) -> &'static M {
        M::null_metric()
    }
}

static NULL_COUNTER_FAMILY: NullFamily<dyn Counter> = NullFamily::new();
static NULL_GAUGE_FAMILY: NullFamily<dyn Gauge> = NullFamily::new();
static NULL_HISTOGRAM_FAMILY: NullFamily<dyn Histogram> = NullFamily::new();

impl NullMetric for dyn Counter {
    fn null_metric() -> &'static Self {
        <dyn Counter>::null()
    }
    fn null_family() -> &'static dyn Family<Self> {
        &NULL_COUNTER_FAMILY
    }
}

impl NullMetric for dyn Gauge {
    fn null_metric() -> &'static Self {
        <dyn Gauge>::null()
    }
    fn null_family() -> &'static dyn Family<Self> {
        &NULL_GAUGE_FAMILY
    }
}

impl NullMetric for dyn Histogram {
    fn null_metric() -> &'static Self {
        <dyn Histogram>::null()
    }
    fn null_family() -> &'static dyn Family<Self> {
        &NULL_HISTOGRAM_FAMILY
    }
}

/// Creates metric families; implemented by a metrics backend (e.g. a
/// Prometheus bridge).
pub trait FamilyFactory {
    /// Creates a counter family.
    fn new_counter_family(
        &mut self,
        name: &str,
        description: &str,
    ) -> &'static dyn Family<dyn Counter>;
    /// Creates a gauge family.
    fn new_gauge_family(&mut self, name: &str, description: &str)
    -> &'static dyn Family<dyn Gauge>;
    /// Creates a histogram family.
    fn new_histogram_family(
        &mut self,
        name: &str,
        description: &str,
        boundaries: &BucketBoundaries,
    ) -> &'static dyn Family<dyn Histogram>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_families_return_null_metrics() {
        let labels: Labels = [("kind".to_string(), "x".to_string())]
            .into_iter()
            .collect();
        <dyn Counter as NullMetric>::null_family()
            .add(&labels)
            .increment();
        <dyn Gauge as NullMetric>::null_family()
            .add(&labels)
            .set(1.);
        <dyn Histogram as NullMetric>::null_family()
            .add(&labels)
            .observe(1.);
    }
}
