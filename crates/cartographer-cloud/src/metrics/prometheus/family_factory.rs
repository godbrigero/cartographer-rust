//! Port of `cartographer/cloud/metrics/prometheus/family_factory` (.h, .cc):
//! a `cartographer::metrics::FamilyFactory` backed by the pure-Rust
//! `prometheus-client` crate (instead of prometheus-cpp).
//!
//! RUST PORT NOTE: Cartographer's metric API hands out `&'static` families
//! and metrics (C++ raw pointers owned by the factory). Families and the
//! per-label-set metric wrappers are therefore leaked; they are created once
//! at registration time (like the C++ statics), so the leak is bounded.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cartographer::metrics::counter::Counter;
use cartographer::metrics::family_factory::{self, Family, Labels};
use cartographer::metrics::gauge::Gauge;
use cartographer::metrics::histogram::{BucketBoundaries, Histogram};
use prometheus_client::metrics::counter::Counter as PrometheusCounter;
use prometheus_client::metrics::family::{Family as PrometheusFamily, MetricConstructor};
use prometheus_client::metrics::gauge::Gauge as PrometheusGauge;
use prometheus_client::metrics::histogram::Histogram as PrometheusHistogram;
use prometheus_client::registry::Registry;

/// Label set of a Prometheus metric (sorted by key, like `std::map`).
type LabelSet = Vec<(String, String)>;
type CounterMetric = PrometheusCounter<f64, AtomicU64>;
type GaugeMetric = PrometheusGauge<f64, AtomicU64>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn to_label_set(labels: &Labels) -> LabelSet {
    labels
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Creates or looks up already existing wrappers (C++ `GetOrCreateWrapper`).
fn get_or_create_wrapper<W: 'static>(
    wrappers: &Mutex<HashMap<LabelSet, &'static W>>,
    labels: &Labels,
    create: impl FnOnce(&LabelSet) -> W,
) -> &'static W {
    let label_set = to_label_set(labels);
    let mut wrappers = lock(wrappers);
    if let Some(wrapper) = wrappers.get(&label_set) {
        return wrapper;
    }
    let wrapper: &'static W = Box::leak(Box::new(create(&label_set)));
    wrappers.insert(label_set, wrapper);
    wrapper
}

struct CounterWrapper(CounterMetric);

impl Counter for CounterWrapper {
    fn increment(&self) {
        self.0.inc();
    }
    fn increment_by(&self, by_value: f64) {
        self.0.inc_by(by_value);
    }
}

struct CounterFamily {
    prometheus: PrometheusFamily<LabelSet, CounterMetric>,
    wrappers: Mutex<HashMap<LabelSet, &'static CounterWrapper>>,
}

impl Family<dyn Counter> for CounterFamily {
    fn add(&self, labels: &Labels) -> &'static dyn Counter {
        get_or_create_wrapper::<CounterWrapper>(&self.wrappers, labels, |label_set| {
            CounterWrapper(self.prometheus.get_or_create_owned(label_set))
        })
    }
}

struct GaugeWrapper(GaugeMetric);

impl Gauge for GaugeWrapper {
    fn increment(&self) {
        self.0.inc();
    }
    fn increment_by(&self, by_value: f64) {
        self.0.inc_by(by_value);
    }
    fn decrement(&self) {
        self.0.dec();
    }
    fn decrement_by(&self, by_value: f64) {
        self.0.dec_by(by_value);
    }
    fn set(&self, value: f64) {
        self.0.set(value);
    }
}

struct GaugeFamily {
    prometheus: PrometheusFamily<LabelSet, GaugeMetric>,
    wrappers: Mutex<HashMap<LabelSet, &'static GaugeWrapper>>,
}

impl Family<dyn Gauge> for GaugeFamily {
    fn add(&self, labels: &Labels) -> &'static dyn Gauge {
        get_or_create_wrapper::<GaugeWrapper>(&self.wrappers, labels, |label_set| {
            GaugeWrapper(self.prometheus.get_or_create_owned(label_set))
        })
    }
}

/// Creates histograms with the family's bucket boundaries.
#[derive(Clone, Debug)]
struct HistogramConstructor {
    boundaries: BucketBoundaries,
}

impl MetricConstructor<PrometheusHistogram> for HistogramConstructor {
    fn new_metric(&self) -> PrometheusHistogram {
        PrometheusHistogram::new(self.boundaries.iter().copied())
    }
}

struct HistogramWrapper(PrometheusHistogram);

impl Histogram for HistogramWrapper {
    fn observe(&self, value: f64) {
        self.0.observe(value);
    }
}

struct HistogramFamily {
    prometheus: PrometheusFamily<LabelSet, PrometheusHistogram, HistogramConstructor>,
    wrappers: Mutex<HashMap<LabelSet, &'static HistogramWrapper>>,
}

impl Family<dyn Histogram> for HistogramFamily {
    fn add(&self, labels: &Labels) -> &'static dyn Histogram {
        get_or_create_wrapper::<HistogramWrapper>(&self.wrappers, labels, |label_set| {
            HistogramWrapper(self.prometheus.get_or_create_owned(label_set))
        })
    }
}

/// A `FamilyFactory` registering all families in a Prometheus registry.
pub struct FamilyFactory {
    registry: Arc<Mutex<Registry>>,
}

impl Default for FamilyFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FamilyFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FamilyFactory").finish_non_exhaustive()
    }
}

impl FamilyFactory {
    /// Creates a factory with an empty registry.
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
        }
    }

    /// The registry (C++ `GetCollectable()`), e.g. for an
    /// [`Exposer`](super::exposer::Exposer).
    pub fn registry(&self) -> Arc<Mutex<Registry>> {
        Arc::clone(&self.registry)
    }

    /// Collects all metrics in the Prometheus text exposition format.
    pub fn collect(&self) -> String {
        encode_registry(&self.registry)
    }
}

/// Encodes `registry` in the Prometheus text exposition format.
pub fn encode_registry(registry: &Mutex<Registry>) -> String {
    let mut text = String::new();
    prometheus_client::encoding::text::encode(&mut text, &lock(registry))
        .expect("writing to a String cannot fail");
    text
}

impl family_factory::FamilyFactory for FamilyFactory {
    fn new_counter_family(
        &mut self,
        name: &str,
        description: &str,
    ) -> &'static dyn Family<dyn Counter> {
        let prometheus = PrometheusFamily::<LabelSet, CounterMetric>::default();
        lock(&self.registry).register(name, description, prometheus.clone());
        Box::leak(Box::new(CounterFamily {
            prometheus,
            wrappers: Mutex::new(HashMap::new()),
        }))
    }

    fn new_gauge_family(
        &mut self,
        name: &str,
        description: &str,
    ) -> &'static dyn Family<dyn Gauge> {
        let prometheus = PrometheusFamily::<LabelSet, GaugeMetric>::default();
        lock(&self.registry).register(name, description, prometheus.clone());
        Box::leak(Box::new(GaugeFamily {
            prometheus,
            wrappers: Mutex::new(HashMap::new()),
        }))
    }

    fn new_histogram_family(
        &mut self,
        name: &str,
        description: &str,
        boundaries: &BucketBoundaries,
    ) -> &'static dyn Family<dyn Histogram> {
        let prometheus = PrometheusFamily::new_with_constructor(HistogramConstructor {
            boundaries: boundaries.clone(),
        });
        lock(&self.registry).register(name, description, prometheus.clone());
        Box::leak(Box::new(HistogramFamily {
            prometheus,
            wrappers: Mutex::new(HashMap::new()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartographer::metrics::family_factory::FamilyFactory as _;
    use cartographer::metrics::histogram::fixed_width;

    const LABEL_KEY: &str = "kind";
    const LABEL_VALUE: &str = "score";
    const OBSERVE_SCORES: [f64; 5] = [-1., 0.11, 0.2, 0.5, 2.];

    fn labels() -> Labels {
        [(LABEL_KEY.to_string(), LABEL_VALUE.to_string())]
            .into_iter()
            .collect()
    }

    /// Returns the sample lines of `text` (no comments).
    fn samples(text: &str) -> Vec<&str> {
        text.lines().filter(|line| !line.starts_with('#')).collect()
    }

    /// Returns the value of the sample line starting with `prefix`.
    fn sample_value(text: &str, prefix: &str) -> f64 {
        samples(text)
            .into_iter()
            .find(|line| line.starts_with(prefix))
            .and_then(|line| line.rsplit(' ').next())
            .unwrap_or_else(|| panic!("no sample {prefix} in:\n{text}"))
            .parse()
            .unwrap()
    }

    /// The `Algorithm` of the C++ test.
    fn register_algorithm_metrics(factory: &mut FamilyFactory) -> &'static dyn Histogram {
        let boundaries = fixed_width(0.05, 20);
        let scores_family =
            factory.new_histogram_family("/algorithm/scores", "Scores achieved", &boundaries);
        scores_family.add(&labels())
    }

    mod metrics_test {
        use super::*;

        #[test]
        fn collect_counter() {
            let mut factory = FamilyFactory::new();
            let counter_family = factory.new_counter_family("/test/hits", "Hits");
            let counter = counter_family.add(&labels());
            counter.increment();
            counter.increment_by(5.);
            let expected_value = 1. + 5.;
            let collected = factory.collect();
            let samples = samples(&collected);
            assert_eq!(samples.len(), 1, "{collected}");
            assert!(
                samples[0].starts_with(&format!(
                    "/test/hits_total{{{LABEL_KEY}=\"{LABEL_VALUE}\"}}"
                )),
                "{collected}"
            );
            assert_eq!(sample_value(&collected, "/test/hits_total"), expected_value);
            // The same labels return the same metric.
            assert!(std::ptr::addr_eq(counter, counter_family.add(&labels())));
        }

        #[test]
        fn collect_gauge() {
            let mut factory = FamilyFactory::new();
            let gauge_family =
                factory.new_gauge_family("/test/queue/length", "Length of some queue");
            let gauge = gauge_family.add(&labels());
            gauge.increment();
            gauge.increment_by(5.);
            gauge.decrement();
            gauge.decrement_by(2.);
            let expected_value = 1. + 5. - 1. - 2.;
            let collected = factory.collect();
            assert!(
                samples(&collected)[0].starts_with(&format!(
                    "/test/queue/length{{{LABEL_KEY}=\"{LABEL_VALUE}\"}}"
                )),
                "{collected}"
            );
            assert_eq!(
                sample_value(&collected, "/test/queue/length"),
                expected_value
            );
            gauge.set(42.);
            assert_eq!(sample_value(&factory.collect(), "/test/queue/length"), 42.);
        }

        #[test]
        fn collect_histogram() {
            let mut factory = FamilyFactory::new();
            let scores = register_algorithm_metrics(&mut factory);
            for score in OBSERVE_SCORES {
                scores.observe(score);
            }
            let collected = factory.collect();
            assert_eq!(
                sample_value(&collected, "/algorithm/scores_count"),
                OBSERVE_SCORES.len() as f64
            );
            // First bucket (le=0.05) holds only the score -1.
            assert_eq!(
                sample_value(&collected, "/algorithm/scores_bucket{le=\"0.05\""),
                1.
            );
        }
    }
}
