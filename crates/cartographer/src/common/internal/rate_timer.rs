//! Port of `cartographer/common/internal/rate_timer` (.h).

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::LazyLock;
use std::time::Instant;

use crate::common::math::pow2;
use crate::common::printf::{format_e, format_f_precision};
use crate::common::time::{Duration, Time, to_seconds, to_seconds_std};

/// A monotonic wall clock (C++ `ClockType` template parameter with a static
/// `now()`).
///
/// RUST PORT NOTE: `now()` returns the time since an arbitrary, fixed origin
/// as a `std::time::Duration` (only differences are used).
pub trait Clock {
    /// Returns the current time since the clock's origin.
    fn now() -> std::time::Duration;
}

/// The default clock (C++ `std::chrono::steady_clock`).
#[derive(Debug)]
pub struct SteadyClock;

static STEADY_CLOCK_ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);

impl Clock for SteadyClock {
    fn now() -> std::time::Duration {
        STEADY_CLOCK_ORIGIN.elapsed()
    }
}

#[derive(Clone, Copy, Debug)]
struct Event {
    time: Time,
    wall_time: std::time::Duration,
}

/// Computes the rate at which pulses come in.
#[derive(Debug)]
pub struct RateTimer<C: Clock = SteadyClock> {
    events: VecDeque<Event>,
    window_duration: Duration,
    clock: PhantomData<C>,
}

impl<C: Clock> RateTimer<C> {
    /// Computes the rate at which pulses come in over `window_duration` in
    /// wall time.
    pub fn new(window_duration: Duration) -> Self {
        RateTimer {
            events: VecDeque::new(),
            window_duration,
            clock: PhantomData,
        }
    }

    /// Returns the pulse rate in Hz.
    pub fn compute_rate(&self) -> f64 {
        let (Some(front), Some(back)) = (self.events.front(), self.events.back()) else {
            return 0.;
        };
        (self.events.len() - 1) as f64 / to_seconds(back.time - front.time)
    }

    /// Returns the ratio of the pulse rate (with supplied times) to the wall
    /// time rate. For example, if a sensor produces pulses at 10 Hz, but we
    /// call Pulse at 20 Hz wall time, this will return 2.
    pub fn compute_wall_time_rate_ratio(&self) -> f64 {
        let (Some(front), Some(back)) = (self.events.front(), self.events.back()) else {
            return 0.;
        };
        to_seconds(back.time - front.time) / wall_seconds(back.wall_time, front.wall_time)
    }

    /// Records an event that will contribute to the computed rate.
    pub fn pulse(&mut self, time: Time) {
        self.events.push_back(Event {
            time,
            wall_time: C::now(),
        });
        while self.events.len() > 2 {
            let span = wall_nanos(self.events.back().unwrap().wall_time)
                - wall_nanos(self.events.front().unwrap().wall_time);
            if span > i128::from(self.window_duration.ticks()) * 100 {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Returns a debug string representation.
    pub fn debug_string(&self) -> String {
        if self.events.len() < 2 {
            return "unknown".to_string();
        }
        format!(
            "{} Hz {} (pulsed at {}% real time)",
            format_f_precision(self.compute_rate(), 2),
            self.deltas_debug_string(),
            format_f_precision(self.compute_wall_time_rate_ratio() * 100., 2)
        )
    }

    /// Computes all differences in seconds between consecutive pulses.
    fn compute_deltas_in_seconds(&self) -> Vec<f64> {
        assert!(
            self.events.len() > 1,
            "Check failed: events_.size() > 1 ({} vs. 1)",
            self.events.len()
        );
        let count = self.events.len() - 1;
        (0..count)
            .map(|i| to_seconds(self.events[i + 1].time - self.events[i].time))
            .collect()
    }

    /// Returns the average and standard deviation of the deltas.
    fn deltas_debug_string(&self) -> String {
        let deltas = self.compute_deltas_in_seconds();
        let sum: f64 = deltas.iter().fold(0., |acc, x| acc + x);
        let mean = sum / deltas.len() as f64;
        let mut squared_sum = 0.;
        for &x in &deltas {
            squared_sum += pow2(x - mean);
        }
        let sigma = (squared_sum / (deltas.len() - 1) as f64).sqrt();
        format!("{} s +/- {} s", format_e(mean, 2), format_e(sigma, 2))
    }
}

fn wall_nanos(d: std::time::Duration) -> i128 {
    d.as_nanos() as i128
}

fn wall_seconds(back: std::time::Duration, front: std::time::Duration) -> f64 {
    if back >= front {
        to_seconds_std(back - front)
    } else {
        -to_seconds_std(front - back)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{from_seconds, from_universal};
    use std::sync::Mutex;

    static SIMULATED_TIME: Mutex<std::time::Duration> = Mutex::new(std::time::Duration::ZERO);

    struct SimulatedClock;

    impl Clock for SimulatedClock {
        fn now() -> std::time::Duration {
            *SIMULATED_TIME.lock().unwrap()
        }
    }

    mod rate_timer_test {
        use super::*;

        #[test]
        fn compute_rate() {
            let mut rate_timer: RateTimer = RateTimer::new(from_seconds(1.));
            let mut time = from_universal(42);
            for _ in 0..100 {
                rate_timer.pulse(time);
                time += from_seconds(0.1);
            }
            assert!((10. - rate_timer.compute_rate()).abs() <= 1e-3);
        }

        #[test]
        fn compute_wall_time_rate_ratio() {
            let mut time = from_universal(42);
            let mut rate_timer: RateTimer<SimulatedClock> = RateTimer::new(from_seconds(1.));
            for _ in 0..100 {
                rate_timer.pulse(time);
                time += from_seconds(0.1);
                *SIMULATED_TIME.lock().unwrap() += std::time::Duration::from_secs_f64(0.05);
            }
            assert!((2. - rate_timer.compute_wall_time_rate_ratio()).abs() <= 1e-3);
        }
    }

    #[test]
    fn debug_string() {
        let mut rate_timer: RateTimer = RateTimer::new(from_seconds(1.));
        assert_eq!(rate_timer.debug_string(), "unknown");
        assert_eq!(rate_timer.compute_rate(), 0.);
        let mut time = from_universal(0);
        for _ in 0..3 {
            rate_timer.pulse(time);
            time += from_seconds(0.1);
        }
        let s = rate_timer.debug_string();
        assert!(
            s.starts_with("10.00 Hz 1.00e-01 s +/- 0.00e+00 s (pulsed at "),
            "{s}"
        );
    }
}
