//! Port of `cartographer/common/time` (.h, .cc).
//!
//! Represents Universal Time Scale durations and timestamps which are 64-bit
//! integers representing the 100 nanosecond ticks since the Epoch which is
//! January 1, 1 at the start of day in UTC.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Seconds from 0001-01-01 to 1970-01-01 (C++
/// `kUtsEpochOffsetFromUnixEpochInSeconds`).
pub const UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS: i64 = 719_162 * 24 * 60 * 60;

/// Number of ticks (100 ns) per second of the Universal Time Scale clock.
pub const TICKS_PER_SECOND: i64 = 10_000_000;

/// A signed duration in Universal Time Scale ticks of 100 ns
/// (C++ `common::Duration`, a `std::chrono::duration<int64, ratio<1, 10^7>>`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(i64);

impl Duration {
    /// The zero duration.
    pub const ZERO: Duration = Duration(0);

    /// Creates a duration from a number of 100 ns ticks.
    pub const fn from_ticks(ticks: i64) -> Duration {
        Duration(ticks)
    }

    /// Returns the number of 100 ns ticks (C++ `count()`).
    pub const fn ticks(self) -> i64 {
        self.0
    }

    /// Same as [`Duration::ticks`] (C++ `count()`).
    pub const fn count(self) -> i64 {
        self.0
    }

    /// The zero duration (C++ `Duration::zero()`).
    pub const fn zero() -> Duration {
        Duration(0)
    }

    /// Smallest representable duration (C++ `Duration::min()`).
    pub const fn min() -> Duration {
        Duration(i64::MIN)
    }

    /// Largest representable duration (C++ `Duration::max()`).
    pub const fn max() -> Duration {
        Duration(i64::MAX)
    }

    /// See [`from_seconds`].
    pub fn from_seconds(seconds: f64) -> Duration {
        from_seconds(seconds)
    }

    /// See [`from_milliseconds`].
    pub fn from_milliseconds(milliseconds: i64) -> Duration {
        from_milliseconds(milliseconds)
    }

    /// See [`to_seconds`].
    pub fn to_seconds(self) -> f64 {
        to_seconds(self)
    }

    /// Converts to a `std::time::Duration`; negative durations become zero.
    pub fn to_std(self) -> std::time::Duration {
        if self.0 <= 0 {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_nanos(self.0 as u64 * 100)
        }
    }

    /// Converts from a `std::time::Duration` (truncating to 100 ns ticks,
    /// saturating at `Duration::max()`).
    pub fn from_std(duration: std::time::Duration) -> Duration {
        Duration(i64::try_from(duration.as_nanos() / 100).unwrap_or(i64::MAX))
    }
}

/// A point in time on the Universal Time Scale (C++ `common::Time`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(Duration);

impl Time {
    /// Creates a time from the duration since the epoch.
    pub const fn from_time_since_epoch(duration: Duration) -> Time {
        Time(duration)
    }

    /// Returns the duration since the epoch (C++ `time_since_epoch()`).
    pub const fn time_since_epoch(self) -> Duration {
        self.0
    }

    /// Smallest representable time (C++ `Time::min()`).
    pub const fn min() -> Time {
        Time(Duration::min())
    }

    /// Largest representable time (C++ `Time::max()`).
    pub const fn max() -> Time {
        Time(Duration::max())
    }

    /// See [`from_universal`].
    pub const fn from_universal(ticks: i64) -> Time {
        from_universal(ticks)
    }

    /// See [`to_universal`].
    pub const fn to_universal(self) -> i64 {
        to_universal(self)
    }
}

/// Creates a duration from seconds. Like C++ `duration_cast` this truncates
/// toward zero.
pub fn from_seconds(seconds: f64) -> Duration {
    // RUST PORT NOTE: `as` saturates (and maps NaN to 0) where C++ is UB.
    Duration((seconds * TICKS_PER_SECOND as f64) as i64)
}

/// Creates a duration from milliseconds.
pub fn from_milliseconds(milliseconds: i64) -> Duration {
    Duration(milliseconds * 10_000)
}

/// Returns the given duration in seconds.
pub fn to_seconds(duration: Duration) -> f64 {
    duration.0 as f64 / TICKS_PER_SECOND as f64
}

/// Returns the given steady clock duration in seconds (C++
/// `ToSeconds(std::chrono::steady_clock::duration)`).
pub fn to_seconds_std(duration: std::time::Duration) -> f64 {
    duration.as_nanos() as f64 / 1e9
}

/// Creates a time from a Universal Time Scale.
pub const fn from_universal(ticks: i64) -> Time {
    Time(Duration(ticks))
}

/// Outputs the Universal Time Scale timestamp for a given time.
pub const fn to_universal(time: Time) -> i64 {
    time.0.0
}

/// CPU time consumed by the thread so far, in seconds.
pub fn get_thread_cpu_time_seconds() -> f64 {
    // RUST PORT NOTE: uses the `cpu-time` crate (CLOCK_THREAD_CPUTIME_ID on
    // Unix) instead of calling `clock_gettime` directly, to avoid `unsafe`.
    match cpu_time::ThreadTime::try_now() {
        Ok(t) => {
            let d = t.as_duration();
            d.as_secs() as f64 + 1e-9 * f64::from(d.subsec_nanos())
        }
        Err(e) => panic!("{e}"),
    }
}

/// For logging and unit tests, outputs the timestamp integer.
impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", to_universal(*self))
    }
}

impl Add for Duration {
    type Output = Duration;
    fn add(self, rhs: Duration) -> Duration {
        Duration(self.0 + rhs.0)
    }
}

impl Sub for Duration {
    type Output = Duration;
    fn sub(self, rhs: Duration) -> Duration {
        Duration(self.0 - rhs.0)
    }
}

impl AddAssign for Duration {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Duration {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

impl Neg for Duration {
    type Output = Duration;
    fn neg(self) -> Duration {
        Duration(-self.0)
    }
}

impl Mul<i64> for Duration {
    type Output = Duration;
    fn mul(self, rhs: i64) -> Duration {
        Duration(self.0 * rhs)
    }
}

impl Mul<Duration> for i64 {
    type Output = Duration;
    fn mul(self, rhs: Duration) -> Duration {
        Duration(self * rhs.0)
    }
}

impl Div<i64> for Duration {
    type Output = Duration;
    fn div(self, rhs: i64) -> Duration {
        Duration(self.0 / rhs)
    }
}

impl Div<Duration> for Duration {
    type Output = i64;
    fn div(self, rhs: Duration) -> i64 {
        self.0 / rhs.0
    }
}

impl Add<Duration> for Time {
    type Output = Time;
    fn add(self, rhs: Duration) -> Time {
        Time(self.0 + rhs)
    }
}

impl Add<Time> for Duration {
    type Output = Time;
    fn add(self, rhs: Time) -> Time {
        Time(self + rhs.0)
    }
}

impl Sub<Duration> for Time {
    type Output = Time;
    fn sub(self, rhs: Duration) -> Time {
        Time(self.0 - rhs)
    }
}

impl Sub<Time> for Time {
    type Output = Duration;
    fn sub(self, rhs: Time) -> Duration {
        self.0 - rhs.0
    }
}

impl AddAssign<Duration> for Time {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs;
    }
}

impl SubAssign<Duration> for Time {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn constants_and_conversions() {
        assert_eq!(UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS, 62_135_596_800);
        assert_eq!(from_seconds(1.5).ticks(), 15_000_000);
        // Truncation toward zero like duration_cast.
        assert_eq!(from_seconds(1.234_567_89e-7).ticks(), 1);
        assert_eq!(from_seconds(-1.234_567_89e-7).ticks(), -1);
        assert_eq!(from_milliseconds(150).ticks(), 1_500_000);
        assert_eq!(to_seconds(Duration::from_ticks(5)), 5.0 / 1e7);
        assert_eq!(to_universal(from_universal(42)), 42);
        assert_eq!(format!("{}", from_universal(-123)), "-123");
        assert_eq!(Time::min().to_universal(), i64::MIN);
        assert_eq!(Time::max().to_universal(), i64::MAX);
        let t = from_universal(100);
        assert_eq!(t + from_seconds(1.0) - t, from_seconds(1.0));
        let mut t2 = t;
        t2 += Duration::from_ticks(5);
        t2 -= Duration::from_ticks(2);
        assert_eq!(to_universal(t2), 103);
        assert!(from_universal(1) < from_universal(2));
        assert_eq!(
            from_milliseconds(150).to_std(),
            std::time::Duration::from_millis(150)
        );
        assert_eq!(
            Duration::from_std(std::time::Duration::from_millis(150)),
            from_milliseconds(150)
        );
        assert_eq!(to_seconds_std(std::time::Duration::from_millis(1500)), 1.5);
    }

    #[test]
    fn thread_cpu_time() {
        let t = get_thread_cpu_time_seconds();
        assert!(t >= 0.0);
    }

    proptest! {
        #[test]
        fn universal_round_trip(ticks in any::<i64>()) {
            prop_assert_eq!(to_universal(from_universal(ticks)), ticks);
        }

        #[test]
        fn milliseconds_to_seconds(ms in -1_000_000_000i64..1_000_000_000) {
            let s = to_seconds(from_milliseconds(ms));
            prop_assert!((s - ms as f64 / 1000.).abs() <= 1e-9 * (1. + s.abs()));
        }

        #[test]
        fn seconds_round_trip_within_a_tick(s in -1e9f64..1e9) {
            let d = from_seconds(s);
            prop_assert!((to_seconds(d) - s).abs() < 1e-7 + 1e-15 * s.abs());
            // Truncation toward zero.
            prop_assert!(to_seconds(d).abs() <= s.abs() + 1e-15 * s.abs());
        }

        #[test]
        fn time_arithmetic(a in -1i64 << 40..1i64 << 40, b in -1i64 << 40..1i64 << 40) {
            let t = from_universal(a);
            let d = Duration::from_ticks(b);
            prop_assert_eq!((t + d) - t, d);
            prop_assert_eq!((t + d) - d, t);
            prop_assert_eq!(-(-d), d);
        }
    }
}
