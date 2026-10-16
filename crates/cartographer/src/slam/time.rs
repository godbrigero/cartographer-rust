//! [`Timestamp`]: sensor time as seen by the high-level API.

use std::fmt;

use crate::common::time::{TICKS_PER_SECOND, Time, UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS};

const NANOS_PER_SECOND: i64 = 1_000_000_000;
const NANOS_PER_TICK: i64 = NANOS_PER_SECOND / TICKS_PER_SECOND;

/// A point in time, stored as integer nanoseconds since an epoch of your
/// choice.
///
/// Any epoch works as long as **all sensors of a [`Slam`](super::Slam) use
/// the same one**: UNIX time (as in ROS bags), seconds since boot, or simply
/// `0.0` at the start of a recording. Only differences between timestamps
/// matter to SLAM; absolute values only show up in saved `.pbstream` files.
///
/// Conversions:
///
/// * from `f64` seconds: [`Timestamp::from_secs_f64`] or `Timestamp::from(1.5)`
///   (rounded to the nearest nanosecond);
/// * from [`std::time::Duration`] since the epoch: `Timestamp::from(duration)`;
/// * from integer nanoseconds: [`Timestamp::from_nanos`] (e.g. a ROS
///   `sec * 1e9 + nsec` stamp, or a `SystemTime` via
///   `duration_since(UNIX_EPOCH)`);
/// * to Cartographer's [`common::time::Time`](crate::common::time::Time):
///   [`Timestamp::to_cartographer_time`], which interprets the epoch as the
///   UNIX epoch exactly like cartographer_ros' `FromRos` (100 ns ticks since
///   0001-01-01, rounded to the nearest tick).
///
/// Every API taking a time accepts `impl Into<Timestamp>`, so plain `f64`
/// seconds can be passed directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    nanos: i64,
}

impl Timestamp {
    /// The epoch itself (`0.0` s).
    pub const ZERO: Timestamp = Timestamp { nanos: 0 };

    /// `nanos` nanoseconds after the epoch.
    pub const fn from_nanos(nanos: i64) -> Self {
        Timestamp { nanos }
    }

    /// `seconds` after the epoch, rounded to the nearest nanosecond.
    ///
    /// Panics if `seconds` is not finite or out of the `i64` nanosecond range
    /// (about ±292 years).
    pub fn from_secs_f64(seconds: f64) -> Self {
        assert!(
            seconds.is_finite() && (seconds * 1e9).abs() < 9.2e18,
            "Timestamp::from_secs_f64: {seconds} is not a representable time"
        );
        Timestamp {
            nanos: (seconds * 1e9).round() as i64,
        }
    }

    /// Nanoseconds since the epoch.
    pub const fn as_nanos(self) -> i64 {
        self.nanos
    }

    /// Seconds since the epoch.
    pub fn as_secs_f64(self) -> f64 {
        // Split to keep full precision for large (UNIX-time) values.
        let seconds = self.nanos.div_euclid(NANOS_PER_SECOND);
        let nanos = self.nanos.rem_euclid(NANOS_PER_SECOND);
        seconds as f64 + nanos as f64 * 1e-9
    }

    /// Converts to Cartographer's time (100 ns ticks since 0001-01-01),
    /// treating the epoch as the UNIX epoch and rounding to the nearest tick
    /// like cartographer_ros' `FromRos`.
    pub fn to_cartographer_time(self) -> Time {
        let ticks = (self.nanos + NANOS_PER_TICK / 2).div_euclid(NANOS_PER_TICK);
        Time::from_universal(UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * TICKS_PER_SECOND + ticks)
    }

    /// Inverse of [`to_cartographer_time`](Self::to_cartographer_time)
    /// (exact within the representable range of about ±292 years around the
    /// UNIX epoch; saturates outside it).
    pub fn from_cartographer_time(time: Time) -> Self {
        let ticks =
            time.to_universal() - UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * TICKS_PER_SECOND;
        Timestamp {
            nanos: ticks.saturating_mul(NANOS_PER_TICK),
        }
    }

    /// `self - earlier` in seconds.
    pub fn seconds_since(self, earlier: Timestamp) -> f64 {
        (self.nanos - earlier.nanos) as f64 * 1e-9
    }

    /// `self + seconds` (rounded to the nearest nanosecond).
    pub fn add_secs_f64(self, seconds: f64) -> Timestamp {
        Timestamp {
            nanos: self.nanos + (seconds * 1e9).round() as i64,
        }
    }
}

impl From<f64> for Timestamp {
    /// Seconds since the epoch; see [`Timestamp::from_secs_f64`].
    fn from(seconds: f64) -> Self {
        Timestamp::from_secs_f64(seconds)
    }
}

impl From<std::time::Duration> for Timestamp {
    /// Time elapsed since the epoch.
    fn from(duration: std::time::Duration) -> Self {
        Timestamp {
            nanos: i64::try_from(duration.as_nanos()).expect("Duration too large for a Timestamp"),
        }
    }
}

impl From<Time> for Timestamp {
    fn from(time: Time) -> Self {
        Timestamp::from_cartographer_time(time)
    }
}

impl From<Timestamp> for Time {
    fn from(timestamp: Timestamp) -> Self {
        timestamp.to_cartographer_time()
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.nanos < 0 { "-" } else { "" };
        let abs = self.nanos.unsigned_abs();
        write!(
            f,
            "{sign}{}.{:09}s",
            abs / NANOS_PER_SECOND as u64,
            abs % NANOS_PER_SECOND as u64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::from_seconds;

    #[test]
    fn seconds_round_trip() {
        for seconds in [0.0, 1.5, -2.25, 1_700_000_000.123_456_7] {
            let t = Timestamp::from_secs_f64(seconds);
            assert!((t.as_secs_f64() - seconds).abs() < 1e-6, "{seconds}");
        }
        assert_eq!(Timestamp::from(0.5).as_nanos(), 500_000_000);
        assert_eq!(Timestamp::from_secs_f64(1e-9).as_nanos(), 1);
    }

    #[test]
    fn duration_conversion() {
        let t = Timestamp::from(std::time::Duration::new(3, 250));
        assert_eq!(t.as_nanos(), 3_000_000_250);
    }

    #[test]
    fn cartographer_time_uses_unix_epoch_like_cartographer_ros() {
        // cartographer_ros: FromUniversal((sec + offset) * 1e7 + (nsec + 50) / 100).
        let t = Timestamp::from_nanos(1_600_000_000 * NANOS_PER_SECOND + 149);
        assert_eq!(
            t.to_cartographer_time().to_universal(),
            (1_600_000_000 + UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS) * 10_000_000 + 1
        );
        let t = Timestamp::from_nanos(150);
        assert_eq!(
            t.to_cartographer_time().to_universal(),
            UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * 10_000_000 + 2
        );
        assert_eq!(
            Timestamp::ZERO.to_cartographer_time(),
            Time::from_universal(UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * 10_000_000)
        );
    }

    #[test]
    fn cartographer_time_round_trip_and_differences() {
        let a = Timestamp::from_secs_f64(10.0);
        let b = Timestamp::from_secs_f64(10.25);
        assert_eq!(
            b.to_cartographer_time() - a.to_cartographer_time(),
            from_seconds(0.25)
        );
        let unix_epoch = UTS_EPOCH_OFFSET_FROM_UNIX_EPOCH_IN_SECONDS * TICKS_PER_SECOND;
        for ticks in [
            unix_epoch,
            unix_epoch + 1,
            unix_epoch - 123_456_789,
            637_000_000_000_000_000,
        ] {
            let time = Time::from_universal(ticks);
            assert_eq!(Timestamp::from(time).to_cartographer_time(), time);
            assert_eq!(Time::from(Timestamp::from(time)), time);
        }
        // Negative times (before the epoch) round to the nearest tick too.
        assert_eq!(
            Timestamp::from_nanos(-149).to_cartographer_time(),
            Timestamp::from_nanos(-100).to_cartographer_time()
        );
    }

    #[test]
    fn arithmetic_and_display() {
        let a = Timestamp::from_secs_f64(1.0);
        assert_eq!(a.add_secs_f64(0.5), Timestamp::from_secs_f64(1.5));
        assert!((a.add_secs_f64(0.5).seconds_since(a) - 0.5).abs() < 1e-12);
        assert_eq!(Timestamp::from_secs_f64(1.5).to_string(), "1.500000000s");
        assert_eq!(Timestamp::from_secs_f64(-0.25).to_string(), "-0.250000000s");
        assert!(Timestamp::from_secs_f64(1.0) < Timestamp::from_secs_f64(1.000_000_001));
    }

    #[test]
    #[should_panic(expected = "not a representable time")]
    fn non_finite_seconds_panic() {
        let _ = Timestamp::from_secs_f64(f64::NAN);
    }
}
