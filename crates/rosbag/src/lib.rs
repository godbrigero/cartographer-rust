//! A small, dependency-light, pure-Rust reader for ROS1 bag files (format
//! 2.0), plus decoders for the handful of message types Cartographer consumes.
//!
//! ```no_run
//! use cartographer_rosbag::{Bag, msgs::Imu};
//!
//! let mut bag = Bag::open("data.bag")?;
//! for msg in bag.messages_for_topics(&["/imu"])? {
//!     let msg = msg?;
//!     let imu: Imu = msg.decode()?;
//!     println!("{} {:?}", msg.time, imu.angular_velocity);
//! }
//! # Ok::<(), cartographer_rosbag::Error>(())
//! ```
//!
//! Supported chunk compression: `none`, `lz4` (via `lz4_flex`) and, with the
//! default `bz2` feature, `bz2` (via the pure-Rust `bzip2-rs`). No C code is
//! compiled.

mod bag;
pub mod decode;
mod error;
pub mod msgs;
mod record;

pub use bag::{Bag, ChunkInfo, Connection, Messages, RawMessage, TopicSummary};
pub use decode::{Decoder, RosMsg};
pub use error::{Error, Result};

use std::fmt;

/// ROS1 time: seconds and nanoseconds since the Unix epoch (or since an
/// arbitrary origin for simulated clocks).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time {
    /// Whole seconds.
    pub sec: u32,
    /// Nanoseconds within the second (normally `< 1_000_000_000`).
    pub nsec: u32,
}

impl Time {
    /// Creates a time from seconds and nanoseconds.
    pub const fn new(sec: u32, nsec: u32) -> Self {
        Self { sec, nsec }
    }

    /// Creates a time from total nanoseconds.
    pub const fn from_nanos(nanos: u64) -> Self {
        Self {
            sec: (nanos / 1_000_000_000) as u32,
            nsec: (nanos % 1_000_000_000) as u32,
        }
    }

    /// Total nanoseconds.
    pub const fn as_nanos(self) -> u64 {
        self.sec as u64 * 1_000_000_000 + self.nsec as u64
    }

    /// Seconds as a floating-point value.
    pub fn as_secs_f64(self) -> f64 {
        f64::from(self.sec) + f64::from(self.nsec) * 1e-9
    }

    /// Decodes the packed little-endian `u64` used in record headers
    /// (low 32 bits = sec, high 32 bits = nsec).
    pub(crate) fn from_packed(v: u64) -> Self {
        Self {
            sec: v as u32,
            nsec: (v >> 32) as u32,
        }
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{:09}", self.sec, self.nsec)
    }
}

/// Normalizes a ROS message type name: `pkg/msg/Name` (ROS2 style, as written
/// by some tools) becomes `pkg/Name`.
pub fn normalize_type(t: &str) -> std::borrow::Cow<'_, str> {
    match t.split_once("/msg/") {
        Some((pkg, name)) if !name.contains('/') => format!("{pkg}/{name}").into(),
        _ => t.into(),
    }
}
