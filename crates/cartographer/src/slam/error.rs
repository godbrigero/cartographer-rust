//! The error type of the high-level API.

use std::fmt;
use std::path::PathBuf;

use super::time::Timestamp;
use super::types::SensorKind;

/// Everything that can go wrong in the high-level [`Slam`](super::Slam) API.
///
/// Input errors ([`UnknownSensor`](Self::UnknownSensor),
/// [`SensorNotConfigured`](Self::SensorNotConfigured),
/// [`NonMonotonicTime`](Self::NonMonotonicTime),
/// [`InvalidData`](Self::InvalidData),
/// [`TrajectoryFinished`](Self::TrajectoryFinished)) are detected before
/// anything is handed to Cartographer: the offending sample is dropped and
/// the [`Slam`](super::Slam) instance stays fully usable, so a caller may
/// log and continue.
#[derive(Debug)]
#[non_exhaustive]
pub enum SlamError {
    /// Data was pushed for a range sensor id that was not declared on the
    /// [`SlamBuilder`](super::SlamBuilder).
    UnknownSensor {
        /// The id that was used.
        id: String,
    },
    /// Data of a kind (IMU, odometry) was pushed although no such sensor was
    /// declared on the [`SlamBuilder`](super::SlamBuilder).
    SensorNotConfigured {
        /// The kind of sensor that is missing.
        kind: SensorKind,
    },
    /// A sensor's timestamps must be strictly increasing.
    NonMonotonicTime {
        /// Sensor id.
        sensor: String,
        /// Timestamp of the previous accepted sample of this sensor.
        previous: Timestamp,
        /// The rejected timestamp.
        current: Timestamp,
    },
    /// A sensor required by the configuration is missing (e.g. no LiDAR, or
    /// no IMU in 3D).
    MissingSensor(String),
    /// The same sensor id was declared twice, or a second IMU / odometry
    /// source was declared.
    DuplicateSensor(String),
    /// A configuration value is out of range or the configuration could not
    /// be loaded (Lua errors are reported here).
    InvalidConfig(String),
    /// The pushed data is malformed (non-finite values, point times after
    /// the message time, unsorted point times, ...).
    InvalidData(String),
    /// [`Slam::finish`](super::Slam::finish) was already called; no more
    /// data is accepted.
    TrajectoryFinished,
    /// The operation is not available in this mode (e.g. a 2D occupancy grid
    /// of a 3D map).
    Unsupported(String),
    /// A serialized map (`.pbstream`) could not be loaded.
    InvalidMap(String),
    /// An I/O error while reading or writing `path`.
    Io {
        /// The file involved.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}

impl SlamError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        SlamError::Io {
            path: path.into(),
            source,
        }
    }
}

impl fmt::Display for SlamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlamError::UnknownSensor { id } => {
                write!(
                    f,
                    "unknown range sensor '{id}' (declare it with SlamBuilder::lidar)"
                )
            }
            SlamError::SensorNotConfigured { kind } => write!(
                f,
                "no {kind} sensor was declared on the SlamBuilder, cannot accept {kind} data"
            ),
            SlamError::NonMonotonicTime {
                sensor,
                previous,
                current,
            } => write!(
                f,
                "timestamps of sensor '{sensor}' must be strictly increasing: got {current} after \
                 {previous}"
            ),
            SlamError::MissingSensor(message) => write!(f, "missing sensor: {message}"),
            SlamError::DuplicateSensor(message) => write!(f, "duplicate sensor: {message}"),
            SlamError::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            SlamError::InvalidData(message) => write!(f, "invalid sensor data: {message}"),
            SlamError::TrajectoryFinished => {
                write!(f, "the trajectory is finished; no more data is accepted")
            }
            SlamError::Unsupported(message) => write!(f, "unsupported: {message}"),
            SlamError::InvalidMap(message) => write!(f, "cannot load map: {message}"),
            SlamError::Io { path, source } => {
                write!(f, "I/O error on '{}': {source}", path.display())
            }
        }
    }
}

impl std::error::Error for SlamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SlamError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Runs `f`, converting a panic (a failed Cartographer `CHECK`) into a
/// message.
pub(crate) fn catch_check_failure<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            "unknown failure".to_string()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_source() {
        let error = SlamError::io(
            "/x/map.pbstream",
            std::io::Error::new(std::io::ErrorKind::NotFound, "gone"),
        );
        assert!(error.to_string().contains("/x/map.pbstream"));
        assert!(std::error::Error::source(&error).is_some());
        let error = SlamError::NonMonotonicTime {
            sensor: "scan".into(),
            previous: Timestamp::from_secs_f64(2.0),
            current: Timestamp::from_secs_f64(1.0),
        };
        assert!(error.to_string().contains("'scan'"));
        assert!(std::error::Error::source(&error).is_none());
        assert!(
            SlamError::SensorNotConfigured {
                kind: SensorKind::Imu
            }
            .to_string()
            .contains("IMU")
        );
    }

    #[test]
    fn catch_check_failure_extracts_messages() {
        assert_eq!(catch_check_failure(|| 3), Ok(3));
        let message = catch_check_failure(|| -> i32 { panic!("Check failed: {}", 42) });
        assert_eq!(message, Err("Check failed: 42".to_string()));
        let message = catch_check_failure(|| -> i32 { panic!("static") });
        assert_eq!(message, Err("static".to_string()));
    }
}
