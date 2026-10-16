//! Error type for bag reading and message decoding.

use std::fmt;

/// Result alias used throughout this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while reading a bag or decoding a message.
#[derive(Debug)]
pub enum Error {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The file does not start with `#ROSBAG V2.0\n`.
    BadMagic,
    /// The bag has no index (`index_pos == 0`), e.g. it was not closed
    /// properly. Run `rosbag reindex` on it first.
    Unindexed,
    /// A record or message is structurally invalid.
    Malformed(String),
    /// A required record header field is missing.
    MissingField {
        /// Record op code.
        op: u8,
        /// Name of the missing field.
        field: &'static str,
    },
    /// The chunk uses a compression scheme this build cannot decode.
    UnsupportedCompression(String),
    /// Decompression of a chunk failed.
    Decompression(String),
    /// Message data ended before decoding finished.
    UnexpectedEof {
        /// Bytes needed by the read that failed.
        needed: usize,
        /// Bytes remaining in the buffer.
        remaining: usize,
    },
    /// Message decoded successfully but bytes were left over.
    TrailingBytes(usize),
    /// A string field is not valid UTF-8.
    InvalidUtf8,
    /// Attempted to decode a message as the wrong type.
    TypeMismatch {
        /// Type requested by the caller.
        expected: &'static str,
        /// Type recorded in the bag's connection.
        actual: String,
    },
    /// A PointCloud2 field has an unknown datatype or lies outside the point.
    BadPointField(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::BadMagic => write!(f, "not a ROS bag v2.0 file (bad magic)"),
            Error::Unindexed => write!(f, "bag is not indexed (index_pos is 0); reindex it first"),
            Error::Malformed(m) => write!(f, "malformed bag: {m}"),
            Error::MissingField { op, field } => {
                write!(f, "record op 0x{op:02x} is missing header field `{field}`")
            }
            Error::UnsupportedCompression(c) => write!(f, "unsupported compression `{c}`"),
            Error::Decompression(m) => write!(f, "decompression failed: {m}"),
            Error::UnexpectedEof { needed, remaining } => write!(
                f,
                "unexpected end of message data (needed {needed} bytes, {remaining} remaining)"
            ),
            Error::TrailingBytes(n) => write!(f, "{n} trailing bytes after decoded message"),
            Error::InvalidUtf8 => write!(f, "string is not valid UTF-8"),
            Error::TypeMismatch { expected, actual } => {
                write!(
                    f,
                    "message type mismatch: expected {expected}, bag has {actual}"
                )
            }
            Error::BadPointField(m) => write!(f, "bad PointCloud2 field: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
