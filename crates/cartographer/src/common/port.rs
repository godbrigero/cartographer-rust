//! Port of `cartographer/common/port` (.h).
//!
//! The C++ integer aliases (`int64` etc.) map to Rust's native integer types.

use std::io::{Read, Write};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

/// Rounds to the nearest integer, halfway cases away from zero
/// (C++ `RoundToInt`, i.e. `std::lround`). Accepts `f32` and `f64` (the
/// C++ overloads); widening `f32` to `f64` is exact so the result is the same.
pub fn round_to_int(x: impl Into<f64>) -> i32 {
    // RUST PORT NOTE: `as` saturates on overflow (UB in C++).
    x.into().round() as i32
}

/// Like [`round_to_int`], returning a 64-bit integer (C++ `RoundToInt64`).
pub fn round_to_int64(x: impl Into<f64>) -> i64 {
    x.into().round() as i64
}

/// Gzip-compresses `uncompressed` with the fastest compression level
/// (zlib `best_speed`) and **appends** the result to `compressed`, like the
/// C++ version which writes through a `back_inserter`.
pub fn fast_gzip_string(uncompressed: &[u8], compressed: &mut Vec<u8>) {
    let mut encoder = GzEncoder::new(compressed, Compression::fast());
    encoder
        .write_all(uncompressed)
        .expect("writing to a Vec cannot fail");
    encoder.finish().expect("writing to a Vec cannot fail");
}

/// Decompresses gzip data `compressed` and **appends** the result to
/// `decompressed`.
///
/// Panics if the input is not valid gzip data (the C++ version throws a
/// boost exception).
pub fn fast_gunzip_string(compressed: &[u8], decompressed: &mut Vec<u8>) {
    let mut decoder = GzDecoder::new(compressed);
    decoder
        .read_to_end(decompressed)
        .unwrap_or_else(|e| panic!("gzip decompression failed: {e}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn round_to_int_rounds_half_away_from_zero() {
        assert_eq!(round_to_int(0.5), 1);
        assert_eq!(round_to_int(-0.5), -1);
        assert_eq!(round_to_int(1.49), 1);
        assert_eq!(round_to_int(-2.5), -3);
        assert_eq!(round_to_int(2.5f32), 3);
        assert_eq!(round_to_int64(1e12 + 0.5), 1_000_000_000_001);
        assert_eq!(round_to_int64(-3.5f32), -4);
    }

    #[test]
    fn gzip_appends() {
        let mut compressed = b"prefix".to_vec();
        fast_gzip_string(b"hello hello hello", &mut compressed);
        assert_eq!(&compressed[..6], b"prefix");
        // gzip magic.
        assert_eq!(&compressed[6..8], &[0x1f, 0x8b]);
        let mut decompressed = b"x".to_vec();
        fast_gunzip_string(&compressed[6..], &mut decompressed);
        assert_eq!(decompressed, b"xhello hello hello");
    }

    proptest! {
        #[test]
        fn gzip_round_trip(data in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let mut compressed = Vec::new();
            fast_gzip_string(&data, &mut compressed);
            let mut decompressed = Vec::new();
            fast_gunzip_string(&compressed, &mut decompressed);
            prop_assert_eq!(decompressed, data);
        }
    }
}
