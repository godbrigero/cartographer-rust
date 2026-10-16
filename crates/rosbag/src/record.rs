//! Low-level record layer of the bag format.
//!
//! Every record is `header_len: u32, header, data_len: u32, data`, where the
//! header is a sequence of `field_len: u32, name '=' value` entries and always
//! contains an `op` field (a single byte) identifying the record type.

use std::io::Read;

use crate::{Error, Result, Time};

pub(crate) const OP_MSG_DATA: u8 = 0x02;
pub(crate) const OP_BAG_HEADER: u8 = 0x03;
// 0x04 = index data: per-connection (time, offset) lists; not needed because
// whole chunks are decompressed and scanned.
pub(crate) const OP_CHUNK: u8 = 0x05;
pub(crate) const OP_CHUNK_INFO: u8 = 0x06;
pub(crate) const OP_CONNECTION: u8 = 0x07;

/// Parsed record header: list of `(name, value)` pairs.
#[derive(Debug, Default)]
pub(crate) struct Fields {
    pub(crate) op: u8,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Fields {
    /// Parses a header block. `require_op` is false for connection-record
    /// data sections, which use the same encoding but carry no `op`.
    pub(crate) fn parse(mut buf: &[u8], require_op: bool) -> Result<Self> {
        let mut fields = Fields::default();
        let mut have_op = false;
        while !buf.is_empty() {
            if buf.len() < 4 {
                return Err(Error::Malformed("truncated header field length".into()));
            }
            let len = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
            buf = &buf[4..];
            if len > buf.len() {
                return Err(Error::Malformed(format!(
                    "header field length {len} exceeds remaining {}",
                    buf.len()
                )));
            }
            let (field, rest) = buf.split_at(len);
            buf = rest;
            let eq = field
                .iter()
                .position(|&b| b == b'=')
                .ok_or_else(|| Error::Malformed("header field without '='".into()))?;
            let (name, value) = (&field[..eq], &field[eq + 1..]);
            if name == b"op" {
                if value.len() != 1 {
                    return Err(Error::Malformed("op field must be 1 byte".into()));
                }
                fields.op = value[0];
                have_op = true;
            }
            fields.entries.push((name.to_vec(), value.to_vec()));
        }
        if require_op && !have_op {
            return Err(Error::Malformed("record header without op field".into()));
        }
        Ok(fields)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(n, _)| n == name.as_bytes())
            .map(|(_, v)| v.as_slice())
    }

    fn need(&self, name: &'static str) -> Result<&[u8]> {
        self.get(name).ok_or(Error::MissingField {
            op: self.op,
            field: name,
        })
    }

    fn fixed<const N: usize>(&self, name: &'static str) -> Result<[u8; N]> {
        self.need(name)?.try_into().map_err(|_| {
            Error::Malformed(format!(
                "field `{name}` of op 0x{:02x} must be {N} bytes",
                self.op
            ))
        })
    }

    pub(crate) fn u32(&self, name: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed(name)?))
    }

    pub(crate) fn u64(&self, name: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed(name)?))
    }

    pub(crate) fn time(&self, name: &'static str) -> Result<Time> {
        Ok(Time::from_packed(self.u64(name)?))
    }

    pub(crate) fn string(&self, name: &'static str) -> Result<String> {
        String::from_utf8(self.need(name)?.to_vec()).map_err(|_| Error::InvalidUtf8)
    }

    pub(crate) fn opt_string(&self, name: &str) -> Result<Option<String>> {
        self.get(name)
            .map(|v| String::from_utf8(v.to_vec()).map_err(|_| Error::InvalidUtf8))
            .transpose()
    }
}

/// Reads a little-endian `u32`; returns `None` on clean EOF before any byte.
fn read_u32_or_eof<R: Read>(r: &mut R) -> Result<Option<u32>> {
    let mut b = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        match r.read(&mut b[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => return Err(Error::Malformed("truncated record length".into())),
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(Some(u32::from_le_bytes(b)))
}

fn read_exact_vec<R: Read>(r: &mut R, len: usize, limit: u64) -> Result<Vec<u8>> {
    if len as u64 > limit {
        return Err(Error::Malformed(format!(
            "record length {len} exceeds remaining {limit} bytes"
        )));
    }
    let mut v = vec![0u8; len];
    r.read_exact(&mut v).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::Malformed("truncated record".into())
        } else {
            e.into()
        }
    })?;
    Ok(v)
}

/// Reads only the header of the next record. Returns `None` at clean EOF.
/// `limit` bounds the lengths accepted (bytes remaining in the source).
pub(crate) fn read_header<R: Read>(r: &mut R, limit: u64) -> Result<Option<Fields>> {
    let Some(hlen) = read_u32_or_eof(r)? else {
        return Ok(None);
    };
    let header = read_exact_vec(r, hlen as usize, limit)?;
    Fields::parse(&header, true).map(Some)
}

/// Reads the `data_len` prefix and the data block that follows a header.
pub(crate) fn read_data<R: Read>(r: &mut R, limit: u64) -> Result<Vec<u8>> {
    let dlen = read_u32_or_eof(r)?
        .ok_or_else(|| Error::Malformed("record without data section".into()))?;
    read_exact_vec(r, dlen as usize, limit)
}

/// Reads one complete record from an in-memory buffer (chunk contents),
/// advancing `buf`. Returns `None` when `buf` is empty.
pub(crate) fn next_in_buf<'a>(buf: &mut &'a [u8]) -> Result<Option<(Fields, &'a [u8])>> {
    if buf.is_empty() {
        return Ok(None);
    }
    let take = |buf: &mut &'a [u8], what: &str| -> Result<&'a [u8]> {
        if buf.len() < 4 {
            return Err(Error::Malformed(format!(
                "truncated {what} length in chunk"
            )));
        }
        let len = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
        let rest = &buf[4..];
        if len > rest.len() {
            return Err(Error::Malformed(format!("truncated {what} in chunk")));
        }
        let (block, rest) = rest.split_at(len);
        *buf = rest;
        Ok(block)
    };
    let header = take(buf, "record header")?;
    let fields = Fields::parse(header, true)?;
    let data = take(buf, "record data")?;
    Ok(Some((fields, data)))
}

/// Decompresses chunk data according to its `compression` field.
pub(crate) fn decompress(compression: &str, data: Vec<u8>, size: usize) -> Result<Vec<u8>> {
    let out = match compression {
        "none" => data,
        "lz4" => {
            let mut out = Vec::with_capacity(size);
            lz4_flex::frame::FrameDecoder::new(data.as_slice())
                .read_to_end(&mut out)
                .map_err(|e| Error::Decompression(format!("lz4: {e}")))?;
            out
        }
        #[cfg(feature = "bz2")]
        "bz2" => {
            let mut out = Vec::with_capacity(size);
            bzip2_rs::DecoderReader::new(data.as_slice())
                .read_to_end(&mut out)
                .map_err(|e| Error::Decompression(format!("bz2: {e}")))?;
            out
        }
        other => return Err(Error::UnsupportedCompression(other.to_string())),
    };
    if out.len() != size {
        return Err(Error::Decompression(format!(
            "{compression} chunk decompressed to {} bytes, header says {size}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_compression_is_reported() {
        let err = decompress("zstd", vec![], 0).unwrap_err();
        assert!(matches!(err, Error::UnsupportedCompression(ref c) if c == "zstd"));
        assert_eq!(err.to_string(), "unsupported compression `zstd`");
    }

    #[test]
    fn size_mismatch_is_reported() {
        let err = decompress("none", vec![1, 2, 3], 4).unwrap_err();
        assert!(matches!(err, Error::Decompression(_)));
    }

    /// `bz2.compress(b"hello rosbag " * 8, 9)` from Python's standard library.
    #[cfg(feature = "bz2")]
    #[test]
    fn bz2_golden() {
        const BZ2_HELLO: [u8; 56] = [
            0x42, 0x5a, 0x68, 0x39, 0x31, 0x41, 0x59, 0x26, 0x53, 0x59, 0x85, 0xb1, 0x1b, 0xa1,
            0x00, 0x00, 0x1b, 0x91, 0x80, 0x40, 0x00, 0x32, 0xc4, 0x98, 0x00, 0x20, 0x00, 0x31,
            0x00, 0x30, 0x0a, 0xa8, 0x34, 0xc2, 0x40, 0xa1, 0x03, 0x05, 0x09, 0x12, 0x28, 0x58,
            0x60, 0xe8, 0xb1, 0xf1, 0x77, 0x24, 0x53, 0x85, 0x09, 0x08, 0x5b, 0x11, 0xba, 0x10,
        ];
        let expected = b"hello rosbag ".repeat(8);
        let out = decompress("bz2", BZ2_HELLO.to_vec(), expected.len()).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn fields_parse_and_lookup() {
        let mut h = Vec::new();
        for f in [
            &b"op=\x05"[..],
            b"compression=lz4",
            b"size=\x10\x00\x00\x00",
        ] {
            h.extend_from_slice(&(f.len() as u32).to_le_bytes());
            h.extend_from_slice(f);
        }
        let fields = Fields::parse(&h, true).unwrap();
        assert_eq!(fields.op, OP_CHUNK);
        assert_eq!(fields.string("compression").unwrap(), "lz4");
        assert_eq!(fields.u32("size").unwrap(), 16);
        assert!(matches!(
            fields.u32("missing"),
            Err(Error::MissingField {
                op: 5,
                field: "missing"
            })
        ));
    }
}
