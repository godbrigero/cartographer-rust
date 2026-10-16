//! ROS1 message deserialization primitives (little-endian, length-prefixed
//! strings and variable-length arrays).

use crate::{Error, Result, Time};

/// A message type that can be decoded from ROS1 serialized bytes.
pub trait RosMsg: Sized {
    /// Canonical ROS1 type name, e.g. `sensor_msgs/Imu`.
    const MSG_TYPE: &'static str;

    /// Decodes the message from the decoder's current position.
    fn decode_from(d: &mut Decoder<'_>) -> Result<Self>;

    /// Decodes a complete serialized message; all bytes must be consumed.
    fn decode(bytes: &[u8]) -> Result<Self> {
        let mut d = Decoder::new(bytes);
        let msg = Self::decode_from(&mut d)?;
        d.finish()?;
        Ok(msg)
    }
}

/// Cursor over serialized message bytes.
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    buf: &'a [u8],
}

impl<'a> Decoder<'a> {
    /// Creates a decoder over `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len()
    }

    /// Errors with [`Error::TrailingBytes`] unless all input was consumed.
    pub fn finish(&self) -> Result<()> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(Error::TrailingBytes(self.buf.len()))
        }
    }

    /// Consumes `n` raw bytes.
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.buf.len() {
            return Err(Error::UnexpectedEof {
                needed: n,
                remaining: self.buf.len(),
            });
        }
        let (head, tail) = self.buf.split_at(n);
        self.buf = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.bytes(N)?.try_into().unwrap())
    }

    /// Reads a `uint8`.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }
    /// Reads a `bool` (any non-zero byte is true).
    pub fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }
    /// Reads a `uint16`.
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    /// Reads a `uint32`.
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    /// Reads an `int32`.
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    /// Reads a `uint64`.
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    /// Reads a `float32`.
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.array()?))
    }
    /// Reads a `float64`.
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }
    /// Reads a `time` (`uint32 sec, uint32 nsec`).
    pub fn time(&mut self) -> Result<Time> {
        Ok(Time::new(self.u32()?, self.u32()?))
    }

    /// Reads a variable-length array length prefix, checking that
    /// `len * elem_size` bytes are actually available (guards allocations).
    pub fn len_prefix(&mut self, elem_size: usize) -> Result<usize> {
        let len = self.u32()? as usize;
        let needed = len.saturating_mul(elem_size.max(1));
        if needed > self.buf.len() {
            return Err(Error::UnexpectedEof {
                needed,
                remaining: self.buf.len(),
            });
        }
        Ok(len)
    }

    /// Reads a `string`.
    pub fn string(&mut self) -> Result<String> {
        let len = self.len_prefix(1)?;
        let b = self.bytes(len)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::InvalidUtf8)
    }

    /// Reads a `uint8[]` (byte blob).
    pub fn byte_vec(&mut self) -> Result<Vec<u8>> {
        let len = self.len_prefix(1)?;
        Ok(self.bytes(len)?.to_vec())
    }

    /// Reads a `float32[]`.
    pub fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let len = self.len_prefix(4)?;
        (0..len).map(|_| self.f32()).collect()
    }

    /// Reads a fixed-size `float64[N]`.
    pub fn f64_array<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut out = [0.0; N];
        for v in &mut out {
            *v = self.f64()?;
        }
        Ok(out)
    }

    /// Reads a variable-length array of messages. `min_elem_size` is a lower
    /// bound on each element's serialized size, used to sanity-check `len`.
    pub fn msg_vec<T: RosMsg>(&mut self, min_elem_size: usize) -> Result<Vec<T>> {
        let len = self.len_prefix(min_elem_size)?;
        (0..len).map(|_| T::decode_from(self)).collect()
    }
}
