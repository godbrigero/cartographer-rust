//! Port of `cartographer/io/proto_stream` (.h, .cc).
//!
//! The `.pbstream` format:
//!
//! ```text
//! file    := magic:u64le record*
//! magic   := 0x7b1d1f7b5bf501db   (bytes on disk: db 01 f5 5b 7b 1f 1d 7b)
//! record  := size:u64le payload[size]
//! payload := gzip(serialized protobuf message)  (one gzip member per message)
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};

use flate2::read::MultiGzDecoder;

use crate::common::port::fast_gzip_string;
use crate::io::proto_stream_interface::{
    ProtoMessage, ProtoStreamReaderInterface, ProtoStreamWriterInterface,
};

/// First eight bytes to identify our proto stream format.
pub const MAGIC: u64 = 0x7b1d_1f7b_5bf5_01db;

fn write_size_as_little_endian(size: u64, out: &mut impl Write) -> std::io::Result<()> {
    out.write_all(&size.to_le_bytes())
}

/// A simple writer of a compressed sequence of protocol buffer messages to a
/// file. The format is not intended to be compatible with any other format
/// used outside of Cartographer.
///
/// TODO(whess): Compress the file instead of individual messages for better
/// compression performance? Should we use LZ4?
pub struct ProtoStreamWriter {
    out: Option<Box<dyn Write + Send>>,
    // C++ `std::ofstream::fail()`.
    failed: bool,
}

impl ProtoStreamWriter {
    /// Creates (truncates) `filename` and writes the magic number.
    ///
    /// Like the C++ `std::ofstream`, failing to open the file is only
    /// reported by [`ProtoStreamWriterInterface::close`].
    pub fn new(filename: &str) -> Self {
        match File::create(filename) {
            Ok(file) => Self::from_writer(BufWriter::new(file)),
            Err(e) => {
                log::error!("Failed to open proto stream '{filename}' for writing: {e}");
                Self {
                    out: None,
                    failed: true,
                }
            }
        }
    }

    /// Creates a writer that writes the stream into `out` (an extension of
    /// the C++ API, e.g. for in-memory buffers).
    pub fn from_writer(out: impl Write + Send + 'static) -> Self {
        let mut writer = Self {
            out: Some(Box::new(out)),
            failed: false,
        };
        writer.write_raw(&MAGIC.to_le_bytes());
        writer
    }

    fn write_raw(&mut self, data: &[u8]) {
        match self.out.as_mut() {
            Some(out) if !self.failed => {
                if out.write_all(data).is_err() {
                    self.failed = true;
                }
            }
            _ => self.failed = true,
        }
    }

    fn write(&mut self, uncompressed_data: &[u8]) {
        let mut compressed_data = Vec::new();
        fast_gzip_string(uncompressed_data, &mut compressed_data);
        let mut size = Vec::with_capacity(8);
        write_size_as_little_endian(compressed_data.len() as u64, &mut size)
            .expect("writing to a Vec cannot fail");
        self.write_raw(&size);
        self.write_raw(&compressed_data);
    }
}

impl ProtoStreamWriterInterface for ProtoStreamWriter {
    fn write_proto(&mut self, proto: &dyn ProtoMessage) {
        let uncompressed_data = proto.serialize_to_bytes();
        self.write(&uncompressed_data);
    }

    fn close(&mut self) -> bool {
        match self.out.take() {
            Some(mut out) => {
                if out.flush().is_err() {
                    self.failed = true;
                }
            }
            // Closing a closed (or never opened) stream fails.
            None => self.failed = true,
        }
        !self.failed
    }
}

/// A reader of the format produced by [`ProtoStreamWriter`].
pub struct ProtoStreamReader {
    input: Box<dyn Read + Send>,
    // C++ `std::istream` state bits.
    eof: bool,
    fail: bool,
}

impl ProtoStreamReader {
    /// Opens `filename` and checks the magic number. Panics if the file
    /// cannot be opened or is not a proto stream (C++ CHECK).
    pub fn new(filename: &str) -> Self {
        let input: Box<dyn Read + Send> = match File::open(filename) {
            Ok(file) => Box::new(BufReader::new(file)),
            Err(_) => {
                panic!("Check failed: in_.good() Failed to open proto stream '{filename}'.")
            }
        };
        Self::from_boxed_reader(input, filename)
    }

    /// Reads a proto stream from `input` (an extension of the C++ API, e.g.
    /// for in-memory buffers). `name` is only used in the error message.
    /// Panics if the magic number does not match.
    pub fn from_reader(input: impl Read + Send + 'static, name: &str) -> Self {
        Self::from_boxed_reader(Box::new(input), name)
    }

    fn from_boxed_reader(input: Box<dyn Read + Send>, name: &str) -> Self {
        let mut reader = Self {
            input,
            eof: false,
            fail: false,
        };
        match reader.read_size_as_little_endian() {
            Some(magic) if magic == MAGIC => {}
            _ => reader.fail = true,
        }
        assert!(
            !reader.fail,
            "Check failed: in_.good() Failed to open proto stream '{name}'."
        );
        reader
    }

    // Reads exactly `buf.len()` bytes, updating the stream state like
    // `std::istream`.
    fn read_exact_or_fail(&mut self, buf: &mut [u8]) -> bool {
        if self.fail {
            return false;
        }
        let mut filled = 0;
        while filled < buf.len() {
            match self.input.read(&mut buf[filled..]) {
                Ok(0) => {
                    self.eof = true;
                    self.fail = true;
                    return false;
                }
                Ok(n) => filled += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    log::error!("Error reading proto stream: {e}");
                    self.fail = true;
                    return false;
                }
            }
        }
        true
    }

    fn read_size_as_little_endian(&mut self) -> Option<u64> {
        let mut bytes = [0u8; 8];
        if self.read_exact_or_fail(&mut bytes) {
            Some(u64::from_le_bytes(bytes))
        } else {
            None
        }
    }

    fn read(&mut self, decompressed_data: &mut Vec<u8>) -> bool {
        let Some(compressed_size) = self.read_size_as_little_endian() else {
            return false;
        };
        // RUST PORT NOTE: C++ allocates `compressed_size` bytes up front; we
        // read incrementally so that a corrupt size cannot exhaust memory.
        let mut compressed_data = Vec::new();
        if self.fail {
            return false;
        }
        match (&mut self.input)
            .take(compressed_size)
            .read_to_end(&mut compressed_data)
        {
            Ok(n) if n as u64 == compressed_size => {}
            Ok(_) => {
                self.eof = true;
                self.fail = true;
                return false;
            }
            Err(e) => {
                log::error!("Error reading proto stream: {e}");
                self.fail = true;
                return false;
            }
        }
        // RUST PORT NOTE: C++ (boost) throws on corrupt gzip data, which
        // terminates the process; we log and report a failed read.
        let mut decoder = MultiGzDecoder::new(compressed_data.as_slice());
        if let Err(e) = decoder.read_to_end(decompressed_data) {
            log::error!("Failed to decompress proto stream message: {e}");
            return false;
        }
        true
    }
}

impl ProtoStreamReaderInterface for ProtoStreamReader {
    fn read_proto(&mut self, proto: &mut dyn ProtoMessage) -> bool {
        let mut decompressed_data = Vec::new();
        self.read(&mut decompressed_data) && proto.parse_from_bytes(&decompressed_data)
    }

    /// True once a read hit the end of the file (like `std::istream::eof()`,
    /// this only becomes true *after* a failed read).
    fn eof(&self) -> bool {
        self.eof
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::{Arc, Mutex};

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use prost::Message;

    use super::*;
    use crate::mapping::proto;

    fn temp_file(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("cartographer_{}_{name}", std::process::id()))
            .to_str()
            .unwrap()
            .to_string()
    }

    mod proto_stream_test {
        use super::*;

        #[test]
        fn write_and_read_back() {
            let test_file = temp_file("test_trajectory.pbstream");
            {
                let mut writer = ProtoStreamWriter::new(&test_file);
                for i in 0..10 {
                    let mut trajectory = proto::Trajectory::default();
                    trajectory.node.push(proto::trajectory::Node {
                        timestamp: i,
                        ..Default::default()
                    });
                    writer.write_proto(&trajectory);
                }
                assert!(writer.close());
            }
            {
                let mut reader = ProtoStreamReader::new(&test_file);
                for i in 0..10 {
                    let mut trajectory = proto::Trajectory::default();
                    assert!(reader.read_proto(&mut trajectory));
                    assert_eq!(1, trajectory.node.len());
                    assert_eq!(i, trajectory.node[0].timestamp);
                }
                let mut trajectory = proto::Trajectory::default();
                assert!(!reader.read_proto(&mut trajectory));
            }
            std::fs::remove_file(&test_file).unwrap();
        }
    }

    /// A `Write` sink sharing its buffer.
    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn sample_serialized_data() -> Vec<proto::SerializedData> {
        use proto::serialized_data::Data;
        vec![
            proto::SerializedData {
                data: Some(Data::PoseGraph(proto::PoseGraph {
                    trajectory: vec![proto::Trajectory {
                        trajectory_id: 3,
                        node: vec![proto::trajectory::Node {
                            node_index: 5,
                            timestamp: -123_456_789_012,
                            pose: Some(crate::transform::proto::Rigid3d {
                                translation: Some(crate::transform::proto::Vector3d {
                                    x: 1.5,
                                    y: -2.25,
                                    z: 1e-300,
                                }),
                                rotation: Some(crate::transform::proto::Quaterniond {
                                    x: 0.,
                                    y: 0.,
                                    z: 0.,
                                    w: 1.,
                                }),
                            }),
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
            },
            proto::SerializedData {
                data: Some(Data::Submap(proto::Submap {
                    submap_id: Some(proto::SubmapId {
                        trajectory_id: 3,
                        submap_index: 1,
                    }),
                    submap_3d: Some(proto::Submap3D {
                        num_range_data: 90,
                        finished: true,
                        rotational_scan_matcher_histogram: vec![0.5; 120],
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            },
            // A oneof set to an empty message must survive the round trip.
            proto::SerializedData {
                data: Some(Data::LandmarkData(proto::LandmarkData::default())),
            },
            proto::SerializedData::default(),
        ]
    }

    #[test]
    fn round_trip_preserves_all_fields() {
        let buffer = SharedBuffer::default();
        let header = proto::SerializationHeader { format_version: 2 };
        let data = sample_serialized_data();
        {
            let mut writer = ProtoStreamWriter::from_writer(buffer.clone());
            writer.write_proto(&header);
            for d in &data {
                writer.write_proto(d);
            }
            assert!(writer.close());
        }
        let bytes = buffer.0.lock().unwrap().clone();
        assert_eq!(
            &bytes[..8],
            &[0xdb, 0x01, 0xf5, 0x5b, 0x7b, 0x1f, 0x1d, 0x7b]
        );

        let mut reader = ProtoStreamReader::from_reader(std::io::Cursor::new(bytes), "buffer");
        let mut read_header = proto::SerializationHeader::default();
        assert!(reader.read_proto(&mut read_header));
        assert_eq!(read_header, header);
        for expected in &data {
            let mut actual = proto::SerializedData::default();
            assert!(!reader.eof());
            assert!(reader.read_proto(&mut actual));
            assert_eq!(&actual, expected);
        }
        let mut actual = proto::SerializedData::default();
        assert!(!reader.eof());
        assert!(!reader.read_proto(&mut actual));
        assert!(reader.eof());
    }

    /// Assembles a pbstream by hand following the documented C++ format
    /// (magic, then u64 LE size + gzip member per message), using an
    /// independent gzip encoder (best compression, unlike the writer).
    #[test]
    fn reads_hand_assembled_cpp_format() {
        let header = proto::SerializationHeader { format_version: 1 };
        let data = sample_serialized_data();
        let mut bytes = vec![0xdb, 0x01, 0xf5, 0x5b, 0x7b, 0x1f, 0x1d, 0x7b];
        let mut append = |payload: Vec<u8>| {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(&payload).unwrap();
            let compressed = encoder.finish().unwrap();
            let size = compressed.len() as u64;
            for i in 0..8 {
                bytes.push(((size >> (8 * i)) & 0xff) as u8);
            }
            bytes.extend_from_slice(&compressed);
        };
        append(header.encode_to_vec());
        for d in &data {
            append(d.encode_to_vec());
        }
        let path = temp_file("hand_assembled.pbstream");
        std::fs::write(&path, &bytes).unwrap();

        let mut reader = ProtoStreamReader::new(&path);
        let mut read_header = proto::SerializationHeader::default();
        assert!(reader.read_proto(&mut read_header));
        assert_eq!(read_header.format_version, 1);
        for expected in &data {
            let mut actual = proto::SerializedData::default();
            assert!(reader.read_proto(&mut actual));
            assert_eq!(&actual, expected);
        }
        assert!(!reader.read_proto(&mut proto::SerializedData::default()));
        assert!(reader.eof());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn writer_output_is_standard_gzip_per_record() {
        let buffer = SharedBuffer::default();
        let mut writer = ProtoStreamWriter::from_writer(buffer.clone());
        let message = proto::SerializationHeader { format_version: 2 };
        writer.write_proto(&message);
        assert!(writer.close());
        let bytes = buffer.0.lock().unwrap().clone();
        let size = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 16 + size);
        let payload = &bytes[16..];
        assert_eq!(&payload[..3], &[0x1f, 0x8b, 0x08]);
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(payload)
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, message.encode_to_vec());
    }

    #[test]
    fn truncated_record_fails_and_sets_eof() {
        let buffer = SharedBuffer::default();
        let mut writer = ProtoStreamWriter::from_writer(buffer.clone());
        writer.write_proto(&proto::SerializationHeader { format_version: 2 });
        assert!(writer.close());
        let mut bytes = buffer.0.lock().unwrap().clone();
        bytes.truncate(bytes.len() - 3);
        let mut reader = ProtoStreamReader::from_reader(std::io::Cursor::new(bytes), "truncated");
        assert!(!reader.eof());
        assert!(!reader.read_proto(&mut proto::SerializationHeader::default()));
        assert!(reader.eof());
    }

    #[test]
    #[should_panic(expected = "Failed to open proto stream 'bad'")]
    fn wrong_magic_panics() {
        ProtoStreamReader::from_reader(std::io::Cursor::new(vec![0u8; 16]), "bad");
    }

    #[test]
    #[should_panic(expected = "Failed to open proto stream")]
    fn missing_file_panics() {
        ProtoStreamReader::new("/nonexistent/file.pbstream");
    }

    #[test]
    fn writer_to_unwritable_path_fails_on_close() {
        let mut writer = ProtoStreamWriter::new("/nonexistent/dir/out.pbstream");
        writer.write_proto(&proto::SerializationHeader::default());
        assert!(!writer.close());
    }
}
