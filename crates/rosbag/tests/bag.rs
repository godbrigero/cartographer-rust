mod common;

use std::io::Cursor;

use cartographer_rosbag::msgs::{Header, Imu, PoseStamped};
use cartographer_rosbag::{Bag, Error, Time};
use common::{BagWriter, Compression, Ser};

fn header_msg(seq: u32, t_ns: u64) -> Vec<u8> {
    Ser::default()
        .header(
            seq,
            (t_ns / 1_000_000_000) as u32,
            (t_ns % 1_000_000_000) as u32,
            "f",
        )
        .take()
}

/// Two topics spread over three chunks whose time ranges overlap, with
/// out-of-order messages inside a chunk.
fn sample_bag(compression: Compression) -> Vec<u8> {
    let mut w = BagWriter::new(compression);
    let a = w.add_connection("/a", "std_msgs/Header");
    let b = w.add_connection("/b", "std_msgs/msg/Header"); // ROS2-style name.
    // Chunk 0: a@10, b@30, a@20 (out of order inside the chunk).
    w.write(a, 10, header_msg(0, 10));
    w.write(b, 30, header_msg(1, 30));
    w.write(a, 20, header_msg(2, 20));
    // Chunk 1 overlaps chunk 0: b@15, a@40.
    w.new_chunk();
    w.write(b, 15, header_msg(3, 15));
    w.write(a, 40, header_msg(4, 40));
    // Chunk 2: later, only /a, includes a tie with chunk 1 (time 40).
    w.new_chunk();
    w.write(a, 40, header_msg(5, 40));
    w.write(a, 1_500_000_050, header_msg(6, 1_500_000_050));
    w.finish()
}

fn check_sample(compression: Compression) {
    let bytes = sample_bag(compression);
    let mut bag = Bag::from_reader(Cursor::new(bytes)).unwrap();

    assert_eq!(bag.connections().len(), 2);
    let c0 = &bag.connections()[0];
    assert_eq!(c0.topic, "/a");
    assert_eq!(c0.msg_type, "std_msgs/Header");
    assert_eq!(c0.md5sum, "0123456789abcdef0123456789abcdef");
    assert_eq!(c0.callerid.as_deref(), Some("/test_node"));
    assert!(!c0.latching);
    assert_eq!(bag.connections()[1].msg_type, "std_msgs/Header");
    assert_eq!(bag.chunks().len(), 3);
    assert_eq!(bag.message_count(), 7);
    assert_eq!(bag.start_time(), Some(Time::new(0, 10)));
    assert_eq!(bag.end_time(), Some(Time::new(1, 500_000_050)));
    let topics = bag.topics();
    assert_eq!(
        topics
            .iter()
            .map(|t| (t.topic.as_str(), t.message_count))
            .collect::<Vec<_>>(),
        vec![("/a", 5), ("/b", 2)]
    );
    let expected_compression = match compression {
        Compression::None => "none",
        Compression::Lz4 => "lz4",
    };
    assert_eq!(
        bag.chunk_compressions().unwrap(),
        vec![(expected_compression.to_string(), 3)]
    );

    let all: Vec<(String, u64, u32)> = bag
        .messages()
        .map(|m| {
            let m = m.unwrap();
            let h: Header = m.decode().unwrap();
            assert_eq!(h.stamp, m.time);
            (m.topic().to_string(), m.time.as_nanos(), h.seq)
        })
        .collect();
    assert_eq!(
        all,
        vec![
            ("/a".into(), 10, 0),
            ("/b".into(), 15, 3),
            ("/a".into(), 20, 2),
            ("/b".into(), 30, 1),
            ("/a".into(), 40, 4), // Tie: chunk 1 before chunk 2 (file order).
            ("/a".into(), 40, 5),
            ("/a".into(), 1_500_000_050, 6),
        ]
    );

    let b_only: Vec<u32> = bag
        .messages_for_topics(&["/b"])
        .unwrap()
        .map(|m| m.unwrap().decode::<Header>().unwrap().seq)
        .collect();
    assert_eq!(b_only, vec![3, 1]);

    // Iterating twice works (the iterator re-seeks).
    assert_eq!(bag.messages().count(), 7);
}

#[test]
fn uncompressed_roundtrip() {
    check_sample(Compression::None);
}

#[test]
fn lz4_roundtrip() {
    check_sample(Compression::Lz4);
}

#[test]
fn open_from_file() {
    let dir = std::env::temp_dir().join(format!("cartographer-rosbag-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample.bag");
    std::fs::write(&path, sample_bag(Compression::Lz4)).unwrap();
    let mut bag = Bag::open(&path).unwrap();
    assert_eq!(bag.messages().count(), 7);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn type_mismatch_is_reported() {
    let mut bag = Bag::from_reader(Cursor::new(sample_bag(Compression::None))).unwrap();
    let m = bag.messages().next().unwrap().unwrap();
    match m.decode::<Imu>() {
        Err(Error::TypeMismatch { expected, actual }) => {
            assert_eq!(expected, "sensor_msgs/Imu");
            assert_eq!(actual, "std_msgs/Header");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(matches!(
        m.decode::<PoseStamped>(),
        Err(Error::TypeMismatch { .. })
    ));
}

#[test]
fn unknown_topic_filter_errors() {
    let mut bag = Bag::from_reader(Cursor::new(sample_bag(Compression::None))).unwrap();
    assert!(bag.messages_for_topics(&["/nope"]).is_err());
}

#[test]
fn bad_magic() {
    let mut bytes = sample_bag(Compression::None);
    bytes[9] = b'1'; // "#ROSBAG V1.0"
    assert!(matches!(
        Bag::from_reader(Cursor::new(bytes)),
        Err(Error::BadMagic)
    ));
    assert!(matches!(
        Bag::from_reader(Cursor::new(b"#ROS".to_vec())),
        Err(Error::BadMagic)
    ));
}

#[test]
fn unindexed_bag() {
    let mut w = BagWriter::new(Compression::None);
    let a = w.add_connection("/a", "std_msgs/Header");
    w.write(a, 1, header_msg(0, 1));
    let mut bytes = w.finish();
    // Zero out index_pos in the bag header ("index_pos=" + 8 bytes).
    let at = find(&bytes, b"index_pos=").unwrap() + b"index_pos=".len();
    bytes[at..at + 8].fill(0);
    assert!(matches!(
        Bag::from_reader(Cursor::new(bytes)),
        Err(Error::Unindexed)
    ));
}

#[test]
fn unsupported_compression() {
    let mut bytes = sample_bag(Compression::None);
    while let Some(at) = find(&bytes, b"compression=none") {
        bytes[at + 12..at + 16].copy_from_slice(b"zstd");
    }
    let mut bag = Bag::from_reader(Cursor::new(bytes)).unwrap();
    let err = bag.messages().next().unwrap().unwrap_err();
    assert!(matches!(err, Error::UnsupportedCompression(ref c) if c == "zstd"));
    assert!(err.to_string().contains("unsupported compression"));
}

#[test]
fn truncated_bags_error_without_panicking() {
    let bytes = sample_bag(Compression::Lz4);
    for len in (0..bytes.len()).step_by(37) {
        let r = Bag::from_reader(Cursor::new(bytes[..len].to_vec()));
        if let Ok(mut bag) = r {
            for m in bag.messages() {
                if m.is_err() {
                    break;
                }
            }
        }
    }
    // Corrupt chunk payload bytes: must yield errors, never panic.
    let mut bytes = sample_bag(Compression::Lz4);
    let at = find(&bytes, b"compression=lz4").unwrap() + 40;
    for b in &mut bytes[at..at + 16] {
        *b ^= 0xA5;
    }
    if let Ok(mut bag) = Bag::from_reader(Cursor::new(bytes)) {
        for m in bag.messages() {
            if m.is_err() {
                break;
            }
        }
    }
}

/// Bags written by the Python `rosbags` library (see `data/make_fixtures.py`).
fn check_python_fixture(name: &str, compression: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    let mut bag = Bag::open(path).unwrap();
    assert!(bag.chunks().len() > 1);
    let comps = bag.chunk_compressions().unwrap();
    assert_eq!(comps, vec![(compression.to_string(), bag.chunks().len())]);
    assert_eq!(bag.connections()[0].msg_type, "sensor_msgs/Imu");
    assert_eq!(bag.start_time(), Some(Time::new(1, 0)));
    assert_eq!(bag.end_time(), Some(Time::new(1, 390_000_000)));
    let mut n = 0;
    for m in bag.messages() {
        let m = m.unwrap();
        let imu: Imu = m.decode().unwrap();
        assert_eq!(imu.header.seq, n);
        assert_eq!(imu.header.stamp, m.time);
        assert_eq!(m.time, Time::new(1, n * 10_000_000));
        assert_eq!(imu.header.frame_id, "imu");
        assert_eq!(imu.angular_velocity.z, f64::from(n) * 0.01);
        assert_eq!(imu.linear_acceleration.z, 9.81);
        assert_eq!(imu.orientation.w, 1.0);
        n += 1;
    }
    assert_eq!(n, 40);
}

#[test]
fn python_rosbags_lz4_fixture() {
    check_python_fixture("py_imu_lz4.bag", "lz4");
}

#[cfg(feature = "bz2")]
#[test]
fn python_rosbags_bz2_fixture() {
    check_python_fixture("py_imu_bz2.bag", "bz2");
}

#[cfg(not(feature = "bz2"))]
#[test]
fn bz2_without_feature_is_unsupported() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/py_imu_bz2.bag");
    let mut bag = Bag::open(path).unwrap();
    let err = bag.messages().next().unwrap().unwrap_err();
    assert!(matches!(err, Error::UnsupportedCompression(ref c) if c == "bz2"));
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
