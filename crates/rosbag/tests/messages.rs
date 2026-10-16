//! Golden decode tests. The hex constants were produced independently by the
//! Python `rosbags` library (`get_typestore(Stores.ROS1_NOETIC)`,
//! `serialize_ros1`); `HEADER_BYTES` is assembled by hand.

mod common;

use cartographer_rosbag::msgs::{
    CloudPoint, Header, Imu, LaserScan, Odometry, PointCloud2, PoseStamped, TFMessage, point_field,
};
use cartographer_rosbag::{Error, RosMsg, Time};
use common::Ser;

const HEADER: &str = "0700000000f1536515cd5b07030000006d6170";
const POSE_STAMPED: &str = "010000000a000000f401000005000000776f726c64000000000000f83f00000000000000c0000000000000d03f00000000000000000000000000000000333333333333e33f9a9999999999e93f";
const IMU: &str = "020000000b0000000000000003000000696d759a9999999999b93f9a9999999999c93f333333333333d33fcdccccccccccec3f000000000000f0bf000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007b14ae47e17a843f7b14ae47e17a94bf000000000000e03f00000000000000009a9999999999b93f9a9999999999c93f343333333333d33f9a9999999999d93f000000000000e03f343333333333e33f676666666666e63f9a9999999999e93f9a9999999999b93f9a9999999999c93f1f85eb51b89e234000000000000000009a9999999999c93f9a9999999999d93f343333333333e33f9a9999999999e93f000000000000f03f343333333333f33f676666666666f63f9a9999999999f93f";
const ODOMETRY: &str = "030000000c00000001000000040000006f646f6d09000000626173655f6c696e6b000000000000f83f00000000000000c0000000000000d03f00000000000000000000000000000000333333333333e33f9a9999999999e93f0000000000000000000000000000f03f000000000000004000000000000008400000000000001040000000000000144000000000000018400000000000001c40000000000000204000000000000022400000000000002440000000000000264000000000000028400000000000002a400000000000002c400000000000002e4000000000000030400000000000003140000000000000324000000000000033400000000000003440000000000000354000000000000036400000000000003740000000000000384000000000000039400000000000003a400000000000003b400000000000003c400000000000003d400000000000003e400000000000003f400000000000004040000000000080404000000000000041400000000000804140000000000000f03f0000000000000000000000000000000000000000000000000000000000000000000000000000e03f0000000000000080000000000000f0bf00000000000000c000000000000008c000000000000010c000000000000014c000000000000018c00000000000001cc000000000000020c000000000000022c000000000000024c000000000000026c000000000000028c00000000000002ac00000000000002cc00000000000002ec000000000000030c000000000000031c000000000000032c000000000000033c000000000000034c000000000000035c000000000000036c000000000000037c000000000000038c000000000000039c00000000000003ac00000000000003bc00000000000003cc00000000000003dc00000000000003ec00000000000003fc000000000000040c000000000008040c000000000000041c000000000008041c0";
const LASER_SCAN: &str = "040000000d00000002000000050000006c617365720000c0bf0000c03f0000803f6f12833acdcccc3dcdcccc3d0000f041040000000000803f000020400000807f0000804000000000";
const POINT_CLOUD2: &str = "050000000f000000040000000e000000756e696c696461725f6c6964617201000000020000000600000001000000780000000007010000000100000079040000000701000000010000007a08000000070100000009000000696e74656e736974791000000007010000000400000072696e671400000004010000000400000074696d65180000000701000000002000000040000000400000000000803f00000040000040400000000000002041000000000000000000000000000060c00000803e000000be0000000000007f43110000000ad7233b0000000001";
const TF: &str = "02000000000000000e0000000300000009000000626173655f6c696e6b03000000696d759a9999999999b93f00000000000000009a9999999999c93f000000000000000000000000000000000000000000000000000000000000f03f000000000e0000000300000009000000626173655f6c696e6b050000006c6964617200000000000000009a9999999999a9bf333333333333d33f000000000000000000000000000000000000000000000000000000000000f03f";

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// seq=7, stamp=1700000000.123456789, frame_id="map".
const HEADER_BYTES: [u8; 19] = [
    0x07, 0x00, 0x00, 0x00, // seq
    0x00, 0xf1, 0x53, 0x65, // sec = 1_700_000_000
    0x15, 0xcd, 0x5b, 0x07, // nsec = 123_456_789
    0x03, 0x00, 0x00, 0x00, b'm', b'a', b'p', // frame_id
];

#[test]
fn header_golden() {
    assert_eq!(hex(HEADER), HEADER_BYTES);
    let h = Header::decode(&HEADER_BYTES).unwrap();
    assert_eq!(
        h,
        Header {
            seq: 7,
            stamp: Time::new(1_700_000_000, 123_456_789),
            frame_id: "map".into(),
        }
    );
}

#[test]
fn pose_stamped_golden() {
    let m = PoseStamped::decode(&hex(POSE_STAMPED)).unwrap();
    assert_eq!(m.header.seq, 1);
    assert_eq!(m.header.stamp, Time::new(10, 500));
    assert_eq!(m.header.frame_id, "world");
    assert_eq!(m.pose.position.to_array(), [1.5, -2.0, 0.25]);
    let q = m.pose.orientation;
    assert_eq!([q.x, q.y, q.z, q.w], [0.0, 0.0, 0.6, 0.8]);
}

#[test]
fn imu_golden() {
    let m = Imu::decode(&hex(IMU)).unwrap();
    assert_eq!(m.header.frame_id, "imu");
    assert_eq!(m.header.stamp, Time::new(11, 0));
    let q = m.orientation;
    assert_eq!([q.x, q.y, q.z, q.w], [0.1, 0.2, 0.3, 0.9]);
    assert_eq!(m.orientation_covariance[0], -1.0);
    assert_eq!(m.angular_velocity.to_array(), [0.01, -0.02, 0.5]);
    assert_eq!(m.linear_acceleration.to_array(), [0.1, 0.2, 9.81]);
    for i in 0..9 {
        assert_eq!(m.angular_velocity_covariance[i], i as f64 * 0.1);
        assert_eq!(m.linear_acceleration_covariance[i], i as f64 * 0.1 * 2.0);
    }
}

#[test]
fn odometry_golden() {
    let m = Odometry::decode(&hex(ODOMETRY)).unwrap();
    assert_eq!(m.header.frame_id, "odom");
    assert_eq!(m.header.stamp, Time::new(12, 1));
    assert_eq!(m.child_frame_id, "base_link");
    assert_eq!(m.pose.pose.position.to_array(), [1.5, -2.0, 0.25]);
    assert_eq!(m.pose.pose.orientation.w, 0.8);
    assert_eq!(m.twist.twist.linear.to_array(), [1.0, 0.0, 0.0]);
    assert_eq!(m.twist.twist.angular.to_array(), [0.0, 0.0, 0.5]);
    for i in 0..36 {
        assert_eq!(m.pose.covariance[i], i as f64);
        assert_eq!(m.twist.covariance[i], -(i as f64));
    }
}

#[test]
fn laser_scan_golden() {
    let m = LaserScan::decode(&hex(LASER_SCAN)).unwrap();
    assert_eq!(m.header.frame_id, "laser");
    assert_eq!(
        (m.angle_min, m.angle_max, m.angle_increment),
        (-1.5, 1.5, 1.0)
    );
    assert_eq!(m.time_increment, 0.001);
    assert_eq!(m.scan_time, 0.1);
    assert_eq!((m.range_min, m.range_max), (0.1, 30.0));
    assert_eq!(m.ranges, vec![1.0, 2.5, f32::INFINITY, 4.0]);
    assert!(m.intensities.is_empty());
}

#[test]
fn tf_message_golden() {
    let m = TFMessage::decode(&hex(TF)).unwrap();
    assert_eq!(m.transforms.len(), 2);
    let t0 = &m.transforms[0];
    assert_eq!(t0.header.frame_id, "base_link");
    assert_eq!(t0.header.stamp, Time::new(14, 3));
    assert_eq!(t0.child_frame_id, "imu");
    assert_eq!(t0.transform.translation.to_array(), [0.1, 0.0, 0.2]);
    assert_eq!(t0.transform.rotation.w, 1.0);
    let t1 = &m.transforms[1];
    assert_eq!(t1.child_frame_id, "lidar");
    assert_eq!(t1.transform.translation.to_array(), [0.0, -0.05, 0.3]);
}

#[test]
fn point_cloud2_golden() {
    let m = PointCloud2::decode(&hex(POINT_CLOUD2)).unwrap();
    assert_eq!(m.header.frame_id, "unilidar_lidar");
    assert_eq!(
        (m.height, m.width, m.point_step, m.row_step),
        (1, 2, 32, 64)
    );
    assert!(!m.is_bigendian && m.is_dense);
    let names: Vec<_> = m.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["x", "y", "z", "intensity", "ring", "time"]);
    let ring = m.field("ring").unwrap();
    assert_eq!(
        (ring.offset, ring.datatype, ring.count),
        (20, point_field::UINT16, 1)
    );
    assert_eq!(m.num_points(), 2);

    let pts = m.read_points().unwrap();
    assert_eq!(
        pts,
        vec![
            CloudPoint {
                position: [1.0, 2.0, 3.0],
                intensity: Some(10.0),
                ring: Some(0),
                time: Some(0.0),
            },
            CloudPoint {
                position: [-3.5, 0.25, -0.125],
                intensity: Some(255.0),
                ring: Some(17),
                time: Some(f64::from(0.0025f32)),
            },
        ]
    );
    assert_eq!(m.read_field("ring").unwrap(), Some(vec![0.0, 17.0]));
    assert_eq!(m.read_field("nope").unwrap(), None);
}

/// Builds a PointCloud2 with the given fields and raw point data.
fn cloud(
    fields: &[(&str, u32, u8)],
    big_endian: bool,
    height: u32,
    width: u32,
    point_step: u32,
    row_step: u32,
    data: &[u8],
) -> Vec<u8> {
    let mut s = Ser::default();
    s.header(0, 1, 2, "c")
        .u32(height)
        .u32(width)
        .u32(fields.len() as u32);
    for &(name, offset, dt) in fields {
        s.string(name).u32(offset).u8(dt).u32(1);
    }
    s.u8(big_endian as u8)
        .u32(point_step)
        .u32(row_step)
        .u32(data.len() as u32)
        .bytes(data)
        .u8(1);
    s.take()
}

#[test]
fn point_cloud2_mixed_datatypes_and_row_padding() {
    // x: f64 @0, y: i16 @8, z: u8 @10, intensity: i32 @12, ring: u8 @11,
    // time: u32 @16 (e.g. nanoseconds). point_step 20; rows padded by 4 bytes.
    let fields = [
        ("x", 0, point_field::FLOAT64),
        ("y", 8, point_field::INT16),
        ("z", 10, point_field::UINT8),
        ("ring", 11, point_field::UINT8),
        ("intensity", 12, point_field::INT32),
        ("time", 16, point_field::UINT32),
    ];
    let point = |x: f64, y: i16, z: u8, ring: u8, i: i32, t: u32| {
        let mut s = Ser::default();
        s.f64(x).u16(y as u16).u8(z).u8(ring).u32(i as u32).u32(t);
        s.take()
    };
    let mut data = Vec::new();
    for r in 0..2u8 {
        data.extend(point(0.5 + f64::from(r), -7, 3, r, -100, 1000));
        data.extend(point(-2.0, 300, 255, 9, 42, 2000 + u32::from(r)));
        data.extend([0xEE; 4]); // Row padding.
    }
    let bytes = cloud(&fields, false, 2, 2, 20, 44, &data);
    let m = PointCloud2::decode(&bytes).unwrap();
    let pts = m.read_points().unwrap();
    assert_eq!(pts.len(), 4);
    assert_eq!(pts[0].position, [0.5, -7.0, 3.0]);
    assert_eq!(pts[0].intensity, Some(-100.0));
    assert_eq!(pts[0].ring, Some(0));
    assert_eq!(pts[0].time, Some(1000.0));
    assert_eq!(pts[1].position, [-2.0, 300.0, 255.0]);
    assert_eq!(pts[2].position, [1.5, -7.0, 3.0]);
    assert_eq!(pts[2].ring, Some(1));
    assert_eq!(pts[3].time, Some(2001.0));
}

#[test]
fn point_cloud2_big_endian() {
    let fields = [
        ("x", 0, point_field::FLOAT32),
        ("y", 4, point_field::FLOAT32),
        ("z", 8, point_field::FLOAT32),
        ("ring", 12, point_field::UINT16),
    ];
    let mut data = Vec::new();
    data.extend(1.25f32.to_be_bytes());
    data.extend((-2.0f32).to_be_bytes());
    data.extend(8.0f32.to_be_bytes());
    data.extend(513u16.to_be_bytes());
    data.extend([0, 0]);
    let m = PointCloud2::decode(&cloud(&fields, true, 1, 1, 16, 16, &data)).unwrap();
    let p = m.read_points().unwrap()[0];
    assert_eq!(p.position, [1.25, -2.0, 8.0]);
    assert_eq!(p.ring, Some(513));
    assert_eq!(p.intensity, None);
    assert_eq!(p.time, None);
}

#[test]
fn point_cloud2_layout_errors() {
    // Field extends past point_step.
    let bad = cloud(
        &[("x", 6, point_field::FLOAT32)],
        false,
        1,
        1,
        8,
        8,
        &[0; 8],
    );
    let m = PointCloud2::decode(&bad).unwrap();
    assert!(matches!(m.read_field("x"), Err(Error::BadPointField(_))));
    // Unknown datatype.
    let bad = cloud(&[("x", 0, 42)], false, 1, 1, 8, 8, &[0; 8]);
    let m = PointCloud2::decode(&bad).unwrap();
    assert!(matches!(m.read_field("x"), Err(Error::BadPointField(_))));
    // Data shorter than width * point_step.
    let xyz = [
        ("x", 0, point_field::FLOAT32),
        ("y", 4, point_field::FLOAT32),
        ("z", 8, point_field::FLOAT32),
    ];
    let m = PointCloud2::decode(&cloud(&xyz, false, 1, 2, 12, 24, &[0; 20])).unwrap();
    assert!(matches!(m.read_points(), Err(Error::BadPointField(_))));
    // Missing z.
    let m = PointCloud2::decode(&cloud(&xyz[..2], false, 1, 1, 12, 12, &[0; 12])).unwrap();
    assert!(matches!(m.read_points(), Err(Error::BadPointField(_))));
}

#[test]
fn truncation_and_trailing_bytes() {
    let full = hex(IMU);
    for len in 0..full.len() {
        assert!(Imu::decode(&full[..len]).is_err(), "len {len}");
    }
    let mut extra = full.clone();
    extra.push(0);
    assert!(matches!(Imu::decode(&extra), Err(Error::TrailingBytes(1))));

    // Huge array length prefix must not allocate or panic.
    let mut s = Ser::default();
    s.header(0, 0, 0, "").u32(u32::MAX);
    assert!(matches!(
        TFMessage::decode(&s.take()[16..]),
        Err(Error::UnexpectedEof { .. })
    ));
    // Invalid UTF-8 frame id.
    let mut s = Ser::default();
    s.u32(0).u32(0).u32(0).u32(2).bytes(&[0xff, 0xfe]);
    assert!(matches!(Header::decode(&s.take()), Err(Error::InvalidUtf8)));
}
