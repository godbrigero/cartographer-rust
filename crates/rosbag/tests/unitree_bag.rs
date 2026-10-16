//! Reads the simulated Unitree L1 bag written by the Python `rosbags` library
//! (`tools/unitree_l1_sim`). Ignored by default; run with
//!
//! ```text
//! UNITREE_L1_BAG=/path/to/unitree_l1_loop.bag \
//!   cargo test -p cartographer-rosbag --test unitree_bag -- --ignored --nocapture
//! ```
//!
//! Without `UNITREE_L1_BAG` the default location
//! `tools/unitree_l1_sim/out/unitree_l1_loop.bag` (repo root) is used; if the
//! file does not exist the test prints a note and passes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cartographer_rosbag::msgs::{Imu, PointCloud2, PoseStamped, TFMessage, point_field};
use cartographer_rosbag::{Bag, Time};

fn bag_path() -> PathBuf {
    std::env::var_os("UNITREE_L1_BAG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tools/unitree_l1_sim/out/unitree_l1_loop.bag")
        })
}

#[test]
#[ignore = "needs the generated Unitree L1 bag (set UNITREE_L1_BAG)"]
fn read_unitree_l1_bag() {
    let path = bag_path();
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let mut bag = Bag::open(&path).unwrap();
    println!(
        "{}: {} messages, {} chunks, {:?}, {} .. {}",
        path.display(),
        bag.message_count(),
        bag.chunks().len(),
        bag.chunk_compressions().unwrap(),
        bag.start_time().unwrap(),
        bag.end_time().unwrap()
    );
    for t in bag.topics() {
        println!("  {:<20} {:>6} x {}", t.topic, t.message_count, t.msg_type);
    }
    let expected_types = [
        ("/unilidar/cloud", "sensor_msgs/PointCloud2"),
        ("/unilidar/imu", "sensor_msgs/Imu"),
        ("/ground_truth/pose", "geometry_msgs/PoseStamped"),
        ("/tf_static", "tf2_msgs/TFMessage"),
    ];
    let topics = bag.topics();
    for (topic, ty) in expected_types {
        let t = topics.iter().find(|t| t.topic == topic).unwrap();
        assert_eq!(t.msg_type, ty);
        assert!(t.message_count > 0, "{topic} empty");
    }
    let index_counts: BTreeMap<String, u64> = topics
        .iter()
        .map(|t| (t.topic.clone(), t.message_count))
        .collect();

    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut last_record_time = Time::default();
    let mut last_stamp: BTreeMap<String, Time> = BTreeMap::new();
    let mut total_points = 0usize;
    let mut first_cloud_printed = false;
    let mut max_ring = 0u16;
    let mut imu_samples = 0;
    let mut gt_first = None;
    let mut gt_last = None;
    let mut tf_static = None;

    for m in bag.messages() {
        let m = m.unwrap();
        assert!(m.time >= last_record_time, "record times not monotonic");
        last_record_time = m.time;
        *counts.entry(m.topic().to_string()).or_default() += 1;
        let stamp = match m.topic() {
            "/unilidar/cloud" => {
                let c: PointCloud2 = m.decode().unwrap();
                assert_eq!(c.point_step, 32);
                let layout: Vec<(&str, u32, u8)> = c
                    .fields
                    .iter()
                    .map(|f| (f.name.as_str(), f.offset, f.datatype))
                    .collect();
                assert_eq!(
                    layout,
                    [
                        ("x", 0, point_field::FLOAT32),
                        ("y", 4, point_field::FLOAT32),
                        ("z", 8, point_field::FLOAT32),
                        ("intensity", 16, point_field::FLOAT32),
                        ("ring", 20, point_field::UINT16),
                        ("time", 24, point_field::FLOAT32),
                    ]
                );
                let pts = c.read_points().unwrap();
                assert_eq!(pts.len(), c.num_points());
                total_points += pts.len();
                for p in &pts {
                    assert!(p.position.iter().all(|v| v.is_finite()));
                    max_ring = max_ring.max(p.ring.unwrap());
                    let t = p.time.unwrap();
                    assert!((-0.001..1.0).contains(&t), "point time {t}");
                }
                if !first_cloud_printed {
                    first_cloud_printed = true;
                    println!(
                        "first cloud: stamp {} frame {} {}x{} points; p0 {:?}; p_last {:?}",
                        c.header.stamp,
                        c.header.frame_id,
                        c.height,
                        c.width,
                        pts.first(),
                        pts.last()
                    );
                }
                c.header.stamp
            }
            "/unilidar/imu" => {
                let imu: Imu = m.decode().unwrap();
                let q = imu.orientation;
                let n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
                let a = imu.linear_acceleration;
                let g = (a.x * a.x + a.y * a.y + a.z * a.z).sqrt();
                assert!((n - 1.0).abs() < 1e-6 || imu.orientation_covariance[0] == -1.0);
                assert!((5.0..15.0).contains(&g), "accel magnitude {g}");
                if imu_samples == 0 {
                    println!(
                        "first imu: stamp {} frame {} gyro {:?} acc {:?} |acc| {g:.3}",
                        imu.header.stamp,
                        imu.header.frame_id,
                        imu.angular_velocity.to_array(),
                        a.to_array()
                    );
                }
                imu_samples += 1;
                imu.header.stamp
            }
            "/ground_truth/pose" => {
                let p: PoseStamped = m.decode().unwrap();
                if gt_first.is_none() {
                    gt_first = Some(p.clone());
                }
                let s = p.header.stamp;
                gt_last = Some(p);
                s
            }
            "/tf_static" => {
                let tf: TFMessage = m.decode().unwrap();
                let s = tf.transforms[0].header.stamp;
                tf_static = Some(tf);
                s
            }
            other => panic!("unexpected topic {other}"),
        };
        let last = last_stamp.entry(m.topic().to_string()).or_default();
        assert!(stamp >= *last, "{} header stamps not monotonic", m.topic());
        *last = stamp;
    }

    assert_eq!(counts, index_counts, "iterated counts differ from index");
    println!("iterated counts: {counts:?}");
    println!(
        "total points: {total_points} (avg {:.1}/cloud), max ring {max_ring}",
        total_points as f64 / counts["/unilidar/cloud"] as f64
    );
    let (a, b) = (gt_first.unwrap(), gt_last.unwrap());
    println!(
        "ground truth: first {} {:?}, last {} {:?}",
        a.header.stamp,
        a.pose.position.to_array(),
        b.header.stamp,
        b.pose.position.to_array()
    );
    let tf = tf_static.unwrap();
    for t in &tf.transforms {
        println!(
            "tf_static: {} -> {} t={:?} q=({:.4}, {:.4}, {:.4}, {:.4})",
            t.header.frame_id,
            t.child_frame_id,
            t.transform.translation.to_array(),
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w
        );
    }
    assert!(
        tf.transforms
            .iter()
            .any(|t| t.header.frame_id == "unilidar_imu" && t.child_frame_id == "unilidar_lidar")
    );

    // Topic filtering yields exactly the IMU messages.
    let n = bag
        .messages_for_topics(&["/unilidar/imu"])
        .unwrap()
        .map(|m| assert_eq!(m.unwrap().topic(), "/unilidar/imu"))
        .count();
    assert_eq!(n as u64, counts["/unilidar/imu"]);
}
