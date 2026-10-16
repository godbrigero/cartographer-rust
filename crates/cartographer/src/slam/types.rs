//! Plain data types of the high-level API: modes, extrinsics, sensor
//! samples, results.

use std::fmt;

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};

use super::time::Timestamp;
use crate::mapping::id::{NodeId, SubmapId};
use crate::transform::rigid_transform::Rigid3d;

/// 2D (planar LiDAR, occupancy grid) or 3D (multi-beam LiDAR + IMU, hybrid
/// voxel grids) SLAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Cartographer's 2D pipeline (`use_trajectory_builder_2d`). IMU and
    /// odometry are optional.
    TwoD,
    /// Cartographer's 3D pipeline (`use_trajectory_builder_3d`). Requires an
    /// IMU.
    ThreeD,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Mode::TwoD => "2D",
            Mode::ThreeD => "3D",
        })
    }
}

/// The kind of a declared sensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensorKind {
    /// A range sensor (2D or 3D LiDAR, depth camera, ...).
    Lidar,
    /// An inertial measurement unit.
    Imu,
    /// Wheel (or any other) odometry.
    Odometry,
}

impl fmt::Display for SensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SensorKind::Lidar => "LiDAR",
            SensorKind::Imu => "IMU",
            SensorKind::Odometry => "odometry",
        })
    }
}

/// Mounting pose of a sensor: the transform **tracking_from_sensor**, i.e.
/// the pose of the sensor frame expressed in the tracking frame (the body
/// frame whose trajectory is estimated; in 3D it should be the IMU frame).
///
/// This is what `tf2_echo tracking_frame sensor_frame` prints in ROS, and
/// what URDF joints from the tracking link to the sensor link describe.
/// Sensor data are transformed into the tracking frame with it, exactly like
/// cartographer_ros does with TF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrinsics {
    /// `tracking_from_sensor`.
    pub tracking_from_sensor: Isometry3<f64>,
}

impl Extrinsics {
    /// The sensor frame is the tracking frame.
    pub fn identity() -> Self {
        Extrinsics {
            tracking_from_sensor: Isometry3::identity(),
        }
    }

    /// From an `Isometry3` (tracking_from_sensor).
    pub fn from_isometry(tracking_from_sensor: Isometry3<f64>) -> Self {
        Extrinsics {
            tracking_from_sensor,
        }
    }

    /// A pure translation: the sensor origin at `(x, y, z)` meters in the
    /// tracking frame, axes aligned.
    pub fn from_translation(x: f64, y: f64, z: f64) -> Self {
        Self::from_isometry(Isometry3::translation(x, y, z))
    }

    /// Translation `(x, y, z)` (m) and rotation as roll, pitch, yaw (rad,
    /// extrinsic X-Y-Z = `R = Rz(yaw) * Ry(pitch) * Rx(roll)`, the ROS/URDF
    /// convention).
    pub fn from_xyz_rpy(x: f64, y: f64, z: f64, roll: f64, pitch: f64, yaw: f64) -> Self {
        Self::from_isometry(Isometry3::from_parts(
            Translation3::new(x, y, z),
            UnitQuaternion::from_euler_angles(roll, pitch, yaw),
        ))
    }

    /// Translation (m) and rotation quaternion.
    pub fn from_translation_rotation(
        translation: Vector3<f64>,
        rotation: UnitQuaternion<f64>,
    ) -> Self {
        Self::from_isometry(Isometry3::from_parts(translation.into(), rotation))
    }
}

impl Default for Extrinsics {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<Isometry3<f64>> for Extrinsics {
    fn from(tracking_from_sensor: Isometry3<f64>) -> Self {
        Self::from_isometry(tracking_from_sensor)
    }
}

/// One IMU measurement in the IMU frame (SI units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuSample {
    /// Specific force in m/s² (what accelerometers measure): a level IMU at
    /// rest reads about `(0, 0, +9.81)`.
    pub linear_acceleration: Vector3<f64>,
    /// Angular velocity in rad/s.
    pub angular_velocity: Vector3<f64>,
}

/// A LiDAR return with its acquisition time.
///
/// # Time convention (Cartographer's)
///
/// `time` is in seconds **relative to the message timestamp** passed to
/// [`Slam::push_lidar`](super::Slam::push_lidar), and the message timestamp
/// is the acquisition time of the **last** point. Hence the last point has
/// `time ≈ 0`, earlier points negative times, and times must be
/// non-decreasing in the slice. Cartographer uses these times to undistort
/// the scan with its motion estimate.
///
/// Many drivers (Velodyne, Ouster, Unitree, Livox, ...) instead stamp the
/// message with the **first** point and give per-point offsets `>= 0` from
/// it. Convert with [`points_relative_to_first`]; for absolute per-point
/// times use [`points_with_absolute_times`]. If a driver provides no
/// per-point times, use `time = 0` for all points
/// ([`Slam::push_lidar_untimed`](super::Slam::push_lidar_untimed)).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TimedPoint {
    /// Position in the sensor frame (m).
    pub position: Vector3<f32>,
    /// Seconds relative to the message timestamp (`<= 0`).
    pub time: f32,
}

impl TimedPoint {
    /// A point at `(x, y, z)` with time offset `time`.
    pub fn new(x: f32, y: f32, z: f32, time: f32) -> Self {
        TimedPoint {
            position: Vector3::new(x, y, z),
            time,
        }
    }
}

/// Converts a scan whose stamp is the time of the **first** point and whose
/// per-point offsets (seconds, `>= 0`) are relative to it (Velodyne, Unitree
/// L1/L2, Ouster, Livox drivers) into Cartographer's convention.
///
/// Returns the message timestamp to pass to
/// [`Slam::push_lidar`](super::Slam::push_lidar) and the points with times
/// relative to it. Points are stably sorted by time first; the returned
/// timestamp is `stamp + max(offset)` and the latest point gets time `0`
/// (cartographer_ros does the same shift for `PointCloud2` messages with a
/// `time` field).
pub fn points_relative_to_first(
    stamp: impl Into<Timestamp>,
    points: impl IntoIterator<Item = (Vector3<f32>, f32)>,
) -> (Timestamp, Vec<TimedPoint>) {
    let stamp = stamp.into();
    let mut points: Vec<(Vector3<f32>, f64)> = points
        .into_iter()
        .map(|(position, offset)| (position, f64::from(offset)))
        .collect();
    points.sort_by(|a, b| a.1.total_cmp(&b.1));
    let Some(&(_, last_offset)) = points.last() else {
        return (stamp, Vec::new());
    };
    let message_time = stamp.add_secs_f64(last_offset);
    let points = points
        .into_iter()
        .map(|(position, offset)| TimedPoint {
            position,
            time: (offset - last_offset) as f32,
        })
        .collect();
    (message_time, points)
}

/// Converts points with **absolute** acquisition times into Cartographer's
/// convention: returns the time of the latest point as message timestamp and
/// the points (stably sorted by time) with times relative to it.
/// Returns `None` for an empty input.
pub fn points_with_absolute_times(
    points: impl IntoIterator<Item = (Vector3<f32>, Timestamp)>,
) -> Option<(Timestamp, Vec<TimedPoint>)> {
    let mut points: Vec<(Vector3<f32>, Timestamp)> = points.into_iter().collect();
    points.sort_by_key(|p| p.1);
    let &(_, last) = points.last()?;
    Some((
        last,
        points
            .into_iter()
            .map(|(position, time)| TimedPoint {
                position,
                time: time.seconds_since(last) as f32,
            })
            .collect(),
    ))
}

/// A pose with its timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StampedPose {
    /// Time of the pose.
    pub time: Timestamp,
    /// Pose of the tracking frame (map_from_tracking for global poses,
    /// local_from_tracking for local poses).
    pub pose: Isometry3<f64>,
}

/// Summary of one submap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapInfo {
    /// Trajectory and index of the submap.
    pub id: SubmapId,
    /// Optimized global pose of the submap frame.
    pub global_pose: Isometry3<f64>,
    /// Number of range data (scans after accumulation) inserted so far.
    pub num_range_data: i32,
    /// True once the submap no longer changes (it is then used for loop
    /// closure).
    pub finished: bool,
}

/// Counters describing the pose graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlamStats {
    /// Nodes (scans kept by the motion filter) in all trajectories.
    pub num_nodes: usize,
    /// Submaps in all trajectories.
    pub num_submaps: usize,
    /// All constraints.
    pub num_constraints: usize,
    /// Constraints of nodes to the submaps they were inserted into (from
    /// local SLAM).
    pub num_intra_submap_constraints: usize,
    /// Constraints found by scan matching a node against a submap it was not
    /// inserted into (loop closure search).
    pub num_inter_submap_constraints: usize,
    /// The subset of inter-submap constraints of a trajectory against its own
    /// submaps that are older than the submap preceding the node's first
    /// insertion submap, i.e. constraints that genuinely close a loop.
    pub num_loop_closure_constraints: usize,
    /// Inter-submap constraints between different trajectories (e.g. a
    /// localization trajectory against a loaded map).
    pub num_inter_trajectory_constraints: usize,
    /// Local SLAM results so far (accumulated scans that were matched).
    pub num_local_results: usize,
    /// Global optimizations that ran so far (counted in the global
    /// optimization callback).
    pub num_global_optimizations: usize,
}

/// Result of local SLAM for one accumulated scan, passed to
/// [`Slam::on_local_result`](super::Slam::on_local_result) callbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalResult {
    /// Trajectory id.
    pub trajectory_id: i32,
    /// Time of the scan.
    pub time: Timestamp,
    /// Pose of the tracking frame in the local SLAM frame (not loop-closed).
    pub local_pose: Isometry3<f64>,
    /// The node id if the scan was inserted into the pose graph, `None` if
    /// the motion filter dropped it.
    pub node_id: Option<NodeId>,
    /// The scan's returns (after filtering) in the local SLAM frame.
    pub points_in_local: Vec<Vector3<f32>>,
}

/// Information passed to
/// [`Slam::on_global_optimization`](super::Slam::on_global_optimization)
/// callbacks after the pose graph was optimized.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationEvent {
    /// Last submap included in the optimization, per trajectory.
    pub last_optimized_submaps: std::collections::BTreeMap<i32, SubmapId>,
    /// Last node included in the optimization, per trajectory.
    pub last_optimized_nodes: std::collections::BTreeMap<i32, NodeId>,
}

/// `Rigid3d` -> `Isometry3<f64>`.
pub(crate) fn to_isometry(rigid: &Rigid3d) -> Isometry3<f64> {
    let q = rigid.rotation();
    Isometry3::from_parts(
        Translation3::from(*rigid.translation()),
        UnitQuaternion::new_normalize(*q),
    )
}

/// `Isometry3<f64>` -> `Rigid3d`.
pub(crate) fn to_rigid3d(isometry: &Isometry3<f64>) -> Rigid3d {
    let q = isometry.rotation.quaternion();
    Rigid3d::new(
        isometry.translation.vector,
        Quaternion::new(q.w, q.i, q.j, q.k),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rigid_isometry_round_trip() {
        let iso = Extrinsics::from_xyz_rpy(1.0, -2.0, 0.5, 0.1, -0.2, 0.3).tracking_from_sensor;
        let rigid = to_rigid3d(&iso);
        let back = to_isometry(&rigid);
        assert_relative_eq!(back, iso, epsilon = 1e-12);
        let p = Vector3::new(0.3, 0.2, 0.1);
        assert_relative_eq!(
            rigid * p,
            (iso * nalgebra::Point3::from(p)).coords,
            epsilon = 1e-12
        );
    }

    #[test]
    fn rpy_convention_is_ros() {
        // Yaw of 90 degrees maps x to y.
        let e = Extrinsics::from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2);
        assert_relative_eq!(
            e.tracking_from_sensor * Vector3::x(),
            Vector3::y(),
            epsilon = 1e-12
        );
        // Roll of 90 degrees maps y to z.
        let e = Extrinsics::from_xyz_rpy(0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        assert_relative_eq!(
            e.tracking_from_sensor * Vector3::y(),
            Vector3::z(),
            epsilon = 1e-12
        );
        assert_eq!(Extrinsics::default(), Extrinsics::identity());
        assert_eq!(
            Extrinsics::from_translation(1.0, 2.0, 3.0)
                .tracking_from_sensor
                .translation
                .vector,
            Vector3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn relative_to_first_shifts_to_last_point() {
        let (time, points) = points_relative_to_first(
            10.0,
            [
                (Vector3::new(1.0, 0.0, 0.0), 0.0_f32),
                (Vector3::new(3.0, 0.0, 0.0), 0.125),
                (Vector3::new(2.0, 0.0, 0.0), 0.0625),
            ],
        );
        assert_eq!(time, Timestamp::from_secs_f64(10.125));
        let xs: Vec<f32> = points.iter().map(|p| p.position.x).collect();
        assert_eq!(xs, [1.0, 2.0, 3.0]);
        assert!((points[0].time + 0.125).abs() < 1e-6);
        assert!((points[1].time + 0.0625).abs() < 1e-6);
        assert_eq!(points[2].time, 0.0);
        let (time, points) = points_relative_to_first(4.0, std::iter::empty());
        assert_eq!(time, Timestamp::from_secs_f64(4.0));
        assert!(points.is_empty());
    }

    #[test]
    fn absolute_times() {
        let (time, points) = points_with_absolute_times([
            (Vector3::new(2.0, 0.0, 0.0), Timestamp::from_secs_f64(5.5)),
            (Vector3::new(1.0, 0.0, 0.0), Timestamp::from_secs_f64(5.0)),
        ])
        .unwrap();
        assert_eq!(time, Timestamp::from_secs_f64(5.5));
        assert_eq!(points[0].position.x, 1.0);
        assert!((points[0].time + 0.5).abs() < 1e-6);
        assert_eq!(points[1].time, 0.0);
        assert!(points_with_absolute_times(std::iter::empty()).is_none());
    }
}
