//! Simulated sensors and the plain message types they produce.
//!
//! Every sensor has an `extrinsic` = `tracking_from_sensor` (default
//! identity) and produces data **in its own sensor frame**; consumers apply
//! the extrinsic (exactly like cartographer_ros does with TF before calling
//! `AddSensorData`).
//!
//! Every sensor draws noise from its own ChaCha8 stream seeded from the
//! dataset seed and the sensor id (see [`sensor_rng`]), so adding or removing
//! a sensor never changes another sensor's data. Noise samples are drawn even
//! when the configured standard deviation is zero, so changing a noise level
//! does not shift the stream.

mod imu;
mod lidar;
mod pose_sensors;

pub use imu::ImuConfig;
pub use lidar::{Lidar2dConfig, Lidar3dConfig, MissHandling, ScanTiming};
pub use pose_sensors::{FixedFramePoseConfig, Landmark, LandmarkConfig, OdometryConfig};

use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// A range point with a relative time stamp (Cartographer's
/// `TimedRangefinderPoint`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedPoint {
    /// Point position in the sensor frame (m).
    pub position: Vector3<f32>,
    /// Time relative to the message time (s); `<= 0`, the last point is `0`.
    pub time: f32,
}

/// A (possibly motion-distorted) point cloud, Cartographer's
/// `TimedPointCloudData` in the sensor frame.
///
/// # Timing convention (matches Cartographer exactly)
/// * `time` is the absolute time of the **last** point in `points`.
/// * Each point's `time` is relative to `time`: the last point has `0.0`,
///   earlier points have negative offsets, non-decreasing along `points`.
/// * Each point was measured from the sensor pose at `time + point.time`.
/// * For an instantaneous scanner all offsets are `0` and `time` is the scan
///   time. If a scan has no points, `time` is the nominal time of its last
///   beam.
///
/// (Cartographer's `LocalTrajectoryBuilder{2D,3D}::AddRangeData` CHECKs that
/// the last point's time is `<= 0` and extrapolates a pose for every point at
/// `time + point.time`; cartographer_ros shifts the message stamp to the last
/// point the same way.)
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloudMessage {
    /// Sensor origin in the sensor frame (always zero; after applying the
    /// extrinsic it becomes the `origin` Cartographer expects).
    pub origin: Vector3<f32>,
    /// Returns in scan order.
    pub points: Vec<TimedPoint>,
    /// Per-point intensities (same length as `points`) or empty.
    pub intensities: Vec<f32>,
    /// Per-point flag (same length as `points`): `true` if this point is a
    /// synthetic "no return" point placed at [`MissHandling::Return`]'s range.
    /// Feed such points to Cartographer as ordinary points beyond its
    /// `max_range`; it then turns them into misses of length
    /// `missing_data_ray_length`.
    pub misses: Vec<bool>,
}

/// An IMU sample in the IMU frame (Cartographer's `ImuData`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuMessage {
    /// Specific force (m/s^2): kinematic acceleration minus gravity, so a
    /// level IMU at rest reads `(0, 0, +g)`.
    pub linear_acceleration: Vector3<f64>,
    /// Angular velocity (rad/s).
    pub angular_velocity: Vector3<f64>,
}

/// An odometry pose (Cartographer's `OdometryData`): pose of the odometry
/// sensor frame in the odometry frame, with accumulated drift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdometryMessage {
    /// `odom_from_sensor`.
    pub pose: Isometry3<f64>,
}

/// A fixed-frame (GPS-like) pose (Cartographer's `FixedFramePoseData`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedFramePoseMessage {
    /// `fixed_frame_from_sensor`, if available.
    pub pose: Option<Isometry3<f64>>,
}

/// One landmark observation (Cartographer's `LandmarkObservation`, but
/// relative to the sensor frame).
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkObservation {
    /// Landmark id.
    pub id: String,
    /// `sensor_from_landmark` (the landmark pose in the sensor frame).
    /// Cartographer's `landmark_to_tracking_transform` is
    /// `extrinsic * landmark_to_sensor`.
    pub landmark_to_sensor: Isometry3<f64>,
    /// Translation weight to report to Cartographer.
    pub translation_weight: f64,
    /// Rotation weight to report to Cartographer.
    pub rotation_weight: f64,
}

/// A set of landmark observations taken at one time.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkMessage {
    /// Observations (never empty).
    pub observations: Vec<LandmarkObservation>,
}

/// Payload of a [`crate::dataset::SensorMessage`].
#[derive(Clone, Debug, PartialEq)]
pub enum SensorData {
    /// From a 2D or 3D LiDAR.
    PointCloud(PointCloudMessage),
    /// From an IMU.
    Imu(ImuMessage),
    /// From odometry.
    Odometry(OdometryMessage),
    /// From a fixed-frame pose sensor.
    FixedFramePose(FixedFramePoseMessage),
    /// From a landmark detector.
    Landmark(LandmarkMessage),
}

/// Kind of a simulated sensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensorKind {
    /// Planar LiDAR.
    Lidar2d,
    /// Multi-beam rotating LiDAR.
    Lidar3d,
    /// IMU.
    Imu,
    /// Wheel/visual odometry.
    Odometry,
    /// GPS-like fixed-frame pose.
    FixedFramePose,
    /// Landmark detector.
    Landmark,
}

/// Configuration of any sensor (see the individual config types).
#[derive(Clone, Debug, PartialEq)]
pub enum SensorConfig {
    /// See [`Lidar2dConfig`].
    Lidar2d(Lidar2dConfig),
    /// See [`Lidar3dConfig`].
    Lidar3d(Lidar3dConfig),
    /// See [`ImuConfig`].
    Imu(ImuConfig),
    /// See [`OdometryConfig`].
    Odometry(OdometryConfig),
    /// See [`FixedFramePoseConfig`].
    FixedFramePose(FixedFramePoseConfig),
    /// See [`LandmarkConfig`].
    Landmark(LandmarkConfig),
}

impl SensorConfig {
    /// The sensor kind.
    pub fn kind(&self) -> SensorKind {
        match self {
            Self::Lidar2d(_) => SensorKind::Lidar2d,
            Self::Lidar3d(_) => SensorKind::Lidar3d,
            Self::Imu(_) => SensorKind::Imu,
            Self::Odometry(_) => SensorKind::Odometry,
            Self::FixedFramePose(_) => SensorKind::FixedFramePose,
            Self::Landmark(_) => SensorKind::Landmark,
        }
    }

    /// `tracking_from_sensor`.
    pub fn extrinsic(&self) -> Isometry3<f64> {
        match self {
            Self::Lidar2d(c) => c.extrinsic,
            Self::Lidar3d(c) => c.extrinsic,
            Self::Imu(c) => c.extrinsic,
            Self::Odometry(c) => c.extrinsic,
            Self::FixedFramePose(c) => c.extrinsic,
            Self::Landmark(c) => c.extrinsic,
        }
    }
}

/// Deterministic per-sensor RNG: ChaCha8 seeded with
/// `seed ^ fnv1a64(sensor_id)`.
pub fn sensor_rng(seed: u64, sensor_id: &str) -> ChaCha8Rng {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in sensor_id.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    ChaCha8Rng::seed_from_u64(seed ^ h)
}

/// Zero-mean Gaussian sample with standard deviation `std` (always consumes
/// one normal sample).
pub(crate) fn gauss(rng: &mut impl Rng, std: f64) -> f64 {
    let z: f64 = rng.sample(StandardNormal);
    z * std
}

/// Isotropic Gaussian 3-vector.
pub(crate) fn gauss3(rng: &mut impl Rng, std: f64) -> Vector3<f64> {
    Vector3::new(gauss(rng, std), gauss(rng, std), gauss(rng, std))
}

/// Small random rotation with isotropic Gaussian rotation vector.
pub(crate) fn gauss_rotation(rng: &mut impl Rng, std: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(gauss3(rng, std))
}
