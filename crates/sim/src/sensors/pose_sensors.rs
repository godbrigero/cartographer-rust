//! Pose-like sensors: odometry, fixed-frame (GPS-like) pose, landmarks.

use std::f64::consts::PI;

use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use rand::Rng;

use super::{
    FixedFramePoseMessage, LandmarkMessage, LandmarkObservation, OdometryMessage, gauss_rotation,
    gauss3,
};
use crate::raycast::RayCast;
use crate::trajectory::{Trajectory, sample_times};

/// Odometry configuration.
///
/// The odometry frame coincides with the world frame at the first sample
/// (i.e. the first message equals the ground-truth sensor pose). Each
/// subsequent message composes the previous odometry pose with the
/// ground-truth increment `delta = prev_gt^-1 * gt` corrupted as follows
/// (`d = |delta.translation|`, `theta = delta.rotation.angle()`, `dt` the
/// sample period):
///
/// * translation `delta.t * translation_scale + N(0, (translation_noise_per_m
///   * d)^2)` per axis (x, y only when `planar`);
/// * rotation `delta.R * exp(n)` with `n ~ N(0, (rotation_noise_per_rad *
///   theta + rotation_noise_per_m * d)^2)` per axis (yaw only when
///   `planar`), plus a systematic yaw drift `yaw_bias * dt`.
///
/// This is a random walk on increments, so drift grows with distance.
#[derive(Clone, Debug, PartialEq)]
pub struct OdometryConfig {
    /// Samples per second.
    pub rate_hz: f64,
    /// Translation noise std per metre travelled in the increment.
    pub translation_noise_per_m: f64,
    /// Rotation noise std (rad) per radian turned in the increment.
    pub rotation_noise_per_rad: f64,
    /// Rotation noise std (rad) per metre travelled in the increment.
    pub rotation_noise_per_m: f64,
    /// Multiplicative scale error on increment translations (1 = none).
    pub translation_scale: f64,
    /// Systematic yaw drift rate (rad/s).
    pub yaw_bias: f64,
    /// Restrict noise to x, y and yaw.
    pub planar: bool,
    /// `tracking_from_odometry_frame`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for OdometryConfig {
    /// 50 Hz, 2 % translation noise, 2 % rotation noise plus 2 mrad/m,
    /// no systematic errors, planar.
    fn default() -> Self {
        Self {
            rate_hz: 50.0,
            translation_noise_per_m: 0.02,
            rotation_noise_per_rad: 0.02,
            rotation_noise_per_m: 0.002,
            translation_scale: 1.0,
            yaw_bias: 0.0,
            planar: true,
            extrinsic: Isometry3::identity(),
        }
    }
}

impl OdometryConfig {
    /// Default configuration with no noise or systematic error (odometry
    /// equals ground truth).
    pub fn ideal() -> Self {
        Self {
            translation_noise_per_m: 0.0,
            rotation_noise_per_rad: 0.0,
            rotation_noise_per_m: 0.0,
            ..Self::default()
        }
    }

    /// Simulates samples at `t0 + k / rate_hz` within `[t0, t1]`.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, OdometryMessage)> {
        let mut out = Vec::new();
        let mut prev: Option<(f64, Isometry3<f64>)> = None;
        let mut odom = Isometry3::identity();
        for t in sample_times(t0, t1, self.rate_hz) {
            let gt = trajectory.state(t).pose * self.extrinsic;
            match prev {
                None => odom = gt,
                Some((tp, gp)) => {
                    let delta = gp.inverse() * gt;
                    let d = delta.translation.vector.norm();
                    let theta = delta.rotation.angle();
                    let mut nt = gauss3(rng, self.translation_noise_per_m * d);
                    let mut nr = gauss3(
                        rng,
                        self.rotation_noise_per_rad * theta + self.rotation_noise_per_m * d,
                    );
                    if self.planar {
                        nt.z = 0.0;
                        nr.x = 0.0;
                        nr.y = 0.0;
                    }
                    nr.z += self.yaw_bias * (t - tp);
                    let noisy = Isometry3::from_parts(
                        Translation3::from(delta.translation.vector * self.translation_scale + nt),
                        delta.rotation * UnitQuaternion::from_scaled_axis(nr),
                    );
                    odom *= noisy;
                }
            }
            prev = Some((t, gt));
            out.push((t, OdometryMessage { pose: odom }));
        }
        out
    }
}

/// Fixed-frame (GPS-like) pose configuration. Reports
/// `fixed_frame_from_world * world_from_tracking * extrinsic`, with additive
/// Gaussian position noise and a small random rotation.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedFramePoseConfig {
    /// Samples per second.
    pub rate_hz: f64,
    /// Position noise std per axis (m).
    pub position_noise_std: f64,
    /// Rotation noise std per axis (rad).
    pub rotation_noise_std: f64,
    /// Transform from the world (ground-truth) frame to the fixed frame.
    pub fixed_frame_from_world: Isometry3<f64>,
    /// `tracking_from_sensor`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for FixedFramePoseConfig {
    /// 1 Hz, 0.3 m / 0.02 rad noise, fixed frame = world frame.
    fn default() -> Self {
        Self {
            rate_hz: 1.0,
            position_noise_std: 0.3,
            rotation_noise_std: 0.02,
            fixed_frame_from_world: Isometry3::identity(),
            extrinsic: Isometry3::identity(),
        }
    }
}

impl FixedFramePoseConfig {
    /// Simulates samples at `t0 + k / rate_hz` within `[t0, t1]`.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, FixedFramePoseMessage)> {
        sample_times(t0, t1, self.rate_hz)
            .into_iter()
            .map(|t| {
                let gt = self.fixed_frame_from_world * trajectory.state(t).pose * self.extrinsic;
                let dp = gauss3(rng, self.position_noise_std);
                let dr = gauss_rotation(rng, self.rotation_noise_std);
                let pose = Isometry3::from_parts(
                    Translation3::from(gt.translation.vector + dp),
                    gt.rotation * dr,
                );
                (t, FixedFramePoseMessage { pose: Some(pose) })
            })
            .collect()
    }
}

/// A landmark with a known world pose.
#[derive(Clone, Debug, PartialEq)]
pub struct Landmark {
    /// Unique id.
    pub id: String,
    /// `world_from_landmark`.
    pub pose: Isometry3<f64>,
}

impl Landmark {
    /// Creates a landmark at `(x, y, z)` with yaw `yaw` (rad).
    pub fn new(id: impl Into<String>, x: f64, y: f64, z: f64, yaw: f64) -> Self {
        Self {
            id: id.into(),
            pose: Isometry3::from_parts(
                Translation3::new(x, y, z),
                UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
            ),
        }
    }

    /// Ten landmarks mounted 1 m high, 0.1 m in front of the outer walls of
    /// [`crate::World2d::loop_corridor`] / [`crate::World3d::loop_corridor`],
    /// facing into the corridor.
    pub fn corridor_set() -> Vec<Self> {
        let h = PI / 2.0;
        vec![
            Self::new("s0", -16.0, -9.9, 1.0, h),
            Self::new("s1", 2.0, -9.9, 1.0, h),
            Self::new("s2", 15.0, -9.9, 1.0, h),
            Self::new("n0", -14.0, 9.9, 1.0, -h),
            Self::new("n1", 0.0, 9.9, 1.0, -h),
            Self::new("n2", 14.0, 9.9, 1.0, -h),
            Self::new("w0", -19.9, -2.0, 1.0, 0.0),
            Self::new("w1", -19.9, 8.0, 1.5, 0.0),
            Self::new("e0", 19.9, 5.0, 1.0, PI),
            Self::new("e1", 19.9, -8.0, 1.0, PI),
        ]
    }
}

/// Landmark detector configuration. A landmark is observed when it is
/// within `max_range` of the sensor, within `field_of_view` (full angle,
/// centred on the sensor x axis) and, if `require_line_of_sight`, not
/// occluded by the world. Observations carry `sensor_from_landmark` with
/// Gaussian translation noise and a small random rotation. Messages without
/// any observation are not emitted.
#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkConfig {
    /// Detection attempts per second.
    pub rate_hz: f64,
    /// Known landmarks.
    pub landmarks: Vec<Landmark>,
    /// Maximum detection range (m).
    pub max_range: f64,
    /// Full field of view (rad); `>= 2 pi` means omnidirectional.
    pub field_of_view: f64,
    /// Whether occluded landmarks are skipped.
    pub require_line_of_sight: bool,
    /// Translation noise std per axis (m).
    pub translation_noise_std: f64,
    /// Rotation noise std per axis (rad).
    pub rotation_noise_std: f64,
    /// Translation weight reported in each observation.
    pub translation_weight: f64,
    /// Rotation weight reported in each observation.
    pub rotation_weight: f64,
    /// `tracking_from_sensor`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for LandmarkConfig {
    /// 2 Hz, [`Landmark::corridor_set`], 8 m range, omnidirectional, line of
    /// sight required, 2 cm / 0.01 rad noise, weights 1e5 / 1e5.
    fn default() -> Self {
        Self {
            rate_hz: 2.0,
            landmarks: Landmark::corridor_set(),
            max_range: 8.0,
            field_of_view: 2.0 * PI,
            require_line_of_sight: true,
            translation_noise_std: 0.02,
            rotation_noise_std: 0.01,
            translation_weight: 1e5,
            rotation_weight: 1e5,
            extrinsic: Isometry3::identity(),
        }
    }
}

impl LandmarkConfig {
    /// Simulates detections at `t0 + k / rate_hz` within `[t0, t1]`.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        world: &dyn RayCast,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, LandmarkMessage)> {
        let mut out = Vec::new();
        for t in sample_times(t0, t1, self.rate_hz) {
            let sensor = trajectory.state(t).pose * self.extrinsic;
            let mut observations = Vec::new();
            for lm in &self.landmarks {
                // Always draw the noise so the stream does not depend on
                // visibility.
                let dp = gauss3(rng, self.translation_noise_std);
                let dr = gauss_rotation(rng, self.rotation_noise_std);
                let rel = sensor.inverse() * lm.pose;
                let v = rel.translation.vector;
                let range = v.norm();
                if range > self.max_range {
                    continue;
                }
                if self.field_of_view < 2.0 * PI && range > 0.0 {
                    let angle = (v.x / range).clamp(-1.0, 1.0).acos();
                    if angle > 0.5 * self.field_of_view {
                        continue;
                    }
                }
                if self.require_line_of_sight
                    && !world.line_of_sight(
                        &sensor.translation.vector.into(),
                        &lm.pose.translation.vector.into(),
                    )
                {
                    continue;
                }
                observations.push(LandmarkObservation {
                    id: lm.id.clone(),
                    landmark_to_sensor: Isometry3::from_parts(
                        Translation3::from(v + dp),
                        rel.rotation * dr,
                    ),
                    translation_weight: self.translation_weight,
                    rotation_weight: self.rotation_weight,
                });
            }
            if !observations.is_empty() {
                out.push((t, LandmarkMessage { observations }));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world2d::World2d;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn ideal_odometry_equals_ground_truth() {
        let traj = Trajectory::corridor_loops(1.0, 1.5);
        let data =
            OdometryConfig::ideal().simulate(&traj, 0.0, 20.0, &mut ChaCha8Rng::seed_from_u64(3));
        assert_eq!(data.len(), 1001);
        for (t, m) in &data {
            let gt = traj.state(*t).pose;
            assert!((m.pose.translation.vector - gt.translation.vector).norm() < 1e-9);
            assert!(m.pose.rotation.angle_to(&gt.rotation) < 1e-9);
        }
    }

    #[test]
    fn odometry_scale_error_gives_known_drift() {
        // Straight line at 1 m/s: 2 % scale error => 0.2 m after 10 m.
        let traj = Trajectory::from_waypoints(
            &[
                nalgebra::Point2::new(0.0, 0.0),
                nalgebra::Point2::new(20.0, 0.0),
            ],
            1.0,
            false,
        );
        let cfg = OdometryConfig {
            translation_scale: 1.02,
            ..OdometryConfig::ideal()
        };
        let data = cfg.simulate(&traj, 0.0, 10.0, &mut ChaCha8Rng::seed_from_u64(3));
        let (_, last) = data.last().unwrap();
        assert!((last.pose.translation.vector.x - 10.2).abs() < 1e-6);
    }

    #[test]
    fn noisy_odometry_drifts() {
        let traj = Trajectory::corridor_loops(1.0, 1.5);
        let data =
            OdometryConfig::default().simulate(&traj, 0.0, 30.0, &mut ChaCha8Rng::seed_from_u64(3));
        let (t, m) = data.last().unwrap();
        let err = (m.pose.translation.vector - traj.state(*t).pose.translation.vector).norm();
        assert!(err > 1e-3 && err < 2.0, "drift {err}");
    }

    #[test]
    fn landmarks_visible_and_accurate() {
        let traj = Trajectory::corridor_loops(1.0, 1.5);
        let world = World2d::loop_corridor();
        let cfg = LandmarkConfig {
            translation_noise_std: 0.0,
            rotation_noise_std: 0.0,
            ..LandmarkConfig::default()
        };
        let data = cfg.simulate(
            &traj,
            &world,
            0.0,
            traj.duration(),
            &mut ChaCha8Rng::seed_from_u64(3),
        );
        let mut seen = std::collections::BTreeSet::new();
        for (t, m) in &data {
            let pose = traj.state(*t).pose;
            for o in &m.observations {
                seen.insert(o.id.clone());
                let lm = cfg.landmarks.iter().find(|l| l.id == o.id).unwrap();
                let back = pose * o.landmark_to_sensor;
                assert!((back.translation.vector - lm.pose.translation.vector).norm() < 1e-9);
                assert!(o.landmark_to_sensor.translation.vector.norm() <= 8.0);
            }
        }
        assert_eq!(seen.len(), cfg.landmarks.len(), "seen {seen:?}");
    }

    #[test]
    fn fixed_frame_pose_noise_level() {
        let traj = Trajectory::corridor_loops(1.0, 1.5);
        let cfg = FixedFramePoseConfig {
            rate_hz: 10.0,
            ..FixedFramePoseConfig::default()
        };
        let data = cfg.simulate(&traj, 0.0, 30.0, &mut ChaCha8Rng::seed_from_u64(3));
        let mse: f64 = data
            .iter()
            .map(|(t, m)| {
                (m.pose.unwrap().translation.vector - traj.state(*t).pose.translation.vector)
                    .norm_squared()
            })
            .sum::<f64>()
            / data.len() as f64;
        // E[|n|^2] = 3 sigma^2 = 0.27.
        assert!((mse - 0.27).abs() < 0.08, "mse {mse}");
    }
}
