//! IMU simulation.

use nalgebra::{Isometry3, Vector3};
use rand::Rng;

use super::{ImuMessage, gauss3};
use crate::trajectory::{Trajectory, sample_times};

/// IMU configuration. Measurements are expressed in the IMU frame
/// (`extrinsic` = `tracking_from_imu`).
///
/// * `angular_velocity = R_ti^T * w_body + gyro_bias + noise`
/// * `linear_acceleration = R_ti^T * R_wb^T * (a_imu_world - g_world) +
///   accel_bias + noise`, with `g_world = (0, 0, -gravity)`, so a level IMU
///   at rest reads `(0, 0, +gravity)` (Cartographer's convention).
///
/// `a_imu_world` includes the lever-arm terms `alpha x r + w x (w x r)` for a
/// non-zero extrinsic translation `r`; the angular acceleration `alpha` is
/// obtained by a central difference of the analytic angular velocity
/// (`h = 1e-4` s, error ~1e-8 rad/s^2), everything else is analytic.
#[derive(Clone, Debug, PartialEq)]
pub struct ImuConfig {
    /// Samples per second; sample `k` is at `t0 + k / rate_hz`.
    pub rate_hz: f64,
    /// White-noise standard deviation per sample on each gyro axis (rad/s).
    pub gyro_noise_std: f64,
    /// White-noise standard deviation per sample on each accel axis (m/s^2).
    pub accel_noise_std: f64,
    /// Constant gyro bias (rad/s).
    pub gyro_bias: Vector3<f64>,
    /// Constant accelerometer bias (m/s^2).
    pub accel_bias: Vector3<f64>,
    /// Gravity magnitude (m/s^2).
    pub gravity: f64,
    /// `tracking_from_imu`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for ImuConfig {
    /// 200 Hz, 1e-3 rad/s gyro noise, 1e-2 m/s^2 accel noise, no bias,
    /// g = 9.81, identity extrinsic.
    fn default() -> Self {
        Self {
            rate_hz: 200.0,
            gyro_noise_std: 1e-3,
            accel_noise_std: 1e-2,
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            gravity: 9.81,
            extrinsic: Isometry3::identity(),
        }
    }
}

impl ImuConfig {
    /// Default configuration without noise or bias.
    pub fn ideal() -> Self {
        Self {
            gyro_noise_std: 0.0,
            accel_noise_std: 0.0,
            ..Self::default()
        }
    }

    /// Noise-free measurement at time `t`.
    pub fn measure(&self, trajectory: &Trajectory, t: f64) -> ImuMessage {
        let s = trajectory.state(t);
        let r = self.extrinsic.translation.vector;
        let r_ti = self.extrinsic.rotation;
        let w = s.angular_velocity;
        let lever = if r.norm_squared() > 0.0 {
            let h = 1e-4;
            let alpha = (trajectory.state(t + h).angular_velocity
                - trajectory.state(t - h).angular_velocity)
                / (2.0 * h);
            alpha.cross(&r) + w.cross(&w.cross(&r))
        } else {
            Vector3::zeros()
        };
        let g_world = Vector3::new(0.0, 0.0, -self.gravity);
        let specific_force_body =
            s.pose.rotation.inverse() * (s.linear_acceleration - g_world) + lever;
        ImuMessage {
            linear_acceleration: r_ti.inverse() * specific_force_body,
            angular_velocity: r_ti.inverse() * w,
        }
    }

    /// Simulates samples at `t0 + k / rate_hz` within `[t0, t1]`.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, ImuMessage)> {
        sample_times(t0, t1, self.rate_hz)
            .into_iter()
            .map(|t| {
                let mut m = self.measure(trajectory, t);
                m.angular_velocity += self.gyro_bias + gauss3(rng, self.gyro_noise_std);
                m.linear_acceleration += self.accel_bias + gauss3(rng, self.accel_noise_std);
                (t, m)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Translation3, UnitQuaternion};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn at_rest_reads_plus_g() {
        let traj =
            Trajectory::from_waypoints(&[Point2::origin(), Point2::new(1e-6, 0.0)], 1e-9, false);
        let m = ImuConfig::ideal().measure(&traj, 1.0);
        assert!((m.linear_acceleration - Vector3::new(0.0, 0.0, 9.81)).norm() < 1e-9);
        assert!(m.angular_velocity.norm() < 1e-9);
        // Upside-down IMU reads -g.
        let cfg = ImuConfig {
            extrinsic: Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
            ),
            ..ImuConfig::ideal()
        };
        let m = cfg.measure(&traj, 1.0);
        assert!((m.linear_acceleration - Vector3::new(0.0, 0.0, -9.81)).norm() < 1e-9);
    }

    /// Strapdown integration of noise-free IMU data (RK4-like midpoint on
    /// rotation, trapezoidal on velocity) reproduces the trajectory.
    fn integrate_and_compare(traj: &Trajectory, cfg: &ImuConfig, t0: f64, t1: f64) -> (f64, f64) {
        let data = cfg.simulate(traj, t0, t1, &mut ChaCha8Rng::seed_from_u64(1));
        let s0 = traj.state(t0);
        // Integrate the tracking frame using the IMU data mapped back through
        // the extrinsic rotation (translation lever-arm is zero here).
        let r_ti = cfg.extrinsic.rotation;
        let g = Vector3::new(0.0, 0.0, -cfg.gravity);
        let mut q = s0.pose.rotation;
        let mut p = s0.pose.translation.vector;
        let mut v = s0.linear_velocity;
        for w in data.windows(2) {
            let (ta, a) = &w[0];
            let (tb, b) = &w[1];
            let dt = tb - ta;
            let wa = r_ti * a.angular_velocity;
            let wb = r_ti * b.angular_velocity;
            let fa = r_ti * a.linear_acceleration;
            let fb = r_ti * b.linear_acceleration;
            let acc_a = q * fa + g;
            let q_next = q * UnitQuaternion::from_scaled_axis(
                (wa + wb) * 0.5 * dt + wa.cross(&wb) * dt * dt / 12.0,
            );
            let acc_b = q_next * fb + g;
            p += v * dt + (acc_a * 2.0 + acc_b) * dt * dt / 6.0;
            v += (acc_a + acc_b) * 0.5 * dt;
            q = q_next;
        }
        let s1 = traj.state(data.last().unwrap().0);
        (
            (p - s1.pose.translation.vector).norm(),
            q.angle_to(&s1.pose.rotation),
        )
    }

    #[test]
    fn integration_reproduces_trajectory_2d() {
        let traj = Trajectory::room_wander(1.0, 0.8);
        let (dp, dq) = integrate_and_compare(&traj, &ImuConfig::ideal(), 2.0, 7.0);
        assert!(dp < 0.01, "position error {dp}");
        assert!(dq < 1e-3, "rotation error {dq}");
    }

    #[test]
    fn integration_reproduces_trajectory_3d_rotated_imu() {
        let traj = Trajectory::room_wander_3d(1.0, 0.8);
        let cfg = ImuConfig {
            rate_hz: 400.0,
            extrinsic: Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(0.3, -0.2, 1.0),
            ),
            ..ImuConfig::ideal()
        };
        let (dp, dq) = integrate_and_compare(&traj, &cfg, 1.0, 5.0);
        assert!(dp < 0.01, "position error {dp}");
        assert!(dq < 1e-3, "rotation error {dq}");
    }

    #[test]
    fn lever_arm_matches_finite_difference() {
        // Offset IMU: its specific force must equal the second derivative of
        // its own world position minus gravity, rotated into the IMU frame.
        let traj = Trajectory::room_wander_3d(1.0, 0.8);
        let ext = Isometry3::from_parts(
            Translation3::new(0.3, -0.2, 0.1),
            UnitQuaternion::from_euler_angles(0.1, 0.2, -0.5),
        );
        let cfg = ImuConfig {
            extrinsic: ext,
            ..ImuConfig::ideal()
        };
        let t = 3.3;
        let h = 1e-3;
        let pos = |t: f64| (traj.state(t).pose * ext).translation.vector;
        let acc_w = (pos(t + h) - pos(t) * 2.0 + pos(t - h)) / (h * h);
        let imu_pose = traj.state(t).pose * ext;
        let expected = imu_pose.rotation.inverse() * (acc_w - Vector3::new(0.0, 0.0, -9.81));
        let m = cfg.measure(&traj, t);
        assert!(
            (m.linear_acceleration - expected).norm() < 1e-3,
            "{} vs {expected}",
            m.linear_acceleration
        );
    }
}
