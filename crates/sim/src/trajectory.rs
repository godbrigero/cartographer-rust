//! Smooth, time-parameterised ground-truth trajectories with analytic
//! derivatives.
//!
//! A [`Trajectory`] is a planar C2 cubic spline `(x(t), y(t))` through
//! waypoints (periodic for closed tours), re-timed so that the speed is
//! approximately constant, with the yaw tangent to the path. Optionally a
//! [`HandheldMotion`] adds sinusoidal height bobbing and roll/pitch sway for
//! 3D datasets. All derivatives are analytic, so a simulated IMU is exactly
//! consistent with the poses.
//!
//! # Frames
//! * `pose` is `world_from_body` (the body / tracking frame expressed in the
//!   world frame); the body x axis points forward along the path, z up.
//! * `linear_velocity` and `linear_acceleration` are in the **world** frame
//!   (kinematic, gravity not included).
//! * `angular_velocity` is in the **body** frame.

use nalgebra::{Isometry2, Isometry3, Point2, Translation3, UnitQuaternion, Vector2, Vector3};

use crate::spline::{CubicSpline, SplineKind};

/// Full kinematic state of the body at one instant. See the module docs for
/// frame conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryState {
    /// Time in seconds.
    pub time: f64,
    /// `world_from_body`.
    pub pose: Isometry3<f64>,
    /// Linear velocity of the body origin, world frame (m/s).
    pub linear_velocity: Vector3<f64>,
    /// Linear acceleration of the body origin, world frame (m/s^2), without
    /// gravity.
    pub linear_acceleration: Vector3<f64>,
    /// Angular velocity, body frame (rad/s).
    pub angular_velocity: Vector3<f64>,
}

impl TrajectoryState {
    /// Planar projection `(x, y, yaw)` of the pose.
    pub fn pose_2d(&self) -> Isometry2<f64> {
        project_2d(&self.pose)
    }

    /// Angular velocity expressed in the world frame.
    pub fn angular_velocity_world(&self) -> Vector3<f64> {
        self.pose.rotation * self.angular_velocity
    }
}

/// Planar projection `(x, y, yaw)` of a 3D pose (yaw of the rotated x axis).
pub fn project_2d(pose: &Isometry3<f64>) -> Isometry2<f64> {
    let fwd = pose.rotation * Vector3::x();
    Isometry2::new(
        Vector2::new(pose.translation.x, pose.translation.y),
        fwd.y.atan2(fwd.x),
    )
}

/// Sinusoidal walking/handheld motion added on top of the planar path.
/// Amplitudes in metres / radians, frequencies in Hz, phases in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandheldMotion {
    /// Vertical bobbing amplitude (m).
    pub z_amplitude: f64,
    /// Vertical bobbing frequency (Hz).
    pub z_frequency: f64,
    /// Roll amplitude (rad).
    pub roll_amplitude: f64,
    /// Roll frequency (Hz).
    pub roll_frequency: f64,
    /// Roll phase (rad).
    pub roll_phase: f64,
    /// Pitch amplitude (rad).
    pub pitch_amplitude: f64,
    /// Pitch frequency (Hz).
    pub pitch_frequency: f64,
    /// Pitch phase (rad).
    pub pitch_phase: f64,
}

impl HandheldMotion {
    /// A plausible walking motion: 3 cm bobbing at 1.8 Hz (step rate), 2 deg
    /// roll at 0.9 Hz (stride), 3 deg pitch at 0.37 Hz.
    pub fn walking() -> Self {
        Self {
            z_amplitude: 0.03,
            z_frequency: 1.8,
            roll_amplitude: 2f64.to_radians(),
            roll_frequency: 0.9,
            roll_phase: 0.0,
            pitch_amplitude: 3f64.to_radians(),
            pitch_frequency: 0.37,
            pitch_phase: 1.0,
        }
    }
}

/// `(value, first derivative, second derivative)` of `a * sin(2 pi f t + p)`.
fn sinusoid(a: f64, f: f64, p: f64, t: f64) -> (f64, f64, f64) {
    let w = 2.0 * std::f64::consts::PI * f;
    let (s, c) = (w * t + p).sin_cos();
    (a * s, a * w * c, -a * w * w * s)
}

/// A smooth ground-truth trajectory. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    x: CubicSpline,
    y: CubicSpline,
    closed: bool,
    start_time: f64,
    end_time: f64,
    height: f64,
    motion: Option<HandheldMotion>,
}

impl Trajectory {
    /// Builds a trajectory through `waypoints` at approximately constant
    /// `speed` (m/s), starting at time 0 at the first waypoint.
    ///
    /// If `closed`, the path is a periodic spline returning to the first
    /// waypoint (do not repeat it at the end) and the trajectory initially
    /// lasts one lap (see [`Trajectory::with_laps`]). Otherwise a natural
    /// spline from the first to the last waypoint is used.
    ///
    /// # Panics
    /// If fewer than 3 waypoints (closed) / 2 (open) are given, consecutive
    /// waypoints coincide, or `speed <= 0`.
    pub fn from_waypoints(waypoints: &[Point2<f64>], speed: f64, closed: bool) -> Self {
        assert!(speed > 0.0, "speed must be positive");
        let mut pts: Vec<Point2<f64>> = waypoints.to_vec();
        if closed {
            assert!(pts.len() >= 3, "closed trajectory needs >= 3 waypoints");
            pts.push(pts[0]);
        } else {
            assert!(pts.len() >= 2, "trajectory needs >= 2 waypoints");
        }
        let kind = if closed {
            SplineKind::Periodic
        } else {
            SplineKind::Natural
        };
        let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.y).collect();
        // Initial knot times from chord lengths, then re-time a few times
        // from the spline's actual segment arc lengths.
        let mut seg_len: Vec<f64> = pts.windows(2).map(|w| (w[1] - w[0]).norm()).collect();
        assert!(seg_len.iter().all(|&l| l > 1e-9), "duplicate waypoints");
        let fit = |seg_len: &[f64]| {
            let mut t = vec![0.0];
            for l in seg_len {
                t.push(t.last().unwrap() + l / speed);
            }
            let sx = CubicSpline::new(&t, &xs, kind);
            let sy = CubicSpline::new(&t, &ys, kind);
            (t, sx, sy)
        };
        let (mut t, mut sx, mut sy) = fit(&seg_len);
        for _ in 0..3 {
            seg_len = t
                .windows(2)
                .map(|w| arc_length(&sx, &sy, w[0], w[1]))
                .collect();
            (t, sx, sy) = fit(&seg_len);
        }
        // Resample the geometric curve at (nearly) equal arc-length spacing
        // and refit with uniform knot times: this makes the speed almost
        // exactly constant even through tight turns.
        let (sx, sy) = resample_uniform_speed(&t, &sx, &sy, speed, kind);
        let end_time = sx.t_max();
        Self {
            x: sx,
            y: sy,
            closed,
            start_time: 0.0,
            end_time,
            height: 0.0,
            motion: None,
        }
    }

    /// For closed trajectories, sets the duration to `laps` periods (may be
    /// fractional). Ignored for open trajectories.
    pub fn with_laps(mut self, laps: f64) -> Self {
        if self.closed {
            self.end_time = self.start_time + laps * self.x.t_max();
        }
        self
    }

    /// Shifts the trajectory so that it starts at `t0` (seconds).
    pub fn with_start_time(mut self, t0: f64) -> Self {
        self.end_time += t0 - self.start_time;
        self.start_time = t0;
        self
    }

    /// Sets the constant base height `z` of the body (m).
    pub fn with_height(mut self, z: f64) -> Self {
        self.height = z;
        self
    }

    /// Adds handheld / walking motion (height bobbing, roll and pitch).
    pub fn with_handheld_motion(mut self, motion: HandheldMotion) -> Self {
        self.motion = Some(motion);
        self
    }

    /// Start time (s).
    pub fn start_time(&self) -> f64 {
        self.start_time
    }

    /// End time (s).
    pub fn end_time(&self) -> f64 {
        self.end_time
    }

    /// Duration (s).
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Lap period (s) for closed trajectories.
    pub fn period(&self) -> Option<f64> {
        self.closed.then(|| self.x.t_max())
    }

    /// Kinematic state at time `t` (clamped to `[start_time, end_time]`).
    pub fn state(&self, t: f64) -> TrajectoryState {
        let t = t.clamp(self.start_time, self.end_time);
        let tau = t - self.start_time;
        let (x, vx, ax) = self.x.eval(tau);
        let (y, vy, ay) = self.y.eval(tau);
        let speed2 = vx * vx + vy * vy;
        let yaw = vy.atan2(vx);
        let yaw_rate = if speed2 > 1e-12 {
            (vx * ay - vy * ax) / speed2
        } else {
            0.0
        };
        let (z, vz, az, roll, roll_d, pitch, pitch_d) = match &self.motion {
            None => (self.height, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Some(m) => {
                let zz = sinusoid(m.z_amplitude, m.z_frequency, 0.0, tau);
                let r = sinusoid(m.roll_amplitude, m.roll_frequency, m.roll_phase, tau);
                let p = sinusoid(m.pitch_amplitude, m.pitch_frequency, m.pitch_phase, tau);
                (self.height + zz.0, zz.1, zz.2, r.0, r.1, p.0, p.1)
            }
        };
        // ZYX Euler (R = Rz(yaw) Ry(pitch) Rx(roll)) rates -> body rates.
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let omega = Vector3::new(
            roll_d - yaw_rate * sp,
            pitch_d * cr + yaw_rate * cp * sr,
            yaw_rate * cp * cr - pitch_d * sr,
        );
        TrajectoryState {
            time: t,
            pose: Isometry3::from_parts(
                Translation3::new(x, y, z),
                UnitQuaternion::from_euler_angles(roll, pitch, yaw),
            ),
            linear_velocity: Vector3::new(vx, vy, vz),
            linear_acceleration: Vector3::new(ax, ay, az),
            angular_velocity: omega,
        }
    }

    /// States sampled at `rate_hz` from `start_time` to `end_time`
    /// (inclusive of the start, and of the end if it falls on the grid).
    pub fn sample(&self, rate_hz: f64) -> Vec<TrajectoryState> {
        sample_times(self.start_time, self.end_time, rate_hz)
            .into_iter()
            .map(|t| self.state(t))
            .collect()
    }

    /// Path length travelled between `t0` and `t1` (numerical integration).
    pub fn path_length(&self, t0: f64, t1: f64) -> f64 {
        let n = ((t1 - t0) * 50.0).ceil().max(1.0) as usize;
        let mut len = 0.0;
        let mut prev = self.state(t0).pose.translation.vector;
        for k in 1..=n {
            let p = self
                .state(t0 + (t1 - t0) * k as f64 / n as f64)
                .pose
                .translation
                .vector;
            len += (p - prev).norm();
            prev = p;
        }
        len
    }

    // ---------------------------------------------------------------- presets

    /// Laps around the centre line of [`crate::World2d::loop_corridor`]
    /// (rounded rectangle `x = +-18.75`, `y = +-8.75`, corner radius 1.2 m),
    /// counter-clockwise, starting at `(0, -8.75)` heading +x. Height 0.
    pub fn corridor_loops(laps: f64, speed: f64) -> Self {
        let wp = rounded_rectangle(18.75, 8.75, 1.2, 2.5);
        Self::from_waypoints(&wp, speed, true).with_laps(laps)
    }

    /// 3D variant of [`Trajectory::corridor_loops`]: 1.2 m base height with
    /// [`HandheldMotion::walking`].
    pub fn corridor_loops_3d(laps: f64, speed: f64) -> Self {
        Self::corridor_loops(laps, speed)
            .with_height(1.2)
            .with_handheld_motion(HandheldMotion::walking())
    }

    /// A closed tour weaving between the obstacles of
    /// [`crate::World2d::rect_room`], repeated `laps` times. Height 0.
    pub fn room_wander(laps: f64, speed: f64) -> Self {
        let wp = [
            (-4.8, -0.2),
            (-4.0, 2.8),
            (-1.5, 3.1),
            (0.5, 1.2),
            (2.5, 1.0),
            (4.8, 1.0),
            (4.8, -2.5),
            (2.5, -3.2),
            (0.2, -1.4),
            (-2.0, -1.3),
            (-4.2, -2.8),
        ];
        let wp: Vec<_> = wp.iter().map(|&(x, y)| Point2::new(x, y)).collect();
        Self::from_waypoints(&wp, speed, true).with_laps(laps)
    }

    /// 3D variant of [`Trajectory::room_wander`]: 1.2 m base height with
    /// [`HandheldMotion::walking`].
    pub fn room_wander_3d(laps: f64, speed: f64) -> Self {
        Self::room_wander(laps, speed)
            .with_height(1.2)
            .with_handheld_motion(HandheldMotion::walking())
    }

    /// A closed tour of [`crate::World2d::office`]: east along the hallway,
    /// a loop through the south-west room, a loop through the north-middle
    /// room (each entered and left through its 1 m door gap), back west along
    /// the hallway and a U-turn at its west end. Height 0.
    pub fn office_tour(laps: f64, speed: f64) -> Self {
        let wp = [
            // Hallway heading east, then south through door x in [-5, -4].
            (-6.0, 0.3),
            (-5.0, 0.0),
            (-4.5, -0.6),
            (-4.5, -1.5),
            // South-west room loop.
            (-5.2, -2.6),
            (-7.0, -3.4),
            (-7.6, -5.0),
            (-6.0, -6.0),
            (-3.6, -5.6),
            (-3.0, -4.2),
            (-3.8, -2.6),
            (-4.5, -1.5),
            (-4.5, -0.6),
            // Hallway east, then north through door x in [-1, 0].
            (-3.8, 0.0),
            (-1.6, 0.0),
            (-0.6, 0.4),
            (-0.5, 1.5),
            // North-middle room loop.
            (0.2, 3.0),
            (1.5, 4.6),
            (0.0, 5.4),
            (-2.4, 4.6),
            (-2.4, 3.0),
            (-1.1, 2.2),
            (-0.5, 1.5),
            (-0.5, 0.5),
            // Hallway west and U-turn at the west end.
            (-1.6, -0.3),
            (-4.5, -0.4),
            (-6.0, -0.45),
            (-7.8, -0.55),
            (-8.9, -0.3),
            (-8.9, 0.3),
            (-7.8, 0.55),
        ];
        let wp: Vec<_> = wp.iter().map(|&(x, y)| Point2::new(x, y)).collect();
        Self::from_waypoints(&wp, speed, true).with_laps(laps)
    }
}

/// Uniform sample times `t0, t0 + 1/rate, ...` up to and including `t1`
/// (with a small tolerance). Times are computed as `t0 + k / rate` to avoid
/// accumulation error.
pub fn sample_times(t0: f64, t1: f64, rate_hz: f64) -> Vec<f64> {
    assert!(rate_hz > 0.0, "rate must be positive");
    let n = ((t1 - t0) * rate_hz + 1e-9).floor();
    if n < 0.0 {
        return Vec::new();
    }
    (0..=n as usize).map(|k| t0 + k as f64 / rate_hz).collect()
}

/// Target knot spacing (m) of the arc-length resampling.
const RESAMPLE_SPACING: f64 = 0.25;

fn resample_uniform_speed(
    knots: &[f64],
    sx: &CubicSpline,
    sy: &CubicSpline,
    speed: f64,
    kind: SplineKind,
) -> (CubicSpline, CubicSpline) {
    // Cumulative arc-length table over the parameter.
    let per_seg = 32;
    let mut table_t = vec![knots[0]];
    let mut table_s = vec![0.0];
    for w in knots.windows(2) {
        for k in 1..=per_seg {
            let a = w[0] + (w[1] - w[0]) * (k - 1) as f64 / per_seg as f64;
            let b = w[0] + (w[1] - w[0]) * k as f64 / per_seg as f64;
            table_s.push(table_s.last().unwrap() + arc_length(sx, sy, a, b));
            table_t.push(b);
        }
    }
    let total = *table_s.last().unwrap();
    let n = (total / RESAMPLE_SPACING).ceil().max(3.0) as usize;
    let ds = total / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys = Vec::with_capacity(n + 1);
    let mut ts = Vec::with_capacity(n + 1);
    let mut j = 0;
    for k in 0..=n {
        let s = (k as f64 * ds).min(total);
        while j + 2 < table_s.len() && table_s[j + 1] < s {
            j += 1;
        }
        let f = ((s - table_s[j]) / (table_s[j + 1] - table_s[j])).clamp(0.0, 1.0);
        let t = table_t[j] + f * (table_t[j + 1] - table_t[j]);
        xs.push(sx.eval(t).0);
        ys.push(sy.eval(t).0);
        ts.push(k as f64 * ds / speed);
    }
    (
        CubicSpline::new(&ts, &xs, kind),
        CubicSpline::new(&ts, &ys, kind),
    )
}

/// Arc length of the planar spline between `t0` and `t1` (Simpson's rule).
fn arc_length(sx: &CubicSpline, sy: &CubicSpline, t0: f64, t1: f64) -> f64 {
    let n = 16;
    let h = (t1 - t0) / n as f64;
    let speed = |t: f64| {
        let (_, vx, _) = sx.eval(t);
        let (_, vy, _) = sy.eval(t);
        (vx * vx + vy * vy).sqrt()
    };
    let mut s = speed(t0) + speed(t1);
    for k in 1..n {
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        s += w * speed(t0 + k as f64 * h);
    }
    s * h / 3.0
}

fn rounded_rectangle(hx: f64, hy: f64, r: f64, spacing: f64) -> Vec<Point2<f64>> {
    // Counter-clockwise from (0, -hy).
    let mut pts = Vec::new();
    let straight = |a: Point2<f64>, b: Point2<f64>, pts: &mut Vec<Point2<f64>>| {
        let n = ((b - a).norm() / spacing).ceil().max(1.0) as usize;
        for k in 0..n {
            pts.push(a + (b - a) * (k as f64 / n as f64));
        }
    };
    let corner = |c: Point2<f64>, a0: f64, pts: &mut Vec<Point2<f64>>| {
        for k in 0..4 {
            let a = a0 + std::f64::consts::FRAC_PI_2 * k as f64 / 4.0;
            pts.push(c + Vector2::new(a.cos(), a.sin()) * r);
        }
    };
    use std::f64::consts::{FRAC_PI_2, PI};
    straight(Point2::new(0.0, -hy), Point2::new(hx - r, -hy), &mut pts);
    corner(Point2::new(hx - r, -hy + r), -FRAC_PI_2, &mut pts);
    straight(Point2::new(hx, -hy + r), Point2::new(hx, hy - r), &mut pts);
    corner(Point2::new(hx - r, hy - r), 0.0, &mut pts);
    straight(Point2::new(hx - r, hy), Point2::new(-hx + r, hy), &mut pts);
    corner(Point2::new(-hx + r, hy - r), FRAC_PI_2, &mut pts);
    straight(
        Point2::new(-hx, hy - r),
        Point2::new(-hx, -hy + r),
        &mut pts,
    );
    corner(Point2::new(-hx + r, -hy + r), PI, &mut pts);
    straight(Point2::new(-hx + r, -hy), Point2::new(0.0, -hy), &mut pts);
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world2d::World2d;

    fn check_derivatives(traj: &Trajectory) {
        let h = 1e-5;
        let mut t = traj.start_time() + 0.1;
        while t < traj.end_time() - 0.1 {
            let s = traj.state(t);
            let (a, b) = (traj.state(t - h), traj.state(t + h));
            let fd_v = (b.pose.translation.vector - a.pose.translation.vector) / (2.0 * h);
            let fd_a = (b.linear_velocity - a.linear_velocity) / (2.0 * h);
            // Body angular velocity from R(t-h)^T R(t+h) = exp(2h [w]x).
            let fd_w = (a.pose.rotation.inverse() * b.pose.rotation).scaled_axis() / (2.0 * h);
            assert!((fd_v - s.linear_velocity).norm() < 1e-6, "v at {t}");
            assert!((fd_a - s.linear_acceleration).norm() < 1e-4, "a at {t}");
            assert!(
                (fd_w - s.angular_velocity).norm() < 1e-5,
                "w at {t}: {fd_w} vs {}",
                s.angular_velocity
            );
            t += 0.37;
        }
    }

    #[test]
    fn derivatives_consistent_2d_and_3d() {
        check_derivatives(&Trajectory::room_wander(1.0, 0.8));
        check_derivatives(&Trajectory::room_wander_3d(1.0, 0.8));
        check_derivatives(&Trajectory::corridor_loops_3d(0.3, 1.5).with_start_time(5.0));
    }

    fn min_clearance(world: &World2d, traj: &Trajectory) -> f64 {
        traj.sample(20.0)
            .iter()
            .map(|s| {
                let p = s.pose_2d().translation.vector.into();
                assert!(world.is_free(&p), "{p} not free");
                world.distance_to_nearest_wall(&p)
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn presets_stay_in_free_space() {
        let c = min_clearance(
            &World2d::loop_corridor(),
            &Trajectory::corridor_loops(1.0, 1.5),
        );
        assert!(c > 0.8, "corridor clearance {c}");
        let c = min_clearance(&World2d::rect_room(), &Trajectory::room_wander(1.0, 0.8));
        assert!(c > 0.4, "room clearance {c}");
        let c = min_clearance(&World2d::office(), &Trajectory::office_tour(1.0, 1.0));
        assert!(c > 0.25, "office clearance {c}");
    }

    #[test]
    fn speed_is_nearly_constant_and_heading_tangent() {
        for traj in [
            Trajectory::corridor_loops(1.0, 1.5),
            Trajectory::room_wander(1.0, 0.8),
            Trajectory::office_tour(1.0, 1.0),
        ] {
            let nominal = traj.path_length(traj.start_time(), traj.end_time()) / traj.duration();
            for s in traj.sample(10.0) {
                let v = s.linear_velocity;
                let speed = v.norm();
                assert!(
                    (speed / nominal - 1.0).abs() < 0.05,
                    "speed {speed} vs {nominal}"
                );
                let fwd = s.pose.rotation * Vector3::x();
                assert!((fwd - v / speed).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn closed_loops_are_periodic() {
        let traj = Trajectory::corridor_loops(3.0, 1.5);
        let p = traj.period().unwrap();
        assert!((traj.duration() - 3.0 * p).abs() < 1e-9);
        // Lap length close to the rounded-rectangle perimeter.
        let perimeter = 2.0 * (37.5 + 17.5) - 8.0 * 1.2 + 2.0 * std::f64::consts::PI * 1.2;
        assert!((p * 1.5 / perimeter - 1.0).abs() < 0.02, "period {p}");
        let (a, b) = (traj.state(1.0), traj.state(1.0 + 2.0 * p));
        assert!((a.pose.translation.vector - b.pose.translation.vector).norm() < 1e-9);
        assert!(a.pose.rotation.angle_to(&b.pose.rotation) < 1e-9);
        let start = traj.state(0.0).pose_2d();
        assert!((start.translation.vector - Vector2::new(0.0, -8.75)).norm() < 1e-9);
        assert!(start.rotation.angle().abs() < 1e-2);
    }

    #[test]
    fn sample_times_grid() {
        assert_eq!(sample_times(1.0, 2.0, 4.0), vec![1.0, 1.25, 1.5, 1.75, 2.0]);
        assert_eq!(sample_times(0.0, 0.3, 4.0), vec![0.0, 0.25]);
    }
}
