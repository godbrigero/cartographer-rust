//! 2D and 3D LiDAR simulation.

use std::f64::consts::PI;

use nalgebra::{Isometry3, Unit, Vector3};
use rand::Rng;

use super::{PointCloudMessage, TimedPoint, gauss};
use crate::raycast::RayCast;
use crate::trajectory::Trajectory;

/// What to do with beams that produce no valid return (nothing hit within
/// `max_range`, or the noisy range fell outside `[min_range, max_range]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MissHandling {
    /// Emit nothing for the beam (typical for real drivers).
    Drop,
    /// Emit a point at `range` metres along the beam, flagged in
    /// [`PointCloudMessage::misses`]. Choose `range` above the consumer's
    /// `max_range` so Cartographer converts it into a miss of length
    /// `missing_data_ray_length`. Beams whose noisy range fell **below**
    /// `min_range` are always dropped.
    Return {
        /// Distance (m) at which the synthetic point is placed.
        range: f64,
    },
}

/// Temporal model of a scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScanTiming {
    /// All beams of a scan are measured at the same instant.
    Instantaneous,
    /// Beams are measured sequentially (one column at a time) at evenly
    /// spaced instants over `scan_duration` seconds starting at the scan start
    /// time, each from the sensor pose at its own time (motion distortion).
    Rotating {
        /// Time from the first to just after the last beam (s); beam `i` of
        /// `n` is at `start + i * scan_duration / n`.
        scan_duration: f64,
    },
}

/// Planar LiDAR configuration. Beams lie in the sensor's x-y plane at angles
/// `angle_min + i * (angle_max - angle_min) / num_beams`, `i = 0..num_beams`
/// (counter-clockwise, i.e. `angle_max` itself is excluded).
#[derive(Clone, Debug, PartialEq)]
pub struct Lidar2dConfig {
    /// Scans per second; scan `k` starts at `t0 + k / rate_hz`.
    pub rate_hz: f64,
    /// Number of beams per scan.
    pub num_beams: usize,
    /// First beam angle (rad).
    pub angle_min: f64,
    /// End of the angular range (rad, exclusive).
    pub angle_max: f64,
    /// Minimum valid range (m).
    pub min_range: f64,
    /// Maximum valid range (m).
    pub max_range: f64,
    /// Standard deviation of additive Gaussian range noise (m).
    pub range_noise_std: f64,
    /// Handling of beams without a valid return.
    pub miss: MissHandling,
    /// Scan timing model.
    pub timing: ScanTiming,
    /// `tracking_from_sensor`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for Lidar2dConfig {
    /// 10 Hz, 720 beams over 360 deg, 0.1-30 m, 1 cm noise, rotating over
    /// the full scan period, misses dropped, identity extrinsic.
    fn default() -> Self {
        Self {
            rate_hz: 10.0,
            num_beams: 720,
            angle_min: -PI,
            angle_max: PI,
            min_range: 0.1,
            max_range: 30.0,
            range_noise_std: 0.01,
            miss: MissHandling::Drop,
            timing: ScanTiming::Rotating { scan_duration: 0.1 },
            extrinsic: Isometry3::identity(),
        }
    }
}

impl Lidar2dConfig {
    /// Default configuration without noise and with instantaneous scans.
    pub fn ideal() -> Self {
        Self {
            range_noise_std: 0.0,
            timing: ScanTiming::Instantaneous,
            ..Self::default()
        }
    }

    /// Beam directions in the sensor frame, in measurement order.
    pub fn beam_directions(&self) -> Vec<Unit<Vector3<f64>>> {
        let inc = (self.angle_max - self.angle_min) / self.num_beams as f64;
        (0..self.num_beams)
            .map(|i| {
                let a = self.angle_min + i as f64 * inc;
                Unit::new_unchecked(Vector3::new(a.cos(), a.sin(), 0.0))
            })
            .collect()
    }

    /// Simulates all scans whose last beam falls in `[t0, t1]`. Returns
    /// `(message time, message)` pairs in time order.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        world: &dyn RayCast,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, PointCloudMessage)> {
        let dirs = self.beam_directions();
        let columns: Vec<Vec<Unit<Vector3<f64>>>> = dirs.into_iter().map(|d| vec![d]).collect();
        let common = ScanCommon {
            rate_hz: self.rate_hz,
            min_range: self.min_range,
            max_range: self.max_range,
            range_noise_std: self.range_noise_std,
            miss: self.miss,
            timing: self.timing,
            extrinsic: self.extrinsic,
            intensities: false,
        };
        simulate_scans(&common, &columns, trajectory, world, t0, t1, rng)
    }
}

/// Rotating multi-beam 3D LiDAR configuration (Velodyne-like). Rings are
/// evenly spaced in elevation from `min_elevation` to `max_elevation`
/// (inclusive); columns are at azimuths `-pi + j * azimuth_step` measured
/// counter-clockwise about the sensor z axis. All rings of a column share a
/// time stamp.
#[derive(Clone, Debug, PartialEq)]
pub struct Lidar3dConfig {
    /// Revolutions per second; scan `k` starts at `t0 + k / rate_hz`.
    pub rate_hz: f64,
    /// Number of rings (lasers).
    pub num_rings: usize,
    /// Lowest ring elevation (rad).
    pub min_elevation: f64,
    /// Highest ring elevation (rad).
    pub max_elevation: f64,
    /// Azimuth increment between columns (rad).
    pub azimuth_step: f64,
    /// Minimum valid range (m).
    pub min_range: f64,
    /// Maximum valid range (m).
    pub max_range: f64,
    /// Standard deviation of additive Gaussian range noise (m).
    pub range_noise_std: f64,
    /// Handling of beams without a valid return.
    pub miss: MissHandling,
    /// Scan timing model.
    pub timing: ScanTiming,
    /// Whether to produce intensities: `100 * |cos(incidence)| /
    /// (1 + (range / 10)^2)` for returns and `0` for misses.
    pub intensities: bool,
    /// `tracking_from_sensor`.
    pub extrinsic: Isometry3<f64>,
}

impl Default for Lidar3dConfig {
    /// VLP-16-like: 10 Hz, 16 rings over +-15 deg, 0.4 deg azimuth step,
    /// 0.3-50 m, 2 cm noise, rotating over the full period, intensities on,
    /// misses dropped, identity extrinsic.
    fn default() -> Self {
        Self {
            rate_hz: 10.0,
            num_rings: 16,
            min_elevation: -15f64.to_radians(),
            max_elevation: 15f64.to_radians(),
            azimuth_step: 0.4f64.to_radians(),
            min_range: 0.3,
            max_range: 50.0,
            range_noise_std: 0.02,
            miss: MissHandling::Drop,
            timing: ScanTiming::Rotating { scan_duration: 0.1 },
            intensities: true,
            extrinsic: Isometry3::identity(),
        }
    }
}

impl Lidar3dConfig {
    /// Default configuration without noise and with instantaneous scans.
    pub fn ideal() -> Self {
        Self {
            range_noise_std: 0.0,
            timing: ScanTiming::Instantaneous,
            ..Self::default()
        }
    }

    /// Number of azimuth columns per revolution.
    pub fn num_columns(&self) -> usize {
        (2.0 * PI / self.azimuth_step).round().max(1.0) as usize
    }

    /// Beam directions in the sensor frame grouped by column (measurement
    /// order), each column ordered from the lowest to the highest ring.
    pub fn beam_columns(&self) -> Vec<Vec<Unit<Vector3<f64>>>> {
        let elevations: Vec<f64> = (0..self.num_rings)
            .map(|r| {
                if self.num_rings == 1 {
                    self.min_elevation
                } else {
                    self.min_elevation
                        + (self.max_elevation - self.min_elevation) * r as f64
                            / (self.num_rings - 1) as f64
                }
            })
            .collect();
        (0..self.num_columns())
            .map(|j| {
                let az = -PI + j as f64 * self.azimuth_step;
                elevations
                    .iter()
                    .map(|el| {
                        Unit::new_unchecked(Vector3::new(
                            el.cos() * az.cos(),
                            el.cos() * az.sin(),
                            el.sin(),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Simulates all revolutions whose last column falls in `[t0, t1]`.
    pub fn simulate(
        &self,
        trajectory: &Trajectory,
        world: &dyn RayCast,
        t0: f64,
        t1: f64,
        rng: &mut impl Rng,
    ) -> Vec<(f64, PointCloudMessage)> {
        let common = ScanCommon {
            rate_hz: self.rate_hz,
            min_range: self.min_range,
            max_range: self.max_range,
            range_noise_std: self.range_noise_std,
            miss: self.miss,
            timing: self.timing,
            extrinsic: self.extrinsic,
            intensities: self.intensities,
        };
        simulate_scans(
            &common,
            &self.beam_columns(),
            trajectory,
            world,
            t0,
            t1,
            rng,
        )
    }
}

struct ScanCommon {
    rate_hz: f64,
    min_range: f64,
    max_range: f64,
    range_noise_std: f64,
    miss: MissHandling,
    timing: ScanTiming,
    extrinsic: Isometry3<f64>,
    intensities: bool,
}

fn simulate_scans(
    c: &ScanCommon,
    columns: &[Vec<Unit<Vector3<f64>>>],
    trajectory: &Trajectory,
    world: &dyn RayCast,
    t0: f64,
    t1: f64,
    rng: &mut impl Rng,
) -> Vec<(f64, PointCloudMessage)> {
    assert!(c.rate_hz > 0.0, "LiDAR rate must be positive");
    let n_cols = columns.len();
    let col_dt = match c.timing {
        ScanTiming::Instantaneous => 0.0,
        ScanTiming::Rotating { scan_duration } => scan_duration / n_cols as f64,
    };
    let mut out = Vec::new();
    for k in 0.. {
        let ts = t0 + k as f64 / c.rate_hz;
        let nominal_last = ts + n_cols.saturating_sub(1) as f64 * col_dt;
        if nominal_last > t1 + 1e-9 {
            break;
        }
        // (absolute time, point, intensity, is_miss)
        let mut raw: Vec<(f64, Vector3<f64>, f32, bool)> = Vec::new();
        for (j, col) in columns.iter().enumerate() {
            let t = ts + j as f64 * col_dt;
            let sensor_pose = trajectory.state(t).pose * c.extrinsic;
            let origin = sensor_pose.translation.vector.into();
            for d in col {
                let noise = gauss(rng, c.range_noise_std);
                let world_dir = Unit::new_unchecked(sensor_pose.rotation * d.into_inner());
                let hit = world.raycast(&origin, &world_dir, c.max_range);
                let valid = hit.and_then(|h| {
                    let r = h.range + noise;
                    (r <= c.max_range).then_some((r, h))
                });
                match (valid, c.miss) {
                    (Some((r, _)), _) if r < c.min_range => {}
                    (Some((r, h)), _) => {
                        let intensity = if c.intensities {
                            let cos_inc = h.normal.dot(&world_dir).abs();
                            (100.0 * cos_inc / (1.0 + (r / 10.0).powi(2))) as f32
                        } else {
                            0.0
                        };
                        raw.push((t, d.into_inner() * r, intensity, false));
                    }
                    (None, MissHandling::Return { range }) => {
                        raw.push((t, d.into_inner() * range, 0.0, true));
                    }
                    (None, MissHandling::Drop) => {}
                }
            }
        }
        let msg_time = raw.last().map_or(nominal_last, |p| p.0);
        let msg = PointCloudMessage {
            origin: Vector3::zeros(),
            points: raw
                .iter()
                .map(|(t, p, _, _)| TimedPoint {
                    position: p.cast::<f32>(),
                    time: (t - msg_time) as f32,
                })
                .collect(),
            intensities: if c.intensities {
                raw.iter().map(|r| r.2).collect()
            } else {
                Vec::new()
            },
            misses: raw.iter().map(|r| r.3).collect(),
        };
        out.push((msg_time, msg));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world2d::{Polygon2, World2d};
    use crate::world3d::World3d;
    use nalgebra::{Point2, Translation3, UnitQuaternion};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(7)
    }

    /// A trajectory that sits (almost) still at the origin facing +x: a
    /// straight line moving at 1 um/s.
    fn nearly_static() -> Trajectory {
        let wp = [Point2::new(0.0, 0.0), Point2::new(1e-3, 0.0)];
        Trajectory::from_waypoints(&wp, 1e-6, false)
    }

    #[test]
    fn lidar2d_exact_ranges_in_square_room() {
        let mut w = World2d::new();
        w.add_enclosure(Polygon2::rect(-5.0, -3.0, 4.0, 2.0));
        let cfg = Lidar2dConfig {
            num_beams: 4,
            ..Lidar2dConfig::ideal()
        };
        let scans = cfg.simulate(&nearly_static(), &w, 0.0, 0.25, &mut rng());
        assert_eq!(scans.len(), 3); // t = 0, 0.1, 0.2
        let (t, msg) = &scans[0];
        assert_eq!(*t, 0.0);
        // Beams at -pi, -pi/2, 0, pi/2.
        let expect = [5.0, 3.0, 4.0, 2.0];
        assert_eq!(msg.points.len(), 4);
        for (p, e) in msg.points.iter().zip(expect) {
            assert!((p.position.norm() - e).abs() < 1e-4, "{p:?} vs {e}");
            assert_eq!(p.time, 0.0);
        }
        assert!(msg.misses.iter().all(|m| !m));
    }

    #[test]
    fn lidar2d_extrinsic_and_misses() {
        let w = World2d::rect_room();
        // Sensor mounted 1 m forward, yawed by +90 deg.
        let extrinsic = Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 0.3),
            UnitQuaternion::from_euler_angles(0.0, 0.0, PI / 2.0),
        );
        let cfg = Lidar2dConfig {
            num_beams: 4,
            max_range: 3.5,
            miss: MissHandling::Return { range: 40.0 },
            extrinsic,
            ..Lidar2dConfig::ideal()
        };
        let scans = cfg.simulate(&nearly_static(), &w, 0.0, 0.0, &mut rng());
        let msg = &scans[0].1;
        // Sensor at world (1, 0) facing +y. Beam -pi -> world -y: wall y=-4
        // at 4 m > 3.5 => miss. Beam -pi/2 -> world +x: wall x = 6 at 5 m =>
        // miss. Beam 0 -> world +y: wall y = 4 at 4 m => miss. Beam pi/2 ->
        // world -x: pillar? (no, y = 0) wall x = -6 at 7 m => miss.
        assert_eq!(msg.misses, vec![true; 4]);
        assert!((msg.points[0].position.norm() - 40.0).abs() < 1e-4);
        let cfg = Lidar2dConfig {
            max_range: 30.0,
            ..cfg
        };
        let msg = &cfg.simulate(&nearly_static(), &w, 0.0, 0.0, &mut rng())[0].1;
        let expect = [4.0, 5.0, 4.0, 7.0];
        for (p, e) in msg.points.iter().zip(expect) {
            assert!((p.position.norm() - e).abs() < 1e-4, "{p:?} vs {e}");
        }
        // Beam 0 is along sensor +x, which is world +y.
        assert!((msg.points[2].position - Vector3::new(4.0, 0.0, 0.0)).norm() < 1e-4);
    }

    #[test]
    fn rotating_timing_convention() {
        let w = World2d::loop_corridor();
        let traj = Trajectory::corridor_loops(1.0, 2.0);
        let cfg = Lidar2dConfig {
            num_beams: 100,
            timing: ScanTiming::Rotating {
                scan_duration: 0.05,
            },
            range_noise_std: 0.0,
            ..Lidar2dConfig::default()
        };
        let scans = cfg.simulate(&traj, &w, 1.0, 3.0, &mut rng());
        let dt = 0.05 / 100.0;
        for (t, msg) in &scans {
            assert!(!msg.points.is_empty());
            assert_eq!(msg.points.last().unwrap().time, 0.0);
            assert!(msg.points.windows(2).all(|p| p[0].time <= p[1].time));
            assert!(msg.points.iter().all(|p| p.time <= 0.0));
            // Points come from beams spaced by dt.
            let first = msg.points[0].time as f64;
            assert!(((-first / dt).round() * dt + first).abs() < 1e-6);
            // Every point lies on a wall when transformed with the pose at
            // its own time (motion distortion is exact).
            for p in &msg.points {
                let pose = traj.state(t + p.time as f64).pose;
                let pw = pose * nalgebra::Point3::from(p.position.cast::<f64>());
                let d = w.distance_to_nearest_wall(&pw.xy());
                assert!(d < 1e-3, "point off wall by {d}");
            }
        }
        // Scan k starts at 1 + k / 10; the last nominal beam must be <= 3.
        assert_eq!(scans.len(), 20);
    }

    #[test]
    fn lidar3d_room_geometry() {
        let w = World3d::room();
        let traj = Trajectory::room_wander_3d(1.0, 0.8);
        let cfg = Lidar3dConfig {
            azimuth_step: 2f64.to_radians(),
            ..Lidar3dConfig::default()
        };
        let scans = cfg.simulate(&traj, &w, 0.0, 0.5, &mut rng());
        assert_eq!(scans.len(), 5);
        let (t, msg) = &scans[2];
        assert_eq!(msg.intensities.len(), msg.points.len());
        assert!(msg.points.len() > 16 * 150);
        assert_eq!(msg.points.last().unwrap().time, 0.0);
        // Transformed points lie inside the room's bounding box (noise 2 cm).
        for p in &msg.points {
            let pose = traj.state(t + p.time as f64).pose;
            let q = pose * nalgebra::Point3::from(p.position.cast::<f64>());
            assert!(q.x.abs() < 6.1 && q.y.abs() < 4.1 && q.z > -0.1 && q.z < 3.1);
        }
        assert!(msg.intensities.iter().all(|&i| (0.0..=100.0).contains(&i)));
    }
}
