//! Running a full simulation into a time-sorted [`Dataset`].

use nalgebra::{Isometry3, Point3, Translation3, Unit, Vector3};

use crate::raycast::{Hit3, RayCast};
use crate::sensors::{
    FixedFramePoseConfig, ImuConfig, ImuMessage, LandmarkConfig, Lidar2dConfig, Lidar3dConfig,
    OdometryConfig, PointCloudMessage, SensorConfig, SensorData, SensorKind, sensor_rng,
};
use crate::trajectory::{Trajectory, TrajectoryState};
use crate::world2d::World2d;
use crate::world3d::World3d;

/// The environment a dataset is simulated in.
#[derive(Clone, Debug, PartialEq)]
pub enum SimWorld {
    /// Planar world; walls are infinitely tall.
    Planar(World2d),
    /// Box/plane world.
    Volumetric(World3d),
}

impl From<World2d> for SimWorld {
    fn from(w: World2d) -> Self {
        Self::Planar(w)
    }
}

impl From<World3d> for SimWorld {
    fn from(w: World3d) -> Self {
        Self::Volumetric(w)
    }
}

impl RayCast for SimWorld {
    fn raycast(
        &self,
        origin: &Point3<f64>,
        direction: &Unit<Vector3<f64>>,
        max_range: f64,
    ) -> Option<Hit3> {
        match self {
            Self::Planar(w) => RayCast::raycast(w, origin, direction, max_range),
            Self::Volumetric(w) => w.raycast(origin, direction, max_range),
        }
    }
}

/// Static description of one simulated sensor.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorInfo {
    /// Sensor id (unique within a dataset).
    pub id: String,
    /// Sensor kind.
    pub kind: SensorKind,
    /// `tracking_from_sensor`.
    pub extrinsic: Isometry3<f64>,
    /// Full configuration used.
    pub config: SensorConfig,
}

/// One time-stamped sensor message.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorMessage {
    /// Id of the producing sensor.
    pub sensor_id: String,
    /// Absolute time (s). For point clouds, the time of the last point.
    pub time: f64,
    /// Payload, expressed in the sensor frame.
    pub data: SensorData,
}

/// A complete simulated dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    /// Simulated interval start (s).
    pub start_time: f64,
    /// Simulated interval end (s).
    pub end_time: f64,
    /// Seed used.
    pub seed: u64,
    /// Sensors in insertion order.
    pub sensors: Vec<SensorInfo>,
    /// All messages sorted by time (stable: ties keep sensor insertion
    /// order, then per-sensor order).
    pub messages: Vec<SensorMessage>,
    /// Ground truth of the tracking frame sampled at the requested rate.
    pub ground_truth: Vec<TrajectoryState>,
    /// The ground-truth trajectory (for exact queries at arbitrary times).
    pub trajectory: Trajectory,
    /// The world the data was simulated in.
    pub world: SimWorld,
}

impl Dataset {
    /// Looks up a sensor by id.
    pub fn sensor(&self, id: &str) -> Option<&SensorInfo> {
        self.sensors.iter().find(|s| s.id == id)
    }

    /// Messages of one sensor, in time order.
    pub fn messages_for<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a SensorMessage> + 'a {
        self.messages.iter().filter(move |m| m.sensor_id == id)
    }

    /// Point clouds of one sensor as `(time, cloud)`.
    pub fn point_clouds<'a>(
        &'a self,
        id: &'a str,
    ) -> impl Iterator<Item = (f64, &'a PointCloudMessage)> + 'a {
        self.messages_for(id).filter_map(|m| match &m.data {
            SensorData::PointCloud(p) => Some((m.time, p)),
            _ => None,
        })
    }

    /// IMU samples of one sensor as `(time, sample)`.
    pub fn imu<'a>(&'a self, id: &'a str) -> impl Iterator<Item = (f64, &'a ImuMessage)> + 'a {
        self.messages_for(id).filter_map(|m| match &m.data {
            SensorData::Imu(p) => Some((m.time, p)),
            _ => None,
        })
    }

    /// Number of messages per sensor id, in sensor order.
    pub fn message_counts(&self) -> Vec<(String, usize)> {
        self.sensors
            .iter()
            .map(|s| (s.id.clone(), self.messages_for(&s.id).count()))
            .collect()
    }

    /// Exact ground-truth state of the tracking frame at time `t`.
    pub fn ground_truth_at(&self, t: f64) -> TrajectoryState {
        self.trajectory.state(t)
    }
}

/// Builder for a [`Dataset`]. Defaults: seed 0, the full trajectory time
/// range, ground truth sampled at 100 Hz, no sensors.
#[derive(Clone, Debug)]
pub struct DatasetBuilder {
    trajectory: Trajectory,
    world: SimWorld,
    sensors: Vec<(String, SensorConfig)>,
    seed: u64,
    time_range: Option<(f64, f64)>,
    ground_truth_rate_hz: f64,
}

impl DatasetBuilder {
    /// Starts a builder for the given trajectory and world.
    pub fn new(trajectory: Trajectory, world: impl Into<SimWorld>) -> Self {
        Self {
            trajectory,
            world: world.into(),
            sensors: Vec::new(),
            seed: 0,
            time_range: None,
            ground_truth_rate_hz: 100.0,
        }
    }

    /// Sets the master seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Restricts the simulated interval to `[t0, t1]` (clamped to the
    /// trajectory's range).
    pub fn time_range(mut self, t0: f64, t1: f64) -> Self {
        self.time_range = Some((t0, t1));
        self
    }

    /// Sets the ground-truth sampling rate.
    pub fn ground_truth_rate(mut self, rate_hz: f64) -> Self {
        self.ground_truth_rate_hz = rate_hz;
        self
    }

    /// Adds any sensor.
    ///
    /// # Panics
    /// If `id` is already used.
    pub fn sensor(mut self, id: impl Into<String>, config: SensorConfig) -> Self {
        let id = id.into();
        assert!(
            self.sensors.iter().all(|(i, _)| *i != id),
            "duplicate sensor id {id}"
        );
        self.sensors.push((id, config));
        self
    }

    /// Adds a 2D LiDAR.
    pub fn lidar_2d(self, id: impl Into<String>, config: Lidar2dConfig) -> Self {
        self.sensor(id, SensorConfig::Lidar2d(config))
    }

    /// Adds a 3D LiDAR.
    pub fn lidar_3d(self, id: impl Into<String>, config: Lidar3dConfig) -> Self {
        self.sensor(id, SensorConfig::Lidar3d(config))
    }

    /// Adds an IMU.
    pub fn imu(self, id: impl Into<String>, config: ImuConfig) -> Self {
        self.sensor(id, SensorConfig::Imu(config))
    }

    /// Adds odometry.
    pub fn odometry(self, id: impl Into<String>, config: OdometryConfig) -> Self {
        self.sensor(id, SensorConfig::Odometry(config))
    }

    /// Adds a fixed-frame pose sensor.
    pub fn fixed_frame_pose(self, id: impl Into<String>, config: FixedFramePoseConfig) -> Self {
        self.sensor(id, SensorConfig::FixedFramePose(config))
    }

    /// Adds a landmark detector.
    pub fn landmarks(self, id: impl Into<String>, config: LandmarkConfig) -> Self {
        self.sensor(id, SensorConfig::Landmark(config))
    }

    /// Preset: [`World2d::loop_corridor`], [`Trajectory::corridor_loops`] at
    /// 1.5 m/s, 2D LiDAR `"scan"` (default config), IMU `"imu"` and
    /// odometry `"odom"` (default configs; the IMU is mounted level, 10 cm
    /// above the tracking frame origin).
    pub fn corridor_2d(laps: f64) -> Self {
        Self::new(
            Trajectory::corridor_loops(laps, 1.5),
            World2d::loop_corridor(),
        )
        .lidar_2d("scan", Lidar2dConfig::default())
        .imu(
            "imu",
            ImuConfig {
                extrinsic: Isometry3::from_parts(
                    Translation3::new(0.0, 0.0, 0.1),
                    Default::default(),
                ),
                ..ImuConfig::default()
            },
        )
        .odometry("odom", OdometryConfig::default())
    }

    /// Preset: [`World3d::loop_corridor`], [`Trajectory::corridor_loops_3d`]
    /// at 1.2 m/s, 3D LiDAR `"points"` (default VLP-16-like config) and IMU
    /// `"imu"`.
    pub fn corridor_3d(laps: f64) -> Self {
        Self::new(
            Trajectory::corridor_loops_3d(laps, 1.2),
            World3d::loop_corridor(),
        )
        .lidar_3d("points", Lidar3dConfig::default())
        .imu("imu", ImuConfig::default())
    }

    /// Runs the simulation.
    pub fn build(self) -> Dataset {
        let (mut t0, mut t1) = self
            .time_range
            .unwrap_or((self.trajectory.start_time(), self.trajectory.end_time()));
        t0 = t0.max(self.trajectory.start_time());
        t1 = t1.min(self.trajectory.end_time());
        let traj = &self.trajectory;
        let world = &self.world;
        let mut messages = Vec::new();
        let mut sensors = Vec::new();
        for (id, config) in &self.sensors {
            let mut rng = sensor_rng(self.seed, id);
            let data: Vec<(f64, SensorData)> = match config {
                SensorConfig::Lidar2d(c) => wrap(
                    c.simulate(traj, world, t0, t1, &mut rng),
                    SensorData::PointCloud,
                ),
                SensorConfig::Lidar3d(c) => wrap(
                    c.simulate(traj, world, t0, t1, &mut rng),
                    SensorData::PointCloud,
                ),
                SensorConfig::Imu(c) => wrap(c.simulate(traj, t0, t1, &mut rng), SensorData::Imu),
                SensorConfig::Odometry(c) => {
                    wrap(c.simulate(traj, t0, t1, &mut rng), SensorData::Odometry)
                }
                SensorConfig::FixedFramePose(c) => wrap(
                    c.simulate(traj, t0, t1, &mut rng),
                    SensorData::FixedFramePose,
                ),
                SensorConfig::Landmark(c) => wrap(
                    c.simulate(traj, world, t0, t1, &mut rng),
                    SensorData::Landmark,
                ),
            };
            messages.extend(data.into_iter().map(|(time, data)| SensorMessage {
                sensor_id: id.clone(),
                time,
                data,
            }));
            sensors.push(SensorInfo {
                id: id.clone(),
                kind: config.kind(),
                extrinsic: config.extrinsic(),
                config: config.clone(),
            });
        }
        messages.sort_by(|a, b| a.time.total_cmp(&b.time));
        let ground_truth = crate::trajectory::sample_times(t0, t1, self.ground_truth_rate_hz)
            .into_iter()
            .map(|t| traj.state(t))
            .collect();
        Dataset {
            start_time: t0,
            end_time: t1,
            seed: self.seed,
            sensors,
            messages,
            ground_truth,
            trajectory: self.trajectory,
            world: self.world,
        }
    }
}

fn wrap<T>(v: Vec<(f64, T)>, f: impl Fn(T) -> SensorData) -> Vec<(f64, SensorData)> {
    v.into_iter().map(|(t, m)| (t, f(m))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensors::{FixedFramePoseConfig, LandmarkConfig};

    fn small(seed: u64) -> Dataset {
        DatasetBuilder::corridor_2d(1.0)
            .fixed_frame_pose("gps", FixedFramePoseConfig::default())
            .landmarks("lm", LandmarkConfig::default())
            .seed(seed)
            .time_range(0.0, 5.0)
            .build()
    }

    #[test]
    fn deterministic_given_seed() {
        let (a, b) = (small(42), small(42));
        assert_eq!(a.messages, b.messages);
        assert_eq!(a.ground_truth, b.ground_truth);
        let c = small(43);
        assert_ne!(a.messages, c.messages);
    }

    #[test]
    fn sensor_streams_are_independent() {
        // Removing a sensor does not change another sensor's data.
        let full = small(5);
        let partial = DatasetBuilder::corridor_2d(1.0)
            .seed(5)
            .time_range(0.0, 5.0)
            .build();
        let a: Vec<_> = full.messages_for("imu").collect();
        let b: Vec<_> = partial.messages_for("imu").collect();
        assert_eq!(a, b);
    }

    #[test]
    fn messages_sorted_and_counted() {
        let d = small(1);
        assert!(d.messages.windows(2).all(|w| w[0].time <= w[1].time));
        let counts: std::collections::HashMap<_, _> = d.message_counts().into_iter().collect();
        assert_eq!(counts["imu"], 1001); // 200 Hz over [0, 5]
        assert_eq!(counts["odom"], 251);
        assert_eq!(counts["gps"], 6);
        assert_eq!(counts["scan"], 50); // last beam of scan 49 at 4.99986
        assert!(counts["lm"] > 0);
        assert_eq!(d.ground_truth.len(), 501);
        assert_eq!(d.sensor("imu").unwrap().kind, SensorKind::Imu);
        assert!((d.sensor("imu").unwrap().extrinsic.translation.z - 0.1).abs() < 1e-12);
        assert_eq!(d.point_clouds("scan").count(), 50);
        assert_eq!(d.imu("imu").count(), 1001);
    }

    #[test]
    fn corridor_3d_preset_builds() {
        let d = DatasetBuilder::corridor_3d(1.0)
            .time_range(0.0, 0.35)
            .build();
        let clouds: Vec<_> = d.point_clouds("points").collect();
        assert_eq!(clouds.len(), 3);
        for (_, c) in clouds {
            assert!(c.points.len() > 10_000);
            assert_eq!(c.points.last().unwrap().time, 0.0);
        }
    }
}
