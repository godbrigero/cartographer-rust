//! [`Slam`]: the running SLAM system.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};

use super::builder::{LocalCallbacks, OptimizationCallbacks, SlamBuilder};
use super::error::SlamError;
use super::map::{OccupancyGrid, PointMap, compose_occupancy_grid, compose_point_map};
use super::time::Timestamp;
use super::types::{
    ImuSample, LocalResult, Mode, OptimizationEvent, SensorKind, SlamStats, StampedPose,
    SubmapInfo, TimedPoint, to_isometry, to_rigid3d,
};
use crate::common::time::Time;
use crate::mapping::id::{NodeId, SubmapId};
use crate::mapping::map_builder::MapBuilder;
use crate::mapping::map_builder_interface::MapBuilderInterface;
use crate::mapping::pose_graph_interface::{PoseGraphInterface, Tag};
use crate::mapping::proto::{MapBuilderOptions, TrajectoryBuilderOptions};
use crate::sensor::imu_data::ImuData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::rangefinder_point::TimedRangefinderPoint;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::Rigid3d;

/// State shared with the callbacks handed to the faithful layer.
#[derive(Default)]
pub(crate) struct Shared {
    pub(crate) local_callbacks: LocalCallbacks,
    pub(crate) optimization_callbacks: OptimizationCallbacks,
    pub(crate) latest_local_pose: Mutex<Option<(Time, Rigid3d)>>,
    pub(crate) num_local_results: AtomicUsize,
    pub(crate) num_optimizations: AtomicUsize,
}

/// The declared sensors with their extrinsics, prepared for transforming
/// data into the tracking frame.
pub(crate) struct SlamSensors {
    /// tracking_from_sensor (f32, like cartographer_ros' `cast<float>()`).
    pub(crate) lidars: HashMap<String, Isometry3<f32>>,
    /// Rotation tracking_from_imu.
    pub(crate) imu: Option<(String, UnitQuaternion<f64>)>,
    /// sensor_from_tracking (the inverse extrinsics).
    pub(crate) odometry: Option<(String, Isometry3<f64>)>,
}

/// Default probability threshold of [`Slam::map_3d`].
pub const DEFAULT_MIN_OCCUPIED_PROBABILITY_3D: f32 = 0.6;

/// A running SLAM system (one trajectory being built or localized), wrapping
/// Cartographer's [`MapBuilder`].
///
/// Create it with [`Slam::builder`]. Feed it sensor data in (roughly) time
/// order with the `push_*` methods; read results at any time; call
/// [`finish`](Self::finish) at the end to run the final optimization.
///
/// `Slam` is `Send` (it can be moved to a worker thread); loop closure and
/// optimization run on Cartographer's background thread pool.
pub struct Slam {
    mode: Mode,
    // Dropped before `pose_graph`'s last handle: MapBuilder's drop waits for
    // the background work.
    map_builder: MapBuilder,
    pose_graph: Arc<dyn PoseGraphInterface>,
    trajectory_id: i32,
    sensors: SlamSensors,
    use_intensities: bool,
    last_times: HashMap<String, Timestamp>,
    finished: bool,
    shared: Arc<Shared>,
    options: (MapBuilderOptions, TrajectoryBuilderOptions),
}

impl std::fmt::Debug for Slam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slam")
            .field("mode", &self.mode)
            .field("trajectory_id", &self.trajectory_id)
            .field("finished", &self.finished)
            .field("lidars", &self.sensors.lidars.keys().collect::<Vec<_>>())
            .field("imu", &self.sensors.imu.as_ref().map(|(id, _)| id))
            .field(
                "odometry",
                &self.sensors.odometry.as_ref().map(|(id, _)| id),
            )
            .finish_non_exhaustive()
    }
}

// `Slam` must stay movable across threads.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Slam>();
};

impl Slam {
    /// Starts building a SLAM system with Cartographer's default
    /// configuration for `mode`.
    pub fn builder(mode: Mode) -> SlamBuilder {
        SlamBuilder::new(mode)
    }

    /// Shorthand for `builder.localize_in(pbstream).build()`: localize in a
    /// saved map with the sensors and configuration of `builder`.
    pub fn load_for_localization(
        pbstream: impl AsRef<Path>,
        builder: SlamBuilder,
    ) -> Result<Slam, SlamError> {
        builder.localize_in(pbstream.as_ref()).build()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mode: Mode,
        map_builder: MapBuilder,
        pose_graph: Arc<dyn PoseGraphInterface>,
        trajectory_id: i32,
        sensors: SlamSensors,
        use_intensities: bool,
        shared: Arc<Shared>,
        options: (MapBuilderOptions, TrajectoryBuilderOptions),
    ) -> Self {
        Slam {
            mode,
            map_builder,
            pose_graph,
            trajectory_id,
            sensors,
            use_intensities,
            last_times: HashMap::new(),
            finished: false,
            shared,
            options,
        }
    }

    // ---------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------

    fn check_accepting(&self) -> Result<(), SlamError> {
        if self.finished {
            Err(SlamError::TrajectoryFinished)
        } else {
            Ok(())
        }
    }

    fn check_time(&self, sensor: &str, time: Timestamp) -> Result<(), SlamError> {
        match self.last_times.get(sensor) {
            Some(&previous) if time <= previous => Err(SlamError::NonMonotonicTime {
                sensor: sensor.to_string(),
                previous,
                current: time,
            }),
            _ => Ok(()),
        }
    }

    fn trajectory_builder(
        &mut self,
    ) -> &mut dyn crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface {
        self.map_builder
            .get_trajectory_builder(self.trajectory_id)
            .expect("the active trajectory has a builder")
    }

    /// Adds a scan of LiDAR `sensor_id` with per-point times.
    ///
    /// `time` is the acquisition time of the **last** point and each
    /// point's [`TimedPoint::time`] is relative to it (`<= 0`, non-decreasing);
    /// see [`TimedPoint`] for converting other conventions. Points are in the
    /// sensor frame; returns beyond `max_range` are treated as free-space
    /// rays (2D). An empty scan is ignored.
    pub fn push_lidar(
        &mut self,
        sensor_id: &str,
        time: impl Into<Timestamp>,
        points: &[TimedPoint],
    ) -> Result<(), SlamError> {
        self.push_range(sensor_id, time.into(), points, &[])
    }

    /// Like [`push_lidar`](Self::push_lidar) with one intensity per point
    /// (used by 3D SLAM when `use_intensities` is enabled via
    /// [`SlamConfig::advanced`](super::SlamConfig::advanced)).
    pub fn push_lidar_with_intensities(
        &mut self,
        sensor_id: &str,
        time: impl Into<Timestamp>,
        points: &[TimedPoint],
        intensities: &[f32],
    ) -> Result<(), SlamError> {
        if intensities.len() != points.len() {
            return Err(SlamError::InvalidData(format!(
                "{} intensities for {} points",
                intensities.len(),
                points.len()
            )));
        }
        self.push_range(sensor_id, time.into(), points, intensities)
    }

    /// Adds a scan without per-point times (all points measured at `time`),
    /// e.g. from a driver that does not report them or an already
    /// motion-compensated cloud.
    pub fn push_lidar_untimed(
        &mut self,
        sensor_id: &str,
        time: impl Into<Timestamp>,
        points: &[Vector3<f32>],
    ) -> Result<(), SlamError> {
        let points: Vec<TimedPoint> = points
            .iter()
            .map(|&position| TimedPoint {
                position,
                time: 0.0,
            })
            .collect();
        self.push_range(sensor_id, time.into(), &points, &[])
    }

    fn push_range(
        &mut self,
        sensor_id: &str,
        time: Timestamp,
        points: &[TimedPoint],
        intensities: &[f32],
    ) -> Result<(), SlamError> {
        self.check_accepting()?;
        let Some(&tracking_from_sensor) = self.sensors.lidars.get(sensor_id) else {
            return Err(SlamError::UnknownSensor {
                id: sensor_id.to_string(),
            });
        };
        if points.is_empty() {
            return Ok(());
        }
        if self.use_intensities && intensities.is_empty() {
            return Err(SlamError::InvalidData(
                "use_intensities is enabled: use push_lidar_with_intensities".to_string(),
            ));
        }
        let mut previous_time = f32::NEG_INFINITY;
        for (index, point) in points.iter().enumerate() {
            if !point.position.iter().all(|v| v.is_finite()) || !point.time.is_finite() {
                return Err(SlamError::InvalidData(format!(
                    "point {index} of '{sensor_id}' is not finite"
                )));
            }
            if point.time < previous_time {
                return Err(SlamError::InvalidData(format!(
                    "point times of '{sensor_id}' must be non-decreasing (point {index}); see \
                     points_relative_to_first / points_with_absolute_times"
                )));
            }
            previous_time = point.time;
        }
        if previous_time > 0.0 {
            return Err(SlamError::InvalidData(format!(
                "the last point of '{sensor_id}' has time {previous_time} > 0: the message time \
                 must be the time of the last point (see points_relative_to_first)"
            )));
        }
        self.check_time(sensor_id, time)?;

        let data = to_point_cloud_data(&tracking_from_sensor, time, points, intensities);
        self.last_times.insert(sensor_id.to_string(), time);
        self.trajectory_builder()
            .add_timed_point_cloud_data(sensor_id, &data);
        Ok(())
    }

    /// Adds an IMU sample (in the IMU frame; rotated into the tracking
    /// frame here).
    pub fn push_imu(
        &mut self,
        time: impl Into<Timestamp>,
        sample: ImuSample,
    ) -> Result<(), SlamError> {
        self.check_accepting()?;
        let time = time.into();
        let Some((id, tracking_from_imu)) = self.sensors.imu.clone() else {
            return Err(SlamError::SensorNotConfigured {
                kind: SensorKind::Imu,
            });
        };
        if !sample
            .linear_acceleration
            .iter()
            .chain(sample.angular_velocity.iter())
            .all(|v| v.is_finite())
        {
            return Err(SlamError::InvalidData(format!(
                "IMU sample of '{id}' is not finite"
            )));
        }
        self.check_time(&id, time)?;
        let data = to_imu_data(&tracking_from_imu, time, &sample);
        self.last_times.insert(id.clone(), time);
        self.trajectory_builder().add_imu_data(&id, &data);
        Ok(())
    }

    /// Adds an odometry pose: the pose of the odometry frame (the tracking
    /// frame unless declared with
    /// [`odometry_with_extrinsics`](SlamBuilder::odometry_with_extrinsics))
    /// in the odometry world frame. Only relative motion is used.
    pub fn push_odometry(
        &mut self,
        time: impl Into<Timestamp>,
        pose: Isometry3<f64>,
    ) -> Result<(), SlamError> {
        self.check_accepting()?;
        let time = time.into();
        let Some((id, sensor_from_tracking)) = self.sensors.odometry.clone() else {
            return Err(SlamError::SensorNotConfigured {
                kind: SensorKind::Odometry,
            });
        };
        if !pose.translation.vector.iter().all(|v| v.is_finite())
            || !pose.rotation.coords.iter().all(|v| v.is_finite())
        {
            return Err(SlamError::InvalidData(format!(
                "odometry pose of '{id}' is not finite"
            )));
        }
        self.check_time(&id, time)?;
        let data = to_odometry_data(&sensor_from_tracking, time, &pose);
        self.last_times.insert(id.clone(), time);
        self.trajectory_builder().add_odometry_data(&id, &data);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Callbacks
    // ---------------------------------------------------------------------

    /// Registers `callback` to run after local SLAM matched an accumulated
    /// scan. It runs synchronously inside the `push_*` call that completed
    /// the scan.
    pub fn on_local_result(&self, callback: impl FnMut(&LocalResult) + Send + 'static) {
        self.shared
            .local_callbacks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::new(callback));
    }

    /// Registers `callback` to run after each global optimization (loop
    /// closure). It runs on a background thread while Cartographer holds
    /// internal locks: keep it short and do not block on the `Slam`.
    pub fn on_global_optimization(
        &self,
        callback: impl FnMut(&OptimizationEvent) + Send + 'static,
    ) {
        self.shared
            .optimization_callbacks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::new(callback));
    }

    // ---------------------------------------------------------------------
    // Control
    // ---------------------------------------------------------------------

    /// Waits for all pending loop closure searches and runs a full
    /// optimization (`max_num_final_iterations`). Can be called at any time,
    /// e.g. before exporting a map mid-run.
    pub fn optimize(&self) {
        self.pose_graph.run_final_optimization();
    }

    /// Ends the trajectory (flushing buffered sensor data) and runs the
    /// final optimization. Further `push_*` calls fail with
    /// [`SlamError::TrajectoryFinished`]. Calling it twice is a no-op.
    ///
    /// With pure localization ([`SlamBuilder::localize_in`]) Cartographer
    /// then trims the whole localization trajectory: read its results
    /// before (after [`optimize`](Self::optimize) if needed).
    pub fn finish(&mut self) -> Result<(), SlamError> {
        if self.finished {
            return Ok(());
        }
        self.map_builder.finish_trajectory(self.trajectory_id);
        self.pose_graph.run_final_optimization();
        self.finished = true;
        Ok(())
    }

    /// True after [`finish`](Self::finish).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Saves the full SLAM state (submaps, nodes, constraints, sensor data
    /// for optimization) as a Cartographer `.pbstream` file, readable by
    /// cartographer_ros tools and by [`SlamBuilder::localize_in`]. Call
    /// [`finish`](Self::finish) first to save the optimized final state.
    pub fn save(&mut self, path: impl AsRef<Path>) -> Result<(), SlamError> {
        let path = path.as_ref();
        let filename = path.to_str().ok_or_else(|| {
            SlamError::io(
                path,
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "path is not UTF-8"),
            )
        })?;
        if self
            .map_builder
            .serialize_state_to_file(/*include_unfinished_submaps=*/ true, filename)
        {
            Ok(())
        } else {
            Err(SlamError::io(
                path,
                std::io::Error::other("failed to write the pbstream"),
            ))
        }
    }

    // ---------------------------------------------------------------------
    // Results
    // ---------------------------------------------------------------------

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Id of the trajectory this instance feeds (0 unless a map was loaded).
    pub fn trajectory_id(&self) -> i32 {
        self.trajectory_id
    }

    /// The option protos the map builder and trajectory were created with.
    pub fn options(&self) -> (&MapBuilderOptions, &TrajectoryBuilderOptions) {
        (&self.options.0, &self.options.1)
    }

    fn node_poses(
        &self,
        trajectory_id: Option<i32>,
        local: bool,
    ) -> BTreeMap<i32, Vec<StampedPose>> {
        let mut trajectories: BTreeMap<i32, Vec<StampedPose>> = BTreeMap::new();
        for node in self.pose_graph.get_trajectory_node_poses().iter() {
            if trajectory_id.is_some_and(|id| id != node.id.trajectory_id) {
                continue;
            }
            let Some(constant) = &node.data.constant_pose_data else {
                continue;
            };
            let pose = if local {
                &constant.local_pose
            } else {
                &node.data.global_pose
            };
            trajectories
                .entry(node.id.trajectory_id)
                .or_default()
                .push(StampedPose {
                    time: Timestamp::from_cartographer_time(constant.time),
                    pose: to_isometry(pose),
                });
        }
        trajectories
    }

    /// Optimized global poses (map_from_tracking) of the nodes of this
    /// instance's trajectory, in time order. Loop closures update them in
    /// the background; [`optimize`](Self::optimize) or
    /// [`finish`](Self::finish) bring them up to date.
    pub fn trajectory(&self) -> Vec<StampedPose> {
        self.node_poses(Some(self.trajectory_id), false)
            .remove(&self.trajectory_id)
            .unwrap_or_default()
    }

    /// Local SLAM poses (local_from_tracking, before loop closure) of the
    /// nodes of this instance's trajectory.
    pub fn local_trajectory(&self) -> Vec<StampedPose> {
        self.node_poses(Some(self.trajectory_id), true)
            .remove(&self.trajectory_id)
            .unwrap_or_default()
    }

    /// Optimized global node poses of all trajectories (including loaded
    /// ones), by trajectory id.
    pub fn all_trajectories(&self) -> BTreeMap<i32, Vec<StampedPose>> {
        self.node_poses(None, false)
    }

    /// The global pose of this instance's trajectory at `time`,
    /// interpolated between the two surrounding nodes (`None` outside the
    /// node time range).
    pub fn pose_at(&self, time: impl Into<Timestamp>) -> Option<Isometry3<f64>> {
        let time = time.into();
        let poses = self.trajectory();
        let after = poses.partition_point(|p| p.time < time);
        let next = poses.get(after)?;
        if next.time == time {
            return Some(next.pose);
        }
        let previous = poses.get(after.checked_sub(1)?)?;
        let fraction = time.seconds_since(previous.time) / next.time.seconds_since(previous.time);
        Some(previous.pose.lerp_slerp(&next.pose, fraction))
    }

    /// The latest local SLAM pose mapped into the global frame with the
    /// current local-to-global correction: the best real-time estimate of
    /// where the tracking frame is now. `None` before the first local
    /// result.
    pub fn current_pose(&self) -> Option<StampedPose> {
        let (time, local_pose) = (*self
            .shared
            .latest_local_pose
            .lock()
            .unwrap_or_else(PoisonError::into_inner))?;
        let global = self
            .pose_graph
            .get_local_to_global_transform(self.trajectory_id)
            * local_pose;
        Some(StampedPose {
            time: Timestamp::from_cartographer_time(time),
            pose: to_isometry(&global),
        })
    }

    /// All submaps (all trajectories) with their optimized global poses.
    pub fn submaps(&self) -> Vec<SubmapInfo> {
        self.pose_graph
            .get_all_submap_data()
            .iter()
            .filter_map(|submap| {
                let handle = submap.data.submap.as_ref()?;
                let guard = handle.read().unwrap_or_else(PoisonError::into_inner);
                Some(SubmapInfo {
                    id: submap.id,
                    global_pose: to_isometry(&submap.data.pose),
                    num_range_data: guard.num_range_data(),
                    finished: guard.insertion_finished(),
                })
            })
            .collect()
    }

    /// Pose graph statistics (constraint counts, loop closures, ...).
    pub fn stats(&self) -> SlamStats {
        let constraints = self.pose_graph.constraints();
        let mut first_insertion_submap: HashMap<NodeId, i32> = HashMap::new();
        for constraint in constraints.iter().filter(|c| c.tag == Tag::IntraSubmap) {
            let index = first_insertion_submap
                .entry(constraint.node_id)
                .or_insert(i32::MAX);
            *index = (*index).min(constraint.submap_id.submap_index);
        }
        let mut stats = SlamStats {
            num_nodes: self.pose_graph.get_trajectory_node_poses().size(),
            num_submaps: self.pose_graph.get_all_submap_poses().size(),
            num_constraints: constraints.len(),
            num_local_results: self.shared.num_local_results.load(Ordering::Relaxed),
            num_global_optimizations: self.shared.num_optimizations.load(Ordering::Relaxed),
            ..SlamStats::default()
        };
        for constraint in &constraints {
            match constraint.tag {
                Tag::IntraSubmap => stats.num_intra_submap_constraints += 1,
                Tag::InterSubmap => {
                    stats.num_inter_submap_constraints += 1;
                    let SubmapId {
                        trajectory_id,
                        submap_index,
                    } = constraint.submap_id;
                    if trajectory_id != constraint.node_id.trajectory_id {
                        stats.num_inter_trajectory_constraints += 1;
                    } else if first_insertion_submap
                        .get(&constraint.node_id)
                        .is_some_and(|&first| submap_index + 1 < first)
                    {
                        stats.num_loop_closure_constraints += 1;
                    }
                }
            }
        }
        stats
    }

    /// The 2D occupancy grid of all submaps at their optimized global poses
    /// (2D mode only), at the submap resolution. Overlapping submaps are
    /// fused in log-odds.
    pub fn map_2d(&self) -> Result<OccupancyGrid, SlamError> {
        if self.mode != Mode::TwoD {
            return Err(SlamError::Unsupported(
                "map_2d needs 2D mode; use map_3d for 3D maps".to_string(),
            ));
        }
        compose_occupancy_grid(&self.pose_graph.get_all_submap_data())
    }

    /// The occupied voxels (probability >=
    /// [`DEFAULT_MIN_OCCUPIED_PROBABILITY_3D`]) of all high-resolution 3D
    /// submaps at their optimized global poses (3D mode only).
    pub fn map_3d(&self) -> Result<PointMap, SlamError> {
        self.map_3d_with_min_probability(DEFAULT_MIN_OCCUPIED_PROBABILITY_3D)
    }

    /// Like [`map_3d`](Self::map_3d) with a custom occupancy threshold.
    pub fn map_3d_with_min_probability(&self, min_probability: f32) -> Result<PointMap, SlamError> {
        if self.mode != Mode::ThreeD {
            return Err(SlamError::Unsupported(
                "map_3d needs 3D mode; use map_2d for 2D maps".to_string(),
            ));
        }
        compose_point_map(&self.pose_graph.get_all_submap_data(), min_probability)
    }

    // ---------------------------------------------------------------------
    // Escape hatches
    // ---------------------------------------------------------------------

    /// The wrapped faithful [`MapBuilder`] (read access).
    pub fn map_builder(&self) -> &MapBuilder {
        &self.map_builder
    }

    /// The wrapped faithful [`MapBuilder`]. Feeding data through it bypasses
    /// the validation and frame transforms of this API.
    pub fn map_builder_mut(&mut self) -> &mut MapBuilder {
        &mut self.map_builder
    }

    /// The pose graph (read API of the faithful layer).
    pub fn pose_graph(&self) -> Arc<dyn PoseGraphInterface> {
        Arc::clone(&self.pose_graph)
    }
}

impl Drop for Slam {
    /// Finishes an unfinished trajectory (without the final optimization):
    /// Cartographer requires all sensor queues to be finished before the
    /// map builder is destroyed.
    fn drop(&mut self) {
        if !self.finished && !std::thread::panicking() {
            self.map_builder.finish_trajectory(self.trajectory_id);
        }
    }
}

/// Transforms a scan into the tracking frame (points and origin by
/// tracking_from_sensor, like cartographer_ros' `SensorBridge`).
pub(crate) fn to_point_cloud_data(
    tracking_from_sensor: &Isometry3<f32>,
    time: Timestamp,
    points: &[TimedPoint],
    intensities: &[f32],
) -> TimedPointCloudData {
    TimedPointCloudData {
        time: time.to_cartographer_time(),
        origin: tracking_from_sensor.translation.vector,
        ranges: points
            .iter()
            .map(|point| TimedRangefinderPoint {
                position: (tracking_from_sensor * Point3::from(point.position)).coords,
                time: point.time,
            })
            .collect(),
        intensities: intensities.to_vec(),
    }
}

/// Rotates an IMU sample into the tracking frame.
pub(crate) fn to_imu_data(
    tracking_from_imu: &UnitQuaternion<f64>,
    time: Timestamp,
    sample: &ImuSample,
) -> ImuData {
    ImuData {
        time: time.to_cartographer_time(),
        linear_acceleration: tracking_from_imu * sample.linear_acceleration,
        angular_velocity: tracking_from_imu * sample.angular_velocity,
    }
}

/// Converts the pose of the odometry frame into the pose of the tracking
/// frame: `odom_from_sensor * sensor_from_tracking`.
pub(crate) fn to_odometry_data(
    sensor_from_tracking: &Isometry3<f64>,
    time: Timestamp,
    pose: &Isometry3<f64>,
) -> OdometryData {
    OdometryData {
        time: time.to_cartographer_time(),
        pose: to_rigid3d(&(pose * sensor_from_tracking)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slam::types::Extrinsics;

    fn imu_at_rest() -> ImuSample {
        ImuSample {
            linear_acceleration: Vector3::new(0.0, 0.0, 9.81),
            angular_velocity: Vector3::zeros(),
        }
    }

    fn scan() -> Vec<TimedPoint> {
        (0..180)
            .map(|i| {
                let a = (i as f32 * 2.0).to_radians();
                TimedPoint::new(3.0 * a.cos(), 3.0 * a.sin(), 0.0, (i as f32 - 179.0) * 1e-4)
            })
            .collect()
    }

    fn slam_2d() -> Slam {
        Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .imu("imu", Extrinsics::identity())
            .odometry("odom")
            .config(|c| c.num_background_threads = 1)
            .build()
            .unwrap()
    }

    #[test]
    fn builder_errors() {
        let error = Slam::builder(Mode::TwoD).build().unwrap_err();
        assert!(matches!(error, SlamError::MissingSensor(_)), "{error}");
        let error = Slam::builder(Mode::ThreeD)
            .lidar("points", Extrinsics::identity())
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::MissingSensor(_)), "{error}");
        let error = Slam::builder(Mode::TwoD)
            .lidar("a", Extrinsics::identity())
            .lidar("a", Extrinsics::identity())
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::DuplicateSensor(_)), "{error}");
        let error = Slam::builder(Mode::TwoD)
            .lidar("a", Extrinsics::identity())
            .imu("i1", Extrinsics::identity())
            .imu("i2", Extrinsics::identity())
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::DuplicateSensor(_)), "{error}");
        let error = Slam::builder(Mode::TwoD)
            .lidar("a", Extrinsics::identity())
            .odometry("a")
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::DuplicateSensor(_)), "{error}");
        // IMU lever arm: an error in 3D, accepted (with a warning) in 2D.
        let error = Slam::builder(Mode::ThreeD)
            .lidar("points", Extrinsics::identity())
            .imu("imu", Extrinsics::from_translation(0.0, 0.0, 0.1))
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::InvalidConfig(_)), "{error}");
        Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .imu("imu", Extrinsics::from_translation(0.0, 0.0, 0.1))
            .build()
            .unwrap();
        // Invalid knob.
        let error = Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .config(|c| c.resolution = -1.0)
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::InvalidConfig(_)), "{error}");
        // Advanced edit contradicting the declared sensors.
        let error = Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .config(|c| {
                c.advanced(|_, tb| {
                    tb.trajectory_builder_2d_options
                        .as_mut()
                        .unwrap()
                        .use_imu_data = true;
                });
            })
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::InvalidConfig(_)), "{error}");
        // Missing map file.
        let error = Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .localize_in("/nonexistent/map.pbstream")
            .build()
            .unwrap_err();
        assert!(matches!(error, SlamError::Io { .. }), "{error}");
    }

    #[test]
    fn derived_options() {
        let slam = slam_2d();
        let (mb, tb) = slam.options();
        assert!(mb.use_trajectory_builder_2d);
        assert!(
            tb.trajectory_builder_2d_options
                .as_ref()
                .unwrap()
                .use_imu_data
        );
        assert!(tb.pure_localization_trimmer.is_none());
        let slam = Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .build()
            .unwrap();
        assert!(
            !slam
                .options()
                .1
                .trajectory_builder_2d_options
                .as_ref()
                .unwrap()
                .use_imu_data
        );
    }

    #[test]
    fn input_errors_leave_the_instance_usable() {
        let mut slam = slam_2d();
        assert!(matches!(
            slam.push_lidar("laser", 0.0, &scan()),
            Err(SlamError::UnknownSensor { .. })
        ));
        slam.push_imu(0.0, imu_at_rest()).unwrap();
        assert!(matches!(
            slam.push_imu(0.0, imu_at_rest()),
            Err(SlamError::NonMonotonicTime { .. })
        ));
        assert!(matches!(
            slam.push_imu(
                0.1,
                ImuSample {
                    linear_acceleration: Vector3::new(f64::NAN, 0.0, 0.0),
                    angular_velocity: Vector3::zeros()
                }
            ),
            Err(SlamError::InvalidData(_))
        ));
        let mut bad = scan();
        bad.last_mut().unwrap().time = 0.01;
        assert!(matches!(
            slam.push_lidar("scan", 0.1, &bad),
            Err(SlamError::InvalidData(_))
        ));
        let mut unsorted = scan();
        unsorted.swap(0, 5);
        assert!(matches!(
            slam.push_lidar("scan", 0.1, &unsorted),
            Err(SlamError::InvalidData(_))
        ));
        assert!(matches!(
            slam.push_lidar_with_intensities("scan", 0.1, &scan(), &[1.0]),
            Err(SlamError::InvalidData(_))
        ));
        // Empty scans are ignored and do not advance the sensor clock.
        slam.push_lidar("scan", 0.1, &[]).unwrap();
        slam.push_lidar("scan", 0.1, &scan()).unwrap();
        assert!(matches!(
            slam.push_lidar("scan", 0.1, &scan()),
            Err(SlamError::NonMonotonicTime { .. })
        ));
        slam.push_odometry(0.1, Isometry3::identity()).unwrap();
        slam.finish().unwrap();
        slam.finish().unwrap();
        assert!(slam.is_finished());
        assert!(matches!(
            slam.push_imu(1.0, imu_at_rest()),
            Err(SlamError::TrajectoryFinished)
        ));
        assert!(matches!(slam.map_3d(), Err(SlamError::Unsupported(_))));
    }

    #[test]
    fn undeclared_imu_and_odometry() {
        let mut slam = Slam::builder(Mode::TwoD)
            .lidar("scan", Extrinsics::identity())
            .build()
            .unwrap();
        assert!(matches!(
            slam.push_imu(0.0, imu_at_rest()),
            Err(SlamError::SensorNotConfigured {
                kind: SensorKind::Imu
            })
        ));
        assert!(matches!(
            slam.push_odometry(0.0, Isometry3::identity()),
            Err(SlamError::SensorNotConfigured {
                kind: SensorKind::Odometry
            })
        ));
    }

    #[test]
    fn data_are_transformed_into_the_tracking_frame() {
        let t = Timestamp::from_secs_f64(2.0);
        // LiDAR turned by 90 degrees of yaw, mounted at (0.2, -0.1, 0.3).
        let lidar = Extrinsics::from_xyz_rpy(0.2, -0.1, 0.3, 0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let mut slam = Slam::builder(Mode::TwoD)
            .lidar("scan", lidar)
            .imu(
                "imu",
                Extrinsics::from_xyz_rpy(0.0, 0.0, 0.0, std::f64::consts::PI, 0.0, 0.0),
            )
            .odometry_with_extrinsics("odom", Extrinsics::from_translation(-0.5, 0.0, 0.0))
            .build()
            .unwrap();
        let tracking_from_lidar = slam.sensors.lidars["scan"];
        let cloud = to_point_cloud_data(
            &tracking_from_lidar,
            t,
            &[
                TimedPoint::new(1.0, 0.0, 0.0, -0.1),
                TimedPoint::new(0.0, 2.0, 0.0, 0.0),
            ],
            &[],
        );
        assert_eq!(cloud.time, t.to_cartographer_time());
        assert!((cloud.origin - Vector3::new(0.2, -0.1, 0.3)).norm() < 1e-6);
        assert!((cloud.ranges[0].position - Vector3::new(0.2, 0.9, 0.3)).norm() < 1e-6);
        assert!((cloud.ranges[1].position - Vector3::new(-1.8, -0.1, 0.3)).norm() < 1e-6);
        assert_eq!(cloud.ranges[0].time, -0.1);
        assert!(cloud.intensities.is_empty());

        // An upside-down IMU (roll 180 degrees) reads -g and a negated yaw
        // rate; in the tracking frame they are +g and the true yaw rate.
        let (_, tracking_from_imu) = slam.sensors.imu.clone().unwrap();
        let imu = to_imu_data(
            &tracking_from_imu,
            t,
            &ImuSample {
                linear_acceleration: Vector3::new(0.0, 0.0, -9.81),
                angular_velocity: Vector3::new(0.0, 0.0, -0.5),
            },
        );
        assert_eq!(imu.time, t.to_cartographer_time());
        assert!((imu.linear_acceleration - Vector3::new(0.0, 0.0, 9.81)).norm() < 1e-9);
        assert!((imu.angular_velocity - Vector3::new(0.0, 0.0, 0.5)).norm() < 1e-9);

        // Odometry of a frame 0.5 m behind the tracking frame, turned by 90
        // degrees: the tracking frame is 0.5 m ahead along the new heading.
        let (_, sensor_from_tracking) = slam.sensors.odometry.clone().unwrap();
        let odom_pose = Isometry3::new(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::z() * std::f64::consts::FRAC_PI_2,
        );
        let odometry = to_odometry_data(&sensor_from_tracking, t, &odom_pose);
        assert!((odometry.pose.translation() - Vector3::new(1.0, 0.5, 0.0)).norm() < 1e-12);

        // The push methods accept these data.
        slam.push_imu(0.0, imu_at_rest()).unwrap();
        slam.push_odometry(0.0, odom_pose).unwrap();
        slam.push_lidar("scan", 0.05, &scan()).unwrap();
    }
}
