//! Port of `cartographer/mapping/internal/2d/local_trajectory_builder_2d` (.h, .cc).

use std::sync::{Arc, PoisonError, RwLock};
use std::time::Instant;

use ceres::SolverSummary;
use nalgebra::{DVector, Quaternion};

use crate::common::time::{Duration, Time, from_seconds, get_thread_cpu_time_seconds, to_seconds};
use crate::mapping::internal::motion_filter::MotionFilter;
use crate::mapping::internal::range_data_collator::RangeDataCollator;
use crate::mapping::internal::two_d::scan_matching::ceres_scan_matcher_2d::CeresScanMatcher2D;
use crate::mapping::internal::two_d::scan_matching::real_time_correlative_scan_matcher_2d::RealTimeCorrelativeScanMatcher2D;
use crate::mapping::pose_extrapolator::PoseExtrapolator;
use crate::mapping::pose_extrapolator_interface::PoseExtrapolatorInterface;
use crate::mapping::proto;
use crate::mapping::submaps::SharedSubmap;
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::mapping::two_d::submap_2d::{ActiveSubmaps2D, Submap2D};
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::metrics::gauge::Gauge;
use crate::metrics::histogram::Histogram;
use crate::sensor::imu_data::ImuData;
use crate::sensor::internal::voxel_filter::{adaptive_voxel_filter, voxel_filter};
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::range_data::{RangeData, crop_range_data, transform_range_data};
use crate::sensor::rangefinder_point::to_rangefinder_point;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::{Rigid2d, Rigid3d, Rigid3f};
use crate::transform::transform::{embed_3d, project_2d};

/// The metrics of the local trajectory builder (C++ file-level statics,
/// null metrics until [`LocalTrajectoryBuilder2D::register_metrics`]).
#[derive(Clone, Copy)]
struct Metrics {
    local_slam_latency: &'static dyn Gauge,
    local_slam_real_time_ratio: &'static dyn Gauge,
    local_slam_cpu_real_time_ratio: &'static dyn Gauge,
    real_time_correlative_scan_matcher_score: &'static dyn Histogram,
    ceres_scan_matcher_cost: &'static dyn Histogram,
    scan_matcher_residual_distance: &'static dyn Histogram,
    scan_matcher_residual_angle: &'static dyn Histogram,
}

// RUST PORT NOTE: the C++ `static auto* k...Metric` globals; `None` means the
// null metrics.
static METRICS: RwLock<Option<Metrics>> = RwLock::new(None);

fn metrics() -> Metrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(|| Metrics {
            local_slam_latency: <dyn Gauge>::null(),
            local_slam_real_time_ratio: <dyn Gauge>::null(),
            local_slam_cpu_real_time_ratio: <dyn Gauge>::null(),
            real_time_correlative_scan_matcher_score: <dyn Histogram>::null(),
            ceres_scan_matcher_cost: <dyn Histogram>::null(),
            scan_matcher_residual_distance: <dyn Histogram>::null(),
            scan_matcher_residual_angle: <dyn Histogram>::null(),
        })
}

/// Result of inserting range data into the active submaps.
#[derive(Clone, Debug)]
pub struct InsertionResult {
    /// The constant data of the new trajectory node.
    pub constant_data: Arc<TrajectoryNodeData>,
    /// The submaps the range data was inserted into (front = older).
    pub insertion_submaps: Vec<SharedSubmap<Submap2D>>,
}

/// Result of local SLAM for one accumulated range data.
#[derive(Clone, Debug)]
pub struct MatchingResult {
    /// Time of the (last) range data.
    pub time: Time,
    /// Pose of the tracking frame in the local SLAM frame.
    pub local_pose: Rigid3d,
    /// The (voxel filtered) range data in the local SLAM frame.
    pub range_data_in_local: RangeData,
    /// `None` if dropped by the motion filter.
    pub insertion_result: Option<InsertionResult>,
}

/// Wires up the local SLAM stack (i.e. pose extrapolator, scan matching,
/// etc.) without loop closure.
pub struct LocalTrajectoryBuilder2D {
    options: proto::LocalTrajectoryBuilderOptions2D,
    active_submaps: ActiveSubmaps2D,

    motion_filter: MotionFilter,
    real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher2D,
    ceres_scan_matcher: CeresScanMatcher2D,

    // RUST PORT NOTE: `std::unique_ptr<PoseExtrapolator>` in C++.
    extrapolator: Option<PoseExtrapolator>,

    num_accumulated: i32,
    accumulated_range_data: RangeData,

    last_wall_time: Option<Instant>,
    last_thread_cpu_time_seconds: Option<f64>,
    last_sensor_time: Option<Time>,

    range_data_collator: RangeDataCollator,
}

impl std::fmt::Debug for LocalTrajectoryBuilder2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalTrajectoryBuilder2D")
            .field("options", &self.options)
            .field("active_submaps", &self.active_submaps)
            .field("num_accumulated", &self.num_accumulated)
            .field("last_sensor_time", &self.last_sensor_time)
            .finish_non_exhaustive()
    }
}

impl LocalTrajectoryBuilder2D {
    /// Creates a builder that expects range data from
    /// `expected_range_sensor_ids`.
    pub fn new<S: AsRef<str>>(
        options: &proto::LocalTrajectoryBuilderOptions2D,
        expected_range_sensor_ids: &[S],
    ) -> Self {
        Self {
            options: *options,
            active_submaps: ActiveSubmaps2D::new(&options.submaps_options.unwrap_or_default()),
            motion_filter: MotionFilter::new(&options.motion_filter_options.unwrap_or_default()),
            real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher2D::new(
                &options
                    .real_time_correlative_scan_matcher_options
                    .unwrap_or_default(),
            ),
            ceres_scan_matcher: CeresScanMatcher2D::new(
                &options.ceres_scan_matcher_options.unwrap_or_default(),
            ),
            extrapolator: None,
            num_accumulated: 0,
            accumulated_range_data: RangeData::default(),
            last_wall_time: None,
            last_thread_cpu_time_seconds: None,
            last_sensor_time: None,
            range_data_collator: RangeDataCollator::new(expected_range_sensor_ids),
        }
    }

    /// Returns a [`MatchingResult`] when range data accumulation completed,
    /// otherwise `None`. Range data must be approximately horizontal for 2D
    /// SLAM. `TimedPointCloudData::time` is when the last point in
    /// `range_data` was acquired, `TimedPointCloudData::ranges` contains the
    /// relative time of point with respect to `TimedPointCloudData::time`.
    pub fn add_range_data(
        &mut self,
        sensor_id: &str,
        unsynchronized_data: TimedPointCloudData,
    ) -> Option<MatchingResult> {
        let synchronized_data = self
            .range_data_collator
            .add_range_data(sensor_id, unsynchronized_data);
        if synchronized_data.ranges.is_empty() {
            log::info!("Range data collator filling buffer.");
            return None;
        }

        let time = synchronized_data.time;
        // Initialize extrapolator now if we do not ever use an IMU.
        if !self.options.use_imu_data {
            self.initialize_extrapolator(time);
        }

        let Some(extrapolator) = self.extrapolator.as_mut() else {
            // Until we've initialized the extrapolator with our first IMU
            // message, we cannot compute the orientation of the rangefinder.
            log::info!("Extrapolator not yet initialized.");
            return None;
        };

        // TODO(gaschler): Check if this can strictly be 0.
        let last_point_time = synchronized_data
            .ranges
            .last()
            .map_or(0.0, |r| r.point_time.time);
        assert!(
            last_point_time <= 0.0,
            "Check failed: synchronized_data.ranges.back().point_time.time <= 0.f ({last_point_time} vs. 0)"
        );
        let time_first_point =
            time + from_seconds(f64::from(synchronized_data.ranges[0].point_time.time));
        if time_first_point < extrapolator.get_last_pose_time() {
            log::info!("Extrapolator is still initializing.");
            return None;
        }

        let mut range_data_poses: Vec<Rigid3f> = Vec::with_capacity(synchronized_data.ranges.len());
        let mut warned = false;
        for range in &synchronized_data.ranges {
            let mut time_point = time + from_seconds(f64::from(range.point_time.time));
            if time_point < extrapolator.get_last_extrapolated_time() {
                if !warned {
                    log::error!(
                        "Timestamp of individual range data point jumps backwards from {} to {}",
                        extrapolator.get_last_extrapolated_time(),
                        time_point
                    );
                    warned = true;
                }
                time_point = extrapolator.get_last_extrapolated_time();
            }
            range_data_poses.push(extrapolator.extrapolate_pose(time_point).cast::<f32>());
        }

        if self.num_accumulated == 0 {
            // 'accumulated_range_data.origin' is uninitialized until the
            // last accumulation (RUST PORT NOTE: zero here).
            self.accumulated_range_data = RangeData::default();
        }

        // Drop any returns below the minimum range and convert returns beyond
        // the maximum range into misses.
        for (range, range_data_pose) in synchronized_data.ranges.iter().zip(&range_data_poses) {
            let hit = &range.point_time;
            let origin_in_local = range_data_pose * &synchronized_data.origins[range.origin_index];
            let mut hit_in_local = range_data_pose * &to_rangefinder_point(hit);
            let delta = hit_in_local.position - origin_in_local;
            let range_length = delta.norm();
            if range_length >= self.options.min_range {
                if range_length <= self.options.max_range {
                    self.accumulated_range_data.returns.push_back(hit_in_local);
                } else {
                    hit_in_local.position = origin_in_local
                        + self.options.missing_data_ray_length / range_length * delta;
                    self.accumulated_range_data.misses.push_back(hit_in_local);
                }
            }
        }
        self.num_accumulated += 1;

        if self.num_accumulated >= self.options.num_accumulated_range_data {
            let current_sensor_time = synchronized_data.time;
            let sensor_duration: Option<Duration> = self
                .last_sensor_time
                .map(|last_sensor_time| current_sensor_time - last_sensor_time);
            self.last_sensor_time = Some(current_sensor_time);
            self.num_accumulated = 0;
            let gravity_alignment =
                Rigid3d::from_rotation(extrapolator.estimate_gravity_orientation(time));
            // TODO(gaschler): This assumes that 'range_data_poses.back()' is
            // at time 'time'.
            let last_range_data_pose = *range_data_poses
                .last()
                .expect("ranges are non-empty, so are the poses");
            self.accumulated_range_data.origin = *last_range_data_pose.translation();
            let gravity_aligned_range_data = self.transform_to_gravity_aligned_frame_and_filter(
                &(gravity_alignment.cast::<f32>() * last_range_data_pose.inverse()),
                &self.accumulated_range_data,
            );
            return self.add_accumulated_range_data(
                time,
                &gravity_aligned_range_data,
                &gravity_alignment,
                sensor_duration,
            );
        }
        None
    }

    /// Adds an IMU measurement. Panics unless `use_imu_data` is set.
    pub fn add_imu_data(&mut self, imu_data: &ImuData) {
        assert!(
            self.options.use_imu_data,
            "An unexpected IMU packet was added."
        );
        self.initialize_extrapolator(imu_data.time);
        self.extrapolator
            .as_mut()
            .expect("initialized above")
            .add_imu_data(imu_data);
    }

    /// Adds an odometry measurement (dropped until the extrapolator is
    /// initialized).
    pub fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        let Some(extrapolator) = self.extrapolator.as_mut() else {
            // Until we've initialized the extrapolator we cannot add odometry
            // data.
            log::info!("Extrapolator not yet initialized.");
            return;
        };
        extrapolator.add_odometry_data(odometry_data);
    }

    /// Registers the local SLAM metrics with `family_factory`.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        let labels = |pairs: &[(&str, &str)]| -> Labels {
            pairs
                .iter()
                .map(|&(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let latency = family_factory.new_gauge_family(
            "mapping_2d_local_trajectory_builder_latency",
            "Duration from first incoming point cloud in accumulation to local slam result",
        );
        let local_slam_latency = latency.add(&Labels::new());
        let real_time_ratio = family_factory.new_gauge_family(
            "mapping_2d_local_trajectory_builder_real_time_ratio",
            "sensor duration / wall clock duration.",
        );
        let local_slam_real_time_ratio = real_time_ratio.add(&Labels::new());

        let cpu_real_time_ratio = family_factory.new_gauge_family(
            "mapping_2d_local_trajectory_builder_cpu_real_time_ratio",
            "sensor duration / cpu duration.",
        );
        let local_slam_cpu_real_time_ratio = cpu_real_time_ratio.add(&Labels::new());
        let score_boundaries = <dyn Histogram>::fixed_width(0.05, 20);
        let scores = family_factory.new_histogram_family(
            "mapping_2d_local_trajectory_builder_scores",
            "Local scan matcher scores",
            &score_boundaries,
        );
        let real_time_correlative_scan_matcher_score =
            scores.add(&labels(&[("scan_matcher", "real_time_correlative")]));
        let cost_boundaries = <dyn Histogram>::scaled_powers_of(2.0, 0.01, 100.0);
        let costs = family_factory.new_histogram_family(
            "mapping_2d_local_trajectory_builder_costs",
            "Local scan matcher costs",
            &cost_boundaries,
        );
        let ceres_scan_matcher_cost = costs.add(&labels(&[("scan_matcher", "ceres")]));
        let distance_boundaries = <dyn Histogram>::scaled_powers_of(2.0, 0.01, 10.0);
        let residuals = family_factory.new_histogram_family(
            "mapping_2d_local_trajectory_builder_residuals",
            "Local scan matcher residuals",
            &distance_boundaries,
        );
        let scan_matcher_residual_distance = residuals.add(&labels(&[("component", "distance")]));
        let scan_matcher_residual_angle = residuals.add(&labels(&[("component", "angle")]));
        *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(Metrics {
            local_slam_latency,
            local_slam_real_time_ratio,
            local_slam_cpu_real_time_ratio,
            real_time_correlative_scan_matcher_score,
            ceres_scan_matcher_cost,
            scan_matcher_residual_distance,
            scan_matcher_residual_angle,
        });
    }

    fn add_accumulated_range_data(
        &mut self,
        time: Time,
        gravity_aligned_range_data: &RangeData,
        gravity_alignment: &Rigid3d,
        sensor_duration: Option<Duration>,
    ) -> Option<MatchingResult> {
        if gravity_aligned_range_data.returns.is_empty() {
            log::warn!("Dropped empty horizontal range data.");
            return None;
        }
        let extrapolator = self
            .extrapolator
            .as_mut()
            .expect("extrapolator is initialized");

        // Computes a gravity aligned pose prediction.
        let non_gravity_aligned_pose_prediction = extrapolator.extrapolate_pose(time);
        let pose_prediction =
            project_2d(&(non_gravity_aligned_pose_prediction * gravity_alignment.inverse()));

        let filtered_gravity_aligned_point_cloud = adaptive_voxel_filter(
            &gravity_aligned_range_data.returns,
            &self
                .options
                .adaptive_voxel_filter_options
                .unwrap_or_default(),
        );
        if filtered_gravity_aligned_point_cloud.is_empty() {
            return None;
        }

        // local map frame <- gravity-aligned frame
        let Some(pose_estimate_2d) = self.scan_match(
            time,
            &pose_prediction,
            &filtered_gravity_aligned_point_cloud,
        ) else {
            log::warn!("Scan matching failed.");
            return None;
        };
        let pose_estimate = embed_3d(&pose_estimate_2d) * *gravity_alignment;
        self.extrapolator
            .as_mut()
            .expect("extrapolator is initialized")
            .add_pose(time, &pose_estimate);

        let range_data_in_local = transform_range_data(
            gravity_aligned_range_data,
            &embed_3d(&pose_estimate_2d.cast::<f32>()),
        );
        let insertion_result = self.insert_into_submap(
            time,
            &range_data_in_local,
            &filtered_gravity_aligned_point_cloud,
            &pose_estimate,
            gravity_alignment.rotation(),
        );

        let metrics = metrics();
        let wall_time = Instant::now();
        if let Some(last_wall_time) = self.last_wall_time {
            let wall_time_duration = (wall_time - last_wall_time).as_secs_f64();
            metrics.local_slam_latency.set(wall_time_duration);
            if let Some(sensor_duration) = sensor_duration {
                metrics
                    .local_slam_real_time_ratio
                    .set(to_seconds(sensor_duration) / wall_time_duration);
            }
        }
        let thread_cpu_time_seconds = get_thread_cpu_time_seconds();
        if let Some(last_thread_cpu_time_seconds) = self.last_thread_cpu_time_seconds {
            let thread_cpu_duration_seconds =
                thread_cpu_time_seconds - last_thread_cpu_time_seconds;
            if let Some(sensor_duration) = sensor_duration {
                metrics
                    .local_slam_cpu_real_time_ratio
                    .set(to_seconds(sensor_duration) / thread_cpu_duration_seconds);
            }
        }
        self.last_wall_time = Some(wall_time);
        self.last_thread_cpu_time_seconds = Some(thread_cpu_time_seconds);
        Some(MatchingResult {
            time,
            local_pose: pose_estimate,
            range_data_in_local,
            insertion_result,
        })
    }

    fn transform_to_gravity_aligned_frame_and_filter(
        &self,
        transform_to_gravity_aligned_frame: &Rigid3f,
        range_data: &RangeData,
    ) -> RangeData {
        let cropped = crop_range_data(
            &transform_range_data(range_data, transform_to_gravity_aligned_frame),
            self.options.min_z,
            self.options.max_z,
        );
        RangeData {
            origin: cropped.origin,
            returns: voxel_filter(&cropped.returns, self.options.voxel_filter_size),
            misses: voxel_filter(&cropped.misses, self.options.voxel_filter_size),
        }
    }

    fn insert_into_submap(
        &mut self,
        time: Time,
        range_data_in_local: &RangeData,
        filtered_gravity_aligned_point_cloud: &PointCloud,
        pose_estimate: &Rigid3d,
        gravity_alignment: &Quaternion<f64>,
    ) -> Option<InsertionResult> {
        if self.motion_filter.is_similar(time, pose_estimate) {
            return None;
        }
        let insertion_submaps = self.active_submaps.insert_range_data(range_data_in_local);
        Some(InsertionResult {
            constant_data: Arc::new(TrajectoryNodeData {
                time,
                gravity_alignment: *gravity_alignment,
                filtered_gravity_aligned_point_cloud: filtered_gravity_aligned_point_cloud.clone(),
                // 'high_resolution_point_cloud' is only used in 3D.
                high_resolution_point_cloud: PointCloud::new(),
                // 'low_resolution_point_cloud' is only used in 3D.
                low_resolution_point_cloud: PointCloud::new(),
                // 'rotational_scan_matcher_histogram' is only used in 3D.
                rotational_scan_matcher_histogram: DVector::zeros(0),
                local_pose: *pose_estimate,
            }),
            insertion_submaps,
        })
    }

    // Scan matches 'filtered_gravity_aligned_point_cloud' and returns the
    // observed pose, or None on failure.
    fn scan_match(
        &self,
        _time: Time,
        pose_prediction: &Rigid2d,
        filtered_gravity_aligned_point_cloud: &PointCloud,
    ) -> Option<Rigid2d> {
        let submaps = self.active_submaps.submaps();
        let Some(matching_submap) = submaps.first() else {
            return Some(*pose_prediction);
        };
        let matching_submap = matching_submap
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let grid = matching_submap.grid();
        let metrics = metrics();
        // The online correlative scan matcher will refine the initial
        // estimate for the Ceres scan matcher.
        let mut initial_ceres_pose = *pose_prediction;

        if self.options.use_online_correlative_scan_matching {
            let (score, pose) = self.real_time_correlative_scan_matcher.match_(
                pose_prediction,
                filtered_gravity_aligned_point_cloud,
                grid,
            );
            initial_ceres_pose = pose;
            metrics
                .real_time_correlative_scan_matcher_score
                .observe(score);
        }

        let mut summary = SolverSummary::default();
        let pose_observation = self.ceres_scan_matcher.match_(
            pose_prediction.translation(),
            &initial_ceres_pose,
            filtered_gravity_aligned_point_cloud,
            grid,
            &mut summary,
        );
        metrics.ceres_scan_matcher_cost.observe(summary.final_cost);
        let residual_distance =
            (pose_observation.translation() - pose_prediction.translation()).norm();
        metrics
            .scan_matcher_residual_distance
            .observe(residual_distance);
        let residual_angle =
            (pose_observation.rotation().angle() - pose_prediction.rotation().angle()).abs();
        metrics.scan_matcher_residual_angle.observe(residual_angle);
        Some(pose_observation)
    }

    // Lazily constructs a PoseExtrapolator.
    fn initialize_extrapolator(&mut self, time: Time) {
        if self.extrapolator.is_some() {
            return;
        }
        let pose_extrapolator_options = self.options.pose_extrapolator_options.unwrap_or_default();
        assert!(
            !pose_extrapolator_options.use_imu_based,
            "Check failed: !options_.pose_extrapolator_options().use_imu_based()"
        );
        let constant_velocity = pose_extrapolator_options
            .constant_velocity
            .unwrap_or_default();
        // TODO(gaschler): Consider using InitializeWithImu as 3D does.
        let mut extrapolator = PoseExtrapolator::new(
            from_seconds(constant_velocity.pose_queue_duration),
            constant_velocity.imu_gravity_time_constant,
        );
        extrapolator.add_pose(time, &Rigid3d::identity());
        self.extrapolator = Some(extrapolator);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use nalgebra::Vector3;

    use super::*;
    use crate::mapping::internal::two_d::local_trajectory_builder_options_2d::testing::load_options_2d;
    use crate::metrics::counter::Counter;
    use crate::metrics::family_factory::Family;
    use crate::metrics::histogram::BucketBoundaries;
    use crate::sensor::rangefinder_point::TimedRangefinderPoint;
    use crate::transform::transform::get_angle;

    const NUM_BEAMS: usize = 1440;

    /// Range of a beam with direction `angle` from `(x, 0)` to the walls of
    /// the room `[-5.02, 6.01] x [-3.97, 4.04]` (walls off cell boundaries).
    fn range_to_walls(x: f32, angle: f32) -> f32 {
        let (s, c) = angle.sin_cos();
        let mut range = f32::INFINITY;
        if c > 1e-6 {
            range = range.min((6.01 - x) / c);
        }
        if c < -1e-6 {
            range = range.min((-5.02 - x) / c);
        }
        if s > 1e-6 {
            range = range.min(4.04 / s);
        }
        if s < -1e-6 {
            range = range.min(-3.97 / s);
        }
        range
    }

    /// A scan of the room from `(x, 0)`, heading +x, taken over 0.1 s.
    /// `extra_points` are appended (in the sensor frame, relative time 0).
    fn scan(time: Time, x: f32, extra_points: &[Vector3<f32>]) -> TimedPointCloudData {
        moving_scan(time, x, 0.0, extra_points)
    }

    /// Like [`scan`], but moving along +x with `velocity`; `x` is the
    /// position at `time` (the last beam). Each beam is measured from the
    /// position at its own time (motion distortion).
    fn moving_scan(
        time: Time,
        x: f32,
        velocity: f32,
        extra_points: &[Vector3<f32>],
    ) -> TimedPointCloudData {
        let mut ranges: Vec<TimedRangefinderPoint> = (0..NUM_BEAMS)
            .map(|i| {
                let angle = -PI + 2.0 * PI * i as f32 / NUM_BEAMS as f32;
                let relative_time = -0.1 + 0.1 * (i + 1) as f32 / NUM_BEAMS as f32;
                let range = range_to_walls(x + velocity * relative_time, angle);
                TimedRangefinderPoint {
                    position: Vector3::new(range * angle.cos(), range * angle.sin(), 0.0),
                    time: relative_time,
                }
            })
            .collect();
        ranges.extend(extra_points.iter().map(|&position| TimedRangefinderPoint {
            position,
            time: 0.0,
        }));
        TimedPointCloudData {
            time,
            origin: Vector3::zeros(),
            ranges,
            intensities: Vec::new(),
        }
    }

    fn time(seconds: f64) -> Time {
        Time::from_universal(1_000_000_000) + from_seconds(seconds)
    }

    // Not in C++ (there is no LocalTrajectoryBuilder2D test); see also the
    // simulation tests in `tests/local_trajectory_builder_2d_sim.rs`.
    mod local_trajectory_builder_2d_test {
        use super::*;

        #[test]
        fn first_scan_dropped_without_imu() {
            let options = load_options_2d("TRAJECTORY_BUILDER_2D.use_imu_data = false");
            let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
            // Odometry before initialization is dropped (no panic).
            builder.add_odometry_data(&OdometryData {
                time: time(0.0),
                pose: Rigid3d::identity(),
            });
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.1), 0.0, &[]))
                    .is_none()
            );
            for i in 2..60 {
                let result = builder
                    .add_range_data("scan", scan(time(0.1 * f64::from(i)), 0.0, &[]))
                    .expect("result");
                assert_eq!(result.time, time(0.1 * f64::from(i)));
                // Matching a stationary scan against its own map is accurate
                // to a fraction of the 5 cm cells (the bicubic cost has a
                // bias of up to ~1/4 cell at walls with free cells inside and
                // unknown cells outside).
                assert!(
                    result.local_pose.translation().norm() < 0.02,
                    "{}",
                    result.local_pose
                );
                assert!(
                    get_angle(&result.local_pose) < 1e-3,
                    "{}",
                    result.local_pose
                );
                // The motion filter lets the first node through and then one
                // every 5 s while stationary.
                assert_eq!(result.insertion_result.is_some(), i == 2 || i == 53, "{i}");
            }
        }

        /// Stationary for 1 s, then moving +x at 0.2 m/s (abrupt start)
        /// until t = 5 s. `transient_tolerance` applies in t in [1, 3).
        fn track_translation(use_odometry: bool, transient_tolerance: f64) {
            let options = load_options_2d("");
            let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
            // Before the first IMU sample nothing can be computed.
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.0), 0.0, &[]))
                    .is_none()
            );
            let position = |t: f64| (0.2 * (t - 1.0)).max(0.0);
            let mut insertion_results = Vec::new();
            for i in 0..=100 {
                let t = 0.05 * f64::from(i);
                builder.add_imu_data(&ImuData {
                    time: time(t),
                    linear_acceleration: Vector3::new(0.0, 0.0, 9.81),
                    angular_velocity: Vector3::zeros(),
                });
                if use_odometry {
                    builder.add_odometry_data(&OdometryData {
                        time: time(t),
                        pose: Rigid3d::from_translation(Vector3::new(position(t), 0.0, 0.0)),
                    });
                }
                if i % 2 == 0 && i > 0 {
                    let x = position(t) as f32;
                    let velocity = if t > 1.0 { 0.2 } else { 0.0 };
                    let result = builder
                        .add_range_data("scan", moving_scan(time(t), x, velocity, &[]))
                        .expect("result");
                    // The pose is tracked to a fraction of a cell (with a
                    // ~5 mm bias, see above), except during the transient
                    // after the abrupt start.
                    let tolerance = if (1.0..3.0).contains(&t) {
                        transient_tolerance
                    } else {
                        0.01
                    };
                    assert!(
                        (result.local_pose.translation().x - f64::from(x)).abs() < tolerance,
                        "t = {t}: {} vs. {x}",
                        result.local_pose
                    );
                    if let Some(insertion_result) = result.insertion_result {
                        assert_eq!(insertion_result.constant_data.time, result.time);
                        assert_eq!(insertion_result.constant_data.local_pose, result.local_pose);
                        insertion_results.push(insertion_result);
                    }
                }
            }
            // 0.8 m of motion: the motion filter (0.2 m) passes about 4 nodes
            // plus the first one.
            assert!(
                (4..=6).contains(&insertion_results.len()),
                "{}",
                insertion_results.len()
            );
            for insertion_result in &insertion_results {
                assert_eq!(insertion_result.insertion_submaps.len(), 1);
                assert!(
                    !insertion_result
                        .constant_data
                        .filtered_gravity_aligned_point_cloud
                        .is_empty()
                );
            }
        }

        #[test]
        fn tracks_translation_with_imu_and_odometry() {
            // Odometry provides the velocity, so there is no transient.
            track_translation(true, 0.01);
        }

        #[test]
        fn tracks_translation_with_imu() {
            // The constant velocity extrapolator learns the new velocity
            // from the scan matches within ~2 s (the Ceres translation prior
            // pulls towards the stale prediction meanwhile).
            track_translation(false, 0.03);
        }

        #[test]
        fn min_range_and_missing_data() {
            let options = load_options_2d(
                "TRAJECTORY_BUILDER_2D.use_imu_data = false\n\
                 TRAJECTORY_BUILDER_2D.min_range = 0.5\n\
                 TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.001",
            );
            let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
            let extra_points = [
                // Below min_range: dropped.
                Vector3::new(0.2, 0.1, 0.0),
                // Beyond max_range (30 m): a miss at missing_data_ray_length.
                Vector3::new(30.0, 40.0, 0.0),
            ];
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.1), 0.0, &extra_points))
                    .is_none()
            );
            let result = builder
                .add_range_data("scan", scan(time(0.2), 0.0, &extra_points))
                .expect("result");
            let misses = result.range_data_in_local.misses.points();
            assert_eq!(misses.len(), 1);
            assert!((misses[0].position - Vector3::new(3.0, 4.0, 0.0)).norm() < 1e-3);
            assert!(
                result
                    .range_data_in_local
                    .returns
                    .iter()
                    .all(|p| p.position.norm() > 3.9)
            );
            assert!(result.range_data_in_local.returns.size() > 300);
        }

        #[test]
        fn accumulates_range_data() {
            let options = load_options_2d(
                "TRAJECTORY_BUILDER_2D.use_imu_data = false\n\
                 TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 2",
            );
            let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
            // First scan dropped (extrapolator initializing).
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.1), 0.0, &[]))
                    .is_none()
            );
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.2), 0.0, &[]))
                    .is_none()
            );
            let result = builder
                .add_range_data("scan", scan(time(0.3), 0.0, &[]))
                .expect("second accumulated scan");
            assert_eq!(result.time, time(0.3));
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.4), 0.0, &[]))
                    .is_none()
            );
            assert!(
                builder
                    .add_range_data("scan", scan(time(0.5), 0.0, &[]))
                    .is_some()
            );
        }

        #[test]
        #[should_panic(expected = "An unexpected IMU packet was added.")]
        fn imu_data_requires_use_imu_data() {
            let options = load_options_2d("TRAJECTORY_BUILDER_2D.use_imu_data = false");
            let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
            builder.add_imu_data(&ImuData::default());
        }
    }

    /// Counts observations of all metrics.
    #[derive(Default)]
    struct RecordingMetric {
        count: AtomicUsize,
    }

    impl Gauge for RecordingMetric {
        fn increment(&self) {}
        fn increment_by(&self, _: f64) {}
        fn decrement(&self) {}
        fn decrement_by(&self, _: f64) {}
        fn set(&self, _: f64) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Histogram for RecordingMetric {
        fn observe(&self, _: f64) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Counter for RecordingMetric {
        fn increment(&self) {}
        fn increment_by(&self, _: f64) {}
    }

    struct RecordingFamily {
        metric: &'static RecordingMetric,
        names: &'static Mutex<Vec<String>>,
        name: String,
    }

    impl RecordingFamily {
        fn record(&self, labels: &Labels) {
            self.names
                .lock()
                .unwrap()
                .push(format!("{} {labels:?}", self.name));
        }
    }

    impl Family<dyn Gauge> for RecordingFamily {
        fn add(&self, labels: &Labels) -> &'static dyn Gauge {
            self.record(labels);
            self.metric
        }
    }

    impl Family<dyn Histogram> for RecordingFamily {
        fn add(&self, labels: &Labels) -> &'static dyn Histogram {
            self.record(labels);
            self.metric
        }
    }

    impl Family<dyn Counter> for RecordingFamily {
        fn add(&self, labels: &Labels) -> &'static dyn Counter {
            self.record(labels);
            self.metric
        }
    }

    struct RecordingFactory {
        metric: &'static RecordingMetric,
        names: &'static Mutex<Vec<String>>,
    }

    impl RecordingFactory {
        fn family(&self, name: &str) -> &'static RecordingFamily {
            Box::leak(Box::new(RecordingFamily {
                metric: self.metric,
                names: self.names,
                name: name.to_string(),
            }))
        }
    }

    impl FamilyFactory for RecordingFactory {
        fn new_counter_family(&mut self, name: &str, _: &str) -> &'static dyn Family<dyn Counter> {
            self.family(name)
        }
        fn new_gauge_family(&mut self, name: &str, _: &str) -> &'static dyn Family<dyn Gauge> {
            self.family(name)
        }
        fn new_histogram_family(
            &mut self,
            name: &str,
            _: &str,
            _: &BucketBoundaries,
        ) -> &'static dyn Family<dyn Histogram> {
            self.family(name)
        }
    }

    #[test]
    fn register_metrics() {
        let metric: &'static RecordingMetric = Box::leak(Box::default());
        let names: &'static Mutex<Vec<String>> = Box::leak(Box::default());
        LocalTrajectoryBuilder2D::register_metrics(&mut RecordingFactory { metric, names });
        {
            let names = names.lock().unwrap();
            assert_eq!(names.len(), 7, "{names:?}");
            assert!(
                names.contains(
                    &"mapping_2d_local_trajectory_builder_residuals {\"component\": \"angle\"}"
                        .to_string()
                )
            );
        }
        let options = load_options_2d(
            "TRAJECTORY_BUILDER_2D.use_imu_data = false\n\
             TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true",
        );
        let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
        for i in 1..5 {
            builder.add_range_data("scan", scan(time(0.1 * f64::from(i)), 0.0, &[]));
        }
        // 3 results; results 2 and 3 observe a real-time CSM score, a Ceres
        // cost and two residuals (a submap exists), plus latency / ratios.
        assert!(metric.count.load(Ordering::Relaxed) >= 8);
    }
}
