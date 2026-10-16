//! Port of `cartographer/mapping/internal/3d/local_trajectory_builder_3d` (.h, .cc).

use std::sync::{Arc, PoisonError, RwLock};
use std::time::Instant;

use nalgebra::{Quaternion, Vector3};

use crate::common::time::{Duration, Time, from_seconds, get_thread_cpu_time_seconds, to_seconds};
use crate::mapping::internal::motion_filter::MotionFilter;
use crate::mapping::internal::range_data_collator::RangeDataCollator;
use crate::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::{
    CeresScanMatcher3D, PointCloudAndHybridGridsPointers,
};
use crate::mapping::internal::three_d::scan_matching::real_time_correlative_scan_matcher_3d::RealTimeCorrelativeScanMatcher3D;
use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
use crate::mapping::pose_extrapolator_interface::{
    ExtrapolationResult, PoseExtrapolatorInterface, create_with_imu_data,
};
use crate::mapping::proto;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::three_d::submap_3d::{ActiveSubmaps3D, Submap3D};
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::metrics::gauge::Gauge;
use crate::metrics::histogram::{Histogram, fixed_width, scaled_powers_of};
use crate::sensor::imu_data::{self, ImuData};
use crate::sensor::internal::voxel_filter::{
    adaptive_voxel_filter, voxel_filter, voxel_filter_range_measurements,
};
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::point_cloud::{PointCloud, transform_point_cloud};
use crate::sensor::range_data::{RangeData, transform_range_data};
use crate::sensor::rangefinder_point::RangefinderPoint;
use crate::sensor::timed_point_cloud_data::{TimedPointCloudData, TimedPointCloudOriginData};
use crate::transform::rigid_transform::{Rigid3d, Rigid3f, quaternion_cast, quaternion_mul};
use crate::transform::timestamped_transform::{self, TimestampedTransform};

// TODO(spielawa): Adjust metrics for multi-trajectory. So far we assume a
// single trajectory.
//
// RUST PORT NOTE: the C++ file-level `static auto* k...Metric` pointers
// (initialised to `Null()` and overwritten by `RegisterMetrics`) are bundled
// into one `Metrics` value behind a global lock; `None` means "all null".
#[derive(Clone, Copy)]
struct Metrics {
    local_slam_latency: &'static dyn Gauge,
    local_slam_voxel_filter_fraction: &'static dyn Gauge,
    local_slam_scan_matcher_fraction: &'static dyn Gauge,
    local_slam_insert_into_submap_fraction: &'static dyn Gauge,
    local_slam_real_time_ratio: &'static dyn Gauge,
    local_slam_cpu_real_time_ratio: &'static dyn Gauge,
    real_time_correlative_scan_matcher_score: &'static dyn Histogram,
    ceres_scan_matcher_cost: &'static dyn Histogram,
    scan_matcher_residual_distance: &'static dyn Histogram,
    scan_matcher_residual_angle: &'static dyn Histogram,
}

impl Metrics {
    fn null() -> Self {
        Metrics {
            local_slam_latency: <dyn Gauge>::null(),
            local_slam_voxel_filter_fraction: <dyn Gauge>::null(),
            local_slam_scan_matcher_fraction: <dyn Gauge>::null(),
            local_slam_insert_into_submap_fraction: <dyn Gauge>::null(),
            local_slam_real_time_ratio: <dyn Gauge>::null(),
            local_slam_cpu_real_time_ratio: <dyn Gauge>::null(),
            real_time_correlative_scan_matcher_score: <dyn Histogram>::null(),
            ceres_scan_matcher_cost: <dyn Histogram>::null(),
            scan_matcher_residual_distance: <dyn Histogram>::null(),
            scan_matcher_residual_angle: <dyn Histogram>::null(),
        }
    }
}

static METRICS: RwLock<Option<Metrics>> = RwLock::new(None);

fn metrics() -> Metrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(Metrics::null)
}

fn labels(pairs: &[(&str, &str)]) -> Labels {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

/// Eigen's `QuaternionBase::angularDistance`.
fn angular_distance(a: &Quaternion<f64>, b: &Quaternion<f64>) -> f64 {
    let d = quaternion_mul(a, &b.conjugate());
    2. * d.imag().norm().atan2(d.w.abs())
}

/// Eigen's `Quaterniond::inverse()` (conjugate divided by the squared norm).
fn quaternion_inverse(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = q.norm_squared();
    if n2 > 0. {
        q.conjugate() / n2
    } else {
        // Eigen returns a zero quaternion in this case.
        Quaternion::new(0., 0., 0., 0.)
    }
}

/// Result of inserting range data into the submaps (C++
/// `LocalTrajectoryBuilder3D::InsertionResult`).
#[derive(Clone)]
pub struct InsertionResult {
    /// The constant data of the new trajectory node.
    pub constant_data: Arc<TrajectoryNodeData>,
    /// The active submaps the data was inserted into (front: the older one,
    /// used for matching).
    pub insertion_submaps: Vec<SharedSubmap<Submap3D>>,
}

/// Result of local SLAM for one accumulated range data (C++
/// `LocalTrajectoryBuilder3D::MatchingResult`).
#[derive(Clone)]
pub struct MatchingResult {
    /// Time of the matched range data.
    pub time: Time,
    /// The scan matched pose of the tracking frame in the local frame.
    pub local_pose: Rigid3d,
    /// The voxel filtered range data in the local frame.
    pub range_data_in_local: RangeData,
    /// `None` if dropped by the motion filter.
    pub insertion_result: Option<InsertionResult>,
}

/// Wires up the local SLAM stack (i.e. pose extrapolator, scan matching,
/// etc.) without loop closure.
pub struct LocalTrajectoryBuilder3D {
    options: proto::LocalTrajectoryBuilderOptions3D,
    active_submaps: ActiveSubmaps3D,

    motion_filter: MotionFilter,
    real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher3D,
    ceres_scan_matcher: CeresScanMatcher3D,

    // RUST PORT NOTE: `None` models the C++ null `unique_ptr` until the first
    // IMU sample arrives.
    extrapolator: Option<Box<dyn PoseExtrapolatorInterface>>,

    num_accumulated: i32,
    accumulated_point_cloud_origin_data: Vec<TimedPointCloudOriginData>,
    last_wall_time: Option<Instant>,

    last_thread_cpu_time_seconds: Option<f64>,

    range_data_collator: RangeDataCollator,

    last_sensor_time: Option<Time>,
}

impl LocalTrajectoryBuilder3D {
    /// Creates a builder for the given options and the ids of the range
    /// sensors whose data is collated.
    pub fn new<S: AsRef<str>>(
        options: &proto::LocalTrajectoryBuilderOptions3D,
        expected_range_sensor_ids: &[S],
    ) -> Self {
        LocalTrajectoryBuilder3D {
            options: options.clone(),
            active_submaps: ActiveSubmaps3D::new(&options.submaps_options.unwrap_or_default()),
            motion_filter: MotionFilter::new(&options.motion_filter_options.unwrap_or_default()),
            real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher3D::new(
                &options
                    .real_time_correlative_scan_matcher_options
                    .unwrap_or_default(),
            ),
            ceres_scan_matcher: CeresScanMatcher3D::new(
                &options
                    .ceres_scan_matcher_options
                    .clone()
                    .unwrap_or_default(),
            ),
            extrapolator: None,
            num_accumulated: 0,
            accumulated_point_cloud_origin_data: Vec::new(),
            last_wall_time: None,
            last_thread_cpu_time_seconds: None,
            range_data_collator: RangeDataCollator::new(expected_range_sensor_ids),
            last_sensor_time: None,
        }
    }

    /// Scan matches using the two point clouds and returns the observed pose,
    /// or `None` on failure.
    fn scan_match(
        &self,
        pose_prediction: &Rigid3d,
        low_resolution_point_cloud_in_tracking: &PointCloud,
        high_resolution_point_cloud_in_tracking: &PointCloud,
    ) -> Option<Rigid3d> {
        let submaps = self.active_submaps.submaps();
        let Some(matching_submap) = submaps.first() else {
            return Some(*pose_prediction);
        };
        let matching_submap = matching_submap
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let metrics = metrics();
        let mut initial_ceres_pose = matching_submap.local_pose().inverse() * *pose_prediction;
        if self.options.use_online_correlative_scan_matching {
            // We take a copy since we use 'initial_ceres_pose' as an output
            // argument.
            let initial_pose = initial_ceres_pose;
            let score = self.real_time_correlative_scan_matcher.match_(
                &initial_pose,
                high_resolution_point_cloud_in_tracking,
                matching_submap.high_resolution_hybrid_grid(),
                &mut initial_ceres_pose,
            );
            metrics
                .real_time_correlative_scan_matcher_score
                .observe(f64::from(score));
        }

        let mut pose_observation_in_submap = Rigid3d::identity();
        let mut summary = ceres::SolverSummary::default();
        let high_resolution_intensity_hybrid_grid = if self.options.use_intensities {
            Some(matching_submap.high_resolution_intensity_hybrid_grid())
        } else {
            None
        };
        self.ceres_scan_matcher.match_(
            (matching_submap.local_pose().inverse() * *pose_prediction).translation(),
            &initial_ceres_pose,
            &[
                PointCloudAndHybridGridsPointers {
                    point_cloud: high_resolution_point_cloud_in_tracking,
                    hybrid_grid: matching_submap.high_resolution_hybrid_grid(),
                    intensity_hybrid_grid: high_resolution_intensity_hybrid_grid,
                },
                PointCloudAndHybridGridsPointers {
                    point_cloud: low_resolution_point_cloud_in_tracking,
                    hybrid_grid: matching_submap.low_resolution_hybrid_grid(),
                    intensity_hybrid_grid: None,
                },
            ],
            &mut pose_observation_in_submap,
            &mut summary,
        );
        metrics.ceres_scan_matcher_cost.observe(summary.final_cost);
        let residual_distance =
            (pose_observation_in_submap.translation() - initial_ceres_pose.translation()).norm();
        metrics
            .scan_matcher_residual_distance
            .observe(residual_distance);
        let residual_angle = angular_distance(
            pose_observation_in_submap.rotation(),
            initial_ceres_pose.rotation(),
        );
        metrics.scan_matcher_residual_angle.observe(residual_angle);
        Some(matching_submap.local_pose() * pose_observation_in_submap)
    }

    /// Adds an IMU measurement. The first one initializes the pose
    /// extrapolator (together with `initial_imu_data` and `initial_poses`
    /// from the options).
    pub fn add_imu_data(&mut self, imu_data: &ImuData) {
        if let Some(extrapolator) = &mut self.extrapolator {
            extrapolator.add_imu_data(imu_data);
            return;
        }
        let initial_poses: Vec<TimestampedTransform> = self
            .options
            .initial_poses
            .iter()
            .map(timestamped_transform::from_proto)
            .collect();
        let mut initial_imu_data: Vec<ImuData> = self
            .options
            .initial_imu_data
            .iter()
            .map(imu_data::from_proto)
            .collect();
        initial_imu_data.push(imu_data.clone());
        self.extrapolator = Some(create_with_imu_data(
            &self.options.pose_extrapolator_options.unwrap_or_default(),
            &initial_imu_data,
            &initial_poses,
        ));
    }

    /// Returns a [`MatchingResult`] when range data accumulation completed,
    /// otherwise `None`. `TimedPointCloudData::time` is when the last point
    /// in `range_data` was acquired, `TimedPointCloudData::ranges` contains
    /// the relative time of point with respect to `TimedPointCloudData::time`.
    pub fn add_range_data(
        &mut self,
        sensor_id: &str,
        unsynchronized_data: &TimedPointCloudData,
    ) -> Option<MatchingResult> {
        if self.options.use_intensities {
            assert_eq!(
                unsynchronized_data.ranges.len(),
                unsynchronized_data.intensities.len(),
                "Passed point cloud has inconsistent number of intensities and ranges."
            );
        }
        let mut synchronized_data = self
            .range_data_collator
            .add_range_data(sensor_id, unsynchronized_data.clone());
        if synchronized_data.ranges.is_empty() {
            log::info!("Range data collator filling buffer.");
            return None;
        }

        let Some(extrapolator) = self.extrapolator.as_mut() else {
            // Until we've initialized the extrapolator with our first IMU
            // message, we cannot compute the orientation of the rangefinder.
            log::info!("IMU not yet initialized.");
            return None;
        };

        assert!(!synchronized_data.ranges.is_empty());
        let last_point_time = synchronized_data.ranges.last().unwrap().point_time.time;
        assert!(
            last_point_time <= 0.,
            "Check failed: synchronized_data.ranges.back().point_time.time <= 0.f ({last_point_time} vs. 0)"
        );
        let time_first_point = synchronized_data.time
            + from_seconds(f64::from(synchronized_data.ranges[0].point_time.time));
        if time_first_point < extrapolator.get_last_pose_time() {
            log::info!("Extrapolator is still initializing.");
            return None;
        }

        if self.num_accumulated == 0 {
            self.accumulated_point_cloud_origin_data.clear();
        }

        synchronized_data.ranges = voxel_filter_range_measurements(
            &synchronized_data.ranges,
            0.5f32 * self.options.voxel_filter_size,
        );
        // RUST PORT NOTE: the C++ reads `synchronized_data.time` after moving
        // the data (valid there since `Time` is trivially copyable); capture
        // it before the move.
        let current_sensor_time = synchronized_data.time;
        self.accumulated_point_cloud_origin_data
            .push(synchronized_data);
        self.num_accumulated += 1;

        if self.num_accumulated < self.options.num_accumulated_range_data {
            return None;
        }
        self.num_accumulated = 0;

        let mut warned = false;
        let mut hit_times: Vec<Time> = Vec::new();
        let mut prev_time_point = extrapolator.get_last_extrapolated_time();
        for point_cloud_origin_data in &self.accumulated_point_cloud_origin_data {
            for hit in &point_cloud_origin_data.ranges {
                let mut time_point =
                    point_cloud_origin_data.time + from_seconds(f64::from(hit.point_time.time));
                if time_point < prev_time_point {
                    if !warned {
                        log::error!(
                            "Timestamp of individual range data point jumps backwards from {prev_time_point} to {time_point}"
                        );
                        warned = true;
                    }
                    time_point = prev_time_point;
                }

                hit_times.push(time_point);
                prev_time_point = time_point;
            }
        }
        hit_times.push(
            self.accumulated_point_cloud_origin_data
                .last()
                .expect("accumulated data is not empty")
                .time,
        );

        let extrapolation_result: ExtrapolationResult =
            extrapolator.extrapolate_poses_with_gravity(&hit_times);
        let mut hits_poses: Vec<Rigid3f> = extrapolation_result.previous_poses.clone();
        hits_poses.push(extrapolation_result.current_pose.cast::<f32>());
        assert_eq!(
            hits_poses.len(),
            hit_times.len(),
            "Check failed: hits_poses.size() == hit_times.size()"
        );

        let max_possible_number_of_accumulated_points = hit_times.len();
        let mut accumulated_points: Vec<RangefinderPoint> =
            Vec::with_capacity(max_possible_number_of_accumulated_points);
        let mut accumulated_intensities: Vec<f32> = Vec::new();
        if self.options.use_intensities {
            accumulated_intensities.reserve(max_possible_number_of_accumulated_points);
        }
        let mut misses = PointCloud::new();
        let mut hits_poses_it = 0usize;
        for point_cloud_origin_data in &self.accumulated_point_cloud_origin_data {
            for hit in &point_cloud_origin_data.ranges {
                let hit_pose = &hits_poses[hits_poses_it];
                let hit_in_local: Vector3<f32> = hit_pose * &hit.point_time.position;
                let origin_in_local: Vector3<f32> =
                    hit_pose * &point_cloud_origin_data.origins[hit.origin_index];
                let delta = hit_in_local - origin_in_local;
                let range = delta.norm();
                if range >= self.options.min_range {
                    if range <= self.options.max_range {
                        accumulated_points.push(RangefinderPoint {
                            position: hit_in_local,
                        });
                        if self.options.use_intensities {
                            accumulated_intensities.push(hit.intensity);
                        }
                    } else {
                        // We insert a ray cropped to 'max_range' as a miss for
                        // hits beyond the maximum range. This way the free
                        // space up to the maximum range will be updated.
                        // TODO(wohe): since `misses` are not used anywhere in
                        // 3D, consider removing `misses` from `range_data`
                        // and/or everywhere in 3D.
                        misses.push_back(RangefinderPoint {
                            position: origin_in_local + self.options.max_range / range * delta,
                        });
                    }
                }
                hits_poses_it += 1;
            }
        }
        assert!(
            hits_poses_it + 1 == hits_poses.len(),
            "Check failed: std::next(hits_poses_it) == hits_poses.end()"
        );
        let returns = PointCloud::with_intensities(accumulated_points, accumulated_intensities);

        let sensor_duration: Option<Duration> = self
            .last_sensor_time
            .map(|last_sensor_time| current_sensor_time - last_sensor_time);
        self.last_sensor_time = Some(current_sensor_time);

        let current_time = *hit_times.last().expect("hit_times is not empty");
        let voxel_filter_start = Instant::now();
        let filtered_range_data = RangeData {
            origin: extrapolation_result
                .current_pose
                .translation()
                .cast::<f32>(),
            returns: voxel_filter(&returns, self.options.voxel_filter_size),
            misses: voxel_filter(&misses, self.options.voxel_filter_size),
        };
        let voxel_filter_duration = voxel_filter_start.elapsed();

        if let Some(sensor_duration) = sensor_duration {
            let voxel_filter_fraction =
                voxel_filter_duration.as_secs_f64() / to_seconds(sensor_duration);
            metrics()
                .local_slam_voxel_filter_fraction
                .set(voxel_filter_fraction);
        }

        self.add_accumulated_range_data(
            current_time,
            &transform_range_data(
                &filtered_range_data,
                &extrapolation_result.current_pose.inverse().cast::<f32>(),
            ),
            sensor_duration,
            &extrapolation_result.current_pose,
            &extrapolation_result.gravity_from_tracking,
        )
    }

    fn add_accumulated_range_data(
        &mut self,
        time: Time,
        filtered_range_data_in_tracking: &RangeData,
        sensor_duration: Option<Duration>,
        pose_prediction: &Rigid3d,
        gravity_alignment: &Quaternion<f64>,
    ) -> Option<MatchingResult> {
        if filtered_range_data_in_tracking.returns.is_empty() {
            log::warn!("Dropped empty range data.");
            return None;
        }
        let metrics = metrics();

        let scan_matcher_start = Instant::now();

        let high_resolution_point_cloud_in_tracking = adaptive_voxel_filter(
            &filtered_range_data_in_tracking.returns,
            &self
                .options
                .high_resolution_adaptive_voxel_filter_options
                .unwrap_or_default(),
        );
        if high_resolution_point_cloud_in_tracking.is_empty() {
            log::warn!("Dropped empty high resolution point cloud data.");
            return None;
        }
        let low_resolution_point_cloud_in_tracking = adaptive_voxel_filter(
            &filtered_range_data_in_tracking.returns,
            &self
                .options
                .low_resolution_adaptive_voxel_filter_options
                .unwrap_or_default(),
        );
        if low_resolution_point_cloud_in_tracking.is_empty() {
            log::warn!("Dropped empty low resolution point cloud data.");
            return None;
        }

        let Some(pose_estimate) = self.scan_match(
            pose_prediction,
            &low_resolution_point_cloud_in_tracking,
            &high_resolution_point_cloud_in_tracking,
        ) else {
            log::warn!("Scan matching failed.");
            return None;
        };
        self.extrapolator
            .as_mut()
            .expect("extrapolator is initialized")
            .add_pose(time, &pose_estimate);

        let scan_matcher_duration = scan_matcher_start.elapsed();
        if let Some(sensor_duration) = sensor_duration {
            let scan_matcher_fraction =
                scan_matcher_duration.as_secs_f64() / to_seconds(sensor_duration);
            metrics
                .local_slam_scan_matcher_fraction
                .set(scan_matcher_fraction);
        }

        let filtered_range_data_in_local = transform_range_data(
            filtered_range_data_in_tracking,
            &pose_estimate.cast::<f32>(),
        );

        let insert_into_submap_start = Instant::now();
        let insertion_result = self.insert_into_submap(
            time,
            &filtered_range_data_in_local,
            filtered_range_data_in_tracking,
            high_resolution_point_cloud_in_tracking,
            low_resolution_point_cloud_in_tracking,
            &pose_estimate,
            gravity_alignment,
        );
        let insert_into_submap_duration = insert_into_submap_start.elapsed();
        if let Some(sensor_duration) = sensor_duration {
            let insert_into_submap_fraction =
                insert_into_submap_duration.as_secs_f64() / to_seconds(sensor_duration);
            metrics
                .local_slam_insert_into_submap_fraction
                .set(insert_into_submap_fraction);
        }
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
            range_data_in_local: filtered_range_data_in_local,
            insertion_result,
        })
    }

    /// Adds an odometry measurement (ignored until the extrapolator is
    /// initialized by the first IMU sample).
    pub fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        let Some(extrapolator) = &mut self.extrapolator else {
            // Until we've initialized the extrapolator we cannot add odometry
            // data.
            log::info!("Extrapolator not yet initialized.");
            return;
        };
        extrapolator.add_odometry_data(odometry_data);
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_into_submap(
        &mut self,
        time: Time,
        filtered_range_data_in_local: &RangeData,
        filtered_range_data_in_tracking: &RangeData,
        high_resolution_point_cloud_in_tracking: PointCloud,
        low_resolution_point_cloud_in_tracking: PointCloud,
        pose_estimate: &Rigid3d,
        gravity_alignment: &Quaternion<f64>,
    ) -> Option<InsertionResult> {
        if self.motion_filter.is_similar(time, pose_estimate) {
            return None;
        }
        let rotational_scan_matcher_histogram_in_gravity = RotationalScanMatcher::compute_histogram(
            &transform_point_cloud(
                &filtered_range_data_in_tracking.returns,
                &Rigid3f::from_rotation(quaternion_cast::<f64, f32>(gravity_alignment)),
            ),
            self.options.rotational_histogram_size,
        );

        let local_from_gravity_aligned = quaternion_mul(
            pose_estimate.rotation(),
            &quaternion_inverse(gravity_alignment),
        );
        let insertion_submaps = self.active_submaps.insert_data(
            filtered_range_data_in_local,
            &local_from_gravity_aligned,
            &rotational_scan_matcher_histogram_in_gravity,
        );
        Some(InsertionResult {
            constant_data: Arc::new(TrajectoryNodeData {
                time,
                gravity_alignment: *gravity_alignment,
                // 'filtered_gravity_aligned_point_cloud' is only used in 2D.
                filtered_gravity_aligned_point_cloud: PointCloud::new(),
                high_resolution_point_cloud: high_resolution_point_cloud_in_tracking,
                low_resolution_point_cloud: low_resolution_point_cloud_in_tracking,
                rotational_scan_matcher_histogram: rotational_scan_matcher_histogram_in_gravity,
                local_pose: *pose_estimate,
            }),
            insertion_submaps,
        })
    }

    /// Registers the local SLAM metrics with `family_factory`.
    pub fn register_metrics(family_factory: &mut dyn FamilyFactory) {
        let latency = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_latency",
            "Duration from first incoming point cloud in accumulation to local slam result",
        );
        let local_slam_latency = latency.add(&Labels::new());

        let voxel_filter_fraction = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_voxel_filter_fraction",
            "Fraction of total sensor time taken up by voxel filter.",
        );
        let local_slam_voxel_filter_fraction = voxel_filter_fraction.add(&Labels::new());

        let scan_matcher_fraction = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_scan_matcher_fraction",
            "Fraction of total sensor time taken up by scan matcher.",
        );
        let local_slam_scan_matcher_fraction = scan_matcher_fraction.add(&Labels::new());

        let insert_into_submap_fraction = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_insert_into_submap_fraction",
            "Fraction of total sensor time taken up by inserting into submap.",
        );
        let local_slam_insert_into_submap_fraction =
            insert_into_submap_fraction.add(&Labels::new());

        let real_time_ratio = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_real_time_ratio",
            "sensor duration / wall clock duration.",
        );
        let local_slam_real_time_ratio = real_time_ratio.add(&Labels::new());

        let cpu_real_time_ratio = family_factory.new_gauge_family(
            "mapping_3d_local_trajectory_builder_cpu_real_time_ratio",
            "sensor duration / cpu duration.",
        );
        let local_slam_cpu_real_time_ratio = cpu_real_time_ratio.add(&Labels::new());

        let score_boundaries = fixed_width(0.05, 20);
        let scores = family_factory.new_histogram_family(
            "mapping_3d_local_trajectory_builder_scores",
            "Local scan matcher scores",
            &score_boundaries,
        );
        let real_time_correlative_scan_matcher_score =
            scores.add(&labels(&[("scan_matcher", "real_time_correlative")]));
        let cost_boundaries = scaled_powers_of(2., 0.01, 100.);
        let costs = family_factory.new_histogram_family(
            "mapping_3d_local_trajectory_builder_costs",
            "Local scan matcher costs",
            &cost_boundaries,
        );
        let ceres_scan_matcher_cost = costs.add(&labels(&[("scan_matcher", "ceres")]));
        let distance_boundaries = scaled_powers_of(2., 0.01, 10.);
        let residuals = family_factory.new_histogram_family(
            "mapping_3d_local_trajectory_builder_residuals",
            "Local scan matcher residuals",
            &distance_boundaries,
        );
        let scan_matcher_residual_distance = residuals.add(&labels(&[("component", "distance")]));
        let scan_matcher_residual_angle = residuals.add(&labels(&[("component", "angle")]));

        *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(Metrics {
            local_slam_latency,
            local_slam_voxel_filter_fraction,
            local_slam_scan_matcher_fraction,
            local_slam_insert_into_submap_fraction,
            local_slam_real_time_ratio,
            local_slam_cpu_real_time_ratio,
            real_time_correlative_scan_matcher_score,
            ceres_scan_matcher_cost,
            scan_matcher_residual_distance,
            scan_matcher_residual_angle,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod local_trajectory_builder_test {
        use super::*;
        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::common::time::from_universal;
        use crate::mapping::internal::three_d::local_trajectory_builder_options_3d::create_local_trajectory_builder_options_3d;
        use crate::sensor::point_cloud::{TimedPointCloud, transform_timed_point_cloud};
        use crate::sensor::rangefinder_point::TimedRangefinderPoint;
        use crate::transform::rigid_transform::{AngleAxis, quaternion_rotate};
        use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
        use std::f64::consts::PI;

        const SENSOR_ID: &str = "sensor_id";

        /// `std::mt19937`, so that the bubbles are exactly the C++ ones.
        pub(super) struct Mt19937 {
            mt: [u32; 624],
            index: usize,
        }

        impl Mt19937 {
            pub(super) fn new(seed: u32) -> Self {
                let mut mt = [0u32; 624];
                mt[0] = seed;
                for i in 1..624 {
                    mt[i] = 1_812_433_253u32
                        .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                        .wrapping_add(i as u32);
                }
                Mt19937 { mt, index: 624 }
            }

            pub(super) fn next_u32(&mut self) -> u32 {
                if self.index >= 624 {
                    for i in 0..624 {
                        let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                        let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                        if y & 1 != 0 {
                            v ^= 0x9908_b0df;
                        }
                        self.mt[i] = v;
                    }
                    self.index = 0;
                }
                let mut y = self.mt[self.index];
                self.index += 1;
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c_5680;
                y ^= (y << 15) & 0xefc6_0000;
                y ^ (y >> 18)
            }

            /// `std::uniform_real_distribution<float>(a, b)` on top of
            /// `std::generate_canonical<float, 24>` (one 32-bit draw).
            fn uniform_f32(&mut self, a: f32, b: f32) -> f32 {
                let mut canonical = self.next_u32() as f32 / 4_294_967_296f32;
                if canonical >= 1. {
                    // std::nextafter(1.f, 0.f)
                    canonical = 1. - f32::EPSILON / 2.;
                }
                canonical * (b - a) + a
            }
        }

        struct TrajectoryNode {
            time: Time,
            pose: Rigid3d,
        }

        struct Fixture {
            local_trajectory_builder: Option<LocalTrajectoryBuilder3D>,
            bubbles: Vec<Vector3<f32>>,
        }

        impl Fixture {
            /// C++ `SetUp()`.
            fn new() -> Self {
                let mut fixture = Fixture {
                    local_trajectory_builder: None,
                    bubbles: Vec::new(),
                };
                fixture.generate_bubbles();
                fixture
            }

            fn create_trajectory_builder_options_3d() -> proto::LocalTrajectoryBuilderOptions3D {
                let parameter_dictionary = make_dictionary(
                    r#"
        return {
          min_range = 0.5,
          max_range = 50.,
          num_accumulated_range_data = 1,
          voxel_filter_size = 0.2,

          high_resolution_adaptive_voxel_filter = {
            max_length = 0.7,
            min_num_points = 200,
            max_range = 50.,
          },

          low_resolution_adaptive_voxel_filter = {
            max_length = 0.7,
            min_num_points = 200,
            max_range = 50.,
          },

          use_online_correlative_scan_matching = false,
          real_time_correlative_scan_matcher = {
            linear_search_window = 0.2,
            angular_search_window = math.rad(1.),
            translation_delta_cost_weight = 1e-1,
            rotation_delta_cost_weight = 1.,
          },

          ceres_scan_matcher = {
            occupied_space_weight_0 = 5.,
            occupied_space_weight_1 = 20.,
            translation_weight = 0.1,
            rotation_weight = 0.3,
            only_optimize_yaw = false,
            ceres_solver_options = {
              use_nonmonotonic_steps = true,
              max_num_iterations = 20,
              num_threads = 1,
            },
          },

          motion_filter = {
            max_time_seconds = 0.2,
            max_distance_meters = 0.02,
            max_angle_radians = 0.001,
          },

          imu_gravity_time_constant = 1.,
          rotational_histogram_size = 120,

          pose_extrapolator = {
            use_imu_based = false,
            constant_velocity = {
              imu_gravity_time_constant = 10.,
              pose_queue_duration = 0.001,
            },
            imu_based = {
              pose_queue_duration = 5.,
              gravity_constant = 9.806,
              pose_translation_weight = 1.,
              pose_rotation_weight = 1.,
              imu_acceleration_weight = 1.,
              imu_rotation_weight = 1.,
              odometry_translation_weight = 1.,
              odometry_rotation_weight = 1.,
              solver_options = {
                use_nonmonotonic_steps = false;
                max_num_iterations = 10;
                num_threads = 1;
              },
            },
          },

          submaps = {
            high_resolution = 0.2,
            high_resolution_max_range = 50.,
            low_resolution = 0.5,
            num_range_data = 45000,
            range_data_inserter = {
              hit_probability = 0.7,
              miss_probability = 0.4,
              num_free_space_voxels = 0,
              intensity_threshold = 100.0,
            },
          },

          use_intensities = false,
        }
        "#,
                );
                create_local_trajectory_builder_options_3d(&parameter_dictionary)
            }

            fn generate_bubbles(&mut self) {
                let mut prng = Mt19937::new(42);
                for _ in 0..100 {
                    let x = prng.uniform_f32(-1., 1.);
                    let y = prng.uniform_f32(-1., 1.);
                    let z = prng.uniform_f32(-1., 1.);
                    self.bubbles.push(10f32 * Vector3::new(x, y, z).normalize());
                }
            }

            /// Computes the earliest intersection of the ray 'from' to 'to'
            /// with the axis-aligned cube with edge length 30 and center at
            /// the origin. Assumes that 'from' is inside the cube.
            fn collide_with_box(from: &Vector3<f32>, to: &Vector3<f32>) -> Vector3<f32> {
                let mut first = 100f32;
                for i in 0..3 {
                    if to[i] > from[i] {
                        first = first.min((15. - from[i]) / (to[i] - from[i]));
                    } else if to[i] < from[i] {
                        first = first.min((-15. - from[i]) / (to[i] - from[i]));
                    }
                }
                first * (to - from) + from
            }

            /// Computes the earliest intersection of the ray 'from' to 'to'
            /// with all bubbles. Returns 'to' if no intersection exists.
            fn collide_with_bubbles(&self, from: &Vector3<f32>, to: &Vector3<f32>) -> Vector3<f32> {
                let mut first = 1f32;
                const BUBBLE_RADIUS: f32 = 0.5;
                for center in &self.bubbles {
                    let a = (to - from).norm_squared();
                    let beta = (to - from).dot(&(from - center));
                    let c = (from - center).norm_squared() - BUBBLE_RADIUS * BUBBLE_RADIUS;
                    let discriminant = beta * beta - a * c;
                    if discriminant < 0. {
                        continue;
                    }
                    let solution = (-beta - discriminant.sqrt()) / a;
                    if solution < 0. {
                        continue;
                    }
                    first = first.min(solution);
                }
                first * (to - from) + from
            }

            fn generate_point_cloud_data(&self, pose: &Rigid3d, time: Time) -> TimedPointCloudData {
                // 360 degree rays at 16 angles.
                let mut directions_in_rangefinder_frame: TimedPointCloud = Vec::new();
                for r in -8..8 {
                    for s in -250..250 {
                        let yaw = AngleAxis::new((PI * f64::from(s) / 250.) as f32, Vector3::z());
                        let pitch =
                            AngleAxis::new((PI / 12. * f64::from(r) / 8.) as f32, Vector3::y());
                        let first_point = TimedRangefinderPoint {
                            position: quaternion_rotate(&(yaw * pitch), &Vector3::x()),
                            time: 0.,
                        };
                        directions_in_rangefinder_frame.push(first_point);
                        // Second orthogonal rangefinder.
                        let roll = AngleAxis::new((PI / 2.) as f32, Vector3::x());
                        let second_point = TimedRangefinderPoint {
                            position: quaternion_rotate(&((roll * yaw) * pitch), &Vector3::x()),
                            time: 0.,
                        };
                        directions_in_rangefinder_frame.push(second_point);
                    }
                }
                // We simulate a 30 m edge length box around the origin, also
                // containing 50 cm radius spheres.
                let pose_f = pose.cast::<f32>();
                let mut returns_in_world_frame: TimedPointCloud = Vec::new();
                for direction_in_world_frame in
                    transform_timed_point_cloud(&directions_in_rangefinder_frame, &pose_f)
                {
                    let origin = &pose_f * &Vector3::<f32>::zeros();
                    returns_in_world_frame.push(TimedRangefinderPoint {
                        position: self.collide_with_bubbles(
                            &origin,
                            &Self::collide_with_box(&origin, &direction_in_world_frame.position),
                        ),
                        time: 0.,
                    });
                }
                TimedPointCloudData {
                    time,
                    origin: Vector3::zeros(),
                    ranges: transform_timed_point_cloud(
                        &returns_in_world_frame,
                        &pose.inverse().cast::<f32>(),
                    ),
                    intensities: Vec::new(),
                }
            }

            fn add_linear_only_imu_observation(&mut self, time: Time, expected_pose: &Rigid3d) {
                let gravity = quaternion_rotate(
                    &quaternion_inverse(expected_pose.rotation()),
                    &Vector3::new(0., 0., 9.81),
                );
                self.local_trajectory_builder
                    .as_mut()
                    .unwrap()
                    .add_imu_data(&ImuData {
                        time,
                        linear_acceleration: gravity,
                        angular_velocity: Vector3::zeros(),
                    });
            }

            fn generate_corkscrew_trajectory() -> Vec<TrajectoryNode> {
                let mut trajectory = Vec::new();
                let mut current_time = from_universal(12345678);
                // One second at zero velocity.
                for _ in 0..5 {
                    current_time += from_seconds(0.3);
                    trajectory.push(TrajectoryNode {
                        time: current_time,
                        pose: Rigid3d::identity(),
                    });
                }
                // Corkscrew translation and constant velocity rotation.
                let mut t = 0f64;
                while t <= 0.6 {
                    current_time += from_seconds(0.3);
                    trajectory.push(TrajectoryNode {
                        time: current_time,
                        pose: Rigid3d::from_translation(Vector3::new(
                            (4. * t).sin(),
                            1. - (4. * t).cos(),
                            t,
                        )) * Rigid3d::from_angle_axis(AngleAxis::new(
                            0.3 * t,
                            Vector3::new(1., -1., 2.).normalize(),
                        )),
                    });
                    t += 0.05;
                }
                trajectory
            }

            /// Returns the number of matched poses (all checked against the
            /// expected trajectory).
            fn verify_accuracy(
                &mut self,
                expected_trajectory: &[TrajectoryNode],
                expected_accuracy: f64,
            ) -> usize {
                let mut num_poses = 0;
                for node in expected_trajectory {
                    self.add_linear_only_imu_observation(node.time, &node.pose);
                    let point_cloud = self.generate_point_cloud_data(&node.pose, node.time);
                    let matching_result = self
                        .local_trajectory_builder
                        .as_mut()
                        .unwrap()
                        .add_range_data(SENSOR_ID, &point_cloud);
                    if let Some(matching_result) = matching_result {
                        assert_is_nearly(
                            &matching_result.local_pose,
                            &node.pose,
                            expected_accuracy,
                        );
                        num_poses += 1;
                        log::info!("num_poses: {num_poses}");
                    }
                }
                num_poses
            }
        }

        #[test]
        fn move_inside_cube_using_only_ceres_scan_matcher() {
            let mut fixture = Fixture::new();
            fixture.local_trajectory_builder = Some(LocalTrajectoryBuilder3D::new(
                &Fixture::create_trajectory_builder_options_3d(),
                &[SENSOR_ID],
            ));
            let num_poses =
                fixture.verify_accuracy(&Fixture::generate_corkscrew_trajectory(), 1e-1);
            // Not in the C++ test: guards against a builder that silently
            // returns no results. Every scan yields a result.
            assert_eq!(num_poses, 18);
        }

        #[test]
        fn mt19937_matches_std() {
            // The 10000th output of a default-seeded std::mt19937 is
            // 4123659995 (C++ standard, [rand.predef]).
            let mut prng = Mt19937::new(5489);
            let mut value = 0;
            for _ in 0..10000 {
                value = prng.next_u32();
            }
            assert_eq!(value, 4_123_659_995);
        }
    }
}
