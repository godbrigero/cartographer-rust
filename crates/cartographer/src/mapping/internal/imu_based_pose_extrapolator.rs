//! Port of `cartographer/mapping/internal/imu_based_pose_extrapolator` (.h, .cc).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

use ceres::{
    AutoDiffLocalParameterization, ParameterBlockId, Problem, QuaternionParameterization,
    SolverOptions, SolverSummary,
};
use nalgebra::{Quaternion, Vector3};

use crate::common::histogram::Histogram;
use crate::common::internal::ceres_solver_options::create_ceres_solver_options;
use crate::common::time::{Time, from_seconds, to_seconds};
use crate::mapping::internal::eigen_quaterniond_from_two_vectors::from_two_vectors;
use crate::mapping::internal::optimization::ceres_pose::CeresPose;
use crate::mapping::internal::optimization::cost_functions::acceleration_cost_function_3d::AccelerationCostFunction3D;
use crate::mapping::internal::optimization::cost_functions::rotation_cost_function_3d::RotationCostFunction3D;
use crate::mapping::internal::optimization::cost_functions::spa_cost_function_3d::SpaCostFunction3D;
use crate::mapping::internal::three_d::imu_integration::{
    extrapolate_pose_with_imu_from_poses, integrate_imu,
};
use crate::mapping::internal::three_d::rotation_parameterization::ConstantYawQuaternionPlus;
use crate::mapping::pose_extrapolator_interface::{ExtrapolationResult, PoseExtrapolatorInterface};
use crate::mapping::pose_graph_interface::Pose;
use crate::mapping::proto;
use crate::sensor::data::HasTime;
use crate::sensor::imu_data::ImuData;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{
    Rigid3d, Rigid3f, quaternion_cast, quaternion_conjugate, quaternion_mul, quaternion_rotate,
    quaternion_slerp,
};
use crate::transform::timestamped_transform::{TimestampedTransform, interpolate};

/// Counter of `LOG_IF_EVERY_N(INFO, !fix_gravity, 20)` (a per-call-site
/// static in C++).
static GRAVITY_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Uses the linear acceleration and rotational velocities to estimate a pose.
///
/// Keeps a sliding window (`pose_queue_duration`) of poses, IMU and odometry
/// data and, on every extrapolation, solves a small nonlinear least squares
/// problem over the poses in the window plus the requested time, using SPA
/// pose constraints, IMU acceleration/rotation constraints and (if present)
/// relative odometry constraints.
#[derive(Debug)]
pub struct ImuBasedPoseExtrapolator {
    timed_pose_queue: VecDeque<TimestampedTransform>,
    previous_solution: VecDeque<TimestampedTransform>,

    imu_data: VecDeque<ImuData>,
    odometry_data: VecDeque<OdometryData>,
    last_extrapolated_time: Time,

    gravity_from_local: Rigid3d,

    options: proto::ImuBasedPoseExtrapolatorOptions,
    solver_options: SolverOptions,

    num_iterations_hist: Histogram,
}

impl ImuBasedPoseExtrapolator {
    /// Creates an extrapolator without any data.
    pub fn new(options: &proto::ImuBasedPoseExtrapolatorOptions) -> Self {
        ImuBasedPoseExtrapolator {
            timed_pose_queue: VecDeque::new(),
            previous_solution: VecDeque::new(),
            imu_data: VecDeque::new(),
            odometry_data: VecDeque::new(),
            last_extrapolated_time: Time::min(),
            gravity_from_local: Rigid3d::identity(),
            options: *options,
            solver_options: create_ceres_solver_options(
                &options.solver_options.unwrap_or_default(),
            ),
            num_iterations_hist: Histogram::new(),
        }
    }

    /// Creates an extrapolator from `imu_data` (must be non-empty) and
    /// `initial_poses`. Initial poses not strictly after the first IMU sample
    /// are ignored; without initial poses, a single gravity-aligned pose at
    /// the time of the last IMU sample is added.
    ///
    /// RUST PORT NOTE: returns `Box<Self>` instead of
    /// `std::unique_ptr<PoseExtrapolatorInterface>`; it coerces to
    /// `Box<dyn PoseExtrapolatorInterface>`.
    pub fn initialize_with_imu(
        options: &proto::ImuBasedPoseExtrapolatorOptions,
        imu_data: &[ImuData],
        initial_poses: &[TimestampedTransform],
    ) -> Box<Self> {
        assert!(!imu_data.is_empty(), "Check failed: !imu_data.empty()");
        log::info!("{options:?}");
        let mut extrapolator = Box::new(ImuBasedPoseExtrapolator::new(options));
        extrapolator.imu_data.extend(imu_data.iter().copied());
        let front_time = imu_data[0].time;
        let back = imu_data.last().expect("imu_data is not empty");
        if !initial_poses.is_empty() {
            for pose in initial_poses {
                if pose.time > front_time {
                    extrapolator.add_pose(pose.time, &pose.transform);
                }
            }
        } else {
            extrapolator.add_pose(
                back.time,
                &Rigid3d::from_rotation(from_two_vectors(&back.linear_acceleration, &Vector3::z())),
            );
        }
        extrapolator
    }

    fn trim_deque_data<T: HasTime>(
        timed_pose_queue: &VecDeque<TimestampedTransform>,
        data: &mut VecDeque<T>,
    ) {
        while data.len() > 1
            && !timed_pose_queue.is_empty()
            && data[1].time() <= timed_pose_queue[0].time
        {
            data.pop_front();
        }
    }

    fn trim_imu_data(&mut self) {
        Self::trim_deque_data(&self.timed_pose_queue, &mut self.imu_data);
    }

    fn trim_odometry_data(&mut self) {
        Self::trim_deque_data(&self.timed_pose_queue, &mut self.odometry_data);
    }

    // Odometry methods.
    fn has_odometry_data(&self) -> bool {
        self.odometry_data.len() >= 2
    }

    fn has_odometry_data_for_time(&self, time: Time) -> bool {
        self.has_odometry_data()
            && self.odometry_data[0].time < time
            && time < self.odometry_data[self.odometry_data.len() - 1].time
    }

    fn interpolate_odometry(&self, time: Time) -> Rigid3d {
        // Only interpolate if time is within odometry data range.
        assert!(
            self.has_odometry_data_for_time(time),
            "Odometry data range does not include time {time}"
        );
        // `std::upper_bound`: first element with `time < data.time`.
        let index = self.odometry_data.partition_point(|data| data.time <= time);
        let prev = &self.odometry_data[index - 1];
        let next = &self.odometry_data[index];
        let first = TimestampedTransform {
            time: prev.time,
            transform: prev.pose,
        };
        let second = TimestampedTransform {
            time: next.time,
            transform: next.pose,
        };
        interpolate(&first, &second, time).transform
    }

    fn calculate_odometry_between_nodes(
        first_node_odometry: &Rigid3d,
        second_node_odometry: &Rigid3d,
    ) -> Rigid3d {
        first_node_odometry.inverse() * *second_node_odometry
    }

    /// Interpolates (linearly in translation, slerp in rotation, in `f32`
    /// like the C++) between `start` and `end` at each of `times`.
    fn interpolate_poses(
        start: &TimestampedTransform,
        end: &TimestampedTransform,
        times: &[Time],
    ) -> Vec<Rigid3f> {
        let duration_scale = (1. / to_seconds(end.time - start.time)) as f32;

        let start_rotation: Quaternion<f32> = quaternion_cast(start.transform.rotation());
        let end_rotation: Quaternion<f32> = quaternion_cast(end.transform.rotation());
        let start_translation = start.transform.translation().cast::<f32>();
        let end_translation = end.transform.translation().cast::<f32>();

        times
            .iter()
            .map(|&time| {
                // C++: `float factor = double * float` (computed in double).
                let factor = (to_seconds(time - start.time) * f64::from(duration_scale)) as f32;
                let origin = start_translation + (end_translation - start_translation) * factor;
                let rotation = quaternion_slerp(&start_rotation, factor, &end_rotation);
                Rigid3f::new(origin, rotation)
            })
            .collect()
    }
}

impl Drop for ImuBasedPoseExtrapolator {
    fn drop(&mut self) {
        log::info!("Number of iterations for pose extrapolation:");
        log::info!("{}", self.num_iterations_hist.to_string(10));
    }
}

impl PoseExtrapolatorInterface for ImuBasedPoseExtrapolator {
    fn get_last_pose_time(&self) -> Time {
        match self.timed_pose_queue.back() {
            Some(pose) => pose.time,
            None => Time::min(),
        }
    }

    fn get_last_extrapolated_time(&self) -> Time {
        self.last_extrapolated_time
    }

    fn add_pose(&mut self, time: Time, pose: &Rigid3d) {
        self.timed_pose_queue.push_back(TimestampedTransform {
            time,
            transform: *pose,
        });
        while self.timed_pose_queue.len() > 3
            && self.timed_pose_queue[1].time
                <= time - from_seconds(self.options.pose_queue_duration)
        {
            if let Some(front) = self.previous_solution.front() {
                assert_eq!(
                    self.timed_pose_queue[0].time, front.time,
                    "Check failed: timed_pose_queue_.front().time == \
                     previous_solution_.front().time"
                );
                self.previous_solution.pop_front();
            }
            self.timed_pose_queue.pop_front();
        }
        self.trim_imu_data();
    }

    fn add_imu_data(&mut self, imu_data: &ImuData) {
        assert!(
            self.timed_pose_queue
                .back()
                .is_none_or(|pose| imu_data.time >= pose.time),
            "Check failed: timed_pose_queue_.empty() || \
             imu_data.time >= timed_pose_queue_.back().time"
        );
        self.imu_data.push_back(*imu_data);
        self.trim_imu_data();
    }

    fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        assert!(
            self.timed_pose_queue
                .back()
                .is_none_or(|pose| odometry_data.time >= pose.time),
            "Check failed: timed_pose_queue_.empty() || \
             odometry_data.time >= timed_pose_queue_.back().time"
        );
        self.odometry_data.push_back(*odometry_data);
        self.trim_odometry_data();
    }

    fn extrapolate_pose(&mut self, time: Time) -> Rigid3d {
        self.extrapolate_poses_with_gravity(&[time]).current_pose
    }

    fn extrapolate_poses_with_gravity(&mut self, times: &[Time]) -> ExtrapolationResult {
        // RUST PORT NOTE: C++ calls `times.back()` / `timed_pose_queue_.back()`
        // before `CHECK_GE(times.size(), 1)` (UB when empty); here both are
        // checked up front.
        assert!(
            !times.is_empty(),
            "Check failed: times.size() >= 1 (0 vs. 1)"
        );
        let time = *times.last().expect("times is not empty");
        let newest_timed_pose = *self
            .timed_pose_queue
            .back()
            .expect("Check failed: !timed_pose_queue_.empty()");
        assert!(
            time >= newest_timed_pose.time,
            "Check failed: time >= newest_timed_pose.time ({time} vs. {})",
            newest_timed_pose.time
        );
        self.last_extrapolated_time = time;

        if self.timed_pose_queue.len() < 3 || to_seconds(time - newest_timed_pose.time) < 1e-6 {
            let last = newest_timed_pose.transform;
            return ExtrapolationResult {
                previous_poses: vec![last.cast::<f32>(); times.len() - 1],
                current_pose: last,
                current_velocity: Vector3::zeros(),
                gravity_from_tracking: *last.rotation(),
            };
        }

        let mut problem = Problem::new();

        // Track gravity alignment over time and use this as a frame here so
        // that we can estimate the gravity alignment of the current pose.
        let gravity_from_local = CeresPose::new(
            &self.gravity_from_local,
            None,
            Some(Box::new(QuaternionParameterization)),
            &mut problem,
        );
        let mut nodes: Vec<CeresPose> = Vec::with_capacity(self.timed_pose_queue.len() + 1);
        let mut node_times: Vec<Time> = Vec::with_capacity(self.timed_pose_queue.len() + 1);

        let queue_len = self.timed_pose_queue.len();
        for (i, timed_pose) in self.timed_pose_queue.iter().enumerate() {
            let is_last = i == queue_len - 1;
            node_times.push(timed_pose.time);

            // Use the last scan match result (timed_pose_queue_.back()) for
            // initialization here instead of the last result from the
            // optimization. This keeps poses from slowly drifting apart due to
            // lack of feedback from the scan matching here.
            let gravity_from_node = if !self.previous_solution.is_empty() && !is_last {
                assert!(
                    self.previous_solution.len() > i,
                    "Check failed: previous_solution_.size() > i ({} vs. {i})",
                    self.previous_solution.len()
                );
                assert_eq!(
                    timed_pose.time, self.previous_solution[i].time,
                    "Check failed: timed_pose.time == previous_solution_[i].time"
                );
                self.previous_solution[i].transform
            } else {
                self.gravity_from_local * timed_pose.transform
            };

            if is_last {
                let node = CeresPose::new(
                    &gravity_from_node,
                    None,
                    Some(Box::new(AutoDiffLocalParameterization::<
                        ConstantYawQuaternionPlus,
                        4,
                        2,
                    >::default())),
                    &mut problem,
                );
                problem.set_parameter_block_constant(node.translation());
                nodes.push(node);
            } else {
                nodes.push(CeresPose::new(
                    &gravity_from_node,
                    None,
                    Some(Box::new(QuaternionParameterization)),
                    &mut problem,
                ));
            }
        }

        let mut gravity_constant = 9.8;
        let mut fix_gravity = false;
        if self.options.gravity_constant > 0. {
            fix_gravity = true;
            gravity_constant = self.options.gravity_constant;
        }

        let last_pose_time = newest_timed_pose.time;
        let mut imu_it_prev_prev = 0;
        while imu_it_prev_prev != self.imu_data.len() - 1
            && self.imu_data[imu_it_prev_prev + 1].time <= last_pose_time
        {
            imu_it_prev_prev += 1;
        }

        let prev_gravity_from_tracking = TimestampedTransform {
            time: node_times[node_times.len() - 1],
            transform: nodes[nodes.len() - 1].to_rigid(&problem),
        };
        let prev_prev_gravity_from_tracking = TimestampedTransform {
            time: node_times[node_times.len() - 2],
            transform: nodes[nodes.len() - 2].to_rigid(&problem),
        };
        let initial_estimate = extrapolate_pose_with_imu_from_poses(
            &prev_gravity_from_tracking.transform,
            prev_gravity_from_tracking.time,
            &prev_prev_gravity_from_tracking.transform,
            prev_prev_gravity_from_tracking.time,
            &(Vector3::z() * gravity_constant),
            time,
            &self.imu_data,
            &mut imu_it_prev_prev,
        )
        .pose;
        nodes.push(CeresPose::new(
            &initial_estimate,
            None,
            Some(Box::new(QuaternionParameterization)),
            &mut problem,
        ));
        node_times.push(time);

        // Add cost functions for node constraints.
        for (i, timed_pose) in self.timed_pose_queue.iter().enumerate() {
            problem.add_residual_block(
                SpaCostFunction3D::create_auto_diff_cost_function(&Pose {
                    zbar_ij: timed_pose.transform,
                    translation_weight: self.options.pose_translation_weight,
                    rotation_weight: self.options.pose_rotation_weight,
                }),
                None, /* loss function */
                &[
                    gravity_from_local.rotation(),
                    gravity_from_local.translation(),
                    nodes[i].rotation(),
                    nodes[i].translation(),
                ],
            );
        }

        assert!(
            !self.imu_data.is_empty(),
            "Check failed: !imu_data_.empty()"
        );
        assert!(
            self.imu_data[0].time <= self.timed_pose_queue[0].time,
            "Check failed: imu_data_.front().time <= timed_pose_queue_.front().time ({} vs. {})",
            self.imu_data[0].time,
            self.timed_pose_queue[0].time
        );

        let imu_calibration = problem.add_parameter_block_with_parameterization(
            &[1., 0., 0., 0.],
            Some(Box::new(QuaternionParameterization)),
        );
        problem.set_parameter_block_constant(imu_calibration);

        let mut imu_it = 0;
        assert!(
            self.imu_data.len() == 1 || self.imu_data[1].time > self.timed_pose_queue[0].time,
            "Check failed: imu_data_.size() == 1 || \
             std::next(imu_it)->time > timed_pose_queue_.front().time"
        );

        // RUST PORT NOTE: C++ passes `&gravity_constant` (a local double) as
        // a parameter block, which Ceres adds on first use; here the handle
        // is created lazily at the same point.
        let mut gravity_block: Option<ParameterBlockId> = None;

        let mut last_node_odometry = Rigid3d::identity();
        let mut last_node_odometry_time = Time::min();

        for i in 1..nodes.len() {
            let first_time = node_times[i - 1];
            let second_time = node_times[i];

            let mut imu_it2 = imu_it;
            let result = integrate_imu(&self.imu_data, first_time, second_time, &mut imu_it);
            if i + 1 < nodes.len() {
                let third_time = node_times[i + 1];
                let first_duration = second_time - first_time;
                let second_duration = third_time - second_time;
                let first_center = first_time + first_duration / 2;
                let second_center = second_time + second_duration / 2;
                let result_to_first_center =
                    integrate_imu(&self.imu_data, first_time, first_center, &mut imu_it2);
                let result_center_to_center =
                    integrate_imu(&self.imu_data, first_center, second_center, &mut imu_it2);
                // 'delta_velocity' is the change in velocity from the point in
                // time halfway between the first and second poses to halfway
                // between second and third pose. It is computed from IMU data
                // and still contains a delta due to gravity. The orientation
                // of this vector is in the IMU frame at the second pose.
                //
                // C++: `(q.inverse() * q2) * v` with Eigen semantics (inverse
                // = conjugate / squared norm, `_transformVector`).
                let delta_velocity = quaternion_rotate(
                    &quaternion_mul(
                        &eigen_quaternion_inverse(&result.delta_rotation),
                        &result_to_first_center.delta_rotation,
                    ),
                    &result_center_to_center.delta_velocity,
                );
                let gravity = *gravity_block
                    .get_or_insert_with(|| problem.add_parameter_block(&[gravity_constant]));
                problem.add_residual_block(
                    AccelerationCostFunction3D::create_auto_diff_cost_function(
                        self.options.imu_acceleration_weight,
                        &delta_velocity,
                        to_seconds(first_duration),
                        to_seconds(second_duration),
                    ),
                    None, /* loss function */
                    &[
                        nodes[i].rotation(),
                        nodes[i - 1].translation(),
                        nodes[i].translation(),
                        nodes[i + 1].translation(),
                        gravity,
                        imu_calibration,
                    ],
                );
                // TODO(danielsievers): Fix gravity in CostFunction.
                if fix_gravity {
                    problem.set_parameter_block_constant(gravity);
                } else {
                    // Force gravity constant to be positive.
                    problem.set_parameter_lower_bound(gravity, 0, 0.0);
                }
            }
            problem.add_residual_block(
                RotationCostFunction3D::create_auto_diff_cost_function(
                    self.options.imu_rotation_weight,
                    &result.delta_rotation,
                ),
                None, /* loss function */
                &[
                    nodes[i - 1].rotation(),
                    nodes[i].rotation(),
                    imu_calibration,
                ],
            );

            // Add a relative pose constraint based on the odometry (if
            // available).
            if self.has_odometry_data_for_time(first_time)
                && self.has_odometry_data_for_time(second_time)
            {
                // Here keep track of last node odometry to avoid double
                // computation. Do this if first loop, or if there were some
                // missing odometry nodes then recalculate.
                if i == 1 || last_node_odometry_time != first_time {
                    last_node_odometry = self.interpolate_odometry(first_time);
                    // RUST PORT NOTE: C++ also sets `last_node_odometry_time =
                    // first_time` here; it is always overwritten below before
                    // being read, so the dead store is omitted.
                }
                let current_node_odometry = self.interpolate_odometry(second_time);
                let relative_odometry = Self::calculate_odometry_between_nodes(
                    &last_node_odometry,
                    &current_node_odometry,
                );

                problem.add_residual_block(
                    SpaCostFunction3D::create_auto_diff_cost_function(&Pose {
                        zbar_ij: relative_odometry,
                        translation_weight: self.options.odometry_translation_weight,
                        rotation_weight: self.options.odometry_rotation_weight,
                    }),
                    None, /* loss function */
                    &[
                        nodes[i - 1].rotation(),
                        nodes[i - 1].translation(),
                        nodes[i].rotation(),
                        nodes[i].translation(),
                    ],
                );
                // Use the current node odometry in the next iteration.
                last_node_odometry = current_node_odometry;
                last_node_odometry_time = second_time;
            }
        }

        // Solve.
        let mut summary = SolverSummary::default();
        ceres::solve(&self.solver_options, &mut problem, &mut summary);
        if !fix_gravity
            && GRAVITY_LOG_COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(20)
        {
            let gravity =
                gravity_block.map_or(gravity_constant, |id| problem.parameter_block(id)[0]);
            log::info!("Gravity was: {gravity}");
        }

        let last_node = nodes[nodes.len() - 1].to_rigid(&problem);
        let gravity_estimate = *last_node.rotation();

        let last_pose = newest_timed_pose;
        let extrapolated_pose = TimestampedTransform {
            time,
            transform: last_pose.transform
                * nodes[nodes.len() - 2].to_rigid(&problem).inverse()
                * last_node,
        };

        self.num_iterations_hist
            .add(summary.iterations.len() as f32);

        self.gravity_from_local = gravity_from_local.to_rigid(&problem);

        self.previous_solution.clear();
        for (node, &node_time) in nodes.iter().zip(&node_times) {
            self.previous_solution.push_back(TimestampedTransform {
                time: node_time,
                transform: node.to_rigid(&problem),
            });
        }

        let current_velocity = (extrapolated_pose.transform.translation()
            - last_pose.transform.translation())
            / to_seconds(time - last_pose.time);

        ExtrapolationResult {
            previous_poses: Self::interpolate_poses(
                &last_pose,
                &extrapolated_pose,
                &times[..times.len() - 1],
            ),
            current_pose: extrapolated_pose.transform,
            current_velocity,
            gravity_from_tracking: gravity_estimate,
        }
    }

    fn estimate_gravity_orientation(&mut self, time: Time) -> Quaternion<f64> {
        self.extrapolate_poses_with_gravity(&[time])
            .gravity_from_tracking
    }
}

/// Eigen `Quaterniond::inverse()`: conjugate divided by the squared norm
/// (zero quaternion for a zero norm).
fn eigen_quaternion_inverse(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = q.norm_squared();
    if n2 > 0. {
        quaternion_conjugate(q) / n2
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

#[cfg(test)]
mod tests {
    use cartographer_sim::{ImuConfig, Trajectory};
    use nalgebra::Isometry3;

    use super::*;
    use crate::common::time::{from_milliseconds, from_universal};
    use crate::transform::rigid_transform::AngleAxis;
    use crate::transform::transform::get_angle;

    fn options(
        pose_queue_duration: f64,
        gravity_constant: f64,
    ) -> proto::ImuBasedPoseExtrapolatorOptions {
        // The values of `configuration_files/trajectory_builder_3d.lua`
        // except for the queue duration and gravity constant.
        proto::ImuBasedPoseExtrapolatorOptions {
            pose_queue_duration,
            gravity_constant,
            pose_translation_weight: 1.,
            pose_rotation_weight: 1.,
            imu_acceleration_weight: 1.,
            imu_rotation_weight: 1.,
            odometry_translation_weight: 1.,
            odometry_rotation_weight: 1.,
            solver_options: Some(crate::common::proto::CeresSolverOptions {
                use_nonmonotonic_steps: false,
                max_num_iterations: 10,
                num_threads: 1,
            }),
        }
    }

    fn to_rigid(pose: &Isometry3<f64>) -> Rigid3d {
        Rigid3d::new(pose.translation.vector, *pose.rotation.quaternion())
    }

    /// IMU tick period (200 Hz).
    const IMU_PERIOD_MS: i64 = 5;
    /// IMU ticks per pose (10 Hz).
    const TICKS_PER_POSE: i64 = 20;
    /// Gravity of the simulation (also the C++ default `gravity_constant`).
    const GRAVITY: f64 = 9.806;

    /// Noise-free simulated 3D motion: IMU at 200 Hz, poses at 10 Hz.
    struct Simulation {
        trajectory: Trajectory,
        imu: ImuConfig,
    }

    impl Simulation {
        fn new() -> Self {
            // Walking-speed (1 m/s) tour with turns, 3 cm vertical bobbing
            // and 2-3 deg roll/pitch sway.
            Simulation {
                trajectory: Trajectory::room_wander_3d(1.0, 1.0),
                imu: ImuConfig {
                    gravity: GRAVITY,
                    ..ImuConfig::ideal()
                },
            }
        }
        fn seconds(tick: i64) -> f64 {
            1.0 + (tick * IMU_PERIOD_MS) as f64 * 1e-3
        }
        fn time(tick: i64) -> Time {
            from_universal(1_000_000_000) + from_milliseconds(tick * IMU_PERIOD_MS)
        }
        fn imu_data(&self, tick: i64) -> ImuData {
            let m = self.imu.measure(&self.trajectory, Self::seconds(tick));
            ImuData {
                time: Self::time(tick),
                linear_acceleration: m.linear_acceleration,
                angular_velocity: m.angular_velocity,
            }
        }
        fn pose(&self, tick: i64) -> Rigid3d {
            to_rigid(&self.trajectory.state(Self::seconds(tick)).pose)
        }
        fn odometry(&self, tick: i64) -> OdometryData {
            OdometryData {
                time: Self::time(tick),
                pose: self.pose(tick),
            }
        }
    }

    #[derive(Default, Debug)]
    struct Errors {
        count: usize,
        max_translation: f64,
        max_rotation: f64,
        max_interpolated_translation: f64,
        max_gravity_angle: f64,
        max_velocity: f64,
        /// Baseline: constant velocity from the last two ground truth poses.
        max_constant_velocity_translation: f64,
    }

    /// Runs the extrapolator over `num_poses` pose intervals. Within each
    /// interval it extrapolates at 25, 50, 75 and 100 ms after the last pose
    /// (asking for all 5 ms ticks since the last pose as `times`, like a
    /// range data accumulation), then adds the ground truth pose. Errors are
    /// measured once `warm_up` poses were added.
    fn run(
        options: &proto::ImuBasedPoseExtrapolatorOptions,
        num_poses: i64,
        warm_up: i64,
        with_odometry: bool,
    ) -> Errors {
        let sim = Simulation::new();
        let initial_imu: Vec<ImuData> = (0..=TICKS_PER_POSE).map(|j| sim.imu_data(j)).collect();
        let mut extrapolator = ImuBasedPoseExtrapolator::initialize_with_imu(
            options,
            &initial_imu,
            &[TimestampedTransform {
                time: Simulation::time(TICKS_PER_POSE),
                transform: sim.pose(TICKS_PER_POSE),
            }],
        );
        assert_eq!(
            extrapolator.get_last_pose_time(),
            Simulation::time(TICKS_PER_POSE)
        );
        if with_odometry {
            extrapolator.add_odometry_data(&sim.odometry(TICKS_PER_POSE));
        }
        let mut errors = Errors::default();
        for k in 1..num_poses {
            let pose_tick = k * TICKS_PER_POSE;
            for j in pose_tick + 1..=pose_tick + TICKS_PER_POSE {
                extrapolator.add_imu_data(&sim.imu_data(j));
                if with_odometry && j % 2 == 0 {
                    extrapolator.add_odometry_data(&sim.odometry(j));
                }
                if (j - pose_tick) % 5 != 0 {
                    continue;
                }
                let times: Vec<Time> = (pose_tick + 1..=j).map(Simulation::time).collect();
                let result = extrapolator.extrapolate_poses_with_gravity(&times);
                assert_eq!(result.previous_poses.len(), times.len() - 1);
                assert_eq!(
                    extrapolator.get_last_extrapolated_time(),
                    Simulation::time(j)
                );
                if k < warm_up {
                    continue;
                }
                errors.count += 1;
                let expected = sim.pose(j);
                let delta = expected.inverse() * result.current_pose;
                errors.max_translation = errors.max_translation.max(delta.translation().norm());
                errors.max_rotation = errors.max_rotation.max(get_angle(&delta));
                for (i, pose) in result.previous_poses.iter().enumerate() {
                    let expected = sim.pose(pose_tick + 1 + i as i64);
                    let error = (pose.translation().cast::<f64>() - expected.translation()).norm();
                    errors.max_interpolated_translation =
                        errors.max_interpolated_translation.max(error);
                }
                // The gravity estimate maps "up" in the tracking frame onto +z.
                let up_in_tracking =
                    quaternion_rotate(&quaternion_conjugate(expected.rotation()), &Vector3::z());
                let up = quaternion_rotate(&result.gravity_from_tracking, &up_in_tracking);
                errors.max_gravity_angle = errors.max_gravity_angle.max(up.angle(&Vector3::z()));
                // `current_velocity` is the average velocity since the last
                // pose.
                let expected_velocity = (expected.translation()
                    - sim.pose(pose_tick).translation())
                    / to_seconds(Simulation::time(j) - Simulation::time(pose_tick));
                errors.max_velocity = errors
                    .max_velocity
                    .max((result.current_velocity - expected_velocity).norm());
                let last = sim.pose(pose_tick);
                let prev = sim.pose(pose_tick - TICKS_PER_POSE);
                let constant_velocity = last.translation()
                    + (last.translation() - prev.translation())
                        * ((j - pose_tick) as f64 / TICKS_PER_POSE as f64);
                errors.max_constant_velocity_translation = errors
                    .max_constant_velocity_translation
                    .max((constant_velocity - expected.translation()).norm());
            }
            let next = pose_tick + TICKS_PER_POSE;
            extrapolator.add_pose(Simulation::time(next), &sim.pose(next));
        }
        errors
    }

    /// Accuracy thresholds for the noise-free simulation, with ~2x margin
    /// over the measured values (printed by the tests).
    ///
    /// The remaining error is inherent to the C++ formulation, not to IMU
    /// integration (near exact at 200 Hz for this motion): the acceleration
    /// residual equates the IMU velocity change between interval midpoints
    /// with the change of *chord* velocities `dp / dt`. With the 3 cm /
    /// 1.8 Hz bobbing (jerk up to ~43 m/s^3) the chord velocity differs from
    /// the midpoint velocity by ~jerk dt^2 / 24 ~ 2 cm/s, i.e. ~2 mm over a
    /// 100 ms horizon. Rotation is integrated from the gyro directly
    /// (~0.3 mrad). `current_velocity` is the average velocity since the last
    /// pose, so its error is the translation error divided by the (as short
    /// as 25 ms) horizon.
    fn check_accuracy(errors: &Errors, max_translation: f64) {
        assert!(errors.count > 0);
        assert!(errors.max_translation < max_translation, "{errors:?}");
        assert!(errors.max_rotation < 1e-3, "{errors:?}");
        // Linear interpolation between the last pose and the extrapolated
        // pose additionally has the chord error of the curved path.
        assert!(errors.max_interpolated_translation < 1e-2, "{errors:?}");
        assert!(errors.max_gravity_angle < 2e-3, "{errors:?}");
        assert!(errors.max_velocity < max_translation / 0.025, "{errors:?}");
        // The IMU must clearly beat constant-velocity extrapolation of the
        // ground truth poses.
        assert!(
            errors.max_translation < 0.5 * errors.max_constant_velocity_translation,
            "{errors:?}"
        );
    }

    #[test]
    fn simulated_motion_fixed_gravity() {
        // Default C++ options: 5 s window (50 poses).
        let errors = run(&options(5., GRAVITY), 70, 5, false);
        eprintln!("fixed gravity: {errors:?}");
        assert_eq!(errors.count, 65 * 4);
        check_accuracy(&errors, 5e-3);
    }

    #[test]
    fn simulated_motion_estimated_gravity() {
        // `gravity_constant <= 0`: the gravity constant is optimized.
        let errors = run(&options(1., 0.), 40, 5, false);
        eprintln!("estimated gravity: {errors:?}");
        check_accuracy(&errors, 1e-2);
    }

    #[test]
    fn simulated_motion_with_odometry() {
        let errors = run(&options(1., GRAVITY), 40, 5, true);
        eprintln!("with odometry: {errors:?}");
        check_accuracy(&errors, 5e-3);
    }

    fn imu(time: Time, linear_acceleration: Vector3<f64>) -> ImuData {
        ImuData {
            time,
            linear_acceleration,
            angular_velocity: Vector3::zeros(),
        }
    }

    /// Feeds ground truth poses `2..=num_poses` (and IMU data), extrapolating
    /// to each pose time before adding it, like the local trajectory builder.
    fn simulated_extrapolator(
        options: &proto::ImuBasedPoseExtrapolatorOptions,
        num_poses: i64,
    ) -> (Simulation, Box<ImuBasedPoseExtrapolator>) {
        let sim = Simulation::new();
        let initial_imu: Vec<ImuData> = (0..=TICKS_PER_POSE).map(|j| sim.imu_data(j)).collect();
        let mut extrapolator = ImuBasedPoseExtrapolator::initialize_with_imu(
            options,
            &initial_imu,
            &[TimestampedTransform {
                time: Simulation::time(TICKS_PER_POSE),
                transform: sim.pose(TICKS_PER_POSE),
            }],
        );
        for k in 2..=num_poses {
            let tick = k * TICKS_PER_POSE;
            for j in tick - TICKS_PER_POSE + 1..=tick {
                extrapolator.add_imu_data(&sim.imu_data(j));
            }
            extrapolator.extrapolate_pose(Simulation::time(tick));
            extrapolator.add_pose(Simulation::time(tick), &sim.pose(tick));
        }
        (sim, extrapolator)
    }

    #[test]
    #[should_panic(expected = "!imu_data.empty()")]
    fn initialize_with_imu_requires_imu_data() {
        ImuBasedPoseExtrapolator::initialize_with_imu(&options(1., GRAVITY), &[], &[]);
    }

    #[test]
    fn initialize_without_poses_is_gravity_aligned() {
        let t0 = from_universal(1000);
        let t1 = t0 + from_milliseconds(5);
        let acceleration = Vector3::new(0., 3., 9.);
        let imu_data = [imu(t0, Vector3::new(0., 0., 9.8)), imu(t1, acceleration)];
        let mut extrapolator =
            ImuBasedPoseExtrapolator::initialize_with_imu(&options(1., GRAVITY), &imu_data, &[]);
        assert_eq!(extrapolator.get_last_pose_time(), t1);
        assert_eq!(extrapolator.get_last_extrapolated_time(), Time::min());

        // Fewer than 3 poses: the last pose is returned for all times.
        let t2 = t1 + from_milliseconds(50);
        let times = [t1, t1 + from_milliseconds(10), t2];
        let result = extrapolator.extrapolate_poses_with_gravity(&times);
        assert_eq!(extrapolator.get_last_extrapolated_time(), t2);
        assert_eq!(result.previous_poses.len(), 2);
        assert_eq!(result.current_pose.translation(), &Vector3::zeros());
        assert_eq!(result.current_velocity, Vector3::zeros());
        assert_eq!(
            result.gravity_from_tracking,
            *result.current_pose.rotation()
        );
        for pose in &result.previous_poses {
            assert_eq!(*pose, result.current_pose.cast::<f32>());
        }
        let up = quaternion_rotate(&result.gravity_from_tracking, &acceleration.normalize());
        assert!((up - Vector3::z()).norm() < 1e-12, "{up:?}");
        assert_eq!(
            extrapolator.estimate_gravity_orientation(t2),
            result.gravity_from_tracking
        );
        assert_eq!(extrapolator.extrapolate_pose(t2), result.current_pose);
    }

    #[test]
    fn initial_poses_not_after_first_imu_are_ignored() {
        let t0 = from_universal(1000);
        let imu_data = [imu(t0, Vector3::new(0., 0., 9.8))];
        let poses = [
            TimestampedTransform {
                time: t0 - from_milliseconds(1),
                transform: Rigid3d::identity(),
            },
            TimestampedTransform {
                time: t0,
                transform: Rigid3d::identity(),
            },
        ];
        let extrapolator =
            ImuBasedPoseExtrapolator::initialize_with_imu(&options(1., GRAVITY), &imu_data, &poses);
        assert_eq!(extrapolator.get_last_pose_time(), Time::min());
    }

    #[test]
    #[should_panic(expected = "!timed_pose_queue_.empty()")]
    fn extrapolate_without_poses_panics() {
        let t0 = from_universal(1000);
        let mut extrapolator = ImuBasedPoseExtrapolator::new(&options(1., GRAVITY));
        extrapolator.add_imu_data(&imu(t0, Vector3::new(0., 0., 9.8)));
        extrapolator.extrapolate_pose(t0);
    }

    #[test]
    #[should_panic(expected = "times.size() >= 1")]
    fn extrapolate_empty_times_panics() {
        let (_, mut extrapolator) = simulated_extrapolator(&options(1., GRAVITY), 3);
        extrapolator.extrapolate_poses_with_gravity(&[]);
    }

    #[test]
    #[should_panic(expected = "time >= newest_timed_pose.time")]
    fn extrapolate_before_last_pose_panics() {
        let (_, mut extrapolator) = simulated_extrapolator(&options(1., GRAVITY), 3);
        extrapolator.extrapolate_pose(Simulation::time(3 * TICKS_PER_POSE - 1));
    }

    #[test]
    #[should_panic(expected = "imu_data.time >= timed_pose_queue_.back().time")]
    fn imu_data_before_last_pose_panics() {
        let (sim, mut extrapolator) = simulated_extrapolator(&options(1., GRAVITY), 3);
        extrapolator.add_imu_data(&sim.imu_data(3 * TICKS_PER_POSE - 1));
    }

    #[test]
    #[should_panic(expected = "odometry_data.time >= timed_pose_queue_.back().time")]
    fn odometry_data_before_last_pose_panics() {
        let (sim, mut extrapolator) = simulated_extrapolator(&options(1., GRAVITY), 3);
        extrapolator.add_odometry_data(&sim.odometry(3 * TICKS_PER_POSE - 1));
    }

    #[test]
    fn extrapolate_at_last_pose_time_returns_last_pose() {
        let (sim, mut extrapolator) = simulated_extrapolator(&options(1., GRAVITY), 5);
        let tick = 5 * TICKS_PER_POSE;
        let result = extrapolator.extrapolate_poses_with_gravity(&[Simulation::time(tick)]);
        assert_eq!(result.current_pose, sim.pose(tick));
        assert_eq!(result.current_velocity, Vector3::zeros());
        assert!(result.previous_poses.is_empty());
    }

    #[test]
    fn trims_poses_and_imu_data() {
        // 0.25 s window with poses every 0.1 s: the front pose is dropped
        // once the second one is at least 0.25 s old (keeping >= 3 poses).
        let (_, extrapolator) = simulated_extrapolator(&options(0.25, GRAVITY), 10);
        let last = Simulation::time(10 * TICKS_PER_POSE);
        let times: Vec<Time> = extrapolator
            .timed_pose_queue
            .iter()
            .map(|p| p.time)
            .collect();
        let expected: Vec<Time> = (7..=10)
            .map(|k| Simulation::time(k * TICKS_PER_POSE))
            .collect();
        assert_eq!(times, expected);
        assert_eq!(extrapolator.get_last_pose_time(), last);
        // IMU data is trimmed to start at the first pose.
        assert_eq!(extrapolator.imu_data[0].time, times[0]);
        assert_eq!(extrapolator.imu_data.back().map(|d| d.time), Some(last));
        // The previous solution is trimmed alongside the pose queue.
        assert_eq!(extrapolator.previous_solution[0].time, times[0]);
    }

    #[test]
    #[should_panic(expected = "timed_pose.time == previous_solution_[i].time")]
    fn two_poses_without_extrapolation_panics() {
        // Like C++: the extrapolated node of the previous solution must be at
        // the time of the next pose, so two poses must not be added after an
        // extrapolation to a different time.
        let (sim, mut extrapolator) = simulated_extrapolator(&options(5., GRAVITY), 4);
        let tick = 4 * TICKS_PER_POSE;
        for j in tick + 1..=tick + 2 * TICKS_PER_POSE {
            extrapolator.add_imu_data(&sim.imu_data(j));
            if j == tick + 10 {
                extrapolator.extrapolate_pose(Simulation::time(j));
            }
            if (j - tick) % TICKS_PER_POSE == 0 {
                extrapolator.add_pose(Simulation::time(j), &sim.pose(j));
            }
        }
        extrapolator.add_imu_data(&sim.imu_data(tick + 2 * TICKS_PER_POSE + 1));
        extrapolator.extrapolate_pose(Simulation::time(tick + 2 * TICKS_PER_POSE + 1));
    }

    #[test]
    fn odometry_interpolation() {
        let mut extrapolator = ImuBasedPoseExtrapolator::new(&options(1., GRAVITY));
        let t0 = from_universal(1000);
        let t1 = t0 + from_milliseconds(100);
        assert!(!extrapolator.has_odometry_data());
        extrapolator.add_odometry_data(&OdometryData {
            time: t0,
            pose: Rigid3d::identity(),
        });
        extrapolator.add_odometry_data(&OdometryData {
            time: t1,
            pose: Rigid3d::from_translation(Vector3::new(2., 0., 0.)),
        });
        assert!(extrapolator.has_odometry_data());
        // The range is open on both ends.
        assert!(!extrapolator.has_odometry_data_for_time(t0));
        assert!(!extrapolator.has_odometry_data_for_time(t1));
        let mid = t0 + from_milliseconds(25);
        assert!(extrapolator.has_odometry_data_for_time(mid));
        let interpolated = extrapolator.interpolate_odometry(mid);
        assert!((interpolated.translation() - Vector3::new(0.5, 0., 0.)).norm() < 1e-12);
        let relative = ImuBasedPoseExtrapolator::calculate_odometry_between_nodes(
            &Rigid3d::from_translation(Vector3::new(1., 0., 0.)),
            &Rigid3d::from_translation(Vector3::new(3., 0., 0.)),
        );
        assert_eq!(relative.translation(), &Vector3::new(2., 0., 0.));
    }

    #[test]
    #[should_panic(expected = "Odometry data range does not include time")]
    fn odometry_interpolation_outside_range_panics() {
        let mut extrapolator = ImuBasedPoseExtrapolator::new(&options(1., GRAVITY));
        let t0 = from_universal(1000);
        for ms in [0, 100] {
            extrapolator.add_odometry_data(&OdometryData {
                time: t0 + from_milliseconds(ms),
                pose: Rigid3d::identity(),
            });
        }
        extrapolator.interpolate_odometry(t0 + from_milliseconds(200));
    }

    #[test]
    fn interpolate_poses_lerps_and_slerps_in_float() {
        let t0 = from_universal(0);
        let start = TimestampedTransform {
            time: t0,
            transform: Rigid3d::identity(),
        };
        let end = TimestampedTransform {
            time: t0 + from_milliseconds(100),
            transform: Rigid3d::new(
                Vector3::new(1., 2., 3.),
                AngleAxis::new(0.4, Vector3::z()).to_quaternion(),
            ),
        };
        let poses = ImuBasedPoseExtrapolator::interpolate_poses(
            &start,
            &end,
            &[t0 + from_milliseconds(25)],
        );
        assert_eq!(poses.len(), 1);
        assert!((poses[0].translation() - Vector3::new(0.25f32, 0.5, 0.75)).norm() < 1e-6);
        let expected = AngleAxis::new(0.1f32, Vector3::z()).to_quaternion();
        assert!((poses[0].rotation() - expected).norm() < 1e-6);
    }
}
