//! Port of `cartographer/mapping/internal/optimization/optimization_problem_3d` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ code keeps the Ceres parameters in `CeresPose`s
//! and in `trajectory_data_` (`imu_calibration`, `gravity_constant`) whose
//! memory Ceres optimizes in place. The Rust `ceres::Problem` owns the
//! parameter values, so `solve` keeps parameter block handles and copies the
//! optimized values back (IMU calibration and gravity constant right after
//! solving, before the summary logging that prints them, as in C++).
//! Parameter blocks that C++ adds implicitly through `AddResidualBlock`
//! (gravity constants) are added explicitly.

use std::collections::BTreeMap;

use ceres::{
    AutoDiffLocalParameterization, LocalParameterization, ParameterBlockId, Problem,
    QuaternionParameterization, SubsetParameterization,
};
use nalgebra::Vector3;

use crate::common::math::rad_to_deg;
use crate::common::time::{Time, to_seconds};
use crate::mapping::id::{IdDataReference, MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::ceres_pose::CeresPose;
use crate::mapping::internal::optimization::cost_functions::acceleration_cost_function_3d::AccelerationCostFunction3D;
use crate::mapping::internal::optimization::cost_functions::cost_helpers::interpolate_nodes_3d;
use crate::mapping::internal::optimization::cost_functions::landmark_cost_function_3d::LandmarkCostFunction3D;
use crate::mapping::internal::optimization::cost_functions::rotation_cost_function_3d::RotationCostFunction3D;
use crate::mapping::internal::optimization::cost_functions::spa_cost_function_3d::SpaCostFunction3D;
use crate::mapping::internal::optimization::optimization_problem_2d::{
    ceres_solver_options, constraint_loss, find_landmark_observation_nodes, fixed_frame_pose_loss,
    frozen_trajectories, interpolate_fixed_frame_pose, interpolate_odometry,
    set_max_num_iterations,
};
use crate::mapping::internal::optimization::optimization_problem_interface::OptimizationProblemInterface;
use crate::mapping::internal::three_d::imu_integration::integrate_imu;
use crate::mapping::internal::three_d::rotation_parameterization::{
    ConstantYawQuaternionPlus, YawOnlyQuaternionPlus,
};
use crate::mapping::optimization::proto;
use crate::mapping::pose_graph_interface::{
    Constraint, LandmarkNode, LandmarkObservation, Pose, TrajectoryData, TrajectoryState,
};
use crate::sensor::data::HasTime;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{AngleAxis, Rigid3d, quaternion_mul, quaternion_rotate};
use crate::transform::transform::get_yaw_quaternion;

/// Per-node data of the 3D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec3D {
    /// Node time.
    pub time: Time,
    /// Pose in the local (non-loop-closed) frame.
    pub local_pose: Rigid3d,
    /// Pose in the global (loop-closed) frame.
    pub global_pose: Rigid3d,
}

impl HasTime for NodeSpec3D {
    fn time(&self) -> Time {
        self.time
    }
}

/// Per-submap data of the 3D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapSpec3D {
    /// Global pose of the submap.
    pub global_pose: Rigid3d,
}

/// Selects a trajectory node closest in time to the landmark observation and
/// applies a relative transform from it.
fn get_initial_landmark_pose(
    observation: &LandmarkObservation,
    prev_node: &NodeSpec3D,
    next_node: &NodeSpec3D,
    prev_node_pose: &CeresPose,
    next_node_pose: &CeresPose,
    problem: &Problem<'_>,
) -> Rigid3d {
    let interpolation_parameter =
        to_seconds(observation.time - prev_node.time) / to_seconds(next_node.time - prev_node.time);

    let (rotation, translation) = interpolate_nodes_3d(
        problem.parameter_block(prev_node_pose.rotation()),
        problem.parameter_block(prev_node_pose.translation()),
        problem.parameter_block(next_node_pose.rotation()),
        problem.parameter_block(next_node_pose.translation()),
        interpolation_parameter,
    );
    Rigid3d::from_arrays(&rotation, &translation) * observation.landmark_to_tracking_transform
}

fn add_landmark_cost_functions(
    landmark_nodes: &BTreeMap<String, LandmarkNode>,
    node_data: &MapById<NodeId, NodeSpec3D>,
    c_nodes: &MapById<NodeId, CeresPose>,
    c_landmarks: &mut BTreeMap<String, CeresPose>,
    problem: &mut Problem<'_>,
    huber_scale: f64,
) {
    for (landmark_id, landmark_node) in landmark_nodes {
        // Do not use landmarks that were not optimized for localization.
        for observation in &landmark_node.landmark_observations {
            let Some((prev, next)) =
                find_landmark_observation_nodes(node_data, observation, landmark_id)
            else {
                continue;
            };
            // Add parameter blocks for the landmark ID if they were not added
            // before.
            let prev_node_pose = *c_nodes.at(&prev.id);
            let next_node_pose = *c_nodes.at(&next.id);
            if !c_landmarks.contains_key(landmark_id) {
                let starting_point = landmark_node.global_landmark_pose.unwrap_or_else(|| {
                    get_initial_landmark_pose(
                        observation,
                        prev.data,
                        next.data,
                        &prev_node_pose,
                        &next_node_pose,
                        problem,
                    )
                });
                let pose = CeresPose::new(
                    &starting_point,
                    None, /* translation_parametrization */
                    Some(Box::new(QuaternionParameterization)),
                    problem,
                );
                // Set landmark constant if it is frozen.
                if landmark_node.frozen {
                    problem.set_parameter_block_constant(pose.translation());
                    problem.set_parameter_block_constant(pose.rotation());
                }
                c_landmarks.insert(landmark_id.clone(), pose);
            }
            let landmark = c_landmarks[landmark_id];
            problem.add_residual_block(
                LandmarkCostFunction3D::create_auto_diff_cost_function(
                    observation,
                    prev.data,
                    next.data,
                ),
                Some(Box::new(ceres::HuberLoss::new(huber_scale))),
                &[
                    prev_node_pose.rotation(),
                    prev_node_pose.translation(),
                    next_node_pose.rotation(),
                    next_node_pose.translation(),
                    landmark.rotation(),
                    landmark.translation(),
                ],
            );
        }
    }
}

/// The 3D sparse pose adjustment problem.
#[derive(Debug)]
pub struct OptimizationProblem3D {
    options: proto::OptimizationProblemOptions,
    node_data: MapById<NodeId, NodeSpec3D>,
    submap_data: MapById<SubmapId, SubmapSpec3D>,
    landmark_data: BTreeMap<String, Rigid3d>,
    imu_data: MapByTime<ImuData>,
    odometry_data: MapByTime<OdometryData>,
    fixed_frame_pose_data: MapByTime<FixedFramePoseData>,
    trajectory_data: BTreeMap<i32, TrajectoryData>,
}

impl OptimizationProblem3D {
    /// Creates an empty problem.
    pub fn new(options: &proto::OptimizationProblemOptions) -> Self {
        OptimizationProblem3D {
            options: *options,
            node_data: MapById::new(),
            submap_data: MapById::new(),
            landmark_data: BTreeMap::new(),
            imu_data: MapByTime::new(),
            odometry_data: MapByTime::new(),
            fixed_frame_pose_data: MapByTime::new(),
            trajectory_data: BTreeMap::new(),
        }
    }

    /// Adds fixed frame pose data of `trajectory_id`.
    pub fn add_fixed_frame_pose_data(
        &mut self,
        trajectory_id: i32,
        fixed_frame_pose_data: &FixedFramePoseData,
    ) {
        self.fixed_frame_pose_data
            .append(trajectory_id, *fixed_frame_pose_data);
    }

    /// Sets the trajectory data of `trajectory_id`.
    pub fn set_trajectory_data(&mut self, trajectory_id: i32, trajectory_data: &TrajectoryData) {
        self.trajectory_data.insert(trajectory_id, *trajectory_data);
    }

    /// The fixed frame pose data.
    pub fn fixed_frame_pose_data(&self) -> &MapByTime<FixedFramePoseData> {
        &self.fixed_frame_pose_data
    }

    /// The per-trajectory data.
    pub fn trajectory_data(&self) -> &BTreeMap<i32, TrajectoryData> {
        &self.trajectory_data
    }

    /// Computes the relative pose between two nodes based on odometry data.
    fn calculate_odometry_between_nodes(
        &self,
        trajectory_id: i32,
        first_node_data: &NodeSpec3D,
        second_node_data: &NodeSpec3D,
    ) -> Option<Rigid3d> {
        if !self.odometry_data.has_trajectory(trajectory_id) {
            return None;
        }
        let first_node_odometry =
            interpolate_odometry(&self.odometry_data, trajectory_id, first_node_data.time)?;
        let second_node_odometry =
            interpolate_odometry(&self.odometry_data, trajectory_id, second_node_data.time)?;
        Some(first_node_odometry.inverse() * second_node_odometry)
    }

    /// `SubsetParameterization(3, {2})` if `fix_z_in_3d`, else none.
    fn translation_parameterization(&self) -> Option<Box<dyn LocalParameterization>> {
        self.options.fix_z_in_3d.then(|| {
            Box::new(SubsetParameterization::new(3, &[2])) as Box<dyn LocalParameterization>
        })
    }

    /// Adds the IMU rotation and acceleration terms of all non-frozen
    /// trajectories. Returns the IMU calibration and gravity constant blocks
    /// per trajectory.
    fn add_imu_cost_functions(
        &self,
        frozen_trajectories: &std::collections::BTreeSet<i32>,
        c_nodes: &MapById<NodeId, CeresPose>,
        problem: &mut Problem<'_>,
    ) -> (
        BTreeMap<i32, ParameterBlockId>,
        BTreeMap<i32, ParameterBlockId>,
    ) {
        let mut c_imu_calibrations = BTreeMap::new();
        let mut c_gravity_constants = BTreeMap::new();
        for trajectory_id in self.node_data.trajectory_ids() {
            if frozen_trajectories.contains(&trajectory_id) {
                // We skip frozen trajectories.
                continue;
            }
            let trajectory_data = self
                .trajectory_data
                .get(&trajectory_id)
                .unwrap_or_else(|| panic!("map::at: no trajectory data for {trajectory_id}"));

            let imu_calibration = problem.add_parameter_block_with_parameterization(
                &trajectory_data.imu_calibration,
                Some(Box::new(QuaternionParameterization)),
            );
            c_imu_calibrations.insert(trajectory_id, imu_calibration);
            if !self.options.use_online_imu_extrinsics_in_3d {
                problem.set_parameter_block_constant(imu_calibration);
            }
            assert!(
                self.imu_data.has_trajectory(trajectory_id),
                "Check failed: imu_data_.HasTrajectory(trajectory_id): 3D optimization without \
                 fix_z_in_3d needs IMU data, but trajectory {trajectory_id} has none."
            );
            let imu_data: Vec<ImuData> = self.imu_data.trajectory(trajectory_id).copied().collect();
            assert!(
                !imu_data.is_empty(),
                "Check failed: imu_data.begin() != imu_data.end()"
            );

            let nodes: Vec<IdDataReference<'_, NodeId, NodeSpec3D>> =
                self.node_data.trajectory(trajectory_id).collect();
            let mut imu_it = 0usize;
            let mut gravity_constant: Option<ParameterBlockId> = None;
            for i in 1..nodes.len() {
                let (first_node_id, first_node_data) = (nodes[i - 1].id, nodes[i - 1].data);
                let (second_node_id, second_node_data) = (nodes[i].id, nodes[i].data);

                if second_node_id.node_index != first_node_id.node_index + 1 {
                    continue;
                }

                // Skip IMU data before the node.
                while imu_it + 1 < imu_data.len()
                    && imu_data[imu_it + 1].time <= first_node_data.time
                {
                    imu_it += 1;
                }

                let mut imu_it2 = imu_it;
                let result = integrate_imu(
                    &imu_data,
                    first_node_data.time,
                    second_node_data.time,
                    &mut imu_it,
                );
                let first_time = first_node_data.time;
                let second_time = second_node_data.time;
                let first_duration = second_time - first_time;
                if let Some(third) = nodes
                    .get(i + 1)
                    .filter(|third| third.id.node_index == second_node_id.node_index + 1)
                {
                    let third_node_id = third.id;
                    let third_time = third.data.time;
                    let second_duration = third_time - second_time;
                    let first_center = first_time + first_duration / 2;
                    let second_center = second_time + second_duration / 2;
                    let result_to_first_center =
                        integrate_imu(&imu_data, first_time, first_center, &mut imu_it2);
                    let result_center_to_center =
                        integrate_imu(&imu_data, first_center, second_center, &mut imu_it2);
                    // 'delta_velocity' is the change in velocity from the point
                    // in time halfway between the first and second poses to
                    // halfway between second and third pose. It is computed
                    // from IMU data and still contains a delta due to gravity.
                    // The orientation of this vector is in the IMU frame at the
                    // second pose.
                    let inverse_delta_rotation = result
                        .delta_rotation
                        .try_inverse()
                        .unwrap_or_else(|| nalgebra::Quaternion::new(0., 0., 0., 0.));
                    let delta_velocity: Vector3<f64> = quaternion_rotate(
                        &quaternion_mul(
                            &inverse_delta_rotation,
                            &result_to_first_center.delta_rotation,
                        ),
                        &result_center_to_center.delta_velocity,
                    );
                    let gravity_block = *gravity_constant.get_or_insert_with(|| {
                        problem.add_parameter_block(&[trajectory_data.gravity_constant])
                    });
                    problem.add_residual_block(
                        AccelerationCostFunction3D::create_auto_diff_cost_function(
                            self.options.acceleration_weight
                                / to_seconds(first_duration + second_duration),
                            &delta_velocity,
                            to_seconds(first_duration),
                            to_seconds(second_duration),
                        ),
                        None, /* loss function */
                        &[
                            c_nodes.at(&second_node_id).rotation(),
                            c_nodes.at(&first_node_id).translation(),
                            c_nodes.at(&second_node_id).translation(),
                            c_nodes.at(&third_node_id).translation(),
                            gravity_block,
                            imu_calibration,
                        ],
                    );
                }
                problem.add_residual_block(
                    RotationCostFunction3D::create_auto_diff_cost_function(
                        self.options.rotation_weight / to_seconds(first_duration),
                        &result.delta_rotation,
                    ),
                    None, /* loss function */
                    &[
                        c_nodes.at(&first_node_id).rotation(),
                        c_nodes.at(&second_node_id).rotation(),
                        imu_calibration,
                    ],
                );
            }
            if let Some(gravity_block) = gravity_constant {
                // Force gravity constant to be positive.
                problem.set_parameter_lower_bound(gravity_block, 0, 0.0);
                c_gravity_constants.insert(trajectory_id, gravity_block);
            }
        }
        (c_imu_calibrations, c_gravity_constants)
    }

    /// Adds penalties for violating odometry (if available) and changes
    /// between consecutive nodes (only used with `fix_z_in_3d`).
    fn add_odometry_and_local_slam_cost_functions(
        &self,
        frozen_trajectories: &std::collections::BTreeSet<i32>,
        c_nodes: &MapById<NodeId, CeresPose>,
        problem: &mut Problem<'_>,
    ) {
        for trajectory_id in self.node_data.trajectory_ids() {
            if frozen_trajectories.contains(&trajectory_id) {
                continue;
            }
            let mut prev_node: Option<IdDataReference<'_, NodeId, NodeSpec3D>> = None;
            for node in self.node_data.trajectory(trajectory_id) {
                let Some(first) = prev_node.replace(node) else {
                    continue;
                };
                let (first_node_id, first_node_data) = (first.id, first.data);
                let (second_node_id, second_node_data) = (node.id, node.data);

                if second_node_id.node_index != first_node_id.node_index + 1 {
                    continue;
                }
                let first_pose = c_nodes.at(&first_node_id);
                let second_pose = c_nodes.at(&second_node_id);
                let blocks = [
                    first_pose.rotation(),
                    first_pose.translation(),
                    second_pose.rotation(),
                    second_pose.translation(),
                ];

                // Add a relative pose constraint based on the odometry (if
                // available).
                if let Some(relative_odometry) = self.calculate_odometry_between_nodes(
                    trajectory_id,
                    first_node_data,
                    second_node_data,
                ) {
                    problem.add_residual_block(
                        SpaCostFunction3D::create_auto_diff_cost_function(&Pose {
                            zbar_ij: relative_odometry,
                            translation_weight: self.options.odometry_translation_weight,
                            rotation_weight: self.options.odometry_rotation_weight,
                        }),
                        None, /* loss function */
                        &blocks,
                    );
                }

                // Add a relative pose constraint based on consecutive local
                // SLAM poses.
                let relative_local_slam_pose =
                    first_node_data.local_pose.inverse() * second_node_data.local_pose;
                problem.add_residual_block(
                    SpaCostFunction3D::create_auto_diff_cost_function(&Pose {
                        zbar_ij: relative_local_slam_pose,
                        translation_weight: self.options.local_slam_pose_translation_weight,
                        rotation_weight: self.options.local_slam_pose_rotation_weight,
                    }),
                    None, /* loss function */
                    &blocks,
                );
            }
        }
    }

    /// Adds fixed frame pose constraints. Returns the fixed frame pose per
    /// trajectory.
    fn add_fixed_frame_cost_functions(
        &self,
        c_nodes: &MapById<NodeId, CeresPose>,
        problem: &mut Problem<'_>,
    ) -> BTreeMap<i32, CeresPose> {
        let mut c_fixed_frames: BTreeMap<i32, CeresPose> = BTreeMap::new();
        for trajectory_id in self.node_data.trajectory_ids() {
            if !self.fixed_frame_pose_data.has_trajectory(trajectory_id) {
                continue;
            }

            let trajectory_data = self
                .trajectory_data
                .get(&trajectory_id)
                .unwrap_or_else(|| panic!("map::at: no trajectory data for {trajectory_id}"));
            let mut fixed_frame_pose_initialized: Option<CeresPose> = None;
            for node in self.node_data.trajectory(trajectory_id) {
                let Some(fixed_frame_pose) = interpolate_fixed_frame_pose(
                    &self.fixed_frame_pose_data,
                    trajectory_id,
                    node.data.time,
                ) else {
                    continue;
                };

                let constraint_pose = Pose {
                    zbar_ij: fixed_frame_pose,
                    translation_weight: self.options.fixed_frame_pose_translation_weight,
                    rotation_weight: self.options.fixed_frame_pose_rotation_weight,
                };

                let fixed_frame = *fixed_frame_pose_initialized.get_or_insert_with(|| {
                    let fixed_frame_pose_in_map = match &trajectory_data.fixed_frame_origin_in_map {
                        Some(origin) => *origin,
                        None => node.data.global_pose * constraint_pose.zbar_ij.inverse(),
                    };
                    let pose = CeresPose::new(
                        &Rigid3d::with_angle_axis(
                            *fixed_frame_pose_in_map.translation(),
                            AngleAxis::new(
                                get_yaw_quaternion(fixed_frame_pose_in_map.rotation()),
                                Vector3::z(),
                            ),
                        ),
                        None,
                        Some(Box::new(AutoDiffLocalParameterization::<
                            YawOnlyQuaternionPlus,
                            4,
                            1,
                        >::default())),
                        problem,
                    );
                    c_fixed_frames.insert(trajectory_id, pose);
                    pose
                });

                let node_pose = c_nodes.at(&node.id);
                problem.add_residual_block(
                    SpaCostFunction3D::create_auto_diff_cost_function(&constraint_pose),
                    fixed_frame_pose_loss(&self.options),
                    &[
                        fixed_frame.rotation(),
                        fixed_frame.translation(),
                        node_pose.rotation(),
                        node_pose.translation(),
                    ],
                );
            }
        }
        c_fixed_frames
    }
}

impl OptimizationProblemInterface for OptimizationProblem3D {
    type NodeData = NodeSpec3D;
    type SubmapData = SubmapSpec3D;
    type RigidTransform = Rigid3d;

    fn add_imu_data(&mut self, trajectory_id: i32, imu_data: &ImuData) {
        self.imu_data.append(trajectory_id, *imu_data);
    }

    fn add_odometry_data(&mut self, trajectory_id: i32, odometry_data: &OdometryData) {
        self.odometry_data.append(trajectory_id, *odometry_data);
    }

    fn add_trajectory_node(&mut self, trajectory_id: i32, node_data: &NodeSpec3D) {
        self.node_data.append(trajectory_id, *node_data);
        self.trajectory_data.entry(trajectory_id).or_default();
    }

    fn insert_trajectory_node(&mut self, node_id: &NodeId, node_data: &NodeSpec3D) {
        self.node_data.insert(node_id, *node_data);
        self.trajectory_data
            .entry(node_id.trajectory_id)
            .or_default();
    }

    fn trim_trajectory_node(&mut self, node_id: &NodeId) {
        self.imu_data
            .trim(&self.node_data, node_id.trajectory_id, node_id.node_index);
        self.odometry_data
            .trim(&self.node_data, node_id.trajectory_id, node_id.node_index);
        self.fixed_frame_pose_data
            .trim(&self.node_data, node_id.trajectory_id, node_id.node_index);
        self.node_data.trim(node_id);
        if self
            .node_data
            .size_of_trajectory_or_zero(node_id.trajectory_id)
            == 0
        {
            self.trajectory_data.remove(&node_id.trajectory_id);
        }
    }

    fn add_submap(&mut self, trajectory_id: i32, global_submap_pose: &Rigid3d) {
        self.submap_data.append(
            trajectory_id,
            SubmapSpec3D {
                global_pose: *global_submap_pose,
            },
        );
    }

    fn insert_submap(&mut self, submap_id: &SubmapId, global_submap_pose: &Rigid3d) {
        self.submap_data.insert(
            submap_id,
            SubmapSpec3D {
                global_pose: *global_submap_pose,
            },
        );
    }

    fn trim_submap(&mut self, submap_id: &SubmapId) {
        self.submap_data.trim(submap_id);
    }

    fn set_max_num_iterations(&mut self, max_num_iterations: i32) {
        set_max_num_iterations(&mut self.options, max_num_iterations);
    }

    fn solve(
        &mut self,
        constraints: &[Constraint],
        trajectories_state: &BTreeMap<i32, TrajectoryState>,
        landmark_nodes: &BTreeMap<String, LandmarkNode>,
    ) {
        if self.node_data.empty() {
            // Nothing to optimize.
            return;
        }

        let frozen_trajectories = frozen_trajectories(trajectories_state);

        let mut problem = Problem::new();

        // Set the starting point.
        assert!(
            !self.submap_data.empty(),
            "Check failed: !submap_data_.empty()"
        );
        let mut c_submaps: MapById<SubmapId, CeresPose> = MapById::new();
        let mut c_nodes: MapById<NodeId, CeresPose> = MapById::new();
        let mut c_landmarks: BTreeMap<String, CeresPose> = BTreeMap::new();
        let mut first_submap = true;
        for submap_id_data in &self.submap_data {
            let frozen = frozen_trajectories.contains(&submap_id_data.id.trajectory_id);
            let pose = if first_submap {
                first_submap = false;
                // Fix the first submap of the first trajectory except for
                // allowing gravity alignment.
                let pose = CeresPose::new(
                    &submap_id_data.data.global_pose,
                    self.translation_parameterization(),
                    Some(Box::new(AutoDiffLocalParameterization::<
                        ConstantYawQuaternionPlus,
                        4,
                        2,
                    >::default())),
                    &mut problem,
                );
                problem.set_parameter_block_constant(pose.translation());
                pose
            } else {
                CeresPose::new(
                    &submap_id_data.data.global_pose,
                    self.translation_parameterization(),
                    Some(Box::new(QuaternionParameterization)),
                    &mut problem,
                )
            };
            if frozen {
                problem.set_parameter_block_constant(pose.rotation());
                problem.set_parameter_block_constant(pose.translation());
            }
            c_submaps.insert(&submap_id_data.id, pose);
        }
        for node_id_data in &self.node_data {
            let frozen = frozen_trajectories.contains(&node_id_data.id.trajectory_id);
            let pose = CeresPose::new(
                &node_id_data.data.global_pose,
                self.translation_parameterization(),
                Some(Box::new(QuaternionParameterization)),
                &mut problem,
            );
            if frozen {
                problem.set_parameter_block_constant(pose.rotation());
                problem.set_parameter_block_constant(pose.translation());
            }
            c_nodes.insert(&node_id_data.id, pose);
        }
        // Add cost functions for intra- and inter-submap constraints.
        for constraint in constraints {
            let submap_pose = c_submaps.at(&constraint.submap_id);
            let node_pose = c_nodes.at(&constraint.node_id);
            problem.add_residual_block(
                SpaCostFunction3D::create_auto_diff_cost_function(&constraint.pose),
                // Loop closure constraints should have a loss function.
                constraint_loss(constraint.tag, self.options.huber_scale),
                &[
                    submap_pose.rotation(),
                    submap_pose.translation(),
                    node_pose.rotation(),
                    node_pose.translation(),
                ],
            );
        }
        // Add cost functions for landmarks.
        add_landmark_cost_functions(
            landmark_nodes,
            &self.node_data,
            &c_nodes,
            &mut c_landmarks,
            &mut problem,
            self.options.huber_scale,
        );
        // Add constraints based on IMU observations of angular velocities and
        // linear acceleration.
        let (c_imu_calibrations, c_gravity_constants) = if !self.options.fix_z_in_3d {
            self.add_imu_cost_functions(&frozen_trajectories, &c_nodes, &mut problem)
        } else {
            (BTreeMap::new(), BTreeMap::new())
        };

        if self.options.fix_z_in_3d {
            // Add penalties for violating odometry (if available) and changes
            // between consecutive nodes.
            self.add_odometry_and_local_slam_cost_functions(
                &frozen_trajectories,
                &c_nodes,
                &mut problem,
            );
        }

        // Add fixed frame pose constraints.
        let c_fixed_frames = self.add_fixed_frame_cost_functions(&c_nodes, &mut problem);

        // Solve.
        let mut summary = ceres::SolverSummary::default();
        ceres::solve(
            &ceres_solver_options(&self.options),
            &mut problem,
            &mut summary,
        );

        // RUST PORT NOTE: in C++ Ceres updated these in place.
        for (trajectory_id, block) in &c_imu_calibrations {
            let values = problem.parameter_block(*block);
            self.trajectory_data
                .get_mut(trajectory_id)
                .expect("trajectory data of an IMU calibration")
                .imu_calibration
                .copy_from_slice(values);
        }
        for (trajectory_id, block) in &c_gravity_constants {
            self.trajectory_data
                .get_mut(trajectory_id)
                .expect("trajectory data of a gravity constant")
                .gravity_constant = problem.parameter_block(*block)[0];
        }

        if self.options.log_solver_summary {
            log::info!("{}", summary.full_report());
            for (trajectory_id, trajectory_data) in &self.trajectory_data {
                if *trajectory_id != 0 {
                    log::info!("Trajectory {trajectory_id}:");
                }
                log::info!("Gravity was: {}", trajectory_data.gravity_constant);
                let imu_calibration = &trajectory_data.imu_calibration;
                log::info!(
                    "IMU correction was: {} deg ({}, {}, {}, {})",
                    rad_to_deg(2. * imu_calibration[0].abs().acos()),
                    imu_calibration[0],
                    imu_calibration[1],
                    imu_calibration[2],
                    imu_calibration[3]
                );
            }
        }

        // Store the result.
        for c_submap_id_data in &c_submaps {
            self.submap_data.at_mut(&c_submap_id_data.id).global_pose =
                c_submap_id_data.data.to_rigid(&problem);
        }
        for c_node_id_data in &c_nodes {
            self.node_data.at_mut(&c_node_id_data.id).global_pose =
                c_node_id_data.data.to_rigid(&problem);
        }
        for (trajectory_id, pose) in &c_fixed_frames {
            self.trajectory_data
                .get_mut(trajectory_id)
                .expect("trajectory data of a fixed frame")
                .fixed_frame_origin_in_map = Some(pose.to_rigid(&problem));
        }
        for (landmark_id, pose) in &c_landmarks {
            self.landmark_data
                .insert(landmark_id.clone(), pose.to_rigid(&problem));
        }
    }

    fn node_data(&self) -> &MapById<NodeId, NodeSpec3D> {
        &self.node_data
    }

    fn submap_data(&self) -> &MapById<SubmapId, SubmapSpec3D> {
        &self.submap_data
    }

    fn landmark_data(&self) -> &BTreeMap<String, Rigid3d> {
        &self.landmark_data
    }

    fn imu_data(&self) -> &MapByTime<ImuData> {
        &self.imu_data
    }

    fn odometry_data(&self) -> &MapByTime<OdometryData> {
        &self.odometry_data
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use nalgebra::Quaternion;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::common::time::{from_seconds, from_universal};
    use crate::mapping::internal::optimization::optimization_problem_options::create_optimization_problem_options;
    use crate::mapping::pose_graph_interface::Tag;
    use crate::transform::rigid_transform::quaternion_normalized;
    use crate::transform::transform::{angle_axis_vector_to_rotation_quaternion, get_angle};

    fn create_options_with(
        fix_z_in_3d: bool,
        max_num_iterations: i32,
    ) -> proto::OptimizationProblemOptions {
        let dictionary = make_dictionary(&format!(
            r#"
            return {{
              acceleration_weight = 1.1e2,
              rotation_weight = 1.6e4,
              huber_scale = 1.,
              local_slam_pose_translation_weight = 1e5,
              local_slam_pose_rotation_weight = 1e5,
              odometry_translation_weight = 1e-2,
              odometry_rotation_weight = 1e-2,
              fixed_frame_pose_translation_weight = 1e1,
              fixed_frame_pose_rotation_weight = 1e2,
              fixed_frame_pose_use_tolerant_loss = false,
              fixed_frame_pose_tolerant_loss_param_a = 1,
              fixed_frame_pose_tolerant_loss_param_b = 1,
              log_solver_summary = true,
              use_online_imu_extrinsics_in_3d = true,
              fix_z_in_3d = {fix_z_in_3d},
              ceres_solver_options = {{
                use_nonmonotonic_steps = false,
                max_num_iterations = {max_num_iterations},
                num_threads = 4,
              }},
            }}"#
        ));
        create_optimization_problem_options(&dictionary)
    }

    /// Fixture of the C++ `OptimizationProblem3DTest`.
    ///
    /// RUST PORT NOTE: `std::mt19937` + `std::uniform_real_distribution`
    /// become a seeded `StdRng` (same seed, different stream).
    struct OptimizationProblem3DTest {
        optimization_problem: OptimizationProblem3D,
        rng: StdRng,
    }

    impl OptimizationProblem3DTest {
        fn new() -> Self {
            OptimizationProblem3DTest {
                optimization_problem: OptimizationProblem3D::new(&Self::create_options()),
                rng: StdRng::seed_from_u64(45387),
            }
        }

        fn create_options() -> proto::OptimizationProblemOptions {
            let dictionary = make_dictionary(
                r#"
                return {
                  acceleration_weight = 2e-5,
                  rotation_weight = 1e-3,
                  huber_scale = 1.,
                  local_slam_pose_translation_weight = 1e-2,
                  local_slam_pose_rotation_weight = 1e-2,
                  odometry_translation_weight = 1e-2,
                  odometry_rotation_weight = 1e-2,
                  fixed_frame_pose_translation_weight = 1e1,
                  fixed_frame_pose_rotation_weight = 1e2,
                  fixed_frame_pose_use_tolerant_loss = false,
                  fixed_frame_pose_tolerant_loss_param_a = 1,
                  fixed_frame_pose_tolerant_loss_param_b = 1,
                  log_solver_summary = true,
                  use_online_imu_extrinsics_in_3d = true,
                  fix_z_in_3d = false,
                  ceres_solver_options = {
                    use_nonmonotonic_steps = false,
                    max_num_iterations = 200,
                    num_threads = 4,
                  },
                }"#,
            );
            create_optimization_problem_options(&dictionary)
        }

        fn random_transform(&mut self, translation_size: f64, rotation_size: f64) -> Rigid3d {
            let x = self.rng.random_range(-translation_size..translation_size);
            let y = self.rng.random_range(-translation_size..translation_size);
            let z = self.rng.random_range(-translation_size..translation_size);
            let rx = self.rng.random_range(-rotation_size..rotation_size);
            let ry = self.rng.random_range(-rotation_size..rotation_size);
            let rz = self.rng.random_range(-rotation_size..rotation_size);
            Rigid3d::new(
                Vector3::new(x, y, z),
                angle_axis_vector_to_rotation_quaternion(&Vector3::new(rx, ry, rz)),
            )
        }

        fn random_yaw_only_transform(
            &mut self,
            translation_size: f64,
            rotation_size: f64,
        ) -> Rigid3d {
            let x = self.rng.random_range(-translation_size..translation_size);
            let y = self.rng.random_range(-translation_size..translation_size);
            let z = self.rng.random_range(-translation_size..translation_size);
            let rz = self.rng.random_range(-rotation_size..rotation_size);
            Rigid3d::new(
                Vector3::new(x, y, z),
                angle_axis_vector_to_rotation_quaternion(&Vector3::new(0., 0., rz)),
            )
        }
    }

    fn add_noise(transform: &Rigid3d, noise: &Rigid3d) -> Rigid3d {
        let noisy_rotation = quaternion_mul(noise.rotation(), transform.rotation());
        Rigid3d::new(
            transform.translation() + noise.translation(),
            noisy_rotation,
        )
    }

    fn constraint(
        submap_id: SubmapId,
        node_id: NodeId,
        zbar_ij: Rigid3d,
        weight: f64,
    ) -> Constraint {
        Constraint {
            submap_id,
            node_id,
            pose: Pose {
                zbar_ij,
                translation_weight: weight,
                rotation_weight: weight,
            },
            // C++ value-initializes the omitted tag to INTRA_SUBMAP.
            tag: Tag::IntraSubmap,
        }
    }

    fn rz(angle: f64) -> Quaternion<f64> {
        AngleAxis::new(angle, Vector3::z()).to_quaternion()
    }

    fn active(trajectory_ids: &[i32]) -> BTreeMap<i32, TrajectoryState> {
        trajectory_ids
            .iter()
            .map(|&id| (id, TrajectoryState::Active))
            .collect()
    }

    fn node(time: Time, local_pose: Rigid3d, global_pose: Rigid3d) -> NodeSpec3D {
        NodeSpec3D {
            time,
            local_pose,
            global_pose,
        }
    }

    mod optimization_problem_3d_test {
        use super::*;

        #[test]
        fn reduces_noise() {
            let mut fixture = OptimizationProblem3DTest::new();
            const NUM_NODES: i32 = 100;
            let submap0_transform = Rigid3d::identity();
            let submap2_transform = Rigid3d::from_rotation(rz(PI));
            const TRAJECTORY_ID: i32 = 0;

            struct NoisyNode {
                ground_truth_pose: Rigid3d,
                noise: Rigid3d,
            }
            let mut test_data = Vec::new();
            for _ in 0..NUM_NODES {
                test_data.push(NoisyNode {
                    ground_truth_pose: fixture.random_transform(10., 3.),
                    noise: fixture.random_yaw_only_transform(0.2, 0.3),
                });
            }

            let mut now = from_universal(0);
            for node in &test_data {
                let pose = add_noise(&node.ground_truth_pose, &node.noise);
                fixture.optimization_problem.add_imu_data(
                    TRAJECTORY_ID,
                    &ImuData {
                        time: now,
                        linear_acceleration: Vector3::z() * 9.81,
                        angular_velocity: Vector3::zeros(),
                    },
                );
                fixture.optimization_problem.add_trajectory_node(
                    TRAJECTORY_ID,
                    &NodeSpec3D {
                        time: now,
                        local_pose: pose,
                        global_pose: pose,
                    },
                );
                now += from_seconds(0.1);
            }

            let mut constraints = Vec::new();
            for j in 0..NUM_NODES {
                let ground_truth_pose = test_data[j as usize].ground_truth_pose;
                let noise = test_data[j as usize].noise;
                constraints.push(constraint(
                    SubmapId::new(TRAJECTORY_ID, 0),
                    NodeId::new(TRAJECTORY_ID, j),
                    add_noise(&ground_truth_pose, &noise),
                    1.,
                ));
                // We add an additional independent, but equally noisy
                // observation.
                let other_noise = fixture.random_yaw_only_transform(0.2, 0.3);
                constraints.push(constraint(
                    SubmapId::new(TRAJECTORY_ID, 1),
                    NodeId::new(TRAJECTORY_ID, j),
                    add_noise(&ground_truth_pose, &other_noise),
                    1.,
                ));
                // We add very noisy data with a low weight to verify it is
                // mostly ignored.
                let very_noisy = fixture.random_transform(1e3, 3.);
                constraints.push(constraint(
                    SubmapId::new(TRAJECTORY_ID, 2),
                    NodeId::new(TRAJECTORY_ID, j),
                    submap2_transform.inverse() * ground_truth_pose * very_noisy,
                    1e-9,
                ));
            }

            let errors = |problem: &OptimizationProblem3D| {
                let node_data = problem.node_data();
                let mut translation_error = 0.;
                let mut rotation_error = 0.;
                for (j, data) in test_data.iter().enumerate() {
                    let ground_truth = &data.ground_truth_pose;
                    let pose = &node_data
                        .at(&NodeId::new(TRAJECTORY_ID, j as i32))
                        .global_pose;
                    translation_error += (ground_truth.translation() - pose.translation()).norm();
                    rotation_error += get_angle(&(ground_truth.inverse() * *pose));
                }
                (translation_error, rotation_error)
            };
            let (translation_error_before, rotation_error_before) =
                errors(&fixture.optimization_problem);

            fixture
                .optimization_problem
                .add_submap(TRAJECTORY_ID, &submap0_transform);
            fixture
                .optimization_problem
                .add_submap(TRAJECTORY_ID, &submap0_transform);
            fixture
                .optimization_problem
                .add_submap(TRAJECTORY_ID, &submap2_transform);
            fixture.optimization_problem.solve(
                &constraints,
                &active(&[TRAJECTORY_ID]),
                &BTreeMap::new(),
            );

            let (translation_error_after, rotation_error_after) =
                errors(&fixture.optimization_problem);

            assert!(
                0.8 * translation_error_before > translation_error_after,
                "{translation_error_before} vs. {translation_error_after}"
            );
            assert!(
                0.8 * rotation_error_before > rotation_error_after,
                "{rotation_error_before} vs. {rotation_error_after}"
            );
        }
    }

    /// Ground truth of a trajectory rotating with constant yaw rate while
    /// moving with constant velocity along x.
    fn ground_truth_at(time_seconds: f64) -> Rigid3d {
        Rigid3d::new(
            Vector3::new(1.5 * time_seconds, 0., 0.),
            rz(0.5 * time_seconds),
        )
    }

    /// Adds `num_nodes` nodes (0.1 s apart) with noisy global poses and
    /// matching IMU data to trajectory `trajectory_id`, returning the ground
    /// truth poses.
    fn add_imu_trajectory(
        problem: &mut OptimizationProblem3D,
        trajectory_id: i32,
        num_nodes: i32,
        rng: &mut StdRng,
    ) -> Vec<Rigid3d> {
        let mut ground_truth = Vec::new();
        for i in 0..num_nodes {
            let time_seconds = 0.1 * f64::from(i);
            let time = from_universal(0) + from_seconds(time_seconds);
            let pose = ground_truth_at(time_seconds);
            // Gravity only (constant velocity), yaw rate 0.5 rad/s.
            problem.add_imu_data(
                trajectory_id,
                &ImuData {
                    time,
                    linear_acceleration: Vector3::new(0., 0., 9.8),
                    angular_velocity: Vector3::new(0., 0., 0.5),
                },
            );
            let noise = Rigid3d::new(
                Vector3::new(
                    rng.random_range(-0.2..0.2),
                    rng.random_range(-0.2..0.2),
                    rng.random_range(-0.2..0.2),
                ),
                angle_axis_vector_to_rotation_quaternion(&Vector3::new(
                    rng.random_range(-0.1..0.1),
                    rng.random_range(-0.1..0.1),
                    rng.random_range(-0.1..0.1),
                )),
            );
            problem.add_trajectory_node(trajectory_id, &node(time, pose, add_noise(&pose, &noise)));
            ground_truth.push(pose);
        }
        ground_truth
    }

    fn sparse_constraints(trajectory_id: i32, ground_truth: &[Rigid3d]) -> Vec<Constraint> {
        (0..ground_truth.len())
            .step_by(5)
            .map(|j| {
                constraint(
                    SubmapId::new(trajectory_id, 0),
                    NodeId::new(trajectory_id, j as i32),
                    ground_truth[j],
                    1e3,
                )
            })
            .collect()
    }

    #[test]
    fn imu_trajectory_converges_to_ground_truth() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut problem = OptimizationProblem3D::new(&create_options_with(false, 200));
        let ground_truth = add_imu_trajectory(&mut problem, 0, 31, &mut rng);
        problem.add_submap(0, &Rigid3d::identity());
        // Every fifth node is observed exactly from the submap; the IMU terms
        // determine the poses in between.
        let constraints = sparse_constraints(0, &ground_truth);
        let initial_error = ground_truth
            .iter()
            .enumerate()
            .map(|(j, gt)| {
                (problem
                    .node_data()
                    .at(&NodeId::new(0, j as i32))
                    .global_pose
                    .translation()
                    - gt.translation())
                .norm()
            })
            .fold(0., f64::max);
        assert!(initial_error > 0.1, "{initial_error}");

        problem.solve(&constraints, &active(&[0]), &BTreeMap::new());

        for (j, gt) in ground_truth.iter().enumerate() {
            let pose = problem
                .node_data()
                .at(&NodeId::new(0, j as i32))
                .global_pose;
            assert!(
                (pose.translation() - gt.translation()).norm() < 1e-4,
                "node {j}: {pose:?} vs. {gt:?}"
            );
            assert!(
                get_angle(&(gt.inverse() * pose)) < 1e-4,
                "node {j}: {pose:?} vs. {gt:?}"
            );
        }
        // The first submap's translation is constant, its yaw cannot change.
        let submap = problem.submap_data().at(&SubmapId::new(0, 0)).global_pose;
        assert_eq!(*submap.translation(), Vector3::zeros());
        assert!(get_angle(&submap) < 1e-4, "{submap:?}");
        let trajectory_data = problem.trajectory_data()[&0];
        assert!(
            (trajectory_data.gravity_constant - 9.8).abs() < 1e-3,
            "{}",
            trajectory_data.gravity_constant
        );
        let [w, x, y, z] = trajectory_data.imu_calibration;
        let imu_calibration = quaternion_normalized(&Quaternion::new(w, x, y, z));
        // With rotation and acceleration only about/along z, a yaw rotation
        // of the IMU calibration is unobservable; roll and pitch are not.
        assert!(
            imu_calibration.i.abs() < 1e-6 && imu_calibration.j.abs() < 1e-6,
            "{imu_calibration:?}"
        );
    }

    #[test]
    fn frozen_trajectory_is_unchanged() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut problem = OptimizationProblem3D::new(&create_options_with(false, 50));
        let ground_truth = add_imu_trajectory(&mut problem, 0, 11, &mut rng);
        problem.add_submap(0, &Rigid3d::identity());
        // Trajectory 1 is frozen and has no IMU data (frozen trajectories get
        // no IMU terms).
        let frozen_submap = Rigid3d::new(Vector3::new(5., 1., 0.3), rz(0.4));
        problem.add_submap(1, &frozen_submap);
        let frozen_nodes: Vec<Rigid3d> = (0..5)
            .map(|i| Rigid3d::new(Vector3::new(5. + f64::from(i), 1., 0.3), rz(0.1)))
            .collect();
        for (i, pose) in frozen_nodes.iter().enumerate() {
            let time = from_universal(0) + from_seconds(0.1 * i as f64);
            problem.add_trajectory_node(1, &node(time, *pose, *pose));
        }
        let mut constraints = sparse_constraints(0, &ground_truth);
        // Inconsistent constraints that would move the frozen trajectory.
        for i in 0..5 {
            let mut c = constraint(
                SubmapId::new(0, 0),
                NodeId::new(1, i),
                Rigid3d::from_translation(Vector3::new(f64::from(i), -3., 2.)),
                1e3,
            );
            c.tag = Tag::InterSubmap;
            constraints.push(c);
            constraints.push(constraint(
                SubmapId::new(1, 0),
                NodeId::new(0, 2 * i),
                Rigid3d::identity(),
                1.,
            ));
        }
        let mut states = active(&[0]);
        states.insert(1, TrajectoryState::Frozen);
        let unfrozen_before = problem.node_data().at(&NodeId::new(0, 1)).global_pose;
        problem.solve(&constraints, &states, &BTreeMap::new());

        assert_eq!(
            problem.submap_data().at(&SubmapId::new(1, 0)).global_pose,
            frozen_submap
        );
        for (i, pose) in frozen_nodes.iter().enumerate() {
            assert_eq!(
                problem
                    .node_data()
                    .at(&NodeId::new(1, i as i32))
                    .global_pose,
                *pose
            );
        }
        assert_ne!(
            problem.node_data().at(&NodeId::new(0, 1)).global_pose,
            unfrozen_before
        );
        // The frozen trajectory had no IMU parameter blocks; its data keeps
        // the defaults.
        assert_eq!(problem.trajectory_data()[&1], TrajectoryData::default());
    }

    #[test]
    fn fix_z_in_3d_keeps_z_and_needs_no_imu() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 50));
        problem.add_submap(0, &Rigid3d::identity());
        let ground_truth: Vec<Rigid3d> = (0..10)
            .map(|i| ground_truth_at(0.1 * f64::from(i)))
            .collect();
        let initial_z: Vec<f64> = (0..10).map(|i| 0.01 * f64::from(i)).collect();
        for (i, gt) in ground_truth.iter().enumerate() {
            let initial = Vector3::new(
                gt.translation().x + 0.1,
                gt.translation().y - 0.1,
                initial_z[i],
            );
            let time = from_universal(0) + from_seconds(0.1 * i as f64);
            problem.add_trajectory_node(0, &node(time, *gt, Rigid3d::new(initial, *gt.rotation())));
        }
        let constraints = vec![constraint(
            SubmapId::new(0, 0),
            NodeId::new(0, 0),
            ground_truth[0],
            1e3,
        )];
        // No IMU data: with fix_z_in_3d only local SLAM (and odometry) terms
        // are added instead of the IMU terms.
        let xy_error = |problem: &OptimizationProblem3D| {
            ground_truth
                .iter()
                .enumerate()
                .map(|(i, gt)| {
                    let pose = problem
                        .node_data()
                        .at(&NodeId::new(0, i as i32))
                        .global_pose;
                    (pose.translation().xy() - gt.translation().xy()).norm()
                })
                .fold(0., f64::max)
        };
        let xy_error_before = xy_error(&problem);
        problem.solve(&constraints, &active(&[0]), &BTreeMap::new());
        // The z coordinates are held constant (even though they are
        // inconsistent with the local SLAM poses, which pulls in roll/pitch).
        for (i, z) in initial_z.iter().enumerate() {
            let pose = problem
                .node_data()
                .at(&NodeId::new(0, i as i32))
                .global_pose;
            assert_eq!(pose.translation().z, *z);
        }
        let xy_error_after = xy_error(&problem);
        assert!(
            xy_error_after < 0.1 * xy_error_before,
            "{xy_error_before} vs. {xy_error_after}"
        );
    }

    #[test]
    #[should_panic(expected = "needs IMU data")]
    fn missing_imu_data_panics_without_fix_z() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(false, 5));
        problem.add_submap(0, &Rigid3d::identity());
        for i in 0..3 {
            problem.add_trajectory_node(
                0,
                &node(from_universal(i), Rigid3d::identity(), Rigid3d::identity()),
            );
        }
        problem.solve(&[], &active(&[0]), &BTreeMap::new());
    }

    #[test]
    #[should_panic(expected = "submap_data_.empty()")]
    fn solve_without_submaps_panics() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 5));
        problem.add_trajectory_node(
            0,
            &node(from_universal(0), Rigid3d::identity(), Rigid3d::identity()),
        );
        problem.solve(&[], &active(&[0]), &BTreeMap::new());
    }

    #[test]
    fn solve_without_nodes_is_a_no_op() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(false, 5));
        problem.add_submap(0, &Rigid3d::identity());
        problem.solve(&[], &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(problem.submap_data().size(), 1);
    }

    #[test]
    fn landmark_is_estimated() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 100));
        problem.add_submap(0, &Rigid3d::identity());
        let ground_truth: Vec<Rigid3d> = (0..4)
            .map(|i| Rigid3d::from_translation(Vector3::new(f64::from(i), 0., 0.)))
            .collect();
        for (i, gt) in ground_truth.iter().enumerate() {
            let time = from_universal(0) + from_seconds(i as f64);
            problem.add_trajectory_node(0, &node(time, *gt, *gt));
        }
        let landmark = Rigid3d::new(Vector3::new(1.5, 2., 0.), rz(0.3));
        // The interpolated tracking pose at t = 1.5 s.
        let tracking = Rigid3d::from_translation(Vector3::new(1.5, 0., 0.));
        let mut landmark_nodes = BTreeMap::new();
        landmark_nodes.insert(
            "landmark".to_string(),
            LandmarkNode {
                landmark_observations: vec![LandmarkObservation {
                    trajectory_id: 0,
                    time: from_universal(0) + from_seconds(1.5),
                    landmark_to_tracking_transform: tracking.inverse() * landmark,
                    translation_weight: 1e3,
                    rotation_weight: 1e3,
                }],
                global_landmark_pose: None,
                frozen: false,
            },
        );
        let constraints: Vec<Constraint> = ground_truth
            .iter()
            .enumerate()
            .map(|(j, gt)| constraint(SubmapId::new(0, 0), NodeId::new(0, j as i32), *gt, 1e3))
            .collect();
        problem.solve(&constraints, &active(&[0]), &landmark_nodes);
        let estimated = problem.landmark_data()["landmark"];
        assert!(
            (estimated.translation() - landmark.translation()).norm() < 1e-6,
            "{estimated:?}"
        );
        assert!(
            get_angle(&(landmark.inverse() * estimated)) < 1e-6,
            "{estimated:?}"
        );
    }

    #[test]
    fn fixed_frame_origin_is_estimated() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 100));
        problem.add_submap(0, &Rigid3d::identity());
        // The fixed frame (e.g. GPS) origin in the map frame.
        let fixed_frame_origin_in_map = Rigid3d::new(Vector3::new(10., -4., 0.), rz(0.7));
        for i in 0..5 {
            let gt = Rigid3d::new(
                Vector3::new(f64::from(i), 0.5 * f64::from(i), 0.),
                rz(0.1 * f64::from(i)),
            );
            let time = from_universal(0) + from_seconds(f64::from(i));
            problem.add_trajectory_node(0, &node(time, gt, gt));
            problem.add_fixed_frame_pose_data(
                0,
                &FixedFramePoseData {
                    time,
                    pose: Some(fixed_frame_origin_in_map.inverse() * gt),
                },
            );
        }
        problem.solve(&[], &active(&[0]), &BTreeMap::new());
        let origin = problem.trajectory_data()[&0]
            .fixed_frame_origin_in_map
            .expect("fixed frame origin");
        assert!(
            (origin.translation() - fixed_frame_origin_in_map.translation()).norm() < 1e-6,
            "{origin:?}"
        );
        assert!(
            get_angle(&(fixed_frame_origin_in_map.inverse() * origin)) < 1e-6,
            "{origin:?}"
        );
    }

    #[test]
    fn set_max_num_iterations_updates_options() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 5));
        problem.set_max_num_iterations(123);
        assert_eq!(
            problem
                .options
                .ceres_solver_options
                .unwrap()
                .max_num_iterations,
            123
        );
    }

    #[test]
    fn trim_trajectory_node_removes_trajectory_data_of_empty_trajectory() {
        let mut problem = OptimizationProblem3D::new(&create_options_with(true, 5));
        for i in 0..2 {
            problem.add_imu_data(
                0,
                &ImuData {
                    time: from_universal(i),
                    ..Default::default()
                },
            );
            problem.add_trajectory_node(
                0,
                &node(from_universal(i), Rigid3d::identity(), Rigid3d::identity()),
            );
        }
        problem.trim_trajectory_node(&NodeId::new(0, 0));
        assert!(problem.trajectory_data().contains_key(&0));
        problem.trim_trajectory_node(&NodeId::new(0, 1));
        assert!(!problem.trajectory_data().contains_key(&0));
        assert!(problem.node_data().empty());
    }
}
