//! Port of `cartographer/mapping/internal/optimization/optimization_problem_2d` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ code keeps the Ceres parameters in
//! `MapById<NodeId, std::array<double, 3>>` containers whose memory Ceres
//! optimizes in place. The Rust `ceres::Problem` owns the parameter values,
//! so `solve` keeps `MapById<_, ParameterBlockId>` handles instead and reads
//! the optimized values back from the problem when storing the result.
//! Parameter blocks that C++ adds implicitly through `AddResidualBlock`
//! (fixed frame poses) are added explicitly, as the handle-based API
//! requires.

use std::collections::{BTreeMap, BTreeSet};

use ceres::{HuberLoss, LossFunction, ParameterBlockId, Problem, QuaternionParameterization};
use nalgebra::{Quaternion, Vector2};

use crate::common::internal::ceres_solver_options::create_ceres_solver_options;
use crate::common::time::{Time, to_seconds};
use crate::mapping::id::{IdDataReference, MapById, NodeId, SubmapId};
use crate::mapping::internal::optimization::ceres_pose::CeresPose;
use crate::mapping::internal::optimization::cost_functions::cost_helpers::interpolate_nodes_2d;
use crate::mapping::internal::optimization::cost_functions::landmark_cost_function_2d::LandmarkCostFunction2D;
use crate::mapping::internal::optimization::cost_functions::spa_cost_function_2d::create_auto_diff_spa_cost_function;
use crate::mapping::internal::optimization::optimization_problem_interface::OptimizationProblemInterface;
use crate::mapping::optimization::proto;
use crate::mapping::pose_graph_interface::{
    Constraint, LandmarkNode, LandmarkObservation, Pose, Tag, TrajectoryData, TrajectoryState,
};
use crate::sensor::data::HasTime;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::map_by_time::MapByTime;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{Rigid2d, Rigid3d};
use crate::transform::timestamped_transform::{TimestampedTransform, interpolate};
use crate::transform::transform::{embed_3d, project_2d};

/// Per-node data of the 2D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec2D {
    /// Node time.
    pub time: Time,
    /// Pose in the local (non-loop-closed) frame.
    pub local_pose_2d: Rigid2d,
    /// Pose in the global (loop-closed) frame.
    pub global_pose_2d: Rigid2d,
    /// Rotation aligning the tracking frame with gravity.
    pub gravity_alignment: Quaternion<f64>,
}

impl HasTime for NodeSpec2D {
    fn time(&self) -> Time {
        self.time
    }
}

/// Per-submap data of the 2D optimization problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmapSpec2D {
    /// Global pose of the submap.
    pub global_pose: Rigid2d,
}

/// Interpolates the odometry of `trajectory_id` at `time` (C++
/// `InterpolateOdometry` / the 3D odometry `Interpolate`). The trajectory must
/// exist in `map_by_time`.
pub(crate) fn interpolate_odometry(
    map_by_time: &MapByTime<OdometryData>,
    trajectory_id: i32,
    time: Time,
) -> Option<Rigid3d> {
    let it = map_by_time.lower_bound(trajectory_id, time).next()?;
    let Some(prev) = map_by_time.before(trajectory_id, time).next_back() else {
        // `it` is the beginning of the trajectory.
        return (it.time == time).then_some(it.pose);
    };
    Some(
        interpolate(
            &TimestampedTransform {
                time: prev.time,
                transform: prev.pose,
            },
            &TimestampedTransform {
                time: it.time,
                transform: it.pose,
            },
            time,
        )
        .transform,
    )
}

/// Interpolates the fixed frame poses of `trajectory_id` at `time` (C++
/// `Interpolate` for fixed frame poses). Returns `None` if a needed sample
/// has no pose. The trajectory must exist in `map_by_time`.
pub(crate) fn interpolate_fixed_frame_pose(
    map_by_time: &MapByTime<FixedFramePoseData>,
    trajectory_id: i32,
    time: Time,
) -> Option<Rigid3d> {
    let it = map_by_time.lower_bound(trajectory_id, time).next()?;
    let it_pose = it.pose?;
    let Some(prev) = map_by_time.before(trajectory_id, time).next_back() else {
        // `it` is the beginning of the trajectory.
        return (it.time == time).then_some(it_pose);
    };
    let prev_pose = prev.pose?;
    Some(
        interpolate(
            &TimestampedTransform {
                time: prev.time,
                transform: prev_pose,
            },
            &TimestampedTransform {
                time: it.time,
                transform: it_pose,
            },
            time,
        )
        .transform,
    )
}

/// Finds the trajectory nodes before and after a landmark observation (the
/// shared part of the C++ 2D/3D `AddLandmarkCostFunctions`). Returns `None`
/// if the observation must be skipped.
///
/// RUST PORT NOTE: C++ dereferences `BeginOfTrajectory(trajectory_id)`
/// without checking that it is valid, which is undefined behaviour if
/// neither the observing trajectory nor any later trajectory has nodes; and
/// for a trajectory with exactly one node it advances `next` past the
/// trajectory's only node, dereferencing the end iterator if there is no
/// later trajectory. Both cases panic here with a clear message. The
/// well-defined (if questionable) C++ behaviour of reading nodes of the
/// *next* trajectory in these situations is replicated.
pub(crate) fn find_landmark_observation_nodes<'a, D: HasTime>(
    node_data: &'a MapById<NodeId, D>,
    observation: &LandmarkObservation,
    landmark_id: &str,
) -> Option<(
    IdDataReference<'a, NodeId, D>,
    IdDataReference<'a, NodeId, D>,
)> {
    let begin_of_trajectory = node_data.begin_of_trajectory(observation.trajectory_id);
    let begin_item = begin_of_trajectory.get().unwrap_or_else(|| {
        panic!(
            "Landmark '{landmark_id}' was observed by trajectory {} which (like all later \
             trajectories) has no nodes (undefined behaviour in C++).",
            observation.trajectory_id
        )
    });
    // The landmark observation was made before the trajectory was created.
    if observation.time < begin_item.data.time() {
        return None;
    }
    // Find the trajectory nodes before and after the landmark observation.
    let mut next = node_data.lower_bound(observation.trajectory_id, observation.time);
    // The landmark observation was made, but the next trajectory node has
    // not been added yet.
    if next == node_data.end_of_trajectory(observation.trajectory_id) {
        return None;
    }
    if next == begin_of_trajectory {
        next.increment();
    }
    let next_item = next.get().unwrap_or_else(|| {
        panic!(
            "Landmark '{landmark_id}' was observed by trajectory {} which has a single node and \
             there is no next node to interpolate with (undefined behaviour in C++).",
            observation.trajectory_id
        )
    });
    let prev_item = next.to_prev().item();
    Some((prev_item, next_item))
}

/// Converts a pose into the 3 optimization variable format used for Ceres:
/// translation in x and y, followed by the rotation angle representing the
/// orientation.
fn from_pose(pose: &Rigid2d) -> [f64; 3] {
    [
        pose.translation().x,
        pose.translation().y,
        pose.normalized_angle(),
    ]
}

/// Converts a pose as represented for Ceres back to a [`Rigid2d`] pose.
fn to_pose(values: &[f64]) -> Rigid2d {
    Rigid2d::new(Vector2::new(values[0], values[1]), values[2])
}

/// Selects a trajectory node closest in time to the landmark observation and
/// applies a relative transform from it.
fn get_initial_landmark_pose(
    observation: &LandmarkObservation,
    prev_node: &NodeSpec2D,
    next_node: &NodeSpec2D,
    prev_node_pose: &[f64],
    next_node_pose: &[f64],
) -> Rigid3d {
    let interpolation_parameter =
        to_seconds(observation.time - prev_node.time) / to_seconds(next_node.time - prev_node.time);

    let (rotation, translation) = interpolate_nodes_2d(
        prev_node_pose,
        &prev_node.gravity_alignment,
        next_node_pose,
        &next_node.gravity_alignment,
        interpolation_parameter,
    );
    Rigid3d::from_arrays(&rotation, &translation) * observation.landmark_to_tracking_transform
}

fn add_landmark_cost_functions(
    landmark_nodes: &BTreeMap<String, LandmarkNode>,
    node_data: &MapById<NodeId, NodeSpec2D>,
    c_nodes: &MapById<NodeId, ParameterBlockId>,
    c_landmarks: &mut BTreeMap<String, CeresPose>,
    problem: &mut Problem<'_>,
    huber_scale: f64,
) {
    for (landmark_id, landmark_node) in landmark_nodes {
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
                        problem.parameter_block(prev_node_pose),
                        problem.parameter_block(next_node_pose),
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
                LandmarkCostFunction2D::create_auto_diff_cost_function(
                    observation,
                    prev.data,
                    next.data,
                ),
                Some(Box::new(HuberLoss::new(huber_scale))),
                &[
                    prev_node_pose,
                    next_node_pose,
                    landmark.rotation(),
                    landmark.translation(),
                ],
            );
        }
    }
}

/// The set of frozen trajectories.
pub(crate) fn frozen_trajectories(
    trajectories_state: &BTreeMap<i32, TrajectoryState>,
) -> BTreeSet<i32> {
    trajectories_state
        .iter()
        .filter(|(_, state)| **state == TrajectoryState::Frozen)
        .map(|(id, _)| *id)
        .collect()
}

/// The fixed frame pose loss (C++ `TolerantLoss` if
/// `fixed_frame_pose_use_tolerant_loss`, else none).
pub(crate) fn fixed_frame_pose_loss(
    options: &proto::OptimizationProblemOptions,
) -> Option<Box<dyn LossFunction>> {
    options.fixed_frame_pose_use_tolerant_loss.then(|| {
        Box::new(ceres::TolerantLoss::new(
            options.fixed_frame_pose_tolerant_loss_param_a,
            options.fixed_frame_pose_tolerant_loss_param_b,
        )) as Box<dyn LossFunction>
    })
}

/// The loss of a constraint: loop closure (inter-submap) constraints use a
/// Huber loss.
pub(crate) fn constraint_loss(tag: Tag, huber_scale: f64) -> Option<Box<dyn LossFunction>> {
    match tag {
        Tag::InterSubmap => Some(Box::new(HuberLoss::new(huber_scale))),
        Tag::IntraSubmap => None,
    }
}

/// The ceres solver options of `options` (which always has them when created
/// by `create_optimization_problem_options`).
pub(crate) fn ceres_solver_options(
    options: &proto::OptimizationProblemOptions,
) -> ceres::SolverOptions {
    create_ceres_solver_options(
        options
            .ceres_solver_options
            .as_ref()
            .expect("OptimizationProblemOptions without ceres_solver_options"),
    )
}

/// Sets `ceres_solver_options.max_num_iterations` (C++
/// `mutable_ceres_solver_options()->set_max_num_iterations`).
pub(crate) fn set_max_num_iterations(
    options: &mut proto::OptimizationProblemOptions,
    max_num_iterations: i32,
) {
    options
        .ceres_solver_options
        .get_or_insert_with(Default::default)
        .max_num_iterations = max_num_iterations;
}

/// The 2D sparse pose adjustment problem.
#[derive(Debug)]
pub struct OptimizationProblem2D {
    options: proto::OptimizationProblemOptions,
    node_data: MapById<NodeId, NodeSpec2D>,
    submap_data: MapById<SubmapId, SubmapSpec2D>,
    landmark_data: BTreeMap<String, Rigid3d>,
    empty_imu_data: MapByTime<ImuData>,
    odometry_data: MapByTime<OdometryData>,
    fixed_frame_pose_data: MapByTime<FixedFramePoseData>,
    trajectory_data: BTreeMap<i32, TrajectoryData>,
}

impl OptimizationProblem2D {
    /// Creates an empty problem.
    pub fn new(options: &proto::OptimizationProblemOptions) -> Self {
        OptimizationProblem2D {
            options: options.clone(),
            node_data: MapById::new(),
            submap_data: MapById::new(),
            landmark_data: BTreeMap::new(),
            empty_imu_data: MapByTime::new(),
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
        first_node_data: &NodeSpec2D,
        second_node_data: &NodeSpec2D,
    ) -> Option<Rigid3d> {
        if !self.odometry_data.has_trajectory(trajectory_id) {
            return None;
        }
        let first_node_odometry =
            interpolate_odometry(&self.odometry_data, trajectory_id, first_node_data.time)?;
        let second_node_odometry =
            interpolate_odometry(&self.odometry_data, trajectory_id, second_node_data.time)?;
        Some(
            Rigid3d::from_rotation(first_node_data.gravity_alignment)
                * first_node_odometry.inverse()
                * second_node_odometry
                * Rigid3d::from_rotation(
                    second_node_data
                        .gravity_alignment
                        .try_inverse()
                        .unwrap_or_else(|| Quaternion::new(0., 0., 0., 0.)),
                ),
        )
    }
}

impl OptimizationProblemInterface for OptimizationProblem2D {
    type NodeData = NodeSpec2D;
    type SubmapData = SubmapSpec2D;
    type RigidTransform = Rigid2d;

    fn add_imu_data(&mut self, _trajectory_id: i32, _imu_data: &ImuData) {
        // IMU data is not used in 2D optimization, so we ignore this part of
        // the interface.
    }

    fn add_odometry_data(&mut self, trajectory_id: i32, odometry_data: &OdometryData) {
        self.odometry_data.append(trajectory_id, *odometry_data);
    }

    fn add_trajectory_node(&mut self, trajectory_id: i32, node_data: &NodeSpec2D) {
        self.node_data.append(trajectory_id, *node_data);
        self.trajectory_data.entry(trajectory_id).or_default();
    }

    fn insert_trajectory_node(&mut self, node_id: &NodeId, node_data: &NodeSpec2D) {
        self.node_data.insert(node_id, *node_data);
        self.trajectory_data
            .entry(node_id.trajectory_id)
            .or_default();
    }

    fn trim_trajectory_node(&mut self, node_id: &NodeId) {
        self.empty_imu_data
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

    fn add_submap(&mut self, trajectory_id: i32, global_submap_pose: &Rigid2d) {
        self.submap_data.append(
            trajectory_id,
            SubmapSpec2D {
                global_pose: *global_submap_pose,
            },
        );
    }

    fn insert_submap(&mut self, submap_id: &SubmapId, global_submap_pose: &Rigid2d) {
        self.submap_data.insert(
            submap_id,
            SubmapSpec2D {
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
        // TODO(hrapp): Move ceres data into SubmapSpec.
        let mut c_submaps: MapById<SubmapId, ParameterBlockId> = MapById::new();
        let mut c_nodes: MapById<NodeId, ParameterBlockId> = MapById::new();
        let mut c_landmarks: BTreeMap<String, CeresPose> = BTreeMap::new();
        let mut first_submap = true;
        for submap_id_data in &self.submap_data {
            let frozen = frozen_trajectories.contains(&submap_id_data.id.trajectory_id);
            let block = problem.add_parameter_block(&from_pose(&submap_id_data.data.global_pose));
            c_submaps.insert(&submap_id_data.id, block);
            if first_submap || frozen {
                first_submap = false;
                // Fix the pose of the first submap or all submaps of a frozen
                // trajectory.
                problem.set_parameter_block_constant(block);
            }
        }
        for node_id_data in &self.node_data {
            let frozen = frozen_trajectories.contains(&node_id_data.id.trajectory_id);
            let block = problem.add_parameter_block(&from_pose(&node_id_data.data.global_pose_2d));
            c_nodes.insert(&node_id_data.id, block);
            if frozen {
                problem.set_parameter_block_constant(block);
            }
        }
        // Add cost functions for intra- and inter-submap constraints.
        for constraint in constraints {
            problem.add_residual_block(
                create_auto_diff_spa_cost_function(&constraint.pose),
                // Loop closure constraints should have a loss function.
                constraint_loss(constraint.tag, self.options.huber_scale),
                &[
                    *c_submaps.at(&constraint.submap_id),
                    *c_nodes.at(&constraint.node_id),
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
        // Add penalties for violating odometry or changes between consecutive
        // nodes if odometry is not available.
        for trajectory_id in self.node_data.trajectory_ids() {
            if frozen_trajectories.contains(&trajectory_id) {
                continue;
            }
            let mut prev_node: Option<IdDataReference<'_, NodeId, NodeSpec2D>> = None;
            for node in self.node_data.trajectory(trajectory_id) {
                let Some(first) = prev_node.replace(node) else {
                    continue;
                };
                let (first_node_id, first_node_data) = (first.id, first.data);
                let (second_node_id, second_node_data) = (node.id, node.data);

                if second_node_id.node_index != first_node_id.node_index + 1 {
                    continue;
                }

                // Add a relative pose constraint based on the odometry (if
                // available).
                if let Some(relative_odometry) = self.calculate_odometry_between_nodes(
                    trajectory_id,
                    first_node_data,
                    second_node_data,
                ) {
                    problem.add_residual_block(
                        create_auto_diff_spa_cost_function(&Pose {
                            zbar_ij: relative_odometry,
                            translation_weight: self.options.odometry_translation_weight,
                            rotation_weight: self.options.odometry_rotation_weight,
                        }),
                        None, /* loss function */
                        &[*c_nodes.at(&first_node_id), *c_nodes.at(&second_node_id)],
                    );
                }

                // Add a relative pose constraint based on consecutive local
                // SLAM poses.
                let relative_local_slam_pose = embed_3d(
                    &(first_node_data.local_pose_2d.inverse() * second_node_data.local_pose_2d),
                );
                problem.add_residual_block(
                    create_auto_diff_spa_cost_function(&Pose {
                        zbar_ij: relative_local_slam_pose,
                        translation_weight: self.options.local_slam_pose_translation_weight,
                        rotation_weight: self.options.local_slam_pose_rotation_weight,
                    }),
                    None, /* loss function */
                    &[*c_nodes.at(&first_node_id), *c_nodes.at(&second_node_id)],
                );
            }
        }

        let mut c_fixed_frames: BTreeMap<i32, ParameterBlockId> = BTreeMap::new();
        for trajectory_id in self.node_data.trajectory_ids() {
            if !self.fixed_frame_pose_data.has_trajectory(trajectory_id) {
                continue;
            }

            let trajectory_data = self
                .trajectory_data
                .get(&trajectory_id)
                .unwrap_or_else(|| panic!("map::at: no trajectory data for {trajectory_id}"));
            let mut fixed_frame_pose_block: Option<ParameterBlockId> = None;
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

                let fixed_frame_block = *fixed_frame_pose_block.get_or_insert_with(|| {
                    let fixed_frame_pose_in_map = match &trajectory_data.fixed_frame_origin_in_map {
                        Some(origin) => project_2d(origin),
                        None => {
                            node.data.global_pose_2d
                                * project_2d(&constraint_pose.zbar_ij).inverse()
                        }
                    };
                    let block = problem.add_parameter_block(&from_pose(&fixed_frame_pose_in_map));
                    c_fixed_frames.insert(trajectory_id, block);
                    block
                });

                problem.add_residual_block(
                    create_auto_diff_spa_cost_function(&constraint_pose),
                    fixed_frame_pose_loss(&self.options),
                    &[fixed_frame_block, *c_nodes.at(&node.id)],
                );
            }
        }

        // Solve.
        let mut summary = ceres::SolverSummary::default();
        ceres::solve(
            &ceres_solver_options(&self.options),
            &mut problem,
            &mut summary,
        );
        if self.options.log_solver_summary {
            log::info!("{}", summary.full_report());
        }

        // Store the result.
        for c_submap_id_data in &c_submaps {
            self.submap_data.at_mut(&c_submap_id_data.id).global_pose =
                to_pose(problem.parameter_block(*c_submap_id_data.data));
        }
        for c_node_id_data in &c_nodes {
            self.node_data.at_mut(&c_node_id_data.id).global_pose_2d =
                to_pose(problem.parameter_block(*c_node_id_data.data));
        }
        for (trajectory_id, block) in &c_fixed_frames {
            self.trajectory_data
                .get_mut(trajectory_id)
                .expect("trajectory data of a fixed frame")
                .fixed_frame_origin_in_map =
                Some(embed_3d(&to_pose(problem.parameter_block(*block))));
        }
        for (landmark_id, pose) in &c_landmarks {
            self.landmark_data
                .insert(landmark_id.clone(), pose.to_rigid(&problem));
        }
    }

    fn node_data(&self) -> &MapById<NodeId, NodeSpec2D> {
        &self.node_data
    }

    fn submap_data(&self) -> &MapById<SubmapId, SubmapSpec2D> {
        &self.submap_data
    }

    fn landmark_data(&self) -> &BTreeMap<String, Rigid3d> {
        &self.landmark_data
    }

    fn imu_data(&self) -> &MapByTime<ImuData> {
        &self.empty_imu_data
    }

    fn odometry_data(&self) -> &MapByTime<OdometryData> {
        &self.odometry_data
    }
}
