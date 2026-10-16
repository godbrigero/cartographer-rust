//! Pose graph optimization benchmarks on synthetic, deterministic graphs.
//!
//! Groups (criterion ids `<group>/<bench>`):
//! * `optimization_problem_2d`: `solve/500_nodes`, `solve/2000_nodes`.
//! * `optimization_problem_3d`: `solve_with_imu/200_nodes`.
//! * `ceres_spa_2d_sparse_normal_cholesky`: `5_iterations/1000_poses`,
//!   `5_iterations/4000_poses`, `5_iterations/16000_poses`.
//!
//! The problems are rebuilt outside the timed section for every iteration
//! (`solve` writes the optimized poses back).

mod common;

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::hint::black_box;

use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::optimization::{
    NodeSpec2D, NodeSpec3D, OptimizationProblem2D, OptimizationProblem3D,
    OptimizationProblemInterface, create_analytical_spa_cost_function,
    create_optimization_problem_options, proto,
};
use cartographer::mapping::pose_graph_interface::{Constraint, Pose, Tag, TrajectoryState};
use cartographer::sensor::imu_data::ImuData;
use cartographer::transform::rigid_transform::{AngleAxis, Rigid2d, Rigid3d};
use cartographer::transform::transform::embed_3d;
use cartographer_sim::{DatasetBuilder, ImuConfig, Trajectory, World3d};
use ceres::{
    HuberLoss, LinearSolverType, ParameterBlockId, Problem, SolverOptions, SolverSummary, solve,
};
use common::{lua_from_file, to_rigid3d, to_time};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{Vector2, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr_free::gauss;

/// Box-Muller normal samples without an extra dependency.
mod rand_distr_free {
    use rand::Rng;

    pub fn gauss(rng: &mut impl Rng, std: f64) -> f64 {
        let u1: f64 = rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

const NODES_PER_SUBMAP: usize = 20;
/// `POSE_GRAPH.matcher_{translation,rotation}_weight` (intra-submap).
const INTRA_WEIGHTS: (f64, f64) = (5e2, 1.6e3);
/// `POSE_GRAPH.constraint_builder.loop_closure_{translation,rotation}_weight`.
const LOOP_CLOSURE_WEIGHTS: (f64, f64) = (1.1e4, 1e5);

fn optimization_options() -> proto::OptimizationProblemOptions {
    create_optimization_problem_options(&lua_from_file(
        "pose_graph.lua",
        "POSE_GRAPH.optimization_problem",
    ))
}

fn constraint(submap: usize, node: usize, zbar_ij: Rigid3d, tag: Tag) -> Constraint {
    let (translation_weight, rotation_weight) = match tag {
        Tag::IntraSubmap => INTRA_WEIGHTS,
        Tag::InterSubmap => LOOP_CLOSURE_WEIGHTS,
    };
    Constraint {
        submap_id: SubmapId::new(0, submap as i32),
        node_id: NodeId::new(0, node as i32),
        pose: Pose {
            zbar_ij,
            translation_weight,
            rotation_weight,
        },
        tag,
    }
}

/// A synthetic single-trajectory 2D pose graph: `num_nodes` nodes 0.2 m apart
/// on laps of a 10 m radius circle (~314 nodes per lap), local SLAM drift
/// (1 cm / 0.1 deg per node), one submap per 20 nodes, every node inserted
/// into the two active submaps (intra constraints from the drifted local
/// poses), and loop closures (2 cm / 0.3 deg noise) from every third node to
/// the closest submap of an earlier lap within 3 m.
struct PoseGraph2D {
    nodes: Vec<NodeSpec2D>,
    submaps: Vec<Rigid2d>,
    constraints: Vec<Constraint>,
}

fn pose_graph_2d(num_nodes: usize) -> PoseGraph2D {
    let mut rng = StdRng::seed_from_u64(num_nodes as u64);
    let radius = 10.0;
    let step = 0.2 / radius;
    let ground_truth: Vec<Rigid2d> = (0..num_nodes)
        .map(|i| {
            let angle = i as f64 * step;
            Rigid2d::new(
                Vector2::new(radius * angle.sin(), radius * (1.0 - angle.cos())),
                angle,
            )
        })
        .collect();
    let mut local = vec![ground_truth[0]];
    for i in 1..num_nodes {
        let delta = ground_truth[i - 1].inverse() * ground_truth[i];
        let noise = Rigid2d::new(
            Vector2::new(gauss(&mut rng, 0.01), gauss(&mut rng, 0.01)),
            gauss(&mut rng, 0.1f64.to_radians()),
        );
        local.push(local[i - 1] * delta * noise);
    }
    let nodes = local
        .iter()
        .enumerate()
        .map(|(i, pose)| NodeSpec2D {
            time: to_time(0.1 * i as f64),
            local_pose_2d: *pose,
            global_pose_2d: *pose,
            gravity_alignment: nalgebra::Quaternion::identity(),
        })
        .collect();
    let num_submaps = num_nodes.div_ceil(NODES_PER_SUBMAP);
    let submaps: Vec<Rigid2d> = (0..num_submaps)
        .map(|s| local[s * NODES_PER_SUBMAP])
        .collect();
    let mut constraints = Vec::new();
    for i in 0..num_nodes {
        let newest = i / NODES_PER_SUBMAP;
        for (submap, submap_pose) in submaps
            .iter()
            .enumerate()
            .take(newest + 1)
            .skip(newest.saturating_sub(1))
        {
            constraints.push(constraint(
                submap,
                i,
                embed_3d(&(submap_pose.inverse() * local[i])),
                Tag::IntraSubmap,
            ));
        }
        if i % 3 != 0 {
            continue;
        }
        let closest = (0..num_submaps)
            .filter(|&s| (s + 4) * NODES_PER_SUBMAP < i)
            .map(|s| {
                let origin = ground_truth[s * NODES_PER_SUBMAP];
                (
                    s,
                    (origin.translation() - ground_truth[i].translation()).norm(),
                )
            })
            .filter(|&(_, distance)| distance < 3.0)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((s, _)) = closest {
            let noise = Rigid2d::new(
                Vector2::new(gauss(&mut rng, 0.02), gauss(&mut rng, 0.02)),
                gauss(&mut rng, 0.3f64.to_radians()),
            );
            constraints.push(constraint(
                s,
                i,
                embed_3d(&(ground_truth[s * NODES_PER_SUBMAP].inverse() * ground_truth[i] * noise)),
                Tag::InterSubmap,
            ));
        }
    }
    PoseGraph2D {
        nodes,
        submaps,
        constraints,
    }
}

fn active_trajectory() -> BTreeMap<i32, TrajectoryState> {
    BTreeMap::from([(0, TrajectoryState::Active)])
}

fn optimization_problem_2d(c: &mut Criterion) {
    let options = optimization_options();
    let mut group = c.benchmark_group("optimization_problem_2d");
    for num_nodes in [500, 2000] {
        let graph = pose_graph_2d(num_nodes);
        let num_loop_closures = graph
            .constraints
            .iter()
            .filter(|c| c.tag == Tag::InterSubmap)
            .count();
        eprintln!(
            "optimization_problem_2d: {num_nodes} nodes, {} submaps, {} constraints ({num_loop_closures} loop closures)",
            graph.submaps.len(),
            graph.constraints.len(),
        );
        let build = || {
            let mut problem = OptimizationProblem2D::new(&options);
            for node in &graph.nodes {
                problem.add_trajectory_node(0, node);
            }
            for submap in &graph.submaps {
                problem.add_submap(0, submap);
            }
            problem
        };
        group.bench_function(
            BenchmarkId::new("solve", format!("{num_nodes}_nodes")),
            |b| {
                b.iter_batched(
                    build,
                    |mut problem| {
                        problem.solve(&graph.constraints, &active_trajectory(), &BTreeMap::new());
                        problem
                    },
                    BatchSize::PerIteration,
                )
            },
        );
    }
    group.finish();
}

fn random_rotation(rng: &mut StdRng, roll_pitch_std: f64, yaw_std: f64) -> Rigid3d {
    Rigid3d::from_rotation(
        AngleAxis::new(gauss(rng, yaw_std), Vector3::z()).to_quaternion()
            * AngleAxis::new(gauss(rng, roll_pitch_std), Vector3::x()).to_quaternion()
            * AngleAxis::new(gauss(rng, roll_pitch_std), Vector3::y()).to_quaternion(),
    )
}

/// 20 s of `Trajectory::corridor_loops_3d` at 1.2 m/s (handheld motion) with
/// a 200 Hz simulated IMU; 199 nodes at 10 Hz with local SLAM drift
/// (1 cm, 0.1 deg yaw, 0.03 deg roll/pitch per node), one submap per 20 nodes,
/// intra constraints to the two active submaps and an inter-submap
/// constraint (2 cm / 0.3 deg noise) from every 5th node to a submap two
/// submaps back.
fn optimization_problem_3d(c: &mut Criterion) {
    let dataset = DatasetBuilder::new(Trajectory::corridor_loops_3d(1.0, 1.2), World3d::room())
        .seed(5)
        .time_range(0.0, 20.0)
        .imu("imu", ImuConfig::default())
        .build();
    let imu_data: Vec<ImuData> = dataset
        .imu("imu")
        .map(|(time, imu)| ImuData {
            time: to_time(time),
            linear_acceleration: imu.linear_acceleration,
            angular_velocity: imu.angular_velocity,
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(3);
    let times: Vec<f64> = (1..200).map(|i| 0.1 * f64::from(i)).collect();
    let ground_truth: Vec<Rigid3d> = times
        .iter()
        .map(|&t| to_rigid3d(&dataset.ground_truth_at(t).pose))
        .collect();
    let mut local = vec![ground_truth[0]];
    for i in 1..ground_truth.len() {
        let noise = Rigid3d::from_translation(Vector3::new(
            gauss(&mut rng, 0.01),
            gauss(&mut rng, 0.01),
            gauss(&mut rng, 0.01),
        )) * random_rotation(&mut rng, 0.03f64.to_radians(), 0.1f64.to_radians());
        local.push(local[i - 1] * (ground_truth[i - 1].inverse() * ground_truth[i]) * noise);
    }
    let num_submaps = local.len().div_ceil(NODES_PER_SUBMAP);
    let submaps: Vec<Rigid3d> = (0..num_submaps)
        .map(|s| local[s * NODES_PER_SUBMAP])
        .collect();
    let mut constraints = Vec::new();
    for i in 0..local.len() {
        let newest = i / NODES_PER_SUBMAP;
        for (submap, submap_pose) in submaps
            .iter()
            .enumerate()
            .take(newest + 1)
            .skip(newest.saturating_sub(1))
        {
            constraints.push(constraint(
                submap,
                i,
                submap_pose.inverse() * local[i],
                Tag::IntraSubmap,
            ));
        }
        if i % 5 == 0 && newest >= 2 {
            let s = newest - 2;
            let noise = Rigid3d::from_translation(Vector3::new(
                gauss(&mut rng, 0.02),
                gauss(&mut rng, 0.02),
                gauss(&mut rng, 0.02),
            )) * random_rotation(&mut rng, 0.1f64.to_radians(), 0.3f64.to_radians());
            constraints.push(constraint(
                s,
                i,
                ground_truth[s * NODES_PER_SUBMAP].inverse() * ground_truth[i] * noise,
                Tag::InterSubmap,
            ));
        }
    }
    eprintln!(
        "optimization_problem_3d: {} nodes, {num_submaps} submaps, {} constraints, {} IMU samples",
        local.len(),
        constraints.len(),
        imu_data.len()
    );
    let options = optimization_options();
    let build = || {
        let mut problem = OptimizationProblem3D::new(&options);
        for imu in &imu_data {
            problem.add_imu_data(0, imu);
        }
        for (i, pose) in local.iter().enumerate() {
            problem.add_trajectory_node(
                0,
                &NodeSpec3D {
                    time: to_time(times[i]),
                    local_pose: *pose,
                    global_pose: *pose,
                },
            );
        }
        for submap in &submaps {
            problem.add_submap(0, submap);
        }
        problem
    };
    let mut group = c.benchmark_group("optimization_problem_3d");
    group.bench_function(BenchmarkId::new("solve_with_imu", "200_nodes"), |b| {
        b.iter_batched(
            build,
            |mut problem| {
                problem.solve(&constraints, &active_trajectory(), &BTreeMap::new());
                problem
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

/// Raw `ceres` 2D SPA: `num_poses` poses on laps of a circle, odometry edges
/// (1 cm / 0.1 deg noise) between consecutive poses and Huber-robustified
/// loop closures to the pose one lap earlier, analytical SPA cost functions,
/// `SPARSE_NORMAL_CHOLESKY`, exactly 5 LM iterations (tolerances disabled).
fn ceres_spa_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceres_spa_2d_sparse_normal_cholesky");
    for num_poses in [1000usize, 4000, 16000] {
        let mut rng = StdRng::seed_from_u64(num_poses as u64);
        let poses_per_lap = 500;
        let radius = 20.0;
        let ground_truth: Vec<Rigid2d> = (0..num_poses)
            .map(|i| {
                let angle = 2.0 * PI * i as f64 / poses_per_lap as f64;
                Rigid2d::new(
                    Vector2::new(radius * angle.sin(), radius * (1.0 - angle.cos())),
                    angle,
                )
            })
            .collect();
        let mut noisy = |pose: Rigid2d, t: f64, r: f64| {
            pose * Rigid2d::new(
                Vector2::new(gauss(&mut rng, t), gauss(&mut rng, t)),
                gauss(&mut rng, r),
            )
        };
        let mut initial = vec![ground_truth[0]];
        let mut edges = Vec::new();
        for i in 1..num_poses {
            let odometry = noisy(
                ground_truth[i - 1].inverse() * ground_truth[i],
                0.01,
                0.1f64.to_radians(),
            );
            initial.push(initial[i - 1] * odometry);
            edges.push((i - 1, i, odometry, false));
            if i >= poses_per_lap && i % 2 == 0 {
                let j = i - poses_per_lap;
                let loop_closure = noisy(
                    ground_truth[j].inverse() * ground_truth[i],
                    0.02,
                    0.3f64.to_radians(),
                );
                edges.push((j, i, loop_closure, true));
            }
        }
        let options = SolverOptions {
            linear_solver_type: LinearSolverType::SparseNormalCholesky,
            max_num_iterations: 5,
            function_tolerance: 0.0,
            gradient_tolerance: 0.0,
            parameter_tolerance: 0.0,
            num_threads: 1,
            ..SolverOptions::default()
        };
        let build = || {
            let mut problem = Problem::new();
            let blocks: Vec<ParameterBlockId> = initial
                .iter()
                .map(|p| {
                    problem.add_parameter_block(&[
                        p.translation().x,
                        p.translation().y,
                        p.rotation().angle(),
                    ])
                })
                .collect();
            problem.set_parameter_block_constant(blocks[0]);
            for &(a, b, ref zbar, is_loop_closure) in &edges {
                let pose = Pose {
                    zbar_ij: embed_3d(zbar),
                    translation_weight: if is_loop_closure { 1.1e4 } else { 1e5 },
                    rotation_weight: 1e5,
                };
                problem.add_residual_block(
                    create_analytical_spa_cost_function(&pose),
                    if is_loop_closure {
                        Some(Box::new(HuberLoss::new(1e1)))
                    } else {
                        None
                    },
                    &[blocks[a], blocks[b]],
                );
            }
            problem
        };
        {
            let mut problem = build();
            let mut summary = SolverSummary::default();
            solve(&options, &mut problem, &mut summary);
            eprintln!(
                "ceres_spa_2d {num_poses} poses, {} edges: {}",
                edges.len(),
                summary.brief_report()
            );
        }
        group.bench_function(
            BenchmarkId::new("5_iterations", format!("{num_poses}_poses")),
            |b| {
                b.iter_batched(
                    build,
                    |mut problem| {
                        let mut summary = SolverSummary::default();
                        solve(&options, &mut problem, &mut summary);
                        black_box(summary.final_cost)
                    },
                    BatchSize::PerIteration,
                )
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = optimization_problem_2d, optimization_problem_3d, ceres_spa_2d
}
criterion_main!(benches);
