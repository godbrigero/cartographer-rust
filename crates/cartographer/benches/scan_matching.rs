//! Scan matcher benchmarks on realistic submaps built from simulated data.
//!
//! Groups (criterion ids `<group>/<bench>`):
//! * `scan_matching_2d`: `ceres_match`, `real_time_correlative_match`,
//!   `fast_correlative_new`, `fast_correlative_match`,
//!   `fast_correlative_match_full_submap`.
//! * `scan_matching_3d`: `ceres_match`, `real_time_correlative_match`,
//!   `fast_correlative_new`, `fast_correlative_match`.

mod common;

use std::hint::black_box;

use cartographer::mapping::scan_matching::{
    CeresScanMatcher2D, CeresScanMatcher3D, FastCorrelativeScanMatcher2D,
    FastCorrelativeScanMatcher3D, PointCloudAndHybridGridsPointers,
    RealTimeCorrelativeScanMatcher2D, RealTimeCorrelativeScanMatcher3D,
    create_ceres_scan_matcher_options_2d, create_ceres_scan_matcher_options_3d,
    create_fast_correlative_scan_matcher_options_2d,
    create_fast_correlative_scan_matcher_options_3d,
    create_real_time_correlative_scan_matcher_options,
};
use cartographer::transform::rigid_transform::Rigid3d;
use ceres::SolverSummary;
use common::{lua_from_file, perturb_2d, perturb_3d};
use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;

fn scan_matching_2d(c: &mut Criterion) {
    let fixture = common::scan_matching_fixture_2d(false);
    let submap = fixture.submap.read().unwrap();
    let grid = submap.grid();
    let point_cloud = &fixture.point_cloud;
    eprintln!(
        "scan_matching_2d fixture: {} points, submap with {} nodes, {}x{} cells",
        point_cloud.size(),
        fixture.num_range_data,
        grid.limits().cell_limits().num_x_cells,
        grid.limits().cell_limits().num_y_cells,
    );

    let mut group = c.benchmark_group("scan_matching_2d");

    // Local SLAM: trajectory_builder_2d.lua ceres_scan_matcher (20 iterations
    // max), initial estimate 3 cm / 2 cm / 0.6 deg off.
    let ceres_scan_matcher =
        CeresScanMatcher2D::new(&create_ceres_scan_matcher_options_2d(&lua_from_file(
            "trajectory_builder_2d.lua",
            "TRAJECTORY_BUILDER_2D.ceres_scan_matcher",
        )));
    let initial = perturb_2d(&fixture.pose, 0.03, -0.02, 0.01);
    group.bench_function("ceres_match", |b| {
        b.iter(|| {
            let mut summary = SolverSummary::default();
            black_box(ceres_scan_matcher.match_(
                initial.translation(),
                &initial,
                black_box(point_cloud),
                grid,
                &mut summary,
            ))
        })
    });

    // Local SLAM online correlative matcher: 0.1 m / 20 deg window.
    let real_time_correlative_scan_matcher = RealTimeCorrelativeScanMatcher2D::new(
        &create_real_time_correlative_scan_matcher_options(&lua_from_file(
            "trajectory_builder_2d.lua",
            "TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher",
        )),
    );
    group.bench_function("real_time_correlative_match", |b| {
        b.iter(|| {
            black_box(real_time_correlative_scan_matcher.match_(
                &initial,
                black_box(point_cloud),
                grid,
            ))
        })
    });

    // Loop closure: pose_graph.lua fast_correlative_scan_matcher (7 m,
    // 30 deg, branch_and_bound_depth 7).
    let fast_options = create_fast_correlative_scan_matcher_options_2d(&lua_from_file(
        "pose_graph.lua",
        "POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher",
    ));
    group.bench_function("fast_correlative_new", |b| {
        b.iter(|| black_box(FastCorrelativeScanMatcher2D::new(grid, &fast_options)))
    });
    let fast_correlative_scan_matcher = FastCorrelativeScanMatcher2D::new(grid, &fast_options);
    let loop_closure_initial = perturb_2d(&fixture.pose, 0.4, -0.3, 0.1);
    let (score, _) = fast_correlative_scan_matcher
        .match_(&loop_closure_initial, point_cloud, 0.55)
        .expect("fast correlative match");
    eprintln!("fast_correlative_match score {score}");
    group.bench_function("fast_correlative_match", |b| {
        b.iter(|| {
            black_box(fast_correlative_scan_matcher.match_(
                &loop_closure_initial,
                black_box(point_cloud),
                0.55,
            ))
        })
    });
    // Global localization (pose_graph.lua global_localization_min_score).
    let (score, _) = fast_correlative_scan_matcher
        .match_full_submap(point_cloud, 0.6)
        .expect("fast correlative full submap match");
    eprintln!("fast_correlative_match_full_submap score {score}");
    group.bench_function("fast_correlative_match_full_submap", |b| {
        b.iter(|| {
            black_box(fast_correlative_scan_matcher.match_full_submap(black_box(point_cloud), 0.6))
        })
    });
    group.finish();
}

fn scan_matching_3d(c: &mut Criterion) {
    let fixture = common::scan_matching_fixture_3d();
    let submap = fixture.submap.read().unwrap();
    let high_resolution_grid = submap.high_resolution_hybrid_grid();
    let low_resolution_grid = submap.low_resolution_hybrid_grid();
    let data = &fixture.constant_data;
    eprintln!(
        "scan_matching_3d fixture: {} high / {} low resolution points",
        data.high_resolution_point_cloud.size(),
        data.low_resolution_point_cloud.size(),
    );

    let mut group = c.benchmark_group("scan_matching_3d");

    // Local SLAM: trajectory_builder_3d.lua ceres_scan_matcher (12 iterations
    // max, no intensity grid), initial estimate 5 cm / 1 deg off.
    let ceres_scan_matcher =
        CeresScanMatcher3D::new(&create_ceres_scan_matcher_options_3d(&lua_from_file(
            "trajectory_builder_3d.lua",
            "TRAJECTORY_BUILDER_3D.ceres_scan_matcher",
        )));
    let initial = perturb_3d(
        &fixture.pose_in_submap,
        Vector3::new(0.05, -0.03, 0.02),
        0.0175,
    );
    let inputs = [
        PointCloudAndHybridGridsPointers {
            point_cloud: &data.high_resolution_point_cloud,
            hybrid_grid: high_resolution_grid,
            intensity_hybrid_grid: None,
        },
        PointCloudAndHybridGridsPointers {
            point_cloud: &data.low_resolution_point_cloud,
            hybrid_grid: low_resolution_grid,
            intensity_hybrid_grid: None,
        },
    ];
    group.bench_function("ceres_match", |b| {
        b.iter(|| {
            let mut pose = Rigid3d::identity();
            let mut summary = SolverSummary::default();
            ceres_scan_matcher.match_(
                initial.translation(),
                &initial,
                black_box(&inputs),
                &mut pose,
                &mut summary,
            );
            black_box(pose)
        })
    });

    // Online correlative matcher: 0.15 m / 1 deg window on the high
    // resolution grid.
    let real_time_correlative_scan_matcher = RealTimeCorrelativeScanMatcher3D::new(
        &create_real_time_correlative_scan_matcher_options(&lua_from_file(
            "trajectory_builder_3d.lua",
            "TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher",
        )),
    );
    group.bench_function("real_time_correlative_match", |b| {
        b.iter(|| {
            let mut pose = Rigid3d::identity();
            black_box(real_time_correlative_scan_matcher.match_(
                &initial,
                black_box(&data.high_resolution_point_cloud),
                high_resolution_grid,
                &mut pose,
            ))
        })
    });

    // Loop closure: pose_graph.lua fast_correlative_scan_matcher_3d (5 m xy,
    // 1 m z, 15 deg, branch_and_bound_depth 8).
    let fast_options = create_fast_correlative_scan_matcher_options_3d(&lua_from_file(
        "pose_graph.lua",
        "POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d",
    ));
    let histogram = submap.rotational_scan_matcher_histogram();
    group.bench_function("fast_correlative_new", |b| {
        b.iter(|| {
            black_box(FastCorrelativeScanMatcher3D::new(
                high_resolution_grid,
                low_resolution_grid,
                histogram,
                &fast_options,
            ))
        })
    });
    let fast_correlative_scan_matcher = FastCorrelativeScanMatcher3D::new(
        high_resolution_grid,
        low_resolution_grid,
        histogram,
        &fast_options,
    );
    let submap_pose = fixture.pose_in_local * fixture.pose_in_submap.inverse();
    let node_pose = perturb_3d(&fixture.pose_in_local, Vector3::new(0.3, -0.2, 0.05), 0.05);
    let result = fast_correlative_scan_matcher
        .match_(&node_pose, &submap_pose, data, 0.55)
        .expect("fast correlative 3D match");
    eprintln!(
        "fast_correlative_match 3D: score {} rotational {} low resolution {}, error {:?}",
        result.score,
        result.rotational_score,
        result.low_resolution_score,
        (result.pose_estimate.translation() - fixture.pose_in_submap.translation()).norm()
    );
    group.bench_function("fast_correlative_match", |b| {
        b.iter(|| {
            black_box(fast_correlative_scan_matcher.match_(
                &node_pose,
                &submap_pose,
                black_box(data),
                0.55,
            ))
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = scan_matching_2d, scan_matching_3d
}
criterion_main!(benches);
