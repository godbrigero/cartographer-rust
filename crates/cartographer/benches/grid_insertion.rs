//! Range data insertion and point cloud filtering benchmarks.
//!
//! Groups (criterion ids `<group>/<bench>`):
//! * `grid_insertion_2d`: `probability_grid_720_beams`, `tsdf_720_beams`.
//! * `grid_insertion_3d`: `hybrid_grid_16_rings_raw`,
//!   `hybrid_grid_16_rings_voxel_filtered`,
//!   `low_resolution_hybrid_grid_16_rings_voxel_filtered`.
//! * `voxel_filter_3d`: `voxel_filter`, `adaptive_voxel_filter_high_resolution`,
//!   `adaptive_voxel_filter_low_resolution`.
//!
//! Insertion benches insert the same scan repeatedly into a grid that already
//! holds 20 other scans of the sequence (grid growth happens before timing).

mod common;

use std::hint::black_box;
use std::sync::Arc;

use cartographer::mapping::internal::two_d::tsdf_2d::TSDF2D;
use cartographer::mapping::internal::two_d::tsdf_range_data_inserter_2d::{
    TSDFRangeDataInserter2D, create_tsdf_range_data_inserter_options_2d,
};
use cartographer::mapping::three_d::hybrid_grid::HybridGrid;
use cartographer::mapping::three_d::range_data_inserter_3d::{
    RangeDataInserter3D, create_range_data_inserter_options_3d,
};
use cartographer::mapping::two_d::map_limits::MapLimits;
use cartographer::mapping::two_d::probability_grid::ProbabilityGrid;
use cartographer::mapping::two_d::probability_grid_range_data_inserter_2d::{
    ProbabilityGridRangeDataInserter2D, create_probability_grid_range_data_inserter_options_2d,
};
use cartographer::mapping::two_d::xy_index::CellLimits;
use cartographer::mapping::value_conversion_tables::ValueConversionTables;
use cartographer::sensor::internal::voxel_filter::{
    adaptive_voxel_filter, create_adaptive_voxel_filter_options, voxel_filter,
};
use cartographer::sensor::point_cloud::{PointCloud, transform_point_cloud};
use cartographer::sensor::range_data::RangeData;
use common::{VOXEL_FILTER_SIZE_3D, lua_from_file};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::Vector2;

/// Initial 2D grid like `ActiveSubmaps2D::CreateGrid` (100 x 100 cells of
/// 5 cm centred at `origin`).
fn initial_limits_2d(origin: &Vector2<f32>) -> MapLimits {
    const INITIAL_SUBMAP_SIZE: i32 = 100;
    let resolution = 0.05;
    MapLimits::new(
        resolution,
        origin.cast::<f64>()
            + 0.5 * f64::from(INITIAL_SUBMAP_SIZE) * resolution * Vector2::new(1., 1.),
        CellLimits::new(INITIAL_SUBMAP_SIZE, INITIAL_SUBMAP_SIZE),
    )
}

fn grid_insertion_2d(c: &mut Criterion) {
    let scans: Vec<RangeData> = common::scans_2d(61).into_iter().step_by(3).collect();
    let (scan, previous) = scans.split_last().unwrap();
    let num_rays = (scan.returns.size() + scan.misses.size()) as u64;
    eprintln!(
        "grid_insertion_2d: {} returns, {} misses",
        scan.returns.size(),
        scan.misses.size()
    );
    let conversion_tables = Arc::new(ValueConversionTables::new());
    let mut group = c.benchmark_group("grid_insertion_2d");
    group.throughput(Throughput::Elements(num_rays));

    let inserter = ProbabilityGridRangeDataInserter2D::new(
        &create_probability_grid_range_data_inserter_options_2d(&lua_from_file(
            "trajectory_builder_2d.lua",
            "TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter",
        )),
    );
    let mut probability_grid = ProbabilityGrid::new(
        initial_limits_2d(&previous[0].origin.xy()),
        &conversion_tables,
    );
    for range_data in previous {
        inserter.insert_into_probability_grid(range_data, &mut probability_grid);
    }
    group.bench_function("probability_grid_720_beams", |b| {
        b.iter(|| inserter.insert_into_probability_grid(black_box(scan), &mut probability_grid))
    });

    let tsdf_options = create_tsdf_range_data_inserter_options_2d(&lua_from_file(
        "trajectory_builder_2d.lua",
        "TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.tsdf_range_data_inserter",
    ));
    let tsdf_inserter = TSDFRangeDataInserter2D::new(&tsdf_options);
    let mut tsdf = TSDF2D::new(
        initial_limits_2d(&previous[0].origin.xy()),
        tsdf_options.truncation_distance as f32,
        tsdf_options.maximum_weight as f32,
        &conversion_tables,
    );
    for range_data in previous {
        tsdf_inserter.insert_into_tsdf(range_data, &mut tsdf);
    }
    group.bench_function("tsdf_720_beams", |b| {
        b.iter(|| tsdf_inserter.insert_into_tsdf(black_box(scan), &mut tsdf))
    });
    group.finish();
}

fn grid_insertion_3d(c: &mut Criterion) {
    let scans = common::scans_3d(21);
    let range_data = |cloud: &PointCloud, scan: &common::Scan3D| {
        let pose = scan.pose.cast::<f32>();
        RangeData {
            origin: *pose.translation(),
            returns: transform_point_cloud(cloud, &pose),
            misses: PointCloud::new(),
        }
    };
    let filtered: Vec<RangeData> = scans
        .iter()
        .map(|scan| range_data(&voxel_filter(&scan.raw, VOXEL_FILTER_SIZE_3D), scan))
        .collect();
    let last = scans.last().unwrap();
    let raw = range_data(&last.raw, last);
    let (scan, previous) = filtered.split_last().unwrap();
    eprintln!(
        "grid_insertion_3d: {} raw returns, {} after voxel filter",
        raw.returns.size(),
        scan.returns.size()
    );
    // trajectory_builder_3d.lua: submaps.range_data_inserter, 0.10 m high /
    // 0.45 m low resolution.
    let inserter =
        RangeDataInserter3D::new(&create_range_data_inserter_options_3d(&lua_from_file(
            "trajectory_builder_3d.lua",
            "TRAJECTORY_BUILDER_3D.submaps.range_data_inserter",
        )));
    let make_grid = |resolution: f32| {
        let mut grid = HybridGrid::new(resolution);
        for range_data in previous {
            inserter.insert(range_data, &mut grid, None);
        }
        grid
    };

    let mut group = c.benchmark_group("grid_insertion_3d");
    let mut high_resolution_grid = make_grid(0.10);
    group.throughput(Throughput::Elements(raw.returns.size() as u64));
    group.bench_function("hybrid_grid_16_rings_raw", |b| {
        b.iter(|| inserter.insert(black_box(&raw), &mut high_resolution_grid, None))
    });
    group.throughput(Throughput::Elements(scan.returns.size() as u64));
    group.bench_function("hybrid_grid_16_rings_voxel_filtered", |b| {
        b.iter(|| inserter.insert(black_box(scan), &mut high_resolution_grid, None))
    });
    let mut low_resolution_grid = make_grid(0.45);
    group.bench_function("low_resolution_hybrid_grid_16_rings_voxel_filtered", |b| {
        b.iter(|| inserter.insert(black_box(scan), &mut low_resolution_grid, None))
    });
    group.finish();

    let mut group = c.benchmark_group("voxel_filter_3d");
    group.throughput(Throughput::Elements(last.raw.size() as u64));
    group.bench_function("voxel_filter", |b| {
        b.iter(|| voxel_filter(black_box(&last.raw), VOXEL_FILTER_SIZE_3D))
    });
    let voxel_filtered = voxel_filter(&last.raw, VOXEL_FILTER_SIZE_3D);
    group.throughput(Throughput::Elements(voxel_filtered.size() as u64));
    for (name, key) in [
        (
            "adaptive_voxel_filter_high_resolution",
            "high_resolution_adaptive_voxel_filter",
        ),
        (
            "adaptive_voxel_filter_low_resolution",
            "low_resolution_adaptive_voxel_filter",
        ),
    ] {
        let options = create_adaptive_voxel_filter_options(&lua_from_file(
            "trajectory_builder_3d.lua",
            &format!("TRAJECTORY_BUILDER_3D.{key}"),
        ));
        group.bench_function(name, |b| {
            b.iter(|| adaptive_voxel_filter(black_box(&voxel_filtered), &options))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = grid_insertion_2d, grid_insertion_3d
}
criterion_main!(benches);
