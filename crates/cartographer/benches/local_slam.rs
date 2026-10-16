//! `LocalTrajectoryBuilder2D` throughput on a pre-generated simulated
//! sequence.
//!
//! Group `local_slam_2d` (criterion ids `local_slam_2d/<bench>`):
//! * `probability_grid_imu/100_scans`: shipped `trajectory_builder_2d.lua`
//!   defaults (IMU on).
//! * `probability_grid_imu_online_correlative/100_scans`: plus
//!   `use_online_correlative_scan_matching = true`.
//! * `tsdf_imu/100_scans`: TSDF submaps and inserter.
//!
//! One iteration feeds the whole 10 s sequence (100 scans of 720 beams, 2000
//! IMU samples) into a fresh builder; the throughput is reported in scans/s.

mod common;

use std::hint::black_box;

use common::{LocalSlamInput, local_slam_inputs, local_trajectory_builder_options_2d};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn local_slam_2d(c: &mut Criterion) {
    let dataset = common::room_dataset_2d(10.0, 7);
    let inputs = local_slam_inputs(&dataset);
    let num_scans = inputs
        .iter()
        .filter(|input| matches!(input, LocalSlamInput::Range(_)))
        .count();
    let mut group = c.benchmark_group("local_slam_2d");
    group.throughput(Throughput::Elements(num_scans as u64));
    for (name, overrides) in [
        ("probability_grid_imu", ""),
        (
            "probability_grid_imu_online_correlative",
            "TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true\n",
        ),
        (
            "tsdf_imu",
            "TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.grid_type = \"TSDF\"\n\
             TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.range_data_inserter_type = \
             \"TSDF_INSERTER_2D\"\n",
        ),
    ] {
        let options = local_trajectory_builder_options_2d(overrides);
        // Every scan yields a result (the IMU initializes the extrapolator).
        assert_eq!(
            common::run_local_slam_2d(&options, inputs.clone()),
            num_scans
        );
        group.bench_function(BenchmarkId::new(name, format!("{num_scans}_scans")), |b| {
            b.iter_batched(
                || inputs.clone(),
                |inputs| black_box(common::run_local_slam_2d(&options, inputs)),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = local_slam_2d
}
criterion_main!(benches);
