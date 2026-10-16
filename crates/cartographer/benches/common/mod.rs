//! Shared, deterministic input generation for the criterion benchmarks.
//!
//! Everything here runs outside the timed closures. All data comes from the
//! `cartographer-sim` simulator with fixed seeds, so every run benchmarks
//! exactly the same inputs.

#![allow(dead_code)] // Each bench target uses a different subset.

use std::sync::Arc;
use std::time::Duration;

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use cartographer::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
use cartographer::mapping::proto::LocalTrajectoryBuilderOptions2D;
use cartographer::mapping::scan_matching::RotationalScanMatcher;
use cartographer::mapping::submaps::{SharedSubmap, Submap};
use cartographer::mapping::three_d::submap_3d::{
    ActiveSubmaps3D, Submap3D, create_submaps_options_3d,
};
use cartographer::mapping::trajectory_node::TrajectoryNodeData;
use cartographer::mapping::two_d::submap_2d::Submap2D;
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::internal::voxel_filter::{
    adaptive_voxel_filter, create_adaptive_voxel_filter_options, voxel_filter,
};
use cartographer::sensor::point_cloud::{PointCloud, transform_point_cloud};
use cartographer::sensor::range_data::RangeData;
use cartographer::sensor::rangefinder_point::{RangefinderPoint, TimedRangefinderPoint};
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::{Rigid2d, Rigid3d};
use cartographer::transform::transform::project_2d;
use cartographer_sim::{
    Dataset, DatasetBuilder, ImuConfig, Lidar2dConfig, Lidar3dConfig, MissHandling, ScanTiming,
    SensorData, Trajectory, World2d, World3d,
};
use criterion::Criterion;
use nalgebra::{Isometry3, Point3, Quaternion, Vector3};

/// Criterion configuration shared by all benches: few samples and short
/// measurement windows so `cargo bench -p cartographer` finishes in minutes.
pub fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .without_plots()
}

/// Offset of the simulated time axis (simulated times start at 0 s).
pub fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

/// Evaluates Lua `code` with the shipped configuration files available for
/// `include`.
pub fn lua(code: &str) -> LuaParameterDictionary {
    LuaParameterDictionary::new(
        code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    )
}

/// `include "<file>"` and return `<expression>` (e.g.
/// `POSE_GRAPH.optimization_problem`).
pub fn lua_from_file(file: &str, expression: &str) -> LuaParameterDictionary {
    lua(&format!("include \"{file}\"\nreturn {expression}"))
}

pub fn to_rigid3d(pose: &Isometry3<f64>) -> Rigid3d {
    let q = pose.rotation.quaternion();
    Rigid3d::new(pose.translation.vector, Quaternion::new(q.w, q.i, q.j, q.k))
}

/// Applies a planar perturbation `(dx, dy, dtheta)` to `pose`.
pub fn perturb_2d(pose: &Rigid2d, dx: f64, dy: f64, dtheta: f64) -> Rigid2d {
    Rigid2d::new(
        pose.translation() + nalgebra::Vector2::new(dx, dy),
        pose.rotation().angle() + dtheta,
    )
}

/// Pre-converted local SLAM input.
#[derive(Clone)]
pub enum LocalSlamInput {
    Range(TimedPointCloudData),
    Imu(ImuData),
}

/// `World2d::rect_room` + `Trajectory::room_wander` at 0.5 m/s with a 720-beam
/// 10 Hz LiDAR (`"scan"`, 1 cm noise, rotating) and a 200 Hz IMU (`"imu"`),
/// like `tests/local_trajectory_builder_2d_sim.rs`.
pub fn room_dataset_2d(duration: f64, seed: u64) -> Dataset {
    DatasetBuilder::new(Trajectory::room_wander(3.0, 0.5), World2d::rect_room())
        .seed(seed)
        .time_range(0.0, duration)
        .lidar_2d("scan", Lidar2dConfig::default())
        .imu("imu", ImuConfig::default())
        .build()
}

/// Converts the dataset's LiDAR and IMU messages into Cartographer inputs (in
/// the tracking frame, like cartographer_ros).
pub fn local_slam_inputs(dataset: &Dataset) -> Vec<LocalSlamInput> {
    let mut inputs = Vec::new();
    for message in &dataset.messages {
        let extrinsic = dataset
            .sensor(&message.sensor_id)
            .expect("known sensor")
            .extrinsic;
        let time = to_time(message.time);
        match &message.data {
            SensorData::PointCloud(cloud) => {
                let extrinsic_f32 = extrinsic.cast::<f32>();
                let transform = |p| (extrinsic_f32 * Point3::from(p)).coords;
                inputs.push(LocalSlamInput::Range(TimedPointCloudData {
                    time,
                    origin: transform(cloud.origin),
                    ranges: cloud
                        .points
                        .iter()
                        .map(|p| TimedRangefinderPoint {
                            position: transform(p.position),
                            time: p.time,
                        })
                        .collect(),
                    intensities: cloud.intensities.clone(),
                }));
            }
            SensorData::Imu(imu) => {
                assert_eq!(extrinsic, Isometry3::identity());
                inputs.push(LocalSlamInput::Imu(ImuData {
                    time,
                    linear_acceleration: imu.linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                }));
            }
            _ => {}
        }
    }
    inputs
}

/// The shipped `trajectory_builder_2d.lua` with `overrides` applied.
pub fn local_trajectory_builder_options_2d(overrides: &str) -> LocalTrajectoryBuilderOptions2D {
    let code =
        format!("include \"trajectory_builder_2d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_2D");
    create_local_trajectory_builder_options_2d(&lua(&code))
}

/// Runs `inputs` through a fresh builder; returns the number of results.
pub fn run_local_slam_2d(
    options: &LocalTrajectoryBuilderOptions2D,
    inputs: Vec<LocalSlamInput>,
) -> usize {
    let mut builder = LocalTrajectoryBuilder2D::new(options, &["scan"]);
    let mut num_results = 0;
    for input in inputs {
        match input {
            LocalSlamInput::Range(data) => {
                if builder.add_range_data("scan", data).is_some() {
                    num_results += 1;
                }
            }
            LocalSlamInput::Imu(imu) => builder.add_imu_data(&imu),
        }
    }
    num_results
}

/// A realistic 2D submap (built by local SLAM on simulated data) and a scan
/// matched against it.
pub struct ScanMatchingFixture2D {
    /// The older (matching) active submap after the last insertion.
    pub submap: SharedSubmap<Submap2D>,
    /// Adaptive-voxel-filtered, gravity-aligned returns of the last node
    /// (`TrajectoryNode::Data::filtered_gravity_aligned_point_cloud`).
    pub point_cloud: PointCloud,
    /// Local-SLAM pose of that node (gravity aligned, 2D, in the grid frame).
    pub pose: Rigid2d,
    /// Number of nodes inserted into `submap`.
    pub num_range_data: i32,
}

/// Runs `LocalTrajectoryBuilder2D` (default 2D config, IMU on) over 8 s of
/// the room dataset and keeps the last inserted node and its submap.
pub fn scan_matching_fixture_2d(tsdf: bool) -> ScanMatchingFixture2D {
    let dataset = room_dataset_2d(8.0, 1);
    let overrides = if tsdf {
        "TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.grid_type = \"TSDF\"\n\
         TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.range_data_inserter_type = \
         \"TSDF_INSERTER_2D\"\n"
    } else {
        ""
    };
    let options = local_trajectory_builder_options_2d(overrides);
    let mut builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
    let mut last = None;
    for input in local_slam_inputs(&dataset) {
        match input {
            LocalSlamInput::Range(data) => {
                if let Some(result) = builder.add_range_data("scan", data)
                    && let Some(insertion_result) = result.insertion_result
                {
                    last = Some(insertion_result);
                }
            }
            LocalSlamInput::Imu(imu) => builder.add_imu_data(&imu),
        }
    }
    let insertion_result = last.expect("local SLAM inserted nodes");
    let constant_data: &TrajectoryNodeData = &insertion_result.constant_data;
    let gravity_aligned_pose = constant_data.local_pose
        * Rigid3d::from_rotation(
            constant_data
                .gravity_alignment
                .try_inverse()
                .expect("unit quaternion"),
        );
    let submap = Arc::clone(&insertion_result.insertion_submaps[0]);
    let num_range_data = submap.read().unwrap().num_range_data();
    ScanMatchingFixture2D {
        submap,
        point_cloud: constant_data.filtered_gravity_aligned_point_cloud.clone(),
        pose: project_2d(&gravity_aligned_pose),
        num_range_data,
    }
}

/// The first `num_scans` simulated 2D LiDAR scans (720 beams, 10 Hz,
/// instantaneous, misses as 5 m rays) along `Trajectory::room_wander` as
/// unfiltered `RangeData` in the world (= local) frame.
pub fn scans_2d(num_scans: usize) -> Vec<RangeData> {
    let dataset = DatasetBuilder::new(Trajectory::room_wander(1.0, 0.5), World2d::rect_room())
        .seed(3)
        .time_range(0.0, 0.1 * num_scans as f64 - 0.05)
        .lidar_2d(
            "scan",
            Lidar2dConfig {
                timing: ScanTiming::Instantaneous,
                miss: MissHandling::Return { range: 35.0 },
                ..Lidar2dConfig::default()
            },
        )
        .build();
    dataset
        .point_clouds("scan")
        .map(|(time, cloud)| {
            let pose = to_rigid3d(&dataset.ground_truth_at(time).pose).cast::<f32>();
            let mut returns = Vec::new();
            let mut misses = Vec::new();
            for (point, &is_miss) in cloud.points.iter().zip(&cloud.misses) {
                if is_miss {
                    // `missing_data_ray_length` = 5 m, like LocalTrajectoryBuilder2D.
                    misses.push(RangefinderPoint {
                        position: pose * (point.position.normalize() * 5.0),
                    });
                } else {
                    returns.push(RangefinderPoint {
                        position: pose * point.position,
                    });
                }
            }
            RangeData {
                origin: *pose.translation(),
                returns: PointCloud::from_points(returns),
                misses: PointCloud::from_points(misses),
            }
        })
        .collect()
}

/// A simulated 3D scan in the tracking frame plus its ground-truth pose.
pub struct Scan3D {
    pub time: f64,
    pub pose: Rigid3d,
    /// Raw returns with range in `[min_range, max_range]` of
    /// `trajectory_builder_3d.lua` (1 m, 60 m), tracking frame.
    pub raw: PointCloud,
}

/// VLP-16-like scans (16 rings, 900 columns, 2 cm noise, instantaneous) in
/// `World3d::room()` along `Trajectory::room_wander_3d` at 1 m/s.
pub fn scans_3d(num_scans: usize) -> Vec<Scan3D> {
    let dataset = DatasetBuilder::new(Trajectory::room_wander_3d(1.0, 1.0), World3d::room())
        .seed(11)
        .time_range(0.0, 0.1 * num_scans as f64 - 0.05)
        .lidar_3d(
            "points",
            Lidar3dConfig {
                timing: ScanTiming::Instantaneous,
                ..Lidar3dConfig::default()
            },
        )
        .build();
    dataset
        .point_clouds("points")
        .map(|(time, cloud)| {
            let points = cloud
                .points
                .iter()
                .filter(|p| {
                    let range = p.position.norm();
                    (1.0..=60.0).contains(&range)
                })
                .map(|p| RangefinderPoint {
                    position: p.position,
                })
                .collect();
            Scan3D {
                time,
                pose: to_rigid3d(&dataset.ground_truth_at(time).pose),
                raw: PointCloud::from_points(points),
            }
        })
        .collect()
}

/// `voxel_filter_size` of `trajectory_builder_3d.lua`.
pub const VOXEL_FILTER_SIZE_3D: f32 = 0.15;
/// `rotational_histogram_size` of `trajectory_builder_3d.lua`.
pub const ROTATIONAL_HISTOGRAM_SIZE: i32 = 120;

/// A 3D submap built from ground-truth-posed simulated scans and a query
/// scan (not inserted) prepared like `LocalTrajectoryBuilder3D` does.
pub struct ScanMatchingFixture3D {
    pub submap: SharedSubmap<Submap3D>,
    /// Ground-truth pose of the query scan in the submap frame.
    pub pose_in_submap: Rigid3d,
    /// Ground-truth pose of the query scan in the local frame.
    pub pose_in_local: Rigid3d,
    /// Node data of the query scan (clouds in the tracking frame).
    pub constant_data: TrajectoryNodeData,
}

pub fn scan_matching_fixture_3d() -> ScanMatchingFixture3D {
    let scans = scans_3d(81);
    let submaps_options = create_submaps_options_3d(&lua_from_file(
        "trajectory_builder_3d.lua",
        "TRAJECTORY_BUILDER_3D.submaps",
    ));
    let high_resolution_options = create_adaptive_voxel_filter_options(&lua_from_file(
        "trajectory_builder_3d.lua",
        "TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter",
    ));
    let low_resolution_options = create_adaptive_voxel_filter_options(&lua_from_file(
        "trajectory_builder_3d.lua",
        "TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter",
    ));
    let mut active_submaps = ActiveSubmaps3D::new(&submaps_options);
    let (query, mapped) = scans.split_last().expect("scans");
    let node_data = |scan: &Scan3D| {
        let filtered = voxel_filter(&scan.raw, VOXEL_FILTER_SIZE_3D);
        let high_resolution_point_cloud =
            adaptive_voxel_filter(&filtered, &high_resolution_options);
        let low_resolution_point_cloud = adaptive_voxel_filter(&filtered, &low_resolution_options);
        // The ground-truth orientation is the gravity alignment (the local
        // frame is the world frame, z up), so local_from_gravity_aligned is
        // the identity.
        let gravity_alignment = *scan.pose.rotation();
        // Like LocalTrajectoryBuilder3D: the histogram of the voxel-filtered
        // returns in the gravity-aligned frame.
        let histogram = RotationalScanMatcher::compute_histogram(
            &transform_point_cloud(
                &filtered,
                &Rigid3d::from_rotation(gravity_alignment).cast::<f32>(),
            ),
            ROTATIONAL_HISTOGRAM_SIZE,
        );
        (
            filtered,
            TrajectoryNodeData {
                time: to_time(scan.time),
                gravity_alignment,
                filtered_gravity_aligned_point_cloud: PointCloud::new(),
                high_resolution_point_cloud,
                low_resolution_point_cloud,
                rotational_scan_matcher_histogram: histogram,
                local_pose: scan.pose,
            },
        )
    };
    for scan in mapped {
        let (filtered, data) = node_data(scan);
        let pose = scan.pose.cast::<f32>();
        let range_data = RangeData {
            origin: *pose.translation(),
            returns: transform_point_cloud(&filtered, &pose),
            misses: PointCloud::new(),
        };
        active_submaps.insert_data(
            &range_data,
            &Quaternion::identity(),
            &data.rotational_scan_matcher_histogram,
        );
    }
    let submap = Arc::clone(&active_submaps.submaps()[0]);
    let submap_pose = submap.read().unwrap().local_pose();
    let (_, constant_data) = node_data(query);
    ScanMatchingFixture3D {
        pose_in_submap: submap_pose.inverse() * query.pose,
        pose_in_local: query.pose,
        submap,
        constant_data,
    }
}

/// Small 3D perturbation: translation `t` and yaw `yaw`.
pub fn perturb_3d(pose: &Rigid3d, t: Vector3<f64>, yaw: f64) -> Rigid3d {
    Rigid3d::from_translation(t)
        * *pose
        * Rigid3d::from_rotation(
            cartographer::transform::rigid_transform::AngleAxis::new(yaw, Vector3::z())
                .to_quaternion(),
        )
}
