//! End-to-end test of `LocalTrajectoryBuilder2D` on simulated data (there is
//! no C++ test for it; see the TODO in `local_trajectory_builder_2d.h`).
//!
//! A 2D LiDAR (10 Hz, 720 beams, 1 cm range noise, rotating scan with motion
//! distortion), optionally an IMU (200 Hz, noisy) and odometry (50 Hz, 2 %
//! noise) move through `World2d::rect_room()` / `World2d::loop_corridor()`.
//! The builder runs with the shipped `trajectory_builder_2d.lua` defaults,
//! adjusted like a cartographer_ros configuration (`use_imu_data`, ...).
//! Local poses are compared with ground truth after a rigid 2D alignment
//! (Cartographer's local frame is anchored at the first tracking pose).

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::LuaParameterDictionary;
use cartographer::common::time::{Time, from_seconds};
use cartographer::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use cartographer::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
use cartographer::mapping::proto::LocalTrajectoryBuilderOptions2D;
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::odometry_data::OdometryData;
use cartographer::sensor::rangefinder_point::TimedRangefinderPoint;
use cartographer::sensor::timed_point_cloud_data::TimedPointCloudData;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_sim::metrics::{Alignment, Association, RpeDelta, TimedPose, associate, ate, rpe};
use cartographer_sim::{
    DatasetBuilder, ImuConfig, Lidar2dConfig, OdometryConfig, SensorData, Trajectory, World2d,
};
use nalgebra::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion};

/// Offset of the simulated time axis (simulated times start at 0 s).
fn to_time(seconds: f64) -> Time {
    Time::from_universal(1_000_000_000_000) + from_seconds(seconds)
}

fn to_seconds_since_base(time: Time) -> f64 {
    (time - to_time(0.0)).to_seconds()
}

fn to_rigid3d(pose: &Isometry3<f64>) -> Rigid3d {
    let q = pose.rotation.quaternion();
    Rigid3d::new(pose.translation.vector, Quaternion::new(q.w, q.i, q.j, q.k))
}

fn to_isometry(pose: &Rigid3d) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(*pose.translation()),
        UnitQuaternion::from_quaternion(*pose.rotation()),
    )
}

/// Loads the shipped `trajectory_builder_2d.lua` and applies `overrides`
/// (Lua statements on `TRAJECTORY_BUILDER_2D`, like a cartographer_ros
/// configuration).
fn load_options(overrides: &str) -> LocalTrajectoryBuilderOptions2D {
    let code =
        format!("include \"trajectory_builder_2d.lua\"\n{overrides}\nreturn TRAJECTORY_BUILDER_2D");
    let dictionary = LuaParameterDictionary::new(
        &code,
        Box::new(ConfigurationFileResolver::new(Vec::<String>::new())),
    );
    create_local_trajectory_builder_options_2d(&dictionary)
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    /// `rect_room` + `room_wander`, else `loop_corridor` + `corridor_loops`.
    room: bool,
    use_imu: bool,
    use_odometry: bool,
    use_online_correlative_scan_matching: bool,
    /// TSDF submaps instead of probability grids.
    tsdf: bool,
    /// Constant speed of the trajectory (m/s), from t = 0 on.
    speed: f64,
    duration: f64,
    /// Results before this time are excluded from the metrics.
    warm_up: f64,
    seed: u64,
    /// Additional Lua overrides.
    extra_overrides: &'static str,
}

impl Default for Scenario {
    fn default() -> Self {
        Self {
            room: true,
            use_imu: true,
            use_odometry: false,
            use_online_correlative_scan_matching: false,
            tsdf: false,
            speed: 0.5,
            duration: 25.0,
            warm_up: 0.0,
            seed: 1,
            extra_overrides: "",
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // Fields are printed.
struct Outcome {
    num_scans: usize,
    num_results: usize,
    num_insertions: usize,
    first_result_time: f64,
    ate_rmse: f64,
    ate_max: f64,
    rpe_translation_rmse: f64,
    rpe_translation_max: f64,
    rpe_rotation_rmse: f64,
    rpe_rotation_max: f64,
}

fn run(scenario: Scenario) -> Outcome {
    let (trajectory, world) = if scenario.room {
        (
            Trajectory::room_wander(3.0, scenario.speed),
            World2d::rect_room(),
        )
    } else {
        (
            Trajectory::corridor_loops(1.0, scenario.speed),
            World2d::loop_corridor(),
        )
    };
    let mut builder = DatasetBuilder::new(trajectory, world)
        .seed(scenario.seed)
        .time_range(0.0, scenario.duration)
        .lidar_2d("scan", Lidar2dConfig::default());
    if scenario.use_imu {
        builder = builder.imu("imu", ImuConfig::default());
    }
    if scenario.use_odometry {
        builder = builder.odometry("odom", OdometryConfig::default());
    }
    let dataset = builder.build();

    let mut overrides = format!(
        "TRAJECTORY_BUILDER_2D.use_imu_data = {}\n\
         TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = {}\n",
        scenario.use_imu, scenario.use_online_correlative_scan_matching
    );
    if scenario.tsdf {
        overrides.push_str(
            "TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.grid_type = \"TSDF\"\n\
             TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.range_data_inserter_type = \
             \"TSDF_INSERTER_2D\"\n",
        );
    }
    overrides.push_str(scenario.extra_overrides);
    let options = load_options(&overrides);
    let mut local_trajectory_builder = LocalTrajectoryBuilder2D::new(&options, &["scan"]);
    let mut estimates = Vec::new();
    let mut num_scans = 0;
    let mut num_insertions = 0;
    for message in &dataset.messages {
        let extrinsic = dataset
            .sensor(&message.sensor_id)
            .expect("known sensor")
            .extrinsic;
        let time = to_time(message.time);
        match &message.data {
            SensorData::PointCloud(cloud) => {
                num_scans += 1;
                // Points into the tracking frame (cartographer_ros applies
                // the sensor extrinsic the same way).
                let extrinsic_f32 = extrinsic.cast::<f32>();
                let transform = |p| (extrinsic_f32 * Point3::from(p)).coords;
                let data = TimedPointCloudData {
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
                };
                if let Some(result) = local_trajectory_builder.add_range_data("scan", data) {
                    if result.insertion_result.is_some() {
                        num_insertions += 1;
                    }
                    estimates.push(TimedPose::new(
                        to_seconds_since_base(result.time),
                        to_isometry(&result.local_pose),
                    ));
                }
            }
            SensorData::Imu(imu) => {
                // The IMU is mounted at the tracking frame in these scenarios.
                assert_eq!(extrinsic, Isometry3::identity());
                local_trajectory_builder.add_imu_data(&ImuData {
                    time,
                    linear_acceleration: imu.linear_acceleration,
                    angular_velocity: imu.angular_velocity,
                });
            }
            SensorData::Odometry(odometry) => {
                local_trajectory_builder.add_odometry_data(&OdometryData {
                    time,
                    pose: to_rigid3d(&(odometry.pose * extrinsic.inverse())),
                });
            }
            _ => {}
        }
    }

    let reference = TimedPose::from_states(&dataset.ground_truth);
    let mut pairs = associate(
        &reference,
        &estimates,
        Association::Interpolate { max_gap: 0.05 },
    );
    assert!(
        pairs.len() + 1 >= estimates.len(),
        "{} of {} estimates associated",
        pairs.len(),
        estimates.len()
    );
    pairs.retain(|pair| pair.time >= scenario.warm_up);
    let ate_result = ate(&pairs, Alignment::Rigid2d);
    let rpe_result = rpe(&pairs, RpeDelta::Distance(1.0));
    Outcome {
        num_scans,
        num_results: estimates.len(),
        num_insertions,
        first_result_time: estimates.first().map_or(f64::NAN, |e| e.time),
        ate_rmse: ate_result.translation.rmse,
        ate_max: ate_result.translation.max,
        rpe_translation_rmse: rpe_result.translation.rmse,
        rpe_translation_max: rpe_result.translation.max,
        rpe_rotation_rmse: rpe_result.rotation.rmse,
        rpe_rotation_max: rpe_result.rotation.max,
    }
}

/// Accuracy thresholds; see [`TIGHT`] and [`RELAXED`].
struct Thresholds {
    /// RPE over 1 m of travel, translation (m).
    rpe_translation_rmse: f64,
    rpe_translation_max: f64,
    /// RPE over 1 m of travel, rotation (degrees).
    rpe_rotation_rmse_deg: f64,
    rpe_rotation_max_deg: f64,
    /// ATE after rigid 2D alignment (m).
    ate_rmse: f64,
    ate_max: f64,
}

/// For runs with a good motion prior (IMU and/or odometry, or the online
/// correlative scan matcher). Justification: submaps have 5 cm cells and the
/// LiDAR has 1 cm range noise; the bicubic Ceres matcher is sub-cell
/// accurate, so the relative error over 1 m (5-15 scans) must stay below
/// half a cell on average (2 cm) and below one cell (5 cm) at worst; heading
/// errors over 1 m below 0.5 deg RMSE / 1.5 deg max (at 15 m range 1.5 deg
/// already displaces points by 0.4 m). Local SLAM has no loop closure, but
/// over 25 s the drift in these feature-rich worlds stays within about one
/// cell: ATE RMSE < 2.5 cm, max < 6 cm. Observed values are 1.5-8x smaller.
const TIGHT: Thresholds = Thresholds {
    rpe_translation_rmse: 0.02,
    rpe_translation_max: 0.05,
    rpe_rotation_rmse_deg: 0.5,
    rpe_rotation_max_deg: 1.5,
    ate_rmse: 0.025,
    ate_max: 0.06,
};

/// Without IMU and odometry the angular velocity is only estimated from past
/// poses (`pose_queue_duration` = 1 ms, i.e. the last two poses), so turns
/// are predicted worse and occasional scans are off by about two cells:
/// RPE RMSE < 3 cm (max < 12 cm), rotation max < 3 deg, ATE RMSE < 3 cm
/// (max < 10 cm).
const RELAXED: Thresholds = Thresholds {
    rpe_translation_rmse: 0.03,
    rpe_translation_max: 0.12,
    rpe_rotation_rmse_deg: 0.5,
    rpe_rotation_max_deg: 3.0,
    ate_rmse: 0.03,
    ate_max: 0.10,
};

fn check_accuracy(name: &str, outcome: &Outcome, thresholds: &Thresholds) {
    println!("{name}: {outcome:?}");
    let checks = [
        (
            "rpe_translation_rmse",
            outcome.rpe_translation_rmse,
            thresholds.rpe_translation_rmse,
        ),
        (
            "rpe_translation_max",
            outcome.rpe_translation_max,
            thresholds.rpe_translation_max,
        ),
        (
            "rpe_rotation_rmse",
            outcome.rpe_rotation_rmse,
            thresholds.rpe_rotation_rmse_deg.to_radians(),
        ),
        (
            "rpe_rotation_max",
            outcome.rpe_rotation_max,
            thresholds.rpe_rotation_max_deg.to_radians(),
        ),
        ("ate_rmse", outcome.ate_rmse, thresholds.ate_rmse),
        ("ate_max", outcome.ate_max, thresholds.ate_max),
    ];
    for (metric, value, threshold) in checks {
        assert!(
            value < threshold,
            "{name}: {metric} = {value} >= {threshold}\n{outcome:?}"
        );
    }
}

/// The simulated platform moves at constant speed from t = 0 on, while the
/// extrapolator starts with zero velocity (Cartographer assumes a start at
/// rest). Without odometry the first ~2 s are a transient in which the
/// velocity estimate converges (the Ceres translation prior pulls towards
/// the zero-velocity prediction), so they are excluded from the metrics.
/// At 1 m/s or more this cold start fails entirely without odometry (the
/// first 10 cm jump is outside the matcher's basin), which is why the
/// odometry-free scenarios use 0.5 m/s.
const WARM_UP: f64 = 3.0;

fn check_counts(outcome: &Outcome, use_imu: bool) {
    if use_imu {
        // The extrapolator is initialized by the first IMU sample (t = 0,
        // before the first scan ends), so every scan yields a result.
        assert_eq!(outcome.num_results, outcome.num_scans, "{outcome:?}");
        assert!(outcome.first_result_time < 0.15, "{outcome:?}");
    } else {
        // The extrapolator is initialized at the time of the first scan,
        // whose earlier points lie before the first pose: the first scan is
        // dropped, like in C++.
        assert_eq!(outcome.num_results, outcome.num_scans - 1, "{outcome:?}");
        assert!(outcome.first_result_time > 0.15, "{outcome:?}");
    }
    // The motion filter drops some, but not all, nodes.
    assert!(outcome.num_insertions > 20, "{outcome:?}");
    assert!(outcome.num_insertions < outcome.num_results, "{outcome:?}");
}

#[test]
fn room_with_imu() {
    let outcome = run(Scenario {
        warm_up: WARM_UP,
        ..Default::default()
    });
    check_counts(&outcome, true);
    check_accuracy("room_with_imu", &outcome, &TIGHT);
}

#[test]
fn room_without_imu() {
    let outcome = run(Scenario {
        use_imu: false,
        warm_up: WARM_UP,
        ..Default::default()
    });
    check_counts(&outcome, false);
    check_accuracy("room_without_imu", &outcome, &RELAXED);
}

#[test]
fn room_without_imu_with_online_correlative_scan_matching() {
    let outcome = run(Scenario {
        use_imu: false,
        use_online_correlative_scan_matching: true,
        warm_up: WARM_UP,
        ..Default::default()
    });
    check_counts(&outcome, false);
    check_accuracy(
        "room_without_imu_with_online_correlative_scan_matching",
        &outcome,
        &TIGHT,
    );
}

#[test]
fn room_with_imu_and_odometry() {
    let outcome = run(Scenario {
        use_odometry: true,
        speed: 1.0,
        ..Default::default()
    });
    check_counts(&outcome, true);
    check_accuracy("room_with_imu_and_odometry", &outcome, &TIGHT);
}

#[test]
fn corridor_with_odometry_without_imu() {
    let outcome = run(Scenario {
        room: false,
        use_imu: false,
        use_odometry: true,
        speed: 1.5,
        ..Default::default()
    });
    check_counts(&outcome, false);
    check_accuracy("corridor_with_odometry_without_imu", &outcome, &TIGHT);
}

#[test]
fn room_tsdf_with_imu_and_odometry() {
    let outcome = run(Scenario {
        use_odometry: true,
        tsdf: true,
        speed: 1.0,
        ..Default::default()
    });
    check_counts(&outcome, true);
    check_accuracy("room_tsdf_with_imu_and_odometry", &outcome, &TIGHT);
}

/// With TSDF submaps the match residuals are about the signed distance (a
/// few cm), so the default priors (translation_weight 10, tuned for
/// probability grids) dominate and pin the pose to the prediction; without
/// odometry a TSDF configuration needs weaker priors.
#[test]
fn room_tsdf_with_imu_reweighted_priors() {
    let outcome = run(Scenario {
        tsdf: true,
        warm_up: WARM_UP,
        extra_overrides: "TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 0.1\n\
                          TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 1.\n",
        ..Default::default()
    });
    check_counts(&outcome, true);
    check_accuracy("room_tsdf_with_imu_reweighted_priors", &outcome, &TIGHT);
}
