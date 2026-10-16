//! [`SlamConfig`]: typed, documented configuration on top of Cartographer's
//! option protos (see [`SlamConfig`] for the knob → proto field table).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::error::{SlamError, catch_check_failure};
use super::types::Mode;
use crate::common::configuration_file_resolver::ConfigurationFileResolver;
use crate::common::lua_parameter_dictionary::{FileResolver, LuaParameterDictionary};
use crate::mapping::map_builder_interface::create_map_builder_options;
use crate::mapping::proto::{
    self, LocalTrajectoryBuilderOptions2D, LocalTrajectoryBuilderOptions3D, MapBuilderOptions,
    TrajectoryBuilderOptions,
};
use crate::mapping::trajectory_builder_interface::create_trajectory_builder_options;

/// Motion filter: a scan becomes a pose graph node only if the tracking
/// frame moved or rotated enough or enough time passed since the last node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionFilterConfig {
    /// Maximum time between nodes (s).
    pub max_time_seconds: f64,
    /// Minimum translation for a new node (m).
    pub max_distance_meters: f64,
    /// Minimum rotation for a new node (rad).
    pub max_angle_radians: f64,
}

/// Weights of the local (Ceres) scan matcher relative to the map fit.
/// Larger values trust the motion prediction (odometry/IMU extrapolation)
/// more.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanMatcherConfig {
    /// Penalty for translating away from the predicted pose.
    pub translation_weight: f64,
    /// Penalty for rotating away from the predicted pose.
    pub rotation_weight: f64,
}

/// Loop closure (global SLAM): background search for constraints between
/// nodes and finished submaps, followed by a sparse pose graph optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopClosureConfig {
    /// Run global SLAM. `false` writes `optimize_every_n_nodes = 0`, which
    /// disables constraint search and optimization (pure odometry-like local
    /// SLAM).
    pub enabled: bool,
    /// Optimize the pose graph every this many nodes (when enabled).
    pub optimize_every_n_nodes: u32,
    /// Minimum scan matching score (0..1) to accept a loop closure.
    pub min_score: f64,
    /// Fraction of nodes for which a constraint search against nearby
    /// submaps is run.
    pub sampling_ratio: f64,
    /// Search only submaps closer than this (m).
    pub max_constraint_distance: f64,
    /// Weight of the translation of loop closure constraints.
    pub translation_weight: f64,
    /// Weight of the rotation of loop closure constraints.
    pub rotation_weight: f64,
    /// Solver iterations of the final optimization ([`Slam::finish`](super::Slam::finish)).
    pub max_num_final_iterations: u32,
}

/// Global localization: constraint search of nodes against submaps of
/// *other* trajectories without a pose prior (e.g. localizing in a loaded
/// map).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalLocalizationConfig {
    /// Fraction of nodes for which a global search is run. The shipped
    /// default (0.003) is tuned for long runs; raise it (e.g. 0.1-0.3) for
    /// faster relocalization at a higher CPU cost.
    pub sampling_ratio: f64,
    /// Minimum score (0..1) to accept a global match.
    pub min_score: f64,
    /// Search globally again if the trajectories were not connected for this
    /// long (s).
    pub search_after_seconds: f64,
}

/// Weights of the pose graph optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationConfig {
    /// Weight of consecutive local SLAM poses (translation).
    pub local_slam_pose_translation_weight: f64,
    /// Weight of consecutive local SLAM poses (rotation).
    pub local_slam_pose_rotation_weight: f64,
    /// Weight of odometry (translation).
    pub odometry_translation_weight: f64,
    /// Weight of odometry (rotation).
    pub odometry_rotation_weight: f64,
}

/// Pure localization: the new trajectory only keeps its latest submaps
/// (older ones are trimmed), so a long localization session does not grow
/// the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PureLocalizationConfig {
    /// Submaps of the localization trajectory to keep (at least 2).
    pub max_submaps_to_keep: u32,
}

type AdvancedFn = Arc<dyn Fn(&mut MapBuilderOptions, &mut TrajectoryBuilderOptions) + Send + Sync>;

/// Typed configuration of a [`Slam`](super::Slam) instance.
///
/// Create it with [`SlamConfig::new`] (Cartographer's shipped defaults) or
/// [`SlamConfig::from_lua`], change the public fields, and optionally edit
/// the raw option protos with [`SlamConfig::advanced`]. Units are SI (m, s,
/// rad).
///
/// The defaults are **not** duplicated in Rust: a `SlamConfig` is read from
/// option protos generated from Lua (by default the configuration files
/// shipped with Cartographer: `map_builder.lua`, `pose_graph.lua`,
/// `trajectory_builder.lua`, `trajectory_builder_2d.lua`,
/// `trajectory_builder_3d.lua`), and writing it back
/// ([`SlamConfig::to_protos`]) only touches the fields listed below, so
/// everything else keeps its Lua value.
///
/// | Knob | 2D proto field (`TrajectoryBuilderOptions.trajectory_builder_2d_options.*`) | 3D proto field (`TrajectoryBuilderOptions.trajectory_builder_3d_options.*`) |
/// |---|---|---|
/// | `resolution` | `submaps_options.grid_options_2d.resolution` | `submaps_options.high_resolution` |
/// | `min_range`, `max_range` | `min_range`, `max_range` | `min_range`, `max_range` |
/// | `min_z`, `max_z` (2D) | `min_z`, `max_z` | – |
/// | `missing_data_ray_length` (2D) | `missing_data_ray_length` | – |
/// | `low_resolution` (3D) | – | `submaps_options.low_resolution` |
/// | `high_resolution_max_range` (3D) | – | `submaps_options.high_resolution_max_range` |
/// | `num_accumulated_range_data` | `num_accumulated_range_data` | `num_accumulated_range_data` |
/// | `voxel_filter_size` | `voxel_filter_size` | `voxel_filter_size` |
/// | `submap_num_range_data` | `submaps_options.num_range_data` | `submaps_options.num_range_data` |
/// | `use_online_correlative_scan_matching` | `use_online_correlative_scan_matching` | `use_online_correlative_scan_matching` |
/// | `scan_matcher.translation_weight` / `.rotation_weight` | `ceres_scan_matcher_options.translation_weight` / `.rotation_weight` | same |
/// | `motion_filter.*` | `motion_filter_options.{max_time_seconds, max_distance_meters, max_angle_radians}` | same |
/// | `imu_gravity_time_constant` | `imu_gravity_time_constant` **and** `pose_extrapolator_options.constant_velocity.imu_gravity_time_constant` | same |
///
/// | Knob | Proto field |
/// |---|---|
/// | `num_background_threads` | `MapBuilderOptions.num_background_threads` |
/// | `loop_closure.enabled` / `.optimize_every_n_nodes` | `MapBuilderOptions.pose_graph_options.optimize_every_n_nodes` (`0` = disabled) |
/// | `loop_closure.min_score` | `pose_graph_options.constraint_builder_options.min_score` |
/// | `loop_closure.sampling_ratio` | `pose_graph_options.constraint_builder_options.sampling_ratio` |
/// | `loop_closure.max_constraint_distance` | `pose_graph_options.constraint_builder_options.max_constraint_distance` |
/// | `loop_closure.translation_weight` / `.rotation_weight` | `pose_graph_options.constraint_builder_options.loop_closure_translation_weight` / `loop_closure_rotation_weight` |
/// | `loop_closure.max_num_final_iterations` | `pose_graph_options.max_num_final_iterations` |
/// | `global_localization.sampling_ratio` | `pose_graph_options.global_sampling_ratio` |
/// | `global_localization.min_score` | `pose_graph_options.constraint_builder_options.global_localization_min_score` |
/// | `global_localization.search_after_seconds` | `pose_graph_options.global_constraint_search_after_n_seconds` |
/// | `optimization.*` | `pose_graph_options.optimization_problem_options.{odometry,local_slam_pose}_{translation,rotation}_weight` |
/// | `pure_localization` | `TrajectoryBuilderOptions.pure_localization_trimmer.max_submaps_to_keep` (`None` = unset) |
///
/// Derived (not knobs): `MapBuilderOptions.use_trajectory_builder_2d/3d`
/// from the [`Mode`], and in 2D `trajectory_builder_2d_options.use_imu_data`
/// from whether an IMU was declared on the [`SlamBuilder`](super::SlamBuilder).
/// [`SlamConfig::advanced`] edits run last and may change anything.
#[derive(Clone)]
pub struct SlamConfig {
    mode: Mode,
    base_map_builder_options: MapBuilderOptions,
    base_trajectory_builder_options: TrajectoryBuilderOptions,
    advanced: Vec<AdvancedFn>,

    /// Grid cell size (m). 2D: the probability grid; 3D: the high-resolution
    /// hybrid grid. Shipped defaults: 0.05 (2D), 0.10 (3D).
    pub resolution: f64,
    /// Returns closer than this are ignored (m).
    pub min_range: f32,
    /// Returns farther than this are ignored (2D: turned into free-space
    /// rays of `missing_data_ray_length`) (m).
    pub max_range: f32,
    /// 2D only: points below this height in the gravity-aligned tracking
    /// frame are ignored (m).
    pub min_z: f32,
    /// 2D only: points above this height are ignored (m).
    pub max_z: f32,
    /// 2D only: length of the free-space ray inserted for returns beyond
    /// `max_range` (m).
    pub missing_data_ray_length: f32,
    /// 3D only: resolution of the low-resolution hybrid grid used for
    /// coarse matching (m).
    pub low_resolution: f64,
    /// 3D only: only returns closer than this are inserted into the
    /// high-resolution grid (m).
    pub high_resolution_max_range: f64,
    /// Number of range messages accumulated into one scan before matching.
    /// Use the number of packets per revolution for sensors that split a
    /// revolution into several messages; 1 for full scans.
    pub num_accumulated_range_data: u32,
    /// Voxel filter edge length applied to each scan (m).
    pub voxel_filter_size: f32,
    /// Scans inserted into a submap before it is finished (submap size).
    pub submap_num_range_data: u32,
    /// Run a correlative search before Ceres scan matching (more robust
    /// without odometry/IMU, more CPU).
    pub use_online_correlative_scan_matching: bool,
    /// Local scan matcher weights.
    pub scan_matcher: ScanMatcherConfig,
    /// Node creation motion filter.
    pub motion_filter: MotionFilterConfig,
    /// Time constant (s) of the IMU gravity estimate (exponential
    /// averaging of the accelerometer).
    pub imu_gravity_time_constant: f64,
    /// Loop closure / global SLAM.
    pub loop_closure: LoopClosureConfig,
    /// Global localization against other trajectories.
    pub global_localization: GlobalLocalizationConfig,
    /// Pose graph optimization weights.
    pub optimization: OptimizationConfig,
    /// Threads for loop closure search and optimization.
    pub num_background_threads: u32,
    /// Trim old submaps of the new trajectory (pure localization). Set
    /// automatically by [`SlamBuilder::localize_in`](super::SlamBuilder::localize_in)
    /// if left `None`.
    pub pure_localization: Option<PureLocalizationConfig>,
}

impl fmt::Debug for SlamConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlamConfig")
            .field("mode", &self.mode)
            .field("resolution", &self.resolution)
            .field("min_range", &self.min_range)
            .field("max_range", &self.max_range)
            .field("min_z", &self.min_z)
            .field("max_z", &self.max_z)
            .field("missing_data_ray_length", &self.missing_data_ray_length)
            .field("low_resolution", &self.low_resolution)
            .field("high_resolution_max_range", &self.high_resolution_max_range)
            .field(
                "num_accumulated_range_data",
                &self.num_accumulated_range_data,
            )
            .field("voxel_filter_size", &self.voxel_filter_size)
            .field("submap_num_range_data", &self.submap_num_range_data)
            .field(
                "use_online_correlative_scan_matching",
                &self.use_online_correlative_scan_matching,
            )
            .field("scan_matcher", &self.scan_matcher)
            .field("motion_filter", &self.motion_filter)
            .field("imu_gravity_time_constant", &self.imu_gravity_time_constant)
            .field("loop_closure", &self.loop_closure)
            .field("global_localization", &self.global_localization)
            .field("optimization", &self.optimization)
            .field("num_background_threads", &self.num_background_threads)
            .field("pure_localization", &self.pure_localization)
            .field("num_advanced_edits", &self.advanced.len())
            .finish()
    }
}

fn to_u32(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

fn to_i32(value: u32, name: &str) -> Result<i32, SlamError> {
    i32::try_from(value)
        .map_err(|_| SlamError::InvalidConfig(format!("{name} = {value} is too large")))
}

impl SlamConfig {
    /// Cartographer's shipped defaults for `mode`.
    pub fn new(mode: Mode) -> Self {
        let dimension = match mode {
            Mode::TwoD => "2d",
            Mode::ThreeD => "3d",
        };
        let code = format!(
            "include \"map_builder.lua\"\n\
             include \"trajectory_builder.lua\"\n\
             MAP_BUILDER.use_trajectory_builder_{dimension} = true\n\
             return {{ map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }}"
        );
        Self::from_lua(&code, &[] as &[&str]).expect("the shipped configuration files are valid")
    }

    /// Loads the configuration from Lua code in the cartographer_ros format:
    /// the code returns a table with (at least) the keys `map_builder` and
    /// `trajectory_builder`; other keys (e.g. cartographer_ros' frame names)
    /// are ignored. The mode comes from `MAP_BUILDER.use_trajectory_builder_2d`
    /// / `_3d` (exactly one must be true).
    ///
    /// `include "file.lua"` searches `search_dirs` in order, then the
    /// configuration directory of this crate, then copies of the shipped
    /// files compiled into the binary, so `include "map_builder.lua"` and
    /// `include "trajectory_builder.lua"` always work.
    ///
    /// ```
    /// use cartographer::{Mode, SlamConfig};
    /// let config = SlamConfig::from_lua(
    ///     r#"
    ///     include "map_builder.lua"
    ///     include "trajectory_builder.lua"
    ///     MAP_BUILDER.use_trajectory_builder_2d = true
    ///     TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35
    ///     return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER,
    ///              tracking_frame = "base_link" }"#,
    ///     &[] as &[&str],
    /// )?;
    /// assert_eq!(config.mode(), Mode::TwoD);
    /// assert_eq!(config.submap_num_range_data, 35);
    /// # Ok::<(), cartographer::SlamError>(())
    /// ```
    pub fn from_lua<S: AsRef<str>>(code: &str, search_dirs: &[S]) -> Result<Self, SlamError> {
        let directories =
            ConfigurationFileResolver::new(search_dirs.iter().map(|dir| dir.as_ref().to_string()))
                .configuration_files_directories()
                .to_vec();
        // Run the user code in a function so that only the two option tables
        // are handed to the (strict, every-key-used-once) dictionary.
        let wrapped = format!(
            "local slam_user_options = (function()\n{code}\nend)()\n\
             return {{ map_builder = slam_user_options.map_builder, \
             trajectory_builder = slam_user_options.trajectory_builder }}"
        );
        let (map_builder_options, trajectory_builder_options) = catch_check_failure(move || {
            let dictionary = LuaParameterDictionary::new(
                &wrapped,
                Box::new(ShippedFilesResolver { directories }),
            );
            let map_builder_options =
                create_map_builder_options(&dictionary.get_dictionary("map_builder"));
            let trajectory_builder_options =
                create_trajectory_builder_options(&dictionary.get_dictionary("trajectory_builder"));
            (map_builder_options, trajectory_builder_options)
        })
        .map_err(|message| SlamError::InvalidConfig(format!("Lua configuration: {message}")))?;
        let mode = if map_builder_options.use_trajectory_builder_3d {
            Mode::ThreeD
        } else {
            Mode::TwoD
        };
        Ok(Self::from_protos(
            mode,
            map_builder_options,
            trajectory_builder_options,
        ))
    }

    /// Reads the knobs from option protos (which stay the base that
    /// [`to_protos`](Self::to_protos) writes into). `from_protos(m, a,
    /// b).to_protos()` returns `(a, b)` up to the derived fields.
    pub fn from_protos(
        mode: Mode,
        map_builder_options: MapBuilderOptions,
        trajectory_builder_options: TrajectoryBuilderOptions,
    ) -> Self {
        let tb2 = trajectory_builder_options
            .trajectory_builder_2d_options
            .unwrap_or_default();
        let tb3 = trajectory_builder_options
            .trajectory_builder_3d_options
            .clone()
            .unwrap_or_default();
        let pose_graph = map_builder_options
            .pose_graph_options
            .clone()
            .unwrap_or_default();
        let constraint_builder = pose_graph
            .constraint_builder_options
            .clone()
            .unwrap_or_default();
        let optimization = pose_graph.optimization_problem_options.unwrap_or_default();
        let submaps_2d = tb2.submaps_options.unwrap_or_default();
        let submaps_3d = tb3.submaps_options.unwrap_or_default();

        struct Common {
            resolution: f64,
            min_range: f32,
            max_range: f32,
            num_accumulated_range_data: i32,
            voxel_filter_size: f32,
            submap_num_range_data: i32,
            use_online_correlative_scan_matching: bool,
            ceres_weights: (f64, f64),
            motion_filter: proto::MotionFilterOptions,
            pose_extrapolator: proto::PoseExtrapolatorOptions,
        }
        let common = match mode {
            Mode::TwoD => {
                let ceres = tb2.ceres_scan_matcher_options.unwrap_or_default();
                Common {
                    resolution: f64::from(
                        submaps_2d.grid_options_2d.unwrap_or_default().resolution,
                    ),
                    min_range: tb2.min_range,
                    max_range: tb2.max_range,
                    num_accumulated_range_data: tb2.num_accumulated_range_data,
                    voxel_filter_size: tb2.voxel_filter_size,
                    submap_num_range_data: submaps_2d.num_range_data,
                    use_online_correlative_scan_matching: tb2.use_online_correlative_scan_matching,
                    ceres_weights: (ceres.translation_weight, ceres.rotation_weight),
                    motion_filter: tb2.motion_filter_options.unwrap_or_default(),
                    pose_extrapolator: tb2.pose_extrapolator_options.unwrap_or_default(),
                }
            }
            Mode::ThreeD => {
                let ceres = tb3.ceres_scan_matcher_options.unwrap_or_default();
                Common {
                    resolution: submaps_3d.high_resolution,
                    min_range: tb3.min_range,
                    max_range: tb3.max_range,
                    num_accumulated_range_data: tb3.num_accumulated_range_data,
                    voxel_filter_size: tb3.voxel_filter_size,
                    submap_num_range_data: submaps_3d.num_range_data,
                    use_online_correlative_scan_matching: tb3.use_online_correlative_scan_matching,
                    ceres_weights: (ceres.translation_weight, ceres.rotation_weight),
                    motion_filter: tb3.motion_filter_options.unwrap_or_default(),
                    pose_extrapolator: tb3.pose_extrapolator_options.unwrap_or_default(),
                }
            }
        };
        let imu_gravity_time_constant = common
            .pose_extrapolator
            .constant_velocity
            .map_or(0.0, |cv| cv.imu_gravity_time_constant);

        SlamConfig {
            mode,
            resolution: common.resolution,
            min_range: common.min_range,
            max_range: common.max_range,
            min_z: tb2.min_z,
            max_z: tb2.max_z,
            missing_data_ray_length: tb2.missing_data_ray_length,
            low_resolution: submaps_3d.low_resolution,
            high_resolution_max_range: submaps_3d.high_resolution_max_range,
            num_accumulated_range_data: to_u32(common.num_accumulated_range_data),
            voxel_filter_size: common.voxel_filter_size,
            submap_num_range_data: to_u32(common.submap_num_range_data),
            use_online_correlative_scan_matching: common.use_online_correlative_scan_matching,
            scan_matcher: ScanMatcherConfig {
                translation_weight: common.ceres_weights.0,
                rotation_weight: common.ceres_weights.1,
            },
            motion_filter: MotionFilterConfig {
                max_time_seconds: common.motion_filter.max_time_seconds,
                max_distance_meters: common.motion_filter.max_distance_meters,
                max_angle_radians: common.motion_filter.max_angle_radians,
            },
            imu_gravity_time_constant,
            loop_closure: LoopClosureConfig {
                enabled: pose_graph.optimize_every_n_nodes > 0,
                optimize_every_n_nodes: to_u32(pose_graph.optimize_every_n_nodes),
                min_score: constraint_builder.min_score,
                sampling_ratio: constraint_builder.sampling_ratio,
                max_constraint_distance: constraint_builder.max_constraint_distance,
                translation_weight: constraint_builder.loop_closure_translation_weight,
                rotation_weight: constraint_builder.loop_closure_rotation_weight,
                max_num_final_iterations: to_u32(pose_graph.max_num_final_iterations),
            },
            global_localization: GlobalLocalizationConfig {
                sampling_ratio: pose_graph.global_sampling_ratio,
                min_score: constraint_builder.global_localization_min_score,
                search_after_seconds: pose_graph.global_constraint_search_after_n_seconds,
            },
            optimization: OptimizationConfig {
                local_slam_pose_translation_weight: optimization.local_slam_pose_translation_weight,
                local_slam_pose_rotation_weight: optimization.local_slam_pose_rotation_weight,
                odometry_translation_weight: optimization.odometry_translation_weight,
                odometry_rotation_weight: optimization.odometry_rotation_weight,
            },
            num_background_threads: to_u32(map_builder_options.num_background_threads),
            pure_localization: trajectory_builder_options.pure_localization_trimmer.map(
                |trimmer| PureLocalizationConfig {
                    max_submaps_to_keep: to_u32(trimmer.max_submaps_to_keep),
                },
            ),
            base_map_builder_options: map_builder_options,
            base_trajectory_builder_options: trajectory_builder_options,
            advanced: Vec::new(),
        }
    }

    /// The SLAM mode (fixed at creation).
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Escape hatch: `edit` runs on the raw option protos after the typed
    /// knobs (and the sensor-derived fields) were written, so it can change
    /// any Cartographer parameter. Several edits run in registration order.
    ///
    /// ```
    /// use cartographer::{Mode, SlamConfig};
    /// let mut config = SlamConfig::new(Mode::TwoD);
    /// config.advanced(|_map_builder, trajectory_builder| {
    ///     let tb2 = trajectory_builder.trajectory_builder_2d_options.as_mut().unwrap();
    ///     tb2.adaptive_voxel_filter_options.as_mut().unwrap().max_length = 0.9;
    /// });
    /// let (_, tb) = config.to_protos()?;
    /// let filter = tb.trajectory_builder_2d_options.unwrap().adaptive_voxel_filter_options.unwrap();
    /// assert_eq!(filter.max_length, 0.9);
    /// # Ok::<(), cartographer::SlamError>(())
    /// ```
    pub fn advanced<F>(&mut self, edit: F) -> &mut Self
    where
        F: Fn(&mut MapBuilderOptions, &mut TrajectoryBuilderOptions) + Send + Sync + 'static,
    {
        self.advanced.push(Arc::new(edit));
        self
    }

    /// Checks the knobs for values Cartographer cannot work with.
    pub fn validate(&self) -> Result<(), SlamError> {
        fn check(condition: bool, message: impl FnOnce() -> String) -> Result<(), SlamError> {
            if condition {
                Ok(())
            } else {
                Err(SlamError::InvalidConfig(message()))
            }
        }
        let positive = |name: &str, value: f64| {
            check(value.is_finite() && value > 0.0, || {
                format!("{name} must be positive and finite, got {value}")
            })
        };
        let non_negative = |name: &str, value: f64| {
            check(value.is_finite() && value >= 0.0, || {
                format!("{name} must be non-negative and finite, got {value}")
            })
        };
        let ratio = |name: &str, value: f64| {
            check((0.0..=1.0).contains(&value), || {
                format!("{name} must be in [0, 1], got {value}")
            })
        };
        positive("resolution", self.resolution)?;
        non_negative("min_range", f64::from(self.min_range))?;
        check(
            self.max_range.is_finite() && self.max_range > self.min_range,
            || {
                format!(
                    "max_range ({}) must be greater than min_range ({})",
                    self.max_range, self.min_range
                )
            },
        )?;
        if self.mode == Mode::TwoD {
            check(self.max_z > self.min_z, || {
                format!(
                    "max_z ({}) must be greater than min_z ({})",
                    self.max_z, self.min_z
                )
            })?;
            positive(
                "missing_data_ray_length",
                f64::from(self.missing_data_ray_length),
            )?;
        } else {
            positive("low_resolution", self.low_resolution)?;
            positive("high_resolution_max_range", self.high_resolution_max_range)?;
        }
        check(self.num_accumulated_range_data >= 1, || {
            "num_accumulated_range_data must be at least 1".to_string()
        })?;
        check(self.submap_num_range_data >= 1, || {
            "submap_num_range_data must be at least 1".to_string()
        })?;
        positive("voxel_filter_size", f64::from(self.voxel_filter_size))?;
        positive(
            "scan_matcher.translation_weight",
            self.scan_matcher.translation_weight,
        )?;
        positive(
            "scan_matcher.rotation_weight",
            self.scan_matcher.rotation_weight,
        )?;
        non_negative(
            "motion_filter.max_time_seconds",
            self.motion_filter.max_time_seconds,
        )?;
        non_negative(
            "motion_filter.max_distance_meters",
            self.motion_filter.max_distance_meters,
        )?;
        non_negative(
            "motion_filter.max_angle_radians",
            self.motion_filter.max_angle_radians,
        )?;
        positive("imu_gravity_time_constant", self.imu_gravity_time_constant)?;
        if self.loop_closure.enabled {
            check(self.loop_closure.optimize_every_n_nodes >= 1, || {
                "loop_closure.optimize_every_n_nodes must be at least 1 when loop closure is \
                 enabled"
                    .to_string()
            })?;
        }
        ratio("loop_closure.min_score", self.loop_closure.min_score)?;
        ratio(
            "loop_closure.sampling_ratio",
            self.loop_closure.sampling_ratio,
        )?;
        non_negative(
            "loop_closure.max_constraint_distance",
            self.loop_closure.max_constraint_distance,
        )?;
        positive(
            "loop_closure.translation_weight",
            self.loop_closure.translation_weight,
        )?;
        positive(
            "loop_closure.rotation_weight",
            self.loop_closure.rotation_weight,
        )?;
        ratio(
            "global_localization.min_score",
            self.global_localization.min_score,
        )?;
        ratio(
            "global_localization.sampling_ratio",
            self.global_localization.sampling_ratio,
        )?;
        non_negative(
            "global_localization.search_after_seconds",
            self.global_localization.search_after_seconds,
        )?;
        for (name, value) in [
            (
                "optimization.local_slam_pose_translation_weight",
                self.optimization.local_slam_pose_translation_weight,
            ),
            (
                "optimization.local_slam_pose_rotation_weight",
                self.optimization.local_slam_pose_rotation_weight,
            ),
            (
                "optimization.odometry_translation_weight",
                self.optimization.odometry_translation_weight,
            ),
            (
                "optimization.odometry_rotation_weight",
                self.optimization.odometry_rotation_weight,
            ),
        ] {
            non_negative(name, value)?;
        }
        check(self.num_background_threads >= 1, || {
            "num_background_threads must be at least 1".to_string()
        })?;
        if let Some(pure_localization) = self.pure_localization {
            check(pure_localization.max_submaps_to_keep >= 2, || {
                "pure_localization.max_submaps_to_keep must be at least 2".to_string()
            })?;
        }
        Ok(())
    }

    /// Validates and produces the option protos: the base protos (from Lua)
    /// with the knobs written in, then the [`advanced`](Self::advanced)
    /// edits.
    pub fn to_protos(&self) -> Result<(MapBuilderOptions, TrajectoryBuilderOptions), SlamError> {
        self.to_protos_with(|_, _| {})
    }

    /// Like [`to_protos`](Self::to_protos), running `derive` between the
    /// knobs and the advanced edits.
    pub(crate) fn to_protos_with(
        &self,
        derive: impl FnOnce(&mut MapBuilderOptions, &mut TrajectoryBuilderOptions),
    ) -> Result<(MapBuilderOptions, TrajectoryBuilderOptions), SlamError> {
        self.validate()?;
        let mut map_builder_options = self.base_map_builder_options.clone();
        let mut trajectory_builder_options = self.base_trajectory_builder_options.clone();
        self.write_knobs(&mut map_builder_options, &mut trajectory_builder_options)?;
        derive(&mut map_builder_options, &mut trajectory_builder_options);
        for edit in &self.advanced {
            edit(&mut map_builder_options, &mut trajectory_builder_options);
        }
        Ok((map_builder_options, trajectory_builder_options))
    }

    fn write_knobs(
        &self,
        mb: &mut MapBuilderOptions,
        tb: &mut TrajectoryBuilderOptions,
    ) -> Result<(), SlamError> {
        mb.use_trajectory_builder_2d = self.mode == Mode::TwoD;
        mb.use_trajectory_builder_3d = self.mode == Mode::ThreeD;
        mb.num_background_threads = to_i32(self.num_background_threads, "num_background_threads")?;
        let pose_graph = mb.pose_graph_options.get_or_insert_with(Default::default);
        pose_graph.optimize_every_n_nodes = if self.loop_closure.enabled {
            to_i32(
                self.loop_closure.optimize_every_n_nodes,
                "loop_closure.optimize_every_n_nodes",
            )?
        } else {
            0
        };
        pose_graph.max_num_final_iterations = to_i32(
            self.loop_closure.max_num_final_iterations,
            "loop_closure.max_num_final_iterations",
        )?;
        pose_graph.global_sampling_ratio = self.global_localization.sampling_ratio;
        pose_graph.global_constraint_search_after_n_seconds =
            self.global_localization.search_after_seconds;
        let constraint_builder = pose_graph
            .constraint_builder_options
            .get_or_insert_with(Default::default);
        constraint_builder.min_score = self.loop_closure.min_score;
        constraint_builder.sampling_ratio = self.loop_closure.sampling_ratio;
        constraint_builder.max_constraint_distance = self.loop_closure.max_constraint_distance;
        constraint_builder.loop_closure_translation_weight = self.loop_closure.translation_weight;
        constraint_builder.loop_closure_rotation_weight = self.loop_closure.rotation_weight;
        constraint_builder.global_localization_min_score = self.global_localization.min_score;
        let optimization = pose_graph
            .optimization_problem_options
            .get_or_insert_with(Default::default);
        optimization.local_slam_pose_translation_weight =
            self.optimization.local_slam_pose_translation_weight;
        optimization.local_slam_pose_rotation_weight =
            self.optimization.local_slam_pose_rotation_weight;
        optimization.odometry_translation_weight = self.optimization.odometry_translation_weight;
        optimization.odometry_rotation_weight = self.optimization.odometry_rotation_weight;

        tb.pure_localization_trimmer = match self.pure_localization {
            Some(pure_localization) => Some(
                proto::trajectory_builder_options::PureLocalizationTrimmerOptions {
                    max_submaps_to_keep: to_i32(
                        pure_localization.max_submaps_to_keep,
                        "pure_localization.max_submaps_to_keep",
                    )?,
                },
            ),
            None => None,
        };

        let num_accumulated_range_data = to_i32(
            self.num_accumulated_range_data,
            "num_accumulated_range_data",
        )?;
        let submap_num_range_data = to_i32(self.submap_num_range_data, "submap_num_range_data")?;
        let motion_filter = proto::MotionFilterOptions {
            max_time_seconds: self.motion_filter.max_time_seconds,
            max_distance_meters: self.motion_filter.max_distance_meters,
            max_angle_radians: self.motion_filter.max_angle_radians,
        };

        // 2D-only knobs always go to the 2D options, 3D-only ones to the 3D
        // options; the shared knobs go to the options of the mode.
        let tb2: &mut LocalTrajectoryBuilderOptions2D = tb
            .trajectory_builder_2d_options
            .get_or_insert_with(Default::default);
        tb2.min_z = self.min_z;
        tb2.max_z = self.max_z;
        tb2.missing_data_ray_length = self.missing_data_ray_length;
        if self.mode == Mode::TwoD {
            tb2.min_range = self.min_range;
            tb2.max_range = self.max_range;
            tb2.num_accumulated_range_data = num_accumulated_range_data;
            tb2.voxel_filter_size = self.voxel_filter_size;
            tb2.use_online_correlative_scan_matching = self.use_online_correlative_scan_matching;
            let ceres = tb2
                .ceres_scan_matcher_options
                .get_or_insert_with(Default::default);
            ceres.translation_weight = self.scan_matcher.translation_weight;
            ceres.rotation_weight = self.scan_matcher.rotation_weight;
            tb2.motion_filter_options = Some(motion_filter);
            tb2.imu_gravity_time_constant = self.imu_gravity_time_constant;
            tb2.pose_extrapolator_options
                .get_or_insert_with(Default::default)
                .constant_velocity
                .get_or_insert_with(Default::default)
                .imu_gravity_time_constant = self.imu_gravity_time_constant;
            let submaps = tb2.submaps_options.get_or_insert_with(Default::default);
            submaps.num_range_data = submap_num_range_data;
            submaps
                .grid_options_2d
                .get_or_insert_with(Default::default)
                .resolution = self.resolution as f32;
        }
        let tb3: &mut LocalTrajectoryBuilderOptions3D = tb
            .trajectory_builder_3d_options
            .get_or_insert_with(Default::default);
        {
            let submaps = tb3.submaps_options.get_or_insert_with(Default::default);
            submaps.low_resolution = self.low_resolution;
            submaps.high_resolution_max_range = self.high_resolution_max_range;
        }
        if self.mode == Mode::ThreeD {
            tb3.min_range = self.min_range;
            tb3.max_range = self.max_range;
            tb3.num_accumulated_range_data = num_accumulated_range_data;
            tb3.voxel_filter_size = self.voxel_filter_size;
            tb3.use_online_correlative_scan_matching = self.use_online_correlative_scan_matching;
            let ceres = tb3
                .ceres_scan_matcher_options
                .get_or_insert_with(Default::default);
            ceres.translation_weight = self.scan_matcher.translation_weight;
            ceres.rotation_weight = self.scan_matcher.rotation_weight;
            tb3.motion_filter_options = Some(motion_filter);
            tb3.imu_gravity_time_constant = self.imu_gravity_time_constant;
            tb3.pose_extrapolator_options
                .get_or_insert_with(Default::default)
                .constant_velocity
                .get_or_insert_with(Default::default)
                .imu_gravity_time_constant = self.imu_gravity_time_constant;
            let submaps = tb3.submaps_options.get_or_insert_with(Default::default);
            submaps.num_range_data = submap_num_range_data;
            submaps.high_resolution = self.resolution;
        }
        Ok(())
    }
}

/// The configuration files shipped with Cartographer, compiled in as a
/// fallback for binaries run away from the source tree.
const SHIPPED_FILES: &[(&str, &str)] = &[
    (
        "map_builder.lua",
        include_str!("../../configuration_files/map_builder.lua"),
    ),
    (
        "pose_graph.lua",
        include_str!("../../configuration_files/pose_graph.lua"),
    ),
    (
        "trajectory_builder.lua",
        include_str!("../../configuration_files/trajectory_builder.lua"),
    ),
    (
        "trajectory_builder_2d.lua",
        include_str!("../../configuration_files/trajectory_builder_2d.lua"),
    ),
    (
        "trajectory_builder_3d.lua",
        include_str!("../../configuration_files/trajectory_builder_3d.lua"),
    ),
];

const EMBEDDED_PREFIX: &str = "<embedded>/";

/// Searches the directories of a [`ConfigurationFileResolver`] (user
/// directories, then this crate's `configuration_files/`), then the
/// compiled-in copies of the shipped files.
struct ShippedFilesResolver {
    directories: Vec<String>,
}

impl FileResolver for ShippedFilesResolver {
    fn get_full_path_or_die(&self, basename: &str) -> String {
        for directory in &self.directories {
            let filename = format!("{directory}/{basename}");
            if std::path::Path::new(&filename).is_file() {
                return filename;
            }
        }
        if SHIPPED_FILES.iter().any(|(name, _)| *name == basename) {
            return format!("{EMBEDDED_PREFIX}{basename}");
        }
        panic!("File '{basename}' was not found.");
    }

    fn get_file_content_or_die(&self, basename: &str) -> String {
        assert!(!basename.is_empty(), "File basename cannot be empty.");
        let filename = self.get_full_path_or_die(basename);
        if let Some(name) = filename.strip_prefix(EMBEDDED_PREFIX) {
            let files: HashMap<&str, &str> = SHIPPED_FILES.iter().copied().collect();
            return files[name].to_string();
        }
        let bytes =
            std::fs::read(&filename).unwrap_or_else(|e| panic!("Failed to read '{filename}': {e}"));
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tb2(tb: &TrajectoryBuilderOptions) -> &LocalTrajectoryBuilderOptions2D {
        tb.trajectory_builder_2d_options.as_ref().unwrap()
    }

    fn tb3(tb: &TrajectoryBuilderOptions) -> &LocalTrajectoryBuilderOptions3D {
        tb.trajectory_builder_3d_options.as_ref().unwrap()
    }

    #[test]
    fn defaults_come_from_shipped_lua() {
        let config = SlamConfig::new(Mode::TwoD);
        assert_eq!(config.mode(), Mode::TwoD);
        assert_eq!(config.resolution, f64::from(0.05_f32));
        assert_eq!(config.min_range, 0.0);
        assert_eq!(config.max_range, 30.0);
        assert_eq!(config.min_z, -0.8);
        assert_eq!(config.max_z, 2.0);
        assert_eq!(config.missing_data_ray_length, 5.0);
        assert_eq!(config.num_accumulated_range_data, 1);
        assert_eq!(config.voxel_filter_size, 0.025);
        assert_eq!(config.submap_num_range_data, 90);
        assert!(!config.use_online_correlative_scan_matching);
        assert_eq!(config.scan_matcher.translation_weight, 10.0);
        assert_eq!(config.scan_matcher.rotation_weight, 40.0);
        assert_eq!(config.motion_filter.max_time_seconds, 5.0);
        assert_eq!(config.motion_filter.max_distance_meters, 0.2);
        assert_eq!(config.motion_filter.max_angle_radians, 1.0_f64.to_radians());
        assert_eq!(config.imu_gravity_time_constant, 10.0);
        assert!(config.loop_closure.enabled);
        assert_eq!(config.loop_closure.optimize_every_n_nodes, 90);
        assert_eq!(config.loop_closure.min_score, 0.55);
        assert_eq!(config.loop_closure.sampling_ratio, 0.3);
        assert_eq!(config.loop_closure.max_constraint_distance, 15.0);
        assert_eq!(config.loop_closure.translation_weight, 1.1e4);
        assert_eq!(config.loop_closure.rotation_weight, 1e5);
        assert_eq!(config.loop_closure.max_num_final_iterations, 200);
        assert_eq!(config.global_localization.sampling_ratio, 0.003);
        assert_eq!(config.global_localization.min_score, 0.6);
        assert_eq!(config.global_localization.search_after_seconds, 10.0);
        assert_eq!(config.optimization.odometry_translation_weight, 1e5);
        assert_eq!(config.optimization.local_slam_pose_rotation_weight, 1e5);
        assert_eq!(config.num_background_threads, 4);
        assert_eq!(config.pure_localization, None);

        let config = SlamConfig::new(Mode::ThreeD);
        assert_eq!(config.mode(), Mode::ThreeD);
        assert_eq!(config.resolution, 0.10);
        assert_eq!(config.low_resolution, 0.45);
        assert_eq!(config.high_resolution_max_range, 20.0);
        assert_eq!(config.min_range, 1.0);
        assert_eq!(config.max_range, 60.0);
        assert_eq!(config.voxel_filter_size, 0.15);
        assert_eq!(config.submap_num_range_data, 160);
        assert_eq!(config.scan_matcher.rotation_weight, 4e2);
        assert_eq!(config.motion_filter.max_time_seconds, 0.5);
        assert_eq!(config.motion_filter.max_angle_radians, 0.004);
    }

    #[test]
    fn unchanged_config_reproduces_the_lua_protos() {
        for mode in [Mode::TwoD, Mode::ThreeD] {
            let config = SlamConfig::new(mode);
            let (mb, tb) = config.to_protos().unwrap();
            assert_eq!(mb, config.base_map_builder_options, "{mode}");
            assert_eq!(tb, config.base_trajectory_builder_options, "{mode}");
            assert_eq!(mb.use_trajectory_builder_2d, mode == Mode::TwoD);
            assert_eq!(mb.use_trajectory_builder_3d, mode == Mode::ThreeD);
        }
    }

    fn customize(config: &mut SlamConfig) {
        config.resolution = 0.0625;
        config.min_range = 0.3;
        config.max_range = 25.0;
        config.min_z = -0.5;
        config.max_z = 1.5;
        config.missing_data_ray_length = 4.5;
        config.low_resolution = 0.5;
        config.high_resolution_max_range = 18.0;
        config.num_accumulated_range_data = 3;
        config.voxel_filter_size = 0.04;
        config.submap_num_range_data = 33;
        config.use_online_correlative_scan_matching = true;
        config.scan_matcher = ScanMatcherConfig {
            translation_weight: 11.0,
            rotation_weight: 41.0,
        };
        config.motion_filter = MotionFilterConfig {
            max_time_seconds: 2.5,
            max_distance_meters: 0.15,
            max_angle_radians: 0.02,
        };
        config.imu_gravity_time_constant = 7.0;
        config.loop_closure = LoopClosureConfig {
            enabled: true,
            optimize_every_n_nodes: 35,
            min_score: 0.62,
            sampling_ratio: 0.25,
            max_constraint_distance: 12.0,
            translation_weight: 2e4,
            rotation_weight: 3e5,
            max_num_final_iterations: 150,
        };
        config.global_localization = GlobalLocalizationConfig {
            sampling_ratio: 0.2,
            min_score: 0.65,
            search_after_seconds: 3.0,
        };
        config.optimization = OptimizationConfig {
            local_slam_pose_translation_weight: 1.5e5,
            local_slam_pose_rotation_weight: 2.5e5,
            odometry_translation_weight: 3.5e5,
            odometry_rotation_weight: 4.5e5,
        };
        config.num_background_threads = 2;
        config.pure_localization = Some(PureLocalizationConfig {
            max_submaps_to_keep: 4,
        });
    }

    fn assert_common_pose_graph_knobs(mb: &MapBuilderOptions, tb: &TrajectoryBuilderOptions) {
        assert_eq!(mb.num_background_threads, 2);
        let pg = mb.pose_graph_options.as_ref().unwrap();
        assert_eq!(pg.optimize_every_n_nodes, 35);
        assert_eq!(pg.max_num_final_iterations, 150);
        assert_eq!(pg.global_sampling_ratio, 0.2);
        assert_eq!(pg.global_constraint_search_after_n_seconds, 3.0);
        let cb = pg.constraint_builder_options.as_ref().unwrap();
        assert_eq!(cb.min_score, 0.62);
        assert_eq!(cb.sampling_ratio, 0.25);
        assert_eq!(cb.max_constraint_distance, 12.0);
        assert_eq!(cb.loop_closure_translation_weight, 2e4);
        assert_eq!(cb.loop_closure_rotation_weight, 3e5);
        assert_eq!(cb.global_localization_min_score, 0.65);
        let opt = pg.optimization_problem_options.as_ref().unwrap();
        assert_eq!(opt.local_slam_pose_translation_weight, 1.5e5);
        assert_eq!(opt.local_slam_pose_rotation_weight, 2.5e5);
        assert_eq!(opt.odometry_translation_weight, 3.5e5);
        assert_eq!(opt.odometry_rotation_weight, 4.5e5);
        assert_eq!(
            tb.pure_localization_trimmer
                .as_ref()
                .unwrap()
                .max_submaps_to_keep,
            4
        );
    }

    #[test]
    fn every_2d_knob_lands_in_its_proto_field() {
        let mut config = SlamConfig::new(Mode::TwoD);
        let base_3d = config
            .base_trajectory_builder_options
            .trajectory_builder_3d_options
            .clone()
            .unwrap();
        customize(&mut config);
        let (mb, tb) = config.to_protos().unwrap();
        assert!(mb.use_trajectory_builder_2d && !mb.use_trajectory_builder_3d);
        assert_common_pose_graph_knobs(&mb, &tb);
        let o = tb2(&tb);
        let submaps = o.submaps_options.as_ref().unwrap();
        assert_eq!(
            submaps.grid_options_2d.as_ref().unwrap().resolution,
            0.0625_f32
        );
        assert_eq!(submaps.num_range_data, 33);
        assert_eq!(o.min_range, 0.3);
        assert_eq!(o.max_range, 25.0);
        assert_eq!(o.min_z, -0.5);
        assert_eq!(o.max_z, 1.5);
        assert_eq!(o.missing_data_ray_length, 4.5);
        assert_eq!(o.num_accumulated_range_data, 3);
        assert_eq!(o.voxel_filter_size, 0.04);
        assert!(o.use_online_correlative_scan_matching);
        let ceres = o.ceres_scan_matcher_options.as_ref().unwrap();
        assert_eq!(ceres.translation_weight, 11.0);
        assert_eq!(ceres.rotation_weight, 41.0);
        // Untouched fields keep their Lua values.
        assert_eq!(ceres.occupied_space_weight, 1.0);
        let mf = o.motion_filter_options.as_ref().unwrap();
        assert_eq!(
            (
                mf.max_time_seconds,
                mf.max_distance_meters,
                mf.max_angle_radians
            ),
            (2.5, 0.15, 0.02)
        );
        assert_eq!(o.imu_gravity_time_constant, 7.0);
        assert_eq!(
            o.pose_extrapolator_options
                .as_ref()
                .unwrap()
                .constant_velocity
                .as_ref()
                .unwrap()
                .imu_gravity_time_constant,
            7.0
        );
        // The 3D options only get the 3D-only knobs.
        let o3 = tb3(&tb);
        let submaps3 = o3.submaps_options.as_ref().unwrap();
        assert_eq!(submaps3.low_resolution, 0.5);
        assert_eq!(submaps3.high_resolution_max_range, 18.0);
        assert_eq!(submaps3.high_resolution, 0.10);
        assert_eq!(o3.min_range, base_3d.min_range);
        assert_eq!(o3.motion_filter_options, base_3d.motion_filter_options);
        // Reading the protos back gives the same knobs.
        let back = SlamConfig::from_protos(Mode::TwoD, mb, tb);
        assert_eq!(
            format!("{back:?}"),
            format!("{:?}", {
                let mut c = config.clone();
                c.advanced.clear();
                c
            })
        );
    }

    #[test]
    fn every_3d_knob_lands_in_its_proto_field() {
        let mut config = SlamConfig::new(Mode::ThreeD);
        let base_2d = config
            .base_trajectory_builder_options
            .trajectory_builder_2d_options
            .unwrap();
        customize(&mut config);
        let (mb, tb) = config.to_protos().unwrap();
        assert!(!mb.use_trajectory_builder_2d && mb.use_trajectory_builder_3d);
        assert_common_pose_graph_knobs(&mb, &tb);
        let o = tb3(&tb);
        let submaps = o.submaps_options.as_ref().unwrap();
        assert_eq!(submaps.high_resolution, 0.0625);
        assert_eq!(submaps.low_resolution, 0.5);
        assert_eq!(submaps.high_resolution_max_range, 18.0);
        assert_eq!(submaps.num_range_data, 33);
        assert_eq!(o.min_range, 0.3);
        assert_eq!(o.max_range, 25.0);
        assert_eq!(o.num_accumulated_range_data, 3);
        assert_eq!(o.voxel_filter_size, 0.04);
        assert!(o.use_online_correlative_scan_matching);
        let ceres = o.ceres_scan_matcher_options.as_ref().unwrap();
        assert_eq!(ceres.translation_weight, 11.0);
        assert_eq!(ceres.rotation_weight, 41.0);
        let mf = o.motion_filter_options.as_ref().unwrap();
        assert_eq!(
            (
                mf.max_time_seconds,
                mf.max_distance_meters,
                mf.max_angle_radians
            ),
            (2.5, 0.15, 0.02)
        );
        assert_eq!(o.imu_gravity_time_constant, 7.0);
        assert_eq!(
            o.pose_extrapolator_options
                .as_ref()
                .unwrap()
                .constant_velocity
                .as_ref()
                .unwrap()
                .imu_gravity_time_constant,
            7.0
        );
        // The 2D options only get the 2D-only knobs.
        let o2 = tb2(&tb);
        assert_eq!(o2.min_z, -0.5);
        assert_eq!(o2.max_z, 1.5);
        assert_eq!(o2.missing_data_ray_length, 4.5);
        assert_eq!(o2.min_range, base_2d.min_range);
        assert_eq!(o2.submaps_options, base_2d.submaps_options);
        let back = SlamConfig::from_protos(Mode::ThreeD, mb, tb);
        assert_eq!(back.resolution, 0.0625);
        assert_eq!(back.submap_num_range_data, 33);
        assert_eq!(back.loop_closure, config.loop_closure);
    }

    #[test]
    fn disabling_loop_closure_writes_zero() {
        let mut config = SlamConfig::new(Mode::TwoD);
        config.loop_closure.enabled = false;
        let (mb, _) = config.to_protos().unwrap();
        assert_eq!(mb.pose_graph_options.unwrap().optimize_every_n_nodes, 0);
        config.loop_closure.enabled = true;
        config.loop_closure.optimize_every_n_nodes = 0;
        assert!(matches!(
            config.to_protos(),
            Err(SlamError::InvalidConfig(_))
        ));
    }

    #[test]
    fn advanced_edits_run_last() {
        let mut config = SlamConfig::new(Mode::TwoD);
        config.submap_num_range_data = 10;
        config
            .advanced(|mb, _| mb.collate_by_trajectory = true)
            .advanced(|_, tb| {
                tb.trajectory_builder_2d_options
                    .as_mut()
                    .unwrap()
                    .submaps_options
                    .as_mut()
                    .unwrap()
                    .num_range_data = 11;
            });
        let (mb, tb) = config
            .to_protos_with(|_, tb| {
                tb.trajectory_builder_2d_options
                    .as_mut()
                    .unwrap()
                    .use_imu_data = false;
            })
            .unwrap();
        assert!(mb.collate_by_trajectory);
        assert_eq!(
            tb2(&tb).submaps_options.as_ref().unwrap().num_range_data,
            11
        );
        assert!(!tb2(&tb).use_imu_data);
        // Clones share the edits.
        let (mb, _) = config.clone().to_protos().unwrap();
        assert!(mb.collate_by_trajectory);
    }

    #[test]
    fn validation_rejects_bad_values() {
        type Mutation = Box<dyn Fn(&mut SlamConfig)>;
        let cases: Vec<(&str, Mutation)> = vec![
            ("resolution", Box::new(|c| c.resolution = 0.0)),
            ("resolution", Box::new(|c| c.resolution = f64::NAN)),
            ("max_range", Box::new(|c| c.max_range = c.min_range)),
            ("min_range", Box::new(|c| c.min_range = -1.0)),
            ("max_z", Box::new(|c| c.max_z = c.min_z - 1.0)),
            (
                "num_accumulated_range_data",
                Box::new(|c| c.num_accumulated_range_data = 0),
            ),
            (
                "submap_num_range_data",
                Box::new(|c| c.submap_num_range_data = 0),
            ),
            ("voxel_filter_size", Box::new(|c| c.voxel_filter_size = 0.0)),
            ("min_score", Box::new(|c| c.loop_closure.min_score = 1.5)),
            (
                "sampling_ratio",
                Box::new(|c| c.global_localization.sampling_ratio = -0.1),
            ),
            (
                "num_background_threads",
                Box::new(|c| c.num_background_threads = 0),
            ),
            (
                "max_submaps_to_keep",
                Box::new(|c| {
                    c.pure_localization = Some(PureLocalizationConfig {
                        max_submaps_to_keep: 1,
                    })
                }),
            ),
            (
                "imu_gravity_time_constant",
                Box::new(|c| c.imu_gravity_time_constant = 0.0),
            ),
        ];
        for (field, mutate) in cases {
            let mut config = SlamConfig::new(Mode::TwoD);
            mutate(&mut config);
            match config.validate() {
                Err(SlamError::InvalidConfig(message)) => {
                    assert!(message.contains(field), "{field}: {message}")
                }
                other => panic!("{field}: expected InvalidConfig, got {other:?}"),
            }
        }
        // 3D-only checks do not apply in 2D and vice versa.
        let mut config = SlamConfig::new(Mode::TwoD);
        config.low_resolution = 0.0;
        assert!(config.validate().is_ok());
        let mut config = SlamConfig::new(Mode::ThreeD);
        config.low_resolution = 0.0;
        assert!(config.validate().is_err());
        config.low_resolution = 0.45;
        config.max_z = -10.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn from_lua_reads_cartographer_ros_style_files() {
        let config = SlamConfig::from_lua(
            r#"
            include "map_builder.lua"
            include "trajectory_builder.lua"
            options = {
              map_builder = MAP_BUILDER,
              trajectory_builder = TRAJECTORY_BUILDER,
              map_frame = "map",
              tracking_frame = "imu_link",
              num_laser_scans = 0,
            }
            MAP_BUILDER.use_trajectory_builder_3d = true
            MAP_BUILDER.num_background_threads = 3
            TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 2
            POSE_GRAPH.optimize_every_n_nodes = 0
            return options"#,
            &[] as &[&str],
        )
        .unwrap();
        assert_eq!(config.mode(), Mode::ThreeD);
        assert_eq!(config.num_background_threads, 3);
        assert_eq!(config.num_accumulated_range_data, 2);
        assert!(!config.loop_closure.enabled);
        let (mb, _) = config.to_protos().unwrap();
        assert_eq!(mb.pose_graph_options.unwrap().optimize_every_n_nodes, 0);
    }

    #[test]
    fn from_lua_searches_user_directories_first() {
        let dir = std::env::temp_dir().join(format!("slam_config_lua_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("my_robot.lua"),
            "include \"map_builder.lua\"\n\
             include \"trajectory_builder.lua\"\n\
             MAP_BUILDER.use_trajectory_builder_2d = true\n\
             TRAJECTORY_BUILDER_2D.max_range = 12.\n\
             MY_OPTIONS = { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }",
        )
        .unwrap();
        let config = SlamConfig::from_lua(
            "include \"my_robot.lua\"\nreturn MY_OPTIONS",
            &[dir.to_str().unwrap()],
        )
        .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(config.mode(), Mode::TwoD);
        assert_eq!(config.max_range, 12.0);
    }

    #[test]
    fn embedded_files_are_byte_identical_to_the_shipped_ones() {
        for (name, content) in SHIPPED_FILES {
            let on_disk = std::fs::read_to_string(format!(
                "{}/{name}",
                crate::common::config::CONFIGURATION_FILES_DIRECTORY
            ))
            .unwrap();
            assert_eq!(&on_disk, content, "{name}");
        }
        let resolver = ShippedFilesResolver {
            directories: vec!["/nonexistent".to_string()],
        };
        assert_eq!(
            resolver.get_full_path_or_die("pose_graph.lua"),
            "<embedded>/pose_graph.lua"
        );
        assert!(
            resolver
                .get_file_content_or_die("pose_graph.lua")
                .contains("POSE_GRAPH")
        );
    }

    #[test]
    fn lua_errors_become_invalid_config() {
        let cases = [
            // Syntax error.
            "return {",
            // Neither 2D nor 3D.
            "include \"map_builder.lua\"\ninclude \"trajectory_builder.lua\"\n\
             return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }",
            // Missing trajectory_builder.
            "include \"map_builder.lua\"\nMAP_BUILDER.use_trajectory_builder_2d = true\n\
             return { map_builder = MAP_BUILDER }",
            // Typo in a key (unused key detected).
            "include \"map_builder.lua\"\ninclude \"trajectory_builder.lua\"\n\
             MAP_BUILDER.use_trajectory_builder_2d = true\n\
             TRAJECTORY_BUILDER_2D.max_rnage = 3.\n\
             return { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER }",
            // Unknown include.
            "include \"does_not_exist.lua\"\nreturn {}",
        ];
        for code in cases {
            match SlamConfig::from_lua(code, &[] as &[&str]) {
                Err(SlamError::InvalidConfig(message)) => {
                    assert!(message.starts_with("Lua configuration"), "{message}")
                }
                other => panic!("{code}: expected InvalidConfig, got {other:?}"),
            }
        }
    }
}
