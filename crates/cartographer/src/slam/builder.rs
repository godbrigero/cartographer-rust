//! [`SlamBuilder`]: declares sensors, configuration and an optional map to
//! load, then wires up a faithful [`MapBuilder`].

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use super::config::{PureLocalizationConfig, SlamConfig};
use super::error::{SlamError, catch_check_failure};
use super::instance::{Shared, Slam, SlamSensors};
use super::time::Timestamp;
use super::types::{Extrinsics, LocalResult, Mode, OptimizationEvent, SensorKind, to_isometry};
use crate::io::proto_stream::ProtoStreamReader;
use crate::io::proto_stream_deserializer::ProtoStreamDeserializer;
use crate::mapping::map_builder::MapBuilder;
use crate::mapping::map_builder_interface::MapBuilderInterface;
use crate::mapping::proto::{SerializedData, serialized_data::Data};
use crate::mapping::trajectory_builder_interface::{LocalSlamResultCallback, SensorId, SensorType};

/// Default number of submaps kept by a pure localization trajectory.
const DEFAULT_PURE_LOCALIZATION_SUBMAPS: u32 = 3;

/// Maximum IMU lever arm (m) accepted in 3D (cartographer_ros' check).
const MAX_IMU_TRANSLATION: f64 = 1e-5;

#[derive(Clone, Debug)]
enum LoadMode {
    /// Frozen map + pure localization trimmer.
    Localize,
    /// Frozen map, new trajectory keeps all its submaps.
    Extend,
}

/// Builder of a [`Slam`] instance: declare the sensors, tweak the
/// configuration, then [`build`](Self::build).
///
/// Sensor ids are free-form strings used in the `push_*` calls (e.g. ROS
/// topic names). Declaration mistakes (duplicate ids, a second IMU, a
/// missing required sensor) are reported by [`build`](Self::build).
#[derive(Clone, Debug)]
pub struct SlamBuilder {
    config: SlamConfig,
    lidars: Vec<(String, Extrinsics)>,
    imu: Option<(String, Extrinsics)>,
    odometry: Option<(String, Extrinsics)>,
    declaration_error: Option<String>,
    load: Option<(PathBuf, LoadMode)>,
}

impl SlamBuilder {
    /// Starts from Cartographer's shipped defaults for `mode`.
    pub fn new(mode: Mode) -> Self {
        Self::with_config(SlamConfig::new(mode))
    }

    /// Starts from a prepared configuration.
    pub fn with_config(config: SlamConfig) -> Self {
        SlamBuilder {
            config,
            lidars: Vec::new(),
            imu: None,
            odometry: None,
            declaration_error: None,
            load: None,
        }
    }

    /// Starts from Lua code (cartographer_ros format; see
    /// [`SlamConfig::from_lua`]). The mode comes from the Lua
    /// `MAP_BUILDER.use_trajectory_builder_2d/3d`.
    pub fn from_lua<S: AsRef<str>>(code: &str, search_dirs: &[S]) -> Result<Self, SlamError> {
        Ok(Self::with_config(SlamConfig::from_lua(code, search_dirs)?))
    }

    fn declare(&mut self, id: &str, kind: SensorKind) {
        let taken = self.lidars.iter().any(|(existing, _)| existing == id)
            || self
                .imu
                .as_ref()
                .is_some_and(|(existing, _)| existing == id)
            || self
                .odometry
                .as_ref()
                .is_some_and(|(existing, _)| existing == id);
        if taken && self.declaration_error.is_none() {
            self.declaration_error = Some(format!("sensor id '{id}' ({kind}) is declared twice"));
        }
    }

    /// Declares a range sensor (2D or 3D LiDAR) with id `id` mounted at
    /// `extrinsics` (tracking_from_sensor). Several LiDARs may be declared;
    /// Cartographer merges their data by time.
    pub fn lidar(mut self, id: impl Into<String>, extrinsics: impl Into<Extrinsics>) -> Self {
        let id = id.into();
        self.declare(&id, SensorKind::Lidar);
        self.lidars.push((id, extrinsics.into()));
        self
    }

    /// Declares the IMU (at most one). Optional in 2D (then it supplies
    /// gravity and rotation to local SLAM), required in 3D. Its data are
    /// rotated into the tracking frame; in 3D the IMU must be at the origin
    /// of the tracking frame (cartographer_ros enforces the same), because
    /// accelerations cannot be transformed exactly across a lever arm.
    pub fn imu(mut self, id: impl Into<String>, extrinsics: impl Into<Extrinsics>) -> Self {
        let id = id.into();
        self.declare(&id, SensorKind::Imu);
        if self.imu.is_some() && self.declaration_error.is_none() {
            self.declaration_error = Some("only one IMU is supported".to_string());
        }
        self.imu = Some((id, extrinsics.into()));
        self
    }

    /// Declares an odometry source whose poses are those of the tracking
    /// frame (at most one odometry source).
    pub fn odometry(self, id: impl Into<String>) -> Self {
        self.odometry_with_extrinsics(id, Extrinsics::identity())
    }

    /// Declares an odometry source reporting the pose of a frame mounted at
    /// `extrinsics` (tracking_from_odometry_frame), e.g. a wheel-odometry
    /// `base_footprint` when tracking an IMU frame.
    pub fn odometry_with_extrinsics(
        mut self,
        id: impl Into<String>,
        extrinsics: impl Into<Extrinsics>,
    ) -> Self {
        let id = id.into();
        self.declare(&id, SensorKind::Odometry);
        if self.odometry.is_some() && self.declaration_error.is_none() {
            self.declaration_error = Some("only one odometry source is supported".to_string());
        }
        self.odometry = Some((id, extrinsics.into()));
        self
    }

    /// Edits the configuration.
    ///
    /// ```
    /// use cartographer::{Mode, Slam};
    /// let builder = Slam::builder(Mode::TwoD).config(|c| {
    ///     c.resolution = 0.1;
    ///     c.max_range = 20.0;
    ///     c.loop_closure.optimize_every_n_nodes = 30;
    /// });
    /// assert_eq!(builder.config_ref().max_range, 20.0);
    /// ```
    pub fn config(mut self, edit: impl FnOnce(&mut SlamConfig)) -> Self {
        edit(&mut self.config);
        self
    }

    /// The configuration so far.
    pub fn config_ref(&self) -> &SlamConfig {
        &self.config
    }

    /// Localize in a saved map instead of building a new one: the map is
    /// loaded frozen (its poses never change) and the new trajectory keeps
    /// only its latest submaps (`pure_localization`, default 3 if not set
    /// in the config). The new trajectory is connected to the map by global
    /// localization (see [`SlamConfig::global_localization`]; the shipped
    /// sampling ratio of 0.003 relocalizes slowly, 0.05-0.3 is typical for
    /// localization).
    ///
    /// Poses are in the frame of the loaded map. Read results before
    /// [`Slam::finish`]: like in Cartographer, the pure localization trimmer
    /// removes the whole localization trajectory once it is finished.
    pub fn localize_in(mut self, pbstream: impl Into<PathBuf>) -> Self {
        self.load = Some((pbstream.into(), LoadMode::Localize));
        self
    }

    /// Continue mapping on top of a saved map: the map is loaded frozen and
    /// the new trajectory adds submaps to it (no trimming).
    pub fn extend_map(mut self, pbstream: impl Into<PathBuf>) -> Self {
        self.load = Some((pbstream.into(), LoadMode::Extend));
        self
    }

    /// Validates the declaration and configuration and starts the SLAM
    /// system.
    pub fn build(self) -> Result<Slam, SlamError> {
        let SlamBuilder {
            mut config,
            lidars,
            imu,
            odometry,
            declaration_error,
            load,
        } = self;
        let mode = config.mode();
        if let Some(message) = declaration_error {
            return Err(SlamError::DuplicateSensor(message));
        }
        if lidars.is_empty() {
            return Err(SlamError::MissingSensor(
                "at least one LiDAR must be declared (SlamBuilder::lidar)".to_string(),
            ));
        }
        if let Some((id, extrinsics)) = &imu {
            let lever_arm = extrinsics.tracking_from_sensor.translation.vector.norm();
            if lever_arm >= MAX_IMU_TRANSLATION {
                match mode {
                    Mode::ThreeD => {
                        return Err(SlamError::InvalidConfig(format!(
                            "the IMU '{id}' must be at the origin of the tracking frame in 3D \
                             (translation is {lever_arm} m); use the IMU frame as tracking \
                             frame and express the other extrinsics relative to it"
                        )));
                    }
                    Mode::TwoD => log::warn!(
                        "IMU '{id}' is {lever_arm} m away from the tracking frame origin; its \
                         data are only rotated, linear accelerations are not compensated for \
                         the lever arm"
                    ),
                }
            }
        } else if mode == Mode::ThreeD {
            return Err(SlamError::MissingSensor(
                "3D SLAM requires an IMU (SlamBuilder::imu)".to_string(),
            ));
        }
        if matches!(load, Some((_, LoadMode::Localize))) && config.pure_localization.is_none() {
            config.pure_localization = Some(PureLocalizationConfig {
                max_submaps_to_keep: DEFAULT_PURE_LOCALIZATION_SUBMAPS,
            });
        }

        let has_imu = imu.is_some();
        let (map_builder_options, trajectory_builder_options) =
            config.to_protos_with(|_, trajectory_builder_options| {
                if mode == Mode::TwoD {
                    trajectory_builder_options
                        .trajectory_builder_2d_options
                        .get_or_insert_with(Default::default)
                        .use_imu_data = has_imu;
                }
            })?;
        let use_intensities = match mode {
            Mode::TwoD => {
                let use_imu_data = trajectory_builder_options
                    .trajectory_builder_2d_options
                    .as_ref()
                    .is_some_and(|options| options.use_imu_data);
                if use_imu_data != has_imu {
                    return Err(SlamError::InvalidConfig(format!(
                        "use_imu_data = {use_imu_data} was set by an advanced edit, but an IMU \
                         was {}declared",
                        if has_imu { "" } else { "not " }
                    )));
                }
                false
            }
            Mode::ThreeD => trajectory_builder_options
                .trajectory_builder_3d_options
                .as_ref()
                .is_some_and(|options| options.use_intensities),
        };

        let mut map_builder = catch_check_failure(|| MapBuilder::new(&map_builder_options))
            .map_err(SlamError::InvalidConfig)?;

        if let Some((path, _)) = &load {
            std::fs::File::open(path).map_err(|e| SlamError::io(path, e))?;
            let filename = path.to_str().ok_or_else(|| {
                SlamError::io(
                    path,
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "path is not UTF-8"),
                )
            })?;
            check_map_mode(filename, mode)?;
            let loaded = catch_check_failure(|| map_builder.load_state_from_file(filename, true))
                .map_err(SlamError::InvalidMap)?;
            log::info!("Loaded {} trajectories from '{filename}'.", loaded.len());
        }

        let mut expected_sensor_ids = BTreeSet::new();
        for (id, _) in &lidars {
            expected_sensor_ids.insert(SensorId {
                r#type: SensorType::Range,
                id: id.clone(),
            });
        }
        if let Some((id, _)) = &imu {
            expected_sensor_ids.insert(SensorId {
                r#type: SensorType::Imu,
                id: id.clone(),
            });
        }
        if let Some((id, _)) = &odometry {
            expected_sensor_ids.insert(SensorId {
                r#type: SensorType::Odometry,
                id: id.clone(),
            });
        }

        let shared = Arc::new(Shared::default());
        let callback: LocalSlamResultCallback = {
            let shared = Arc::clone(&shared);
            Arc::new(
                move |trajectory_id, time, local_pose, range_data, insertion_result| {
                    shared
                        .num_local_results
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    *shared
                        .latest_local_pose
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner) = Some((time, local_pose));
                    let mut callbacks = shared
                        .local_callbacks
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    if callbacks.is_empty() {
                        return;
                    }
                    let result = LocalResult {
                        trajectory_id,
                        time: Timestamp::from_cartographer_time(time),
                        local_pose: to_isometry(&local_pose),
                        node_id: insertion_result.map(|insertion| insertion.node_id),
                        points_in_local: range_data
                            .returns
                            .points()
                            .iter()
                            .map(|point| point.position)
                            .collect(),
                    };
                    for callback in callbacks.iter_mut() {
                        callback(&result);
                    }
                },
            )
        };
        let trajectory_id = catch_check_failure(|| {
            map_builder.add_trajectory_builder(
                &expected_sensor_ids,
                &trajectory_builder_options,
                Some(callback),
            )
        })
        .map_err(SlamError::InvalidConfig)?;

        let pose_graph = map_builder.pose_graph();
        {
            let shared = Arc::clone(&shared);
            pose_graph.set_global_slam_optimization_callback(Box::new(
                move |last_optimized_submaps, last_optimized_nodes| {
                    shared
                        .num_optimizations
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let mut callbacks = shared
                        .optimization_callbacks
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    if callbacks.is_empty() {
                        return;
                    }
                    let event = OptimizationEvent {
                        last_optimized_submaps: last_optimized_submaps.clone(),
                        last_optimized_nodes: last_optimized_nodes.clone(),
                    };
                    for callback in callbacks.iter_mut() {
                        callback(&event);
                    }
                },
            ));
        }

        let sensors = SlamSensors {
            lidars: lidars
                .into_iter()
                .map(|(id, extrinsics)| (id, extrinsics.tracking_from_sensor.cast::<f32>()))
                .collect::<HashMap<_, _>>(),
            imu: imu.map(|(id, extrinsics)| (id, extrinsics.tracking_from_sensor.rotation)),
            odometry: odometry
                .map(|(id, extrinsics)| (id, extrinsics.tracking_from_sensor.inverse())),
        };
        Ok(Slam::new(
            mode,
            map_builder,
            pose_graph,
            trajectory_id,
            sensors,
            use_intensities,
            shared,
            (map_builder_options, trajectory_builder_options),
        ))
    }
}

/// Reads the pbstream once to check that it holds submaps of `mode`: a
/// mismatch would otherwise only fail later on a background thread (which
/// aborts the process, like a failed `CHECK` in C++).
fn check_map_mode(filename: &str, mode: Mode) -> Result<(), SlamError> {
    let found = catch_check_failure(|| {
        let mut reader = ProtoStreamReader::new(filename);
        let mut deserializer = ProtoStreamDeserializer::new(&mut reader);
        let mut data = SerializedData::default();
        while deserializer.read_next_serialized_data(&mut data) {
            if let Some(Data::Submap(submap)) = &data.data {
                if submap.submap_2d.is_some() {
                    return Some(Mode::TwoD);
                }
                if submap.submap_3d.is_some() {
                    return Some(Mode::ThreeD);
                }
            }
        }
        None
    })
    .map_err(SlamError::InvalidMap)?;
    match found {
        Some(found) if found != mode => Err(SlamError::InvalidMap(format!(
            "'{filename}' contains a {found} map, but this SLAM system is {mode}"
        ))),
        _ => Ok(()),
    }
}

/// Keeps the user callback lists (exposed for [`Slam`]).
pub(crate) type LocalCallbacks = Mutex<Vec<Box<dyn FnMut(&LocalResult) + Send>>>;
/// Keeps the user optimization callback lists (exposed for [`Slam`]).
pub(crate) type OptimizationCallbacks = Mutex<Vec<Box<dyn FnMut(&OptimizationEvent) + Send>>>;
