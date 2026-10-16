//! Port of `cartographer/mapping/internal/collated_trajectory_builder` (.h, .cc).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use crate::common::internal::rate_timer::RateTimer;
use crate::common::time::from_seconds;
use crate::mapping::internal::local_slam_result_data::LocalSlamResultData;
use crate::mapping::proto;
use crate::mapping::trajectory_builder_interface::{
    SensorId, SensorType, TrajectoryBuilderInterface,
};
use crate::sensor::collator_interface::CollatorInterface;
use crate::sensor::data::Data;
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::internal::dispatchable::make_dispatchable;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;

const SENSOR_DATA_RATES_LOGGING_PERIOD_SECONDS: f64 = 15.;

/// A sensor collator shared between the map builder and its trajectory
/// builders (C++ non-owning `sensor::CollatorInterface*`).
pub type SharedCollator = Arc<Mutex<dyn CollatorInterface + Send>>;

/// The state touched by collated data (C++ members used by
/// `HandleCollatedSensorData`).
struct CollatedHandler {
    wrapped_trajectory_builder: Box<dyn TrajectoryBuilderInterface>,
    // Time at which we last logged the rates of incoming sensor data.
    last_logging_time: Instant,
    rate_timers: BTreeMap<String, RateTimer>,
}

impl CollatedHandler {
    fn handle_collated_sensor_data(&mut self, sensor_id: &str, data: Box<dyn Data>) {
        self.rate_timers
            .entry(sensor_id.to_string())
            .or_insert_with(|| {
                RateTimer::new(from_seconds(SENSOR_DATA_RATES_LOGGING_PERIOD_SECONDS))
            })
            .pulse(data.get_time());

        if self.last_logging_time.elapsed().as_secs_f64() > SENSOR_DATA_RATES_LOGGING_PERIOD_SECONDS
        {
            for (sensor_id, rate_timer) in &self.rate_timers {
                log::info!("{} rate: {}", sensor_id, rate_timer.debug_string());
            }
            self.last_logging_time = Instant::now();
        }

        data.add_to_trajectory_builder(self.wrapped_trajectory_builder.as_mut());
    }
}

fn lock(handler: &Mutex<CollatedHandler>) -> MutexGuard<'_, CollatedHandler> {
    handler.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Collates sensor data using a `sensor::CollatorInterface`, then passes it on
/// to a `mapping::TrajectoryBuilderInterface` which is common for 2D and 3D.
///
/// RUST PORT NOTE: the C++ registers a callback capturing `this` with the
/// (non-owning) collator. Here the wrapped builder, rate timers and logging
/// time live in an `Arc<Mutex<..>>` shared with that callback; the collator
/// is shared as [`SharedCollator`]. The collator lock is taken first and the
/// handler lock second (from within the callback); never the other way
/// round.
pub struct CollatedTrajectoryBuilder {
    sensor_collator: SharedCollator,
    collate_landmarks: bool,
    collate_fixed_frame: bool,
    trajectory_id: i32,
    handler: Arc<Mutex<CollatedHandler>>,
}

impl std::fmt::Debug for CollatedTrajectoryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollatedTrajectoryBuilder")
            .field("collate_landmarks", &self.collate_landmarks)
            .field("collate_fixed_frame", &self.collate_fixed_frame)
            .field("trajectory_id", &self.trajectory_id)
            .finish_non_exhaustive()
    }
}

impl CollatedTrajectoryBuilder {
    /// Creates the builder and registers `trajectory_id` with
    /// `sensor_collator`, expecting data from `expected_sensor_ids` (landmark
    /// and fixed frame sensors only if they are collated).
    pub fn new(
        trajectory_options: &proto::TrajectoryBuilderOptions,
        sensor_collator: SharedCollator,
        trajectory_id: i32,
        expected_sensor_ids: &BTreeSet<SensorId>,
        wrapped_trajectory_builder: Box<dyn TrajectoryBuilderInterface>,
    ) -> Self {
        let collate_landmarks = trajectory_options.collate_landmarks;
        let collate_fixed_frame = trajectory_options.collate_fixed_frame;
        let handler = Arc::new(Mutex::new(CollatedHandler {
            wrapped_trajectory_builder,
            last_logging_time: Instant::now(),
            rate_timers: BTreeMap::new(),
        }));

        let mut expected_sensor_id_strings = HashSet::new();
        for sensor_id in expected_sensor_ids {
            if sensor_id.r#type == SensorType::Landmark && !collate_landmarks {
                continue;
            }
            if sensor_id.r#type == SensorType::FixedFramePose && !collate_fixed_frame {
                continue;
            }
            expected_sensor_id_strings.insert(sensor_id.id.clone());
        }
        let callback_handler = Arc::clone(&handler);
        sensor_collator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .add_trajectory(
                trajectory_id,
                &expected_sensor_id_strings,
                Box::new(move |sensor_id: &str, data: Box<dyn Data>| {
                    lock(&callback_handler).handle_collated_sensor_data(sensor_id, data);
                }),
            );
        Self {
            sensor_collator,
            collate_landmarks,
            collate_fixed_frame,
            trajectory_id,
            handler,
        }
    }

    fn add_data(&mut self, data: Box<dyn Data>) {
        self.sensor_collator
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .add_sensor_data(self.trajectory_id, data);
    }
}

impl TrajectoryBuilderInterface for CollatedTrajectoryBuilder {
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) {
        self.add_data(make_dispatchable(sensor_id, timed_point_cloud_data.clone()));
    }

    fn add_imu_data(&mut self, sensor_id: &str, imu_data: &ImuData) {
        self.add_data(make_dispatchable(sensor_id, imu_data.clone()));
    }

    fn add_odometry_data(&mut self, sensor_id: &str, odometry_data: &OdometryData) {
        self.add_data(make_dispatchable(sensor_id, odometry_data.clone()));
    }

    fn add_fixed_frame_pose_data(
        &mut self,
        sensor_id: &str,
        fixed_frame_pose_data: &FixedFramePoseData,
    ) {
        if self.collate_fixed_frame {
            self.add_data(make_dispatchable(sensor_id, fixed_frame_pose_data.clone()));
            return;
        }
        lock(&self.handler)
            .wrapped_trajectory_builder
            .add_fixed_frame_pose_data(sensor_id, fixed_frame_pose_data);
    }

    fn add_landmark_data(&mut self, sensor_id: &str, landmark_data: &LandmarkData) {
        if self.collate_landmarks {
            self.add_data(make_dispatchable(sensor_id, landmark_data.clone()));
            return;
        }
        lock(&self.handler)
            .wrapped_trajectory_builder
            .add_landmark_data(sensor_id, landmark_data);
    }

    fn add_local_slam_result_data(&mut self, local_slam_result_data: Box<dyn LocalSlamResultData>) {
        self.add_data(local_slam_result_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::{Time, from_universal};
    use crate::mapping::internal::testing::mock_trajectory_builder::{
        CallKind, MockTrajectoryBuilder,
    };
    use crate::sensor::internal::collator::Collator;
    use crate::sensor::internal::trajectory_collator::TrajectoryCollator;
    use std::any::Any;

    fn sensor_id(r#type: SensorType, id: &str) -> SensorId {
        SensorId {
            r#type,
            id: id.to_string(),
        }
    }

    fn options(
        collate_fixed_frame: bool,
        collate_landmarks: bool,
    ) -> proto::TrajectoryBuilderOptions {
        proto::TrajectoryBuilderOptions {
            collate_fixed_frame,
            collate_landmarks,
            ..Default::default()
        }
    }

    fn imu(time: i64) -> ImuData {
        ImuData {
            time: from_universal(time),
            ..Default::default()
        }
    }

    fn odometry(time: i64) -> OdometryData {
        OdometryData {
            time: from_universal(time),
            ..Default::default()
        }
    }

    fn range(time: i64) -> TimedPointCloudData {
        TimedPointCloudData {
            time: from_universal(time),
            ..Default::default()
        }
    }

    #[derive(Clone)]
    struct FakeLocalSlamResult {
        time: Time,
    }

    impl Data for FakeLocalSlamResult {
        fn get_time(&self) -> Time {
            self.time
        }
        fn get_sensor_id(&self) -> &str {
            "local_slam"
        }
        fn add_to_trajectory_builder(
            self: Box<Self>,
            trajectory_builder: &mut dyn TrajectoryBuilderInterface,
        ) {
            trajectory_builder.add_local_slam_result_data(self);
        }
    }

    impl LocalSlamResultData for FakeLocalSlamResult {
        fn add_to_pose_graph(&self, _trajectory_id: i32, _pose_graph: &dyn Any) {}
    }

    fn kinds_and_times(builder: &MockTrajectoryBuilder) -> Vec<(CallKind, String, i64)> {
        builder
            .recorded_calls()
            .into_iter()
            .map(|(kind, id, time)| (kind, id, crate::common::time::to_universal(time)))
            .collect()
    }

    #[test]
    fn collates_sensor_data_in_time_order() {
        let collator: SharedCollator = Arc::new(Mutex::new(Collator::new()));
        let mock = MockTrajectoryBuilder::new();
        let expected: BTreeSet<SensorId> = [
            sensor_id(SensorType::Range, "points"),
            sensor_id(SensorType::Imu, "imu"),
            sensor_id(SensorType::Odometry, "odom"),
        ]
        .into_iter()
        .collect();
        let mut builder = CollatedTrajectoryBuilder::new(
            &options(true, false),
            Arc::clone(&collator),
            0,
            &expected,
            Box::new(mock.clone()),
        );
        builder.add_imu_data("imu", &imu(10));
        builder.add_imu_data("imu", &imu(30));
        builder.add_timed_point_cloud_data("points", &range(20));
        // Nothing is dispatched before every expected sensor has data.
        assert!(mock.recorded_calls().is_empty());
        builder.add_odometry_data("odom", &odometry(25));
        builder.add_timed_point_cloud_data("points", &range(40));
        builder.add_odometry_data("odom", &odometry(50));
        collator.lock().unwrap().finish_trajectory(0);
        collator.lock().unwrap().flush();
        assert_eq!(
            kinds_and_times(&mock),
            vec![
                (CallKind::Imu, "imu".to_string(), 10),
                (CallKind::TimedPointCloud, "points".to_string(), 20),
                (CallKind::Odometry, "odom".to_string(), 25),
                (CallKind::Imu, "imu".to_string(), 30),
                (CallKind::TimedPointCloud, "points".to_string(), 40),
                (CallKind::Odometry, "odom".to_string(), 50),
            ]
        );
    }

    #[test]
    fn uncollated_landmarks_and_fixed_frame_bypass_the_collator() {
        let collator: SharedCollator = Arc::new(Mutex::new(TrajectoryCollator::new()));
        let mock = MockTrajectoryBuilder::new();
        let expected: BTreeSet<SensorId> = [
            sensor_id(SensorType::Imu, "imu"),
            sensor_id(SensorType::Landmark, "landmarks"),
            sensor_id(SensorType::FixedFramePose, "gps"),
        ]
        .into_iter()
        .collect();
        let mut builder = CollatedTrajectoryBuilder::new(
            &options(false, false),
            Arc::clone(&collator),
            3,
            &expected,
            Box::new(mock.clone()),
        );
        builder.add_landmark_data(
            "landmarks",
            &LandmarkData {
                time: from_universal(5),
                landmark_observations: Vec::new(),
            },
        );
        builder.add_fixed_frame_pose_data(
            "gps",
            &FixedFramePoseData {
                time: from_universal(6),
                pose: None,
            },
        );
        // Delivered immediately, without waiting for other sensors.
        assert_eq!(
            kinds_and_times(&mock),
            vec![
                (CallKind::Landmark, "landmarks".to_string(), 5),
                (CallKind::FixedFramePose, "gps".to_string(), 6),
            ]
        );
        // Only "imu" is expected by the collator, so IMU data flows through
        // without waiting for landmarks or fixed frame poses.
        builder.add_imu_data("imu", &imu(7));
        builder.add_imu_data("imu", &imu(8));
        collator.lock().unwrap().finish_trajectory(3);
        collator.lock().unwrap().flush();
        assert_eq!(mock.recorded_calls().len(), 4);
        assert_eq!(mock.recorded_calls()[3].0, CallKind::Imu);
    }

    #[test]
    fn collated_landmarks_and_fixed_frame_go_through_the_collator() {
        let collator: SharedCollator = Arc::new(Mutex::new(TrajectoryCollator::new()));
        let mock = MockTrajectoryBuilder::new();
        let expected: BTreeSet<SensorId> = [
            sensor_id(SensorType::Imu, "imu"),
            sensor_id(SensorType::Landmark, "landmarks"),
            sensor_id(SensorType::FixedFramePose, "gps"),
        ]
        .into_iter()
        .collect();
        let mut builder = CollatedTrajectoryBuilder::new(
            &options(true, true),
            Arc::clone(&collator),
            1,
            &expected,
            Box::new(mock.clone()),
        );
        builder.add_imu_data("imu", &imu(30));
        builder.add_landmark_data(
            "landmarks",
            &LandmarkData {
                time: from_universal(20),
                landmark_observations: Vec::new(),
            },
        );
        // Waiting for "gps".
        assert!(mock.recorded_calls().is_empty());
        builder.add_fixed_frame_pose_data(
            "gps",
            &FixedFramePoseData {
                time: from_universal(10),
                pose: None,
            },
        );
        collator.lock().unwrap().finish_trajectory(1);
        collator.lock().unwrap().flush();
        assert_eq!(
            kinds_and_times(&mock),
            vec![
                (CallKind::FixedFramePose, "gps".to_string(), 10),
                (CallKind::Landmark, "landmarks".to_string(), 20),
                (CallKind::Imu, "imu".to_string(), 30),
            ]
        );
    }

    #[test]
    fn local_slam_result_data_is_collated() {
        let collator: SharedCollator = Arc::new(Mutex::new(TrajectoryCollator::new()));
        let mock = MockTrajectoryBuilder::new();
        let expected: BTreeSet<SensorId> = [sensor_id(SensorType::LocalSlamResult, "local_slam")]
            .into_iter()
            .collect();
        let mut builder = CollatedTrajectoryBuilder::new(
            &options(true, false),
            Arc::clone(&collator),
            0,
            &expected,
            Box::new(mock.clone()),
        );
        builder.add_local_slam_result_data(Box::new(FakeLocalSlamResult {
            time: from_universal(42),
        }));
        collator.lock().unwrap().finish_trajectory(0);
        collator.lock().unwrap().flush();
        assert_eq!(
            kinds_and_times(&mock),
            vec![(CallKind::LocalSlamResult, "local_slam".to_string(), 42)]
        );
    }
}
