//! Port of `cartographer/mapping/internal/global_trajectory_builder` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ class template
//! `GlobalTrajectoryBuilder<LocalTrajectoryBuilder, PoseGraph>` relies on
//! duck typing. Here the two template parameters are bounded by the traits
//! [`LocalTrajectoryBuilder`] (implemented by `LocalTrajectoryBuilder2D` and
//! `LocalTrajectoryBuilder3D`) and [`AddNodePoseGraph`] (implemented by
//! `PoseGraph2D` and `PoseGraph3D`). The template is
//! public (in C++ it is in an anonymous namespace) so that tests and the
//! map builder can name it.

use std::any::Any;
use std::sync::{Arc, PoisonError, RwLock};

use crate::common::time::Time;
use crate::mapping::id::NodeId;
use crate::mapping::internal::local_slam_result_data::LocalSlamResultData;
use crate::mapping::internal::motion_filter::MotionFilter;
use crate::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D;
use crate::mapping::internal::three_d::pose_graph_3d::PoseGraph3D;
use crate::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D;
use crate::mapping::internal::two_d::pose_graph_2d::PoseGraph2D;
use crate::mapping::pose_graph::PoseGraph;
use crate::mapping::submaps::{SharedSubmap, Submap};
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::trajectory_builder_interface::{
    InsertionResult, LocalSlamResultCallback, TrajectoryBuilderInterface,
};
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::metrics::counter::Counter;
use crate::metrics::family_factory::{FamilyFactory, Labels};
use crate::sensor::fixed_frame_pose_data::FixedFramePoseData;
use crate::sensor::imu_data::ImuData;
use crate::sensor::landmark_data::LandmarkData;
use crate::sensor::odometry_data::OdometryData;
use crate::sensor::range_data::RangeData;
use crate::sensor::timed_point_cloud_data::TimedPointCloudData;
use crate::transform::rigid_transform::Rigid3d;

/// The metrics (C++ file-level statics `kLocalSlamMatchingResults` and
/// `kLocalSlamInsertionResults`, null counters until
/// [`global_trajectory_builder_register_metrics`]).
#[derive(Clone, Copy)]
struct Metrics {
    local_slam_matching_results: &'static dyn Counter,
    local_slam_insertion_results: &'static dyn Counter,
}

static METRICS: RwLock<Option<Metrics>> = RwLock::new(None);

fn metrics() -> Metrics {
    METRICS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .unwrap_or_else(|| Metrics {
            local_slam_matching_results: <dyn Counter>::null(),
            local_slam_insertion_results: <dyn Counter>::null(),
        })
}

/// `LocalTrajectoryBuilder{2D,3D}::InsertionResult`, generic over the submap
/// type.
#[derive(Clone, Debug)]
pub struct LocalInsertionResult<S> {
    /// The constant data of the new trajectory node.
    pub constant_data: Arc<TrajectoryNodeData>,
    /// The submaps the range data was inserted into.
    pub insertion_submaps: Vec<SharedSubmap<S>>,
}

/// `LocalTrajectoryBuilder{2D,3D}::MatchingResult`, generic over the submap
/// type.
#[derive(Clone, Debug)]
pub struct LocalMatchingResult<S> {
    /// Time of the (last) range data.
    pub time: Time,
    /// Pose of the tracking frame in the local SLAM frame.
    pub local_pose: Rigid3d,
    /// The range data in the local SLAM frame.
    pub range_data_in_local: RangeData,
    /// `None` if dropped by the motion filter.
    pub insertion_result: Option<LocalInsertionResult<S>>,
}

/// The local trajectory builder interface used by [`GlobalTrajectoryBuilder`]
/// (the operations the C++ template calls on `LocalTrajectoryBuilder`).
pub trait LocalTrajectoryBuilder: Send + 'static {
    /// The submap type (`Submap2D` or `Submap3D`).
    type Submap: Submap;

    /// C++ `AddRangeData`: returns a matching result when range data
    /// accumulation completed.
    fn add_range_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) -> Option<LocalMatchingResult<Self::Submap>>;

    /// C++ `AddImuData`.
    fn add_imu_data(&mut self, imu_data: &ImuData);

    /// C++ `AddOdometryData`.
    fn add_odometry_data(&mut self, odometry_data: &OdometryData);
}

impl LocalTrajectoryBuilder for LocalTrajectoryBuilder2D {
    type Submap = Submap2D;

    fn add_range_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) -> Option<LocalMatchingResult<Submap2D>> {
        let result = LocalTrajectoryBuilder2D::add_range_data(
            self,
            sensor_id,
            timed_point_cloud_data.clone(),
        )?;
        Some(LocalMatchingResult {
            time: result.time,
            local_pose: result.local_pose,
            range_data_in_local: result.range_data_in_local,
            insertion_result: result.insertion_result.map(|insertion_result| {
                LocalInsertionResult {
                    constant_data: insertion_result.constant_data,
                    insertion_submaps: insertion_result.insertion_submaps,
                }
            }),
        })
    }

    fn add_imu_data(&mut self, imu_data: &ImuData) {
        LocalTrajectoryBuilder2D::add_imu_data(self, imu_data);
    }

    fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        LocalTrajectoryBuilder2D::add_odometry_data(self, odometry_data);
    }
}

impl LocalTrajectoryBuilder for LocalTrajectoryBuilder3D {
    type Submap = Submap3D;

    fn add_range_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) -> Option<LocalMatchingResult<Submap3D>> {
        let result =
            LocalTrajectoryBuilder3D::add_range_data(self, sensor_id, timed_point_cloud_data)?;
        Some(LocalMatchingResult {
            time: result.time,
            local_pose: result.local_pose,
            range_data_in_local: result.range_data_in_local,
            insertion_result: result.insertion_result.map(|insertion_result| {
                LocalInsertionResult {
                    constant_data: insertion_result.constant_data,
                    insertion_submaps: insertion_result.insertion_submaps,
                }
            }),
        })
    }

    fn add_imu_data(&mut self, imu_data: &ImuData) {
        LocalTrajectoryBuilder3D::add_imu_data(self, imu_data);
    }

    fn add_odometry_data(&mut self, odometry_data: &OdometryData) {
        LocalTrajectoryBuilder3D::add_odometry_data(self, odometry_data);
    }
}

/// A pose graph that accepts nodes from local SLAM (C++
/// `PoseGraph2D::AddNode` / `PoseGraph3D::AddNode`, which are not part of the
/// `PoseGraph` interface).
pub trait AddNodePoseGraph: PoseGraph + 'static {
    /// The submap type (`Submap2D` or `Submap3D`).
    type Submap: Submap;

    /// Adds a new node with `constant_data`. Its `constant_data.local_pose`
    /// was determined by scan matching against `insertion_submaps[0]` and the
    /// node data was inserted into the `insertion_submaps`.
    fn add_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Self::Submap>],
    ) -> NodeId;
}

impl AddNodePoseGraph for PoseGraph2D {
    type Submap = Submap2D;

    fn add_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Submap2D>],
    ) -> NodeId {
        PoseGraph2D::add_node(self, constant_data, trajectory_id, insertion_submaps)
    }
}

impl AddNodePoseGraph for PoseGraph3D {
    type Submap = Submap3D;

    fn add_node(
        &self,
        constant_data: Arc<TrajectoryNodeData>,
        trajectory_id: i32,
        insertion_submaps: &[SharedSubmap<Submap3D>],
    ) -> NodeId {
        PoseGraph3D::add_node(self, constant_data, trajectory_id, insertion_submaps)
    }
}

/// Wires a local trajectory builder to a pose graph: local SLAM results are
/// added as nodes, other sensor data is forwarded to the pose graph.
///
/// RUST PORT NOTE: the non-owning C++ `PoseGraph*` is an `Arc<P>`; the
/// possibly null `std::unique_ptr<LocalTrajectoryBuilder>` is an
/// `Option<L>`.
pub struct GlobalTrajectoryBuilder<L, P> {
    trajectory_id: i32,
    pose_graph: Arc<P>,
    local_trajectory_builder: Option<L>,
    local_slam_result_callback: Option<LocalSlamResultCallback>,
    pose_graph_odometry_motion_filter: Option<MotionFilter>,
}

impl<L, P> std::fmt::Debug for GlobalTrajectoryBuilder<L, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalTrajectoryBuilder")
            .field("trajectory_id", &self.trajectory_id)
            .field(
                "has_local_trajectory_builder",
                &self.local_trajectory_builder.is_some(),
            )
            .field(
                "pose_graph_odometry_motion_filter",
                &self.pose_graph_odometry_motion_filter,
            )
            .finish_non_exhaustive()
    }
}

impl<L, P> GlobalTrajectoryBuilder<L, P>
where
    L: LocalTrajectoryBuilder,
    P: AddNodePoseGraph<Submap = L::Submap>,
{
    /// Passing `None` for `local_trajectory_builder` is acceptable, but no
    /// `TimedPointCloudData` may be added in that case.
    pub fn new(
        local_trajectory_builder: Option<L>,
        trajectory_id: i32,
        pose_graph: Arc<P>,
        local_slam_result_callback: Option<LocalSlamResultCallback>,
        pose_graph_odometry_motion_filter: Option<MotionFilter>,
    ) -> Self {
        Self {
            trajectory_id,
            pose_graph,
            local_trajectory_builder,
            local_slam_result_callback,
            pose_graph_odometry_motion_filter,
        }
    }
}

impl<L, P> TrajectoryBuilderInterface for GlobalTrajectoryBuilder<L, P>
where
    L: LocalTrajectoryBuilder,
    P: AddNodePoseGraph<Submap = L::Submap>,
{
    fn add_timed_point_cloud_data(
        &mut self,
        sensor_id: &str,
        timed_point_cloud_data: &TimedPointCloudData,
    ) {
        let local_trajectory_builder = self
            .local_trajectory_builder
            .as_mut()
            .expect("Cannot add TimedPointCloudData without a LocalTrajectoryBuilder.");
        let Some(matching_result) =
            local_trajectory_builder.add_range_data(sensor_id, timed_point_cloud_data)
        else {
            // The range data has not been fully accumulated yet.
            return;
        };
        let metrics = metrics();
        metrics.local_slam_matching_results.increment();
        let mut insertion_result = None;
        if let Some(local_insertion_result) = matching_result.insertion_result {
            metrics.local_slam_insertion_results.increment();
            let node_id = self.pose_graph.add_node(
                Arc::clone(&local_insertion_result.constant_data),
                self.trajectory_id,
                &local_insertion_result.insertion_submaps,
            );
            assert_eq!(
                node_id.trajectory_id, self.trajectory_id,
                "Check failed: node_id.trajectory_id == trajectory_id_"
            );
            insertion_result = Some(InsertionResult {
                node_id,
                constant_data: local_insertion_result.constant_data,
                insertion_submaps: local_insertion_result
                    .insertion_submaps
                    .into_iter()
                    .map(|submap| submap as Arc<RwLock<dyn Submap>>)
                    .collect(),
            });
        }
        if let Some(local_slam_result_callback) = &self.local_slam_result_callback {
            local_slam_result_callback(
                self.trajectory_id,
                matching_result.time,
                matching_result.local_pose,
                matching_result.range_data_in_local,
                insertion_result,
            );
        }
    }

    fn add_imu_data(&mut self, _sensor_id: &str, imu_data: &ImuData) {
        if let Some(local_trajectory_builder) = &mut self.local_trajectory_builder {
            local_trajectory_builder.add_imu_data(imu_data);
        }
        self.pose_graph.add_imu_data(self.trajectory_id, imu_data);
    }

    fn add_odometry_data(&mut self, _sensor_id: &str, odometry_data: &OdometryData) {
        assert!(odometry_data.pose.is_valid(), "{:?}", odometry_data.pose);
        if let Some(local_trajectory_builder) = &mut self.local_trajectory_builder {
            local_trajectory_builder.add_odometry_data(odometry_data);
        }
        // TODO(MichaelGrupp): Instead of having an optional filter on this
        // level, odometry could be marginalized between nodes in the pose
        // graph. Related issue: cartographer-project/cartographer/#1768
        if let Some(filter) = &mut self.pose_graph_odometry_motion_filter
            && filter.is_similar(odometry_data.time, &odometry_data.pose)
        {
            return;
        }
        self.pose_graph
            .add_odometry_data(self.trajectory_id, odometry_data);
    }

    fn add_fixed_frame_pose_data(
        &mut self,
        _sensor_id: &str,
        fixed_frame_pose: &FixedFramePoseData,
    ) {
        if let Some(pose) = &fixed_frame_pose.pose {
            assert!(pose.is_valid(), "{pose:?}");
        }
        self.pose_graph
            .add_fixed_frame_pose_data(self.trajectory_id, fixed_frame_pose);
    }

    fn add_landmark_data(&mut self, _sensor_id: &str, landmark_data: &LandmarkData) {
        self.pose_graph
            .add_landmark_data(self.trajectory_id, landmark_data);
    }

    fn add_local_slam_result_data(&mut self, local_slam_result_data: Box<dyn LocalSlamResultData>) {
        assert!(
            self.local_trajectory_builder.is_none(),
            "Can't add LocalSlamResultData with local_trajectory_builder_ present."
        );
        local_slam_result_data
            .add_to_pose_graph(self.trajectory_id, self.pose_graph.as_ref() as &dyn Any);
    }
}

/// Creates a 2D global trajectory builder (C++
/// `CreateGlobalTrajectoryBuilder2D`).
pub fn create_global_trajectory_builder_2d(
    local_trajectory_builder: Option<LocalTrajectoryBuilder2D>,
    trajectory_id: i32,
    pose_graph: Arc<PoseGraph2D>,
    local_slam_result_callback: Option<LocalSlamResultCallback>,
    pose_graph_odometry_motion_filter: Option<MotionFilter>,
) -> Box<dyn TrajectoryBuilderInterface> {
    Box::new(GlobalTrajectoryBuilder::new(
        local_trajectory_builder,
        trajectory_id,
        pose_graph,
        local_slam_result_callback,
        pose_graph_odometry_motion_filter,
    ))
}

/// Creates a 3D global trajectory builder (C++
/// `CreateGlobalTrajectoryBuilder3D`).
///
pub fn create_global_trajectory_builder_3d(
    local_trajectory_builder: Option<LocalTrajectoryBuilder3D>,
    trajectory_id: i32,
    pose_graph: Arc<PoseGraph3D>,
    local_slam_result_callback: Option<LocalSlamResultCallback>,
    pose_graph_odometry_motion_filter: Option<MotionFilter>,
) -> Box<dyn TrajectoryBuilderInterface> {
    Box::new(GlobalTrajectoryBuilder::new(
        local_trajectory_builder,
        trajectory_id,
        pose_graph,
        local_slam_result_callback,
        pose_graph_odometry_motion_filter,
    ))
}

/// Registers the global trajectory builder metrics (C++
/// `GlobalTrajectoryBuilderRegisterMetrics`).
pub fn global_trajectory_builder_register_metrics(family_factory: &mut dyn FamilyFactory) {
    let results = family_factory.new_counter_family(
        "mapping_global_trajectory_builder_local_slam_results",
        "Local SLAM results",
    );
    let labels = |value: &str| -> Labels {
        [("type".to_string(), value.to_string())]
            .into_iter()
            .collect()
    };
    *METRICS.write().unwrap_or_else(PoisonError::into_inner) = Some(Metrics {
        local_slam_matching_results: results.add(&labels("MatchingResult")),
        local_slam_insertion_results: results.add(&labels("InsertionResult")),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::thread_pool::{ThreadPool, ThreadPoolInterface};
    use crate::common::time::from_universal;
    use crate::mapping::internal::optimization::optimization_problem_2d::OptimizationProblem2D;
    use crate::mapping::internal::testing::mock_pose_graph::{MockPoseGraph, PoseGraphCall};
    use crate::mapping::internal::testing::test_helpers::{
        generate_fake_range_measurements, resolve_lua_parameters,
    };
    use crate::mapping::internal::two_d::local_trajectory_builder_options_2d::create_local_trajectory_builder_options_2d;
    use crate::mapping::pose_graph::create_pose_graph_options;
    use crate::mapping::pose_graph_interface::PoseGraphInterface;
    use crate::mapping::proto;
    use crate::mapping::two_d::submap_2d::{ActiveSubmaps2D, create_submaps_options_2d};
    use crate::sensor::data::Data;
    use crate::sensor::point_cloud::PointCloud;
    use nalgebra::{DVector, Quaternion, Vector3};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    type Log = Arc<Mutex<Vec<&'static str>>>;

    /// Fake local trajectory builder returning scripted results.
    struct FakeLocalTrajectoryBuilder {
        results: VecDeque<Option<LocalMatchingResult<Submap2D>>>,
        log: Log,
    }

    impl LocalTrajectoryBuilder for FakeLocalTrajectoryBuilder {
        type Submap = Submap2D;

        fn add_range_data(
            &mut self,
            _sensor_id: &str,
            _timed_point_cloud_data: &TimedPointCloudData,
        ) -> Option<LocalMatchingResult<Submap2D>> {
            self.log.lock().unwrap().push("AddRangeData");
            self.results.pop_front().flatten()
        }

        fn add_imu_data(&mut self, _imu_data: &ImuData) {
            self.log.lock().unwrap().push("AddImuData");
        }

        fn add_odometry_data(&mut self, _odometry_data: &OdometryData) {
            self.log.lock().unwrap().push("AddOdometryData");
        }
    }

    /// `(trajectory_id, time, local_pose, num_returns, insertion_result)`.
    type Recorded = Arc<Mutex<Vec<(i32, Time, Rigid3d, usize, Option<InsertionResult>)>>>;

    fn recording_callback(recorded: &Recorded) -> LocalSlamResultCallback {
        let recorded = Arc::clone(recorded);
        Arc::new(
            move |trajectory_id, time, local_pose, range_data: RangeData, insertion_result| {
                recorded.lock().unwrap().push((
                    trajectory_id,
                    time,
                    local_pose,
                    range_data.returns.len(),
                    insertion_result,
                ));
            },
        )
    }

    fn submap() -> SharedSubmap<Submap2D> {
        let mut active_submaps =
            ActiveSubmaps2D::new(&create_submaps_options_2d(&resolve_lua_parameters(
                "include 'trajectory_builder_2d.lua' return TRAJECTORY_BUILDER_2D.submaps",
            )));
        active_submaps
            .insert_range_data(&RangeData::default())
            .remove(0)
    }

    fn node_data(time: i64) -> Arc<TrajectoryNodeData> {
        Arc::new(TrajectoryNodeData {
            time: from_universal(time),
            gravity_alignment: Quaternion::identity(),
            filtered_gravity_aligned_point_cloud: PointCloud::default(),
            high_resolution_point_cloud: PointCloud::default(),
            low_resolution_point_cloud: PointCloud::default(),
            rotational_scan_matcher_histogram: DVector::zeros(0),
            local_pose: Rigid3d::identity(),
        })
    }

    fn matching_result(time: i64, inserted: bool) -> LocalMatchingResult<Submap2D> {
        LocalMatchingResult {
            time: from_universal(time),
            local_pose: Rigid3d::from_translation(Vector3::new(1., 2., 3.)),
            range_data_in_local: RangeData::default(),
            insertion_result: inserted.then(|| LocalInsertionResult {
                constant_data: node_data(time),
                insertion_submaps: vec![submap()],
            }),
        }
    }

    type Fixture = (
        GlobalTrajectoryBuilder<FakeLocalTrajectoryBuilder, MockPoseGraph>,
        Arc<MockPoseGraph>,
        Log,
        Recorded,
    );

    fn builder(
        results: Vec<Option<LocalMatchingResult<Submap2D>>>,
        with_local_builder: bool,
        filter: Option<MotionFilter>,
    ) -> Fixture {
        let pose_graph = MockPoseGraph::new();
        let log = Log::default();
        let recorded = Recorded::default();
        let local = with_local_builder.then(|| FakeLocalTrajectoryBuilder {
            results: results.into(),
            log: Arc::clone(&log),
        });
        let builder = GlobalTrajectoryBuilder::new(
            local,
            7,
            Arc::clone(&pose_graph),
            Some(recording_callback(&recorded)),
            filter,
        );
        (builder, pose_graph, log, recorded)
    }

    fn range(time: i64) -> TimedPointCloudData {
        TimedPointCloudData {
            time: from_universal(time),
            ..Default::default()
        }
    }

    fn odometry(time: i64, x: f64) -> OdometryData {
        OdometryData {
            time: from_universal(time),
            pose: Rigid3d::from_translation(Vector3::new(x, 0., 0.)),
        }
    }

    #[test]
    fn accumulating_range_data_reports_nothing() {
        let (mut builder, pose_graph, log, recorded) = builder(vec![None], true, None);
        builder.add_timed_point_cloud_data("points", &range(10));
        assert_eq!(*log.lock().unwrap(), ["AddRangeData"]);
        assert!(pose_graph.recorded_calls().is_empty());
        assert!(recorded.lock().unwrap().is_empty());
    }

    #[test]
    fn filtered_range_data_reports_without_insertion_result() {
        let (mut builder, pose_graph, _log, recorded) =
            builder(vec![Some(matching_result(10, false))], true, None);
        builder.add_timed_point_cloud_data("points", &range(10));
        assert!(pose_graph.recorded_calls().is_empty());
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let (trajectory_id, time, local_pose, _, insertion_result) = &recorded[0];
        assert_eq!(*trajectory_id, 7);
        assert_eq!(*time, from_universal(10));
        assert_eq!(local_pose.translation(), &Vector3::new(1., 2., 3.));
        assert!(insertion_result.is_none());
    }

    #[test]
    fn inserted_range_data_adds_node_and_reports_it() {
        let (mut builder, pose_graph, _log, recorded) = builder(
            vec![
                Some(matching_result(10, true)),
                Some(matching_result(20, true)),
            ],
            true,
            None,
        );
        builder.add_timed_point_cloud_data("points", &range(10));
        builder.add_timed_point_cloud_data("points", &range(20));
        assert_eq!(
            pose_graph.recorded_calls(),
            vec![
                PoseGraphCall::AddNode {
                    trajectory_id: 7,
                    time: from_universal(10),
                    num_insertion_submaps: 1,
                },
                PoseGraphCall::AddNode {
                    trajectory_id: 7,
                    time: from_universal(20),
                    num_insertion_submaps: 1,
                },
            ]
        );
        let recorded = recorded.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        for (i, entry) in recorded.iter().enumerate() {
            let insertion_result = entry.4.as_ref().unwrap();
            assert_eq!(insertion_result.node_id, NodeId::new(7, i as i32));
            assert_eq!(insertion_result.insertion_submaps.len(), 1);
            assert_eq!(insertion_result.constant_data.time, entry.1);
        }
    }

    #[test]
    fn imu_data_goes_to_local_builder_and_pose_graph() {
        let imu = ImuData {
            time: from_universal(5),
            ..Default::default()
        };
        let (mut with_local, pose_graph, log, _) = builder(vec![], true, None);
        with_local.add_imu_data("imu", &imu);
        assert_eq!(*log.lock().unwrap(), ["AddImuData"]);
        assert_eq!(
            pose_graph.recorded_calls(),
            vec![PoseGraphCall::AddImuData(7, from_universal(5))]
        );

        let (mut without_local, pose_graph, log, _) = builder(vec![], false, None);
        without_local.add_imu_data("imu", &imu);
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(pose_graph.recorded_calls().len(), 1);
    }

    #[test]
    fn odometry_motion_filter_only_affects_pose_graph() {
        let filter = MotionFilter::new(&proto::MotionFilterOptions {
            max_time_seconds: 10.,
            max_distance_meters: 1.,
            max_angle_radians: 1.,
        });
        let (mut builder, pose_graph, log, _) = builder(vec![], true, Some(filter));
        builder.add_odometry_data("odom", &odometry(0, 0.));
        // Similar to the previous one: filtered out for the pose graph.
        builder.add_odometry_data("odom", &odometry(1, 0.5));
        // Moved far enough.
        builder.add_odometry_data("odom", &odometry(2, 2.));
        assert_eq!(
            *log.lock().unwrap(),
            ["AddOdometryData", "AddOdometryData", "AddOdometryData"]
        );
        assert_eq!(
            pose_graph.recorded_calls(),
            vec![
                PoseGraphCall::AddOdometryData(7, from_universal(0)),
                PoseGraphCall::AddOdometryData(7, from_universal(2)),
            ]
        );
    }

    #[test]
    fn odometry_without_filter_always_goes_to_pose_graph() {
        let (mut builder, pose_graph, _, _) = builder(vec![], true, None);
        builder.add_odometry_data("odom", &odometry(0, 0.));
        builder.add_odometry_data("odom", &odometry(1, 0.));
        assert_eq!(pose_graph.recorded_calls().len(), 2);
    }

    #[test]
    #[should_panic]
    fn invalid_odometry_pose_panics() {
        let (mut builder, _, _, _) = builder(vec![], true, None);
        builder.add_odometry_data("odom", &odometry(0, f64::NAN));
    }

    #[test]
    fn fixed_frame_and_landmarks_go_to_pose_graph() {
        let (mut builder, pose_graph, log, _) = builder(vec![], true, None);
        builder.add_fixed_frame_pose_data(
            "gps",
            &FixedFramePoseData {
                time: from_universal(3),
                pose: Some(Rigid3d::identity()),
            },
        );
        builder.add_fixed_frame_pose_data(
            "gps",
            &FixedFramePoseData {
                time: from_universal(4),
                pose: None,
            },
        );
        builder.add_landmark_data(
            "landmarks",
            &LandmarkData {
                time: from_universal(5),
                landmark_observations: Vec::new(),
            },
        );
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(
            pose_graph.recorded_calls(),
            vec![
                PoseGraphCall::AddFixedFramePoseData(7, from_universal(3)),
                PoseGraphCall::AddFixedFramePoseData(7, from_universal(4)),
                PoseGraphCall::AddLandmarkData(7, from_universal(5)),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Cannot add TimedPointCloudData without a LocalTrajectoryBuilder.")]
    fn range_data_without_local_builder_panics() {
        let (mut builder, _, _, _) = builder(vec![], false, None);
        builder.add_timed_point_cloud_data("points", &range(1));
    }

    struct FakeLocalSlamResult {
        added_to: Arc<Mutex<Vec<i32>>>,
    }

    impl Data for FakeLocalSlamResult {
        fn get_time(&self) -> Time {
            from_universal(1)
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
        fn add_to_pose_graph(&self, trajectory_id: i32, pose_graph: &dyn Any) {
            assert!(pose_graph.downcast_ref::<MockPoseGraph>().is_some());
            self.added_to.lock().unwrap().push(trajectory_id);
        }
    }

    #[test]
    fn local_slam_result_data_is_added_to_pose_graph() {
        let (mut builder, _, _, _) = builder(vec![], false, None);
        let added_to = Arc::new(Mutex::new(Vec::new()));
        builder.add_local_slam_result_data(Box::new(FakeLocalSlamResult {
            added_to: Arc::clone(&added_to),
        }));
        assert_eq!(*added_to.lock().unwrap(), [7]);
    }

    #[test]
    #[should_panic(
        expected = "Can't add LocalSlamResultData with local_trajectory_builder_ present."
    )]
    fn local_slam_result_data_with_local_builder_panics() {
        let (mut builder, _, _, _) = builder(vec![], true, None);
        builder.add_local_slam_result_data(Box::new(FakeLocalSlamResult {
            added_to: Arc::default(),
        }));
    }

    #[test]
    fn global_trajectory_builder_2d_end_to_end() {
        let thread_pool = Arc::new(ThreadPool::new(1));
        let pose_graph_options = create_pose_graph_options(&resolve_lua_parameters(
            "include 'pose_graph.lua' return POSE_GRAPH",
        ));
        let pose_graph = Arc::new(PoseGraph2D::new(
            &pose_graph_options,
            Box::new(OptimizationProblem2D::new(
                &pose_graph_options
                    .optimization_problem_options
                    .unwrap_or_default(),
            )),
            Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
        ));
        let local_options = create_local_trajectory_builder_options_2d(&resolve_lua_parameters(
            "include 'trajectory_builder_2d.lua' \
             TRAJECTORY_BUILDER_2D.use_imu_data = false \
             return TRAJECTORY_BUILDER_2D",
        ));
        let recorded = Recorded::default();
        let mut builder = create_global_trajectory_builder_2d(
            Some(LocalTrajectoryBuilder2D::new(&local_options, &["points"])),
            0,
            Arc::clone(&pose_graph),
            Some(recording_callback(&recorded)),
            None,
        );
        for measurement in generate_fake_range_measurements(5., 5., 0.1) {
            builder.add_timed_point_cloud_data("points", &measurement);
        }
        let recorded = recorded.lock().unwrap();
        assert!(!recorded.is_empty());
        let node_ids: Vec<NodeId> = recorded
            .iter()
            .filter_map(|entry| entry.4.as_ref().map(|result| result.node_id))
            .collect();
        assert!(!node_ids.is_empty());
        for (i, node_id) in node_ids.iter().enumerate() {
            assert_eq!(*node_id, NodeId::new(0, i as i32));
        }
        assert!(recorded.iter().all(|entry| entry.0 == 0 && entry.3 > 0));
        pose_graph.run_final_optimization();
        assert_eq!(pose_graph.get_trajectory_nodes().size(), node_ids.len());
        // The pose graph must be dropped (waiting for its computations)
        // before the thread pool.
        drop(builder);
        drop(pose_graph);
    }
}
