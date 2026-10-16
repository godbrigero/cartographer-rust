//! Port of `cartographer/cloud/internal/client_server_test.cc`.
//!
//! All servers listen on ephemeral localhost ports (the C++ hard-codes
//! `0.0.0.0:50051`/`50052`).
//!
//! The C++ tests that need the real `MapBuilder` (`CreateMapBuilder`) are
//! ported faithfully but `#[ignore]`d until the concrete `MapBuilder` port is
//! merged (replace the body of [`create_map_builder`] then). The tests using
//! gmock mocks in C++ (`*WithMock`) run against the scriptable
//! `FakeMapBuilder`; additional `fake_*` tests exercise the same client/server
//! paths (local SLAM result streaming, uplink uploading, state loading and
//! serialization, invalid requests) with the fake map builder.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use cartographer::common::time::{Time, from_seconds, from_universal};
use cartographer::io::internal::in_memory_proto_stream::{
    ForwardingProtoStreamWriter, InMemoryProtoStreamReader,
};
use cartographer::io::internal::testing::test_helpers::proto_reader_from_protos;
use cartographer::io::proto_stream_deserializer::ProtoStreamDeserializer;
use cartographer::io::proto_stream_interface::{ProtoMessage, ProtoStreamWriterInterface};
use cartographer::mapping::id::{NodeId, SubmapId};
use cartographer::mapping::internal::testing::mock_trajectory_builder::{
    CallKind, MockTrajectoryBuilder,
};
use cartographer::mapping::internal::testing::test_helpers::{
    generate_fake_range_measurements, resolve_lua_parameters,
};
use cartographer::mapping::map_builder_interface::MapBuilderInterface;
use cartographer::mapping::pose_graph_interface::TrajectoryState;
use cartographer::mapping::proto as mapping_proto;
use cartographer::mapping::submaps::Submap;
use cartographer::mapping::trajectory_builder_interface::{
    InsertionResult, LocalSlamResultCallback, SensorId, SensorType,
    create_trajectory_builder_options,
};
use cartographer::mapping::trajectory_node::TrajectoryNodeData;
use cartographer::mapping::two_d::submap_2d::{ActiveSubmaps2D, create_submaps_options_2d};
use cartographer::sensor::imu_data::ImuData;
use cartographer::sensor::point_cloud::PointCloud;
use cartographer::sensor::range_data::RangeData;
use cartographer::sensor::rangefinder_point::RangefinderPoint;
use cartographer::transform::rigid_transform::Rigid3d;
use cartographer_cloud::internal::map_builder_context_interface::BoxedMapBuilder;
use cartographer_cloud::internal::testing::fake_map_builder::{
    FakeMapBuilder, FakePoseGraph, MapBuilderCall, MapBuilderScript, PoseGraphCall,
};
use cartographer_cloud::{
    MapBuilderServer, MapBuilderServerInterface, MapBuilderStub, create_map_builder_server_options,
    proto,
};
use nalgebra::{DVector, Quaternion, Vector3};

const CLIENT_ID: &str = "CLIENT_ID";
const DURATION: f64 = 4.; // Seconds.
const TIME_STEP: f64 = 0.1; // Seconds.
const TRAVEL_DISTANCE: f64 = 1.2; // Meters.

fn imu_sensor_id() -> SensorId {
    SensorId {
        r#type: SensorType::Imu,
        id: "imu".to_string(),
    }
}

fn range_sensor_id() -> SensorId {
    SensorId {
        r#type: SensorType::Range,
        id: "range".to_string(),
    }
}

fn sensor_ids(ids: &[SensorId]) -> BTreeSet<SensorId> {
    ids.iter().cloned().collect()
}

/// Creates the real map builder (C++ `mapping::CreateMapBuilder`).
///
/// GAP: the concrete `MapBuilder` is ported separately; once merged, replace
/// the body with
/// `Box::new(cartographer::mapping::map_builder::MapBuilder::new(options))`
/// (or the corresponding factory) and remove the `#[ignore]`s.
fn create_map_builder(_options: &mapping_proto::MapBuilderOptions) -> BoxedMapBuilder {
    panic!("needs MapBuilder; enabled after merge")
}

/// Returns `127.0.0.1:<port>` of a free ephemeral port (for servers that
/// must be reachable before they start).
fn reserve_local_address() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    drop(listener);
    address
}

fn wait_until(what: &str, timeout: Duration, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridType {
    ProbabilityGrid,
    Tsdf,
}

/// Collects the local SLAM results received by the client (C++
/// `local_slam_result_callback_` and its condition variable).
#[derive(Default)]
struct LocalSlamResults {
    state: Mutex<(Vec<Rigid3d>, usize /* number_of_insertion_results */)>,
    condition: Condvar,
}

impl LocalSlamResults {
    fn callback(self: &Arc<Self>) -> LocalSlamResultCallback {
        let results = Arc::clone(self);
        Arc::new(
            move |_trajectory_id, _time, local_pose, _range_data, insertion_result| {
                let mut state = results.state.lock().unwrap();
                if insertion_result.is_some() {
                    state.1 += 1;
                }
                state.0.push(local_pose);
                drop(state);
                results.condition.notify_all();
            },
        )
    }

    /// C++ `WaitForLocalSlamResults(size)`.
    fn wait_for(&self, size: usize) {
        let state = self.state.lock().unwrap();
        let (state, timeout) = self
            .condition
            .wait_timeout_while(state, Duration::from_secs(120), |state| state.0.len() < size)
            .unwrap();
        assert!(
            !timeout.timed_out(),
            "only {} local SLAM results",
            state.0.len()
        );
    }

    fn poses(&self) -> Vec<Rigid3d> {
        self.state.lock().unwrap().0.clone()
    }

    fn number_of_insertion_results(&self) -> usize {
        self.state.lock().unwrap().1
    }
}

/// C++ `ClientServerTestBase`.
struct ClientServerTest {
    map_builder_server_options: proto::MapBuilderServerOptions,
    uploading_map_builder_server_options: proto::MapBuilderServerOptions,
    trajectory_builder_options: mapping_proto::TrajectoryBuilderOptions,
    server: Option<MapBuilderServer>,
    uploading_server: Option<MapBuilderServer>,
    stub: Option<MapBuilderStub>,
    stub_for_uploading_server: Option<MapBuilderStub>,
    local_slam_results: Arc<LocalSlamResults>,
}

impl ClientServerTest {
    /// C++ `SetUp()`. The main server address is reserved up front so that
    /// the uploading server can be configured before the main server starts.
    fn set_up() -> Self {
        let main_address = reserve_local_address();
        let map_builder_server_options = create_map_builder_server_options(&resolve_lua_parameters(
            &format!(
                r#"
                include "map_builder_server.lua"
                MAP_BUILDER.use_trajectory_builder_2d = true
                MAP_BUILDER.pose_graph.optimize_every_n_nodes = 0
                MAP_BUILDER_SERVER.num_event_threads = 1
                MAP_BUILDER_SERVER.num_grpc_threads = 1
                MAP_BUILDER_SERVER.uplink_server_address = ""
                MAP_BUILDER_SERVER.server_address = "{main_address}"
                return MAP_BUILDER_SERVER"#
            ),
        ));
        let uploading_map_builder_server_options =
            create_map_builder_server_options(&resolve_lua_parameters(&format!(
                r#"
                include "map_builder_server.lua"
                MAP_BUILDER.use_trajectory_builder_2d = true
                MAP_BUILDER.pose_graph.optimize_every_n_nodes = 0
                MAP_BUILDER_SERVER.num_event_threads = 1
                MAP_BUILDER_SERVER.num_grpc_threads = 1
                MAP_BUILDER_SERVER.uplink_server_address = "{main_address}"
                MAP_BUILDER_SERVER.server_address = "127.0.0.1:0"
                MAP_BUILDER_SERVER.upload_batch_size = 1
                return MAP_BUILDER_SERVER"#
            )));
        assert_ne!(
            map_builder_server_options.server_address,
            uploading_map_builder_server_options.server_address
        );
        let trajectory_builder_options = create_trajectory_builder_options(&resolve_lua_parameters(
            r#"
            include "trajectory_builder.lua"
            TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false
            TRAJECTORY_BUILDER.trajectory_builder_2d.submaps.num_range_data = 4
            return TRAJECTORY_BUILDER"#,
        ));
        Self {
            map_builder_server_options,
            uploading_map_builder_server_options,
            trajectory_builder_options,
            server: None,
            uploading_server: None,
            stub: None,
            stub_for_uploading_server: None,
            local_slam_results: Arc::new(LocalSlamResults::default()),
        }
    }

    fn set_up_with_grid_type(grid_type: GridType) -> Self {
        let mut test = Self::set_up();
        if grid_type == GridType::Tsdf {
            test.set_options_to_tsdf_2d();
        }
        test
    }

    fn server(&self) -> &MapBuilderServer {
        self.server.as_ref().unwrap()
    }

    fn uploading_server(&self) -> &MapBuilderServer {
        self.uploading_server.as_ref().unwrap()
    }

    fn stub(&mut self) -> &mut MapBuilderStub {
        self.stub.as_mut().unwrap()
    }

    fn stub_for_uploading_server(&mut self) -> &mut MapBuilderStub {
        self.stub_for_uploading_server.as_mut().unwrap()
    }

    /// C++ `InitializeRealServer()`.
    fn initialize_real_server(&mut self) {
        let map_builder = create_map_builder(
            self.map_builder_server_options
                .map_builder_options
                .as_ref()
                .unwrap(),
        );
        self.server = Some(MapBuilderServer::new(
            &self.map_builder_server_options,
            map_builder,
        ));
    }

    /// C++ `InitializeRealUploadingServer()`.
    fn initialize_real_uploading_server(&mut self) {
        let map_builder = create_map_builder(
            self.uploading_map_builder_server_options
                .map_builder_options
                .as_ref()
                .unwrap(),
        );
        self.uploading_server = Some(MapBuilderServer::new(
            &self.uploading_map_builder_server_options,
            map_builder,
        ));
    }

    /// C++ `InitializeServerWithMockMapBuilder()`; returns the fake's script
    /// and pose graph.
    fn initialize_server_with_fake_map_builder(
        &mut self,
        trajectory_builder: Option<MockTrajectoryBuilder>,
    ) -> (Arc<Mutex<MapBuilderScript>>, Arc<FakePoseGraph>) {
        let map_builder = FakeMapBuilder::new(trajectory_builder);
        let script = map_builder.script();
        let pose_graph = map_builder.fake_pose_graph();
        self.server = Some(MapBuilderServer::new(
            &self.map_builder_server_options,
            Box::new(map_builder),
        ));
        assert_eq!(
            pose_graph.calls(),
            vec![PoseGraphCall::SetGlobalSlamOptimizationCallback]
        );
        (script, pose_graph)
    }

    /// Like [`Self::initialize_server_with_fake_map_builder`] for the
    /// uploading server.
    fn initialize_uploading_server_with_fake_map_builder(
        &mut self,
    ) -> (Arc<Mutex<MapBuilderScript>>, Arc<FakePoseGraph>) {
        let map_builder = FakeMapBuilder::new(None);
        let script = map_builder.script();
        let pose_graph = map_builder.fake_pose_graph();
        self.uploading_server = Some(MapBuilderServer::new(
            &self.uploading_map_builder_server_options,
            Box::new(map_builder),
        ));
        (script, pose_graph)
    }

    /// C++ `InitializeStub()`.
    fn initialize_stub(&mut self) {
        let address = self.server().local_address().unwrap().to_string();
        self.stub = Some(MapBuilderStub::new(&address, CLIENT_ID));
    }

    /// C++ `InitializeStubForUploadingServer()`.
    fn initialize_stub_for_uploading_server(&mut self) {
        let address = self.uploading_server().local_address().unwrap().to_string();
        self.stub_for_uploading_server = Some(MapBuilderStub::new(&address, CLIENT_ID));
    }

    /// C++ `SetOptionsToTSDF2D()`.
    fn set_options_to_tsdf_2d(&mut self) {
        let options_2d = self
            .trajectory_builder_options
            .trajectory_builder_2d_options
            .as_mut()
            .unwrap();
        let submaps_options = options_2d.submaps_options.as_mut().unwrap();
        submaps_options
            .range_data_inserter_options
            .as_mut()
            .unwrap()
            .range_data_inserter_type =
            mapping_proto::range_data_inserter_options::RangeDataInserterType::TsdfInserter2d
                as i32;
        submaps_options.grid_options_2d.as_mut().unwrap().grid_type =
            mapping_proto::grid_options2_d::GridType::Tsdf as i32;
        options_2d
            .ceres_scan_matcher_options
            .as_mut()
            .unwrap()
            .occupied_space_weight = 10.0;
        for options in [
            &mut self.map_builder_server_options,
            &mut self.uploading_map_builder_server_options,
        ] {
            options
                .map_builder_options
                .as_mut()
                .unwrap()
                .pose_graph_options
                .as_mut()
                .unwrap()
                .constraint_builder_options
                .as_mut()
                .unwrap()
                .ceres_scan_matcher_options
                .as_mut()
                .unwrap()
                .occupied_space_weight = 50.0;
        }
    }

    /// C++ `WaitForLocalSlamResultUploads(size)`.
    fn wait_for_local_slam_result_uploads(&mut self, size: usize) {
        let deadline = Instant::now() + Duration::from_secs(120);
        while self.stub().pose_graph().get_trajectory_node_poses().size() < size {
            assert!(Instant::now() < deadline, "timed out waiting for uploads");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn expect_travelled_distance(&self) {
        let poses = self.local_slam_results.poses();
        let distance = (poses.last().unwrap().translation() - poses.first().unwrap().translation())
            .norm();
        assert!(
            (TRAVEL_DISTANCE - distance).abs() <= 0.1 * TRAVEL_DISTANCE,
            "travelled {distance}"
        );
    }
}

/// The serialized state of the C++ load tests (`kPoseGraphProtoString` etc.)
/// built in code: a pose graph with one trajectory holding one node and one
/// submap.
fn load_state_reader() -> Box<InMemoryProtoStreamReader> {
    use mapping_proto::serialized_data::Data;
    let data = |data: Data| mapping_proto::SerializedData { data: Some(data) };
    proto_reader_from_protos(
        mapping_proto::SerializationHeader { format_version: 2 },
        vec![
            data(Data::PoseGraph(mapping_proto::PoseGraph {
                trajectory: vec![mapping_proto::Trajectory {
                    trajectory_id: 0,
                    node: vec![Default::default()],
                    submap: vec![Default::default()],
                }],
                ..Default::default()
            })),
            data(Data::AllTrajectoryBuilderOptions(
                mapping_proto::AllTrajectoryBuilderOptions {
                    options_with_sensor_ids: vec![Default::default()],
                },
            )),
            data(Data::Submap(Default::default())),
            data(Data::Node(Default::default())),
            data(Data::ImuData(Default::default())),
            data(Data::OdometryData(Default::default())),
            data(Data::LandmarkData(Default::default())),
        ],
    )
}

// ---------------------------------------------------------------------------
// Tests runnable with the fake map builder.
// ---------------------------------------------------------------------------

#[test]
fn fake_start_and_stop_server() {
    let mut test = ClientServerTest::set_up();
    test.initialize_server_with_fake_map_builder(None);
    test.server().start();
    assert!(test.server().local_address().is_some());
    test.server().shutdown();
    // Returns immediately after `shutdown`.
    test.server().wait_for_shutdown();
}

/// C++ `ClientServerTestByGridType.AddTrajectoryBuilderWithMock`.
fn add_trajectory_builder_with_mock(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    let (script, _) = test.initialize_server_with_fake_map_builder(None);
    script.lock().unwrap().next_trajectory_id = 3;
    test.server().start();
    test.initialize_stub();
    let expected_sensor_ids = sensor_ids(&[imu_sensor_id()]);
    let options = test.trajectory_builder_options.clone();
    let trajectory_id = test
        .stub()
        .add_trajectory_builder(&expected_sensor_ids, &options, None);
    assert_eq!(trajectory_id, 3);
    test.stub().finish_trajectory(trajectory_id);
    assert_eq!(
        script.lock().unwrap().calls,
        vec![
            MapBuilderCall::AddTrajectoryBuilder {
                expected_sensor_ids,
                trajectory_options: Box::new(options),
                // The server always subscribes for its local SLAM callback.
                has_local_slam_result_callback: true,
            },
            MapBuilderCall::FinishTrajectory(3),
        ]
    );
    test.server().shutdown();
}

#[test]
fn add_trajectory_builder_with_mock_probability_grid() {
    add_trajectory_builder_with_mock(GridType::ProbabilityGrid);
}

#[test]
fn add_trajectory_builder_with_mock_tsdf() {
    add_trajectory_builder_with_mock(GridType::Tsdf);
}

/// C++ `ClientServerTestByGridType.AddSensorDataWithMock`.
fn add_sensor_data_with_mock(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    let mock_trajectory_builder = MockTrajectoryBuilder::new();
    let (script, _) =
        test.initialize_server_with_fake_map_builder(Some(mock_trajectory_builder.clone()));
    script.lock().unwrap().next_trajectory_id = 3;
    test.server().start();
    test.initialize_stub();
    let expected_sensor_ids = sensor_ids(&[imu_sensor_id()]);
    let options = test.trajectory_builder_options.clone();
    let trajectory_id = test
        .stub()
        .add_trajectory_builder(&expected_sensor_ids, &options, None);
    assert_eq!(trajectory_id, 3);
    let imu_data = ImuData {
        time: from_universal(42),
        linear_acceleration: Vector3::new(0., 0., 9.8),
        angular_velocity: Vector3::zeros(),
    };
    test.stub()
        .get_trajectory_builder(trajectory_id)
        .unwrap()
        .add_imu_data(&imu_sensor_id().id, &imu_data);
    test.stub().finish_trajectory(trajectory_id);
    // The SLAM thread hands the data to the trajectory builder.
    wait_until("AddSensorData", Duration::from_secs(10), || {
        !mock_trajectory_builder.recorded_calls().is_empty()
    });
    assert_eq!(
        mock_trajectory_builder.recorded_calls(),
        vec![(CallKind::Imu, imu_sensor_id().id, from_universal(42))]
    );
    assert!(
        script
            .lock()
            .unwrap()
            .calls
            .contains(&MapBuilderCall::FinishTrajectory(3))
    );
    test.server().shutdown();
}

#[test]
fn add_sensor_data_with_mock_probability_grid() {
    add_sensor_data_with_mock(GridType::ProbabilityGrid);
}

#[test]
fn add_sensor_data_with_mock_tsdf() {
    add_sensor_data_with_mock(GridType::Tsdf);
}

/// `AddTrajectoryBuilder` flow (trajectory states) with the fake.
#[test]
fn fake_add_trajectory_builder() {
    let mut test = ClientServerTest::set_up();
    test.initialize_server_with_fake_map_builder(None);
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let trajectory_id =
        test.stub()
            .add_trajectory_builder(&sensor_ids(&[imu_sensor_id()]), &options, None);
    assert!(!test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    assert_eq!(test.stub().num_trajectory_builders(), 1);
    test.stub().finish_trajectory(trajectory_id);
    assert_eq!(test.stub().num_trajectory_builders(), 0);
    assert!(test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    assert!(!test.stub().pose_graph().is_trajectory_frozen(trajectory_id));
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states(),
        [(trajectory_id, TrajectoryState::Finished)].into_iter().collect()
    );
    test.server().shutdown();
}

/// A fake local SLAM result with an inserted node and a submap.
fn fake_insertion_result(trajectory_id: i32, node_index: i32) -> InsertionResult {
    let mut active_submaps = ActiveSubmaps2D::new(&create_submaps_options_2d(
        &resolve_lua_parameters(
            r#"include "trajectory_builder_2d.lua"
            return TRAJECTORY_BUILDER_2D.submaps"#,
        ),
    ));
    let submaps = active_submaps.insert_range_data(&RangeData {
        origin: Vector3::zeros(),
        returns: PointCloud::from_points(vec![RangefinderPoint {
            position: Vector3::new(1., 0., 0.),
        }]),
        misses: PointCloud::default(),
    });
    InsertionResult {
        node_id: NodeId::new(trajectory_id, node_index),
        constant_data: Arc::new(TrajectoryNodeData {
            time: from_universal(1000),
            gravity_alignment: Quaternion::identity(),
            filtered_gravity_aligned_point_cloud: PointCloud::default(),
            high_resolution_point_cloud: PointCloud::default(),
            low_resolution_point_cloud: PointCloud::default(),
            rotational_scan_matcher_histogram: DVector::zeros(0),
            local_pose: Rigid3d::identity(),
        }),
        insertion_submaps: submaps
            .into_iter()
            .map(|submap| submap as Arc<std::sync::RwLock<dyn Submap>>)
            .collect(),
    }
}

/// Local SLAM results produced by the server's map builder are streamed to
/// the stub's callback (`ReceiveLocalSlamResults`), and the stream ends when
/// the trajectory is finished.
#[test]
fn fake_local_slam_results_are_streamed_to_the_client() {
    let mut test = ClientServerTest::set_up();
    let (script, _) = test.initialize_server_with_fake_map_builder(None);
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    let server_callback = script
        .lock()
        .unwrap()
        .local_slam_result_callbacks
        .get(&trajectory_id)
        .cloned()
        .expect("the server registers a local SLAM callback");
    for i in 0..5 {
        server_callback(
            trajectory_id,
            from_universal(i),
            Rigid3d::new(
                Vector3::new(0.3 * i as f64, 0., 0.),
                Quaternion::new(1., 0., 0., 0.),
            ),
            RangeData::default(),
            (i % 2 == 0).then(|| fake_insertion_result(trajectory_id, i as i32)),
        );
    }
    test.local_slam_results.wait_for(5);
    assert_eq!(test.local_slam_results.number_of_insertion_results(), 3);
    let poses = test.local_slam_results.poses();
    assert_eq!(poses[4].translation().x, 1.2);
    // Finishing ends the stream (and joins the reader thread).
    test.stub().finish_trajectory(trajectory_id);
    test.server().shutdown();
}

/// Global SLAM optimizations of the server's pose graph reach subscribers of
/// `ReceiveGlobalSlamOptimizations`.
#[test]
fn fake_global_slam_optimizations_are_streamed() {
    let mut test = ClientServerTest::set_up();
    let (_, pose_graph) = test.initialize_server_with_fake_map_builder(None);
    test.server().start();
    let address = test.server().local_address().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut stream = runtime.block_on(async {
        let mut client =
            proto::map_builder_service_client::MapBuilderServiceClient::connect(format!(
                "http://{address}"
            ))
            .await
            .unwrap();
        client
            .receive_global_slam_optimizations(())
            .await
            .unwrap()
            .into_inner()
    });
    pose_graph.run_global_slam_optimization_callback(
        &[(0, SubmapId::new(0, 4))].into_iter().collect(),
        &[(0, NodeId::new(0, 11))].into_iter().collect(),
    );
    let response = runtime.block_on(stream.message()).unwrap().unwrap();
    assert_eq!(
        response.last_optimized_node_ids[&0],
        mapping_proto::NodeId {
            trajectory_id: 0,
            node_index: 11,
        }
    );
    assert_eq!(response.last_optimized_submap_ids[&0].submap_index, 4);
    // Shutdown ends the stream.
    test.server().shutdown();
    assert!(runtime.block_on(stream.message()).ok().flatten().is_none());
}

/// The uploading server creates the trajectory in the uplink (without range
/// sensors, plus the local SLAM result sensor, with stripped options) and
/// uploads local SLAM results, which the uplink reassembles into
/// `LocalSlamResult2D` packets for its trajectory builder.
#[test]
fn fake_uploading_server_uploads_trajectory_and_local_slam_results() {
    let mut test = ClientServerTest::set_up();
    let uplink_trajectory_builder = MockTrajectoryBuilder::new();
    let (uplink_script, uplink_pose_graph) =
        test.initialize_server_with_fake_map_builder(Some(uplink_trajectory_builder.clone()));
    uplink_script.lock().unwrap().next_trajectory_id = 5;
    test.server().start();
    let (uploading_script, uploading_pose_graph) =
        test.initialize_uploading_server_with_fake_map_builder();
    test.uploading_server().start();
    test.initialize_stub_for_uploading_server();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub_for_uploading_server().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id(), imu_sensor_id()]),
        &options,
        Some(callback),
    );
    assert_eq!(trajectory_id, 0);
    // The uplink got the trajectory.
    let mut stripped_options = options.clone();
    stripped_options.trajectory_builder_2d_options = None;
    stripped_options.trajectory_builder_3d_options = None;
    stripped_options.pure_localization_trimmer = None;
    stripped_options.initial_trajectory_pose = None;
    assert_eq!(
        uplink_script.lock().unwrap().calls[0],
        MapBuilderCall::AddTrajectoryBuilder {
            expected_sensor_ids: sensor_ids(&[
                imu_sensor_id(),
                SensorId {
                    r#type: SensorType::LocalSlamResult,
                    id: "local_slam_result_0".to_string(),
                },
            ]),
            trajectory_options: Box::new(stripped_options),
            has_local_slam_result_callback: true,
        }
    );
    // A local SLAM result with an insertion is uploaded (batch size 1).
    let server_callback = uploading_script
        .lock()
        .unwrap()
        .local_slam_result_callbacks[&trajectory_id]
        .clone();
    server_callback(
        trajectory_id,
        from_universal(1000),
        Rigid3d::identity(),
        RangeData::default(),
        Some(fake_insertion_result(trajectory_id, 0)),
    );
    test.local_slam_results.wait_for(1);
    wait_until("upload", Duration::from_secs(30), || {
        !uplink_trajectory_builder.recorded_calls().is_empty()
    });
    assert_eq!(
        uplink_trajectory_builder.recorded_calls(),
        vec![(
            CallKind::LocalSlamResult,
            "local_slam_result_0".to_string(),
            from_universal(1000)
        )]
    );
    assert!(
        uplink_script
            .lock()
            .unwrap()
            .calls
            .contains(&MapBuilderCall::GetTrajectoryBuilder(5))
    );
    // Finishing finishes the uplink trajectory too.
    test.stub_for_uploading_server()
        .finish_trajectory(trajectory_id);
    assert!(uploading_pose_graph.is_trajectory_finished(0));
    assert!(uplink_pose_graph.is_trajectory_finished(5));
    test.uploading_server().shutdown();
    test.server().shutdown();
}

/// `LoadState` streams the whole serialized state; the server registers the
/// client id for the loaded trajectories.
#[test]
fn fake_load_state() {
    let mut test = ClientServerTest::set_up();
    let (script, _) = test.initialize_server_with_fake_map_builder(None);
    script.lock().unwrap().trajectory_remapping = [(0, 0)].into_iter().collect();
    test.server().start();
    test.initialize_stub();
    let mut reader = load_state_reader();
    let trajectory_remapping = test.stub().load_state(reader.as_mut(), true);
    assert_eq!(trajectory_remapping, [(0, 0)].into_iter().collect());
    // Header + pose graph + options + 5 data messages.
    assert_eq!(
        script.lock().unwrap().calls,
        vec![MapBuilderCall::LoadState(true, 8)]
    );
    // The client id was registered for the loaded trajectory, so it can be
    // deleted by this client.
    test.stub().pose_graph().delete_trajectory(0);
    let trajectory_remapping = test.stub().load_state_from_file("some.pbstream", false);
    assert_eq!(trajectory_remapping, [(0, 0)].into_iter().collect());
    test.server().shutdown();
}

/// `SerializeState` reads the `WriteState` stream; the result can be
/// deserialized again.
#[test]
fn fake_serialize_state() {
    let mut test = ClientServerTest::set_up();
    let (script, _) = test.initialize_server_with_fake_map_builder(None);
    script.lock().unwrap().serialized_state = vec![mapping_proto::SerializedData {
        data: Some(mapping_proto::serialized_data::Data::PoseGraph(
            mapping_proto::PoseGraph {
                trajectory: vec![mapping_proto::Trajectory::default()],
                ..Default::default()
            },
        )),
    }];
    test.server().start();
    test.initialize_stub();
    let chunks = Arc::new(Mutex::new(VecDeque::<Box<dyn ProtoMessage>>::new()));
    let sink = Arc::clone(&chunks);
    let mut writer = ForwardingProtoStreamWriter::new(Box::new(move |proto| {
        if let Some(proto) = proto {
            sink.lock().unwrap().push_back(proto.clone_message());
        }
        true
    }));
    test.stub().serialize_state(false, &mut writer);
    assert!(writer.close());
    let chunks = std::mem::take(&mut *chunks.lock().unwrap());
    // Only the header and the pose graph: the fake has no trajectory options.
    assert_eq!(chunks.len(), 2);
    assert!(test.stub().serialize_state_to_file(false, "state.pbstream"));
    assert_eq!(
        script.lock().unwrap().calls,
        vec![
            MapBuilderCall::SerializeState(false),
            MapBuilderCall::SerializeStateToFile(false, "state.pbstream".to_string()),
        ]
    );
    test.server().shutdown();
}

/// The invalid-id part of `LocalSlam2DHandlesInvalidRequests` with the fake.
#[test]
fn fake_handles_invalid_requests() {
    let mut test = ClientServerTest::set_up();
    let (script, _) = test.initialize_server_with_fake_map_builder(None);
    script.lock().unwrap().submap_to_proto_error = "Requested submap does not exist".to_string();
    test.server().start();
    test.initialize_stub();
    const INVALID_TRAJECTORY_ID: i32 = 7;
    test.stub().pose_graph().delete_trajectory(INVALID_TRAJECTORY_ID);
    assert!(
        !test
            .stub()
            .pose_graph()
            .is_trajectory_finished(INVALID_TRAJECTORY_ID)
    );
    assert!(
        !test
            .stub()
            .pose_graph()
            .is_trajectory_frozen(INVALID_TRAJECTORY_ID)
    );
    assert!(
        test.stub()
            .get_trajectory_builder(INVALID_TRAJECTORY_ID)
            .is_none()
    );
    test.stub().finish_trajectory(INVALID_TRAJECTORY_ID);
    let mut submap_query_response = Default::default();
    assert_ne!(
        test.stub().submap_to_proto(
            &SubmapId::new(INVALID_TRAJECTORY_ID, 0),
            &mut submap_query_response
        ),
        ""
    );
    // Neither the unknown delete nor the unknown finish reached the map
    // builder (NOT_FOUND).
    assert_eq!(
        script.lock().unwrap().calls,
        vec![MapBuilderCall::SubmapToProto(SubmapId::new(
            INVALID_TRAJECTORY_ID,
            0
        ))]
    );
    test.server().shutdown();
}

/// The stub panics like the C++ `LOG(FATAL)` if the server is unreachable.
#[test]
fn stub_fails_without_server() {
    let address = reserve_local_address();
    let start = Instant::now();
    assert!(MapBuilderStub::try_new(&address, CLIENT_ID).is_err());
    assert!(start.elapsed() >= Duration::from_secs(9));
}

// ---------------------------------------------------------------------------
// C++ tests with the real MapBuilder.
// ---------------------------------------------------------------------------

fn add_measurements(test: &mut ClientServerTest, trajectory_id: i32, range: std::ops::Range<usize>) {
    let measurements = generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP);
    let stub = test.stub_for_uploading_server.as_mut().or(test.stub.as_mut()).unwrap();
    let trajectory_stub = stub.get_trajectory_builder(trajectory_id).unwrap();
    for measurement in &measurements[range] {
        trajectory_stub.add_timed_point_cloud_data(&range_sensor_id().id, measurement);
    }
}

fn number_of_measurements() -> usize {
    generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP).len()
}

#[test]
#[ignore = "needs MapBuilder; enabled after merge"]
fn start_and_stop_server() {
    let mut test = ClientServerTest::set_up();
    test.initialize_real_server();
    test.server().start();
    test.server().shutdown();
}

fn add_trajectory_builder(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let trajectory_id =
        test.stub()
            .add_trajectory_builder(&sensor_ids(&[imu_sensor_id()]), &options, None);
    assert!(!test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    test.stub().finish_trajectory(trajectory_id);
    assert!(test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    assert!(!test.stub().pose_graph().is_trajectory_frozen(trajectory_id));
    test.server().shutdown();
}

fn add_sensor_data(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.trajectory_builder_options
        .trajectory_builder_2d_options
        .as_mut()
        .unwrap()
        .use_imu_data = true;
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let trajectory_id =
        test.stub()
            .add_trajectory_builder(&sensor_ids(&[imu_sensor_id()]), &options, None);
    let imu_data = ImuData {
        time: from_universal(42),
        linear_acceleration: Vector3::new(0., 0., 9.8),
        angular_velocity: Vector3::zeros(),
    };
    test.stub()
        .get_trajectory_builder(trajectory_id)
        .unwrap()
        .add_imu_data(&imu_sensor_id().id, &imu_data);
    test.stub().finish_trajectory(trajectory_id);
    test.server().shutdown();
}

fn local_slam_2d(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    assert!(test.stub().pose_graph().get_trajectory_states().is_empty());
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    add_measurements(&mut test, trajectory_id, 0..number_of_measurements());
    test.local_slam_results.wait_for(number_of_measurements());
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Active
    );
    test.stub().finish_trajectory(trajectory_id);
    test.stub().pose_graph().run_final_optimization();
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Finished
    );
    assert_eq!(test.local_slam_results.poses().len(), number_of_measurements());
    test.expect_travelled_distance();
    test.server().shutdown();
}

fn local_slam_and_delete_2d(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    add_measurements(&mut test, trajectory_id, 0..number_of_measurements());
    test.local_slam_results.wait_for(number_of_measurements());
    test.stub().pose_graph().run_final_optimization();
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Active
    );
    assert!(test.stub().pose_graph().get_all_submap_poses().size() > 0);
    assert!(test.stub().pose_graph().get_trajectory_node_poses().size() > 0);
    test.stub().finish_trajectory(trajectory_id);
    test.stub().pose_graph().delete_trajectory(trajectory_id);
    test.stub().pose_graph().run_final_optimization();
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Deleted
    );
    assert_eq!(test.stub().pose_graph().get_all_submap_poses().size(), 0);
    assert_eq!(test.stub().pose_graph().get_trajectory_node_poses().size(), 0);
    test.server().shutdown();
}

#[test]
#[ignore = "needs MapBuilder; enabled after merge"]
fn global_slam_3d() {
    let mut test = ClientServerTest::set_up();
    {
        let map_builder_options = test
            .map_builder_server_options
            .map_builder_options
            .as_mut()
            .unwrap();
        map_builder_options.use_trajectory_builder_2d = false;
        map_builder_options.use_trajectory_builder_3d = true;
        map_builder_options
            .pose_graph_options
            .as_mut()
            .unwrap()
            .optimize_every_n_nodes = 3;
        let motion_filter_options = test
            .trajectory_builder_options
            .trajectory_builder_3d_options
            .as_mut()
            .unwrap()
            .motion_filter_options
            .as_mut()
            .unwrap();
        motion_filter_options.max_distance_meters = 0.;
        motion_filter_options.max_angle_radians = 0.;
    }
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id(), imu_sensor_id()]),
        &options,
        Some(callback),
    );
    let measurements = generate_fake_range_measurements(TRAVEL_DISTANCE, DURATION, TIME_STEP);
    {
        let trajectory_stub = test.stub().get_trajectory_builder(trajectory_id).unwrap();
        for measurement in &measurements {
            let imu_data = ImuData {
                time: measurement.time - from_seconds(TIME_STEP / 2.),
                linear_acceleration: Vector3::new(0., 0., 9.8),
                angular_velocity: Vector3::zeros(),
            };
            trajectory_stub.add_imu_data(&imu_sensor_id().id, &imu_data);
            trajectory_stub.add_timed_point_cloud_data(&range_sensor_id().id, measurement);
        }
    }
    // First range data will not call back while initializing
    // PoseExtrapolator, so expect one less.
    test.local_slam_results.wait_for(measurements.len() - 1);
    test.stub().finish_trajectory(trajectory_id);
    test.expect_travelled_distance();
    test.server().shutdown();
}

fn start_and_stop_uploading_server_and_server(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_real_uploading_server();
    test.uploading_server().start();
    test.uploading_server().shutdown();
    test.server().shutdown();
}

fn add_trajectory_builder_with_uploading_server(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_real_uploading_server();
    test.uploading_server().start();
    test.initialize_stub();
    test.initialize_stub_for_uploading_server();
    let options = test.trajectory_builder_options.clone();
    let trajectory_id = test.stub_for_uploading_server().add_trajectory_builder(
        &sensor_ids(&[imu_sensor_id()]),
        &options,
        None,
    );
    assert!(
        !test
            .stub_for_uploading_server()
            .pose_graph()
            .is_trajectory_finished(trajectory_id)
    );
    assert!(!test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    test.stub_for_uploading_server()
        .finish_trajectory(trajectory_id);
    assert!(
        test.stub_for_uploading_server()
            .pose_graph()
            .is_trajectory_finished(trajectory_id)
    );
    assert!(test.stub().pose_graph().is_trajectory_finished(trajectory_id));
    assert!(
        !test
            .stub_for_uploading_server()
            .pose_graph()
            .is_trajectory_frozen(trajectory_id)
    );
    assert!(!test.stub().pose_graph().is_trajectory_frozen(trajectory_id));
    test.uploading_server().shutdown();
    test.server().shutdown();
}

fn local_slam_2d_with_uploading_server(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    test.initialize_real_uploading_server();
    test.uploading_server().start();
    test.initialize_stub_for_uploading_server();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub_for_uploading_server().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    add_measurements(&mut test, trajectory_id, 0..number_of_measurements());
    test.local_slam_results.wait_for(number_of_measurements());
    let uploads = test.local_slam_results.number_of_insertion_results();
    test.wait_for_local_slam_result_uploads(uploads);

    let chunks = Arc::new(Mutex::new(VecDeque::<Box<dyn ProtoMessage>>::new()));
    let sink = Arc::clone(&chunks);
    let mut writer = ForwardingProtoStreamWriter::new(Box::new(move |proto| {
        if let Some(proto) = proto {
            sink.lock().unwrap().push_back(proto.clone_message());
        }
        true
    }));
    test.stub().serialize_state(false, &mut writer);
    assert!(writer.close());
    // Ensure it can be read.
    let mut reader = InMemoryProtoStreamReader::new(std::mem::take(&mut *chunks.lock().unwrap()));
    let deserializer = ProtoStreamDeserializer::new(&mut reader);
    assert_eq!(deserializer.pose_graph().trajectory.len(), 1);
    drop(deserializer);
    test.stub_for_uploading_server()
        .finish_trajectory(trajectory_id);
    assert_eq!(test.local_slam_results.poses().len(), number_of_measurements());
    test.expect_travelled_distance();
    test.uploading_server().shutdown();
    test.server().shutdown();
}

fn local_slam_2d_uplink_server_restarting(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    test.initialize_real_uploading_server();
    test.uploading_server().start();
    test.initialize_stub_for_uploading_server();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub_for_uploading_server().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    let half = number_of_measurements() / 2;
    // Insert half of the measurements.
    add_measurements(&mut test, trajectory_id, 0..half);
    test.local_slam_results.wait_for(half);
    let uploads = test.local_slam_results.number_of_insertion_results();
    test.wait_for_local_slam_result_uploads(uploads);

    // Simulate a cloud server restart.
    log::info!("Simulating server restart.");
    const UPLINK_TRAJECTORY_ID: i32 = 0;
    test.stub().finish_trajectory(UPLINK_TRAJECTORY_ID);
    test.server().shutdown();
    test.server().wait_for_shutdown();
    test.stub = None;
    test.server = None;
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();

    // Insert the second half of the measurements.
    add_measurements(&mut test, trajectory_id, half..number_of_measurements());
    test.local_slam_results.wait_for(half);
    test.wait_for_local_slam_result_uploads(2);
    test.stub_for_uploading_server()
        .finish_trajectory(trajectory_id);
    test.uploading_server().shutdown();
    test.uploading_server().wait_for_shutdown();
    test.server().shutdown();
    test.server().wait_for_shutdown();
}

#[test]
#[ignore = "needs MapBuilder; enabled after merge"]
fn delayed_connection_to_uplink_server() {
    let mut test = ClientServerTest::set_up();
    test.initialize_real_uploading_server();
    test.uploading_server().start();
    test.initialize_stub_for_uploading_server();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub_for_uploading_server().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    // Insert the first measurement.
    add_measurements(&mut test, trajectory_id, 0..1);
    test.local_slam_results.wait_for(1);
    log::info!("Delayed start of uplink server.");
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    // Insert all other measurements.
    add_measurements(&mut test, trajectory_id, 1..number_of_measurements());
    test.local_slam_results.wait_for(number_of_measurements());
    test.wait_for_local_slam_result_uploads(2);
    test.stub_for_uploading_server()
        .finish_trajectory(trajectory_id);
    test.uploading_server().shutdown();
    test.uploading_server().wait_for_shutdown();
    test.server().shutdown();
    test.server().wait_for_shutdown();
}

fn load_state_and_delete(grid_type: GridType, load_frozen_state: bool) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let mut reader = load_state_reader();
    let trajectory_remapping = test.stub().load_state(reader.as_mut(), load_frozen_state);
    let expected_trajectory_id = 0;
    assert_eq!(trajectory_remapping.len(), 1);
    assert_eq!(trajectory_remapping[&0], expected_trajectory_id);
    test.stub().pose_graph().run_final_optimization();
    assert_eq!(
        test.stub()
            .pose_graph()
            .is_trajectory_frozen(expected_trajectory_id),
        load_frozen_state
    );
    assert!(
        !test
            .stub()
            .pose_graph()
            .is_trajectory_finished(expected_trajectory_id)
    );
    if !load_frozen_state {
        assert_eq!(
            test.stub().pose_graph().get_trajectory_states()[&expected_trajectory_id],
            TrajectoryState::Active
        );
        test.stub().finish_trajectory(expected_trajectory_id);
        assert_eq!(
            test.stub().pose_graph().get_trajectory_states()[&expected_trajectory_id],
            TrajectoryState::Finished
        );
    }
    for trajectory_id in trajectory_remapping.values() {
        test.stub().pose_graph().delete_trajectory(*trajectory_id);
        test.stub().pose_graph().run_final_optimization();
        assert_eq!(
            test.stub().pose_graph().get_trajectory_states()[trajectory_id],
            TrajectoryState::Deleted
        );
    }
    test.server().shutdown();
}

fn local_slam_2d_handles_invalid_requests(grid_type: GridType) {
    let mut test = ClientServerTest::set_up_with_grid_type(grid_type);
    test.initialize_real_server();
    test.server().start();
    test.initialize_stub();
    let options = test.trajectory_builder_options.clone();
    let callback = test.local_slam_results.callback();
    let trajectory_id = test.stub().add_trajectory_builder(
        &sensor_ids(&[range_sensor_id()]),
        &options,
        Some(callback),
    );
    add_measurements(&mut test, trajectory_id, 0..number_of_measurements());
    test.local_slam_results.wait_for(number_of_measurements());
    test.stub().pose_graph().run_final_optimization();
    const INVALID_TRAJECTORY_ID: i32 = 7;
    test.stub().pose_graph().delete_trajectory(INVALID_TRAJECTORY_ID);
    assert!(
        !test
            .stub()
            .pose_graph()
            .is_trajectory_finished(INVALID_TRAJECTORY_ID)
    );
    assert!(
        !test
            .stub()
            .pose_graph()
            .is_trajectory_frozen(INVALID_TRAJECTORY_ID)
    );
    assert!(
        test.stub()
            .get_trajectory_builder(INVALID_TRAJECTORY_ID)
            .is_none()
    );
    test.stub().finish_trajectory(INVALID_TRAJECTORY_ID);
    let invalid_submap_id_0 = SubmapId::new(INVALID_TRAJECTORY_ID, 0);
    let invalid_submap_id_1 = SubmapId::new(trajectory_id, 424242);
    let mut submap_query_response = Default::default();
    // Expect that it returns non-empty error string.
    assert_ne!(
        test.stub()
            .submap_to_proto(&invalid_submap_id_0, &mut submap_query_response),
        ""
    );
    assert_ne!(
        test.stub()
            .submap_to_proto(&invalid_submap_id_1, &mut submap_query_response),
        ""
    );
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Active
    );
    let submap_poses = test.stub().pose_graph().get_all_submap_poses();
    assert!(submap_poses.size() > 0);
    assert!(test.stub().pose_graph().get_trajectory_node_poses().size() > 0);
    test.stub().finish_trajectory(trajectory_id);
    test.stub().pose_graph().delete_trajectory(trajectory_id);
    test.stub().pose_graph().run_final_optimization();
    let deleted_submap_id = submap_poses.iter().next().unwrap().id;
    assert_ne!(
        test.stub()
            .submap_to_proto(&deleted_submap_id, &mut submap_query_response),
        ""
    );
    assert_eq!(
        test.stub().pose_graph().get_trajectory_states()[&trajectory_id],
        TrajectoryState::Deleted
    );
    // Make sure optimization runs with a deleted trajectory.
    test.stub().pose_graph().run_final_optimization();
    test.server().shutdown();
}

/// Instantiates a `ClientServerTestByGridType` test for both grid types
/// (C++ `INSTANTIATE_TEST_CASE_P`).
macro_rules! by_grid_type {
    ($($name:ident => $body:expr;)*) => {
        mod client_server_test_by_grid_type {
            use super::*;
            $(
                mod $name {
                    use super::*;

                    #[test]
                    #[ignore = "needs MapBuilder; enabled after merge"]
                    fn probability_grid() {
                        ($body)(GridType::ProbabilityGrid);
                    }

                    #[test]
                    #[ignore = "needs MapBuilder; enabled after merge"]
                    fn tsdf() {
                        ($body)(GridType::Tsdf);
                    }
                }
            )*
        }
    };
}

by_grid_type! {
    add_trajectory_builder => add_trajectory_builder;
    add_sensor_data => add_sensor_data;
    local_slam_2d => local_slam_2d;
    local_slam_and_delete_2d => local_slam_and_delete_2d;
    start_and_stop_uploading_server_and_server => start_and_stop_uploading_server_and_server;
    add_trajectory_builder_with_uploading_server => add_trajectory_builder_with_uploading_server;
    local_slam_2d_with_uploading_server => local_slam_2d_with_uploading_server;
    local_slam_2d_uplink_server_restarting => local_slam_2d_uplink_server_restarting;
    load_state_and_delete => |grid_type| load_state_and_delete(grid_type, true);
    load_unfrozen_state_and_delete => |grid_type| load_state_and_delete(grid_type, false);
    local_slam_2d_handles_invalid_requests => local_slam_2d_handles_invalid_requests;
}
