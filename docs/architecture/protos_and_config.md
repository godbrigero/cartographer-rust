# Protos and Lua configuration: architecture report (Phase 0)

Source of truth: `cartographer/` (read-only C++ tree). Paths below are relative to
`cartographer/cartographer/` unless they start with `configuration_files/`.

Contents:
1. Proto inventory, plus prost/protox hazards
2. Lua option table: every key → C++ consumer → proto field
3. The Lua language subset that configs actually use, and the interpreter requirements
4. How `LuaParameterDictionary` behaves and how tests use it

---

## 0. Key facts (TL;DR)

| Topic | Fact |
|---|---|
| Proto syntax | **All 43 files are `proto3`.** No proto2, no `optional` keyword, no extensions, no groups, no `Any`/`Timestamp`/`Duration`. |
| Well-known types | Only `google/protobuf/empty.proto` (in `cloud/proto/map_builder_service.proto`). |
| Packages | 9: `cartographer.{common,sensor,transform,ground_truth,cloud}.proto`, `cartographer.mapping.proto`, `cartographer.mapping.{scan_matching,constraints,optimization}.proto`. |
| Include root | Imports are written `cartographer/<pkg>/proto/...`, so the include path is the **outer** repo dir (`.../cartographer/`). This matches CMake's `-I ${PROJECT_SOURCE_DIR}`. |
| Dir and package don't match | `mapping/proto/pose_graph/*.proto` → packages `cartographer.mapping.constraints.proto` and `cartographer.mapping.optimization.proto`. `mapping/proto/scan_matching/*.proto` → `cartographer.mapping.scan_matching.proto`. |
| Rust-keyword field | `type` in `mapping.proto.SensorId` (field 1) and `cloud.proto.SensorId` (field 2) → prost emits `r#type`. |
| Oneofs | 5: `Grid2D.grid`, `SerializedData.data`, `cloud.SensorData.sensor_data`, `cloud.LoadStateRequest.state_chunk`, `cloud.WriteStateResponse.state_chunk`. |
| Maps | 4, all in the cloud service: `map<int32,NodeId>`, `map<int32,SubmapId>`, `map<int32,int32>`, `map<int32,TrajectoryState(enum)>`. The map type params contain inline `/* */` comments. |
| Recursion | None. No message is self- or mutually recursive, so prost does not need to `Box` anything. |
| Lua runtime | **Lua 5.2.4** (Bazel pins `lua-5.2.4`; `package.xml` depends on `lua5.2-dev`; CMake's FindLuaGoogle accepts 5.3 or 5.2). All numbers are doubles; there is no integer subtype. |
| Config→proto | Hand-written `Create*Options(LuaParameterDictionary*)` functions, one per options message. **Lua key names often differ from proto field names** (e.g. `submaps` → `submaps_options`), so a generic serde mapping won't work (see §2.0). |
| Strictness | Every non-numeric key in every (sub)dictionary must be read **exactly once**, or the destructor CHECK-fails. A typo or unused key in a config is fatal. |
| Base configs invalid alone | `map_builder.lua` sets both `use_trajectory_builder_2d/3d = false`, but `CreateMapBuilderOptions` does `CHECK_NE(2d, 3d)`. A consumer must flip one of them. |

---

## 1. Proto inventory

### 1.1 Per-file table

Category: **O** = options proto (filled from Lua), **D** = data/serialization (pbstream, in-memory), **S** = gRPC service, **O+D** = options message that is also serialized into pbstream.
"Imports" lists only cross-file imports. `xpkg` marks an import from a different package.

| # | File | Package | Imports | Messages / enums (brief) | Cat |
|---|---|---|---|---|---|
| 1 | `common/proto/ceres_solver_options.proto` | cartographer.common.proto | – | `CeresSolverOptions{bool use_nonmonotonic_steps=1; int32 max_num_iterations=2; int32 num_threads=3}` | O |
| 2 | `sensor/proto/adaptive_voxel_filter_options.proto` | cartographer.sensor.proto | – | `AdaptiveVoxelFilterOptions{float max_length=1; float min_num_points=2 (sic, float); float max_range=3}` | O |
| 3 | `sensor/proto/sensor.proto` | cartographer.sensor.proto | transform.proto (xpkg) | `RangefinderPoint{Vector3f position}`, `TimedRangefinderPoint{Vector3f position; float time}`, `CompressedPointCloud{int32 num_points=1; repeated int32 point_data=3}`, `TimedPointCloudData{int64 timestamp; Vector3f origin; repeated Vector4f point_data_legacy=3; repeated TimedRangefinderPoint point_data=4; repeated float intensities=5}`, `RangeData{origin; repeated Vector3f returns_legacy/misses_legacy (2,3); repeated RangefinderPoint returns/misses (4,5)}`, `ImuData{int64 timestamp; Vector3d linear_acceleration, angular_velocity}`, `OdometryData{int64 timestamp; Rigid3d pose}`, `FixedFramePoseData{same}`, `LandmarkData{int64 timestamp; repeated LandmarkObservation{bytes id; Rigid3d landmark_to_tracking_transform; double translation_weight, rotation_weight}}`. Option `java_outer_classname="Sensor"`. | D |
| 4 | `transform/proto/transform.proto` | cartographer.transform.proto | – | `Vector2d/2f{x,y}`, `Vector3d/3f{x,y,z}`, `Vector4f{x,y,z,t}`, `Quaterniond/f{x=1,y=2,z=3,w=4}` (**w is last**), `Rigid2d{Vector2d translation; double rotation}`, `Rigid2f`, `Rigid3d{Vector3d translation; Quaterniond rotation}`, `Rigid3f` | D |
| 5 | `transform/proto/timestamped_transform.proto` | cartographer.transform.proto | transform.proto | `TimestampedTransform{int64 time=1; Rigid3d transform=2}` | D |
| 6 | `ground_truth/proto/relations.proto` | cartographer.ground_truth.proto | transform.proto (xpkg) | `Relation{int64 timestamp1, timestamp2; Rigid3d expected; double covered_distance}`, `GroundTruth{repeated Relation relation}` | D |
| 7 | `mapping/proto/cell_limits_2d.proto` | cartographer.mapping.proto | – | `CellLimits{int32 num_x_cells, num_y_cells}` | D |
| 8 | `mapping/proto/map_limits.proto` | cartographer.mapping.proto | cell_limits_2d, transform (xpkg) | `MapLimits{double resolution; Vector2d max; CellLimits cell_limits}` | D |
| 9 | `mapping/proto/connected_components.proto` | cartographer.mapping.proto | – | `ConnectedComponents{repeated ConnectedComponent{repeated int32 trajectory_id} connected_component}`. Option `java_outer_classname="ConnectedComponentsOuterClass"` (file and message share a name). | D |
| 10 | `mapping/proto/probability_grid.proto` | cartographer.mapping.proto | – | `ProbabilityGrid{}`, an **empty** message used only as a oneof discriminator in Grid2D | D |
| 11 | `mapping/proto/tsdf_2d.proto` | cartographer.mapping.proto | – | `TSDF2D{float truncation_distance; float max_weight; repeated int32 weight_cells}` | D |
| 12 | `mapping/proto/grid_2d.proto` | cartographer.mapping.proto | map_limits, probability_grid, tsdf_2d | `Grid2D{MapLimits limits=1; repeated int32 cells=2 (int16 values); CellBox{int32 max_x,max_y,min_x,min_y} known_cells_box=3; oneof grid{ProbabilityGrid probability_grid_2d=4; TSDF2D tsdf_2d=5}; float min_correspondence_cost=6; float max_correspondence_cost=7}` | D |
| 13 | `mapping/proto/hybrid_grid.proto` | cartographer.mapping.proto | – | `HybridGrid{float resolution=1; repeated sint32 x_indices=3,y_indices=4,z_indices=5; repeated int32 values=6 (uint16 values)}`. Field 2 is unused. | D |
| 14 | `mapping/proto/submap.proto` | cartographer.mapping.proto | grid_2d, hybrid_grid, transform (xpkg) | `Submap2D{Rigid3d local_pose; int32 num_range_data; bool finished; Grid2D grid}`, `Submap3D{local_pose; num_range_data; finished; HybridGrid high_resolution_hybrid_grid=4, low_resolution_hybrid_grid=5; repeated float rotational_scan_matcher_histogram=6}` | D |
| 15 | `mapping/proto/trajectory.proto` | cartographer.mapping.proto | transform (xpkg) | `Trajectory{int32 trajectory_id=3; repeated Node{int32 node_index=7; int64 timestamp=1; Rigid3d pose=5} node=1; repeated Submap{int32 submap_index=2; Rigid3d pose=1} submap=2}`. Option `java_outer_classname="TrajectoryOuterClass"`. | D |
| 16 | `mapping/proto/pose_graph.proto` | cartographer.mapping.proto | trajectory, transform (xpkg) | `SubmapId{int32 trajectory_id, submap_index}`, `NodeId{int32 trajectory_id, node_index}`, `PoseGraph{repeated Constraint constraint=2; repeated Trajectory trajectory=4; repeated LandmarkPose landmark_poses=5}`, nested `Constraint{SubmapId=1; NodeId=2; Rigid3d relative_pose=3; double translation_weight=6, rotation_weight=7; enum Tag{INTRA_SUBMAP=0, INTER_SUBMAP=1} tag=5}`, `LandmarkPose{string landmark_id; Rigid3d global_pose}` | D |
| 17 | `mapping/proto/trajectory_node_data.proto` | cartographer.mapping.proto | sensor (xpkg), transform (xpkg) | `TrajectoryNodeData{int64 timestamp; Quaterniond gravity_alignment; CompressedPointCloud filtered_gravity_aligned_point_cloud=3, high_resolution_point_cloud=4, low_resolution_point_cloud=5; repeated float rotational_scan_matcher_histogram=6; Rigid3d local_pose=7}` | D |
| 18 | `mapping/proto/submap_visualization.proto` | cartographer.mapping.proto | transform (xpkg) | `SubmapList{repeated TrajectorySubmapList{repeated SubmapEntry{int32 submap_version=1; Rigid3d pose=3} submap} trajectory=2}`, `SubmapQuery{Request{int32 submap_index, trajectory_id} request; Response{int32 submap_version=2; repeated SubmapTexture{bytes cells (gzip); int32 width,height; double resolution; Rigid3d slice_pose} textures=10; string error_message=8} response}` | D |
| 19 | `mapping/proto/serialization.proto` | cartographer.mapping.proto | pose_graph, submap, trajectory_node_data, sensor (xpkg), trajectory_builder_options, transform (xpkg) | `Submap{SubmapId; Submap2D submap_2d; Submap3D submap_3d}` (two optional fields, **not** a oneof), `Node{NodeId node_id=1; TrajectoryNodeData node_data=5}`, `ImuData/OdometryData/FixedFramePoseData/LandmarkData{int32 trajectory_id; sensor.proto.X}` (same names as the sensor-package messages but a different package), `TrajectoryData{int32 trajectory_id; double gravity_constant; Quaterniond imu_calibration; Rigid3d fixed_frame_origin_in_map}`, `LocalSlamResultData{int64 timestamp; TrajectoryNodeData; repeated Submap submaps}`, `SerializationHeader{uint32 format_version}`, `SerializedData{oneof data{PoseGraph=1; AllTrajectoryBuilderOptions=2; Submap=3; Node=4; TrajectoryData=5; ImuData=6; OdometryData=7; FixedFramePoseData=8; LandmarkData=9}}` | D (pbstream) |
| 20 | `mapping/proto/grid_2d_options.proto` | cartographer.mapping.proto | – | `GridOptions2D{enum GridType{INVALID_GRID=0, PROBABILITY_GRID=1, TSDF=2} grid_type=1; float resolution=2}` | O |
| 21 | `mapping/proto/motion_filter_options.proto` | cartographer.mapping.proto | – | `MotionFilterOptions{double max_time_seconds, max_distance_meters, max_angle_radians}` | O |
| 22 | `mapping/proto/normal_estimation_options_2d.proto` | cartographer.mapping.proto | – | `NormalEstimationOptions2D{int32 num_normal_samples; float sample_radius}` | O |
| 23 | `mapping/proto/pose_extrapolator_options.proto` | cartographer.mapping.proto | ceres_solver_options (xpkg) | `ConstantVelocityPoseExtrapolatorOptions{double imu_gravity_time_constant=1, pose_queue_duration=2}`, `ImuBasedPoseExtrapolatorOptions{double pose_queue_duration=1, gravity_constant=2, pose_translation_weight=3, pose_rotation_weight=4, imu_acceleration_weight=5, imu_rotation_weight=6; CeresSolverOptions solver_options=7; double odometry_translation_weight=8, odometry_rotation_weight=9}`, `PoseExtrapolatorOptions{bool use_imu_based; constant_velocity; imu_based}` | O |
| 24 | `mapping/proto/probability_grid_range_data_inserter_options_2d.proto` | cartographer.mapping.proto | – | `ProbabilityGridRangeDataInserterOptions2D{double hit_probability, miss_probability; bool insert_free_space}` | O |
| 25 | `mapping/proto/tsdf_range_data_inserter_options_2d.proto` | cartographer.mapping.proto | normal_estimation_options_2d | `TSDFRangeDataInserterOptions2D{double truncation_distance=1, maximum_weight=2; bool update_free_space=3; NormalEstimationOptions2D=4; bool project_sdf_distance_to_scan_normal=5; int32 update_weight_range_exponent=6; double update_weight_angle_scan_normal_to_ray_kernel_bandwidth=7, update_weight_distance_cell_to_hit_kernel_bandwidth=8}` | O |
| 26 | `mapping/proto/range_data_inserter_options.proto` | cartographer.mapping.proto | #24, #25 | `RangeDataInserterOptions{enum RangeDataInserterType{INVALID_INSERTER=0, PROBABILITY_GRID_INSERTER_2D=1, TSDF_INSERTER_2D=2} range_data_inserter_type=1; probability_grid_range_data_inserter_options_2d=2; tsdf_range_data_inserter_options_2d=3}` | O |
| 27 | `mapping/proto/range_data_inserter_options_3d.proto` | cartographer.mapping.proto | – | `RangeDataInserterOptions3D{double hit_probability, miss_probability; int32 num_free_space_voxels; float intensity_threshold}` | O |
| 28 | `mapping/proto/submaps_options_2d.proto` | cartographer.mapping.proto | #20, #26 | `SubmapsOptions2D{int32 num_range_data=1; GridOptions2D grid_options_2d=2; RangeDataInserterOptions range_data_inserter_options=3}` | O |
| 29 | `mapping/proto/submaps_options_3d.proto` | cartographer.mapping.proto | #27 | `SubmapsOptions3D{double high_resolution=1, high_resolution_max_range=4, low_resolution=5; int32 num_range_data=2; RangeDataInserterOptions3D=3}` | O |
| 30 | `mapping/proto/scan_matching/ceres_scan_matcher_options_2d.proto` | cartographer.mapping.scan_matching.proto | #1 (xpkg) | `CeresScanMatcherOptions2D{double occupied_space_weight=1, translation_weight=2, rotation_weight=3; CeresSolverOptions ceres_solver_options=9}` | O |
| 31 | `mapping/proto/scan_matching/ceres_scan_matcher_options_3d.proto` | cartographer.mapping.scan_matching.proto | #1 (xpkg) | `IntensityCostFunctionOptions{double weight, huber_scale; float intensity_threshold}`, `CeresScanMatcherOptions3D{repeated double occupied_space_weight=1; double translation_weight=2, rotation_weight=3; bool only_optimize_yaw=5; CeresSolverOptions=6; repeated IntensityCostFunctionOptions intensity_cost_function_options=7}` | O |
| 32 | `mapping/proto/scan_matching/fast_correlative_scan_matcher_options_2d.proto` | cartographer.mapping.scan_matching.proto | – | `FastCorrelativeScanMatcherOptions2D{double linear_search_window=3, angular_search_window=4; int32 branch_and_bound_depth=2}` | O |
| 33 | `mapping/proto/scan_matching/fast_correlative_scan_matcher_options_3d.proto` | cartographer.mapping.scan_matching.proto | – | `FastCorrelativeScanMatcherOptions3D{int32 branch_and_bound_depth=2, full_resolution_depth=8; double min_rotational_score=4, min_low_resolution_score=9, linear_xy_search_window=5, linear_z_search_window=6, angular_search_window=7}` | O |
| 34 | `mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.proto` | cartographer.mapping.scan_matching.proto | – | `RealTimeCorrelativeScanMatcherOptions{double linear_search_window, angular_search_window, translation_delta_cost_weight, rotation_delta_cost_weight}` | O |
| 35 | `mapping/proto/pose_graph/constraint_builder_options.proto` | cartographer.mapping.constraints.proto | #30–#33 (xpkg) | `ConstraintBuilderOptions{double sampling_ratio=1, max_constraint_distance=2, min_score=4, global_localization_min_score=5, loop_closure_translation_weight=13, loop_closure_rotation_weight=14; bool log_matches=8; FCSM2D fast_correlative_scan_matcher_options=9; CSM2D ceres_scan_matcher_options=11; FCSM3D fast_correlative_scan_matcher_options_3d=10; CSM3D ceres_scan_matcher_options_3d=12}` | O |
| 36 | `mapping/proto/pose_graph/optimization_problem_options.proto` | cartographer.mapping.optimization.proto | #1 (xpkg) | `OptimizationProblemOptions{reserved 20 to 22; double huber_scale=1, acceleration_weight=8, rotation_weight=9, local_slam_pose_translation_weight=14, local_slam_pose_rotation_weight=15, odometry_translation_weight=16, odometry_rotation_weight=17, fixed_frame_pose_translation_weight=11, fixed_frame_pose_rotation_weight=12; bool fixed_frame_pose_use_tolerant_loss=23; double fixed_frame_pose_tolerant_loss_param_a=24, _b=25; bool fix_z_in_3d=13, use_online_imu_extrinsics_in_3d=18, log_solver_summary=5; CeresSolverOptions=7}` | O |
| 37 | `mapping/proto/pose_graph_options.proto` | cartographer.mapping.proto | #35, #36 (xpkg) | `PoseGraphOptions{int32 optimize_every_n_nodes=1; ConstraintBuilderOptions constraint_builder_options=3; double matcher_translation_weight=7, matcher_rotation_weight=8; OptimizationProblemOptions optimization_problem_options=4; int32 max_num_final_iterations=6; double global_sampling_ratio=5; bool log_residual_histograms=9; double global_constraint_search_after_n_seconds=10; OverlappingSubmapsTrimmerOptions2D{int32 fresh_submaps_count; double min_covered_area; int32 min_added_submaps_count} overlapping_submaps_trimmer_2d=11}` | O |
| 38 | `mapping/proto/local_trajectory_builder_options_2d.proto` | cartographer.mapping.proto | #21, #23, #2 (xpkg), #30 (xpkg), #34 (xpkg), #28 | `LocalTrajectoryBuilderOptions2D` (NEXT ID 22): `float min_range=14, max_range=15, min_z=1, max_z=2, missing_data_ray_length=16; int32 num_accumulated_range_data=19; float voxel_filter_size=3; AVF adaptive_voxel_filter_options=6, loop_closure_adaptive_voxel_filter_options=20; bool use_online_correlative_scan_matching=5; RTCSM=7; CSM2D ceres_scan_matcher_options=8; MotionFilterOptions=13; double imu_gravity_time_constant=17; PoseExtrapolatorOptions=21; SubmapsOptions2D=11; bool use_imu_data=12` | O+D |
| 39 | `mapping/proto/local_trajectory_builder_options_3d.proto` | cartographer.mapping.proto | #21, #23, #31, #34, #29, #2, sensor, timestamped_transform (xpkg) | `LocalTrajectoryBuilderOptions3D`: `float min_range=1, max_range=2; int32 num_accumulated_range_data=3; float voxel_filter_size=4; AVF high_resolution_..._options=5, low_resolution_..._options=12; bool use_online_correlative_scan_matching=13; RTCSM=14; CSM3D=6; MotionFilterOptions=7; double imu_gravity_time_constant=15; int32 rotational_histogram_size=17; PoseExtrapolatorOptions=18; repeated TimestampedTransform initial_poses=19; repeated sensor.ImuData initial_imu_data=20; SubmapsOptions3D=8; bool use_intensities=21`. Fields 19 and 20 are **not** filled from Lua. | O+D |
| 40 | `mapping/proto/trajectory_builder_options.proto` | cartographer.mapping.proto | transform (xpkg), #21, #38, #39 | `InitialTrajectoryPose{Rigid3d relative_pose; int32 to_trajectory_id; int64 timestamp}`, `TrajectoryBuilderOptions{2d_options=1; 3d_options=2; InitialTrajectoryPose=4; reserved 5; bool pure_localization=3 [deprecated=true]; PureLocalizationTrimmerOptions{int32 max_submaps_to_keep} pure_localization_trimmer=6; bool collate_fixed_frame=7, collate_landmarks=8; MotionFilterOptions pose_graph_odometry_motion_filter=9}`, `SensorId{enum SensorType{RANGE,IMU,ODOMETRY,FIXED_FRAME_POSE,LANDMARK,LOCAL_SLAM_RESULT} type=1; string id=2}`, `TrajectoryBuilderOptionsWithSensorIds{repeated SensorId=1; TrajectoryBuilderOptions=2}`, `AllTrajectoryBuilderOptions{repeated ...WithSensorIds options_with_sensor_ids=1}` | O+D |
| 41 | `mapping/proto/map_builder_options.proto` | cartographer.mapping.proto | #37 | `MapBuilderOptions{bool use_trajectory_builder_2d=1, use_trajectory_builder_3d=2; int32 num_background_threads=3; PoseGraphOptions=4; bool collate_by_trajectory=5}` | O |
| 42 | `cloud/proto/map_builder_server_options.proto` | cartographer.cloud.proto | #41 (xpkg) | `MapBuilderServerOptions{string server_address=1; int32 num_grpc_threads=2, num_event_threads=3; MapBuilderOptions map_builder_options=4; string uplink_server_address=5; int32 upload_batch_size=6; bool enable_ssl_encryption=7, enable_google_auth=9}`. Field 8 is unused. | O |
| 43 | `cloud/proto/map_builder_service.proto` | cartographer.cloud.proto | pose_graph, serialization, submap_visualization, trajectory_builder_options, sensor, transform (all xpkg), **google/protobuf/empty.proto** | Top-level enums `SensorType{RANGE..LOCAL_SLAM_RESULT}` and `TrajectoryState{ACTIVE,FINISHED,FROZEN,DELETED}`; 41 top-level messages (request/response plus helpers) (see §1.2); `service MapBuilderService` with 26 RPCs | S |

### 1.2 gRPC service (`MapBuilderService`, 26 RPCs)

| RPC | Request | Response | Streaming |
|---|---|---|---|
| AddTrajectory | AddTrajectoryRequest{repeated SensorId expected_sensor_ids=3; TrajectoryBuilderOptions=2; string client_id=4} | AddTrajectoryResponse{int32 trajectory_id} | unary |
| AddSensorDataBatch | AddSensorDataBatchRequest{repeated SensorData} | Empty | unary |
| AddOdometryData / AddImuData / AddRangefinderData / AddFixedFramePoseData / AddLandmarkData | Add*Request{SensorMetadata{int32 trajectory_id; string sensor_id; string client_id}; payload} | Empty | **client stream** |
| ReceiveLocalSlamResults | {int32 trajectory_id} | ReceiveLocalSlamResultsResponse{trajectory_id; int64 timestamp; Rigid3d local_pose; sensor.RangeData; LocalSlamInsertionResult{NodeId}} | **server stream** |
| ReceiveGlobalSlamOptimizations | Empty | {map<int32,NodeId> last_optimized_node_ids; map<int32,SubmapId> last_optimized_submap_ids} | **server stream** |
| FinishTrajectory / DeleteTrajectory | {int32 trajectory_id; string client_id} | Empty | unary |
| GetSubmap | {SubmapId} | {SubmapQuery.Response submap_query_response; string error_msg} | unary |
| GetTrajectoryNodePoses | Empty | {repeated TrajectoryNodePose{NodeId; Rigid3d global_pose; ConstantPoseData{int64 timestamp; Rigid3d local_pose}}} | unary |
| GetTrajectoryStates | Empty | {map<int32, TrajectoryState> trajectories_state} | unary |
| GetLandmarkPoses | Empty | {repeated PoseGraph.LandmarkPose} | unary |
| GetAllSubmapPoses | Empty | {repeated SubmapPose{SubmapId; int32 submap_version; Rigid3d global_pose}} | unary |
| GetLocalToGlobalTransform | {int32 trajectory_id} | {Rigid3d local_to_global} | unary |
| GetConstraints | Empty | {repeated PoseGraph.Constraint} | unary |
| SetLandmarkPose | {PoseGraph.LandmarkPose} | Empty | unary |
| IsTrajectoryFinished / IsTrajectoryFrozen | {int32 trajectory_id} | {bool is_finished / is_frozen} | unary |
| RunFinalOptimization | Empty | Empty | unary |
| LoadState | LoadStateRequest{oneof state_chunk{SerializedData=1; SerializationHeader=2; string client_id=3}; bool load_frozen_state=4} | {TrajectoryRemapping{map<int32,int32>}} | **client stream** |
| LoadStateFromFile | {string file_path; client_id; bool load_frozen_state} | {TrajectoryRemapping} | unary |
| WriteState | Empty | WriteStateResponse{oneof state_chunk{SerializationHeader header=1; SerializedData=2}} | **server stream** |
| WriteStateToFile | {string filename} | {bool success} | unary |

`SensorData{SensorMetadata=1; oneof sensor_data{sensor.OdometryData=2; ImuData=3; TimedPointCloudData=4; FixedFramePoseData=5; LandmarkData=6; mapping.LocalSlamResultData=7}}`.

### 1.3 Things that may trip up prost / protox / tonic

| Issue | Where | Impact and mitigation |
|---|---|---|
| Rust keyword field `type` | `mapping.proto.SensorId.type`, `cloud.proto.SensorId.type` | prost emits `r#type`. Works, but is ugly; wrap it in accessors if desired. No other field or oneof name collides with a Rust keyword (checked against the full 2021/2024 keyword list). |
| Package nesting vs `proto` leaf module | All packages end in `.proto`, and `cartographer.mapping.proto` is a sibling of `cartographer.mapping.scan_matching.proto`. | prost writes one `.rs` per package (9 files). Cross-package refs use `super::super::...` paths, so the generated modules **must** be nested exactly by package: `cartographer::mapping::{proto, scan_matching::proto, constraints::proto, optimization::proto}`. Use `prost_build::Config::include_file("mod.rs")` (or tonic-build's equivalent) to auto-generate that tree. |
| Partially qualified type refs | e.g. `sensor.proto.AdaptiveVoxelFilterOptions` from `cartographer.mapping.proto`; `mapping.scan_matching.proto.X` from `cartographer.mapping.constraints.proto`; `common.proto.CeresSolverOptions` from `cartographer.mapping.scan_matching.proto`; `transform.proto.Rigid3d` | These resolve by protobuf scoping rules (innermost outward). protoc accepts them and protox implements the same rules. If you write a custom descriptor pipeline, implement C++-style scope walking. |
| Inline block comments in map types | `map<int32 /* trajectory_id */, ...>` (4×) | protoc and protox handle them. A hand-rolled `.proto` parser would need to too. |
| Well-known type `google.protobuf.Empty` | service only | prost maps it to `()` by default. protox bundles the WKT sources. tonic codegen handles `()` request/response. |
| `oneof`s (5) | see §0 | prost generates `enum` types in a snake_case module named after the message, e.g. `serialized_data::Data`, `grid2_d::Grid` (verify the exact casing), `sensor_data::SensorData`, `load_state_request::StateChunk`, `write_state_response::StateChunk`. `LoadStateRequest.state_chunk` mixes message and `string` variants. |
| `map<,>` with enum value | `map<int32, TrajectoryState>` | prost stores the value as `i32` (`HashMap<i32, i32>` with `enumeration(TrajectoryState)` attr), so conversion is needed. Default map type is `HashMap`; use `btree_map(["."])` if you need deterministic iteration. |
| Enums stored as `i32` | every enum field | Needs `GridType::try_from(i32)`. For Lua string → enum (C++ `*_Parse`), use prost's generated `from_str_name("PROBABILITY_GRID")`. The names must match **exactly** as in the proto. |
| Acronym and digit re-casing | `TSDF2D`, `TSDFRangeDataInserterOptions2D`, `Grid2D`, `Submap2D`, enum values `TSDF`, `PROBABILITY_GRID_INSERTER_2D` | heck converts these to e.g. `Tsdf2d`, `TsdfRangeDataInserterOptions2D`, `GridType::Tsdf`, `RangeDataInserterType::ProbabilityGridInserter2d`. Generated names won't match C++ spelling, so check the generated code before writing hand code against it. prost does **not** strip enum prefixes here: values are not prefixed with the enum name. |
| Same message name in different packages | `ImuData`, `OdometryData`, `FixedFramePoseData`, `LandmarkData` exist in both `cartographer.sensor.proto` and `cartographer.mapping.proto` (wrappers with `trajectory_id`). `SensorId` and `SensorType` exist in both `mapping.proto` and `cloud.proto`. | Fine in Rust (different modules), but easy to import the wrong one. |
| Same name, nested vs top-level | `mapping.proto.Submap` (serialization) vs `Trajectory.Submap`; `mapping.proto.Node` vs `Trajectory.Node` | No clash: nested types live in the `trajectory::` module. |
| Empty message | `ProbabilityGrid {}` | Fine. It is used as a oneof variant marker. |
| `[deprecated = true]` | `TrajectoryBuilderOptions.pure_localization` | prost puts `#[deprecated]` on the field, so any struct literal or field access warns. Add `#![allow(deprecated)]` on the generated module or use `..Default::default()`. |
| `reserved` | `TrajectoryBuilderOptions` 5; `OptimizationProblemOptions` 20–22 | No impact. |
| `java_outer_classname` | sensor.proto, trajectory.proto, connected_components.proto | Ignored by prost. |
| `sint32` | `HybridGrid.{x,y,z}_indices` | Zigzag encoding; prost supports it. Don't map these to `int32`. |
| Packed repeated scalars | many (`cells`, `point_data`, `values`, histograms, `occupied_space_weight`, `intensities`) | proto3 packs by default. prost encodes packed and decodes both packed and unpacked forms. This is required for **.pbstream byte-compat** with C++ output. |
| `float` vs `double` | Many options fields are `float` (see §2: min/max range, voxel_filter_size, grid resolution, sample_radius, AVF fields, intensity_threshold) | Lua doubles get narrowed to `f32` on `set_*`. Replicate this to stay bit-identical (e.g. `resolution = 0.05` becomes `0.05f32`). |
| Timestamps | All `int64 timestamp`/`time` fields | These are cartographer "universal time" ticks (100 ns since 0001-01-01; see `common/time.h`), **not** `google.protobuf.Timestamp`. |
| `bytes` | `LandmarkObservation.id`, `SubmapTexture.cells` | `Vec<u8>` by default. Landmark ids are UTF-8 strings in C++. Use `Config::bytes(["."])` if you want `bytes::Bytes`. |
| Presence | proto3 scalars have no presence, but message fields do (`Option<T>` in prost) | C++ relies on `has_*()` only for message fields (`has_overlapping_submaps_trimmer_2d`, `has_pure_localization_trimmer`, `has_pose_graph_odometry_motion_filter`, `has_initial_trajectory_pose`, `Submap2D.has_grid`, `Submap.has_submap_2d/3d`, Grid2D oneof). These map cleanly to `Option`. |
| No recursion | – | No `Box` is needed. |
| Options protos are also wire data | `AllTrajectoryBuilderOptions` is `SerializedData` variant 2, so `TrajectoryBuilderOptions` → `LocalTrajectoryBuilderOptions{2D,3D}` and all their sub-options are written to `.pbstream`. | Options protos must be wire-compatible too, not just used as in-memory config. |
| CMake quirk (FYI) | `list(REMOVE_ITEM ALL_PROTOS ALL_GRPC_SERVICES)` lacks `${}`, so it is a no-op. Without `BUILD_GRPC` all `cloud/*` files are excluded, including both cloud protos. | In Rust, gate the cloud protos and tonic behind a cargo feature (`grpc`). |

---

## 2. Configuration: every Lua option

### 2.0 Conventions and consumers

**Getter semantics** (from `common/lua_parameter_dictionary.cc`, Lua 5.2):

| Getter | Lua check | Returns | Notes |
|---|---|---|---|
| `GetDouble` | `lua_isnumber` (true for numbers **and numeric strings**) | `lua_tonumber` | |
| `GetInt` | `lua_isnumber` | `int(lua_tointeger)` | In 5.2 on 64-bit, a non-integral value is converted by C cast, i.e. **truncation toward zero**, then narrowed to `int`. Lua 5.3 would return 0 instead. No shipped config passes a non-integral value to `GetInt`. |
| `GetNonNegativeInt` | `GetInt` + `CHECK_GE(v, 0)` | int | Counts as one reference. |
| `GetBool` | `lua_isboolean` (strict; `1`/`"true"` are rejected) | bool | |
| `GetString` | `lua_isstring` (true for **numbers** too) | `lua_tostring` | A number is formatted as `%.14g` (`10` → `"10"`). The test `GetDictionary` relies on this. |
| `GetDictionary` | `lua_istable` | child dict | Child inherits the parent's ref-count mode. Its unused-key check fires when the returned `unique_ptr` dies, which in `Create*` calls is at the end of the full expression. |
| `HasKey` | `rawget(key) != nil` | bool | **Not** counted as a reference. |
| `GetArrayValuesAs{Doubles,Strings,Dictionaries}` | iterates `t[1], t[2], …` until the first `nil` | vector | Not reference-counted. Integer keys are never checked. |
| `GetKeys` | `lua_next`; skips keys where `lua_isnumber(key)` | string keys | A string key like `"1"` would also be skipped (edge case). |

Lookups use `lua_rawget`, so metatables are ignored.

**Lua key ≠ proto field.** Renames done by the `Create*` functions:

| Lua key | Proto field |
|---|---|
| `adaptive_voxel_filter`, `loop_closure_adaptive_voxel_filter`, `high_resolution_adaptive_voxel_filter`, `low_resolution_adaptive_voxel_filter` | `*_options` suffix |
| `real_time_correlative_scan_matcher`, `ceres_scan_matcher`, `motion_filter`, `pose_extrapolator`, `submaps`, `range_data_inserter` | `*_options` suffix |
| `probability_grid_range_data_inserter` / `tsdf_range_data_inserter` | `probability_grid_range_data_inserter_options_2d` / `tsdf_range_data_inserter_options_2d` |
| `constraint_builder` / `optimization_problem` | `constraint_builder_options` / `optimization_problem_options` |
| `fast_correlative_scan_matcher` / `fast_correlative_scan_matcher_3d` / `ceres_scan_matcher_3d` | `fast_correlative_scan_matcher_options` / `fast_correlative_scan_matcher_options_3d` / `ceres_scan_matcher_options_3d` |
| `map_builder` / `pose_graph` | `map_builder_options` / `pose_graph_options` |
| `trajectory_builder_2d` / `trajectory_builder_3d` | `trajectory_builder_2d_options` / `trajectory_builder_3d_options` |
| `occupied_space_weight_0`, `_1`, … | `repeated double occupied_space_weight` (index order) |
| `intensity_cost_function_options_0`, `_1`, … | `repeated IntensityCostFunctionOptions intensity_cost_function_options` |

Unchanged names: `ceres_solver_options`, `solver_options`, `constant_velocity`, `imu_based`, `grid_options_2d`, `normal_estimation_options`, `overlapping_submaps_trimmer_2d`, `pure_localization_trimmer`, `pose_graph_odometry_motion_filter`.

**Consumer functions** (abbreviation → function @ file):

| Abbr | Function | File | Proto produced |
|---|---|---|---|
| MBSO | `cloud::CreateMapBuilderServerOptions` (+ `LoadMapBuilderServerOptions(dir, basename)`) | `cloud/map_builder_server_options.cc` | cloud.MapBuilderServerOptions |
| MBO | `mapping::CreateMapBuilderOptions` | `mapping/map_builder_interface.cc` | MapBuilderOptions |
| PGO | `mapping::CreatePoseGraphOptions` (+ static `PopulateOverlappingSubmapsTrimmerOptions2D`) | `mapping/pose_graph.cc` | PoseGraphOptions |
| CBO | `mapping::constraints::CreateConstraintBuilderOptions` | `mapping/internal/constraints/constraint_builder.cc` | ConstraintBuilderOptions |
| OPO | `mapping::optimization::CreateOptimizationProblemOptions` | `mapping/internal/optimization/optimization_problem_options.cc` | OptimizationProblemOptions |
| FC2 | `mapping::scan_matching::CreateFastCorrelativeScanMatcherOptions2D` | `mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d.cc` | FastCorrelativeScanMatcherOptions2D |
| FC3 | `mapping::scan_matching::CreateFastCorrelativeScanMatcherOptions3D` | `mapping/internal/3d/scan_matching/fast_correlative_scan_matcher_3d.cc` | FastCorrelativeScanMatcherOptions3D |
| CS2 | `mapping::scan_matching::CreateCeresScanMatcherOptions2D` | `mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.cc` | CeresScanMatcherOptions2D |
| CS3 | `mapping::scan_matching::CreateCeresScanMatcherOptions3D` | `mapping/internal/3d/scan_matching/ceres_scan_matcher_3d.cc` | CeresScanMatcherOptions3D (+ inline IntensityCostFunctionOptions) |
| CSO | `common::CreateCeresSolverOptionsProto` | `common/internal/ceres_solver_options.cc` | CeresSolverOptions |
| TBO | `mapping::CreateTrajectoryBuilderOptions` (+ static `PopulatePureLocalizationTrimmerOptions`, `PopulatePoseGraphOdometryMotionFilterOptions`) | `mapping/trajectory_builder_interface.cc` | TrajectoryBuilderOptions |
| LT2 | `mapping::CreateLocalTrajectoryBuilderOptions2D` | `mapping/internal/2d/local_trajectory_builder_options_2d.cc` | LocalTrajectoryBuilderOptions2D |
| LT3 | `mapping::CreateLocalTrajectoryBuilderOptions3D` | `mapping/internal/3d/local_trajectory_builder_options_3d.cc` | LocalTrajectoryBuilderOptions3D |
| AVF | `sensor::CreateAdaptiveVoxelFilterOptions` | `sensor/internal/voxel_filter.cc` | AdaptiveVoxelFilterOptions |
| RTC | `mapping::scan_matching::CreateRealTimeCorrelativeScanMatcherOptions` | `mapping/internal/scan_matching/real_time_correlative_scan_matcher.cc` | RealTimeCorrelativeScanMatcherOptions |
| MF | `mapping::CreateMotionFilterOptions` | `mapping/internal/motion_filter.cc` | MotionFilterOptions |
| PEO | `mapping::CreatePoseExtrapolatorOptions` (+ anon-ns `CreateConstantVelocityPoseExtrapolatorOptions`, `CreateImuBasedPoseExtrapolatorOptions`) | `mapping/pose_extrapolator_interface.cc` | PoseExtrapolatorOptions |
| SO2 | `mapping::CreateSubmapsOptions2D` | `mapping/2d/submap_2d.cc` | SubmapsOptions2D |
| GO2 | `mapping::CreateGridOptions2D` | `mapping/2d/grid_2d.cc` | GridOptions2D |
| RDI | `mapping::CreateRangeDataInserterOptions` | `mapping/range_data_inserter_interface.cc` | RangeDataInserterOptions |
| PGI | `mapping::CreateProbabilityGridRangeDataInserterOptions2D` | `mapping/2d/probability_grid_range_data_inserter_2d.cc` | ProbabilityGridRangeDataInserterOptions2D |
| TSI | `mapping::CreateTSDFRangeDataInserterOptions2D` | `mapping/internal/2d/tsdf_range_data_inserter_2d.cc` | TSDFRangeDataInserterOptions2D |
| NE2 | `mapping::CreateNormalEstimationOptions2D` | `mapping/internal/2d/normal_estimation_2d.cc` | NormalEstimationOptions2D |
| SO3 | `mapping::CreateSubmapsOptions3D` | `mapping/3d/submap_3d.cc` | SubmapsOptions3D |
| RD3 | `mapping::CreateRangeDataInserterOptions3D` | `mapping/3d/range_data_inserter_3d.cc` | RangeDataInserterOptions3D |

That is 25 functions. All of them are listed above; there are no other `Create*Options(LuaParameterDictionary*)` in the tree. `transform::FromDictionary` and the `io::*PointsProcessor::FromDictionary` functions also read Lua (§2.8).

**Getter abbreviations** used in the tables: `D`=GetDouble, `I`=GetInt, `NNI`=GetNonNegativeInt, `B`=GetBool, `S`=GetString, `Dict`=GetDictionary. Proto types: `f`=float, `d`=double, `i32`, `b`=bool, `str`, `msg`.

**Validation CHECKs inside `Create*`:** CSO: `max_num_iterations > 0`, `num_threads > 0`. PGI and RD3: `hit_probability > 0.5`, `miss_probability < 0.5`. SO2: valid (grid_type, inserter_type) pair, i.e. (PROBABILITY_GRID, PROBABILITY_GRID_INSERTER_2D) or (TSDF, TSDF_INSERTER_2D), and `num_range_data > 0`. SO3: `num_range_data > 0`. NE2: `num_normal_samples > 0`, `sample_radius > 0`. RTC: both cost weights `>= 0`. PGO: `max_num_final_iterations > 0`. MBO: `use_trajectory_builder_2d != use_trajectory_builder_3d`. GO2 and RDI: the enum string must parse.

### 2.1 `pose_graph.lua` → `POSE_GRAPH` (PGO → PoseGraphOptions)

| Full key path | Lua default | Getter | Consumer | Proto field (type) | Notes |
|---|---|---|---|---|---|
| POSE_GRAPH.optimize_every_n_nodes | `90` | I | PGO | optimize_every_n_nodes (i32) | 0 disables online optimization |
| POSE_GRAPH.constraint_builder | table | Dict | PGO→CBO | constraint_builder_options (msg) | |
| POSE_GRAPH.constraint_builder.sampling_ratio | `0.3` | D | CBO | sampling_ratio (d) | |
| POSE_GRAPH.constraint_builder.max_constraint_distance | `15.` | D | CBO | max_constraint_distance (d) | |
| POSE_GRAPH.constraint_builder.min_score | `0.55` | D | CBO | min_score (d) | |
| POSE_GRAPH.constraint_builder.global_localization_min_score | `0.6` | D | CBO | global_localization_min_score (d) | |
| POSE_GRAPH.constraint_builder.loop_closure_translation_weight | `1.1e4` | D | CBO | loop_closure_translation_weight (d) | |
| POSE_GRAPH.constraint_builder.loop_closure_rotation_weight | `1e5` | D | CBO | loop_closure_rotation_weight (d) | |
| POSE_GRAPH.constraint_builder.log_matches | `true` | B | CBO | log_matches (b) | |
| POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher | table | Dict | CBO→FC2 | fast_correlative_scan_matcher_options (msg) | |
| …fast_correlative_scan_matcher.linear_search_window | `7.` | D | FC2 | linear_search_window (d) | |
| …fast_correlative_scan_matcher.angular_search_window | `math.rad(30.)` = 0.5235987755982988 | D | FC2 | angular_search_window (d) | degrees→radians conversion happens **in Lua** |
| …fast_correlative_scan_matcher.branch_and_bound_depth | `7` | I | FC2 | branch_and_bound_depth (i32) | |
| POSE_GRAPH.constraint_builder.ceres_scan_matcher | table | Dict | CBO→CS2 | ceres_scan_matcher_options (msg) | |
| …ceres_scan_matcher.occupied_space_weight | `20.` | D | CS2 | occupied_space_weight (d) | |
| …ceres_scan_matcher.translation_weight | `10.` | D | CS2 | translation_weight (d) | |
| …ceres_scan_matcher.rotation_weight | `1.` | D | CS2 | rotation_weight (d) | |
| …ceres_scan_matcher.ceres_solver_options | table | Dict | CS2→CSO | ceres_solver_options (msg) | |
| …ceres_scan_matcher.ceres_solver_options.use_nonmonotonic_steps | `true` | B | CSO | use_nonmonotonic_steps (b) | |
| …ceres_scan_matcher.ceres_solver_options.max_num_iterations | `10` | NNI | CSO | max_num_iterations (i32) | CHECK > 0 |
| …ceres_scan_matcher.ceres_solver_options.num_threads | `1` | NNI | CSO | num_threads (i32) | CHECK > 0 |
| POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d | table | Dict | CBO→FC3 | fast_correlative_scan_matcher_options_3d (msg) | |
| …fast_correlative_scan_matcher_3d.branch_and_bound_depth | `8` | I | FC3 | branch_and_bound_depth (i32) | |
| …fast_correlative_scan_matcher_3d.full_resolution_depth | `3` | I | FC3 | full_resolution_depth (i32) | |
| …fast_correlative_scan_matcher_3d.min_rotational_score | `0.77` | D | FC3 | min_rotational_score (d) | |
| …fast_correlative_scan_matcher_3d.min_low_resolution_score | `0.55` | D | FC3 | min_low_resolution_score (d) | |
| …fast_correlative_scan_matcher_3d.linear_xy_search_window | `5.` | D | FC3 | linear_xy_search_window (d) | |
| …fast_correlative_scan_matcher_3d.linear_z_search_window | `1.` | D | FC3 | linear_z_search_window (d) | |
| …fast_correlative_scan_matcher_3d.angular_search_window | `math.rad(15.)` = 0.2617993877991494 | D | FC3 | angular_search_window (d) | deg→rad in Lua |
| POSE_GRAPH.constraint_builder.ceres_scan_matcher_3d | table | Dict | CBO→CS3 | ceres_scan_matcher_options_3d (msg) | |
| …ceres_scan_matcher_3d.occupied_space_weight_0 | `5.` | HasKey+D | CS3 | occupied_space_weight[0] (repeated d) | loop `_0,_1,…` until the first missing key |
| …ceres_scan_matcher_3d.occupied_space_weight_1 | `30.` | HasKey+D | CS3 | occupied_space_weight[1] | |
| …ceres_scan_matcher_3d.translation_weight | `10.` | D | CS3 | translation_weight (d) | |
| …ceres_scan_matcher_3d.rotation_weight | `1.` | D | CS3 | rotation_weight (d) | |
| …ceres_scan_matcher_3d.only_optimize_yaw | `false` | B | CS3 | only_optimize_yaw (b) | |
| …ceres_scan_matcher_3d.ceres_solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `10`, `1` | B, NNI, NNI | CSO | ceres_solver_options.* | |
| (…ceres_scan_matcher_3d.intensity_cost_function_options_N) | absent | – | CS3 | intensity_cost_function_options = [] | absent means an empty repeated field |
| POSE_GRAPH.matcher_translation_weight | `5e2` | D | PGO | matcher_translation_weight (d) | |
| POSE_GRAPH.matcher_rotation_weight | `1.6e3` | D | PGO | matcher_rotation_weight (d) | |
| POSE_GRAPH.optimization_problem | table | Dict | PGO→OPO | optimization_problem_options (msg) | |
| …optimization_problem.huber_scale | `1e1` | D | OPO | huber_scale (d) | |
| …optimization_problem.acceleration_weight | `1.1e2` | D | OPO | acceleration_weight (d) | |
| …optimization_problem.rotation_weight | `1.6e4` | D | OPO | rotation_weight (d) | |
| …optimization_problem.local_slam_pose_translation_weight | `1e5` | D | OPO | local_slam_pose_translation_weight (d) | |
| …optimization_problem.local_slam_pose_rotation_weight | `1e5` | D | OPO | local_slam_pose_rotation_weight (d) | |
| …optimization_problem.odometry_translation_weight | `1e5` | D | OPO | odometry_translation_weight (d) | |
| …optimization_problem.odometry_rotation_weight | `1e5` | D | OPO | odometry_rotation_weight (d) | |
| …optimization_problem.fixed_frame_pose_translation_weight | `1e1` | D | OPO | fixed_frame_pose_translation_weight (d) | |
| …optimization_problem.fixed_frame_pose_rotation_weight | `1e2` | D | OPO | fixed_frame_pose_rotation_weight (d) | |
| …optimization_problem.fixed_frame_pose_use_tolerant_loss | `false` | B | OPO | fixed_frame_pose_use_tolerant_loss (b) | |
| …optimization_problem.fixed_frame_pose_tolerant_loss_param_a | `1` | D | OPO | fixed_frame_pose_tolerant_loss_param_a (d) | integer literal read as a double |
| …optimization_problem.fixed_frame_pose_tolerant_loss_param_b | `1` | D | OPO | fixed_frame_pose_tolerant_loss_param_b (d) | |
| …optimization_problem.log_solver_summary | `false` | B | OPO | log_solver_summary (b) | |
| …optimization_problem.use_online_imu_extrinsics_in_3d | `true` | B | OPO | use_online_imu_extrinsics_in_3d (b) | |
| …optimization_problem.fix_z_in_3d | `false` | B | OPO | fix_z_in_3d (b) | |
| …optimization_problem.ceres_solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `50`, `7` | B, NNI, NNI | CSO | ceres_solver_options.* | |
| POSE_GRAPH.max_num_final_iterations | `200` | NNI | PGO | max_num_final_iterations (i32) | CHECK > 0 |
| POSE_GRAPH.global_sampling_ratio | `0.003` | D | PGO | global_sampling_ratio (d) | |
| POSE_GRAPH.log_residual_histograms | `true` | B | PGO | log_residual_histograms (b) | |
| POSE_GRAPH.global_constraint_search_after_n_seconds | `10.` | D | PGO | global_constraint_search_after_n_seconds (d) | Converted at the use site with `common::FromSeconds` (pose_graph_2d.cc:283, pose_graph_3d.cc:277) |
| POSE_GRAPH.overlapping_submaps_trimmer_2d | **commented out** | HasKey→Dict | PGO (Populate…) | overlapping_submaps_trimmer_2d (msg, optional) | Set only if the key is present |
| …overlapping_submaps_trimmer_2d.fresh_submaps_count | (`1` in the comment) | I | PGO | fresh_submaps_count (i32) | |
| …overlapping_submaps_trimmer_2d.min_covered_area | (`2`) | D | PGO | min_covered_area (d) | |
| …overlapping_submaps_trimmer_2d.min_added_submaps_count | (`5`) | I | PGO | min_added_submaps_count (i32) | |

### 2.2 `map_builder.lua` → `MAP_BUILDER` (MBO → MapBuilderOptions)

| Full key path | Lua default | Getter | Consumer | Proto field | Notes |
|---|---|---|---|---|---|
| MAP_BUILDER.use_trajectory_builder_2d | `false` | B | MBO | use_trajectory_builder_2d (b) | CHECK_NE with the 3D flag, so the user must set one to true |
| MAP_BUILDER.use_trajectory_builder_3d | `false` | B | MBO | use_trajectory_builder_3d (b) | |
| MAP_BUILDER.num_background_threads | `4` | NNI | MBO | num_background_threads (i32) | |
| MAP_BUILDER.pose_graph | `POSE_GRAPH` (**same table reference**) | Dict | MBO→PGO | pose_graph_options (msg) | |
| MAP_BUILDER.collate_by_trajectory | `false` (set to `true` by map_builder_server.lua) | B | MBO | collate_by_trajectory (b) | |

### 2.3 `map_builder_server.lua` → `MAP_BUILDER_SERVER` (MBSO → cloud.MapBuilderServerOptions)

| Full key path | Lua default | Getter | Consumer | Proto field | Notes |
|---|---|---|---|---|---|
| MAP_BUILDER_SERVER.map_builder | `MAP_BUILDER` (reference) | Dict | MBSO→MBO | map_builder_options (msg) | |
| MAP_BUILDER_SERVER.num_event_threads | `4` | I | MBSO | num_event_threads (i32) | |
| MAP_BUILDER_SERVER.num_grpc_threads | `4` | I | MBSO | num_grpc_threads (i32) | |
| MAP_BUILDER_SERVER.server_address | `"0.0.0.0:50051"` | S | MBSO | server_address (str) | |
| MAP_BUILDER_SERVER.uplink_server_address | `""` | S | MBSO | uplink_server_address (str) | empty means no uplink |
| MAP_BUILDER_SERVER.upload_batch_size | `100` | I | MBSO | upload_batch_size (i32) | |
| MAP_BUILDER_SERVER.enable_ssl_encryption | `false` | B | MBSO | enable_ssl_encryption (b) | |
| MAP_BUILDER_SERVER.enable_google_auth | `false` | B | MBSO | enable_google_auth (b) | |
| (statement) `MAP_BUILDER.collate_by_trajectory = true` | – | – | – | – | Runs **after** the `MAP_BUILDER_SERVER` table is built, and still takes effect through reference aliasing |

### 2.4 `trajectory_builder.lua` → `TRAJECTORY_BUILDER` (TBO → TrajectoryBuilderOptions)

| Full key path | Lua default | Getter | Consumer | Proto field | Notes |
|---|---|---|---|---|---|
| TRAJECTORY_BUILDER.trajectory_builder_2d | `TRAJECTORY_BUILDER_2D` (ref) | Dict | TBO→LT2 | trajectory_builder_2d_options (msg) | **Always required**, even for 3D use |
| TRAJECTORY_BUILDER.trajectory_builder_3d | `TRAJECTORY_BUILDER_3D` (ref) | Dict | TBO→LT3 | trajectory_builder_3d_options (msg) | **Always required**, even for 2D use |
| TRAJECTORY_BUILDER.collate_fixed_frame | `true` | B | TBO | collate_fixed_frame (b) | |
| TRAJECTORY_BUILDER.collate_landmarks | `false` | B | TBO | collate_landmarks (b) | |
| TRAJECTORY_BUILDER.pure_localization_trimmer | **commented out** | HasKey→Dict | TBO (Populate…) | pure_localization_trimmer (msg, optional) | |
| …pure_localization_trimmer.max_submaps_to_keep | (`3` in the comment) | I | TBO | max_submaps_to_keep (i32) | |
| TRAJECTORY_BUILDER.pose_graph_odometry_motion_filter | **not in any Lua file** | HasKey→Dict | TBO (Populate…) | pose_graph_odometry_motion_filter (MotionFilterOptions, optional) | Read with direct `GetDouble` calls, not via MF |
| …pose_graph_odometry_motion_filter.max_time_seconds / max_distance_meters / max_angle_radians | – | D ×3 | TBO | max_time_seconds / max_distance_meters / max_angle_radians (d) | |
| (no Lua) | – | – | – | initial_trajectory_pose, pure_localization (deprecated) | Set programmatically, never from Lua |

### 2.5 `trajectory_builder_2d.lua` → `TRAJECTORY_BUILDER_2D` (LT2 → LocalTrajectoryBuilderOptions2D)

| Full key path | Lua default | Getter | Consumer | Proto field | Notes |
|---|---|---|---|---|---|
| TRAJECTORY_BUILDER_2D.use_imu_data | `true` | B | LT2 | use_imu_data (b) | |
| TRAJECTORY_BUILDER_2D.min_range | `0.` | D | LT2 | min_range (**f**) | double→float narrowing |
| TRAJECTORY_BUILDER_2D.max_range | `30.` | D | LT2 | max_range (f) | |
| TRAJECTORY_BUILDER_2D.min_z | `-0.8` | D | LT2 | min_z (f) | unary minus |
| TRAJECTORY_BUILDER_2D.max_z | `2.` | D | LT2 | max_z (f) | |
| TRAJECTORY_BUILDER_2D.missing_data_ray_length | `5.` | D | LT2 | missing_data_ray_length (f) | |
| TRAJECTORY_BUILDER_2D.num_accumulated_range_data | `1` | I | LT2 | num_accumulated_range_data (i32) | |
| TRAJECTORY_BUILDER_2D.voxel_filter_size | `0.025` | D | LT2 | voxel_filter_size (f) | |
| TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length | `0.5` | D | AVF | adaptive_voxel_filter_options.max_length (f) | |
| TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points | `200` | NNI | AVF | …min_num_points (**f**) | int read, stored as float |
| TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range | `50.` | D | AVF | …max_range (f) | |
| TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.max_length | `0.9` | D | AVF | loop_closure_adaptive_voxel_filter_options.max_length (f) | |
| TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.min_num_points | `100` | NNI | AVF | …min_num_points (f) | |
| TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter.max_range | `50.` | D | AVF | …max_range (f) | |
| TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching | `false` | B | LT2 | use_online_correlative_scan_matching (b) | |
| TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window | `0.1` | D | RTC | real_time_correlative_scan_matcher_options.linear_search_window (d) | |
| …real_time_correlative_scan_matcher.angular_search_window | `math.rad(20.)` = 0.3490658503988659 | D | RTC | …angular_search_window (d) | deg→rad in Lua |
| …real_time_correlative_scan_matcher.translation_delta_cost_weight | `1e-1` | D | RTC | …translation_delta_cost_weight (d) | CHECK ≥ 0 |
| …real_time_correlative_scan_matcher.rotation_delta_cost_weight | `1e-1` | D | RTC | …rotation_delta_cost_weight (d) | CHECK ≥ 0 |
| TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight | `1.` | D | CS2 | ceres_scan_matcher_options.occupied_space_weight (d) | |
| …ceres_scan_matcher.translation_weight | `10.` | D | CS2 | …translation_weight (d) | |
| …ceres_scan_matcher.rotation_weight | `40.` | D | CS2 | …rotation_weight (d) | |
| …ceres_scan_matcher.ceres_solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `20`, `1` | B, NNI, NNI | CSO | …ceres_solver_options.* | |
| TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds | `5.` | D | MF | motion_filter_options.max_time_seconds (d) | Use site: `common::FromSeconds` (motion_filter.cc:47) |
| …motion_filter.max_distance_meters | `0.2` | D | MF | …max_distance_meters (d) | |
| …motion_filter.max_angle_radians | `math.rad(1.)` = 0.017453292519943295 | D | MF | …max_angle_radians (d) | deg→rad in Lua |
| TRAJECTORY_BUILDER_2D.imu_gravity_time_constant | `10.` | D | LT2 | imu_gravity_time_constant (d) | Legacy key (ROS), still mandatory |
| TRAJECTORY_BUILDER_2D.pose_extrapolator.use_imu_based | `false` | B | PEO | pose_extrapolator_options.use_imu_based (b) | |
| …pose_extrapolator.constant_velocity.imu_gravity_time_constant | `10.` | D | PEO(CV) | …constant_velocity.imu_gravity_time_constant (d) | |
| …pose_extrapolator.constant_velocity.pose_queue_duration | `0.001` | D | PEO(CV) | …constant_velocity.pose_queue_duration (d) | seconds; `FromSeconds` at use (local_trajectory_builder_2d.cc:325, pose_extrapolator_interface.cc:92) |
| …pose_extrapolator.imu_based.pose_queue_duration | `5.` | D | PEO(IMU) | …imu_based.pose_queue_duration (d) | seconds; `FromSeconds` at use (imu_based_pose_extrapolator.cc:90) |
| …pose_extrapolator.imu_based.gravity_constant | `9.806` | D | PEO(IMU) | …gravity_constant (d) | |
| …pose_extrapolator.imu_based.pose_translation_weight | `1.` | D | PEO(IMU) | …pose_translation_weight (d) | |
| …pose_extrapolator.imu_based.pose_rotation_weight | `1.` | D | PEO(IMU) | …pose_rotation_weight (d) | |
| …pose_extrapolator.imu_based.imu_acceleration_weight | `1.` | D | PEO(IMU) | …imu_acceleration_weight (d) | |
| …pose_extrapolator.imu_based.imu_rotation_weight | `1.` | D | PEO(IMU) | …imu_rotation_weight (d) | |
| …pose_extrapolator.imu_based.odometry_translation_weight | `1.` | D | PEO(IMU) | …odometry_translation_weight (d) | |
| …pose_extrapolator.imu_based.odometry_rotation_weight | `1.` | D | PEO(IMU) | …odometry_rotation_weight (d) | |
| …pose_extrapolator.imu_based.solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `10`, `1` | B, NNI, NNI | CSO | …imu_based.solver_options.* | **`;` used as the field separator** in the Lua source |
| TRAJECTORY_BUILDER_2D.submaps.num_range_data | `90` | NNI | SO2 | submaps_options.num_range_data (i32) | CHECK > 0 |
| …submaps.grid_options_2d.grid_type | `"PROBABILITY_GRID"` | S | GO2 | …grid_options_2d.grid_type (enum GridType) | string→enum via `GridOptions2D_GridType_Parse`. Valid: `INVALID_GRID`, `PROBABILITY_GRID`, `TSDF` |
| …submaps.grid_options_2d.resolution | `0.05` | D | GO2 | …grid_options_2d.resolution (**f**) | |
| …submaps.range_data_inserter.range_data_inserter_type | `"PROBABILITY_GRID_INSERTER_2D"` | S | RDI | …range_data_inserter_options.range_data_inserter_type (enum) | string→enum via `_Parse`. Valid: `INVALID_INSERTER`, `PROBABILITY_GRID_INSERTER_2D`, `TSDF_INSERTER_2D`. Must pair with grid_type |
| …range_data_inserter.probability_grid_range_data_inserter.insert_free_space | `true` | **HasKey ? B : true** | PGI | …probability_grid_range_data_inserter_options_2d.insert_free_space (b) | **The only key with a C++ default** (true) |
| …probability_grid_range_data_inserter.hit_probability | `0.55` | D | PGI | …hit_probability (d) | CHECK > 0.5. Converted to odds and lookup tables at the use site |
| …probability_grid_range_data_inserter.miss_probability | `0.49` | D | PGI | …miss_probability (d) | CHECK < 0.5 |
| …range_data_inserter.tsdf_range_data_inserter.truncation_distance | `0.3` | D | TSI | …tsdf_range_data_inserter_options_2d.truncation_distance (d) | Always parsed, even when the PROBABILITY_GRID inserter is selected |
| …tsdf_range_data_inserter.maximum_weight | `10.` | D | TSI | …maximum_weight (d) | |
| …tsdf_range_data_inserter.update_free_space | `false` | B | TSI | …update_free_space (b) | |
| …tsdf_range_data_inserter.normal_estimation_options.num_normal_samples | `4` | I | NE2 | …normal_estimation_options.num_normal_samples (i32) | CHECK > 0 |
| …tsdf_range_data_inserter.normal_estimation_options.sample_radius | `0.5` | D | NE2 | …sample_radius (**f**) | CHECK > 0 |
| …tsdf_range_data_inserter.project_sdf_distance_to_scan_normal | `true` | B | TSI | …project_sdf_distance_to_scan_normal (b) | |
| …tsdf_range_data_inserter.update_weight_range_exponent | `0` | I | TSI | …update_weight_range_exponent (i32) | |
| …tsdf_range_data_inserter.update_weight_angle_scan_normal_to_ray_kernel_bandwidth | `0.5` | D | TSI | same name (d) | |
| …tsdf_range_data_inserter.update_weight_distance_cell_to_hit_kernel_bandwidth | `0.5` | D | TSI | same name (d) | |

### 2.6 `trajectory_builder_3d.lua` → `TRAJECTORY_BUILDER_3D` (LT3 → LocalTrajectoryBuilderOptions3D)

The file also defines globals `MAX_3D_RANGE = 60.` and `INTENSITY_THRESHOLD = 40`. These are not option keys; they are referenced by identifier below. Globals are not part of any returned table, so they are never ref-counted.

| Full key path | Lua default | Getter | Consumer | Proto field | Notes |
|---|---|---|---|---|---|
| TRAJECTORY_BUILDER_3D.min_range | `1.` | D | LT3 | min_range (f) | |
| TRAJECTORY_BUILDER_3D.max_range | `MAX_3D_RANGE` (=60.) | D | LT3 | max_range (f) | value comes from a global variable |
| TRAJECTORY_BUILDER_3D.num_accumulated_range_data | `1` | I | LT3 | num_accumulated_range_data (i32) | |
| TRAJECTORY_BUILDER_3D.voxel_filter_size | `0.15` | D | LT3 | voxel_filter_size (f) | |
| …high_resolution_adaptive_voxel_filter.{max_length, min_num_points, max_range} | `2.`, `150`, `15.` | D, NNI, D | AVF | high_resolution_adaptive_voxel_filter_options.* (f, f, f) | |
| …low_resolution_adaptive_voxel_filter.{max_length, min_num_points, max_range} | `4.`, `200`, `MAX_3D_RANGE` | D, NNI, D | AVF | low_resolution_adaptive_voxel_filter_options.* (f, f, f) | |
| TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching | `false` | B | LT3 | use_online_correlative_scan_matching (b) | |
| …real_time_correlative_scan_matcher.linear_search_window | `0.15` | D | RTC | real_time_correlative_scan_matcher_options.linear_search_window (d) | |
| …real_time_correlative_scan_matcher.angular_search_window | `math.rad(1.)` | D | RTC | …angular_search_window (d) | deg→rad in Lua |
| …real_time_correlative_scan_matcher.translation_delta_cost_weight | `1e-1` | D | RTC | … (d) | CHECK ≥ 0 |
| …real_time_correlative_scan_matcher.rotation_delta_cost_weight | `1e-1` | D | RTC | … (d) | CHECK ≥ 0 |
| …ceres_scan_matcher.occupied_space_weight_0 | `1.` | HasKey+D | CS3 | ceres_scan_matcher_options.occupied_space_weight[0] (repeated d) | |
| …ceres_scan_matcher.occupied_space_weight_1 | `6.` | HasKey+D | CS3 | occupied_space_weight[1] | |
| …ceres_scan_matcher.intensity_cost_function_options_0.weight | `0.5` | D | CS3 | intensity_cost_function_options[0].weight (d) | loop `_0,_1,…` |
| …ceres_scan_matcher.intensity_cost_function_options_0.huber_scale | `0.3` | D | CS3 | [0].huber_scale (d) | |
| …ceres_scan_matcher.intensity_cost_function_options_0.intensity_threshold | `INTENSITY_THRESHOLD` (=40) | D | CS3 | [0].intensity_threshold (**f**) | |
| …ceres_scan_matcher.translation_weight | `5.` | D | CS3 | translation_weight (d) | |
| …ceres_scan_matcher.rotation_weight | `4e2` | D | CS3 | rotation_weight (d) | |
| …ceres_scan_matcher.only_optimize_yaw | `false` | B | CS3 | only_optimize_yaw (b) | |
| …ceres_scan_matcher.ceres_solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `12`, `1` | B, NNI, NNI | CSO | ceres_solver_options.* | |
| …motion_filter.{max_time_seconds, max_distance_meters, max_angle_radians} | `0.5`, `0.1`, `0.004` | D ×3 | MF | motion_filter_options.* (d) | the angle is given in radians directly (no `math.rad`) |
| TRAJECTORY_BUILDER_3D.rotational_histogram_size | `120` | I | LT3 | rotational_histogram_size (i32) | |
| TRAJECTORY_BUILDER_3D.imu_gravity_time_constant | `10.` | D | LT3 | imu_gravity_time_constant (d) | legacy |
| …pose_extrapolator.use_imu_based | `false` | B | PEO | pose_extrapolator_options.use_imu_based | |
| …pose_extrapolator.constant_velocity.{imu_gravity_time_constant, pose_queue_duration} | `10.`, `0.001` | D, D | PEO(CV) | …constant_velocity.* (d) | |
| …pose_extrapolator.imu_based.{pose_queue_duration, gravity_constant, pose_translation_weight, pose_rotation_weight, imu_acceleration_weight, imu_rotation_weight, odometry_translation_weight, odometry_rotation_weight} | `5.`, `9.806`, `1.` ×6 | D ×8 | PEO(IMU) | …imu_based.* (d) | identical to 2D |
| …pose_extrapolator.imu_based.solver_options.{use_nonmonotonic_steps, max_num_iterations, num_threads} | `false`, `10`, `1` | B, NNI, NNI | CSO | …solver_options.* | `;` separators |
| …submaps.high_resolution | `0.10` | D | SO3 | submaps_options.high_resolution (d) | |
| …submaps.high_resolution_max_range | `20.` | D | SO3 | …high_resolution_max_range (d) | |
| …submaps.low_resolution | `0.45` | D | SO3 | …low_resolution (d) | |
| …submaps.num_range_data | `160` | NNI | SO3 | …num_range_data (i32) | CHECK > 0 |
| …submaps.range_data_inserter.hit_probability | `0.55` | D | RD3 | …range_data_inserter_options.hit_probability (d) | CHECK > 0.5 |
| …submaps.range_data_inserter.miss_probability | `0.49` | D | RD3 | …miss_probability (d) | CHECK < 0.5 |
| …submaps.range_data_inserter.num_free_space_voxels | `2` | I | RD3 | …num_free_space_voxels (i32) | |
| …submaps.range_data_inserter.intensity_threshold | `INTENSITY_THRESHOLD` (=40) | D | RD3 | …intensity_threshold (**f**) | |
| TRAJECTORY_BUILDER_3D.use_intensities | `false` | B | LT3 | use_intensities (b) | When true, CS3 needs `intensity_cost_function_options_0` (comment in the Lua file) |
| (no Lua) | – | – | – | initial_poses, initial_imu_data | Set programmatically |

### 2.7 Non-trivial conversions (summary)

| Kind | Where | Detail |
|---|---|---|
| deg→rad | Lua `math.rad(x)` in all `angular_search_window` and `motion_filter.max_angle_radians` (2D) values | Done by the **Lua interpreter**; C++ only sees radians. Lua 5.2 computes `x * (PI/180.0)`, which matches Rust `f64::to_radians` bit-for-bit. |
| seconds→Duration | `pose_queue_duration` (CV and IMU), `motion_filter.max_time_seconds`, `global_constraint_search_after_n_seconds` | Stored as `double` seconds in the proto. At the use site: `common::FromSeconds(d)` = `duration_cast<Duration>(duration<double>(s))`. Duration ticks are 100 ns and the cast **truncates** toward zero. |
| string→enum | `grid_type`, `range_data_inserter_type` | `*_Parse`. Exact, case-sensitive proto value name. Unknown names are fatal. |
| indexed keys→repeated | `occupied_space_weight_N`, `intensity_cost_function_options_N` | Probes N = 0, 1, … with `HasKey` and stops at the first gap (a later `_5` after a missing `_3` would be **unused → fatal** at dict destruction). |
| optional subtables | `overlapping_submaps_trimmer_2d`, `pure_localization_trimmer`, `pose_graph_odometry_motion_filter` | `HasKey` gate. If absent, the proto sub-message is unset (`Option::None`). |
| key-level default | `probability_grid_range_data_inserter.insert_free_space` | `true` if absent. |
| double→float | `min_range`, `max_range`, `min_z`, `max_z`, `missing_data_ray_length`, `voxel_filter_size`, AVF `max_length`/`max_range`, `grid_options_2d.resolution`, `sample_radius`, `intensity_threshold` (×2) | `f64 as f32` (round-to-nearest). |
| int→float | AVF `min_num_points` (`GetNonNegativeInt` → float field) | |
| probability→odds | hit/miss probabilities | At the use site (`ComputeLookupTableToApplyOdds` etc.), not in `Create*`. |

### 2.8 Other Lua consumers (not `Create*Options`, but they share the dictionary)

| Consumer | Keys (getter) | Notes |
|---|---|---|
| `transform::FromDictionary` (`transform/rigid_transform.cc`) | `translation` = array of 3 (`GetArrayValuesAsDoubles`). `rotation` = either `{w=,x=,y=,z=}` (D ×4, `CHECK_NEAR(norm,1,1e-9)`) or the array `{roll,pitch,yaw}` | RPY → `Quaterniond(yaw·Z) * (pitch·Y) * (roll·X)`. Used by `write_xray_image.transform` and by cartographer_ros. |
| `io::PointsProcessorPipelineBuilder::CreatePipeline` (`io/points_processor_pipeline_builder.cc`) | the dict itself is an **array** of tables (`GetArrayValuesAsDictionaries`). Each has `action` (S) | Built back to front. Actions and keys: `min_max_range_filter{min_range D, max_range D}`, `dump_num_points{}`, `fixed_ratio_sampler{sampling_ratio D}`, `frame_id_filter{keep_frames?/drop_frames? = string arrays}`, `vertical_range_filter{min_z, max_z}`, `voxel_filter_and_remove_moving_objects{voxel_size D, miss_per_hit_limit? D=3}`, `color_points{frame_id S, color = array of 3 doubles → uint8}`, `intensity_to_color{frame_id? S="", min_intensity D, max_intensity D}`, `write_ply{filename}`, `write_pcd{filename}`, `write_xyz{filename}`, `write_hybrid_grid{voxel_size, range_data_inserter→RD3, filename}`, `write_probability_grid{resolution, range_data_inserter→PGI, draw_trajectories? B=true, output_type? "png"/"pb" (default png), filename}`, `write_xray_image{voxel_size, filename, transform→FromDictionary, separate_floors? B=false, draw_trajectories? B=true, saturation_factor? D=1}` |
| `io::MigrateStreamVersion1ToVersion2` (`io/serialization_format_migration.cc`) | inline Lua: `include "map_builder.lua"; MAP_BUILDER.use_trajectory_builder_3d = true; return MAP_BUILDER` | Uses ConfigurationFileResolver on `kSourceDirectory/configuration_files`. |
| `common/print_configuration_main.cc` | any | `NonReferenceCounted`; prints `"return " + ToString()`, or a sub-dictionary chosen by the dotted path `--subdictionary a.b.c`. |

---

## 3. Lua language subset

### 3.1 Host environment provided by `LuaParameterDictionary`

| Item | Behavior |
|---|---|
| State | A fresh `luaL_newstate()` per top-level dictionary, with `luaL_openlibs()` (the **full** stdlib is available, so configs could use anything; see §3.3 for what is actually used). |
| `include(basename)` | `FileResolver::GetFullPathOrDie(basename)`, then **fatal if that full path was already included** in this state (a double include, even an indirect one, aborts). Loads the file as a chunk named by its full path and runs it in the **same global environment** (`lua_pcall(..., LUA_MULTRET)`). Returns whatever the chunk returns; config files return nothing. |
| `read(basename)` | Returns the file contents as a string. Not used by any shipped config. |
| `choose(cond, a, b)` | Exactly 3 args. `cond` must be a boolean (else fatal "condition is not a boolean value."). Returns `a` or `b`. Not used by shipped configs; only by tests. |
| File resolution (`ConfigurationFileResolver`) | Searches the caller's dirs in order, then **always appends** `kConfigurationFilesDirectory` (the installed `configuration_files/`). First match of `dir + "/" + basename` that opens wins. A missing file is fatal. |
| Top-level contract | The top-level code is run with `lua_pcall(L,0,1,0)` and **must `return` a table** (CHECK). Included base files only define globals. |
| Errors | Every Lua syntax or runtime error is a `CHECK` failure (process abort) in C++. |

### 3.2 Constructs used in `cartographer/configuration_files/*.lua` (exhaustive)

| Construct | Example | Files |
|---|---|---|
| Line comment `--` to EOL (including the license header, commented-out table blocks, and trailing comment lines inside table constructors) | `--  overlapping_submaps_trimmer_2d = {` | all |
| `include "<file>"` (call **without parentheses**, single string-literal argument) | `include "pose_graph.lua"` | map_builder, map_builder_server, trajectory_builder |
| Global assignment of a table constructor | `POSE_GRAPH = { ... }` | all |
| Global assignment of a number | `MAX_3D_RANGE = 60.`, `INTENSITY_THRESHOLD = 40` | trajectory_builder_3d |
| Global variable reference as a value (**table by reference**) | `pose_graph = POSE_GRAPH`, `map_builder = MAP_BUILDER`, `trajectory_builder_2d = TRAJECTORY_BUILDER_2D` | map_builder, map_builder_server, trajectory_builder |
| Global variable reference (number) | `max_range = MAX_3D_RANGE`, `intensity_threshold = INTENSITY_THRESHOLD` | trajectory_builder_3d |
| Field assignment on an existing table (dotted path) | `MAP_BUILDER.collate_by_trajectory = true` | map_builder_server |
| Nested table constructors with `name = value` fields | everywhere | all |
| Field separator `,` **and `;`** (mixed in the same file), trailing separator allowed, blank lines inside constructors | `use_nonmonotonic_steps = false;` | trajectory_builder_2d/3d (`solver_options`) |
| Number literals: integer `90`; trailing-dot float `15.`, `0.`; decimal `0.025`, `0.10`; exponent `1.1e4`, `1e5`, `5e2`, `1.6e3`, `1e-1`, `4e2`, `1e1` | | all |
| Unary minus on a literal | `min_z = -0.8` | trajectory_builder_2d |
| String literals (double-quoted), including the empty string | `"PROBABILITY_GRID"`, `"0.0.0.0:50051"`, `""` | trajectory_builder_2d, map_builder_server |
| Booleans | `true`, `false` | all |
| Function call with parentheses, library field access | `math.rad(30.)` | pose_graph, trajectory_builder_2d/3d |
| **Not used** in the base configs | `local`, `return`, arithmetic binary ops, `..`, `if`/`for`/`while`, function definitions, `[k]=v` keys, positional array items, `--[[ ]]` block comments, long strings, single-quoted strings, `nil` | |

### 3.3 Additional constructs used in C++ test Lua strings (§4) and `ToString` round-trips

| Construct | Example | Where |
|---|---|---|
| `return <table constructor>` | `return { blah = 100 }` | most `MakeDictionary` tests |
| `return <global>` / `return <global>.<field>` | `return MAP_BUILDER`, `return POSE_GRAPH.constraint_builder` | map_builder_test, constraint_builder_{2d,3d}_test, pose_graph_3d_test, client_server_test, configuration_files_test |
| Multi-level dotted assignment | `MAP_BUILDER.pose_graph.optimize_every_n_nodes = 0`, `TRAJECTORY_BUILDER.trajectory_builder_2d.submaps.num_range_data = 4`, `POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher_3d.min_low_resolution_score = 0` | map_builder_test, client_server_test, constraint_builder_3d_test |
| Single-quoted strings | `'is_a_string'`, `'hello'`, `{ 'a', 'b', 'c' }` | lua_parameter_dictionary_test |
| Positional (array) items, including mixed with keyed fields | `{ 1., 2., 3. }`, `{ { a = 1 }, { b = 3 } }`, `{ "blub", 3, 3.1, foo = "ups" }` | lua_parameter_dictionary_test |
| `math.huge`, `-math.huge` (unary minus on an expression) | `fasel1 = -math.huge` | lua_parameter_dictionary_test (also emitted by `ToString`) |
| `choose(bool, a, b)` | `choose(true, 1, 0)` | lua_parameter_dictionary_test |
| Global assigned, then returned | `pipeline = { {...} } return pipeline` | io/probability_grid_points_processor_test |
| Negative integer | `blah = -100` | lua_parameter_dictionary_test |
| Missing trailing comma before `}` | `branch_and_bound_depth = 3}` | fast_correlative_scan_matcher_2d_test |
| `%q`-quoted strings (from `ToString`) | `"hello"` with `\"`, `\\`, `\` + newline, `\r`, `\0` escapes | ToString → reparse round-trip test |

### 3.4 cartographer_ros-style extension patterns (from upstream knowledge; **not verified against a checkout**, since cartographer_ros is not in this repo)

| Pattern | Example |
|---|---|
| Include both base files, then build a node-options table that references them | `include "map_builder.lua"` `include "trajectory_builder.lua"` `options = { map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER, map_frame = "map", tracking_frame = "base_link", ..., pose_publish_period_sec = 5e-3, ... }` |
| Mutate the base globals **after** building `options` (this works only because of reference semantics) | `MAP_BUILDER.use_trajectory_builder_2d = true` `TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 10` `POSE_GRAPH.optimize_every_n_nodes = 0` `TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)` |
| Config inheritance: include another *user* config, then tweak it | `include "backpack_2d.lua"` then `TRAJECTORY_BUILDER.pure_localization_trimmer = { max_submaps_to_keep = 3 }` and `POSE_GRAPH.optimize_every_n_nodes = 20` |
| Assigning a new subtable to enable optional features | `POSE_GRAPH.overlapping_submaps_trimmer_2d = { fresh_submaps_count = 1, min_covered_area = 2, min_added_submaps_count = 5 }` |
| Global constants and arithmetic | `VOXEL_SIZE = 5e-2`; `transform.lua`: `XY_TRANSFORM = { translation = {0., 0., 0.}, rotation = {0., -math.pi / 2., 0.} }` (`math.pi`, `/`, unary minus on an expression) |
| Assets-writer pipelines: array of action tables | `options = { tracking_frame = "base_link", pipeline = { { action = "min_max_range_filter", min_range = 1., max_range = 60. }, { action = "write_xray_image", voxel_size = VOXEL_SIZE, filename = "xray_yz_all", transform = YZ_TRANSFORM }, ... } }` |
| Final line | `return options` |

The top-level `options` dict is itself reference-counted. cartographer_ros' `CreateNodeOptions`/`CreateTrajectoryOptions` consume `map_builder` and `trajectory_builder` through `CreateMapBuilderOptions` / `CreateTrajectoryBuilderOptions`.

### 3.5 Interpreter requirements derived from the above

| Requirement | Must / Should | Reason |
|---|---|---|
| Tables are **reference types** with shared mutation (`Rc<RefCell<Table>>`) | **Must** | `MAP_BUILDER_SERVER.map_builder = MAP_BUILDER` followed by `MAP_BUILDER.collate_by_trajectory = true`; tests set `MAP_BUILDER.pose_graph.*` and expect POSE_GRAPH-derived values; ROS configs mutate globals after building `options`. |
| One global environment shared across `include`d chunks; `include` returns the chunk's results | Must | |
| Include-once, keyed by resolved full path → error | Must | Matches C++ (fatal). |
| Lexer: `--` comments, `--[[ ]]`/`--[==[ ]==]` block comments, `"..."` and `'...'` strings with the standard escapes (`\n \t \\ \" \' \r \a \b \f \v \ddd \xhh \z`, backslash-newline), long strings `[[...]]` | Must (comments and strings) / Should (long strings, `\z`) | `%q` round-trips; ROS usage. |
| Numbers: decimal, trailing dot, leading dot (`.5`), exponents, hex `0x..` | Must (all f64) | Lua 5.2 has no integer type. Keep `f64` throughout, but keep enough information that `ToString` prints `123.000000` (it uses `%f`, see §4.3). |
| Expressions: unary `-`, `not`; binary `+ - * / % ^ ..`, comparisons, `and`/`or` with Lua short-circuit and truthiness; parentheses | Must: unary `-`, `* /`, `+ -`. Should: the rest | `-math.pi / 2.` in ROS transform.lua; users write arithmetic like `1/3`. |
| Table constructors: `name = e`, `[e] = e`, positional items, `,`/`;` separators, trailing separator | Must (incl. positional) | Arrays are used by FromDictionary and pipelines. |
| Statements: global assignment, dotted/indexed assignment (`a.b.c = v`, `a["k"] = v`), `local x = e`, function-call statement (`include "x"`, `f(...)`, `f{...}`, `f"str"` sugar), `return e` (trailing) | Must | |
| Statements: `if/elseif/else`, numeric/generic `for`, `while`, `function` defs, multiple assignment | Should (cheap to add; real-world configs occasionally use them) | Not used in this repo's configs or tests. |
| Library: `math.rad`, `math.pi`, `math.huge` | Must | |
| Library: `math.deg/sin/cos/sqrt/abs/floor/ceil/min/max`, `string.format`, `tostring`, `tonumber`, `print`, `ipairs/pairs` | Should | Commonly used in hand-written configs. |
| Builtins: `include`, `read`, `choose` | Must | Same semantics as §3.1. |
| Assigning `nil` deletes the key; reading a missing global gives `nil`; indexing `nil` is an error | Must | |
| `GetString` accepts numbers (`%.14g`); `GetDouble`/`GetInt` accept numeric strings | Must, for C++ parity | |
| `GetInt` on a non-integral value truncates toward zero | Must, for 5.2 parity | (or reject it, as a deliberate stricter choice) |
| Exactly-once key usage check per dictionary wrapper (§4.1) | Must | Typo detection. |

---

## 4. `LuaParameterDictionary` in tests

### 4.1 Core API semantics worth replicating

| Aspect | C++ behavior |
|---|---|
| Construction | `LuaParameterDictionary(code, unique_ptr<FileResolver>)` is reference-counted. `NonReferenceCounted(code, resolver)` does not check (used by `print_configuration` and some tests). |
| Ref counting | Each `Get*` / `GetDictionary` call does `reference_counts_[key]++`. On destruction (ref-counted mode), every non-numeric key from `GetKeys()` must have count **== 1**; otherwise `CHECK` "Key 'x' was used the wrong number of times." Reading a key twice is also fatal. `HasKey` and the array getters are not counted. Counts are kept **per wrapper object**: each `GetDictionary` creates a new wrapper (via `lua_newthread`), so an aliased table reached by two paths is counted independently per path. |
| Missing key | `CHECK` "Key 'k' not in dictionary:\n" + `ToString()`. |
| `ToString()` | Array part first (`t[1..n]` until nil), then string keys **sorted**. Format: `{\n<indent+2>value,\n...<indent>}`, with `key = value,` per entry. Numbers use `std::to_string` (`%f`, 6 decimals, e.g. `123.000000`; tiny values like `1e-7` print `0.000000`, which is **lossy**). ±inf prints as `math.huge` / `-math.huge`. Strings are `%q`-quoted. Nested indent is +2. An empty table prints `{}`. Other types (functions, etc.) are fatal. |
| `GetKeys()` | String keys only (numeric-looking keys are excluded). |

### 4.2 Test helpers

| Helper | Location | Behavior |
|---|---|---|
| `common::DummyFileResolver` | `common/internal/testing/lua_parameter_dictionary_test_helpers.h` | Both methods `LOG(FATAL) "Not implemented"`, so `include`/`read` are fatal. Snippets must be self-contained. |
| `common::MakeDictionary(code)` | same header (a non-inline free function defined in the header) | `make_unique<LuaParameterDictionary>(code, DummyFileResolver)`, ref-counted. |
| `MakeNonReferenceCounted(code)` | `common/lua_parameter_dictionary_test.cc` (local) | `NonReferenceCounted(code, DummyFileResolver)`. |
| `mapping::testing::ResolveLuaParameters(code)` | `mapping/internal/testing/test_helpers.{h,cc}` | `ConfigurationFileResolver({kSourceDirectory + "/configuration_files"})` (plus the implicit install dir). Ref-counted. Lets snippets `include` the real base configs. |
| Inline equivalent | `common/configuration_files_test.cc`, `io/serialization_format_migration.cc` | Same as `ResolveLuaParameters`, written out by hand. |

Rust analogue: a `FileResolver` trait with a `DummyResolver` (errors on include), a `DirResolver(Vec<PathBuf>)` that appends `concat!(env!("CARGO_MANIFEST_DIR"), "/configuration_files")`, and `make_dictionary(code)` / `resolve_lua_parameters(code)` test helpers.

### 4.3 Inventory of test Lua snippets and what they build

All of these are ref-counted, so **each snippet is an exact, complete key set** for the `Create*` it feeds. That makes them good golden fixtures.

| Test file | Helper | Snippet shape | Fed to |
|---|---|---|---|
| `common/lua_parameter_dictionary_test.cc` | MakeDictionary / MakeNonReferenceCounted | Scalars (`100`, `'is_a_string'`, `3.1415`, `true/false`, `-100`), nested dict, `math.huge`, arrays of strings/doubles/dicts, mixed array+keys, `choose(...)`, ToString golden output plus a re-parse of `"return " .. ToString()` | API tests: `GetInt` on a missing key → death; `GetNonNegativeInt(-100)` → death "-100 is negative."; `GetString` on the number `10` → `"10"`; `choose('truish',…)` → death |
| `common/configuration_files_test.cc` | inline resolver | `include "map_builder.lua"; MAP_BUILDER.use_trajectory_builder_2d = true; return MAP_BUILDER` and `include "trajectory_builder.lua"; TRAJECTORY_BUILDER.trajectory_builder_2d.use_imu_data = false; return TRAJECTORY_BUILDER` | `CreateMapBuilderOptions`, `CreateTrajectoryBuilderOptions` (validates that the shipped configs are fully consumed) |
| `mapping/map_builder_test.cc` | ResolveLuaParameters | `include "map_builder.lua"` + `MAP_BUILDER.use_trajectory_builder_2d = true`, `.pose_graph.optimize_every_n_nodes = 0`, `.pose_graph.global_sampling_ratio = 0.05`, `.pose_graph.global_constraint_search_after_n_seconds = 0`; `include "trajectory_builder.lua"` + `trajectory_builder_2d.use_imu_data = false`, `trajectory_builder_{2d,3d}.submaps.num_range_data = 4` | MBO, TBO |
| `cloud/internal/client_server_test.cc` | ResolveLuaParameters | `include "map_builder_server.lua"` + `MAP_BUILDER.use_trajectory_builder_2d = true`, `MAP_BUILDER.pose_graph.optimize_every_n_nodes = 0`, `MAP_BUILDER_SERVER.{num_event_threads=1, num_grpc_threads=1, uplink_server_address="" or "localhost:50051", server_address="0.0.0.0:50051"/"…50052", upload_batch_size=1}`; trajectory_builder snippet as above | MBSO, TBO |
| `mapping/internal/3d/pose_graph_3d_test.cc` | ResolveLuaParameters | `include "pose_graph.lua"; return POSE_GRAPH` | PGO |
| `mapping/internal/constraints/constraint_builder_2d_test.cc` | ResolveLuaParameters | `include "pose_graph.lua"` + `POSE_GRAPH.constraint_builder.{sampling_ratio=1, min_score=0, global_localization_min_score=0}`; `return POSE_GRAPH.constraint_builder` | CBO |
| `mapping/internal/constraints/constraint_builder_3d_test.cc` | ResolveLuaParameters | as 2D, plus `…fast_correlative_scan_matcher_3d.{min_low_resolution_score=0, min_rotational_score=0}` | CBO |
| `mapping/internal/2d/pose_graph_2d_test.cc` | MakeDictionary | full inline SubmapsOptions2D (num_range_data=1, hit 0.53 / miss 0.495, tsdf with zero bandwidths) and full inline PoseGraphOptions (ceres_scan_matcher_3d with only `occupied_space_weight_0`, no intensity) | SO2, PGO |
| `mapping/internal/optimization/optimization_problem_3d_test.cc` | MakeDictionary | full OptimizationProblemOptions (acceleration_weight 2e-5, …, num_threads 4) | OPO |
| `mapping/internal/3d/local_trajectory_builder_3d_test.cc` | MakeDictionary | full LocalTrajectoryBuilderOptions3D (uses `math.rad(1.)`, `;` separators, no `intensity_cost_function_options_*`, num_range_data 45000) | LT3 |
| `mapping/internal/3d/scan_matching/ceres_scan_matcher_3d_test.cc` | MakeDictionary | `occupied_space_weight_0`, `intensity_cost_function_options_0{weight, huber_scale=55, intensity_threshold=100}`, … | CS3 |
| `mapping/internal/3d/scan_matching/fast_correlative_scan_matcher_3d_test.cc` | MakeDictionary | built by **C++ string concatenation**: `"branch_and_bound_depth = " + std::to_string(d) + ", " ...`; also an RD3 dict | FC3, RD3 |
| `mapping/internal/3d/scan_matching/real_time_correlative_scan_matcher_3d_test.cc` | MakeDictionary | RTC with `math.rad(1.)` | RTC |
| `mapping/3d/range_data_inserter_3d_test.cc` | MakeDictionary | RD3 (`num_free_space_voxels = 1000`, `intensity_threshold = 100`) | RD3 |
| `mapping/internal/2d/scan_matching/ceres_scan_matcher_2d_test.cc` | MakeDictionary | CS2 + ceres_solver_options | CS2 |
| `mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d_test.cc` | MakeDictionary | `R"text(return { linear_search_window = 3., angular_search_window = 1., branch_and_bound_depth = )text" + std::to_string(d) + "}"`; plus PGI | FC2, PGI |
| `mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d_test.cc` | MakeDictionary | RTC (weights 0.); also TSI and PGI dicts | RTC, TSI, PGI |
| `mapping/internal/2d/scan_matching/tsdf_match_cost_function_2d_test.cc` | MakeDictionary | TSI (concatenated string literals) | TSI |
| `mapping/internal/2d/tsdf_range_data_inserter_2d_test.cc` | MakeDictionary | TSI | TSI |
| `mapping/internal/2d/normal_estimation_2d_test.cc` | MakeDictionary | NE2. The parameterized test builds `num_normal_samples` via `std::to_string(GetParam())` | NE2 |
| `mapping/2d/submap_2d_test.cc` | MakeDictionary | full SO2 via concatenation with `std::to_string(kNumRangeData)`; strings escaped as `\"PROBABILITY_GRID\"` in C++ | SO2 |
| `mapping/2d/range_data_inserter_2d_test.cc` | MakeDictionary | PGI | PGI |
| `mapping/internal/motion_filter_test.cc` | MakeDictionary | MF (`max_angle_radians = 2.`) | MF |
| `io/probability_grid_points_processor_test.cc` | `NonReferenceCounted` + DummyFileResolver | `pipeline = { { action = "write_probability_grid", resolution = 0.05, range_data_inserter = {...}, draw_trajectories = false, output_type = "pb", filename = "map" } } return pipeline` | `PointsProcessorPipelineBuilder::CreatePipeline` |

**C++ helpers that build Lua code strings:** only ad-hoc `std::string` concatenation inside the tests (`fast_correlative_scan_matcher_{2d,3d}_test`, `normal_estimation_2d_test`, `submap_2d_test`, `lua_parameter_dictionary_test` (`"return " + dict->ToString()`)). There is no shared builder utility. In the Rust port, `format!`-based fixtures (or `include_str!` fixture files) are enough.

### 4.4 Porting checklist (config layer)

1. A `lua-subset` crate: lexer, parser, tree-walking evaluator, `Rc<RefCell<Table>>` values, `include`/`read`/`choose`, `math.{rad,pi,huge}`.
2. A `LuaParameterDictionary` equivalent over the evaluated table: typed getters with C++ coercion rules, per-wrapper exactly-once ref counting enforced on `Drop` (or an explicit `finish()` returning `Result`), `HasKey`, array getters, and `ToString` byte-compatible with C++ (`%f`, `%q`, sorted keys).
3. 25 hand-written `create_*_options(&mut Dict) -> Result<Proto>` functions, mirroring §2 exactly (renames, f32 narrowing, the `_N` loops, the `insert_free_space` default, optional subtables, CHECKs → `Err`).
4. Golden tests: port every snippet in §4.3, plus `configuration_files_test` against copied `configuration_files/*.lua`.
5. Protos: `prost-build` + `protox` (pure Rust, no protoc) with include root = the outer `cartographer/` dir, `include_file` for module nesting, `#![allow(deprecated)]`, cloud protos and tonic behind a `grpc` feature.
