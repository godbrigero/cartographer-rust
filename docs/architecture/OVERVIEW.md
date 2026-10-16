# Cartographer architecture overview (for the Rust port)

Detailed per-area reports:

| Report | Scope |
|---|---|
| [common_transform_metrics.md](common_transform_metrics.md) | time, math, thread pool, Lua dictionary, rigid transforms, metrics |
| [sensor.md](sensor.md) | sensor data types, collators, ordered multi-queue, voxel filters, compressed point clouds |
| [mapping_2d.md](mapping_2d.md) | probability values, 2D grids/submaps, inserters, 2D scan matchers, local trajectory builder 2D, extrapolator |
| [mapping_3d.md](mapping_3d.md) | hybrid grid, 3D submaps, 3D scan matchers, rotational matcher, IMU-based extrapolator |
| [pose_graph.md](pose_graph.md) | pose graph 2D/3D, constraint builder, optimization problems, trimmers, map builder |
| [io_ground_truth_cloud.md](io_ground_truth_cloud.md) | points processors, pbstream, serialization/migration, submap painting, ground truth, cloud |
| [protos_and_config.md](protos_and_config.md) | every proto and every Lua option, the Lua subset |

## Module dependency graph

Arrows point from a module to the modules it depends on.

```mermaid
graph TD
  subgraph external[External replacements]
    nalgebra[nalgebra = Eigen]
    ceres[cartographer-ceres = Ceres]
    prost[prost/protox = protobuf]
    lua[common::lua = Lua 5.2 subset]
  end
  common --> lua
  common --> prost
  metrics --> common
  transform --> common
  transform --> nalgebra
  sensor --> transform
  sensor --> common
  mapping_prim[mapping primitives<br/>probability_values, id, submaps,<br/>trajectory_node, imu_tracker,<br/>pose_extrapolator, motion_filter] --> sensor
  grids2d[mapping/2d<br/>grid_2d, probability_grid, tsdf,<br/>range_data_inserter, submap_2d] --> mapping_prim
  grids3d[mapping/3d<br/>hybrid_grid, range_data_inserter_3d,<br/>submap_3d] --> mapping_prim
  local2d[internal/2d<br/>scan_matching + local_trajectory_builder_2d] --> grids2d
  local2d --> ceres
  local3d[internal/3d<br/>scan_matching + local_trajectory_builder_3d] --> grids3d
  local3d --> ceres
  optimization[internal/optimization<br/>optimization_problem_2d/3d, cost functions] --> ceres
  optimization --> mapping_prim
  constraints[internal/constraints<br/>constraint_builder_2d/3d] --> local2d
  constraints --> local3d
  posegraph[pose_graph_2d/3d, trimmers,<br/>work_queue] --> optimization
  posegraph --> constraints
  mapbuilder[map_builder, collated/global<br/>trajectory builders] --> posegraph
  mapbuilder --> local2d
  mapbuilder --> local3d
  mapbuilder --> sensor_collator[sensor::internal collator]
  io --> mapbuilder
  ground_truth --> io
  cloud --> mapbuilder
  slam[high-level cartographer::Slam API] --> mapbuilder
  slam --> io
```

## End-to-end data flow

```mermaid
sequenceDiagram
  participant U as User / ROS node
  participant MB as MapBuilder
  participant CTB as CollatedTrajectoryBuilder
  participant COL as Collator (OrderedMultiQueue)
  participant GTB as GlobalTrajectoryBuilder
  participant LTB as LocalTrajectoryBuilder2D/3D
  participant PG as PoseGraph2D/3D
  participant CB as ConstraintBuilder
  participant OP as OptimizationProblem (Ceres SPA)
  U->>MB: AddTrajectoryBuilder(sensor ids, options, callback)
  U->>CTB: AddSensorData(range / IMU / odometry / fixed frame / landmark)
  CTB->>COL: AddSensorData(trajectory_id, Dispatchable)
  COL-->>CTB: time-ordered data across all sensors
  CTB->>GTB: AddRangefinderData / AddImuData / AddOdometryData ...
  GTB->>LTB: AddRangeData (+ IMU/odometry -> PoseExtrapolator)
  Note over LTB: range data collator, motion compensation,<br/>voxel filter, gravity alignment,<br/>real-time CSM + Ceres scan match,<br/>motion filter, submap insertion
  LTB-->>GTB: MatchingResult (local pose, InsertionResult)
  GTB->>PG: AddNode(constant data, insertion submaps)
  GTB-->>U: LocalSlamResultCallback
  PG->>CB: MaybeAddConstraint / MaybeAddGlobalConstraint (thread pool)
  Note over CB: fast correlative (branch & bound)<br/>+ Ceres refinement
  CB-->>PG: constraints (when done)
  PG->>OP: Solve every optimize_every_n_nodes nodes
  OP-->>PG: optimized submap + node poses
  PG->>PG: trimmers, global-to-local transforms
  U->>MB: SerializeState -> .pbstream (io::ProtoStreamWriter)
```

Threading in C++: sensor data is handled on the caller's thread up to `AddNode`.
`PoseGraph` owns a `WorkQueue` executed on the shared `ThreadPool`
(`num_background_threads`). Constraint search tasks run on the same pool;
optimization runs on a pool thread, with the pose graph mutex released during
`Solve`.

## Port order (leaves first)

1. `cartographer-ceres` (independent; the C++ uses it from `mapping/` only) and
   `common` (time, math, Lua dictionary, thread pool, ceres options), then `transform`, then `metrics`.
2. `sensor` (data types, proto conversions, compressed point cloud, voxel filters,
   map_by_time, ordered multi-queue, collators).
3. Mapping primitives: `id`, `probability_values`, `value_conversion_tables`,
   `trajectory_node`, `imu_tracker`, `pose_extrapolator(_interface)`,
   `motion_filter`, `range_data_collator`, `submaps`, `range_data_inserter_interface`,
   `detect_floors`, `eigen_quaterniond_from_two_vectors`.
4. 2D maps: `xy_index`, `map_limits`, `grid_2d`, `probability_grid`, `tsdf_2d`, `tsd_value_converter`,
   inserters, `ray_to_pixel_mask`, `normal_estimation_2d`, `submap_2d`.
   3D maps (in parallel): `hybrid_grid`, `range_data_inserter_3d`, `submap_3d`.
5. 2D local SLAM (scan matchers, `local_trajectory_builder_2d`). 3D local SLAM in
   parallel (scan matchers, rotational scan matcher, `imu_integration`,
   `imu_based_pose_extrapolator`, `local_trajectory_builder_3d`).
6. Optimization (cost functions, `optimization_problem_2d/3d`) and constraints
   (`constraint_builder_2d/3d`), then `pose_graph_2d/3d`, `pose_graph_trimmer`,
   `overlapping_submaps_trimmer_2d`, `connected_components`,
   `trajectory_connectivity_state`, `work_queue`, `pose_graph_data`.
7. `map_builder`, `collated_trajectory_builder`, `global_trajectory_builder`,
   `local_slam_result_2d/3d`, `submap_controller`.
8. `io` (proto streams, serialization/migration, points processors, painting),
   then `ground_truth`.
9. The high-level `cartographer::Slam` API, end-to-end tests, benchmarks and the report.
10. `cloud` (optional `tonic` feature).
