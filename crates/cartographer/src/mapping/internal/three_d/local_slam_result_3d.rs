//! Port of `cartographer/mapping/internal/3d/local_slam_result_3d` (.h, .cc).
//!
//! GAP: not ported yet because `PoseGraph3D` (which `AddToPoseGraph`
//! downcasts to) is ported concurrently in a separate branch. Once
//! `crate::mapping::internal::three_d::pose_graph_3d::PoseGraph3D` exists
//! with `add_node(&self, Arc<TrajectoryNodeData>, i32,
//! &[SharedSubmap<Submap3D>]) -> NodeId`, this file is a copy of
//! `two_d/local_slam_result_2d.rs` with `2D` -> `3D` and these C++
//! differences in `AddToPoseGraph`:
//!
//! * it checks `local_slam_result_data_.submaps(0).has_submap_3d()`;
//! * it does not skip refused submaps: the C++ pushes the (possibly null)
//!   result of `submap_controller_->UpdateSubmap(submap_proto)` for every
//!   submap and never returns early ("Ignoring submap"/"Ignoring node" exist
//!   only in 2D). A null submap would crash `PoseGraph3D::AddNode`, so the
//!   Rust port should `expect` the submap with a message instead.
//!
//! `SubmapController<Submap3D>` is already available in
//! `crate::mapping::internal::submap_controller`.
