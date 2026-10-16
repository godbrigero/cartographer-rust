//! Port of `cartographer/mapping/internal/3d/local_slam_result_3d` (.h, .cc).

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use crate::common::time::{Time, from_universal};
use crate::mapping::internal::local_slam_result_data::LocalSlamResultData;
use crate::mapping::internal::submap_controller::SubmapController;
use crate::mapping::internal::three_d::pose_graph_3d::PoseGraph3D;
use crate::mapping::proto;
use crate::mapping::submaps::SharedSubmap;
use crate::mapping::three_d::submap_3d::Submap3D;
use crate::mapping::trajectory_builder_interface::TrajectoryBuilderInterface;
use crate::mapping::trajectory_node;
use crate::sensor::data::Data;

/// A serialized 3D local SLAM result (node plus the submaps it was inserted
/// into), e.g. received from a remote local SLAM, that is added to a
/// [`PoseGraph3D`].
///
/// RUST PORT NOTE: the C++ holds a raw, non-owning
/// `SubmapController<Submap3D>*`; here the controller is shared as
/// `Arc<Mutex<..>>`.
#[derive(Clone)]
pub struct LocalSlamResult3D {
    sensor_id: String,
    time: Time,
    local_slam_result_data: proto::LocalSlamResultData,
    submap_controller: Arc<Mutex<SubmapController<Submap3D>>>,
}

impl std::fmt::Debug for LocalSlamResult3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalSlamResult3D")
            .field("sensor_id", &self.sensor_id)
            .field("time", &self.time)
            .finish_non_exhaustive()
    }
}

impl LocalSlamResult3D {
    /// Creates the packet; its time is `local_slam_result_data.timestamp`.
    pub fn new(
        sensor_id: &str,
        local_slam_result_data: proto::LocalSlamResultData,
        submap_controller: Arc<Mutex<SubmapController<Submap3D>>>,
    ) -> Self {
        Self {
            sensor_id: sensor_id.to_string(),
            time: from_universal(local_slam_result_data.timestamp),
            local_slam_result_data,
            submap_controller,
        }
    }
}

impl Data for LocalSlamResult3D {
    fn get_time(&self) -> Time {
        self.time
    }

    fn get_sensor_id(&self) -> &str {
        &self.sensor_id
    }

    fn add_to_trajectory_builder(
        self: Box<Self>,
        trajectory_builder: &mut dyn TrajectoryBuilderInterface,
    ) {
        trajectory_builder.add_local_slam_result_data(self);
    }
}

impl LocalSlamResultData for LocalSlamResult3D {
    /// RUST PORT NOTE: unlike 2D, the C++ does not skip submaps refused by
    /// the `SubmapController` (it pushes a null `shared_ptr`, which crashes
    /// `PoseGraph3D::AddNode`); here a refused submap panics with a message.
    fn add_to_pose_graph(&self, trajectory_id: i32, pose_graph: &dyn Any) {
        let pose_graph = pose_graph
            .downcast_ref::<PoseGraph3D>()
            .expect("Check failed: dynamic_cast<PoseGraph3D*>(pose_graph)");
        let submap_protos = &self.local_slam_result_data.submaps;
        assert!(
            !submap_protos.is_empty(),
            "Check failed: local_slam_result_data_.submaps().size() >= 1 ({} vs. 1)",
            submap_protos.len()
        );
        assert!(
            submap_protos[0].submap_3d.is_some(),
            "Check failed: local_slam_result_data_.submaps(0).has_submap_3d()"
        );
        let submaps: Vec<SharedSubmap<Submap3D>> = {
            let mut submap_controller = self
                .submap_controller
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            submap_protos
                .iter()
                .map(|submap_proto| {
                    submap_controller
                        .update_submap(submap_proto)
                        .expect("SubmapController refused a partially filled 3D submap")
                })
                .collect()
        };
        pose_graph.add_node(
            Arc::new(trajectory_node::from_proto(
                &self
                    .local_slam_result_data
                    .node_data
                    .clone()
                    .unwrap_or_default(),
            )),
            trajectory_id,
            &submaps,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::thread_pool::{ThreadPool, ThreadPoolInterface};
    use crate::mapping::internal::optimization::optimization_problem_3d::OptimizationProblem3D;
    use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;
    use crate::mapping::pose_graph::create_pose_graph_options;
    use crate::mapping::pose_graph_interface::PoseGraphInterface;
    use crate::mapping::submaps::Submap;
    use crate::mapping::three_d::submap_3d::{ActiveSubmaps3D, create_submaps_options_3d};
    use crate::mapping::trajectory_node::TrajectoryNodeData;
    use crate::sensor::point_cloud::PointCloud;
    use crate::sensor::range_data::RangeData;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::Rigid3d;
    use nalgebra::{DVector, Quaternion, Vector3};

    /// Field order: the pose graph is dropped before the thread pool.
    struct Fixture {
        pose_graph: PoseGraph3D,
        _thread_pool: Arc<ThreadPool>,
    }

    fn fixture() -> Fixture {
        let thread_pool = Arc::new(ThreadPool::new(1));
        let options = create_pose_graph_options(&resolve_lua_parameters(
            "include 'pose_graph.lua' return POSE_GRAPH",
        ));
        let pose_graph = PoseGraph3D::new(
            &options,
            Box::new(OptimizationProblem3D::new(
                &options.optimization_problem_options.unwrap_or_default(),
            )),
            Arc::clone(&thread_pool) as Arc<dyn ThreadPoolInterface>,
        );
        Fixture {
            pose_graph,
            _thread_pool: thread_pool,
        }
    }

    fn point_cloud() -> PointCloud {
        PointCloud::from_points(
            (0..100)
                .map(|i| {
                    let angle = f64::from(i) * 0.0628;
                    RangefinderPoint {
                        position: Vector3::new(
                            (3. * angle.cos()) as f32,
                            (3. * angle.sin()) as f32,
                            (0.01 * f64::from(i)) as f32,
                        ),
                    }
                })
                .collect(),
        )
    }

    /// A local SLAM result with one submap holding `num_range_data` range
    /// data.
    fn local_slam_result_proto(num_range_data: i32) -> proto::LocalSlamResultData {
        let mut active_submaps =
            ActiveSubmaps3D::new(&create_submaps_options_3d(&resolve_lua_parameters(
                "include 'trajectory_builder_3d.lua' return TRAJECTORY_BUILDER_3D.submaps",
            )));
        let histogram = DVector::from_element(120, 0.0f32);
        let submaps = active_submaps.insert_data(
            &RangeData {
                origin: Vector3::zeros(),
                returns: point_cloud(),
                misses: PointCloud::default(),
            },
            &Quaternion::identity(),
            &histogram,
        );
        let mut submap_proto = submaps[0].read().unwrap().to_proto(true);
        submap_proto.submap_id = Some(proto::SubmapId {
            trajectory_id: 0,
            submap_index: 0,
        });
        submap_proto.submap_3d.as_mut().unwrap().num_range_data = num_range_data;
        let node_data = TrajectoryNodeData {
            time: from_universal(1000),
            gravity_alignment: Quaternion::identity(),
            filtered_gravity_aligned_point_cloud: PointCloud::default(),
            high_resolution_point_cloud: point_cloud(),
            low_resolution_point_cloud: point_cloud(),
            rotational_scan_matcher_histogram: histogram,
            local_pose: Rigid3d::identity(),
        };
        proto::LocalSlamResultData {
            timestamp: 1000,
            node_data: Some(trajectory_node::to_proto(&node_data)),
            submaps: vec![submap_proto],
        }
    }

    #[test]
    fn adds_node_to_pose_graph() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result = LocalSlamResult3D::new("remote", local_slam_result_proto(1), controller);
        assert_eq!(result.get_time(), from_universal(1000));
        assert_eq!(result.get_sensor_id(), "remote");
        result.add_to_pose_graph(0, &fixture.pose_graph);
        // No final optimization: 3D optimization without IMU data would
        // (correctly) CHECK-fail.
        assert_eq!(fixture.pose_graph.get_trajectory_nodes().size(), 1);
        assert_eq!(fixture.pose_graph.get_all_submap_data().size(), 1);
    }

    #[test]
    #[should_panic(expected = "refused a partially filled 3D submap")]
    fn panics_on_refused_submap() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result = LocalSlamResult3D::new("remote", local_slam_result_proto(2), controller);
        result.add_to_pose_graph(0, &fixture.pose_graph);
    }

    #[test]
    #[should_panic(expected = "dynamic_cast<PoseGraph3D*>")]
    fn panics_on_wrong_pose_graph_type() {
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result = LocalSlamResult3D::new("remote", local_slam_result_proto(1), controller);
        result.add_to_pose_graph(0, &42_i32);
    }
}
