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
    /// RUST PORT NOTE: the C++ pushes the result of
    /// `submap_controller_->UpdateSubmap()` for every submap, including null
    /// pointers for refused (partially filled, previously unknown) submaps,
    /// which `PoseGraph3D::AddNode` would then dereference. Like the 2D
    /// version, refused submaps are dropped here with a logged note, and the
    /// node is ignored if no submap is left.
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
        let mut submaps: Vec<SharedSubmap<Submap3D>> = Vec::new();
        {
            let mut submap_controller = self
                .submap_controller
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for submap_proto in submap_protos {
                match submap_controller.update_submap(submap_proto) {
                    Some(submap_ptr) => submaps.push(submap_ptr),
                    None => log::info!("Ignoring refused (null) submap"),
                }
            }
        }
        if submaps.is_empty() {
            log::info!("Ignoring node");
            return;
        }
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
    use crate::mapping::internal::testing::test_helpers::{
        create_fake_node, create_fake_submap_3d, resolve_lua_parameters,
    };
    use crate::mapping::pose_graph::create_pose_graph_options;
    use crate::mapping::pose_graph_interface::PoseGraphInterface;
    use crate::transform::rigid_transform::Rigid3d;
    use crate::transform::transform::to_proto;

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

    /// A local SLAM result with the given submaps (as `(num_range_data,
    /// submap_index)`) for trajectory 0.
    fn local_slam_result_proto(submaps: &[(i32, i32)]) -> proto::LocalSlamResultData {
        let mut node = create_fake_node(0, 0);
        let node_data = node.node_data.as_mut().unwrap();
        node_data.timestamp = 1000;
        node_data.gravity_alignment = Some(to_proto(Rigid3d::identity().rotation()));
        proto::LocalSlamResultData {
            timestamp: 1000,
            node_data: node.node_data,
            submaps: submaps
                .iter()
                .map(|&(num_range_data, submap_index)| {
                    let mut submap = create_fake_submap_3d(0, submap_index, false);
                    submap.submap_3d.as_mut().unwrap().num_range_data = num_range_data;
                    submap
                })
                .collect(),
        }
    }

    #[test]
    fn adds_node_to_pose_graph() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result =
            LocalSlamResult3D::new("remote", local_slam_result_proto(&[(1, 0)]), controller);
        assert_eq!(result.get_time(), from_universal(1000));
        assert_eq!(result.get_sensor_id(), "remote");
        result.add_to_pose_graph(0, &fixture.pose_graph);
        fixture.pose_graph.wait_for_all_computations();
        assert_eq!(fixture.pose_graph.get_trajectory_nodes().size(), 1);
        assert_eq!(fixture.pose_graph.get_all_submap_data().size(), 1);
    }

    #[test]
    fn drops_refused_submaps() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        // The second submap is new and partially filled: it is refused (null
        // in C++) and dropped, the node is added with the first one only.
        let result = LocalSlamResult3D::new(
            "remote",
            local_slam_result_proto(&[(1, 0), (2, 1)]),
            controller,
        );
        result.add_to_pose_graph(0, &fixture.pose_graph);
        fixture.pose_graph.wait_for_all_computations();
        assert_eq!(fixture.pose_graph.get_trajectory_nodes().size(), 1);
        assert_eq!(fixture.pose_graph.get_all_submap_data().size(), 1);
    }

    #[test]
    fn ignores_node_without_usable_submaps() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result =
            LocalSlamResult3D::new("remote", local_slam_result_proto(&[(2, 0)]), controller);
        result.add_to_pose_graph(0, &fixture.pose_graph);
        fixture.pose_graph.wait_for_all_computations();
        assert_eq!(fixture.pose_graph.get_trajectory_nodes().size(), 0);
    }

    #[test]
    #[should_panic(expected = "has_submap_3d()")]
    fn panics_on_2d_submap() {
        let fixture = fixture();
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let mut proto = local_slam_result_proto(&[(1, 0)]);
        proto.submaps[0].submap_3d = None;
        proto.submaps[0].submap_2d = Some(proto::Submap2D::default());
        LocalSlamResult3D::new("remote", proto, controller)
            .add_to_pose_graph(0, &fixture.pose_graph);
    }

    #[test]
    #[should_panic(expected = "dynamic_cast<PoseGraph3D*>")]
    fn panics_on_wrong_pose_graph_type() {
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result =
            LocalSlamResult3D::new("remote", local_slam_result_proto(&[(1, 0)]), controller);
        result.add_to_pose_graph(0, &42_i32);
    }

    #[test]
    fn add_to_trajectory_builder_forwards_packet() {
        use crate::mapping::internal::testing::mock_trajectory_builder::MockTrajectoryBuilder;
        let controller = Arc::new(Mutex::new(SubmapController::new()));
        let result =
            LocalSlamResult3D::new("remote", local_slam_result_proto(&[(1, 0)]), controller);
        let mut builder = MockTrajectoryBuilder::new();
        Box::new(result).add_to_trajectory_builder(&mut builder);
        assert_eq!(builder.recorded_calls().len(), 1);
    }
}
