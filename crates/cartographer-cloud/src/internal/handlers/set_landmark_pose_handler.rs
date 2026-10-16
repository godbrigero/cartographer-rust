//! Port of `cartographer/cloud/internal/handlers/set_landmark_pose_handler` (.h, .cc).

use cartographer::transform::transform::to_rigid3;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `SetLandmarkPose` (`frozen` keeps its default `false`).
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
    request: &proto::SetLandmarkPoseRequest,
) -> Result<(), Status> {
    let landmark_pose = request.landmark_pose.clone().unwrap_or_default();
    let pose_graph = context.map_builder().pose_graph();
    pose_graph.set_landmark_pose(
        &landmark_pose.landmark_id,
        &to_rigid3(&landmark_pose.global_pose.unwrap_or_default()),
        false,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use cartographer::mapping::proto::pose_graph::LandmarkPose;
    use cartographer::transform::proto as transform_proto;
    use cartographer::transform::rigid_transform::Rigid3d;
    use cartographer::transform::rigid_transform_test_helpers::assert_is_nearly;
    use nalgebra::{Quaternion, Vector3};

    use crate::internal::testing::fake_map_builder::PoseGraphCall;
    use crate::internal::testing::handler_test::HandlerTest;
    use crate::proto;

    mod set_landmark_pose_handler_test {
        use super::*;

        #[test]
        fn set_landmark_pose() {
            const EPS: f64 = 1e-10;
            let test = HandlerTest::new();
            // C++ `kMessage`.
            let request = proto::SetLandmarkPoseRequest {
                landmark_pose: Some(LandmarkPose {
                    landmark_id: "landmark_1".to_string(),
                    global_pose: Some(transform_proto::Rigid3d {
                        translation: Some(transform_proto::Vector3d {
                            x: 1.,
                            y: 2.,
                            z: 3.,
                        }),
                        rotation: Some(transform_proto::Quaterniond {
                            w: 4.,
                            x: 5.,
                            y: 6.,
                            z: 7.,
                        }),
                    }),
                }),
            };
            test.call(|mut client| async move { client.set_landmark_pose(request).await })
                .unwrap();
            let calls = test.mock_pose_graph.calls();
            assert_eq!(calls.len(), 1, "{calls:?}");
            let PoseGraphCall::SetLandmarkPose(landmark_id, global_pose, frozen) = &calls[0] else {
                panic!("unexpected call {:?}", calls[0]);
            };
            assert_eq!(landmark_id, "landmark_1");
            assert!(!frozen);
            assert_is_nearly(
                global_pose,
                &Rigid3d::new(Vector3::new(1., 2., 3.), Quaternion::new(4., 5., 6., 7.)),
                EPS,
            );
        }
    }
}
