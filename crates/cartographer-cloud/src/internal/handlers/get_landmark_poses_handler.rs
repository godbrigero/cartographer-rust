//! Port of `cartographer/cloud/internal/handlers/get_landmark_poses_handler` (.h, .cc).

use cartographer::mapping::proto::pose_graph::LandmarkPose;
use cartographer::transform::transform::to_proto;
use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::proto;

/// Handles `GetLandmarkPoses`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetLandmarkPosesResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let landmark_poses = pose_graph.get_landmark_poses();
    Ok(proto::GetLandmarkPosesResponse {
        landmark_poses: landmark_poses
            .iter()
            .map(|(landmark_id, global_pose)| LandmarkPose {
                landmark_id: landmark_id.clone(),
                global_pose: Some(to_proto(global_pose)),
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use cartographer::mapping::proto::pose_graph::LandmarkPose;
    use cartographer::transform::proto as transform_proto;
    use cartographer::transform::rigid_transform::Rigid3d;
    use nalgebra::{Quaternion, Vector3};

    use crate::internal::testing::handler_test::HandlerTest;
    use crate::proto;

    fn rigid3d_proto(t: [f64; 3], q: [f64; 4]) -> transform_proto::Rigid3d {
        transform_proto::Rigid3d {
            translation: Some(transform_proto::Vector3d {
                x: t[0],
                y: t[1],
                z: t[2],
            }),
            rotation: Some(transform_proto::Quaterniond {
                w: q[0],
                x: q[1],
                y: q[2],
                z: q[3],
            }),
        }
    }

    mod get_landmark_poses_handler_test {
        use super::*;

        #[test]
        fn no_local_slam_uploader() {
            let test = HandlerTest::new();
            test.mock_pose_graph.lock().landmark_poses = [
                (
                    "landmark_1".to_string(),
                    Rigid3d::new(Vector3::new(1., 2., 3.), Quaternion::new(1., 0., 0., 0.)),
                ),
                (
                    "landmark_2".to_string(),
                    Rigid3d::new(Vector3::new(3., 2., 1.), Quaternion::new(0., 1., 0., 0.)),
                ),
            ]
            .into_iter()
            .collect();
            let response = test
                .call(|mut client| async move { client.get_landmark_poses(()).await })
                .unwrap()
                .into_inner();
            // C++ `kMessage`.
            let expected_response = proto::GetLandmarkPosesResponse {
                landmark_poses: vec![
                    LandmarkPose {
                        landmark_id: "landmark_1".to_string(),
                        global_pose: Some(rigid3d_proto([1., 2., 3.], [1., 0., 0., 0.])),
                    },
                    LandmarkPose {
                        landmark_id: "landmark_2".to_string(),
                        global_pose: Some(rigid3d_proto([3., 2., 1.], [0., 1., 0., 0.])),
                    },
                ],
            };
            assert_eq!(response, expected_response);
        }
    }
}
