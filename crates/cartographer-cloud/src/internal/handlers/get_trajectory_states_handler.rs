//! Port of `cartographer/cloud/internal/handlers/get_trajectory_states_handler` (.h, .cc).

use tonic::Status;

use crate::internal::map_builder_context_interface::MapBuilderContextInterface;
use crate::internal::mapping::serialization::trajectory_state_to_proto;
use crate::proto;

/// Handles `GetTrajectoryStates`.
pub fn on_request(
    context: &dyn MapBuilderContextInterface,
) -> Result<proto::GetTrajectoryStatesResponse, Status> {
    let pose_graph = context.map_builder().pose_graph();
    let trajectories_state = pose_graph.get_trajectory_states();
    Ok(proto::GetTrajectoryStatesResponse {
        trajectories_state: trajectories_state
            .into_iter()
            .map(|(trajectory_id, state)| (trajectory_id, trajectory_state_to_proto(state) as i32))
            .collect(),
    })
}
