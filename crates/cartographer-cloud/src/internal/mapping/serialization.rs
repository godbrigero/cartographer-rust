//! Port of `cartographer/cloud/internal/mapping/serialization` (.h, .cc).

use std::collections::BTreeMap;

use cartographer::mapping::pose_graph_interface::TrajectoryState;

use crate::proto;

/// Converts a `TrajectoryState` to its cloud proto enum.
pub fn trajectory_state_to_proto(trajectory_state: TrajectoryState) -> proto::TrajectoryState {
    match trajectory_state {
        TrajectoryState::Active => proto::TrajectoryState::Active,
        TrajectoryState::Finished => proto::TrajectoryState::Finished,
        TrajectoryState::Frozen => proto::TrajectoryState::Frozen,
        TrajectoryState::Deleted => proto::TrajectoryState::Deleted,
    }
}

/// Converts a cloud proto enum value to a `TrajectoryState`. Panics on an
/// unknown value (C++ `LOG(FATAL) << "Unknown proto::TrajectoryState"`).
pub fn trajectory_state_from_proto(proto: i32) -> TrajectoryState {
    match proto::TrajectoryState::try_from(proto) {
        Ok(proto::TrajectoryState::Active) => TrajectoryState::Active,
        Ok(proto::TrajectoryState::Finished) => TrajectoryState::Finished,
        Ok(proto::TrajectoryState::Frozen) => TrajectoryState::Frozen,
        Ok(proto::TrajectoryState::Deleted) => TrajectoryState::Deleted,
        Err(_) => panic!("Unknown proto::TrajectoryState"),
    }
}

/// Converts a trajectory remapping (serialized id -> new id) to its proto.
pub fn trajectory_remapping_to_proto(
    trajectory_remapping: &BTreeMap<i32, i32>,
) -> proto::TrajectoryRemapping {
    proto::TrajectoryRemapping {
        serialized_trajectories_to_trajectories: trajectory_remapping.clone(),
    }
}

/// Converts a trajectory remapping proto to a map.
pub fn trajectory_remapping_from_proto(proto: &proto::TrajectoryRemapping) -> BTreeMap<i32, i32> {
    proto.serialized_trajectories_to_trajectories.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_state_round_trip() {
        for (state, value) in [
            (TrajectoryState::Active, 0),
            (TrajectoryState::Finished, 1),
            (TrajectoryState::Frozen, 2),
            (TrajectoryState::Deleted, 3),
        ] {
            let proto = trajectory_state_to_proto(state);
            assert_eq!(proto as i32, value);
            assert_eq!(trajectory_state_from_proto(proto as i32), state);
        }
    }

    #[test]
    #[should_panic(expected = "Unknown proto::TrajectoryState")]
    fn unknown_trajectory_state_panics() {
        let _ = trajectory_state_from_proto(9);
    }

    #[test]
    fn trajectory_remapping_round_trip() {
        let remapping: BTreeMap<i32, i32> = [(0, 3), (5, 4)].into_iter().collect();
        let proto = trajectory_remapping_to_proto(&remapping);
        assert_eq!(proto.serialized_trajectories_to_trajectories.len(), 2);
        assert_eq!(trajectory_remapping_from_proto(&proto), remapping);
    }
}
