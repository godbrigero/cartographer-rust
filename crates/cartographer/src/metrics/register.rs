//! Port of `cartographer/metrics/register` (.h, .cc).

use super::family_factory::FamilyFactory;

/// Registers the metrics of all Cartographer components with `registry`.
///
/// Same calls, in the same order, as the C++ version:
/// `mapping::constraints::ConstraintBuilder2D::RegisterMetrics(registry)`,
/// `mapping::constraints::ConstraintBuilder3D::RegisterMetrics(registry)`,
/// `mapping::GlobalTrajectoryBuilderRegisterMetrics(registry)`,
/// `mapping::LocalTrajectoryBuilder2D::RegisterMetrics(registry)`,
/// `mapping::LocalTrajectoryBuilder3D::RegisterMetrics(registry)`,
/// `mapping::PoseGraph2D::RegisterMetrics(registry)`,
/// `mapping::PoseGraph3D::RegisterMetrics(registry)` and
/// `sensor::TrajectoryCollator::RegisterMetrics(registry)`.
pub fn register_all_metrics(registry: &mut dyn FamilyFactory) {
    crate::mapping::internal::constraints::constraint_builder_2d::ConstraintBuilder2D::register_metrics(registry);
    crate::mapping::internal::constraints::constraint_builder_3d::ConstraintBuilder3D::register_metrics(registry);
    crate::mapping::internal::global_trajectory_builder::global_trajectory_builder_register_metrics(
        registry,
    );
    crate::mapping::internal::two_d::local_trajectory_builder_2d::LocalTrajectoryBuilder2D::register_metrics(registry);
    crate::mapping::internal::three_d::local_trajectory_builder_3d::LocalTrajectoryBuilder3D::register_metrics(registry);
    crate::mapping::internal::two_d::pose_graph_2d::PoseGraph2D::register_metrics(registry);
    crate::mapping::internal::three_d::pose_graph_3d::PoseGraph3D::register_metrics(registry);
    crate::sensor::internal::trajectory_collator::TrajectoryCollator::register_metrics(registry);
}
