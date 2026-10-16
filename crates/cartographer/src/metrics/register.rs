//! Port of `cartographer/metrics/register` (.h, .cc).

use super::family_factory::FamilyFactory;

/// Registers the metrics of all Cartographer components with `registry`.
///
/// PENDING(mapping): the C++ version calls, in this order,
/// `mapping::constraints::ConstraintBuilder2D::RegisterMetrics(registry)`,
/// `mapping::constraints::ConstraintBuilder3D::RegisterMetrics(registry)`,
/// `mapping::GlobalTrajectoryBuilderRegisterMetrics(registry)`,
/// `mapping::LocalTrajectoryBuilder2D::RegisterMetrics(registry)`,
/// `mapping::LocalTrajectoryBuilder3D::RegisterMetrics(registry)`,
/// `mapping::PoseGraph2D::RegisterMetrics(registry)`,
/// `mapping::PoseGraph3D::RegisterMetrics(registry)` and
/// `sensor::TrajectoryCollator::RegisterMetrics(registry)`. The mapping calls
/// must be added here (before the sensor one) once those modules are ported.
pub fn register_all_metrics(registry: &mut dyn FamilyFactory) {
    crate::sensor::internal::trajectory_collator::TrajectoryCollator::register_metrics(registry);
}
