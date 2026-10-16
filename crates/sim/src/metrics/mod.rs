//! Evaluation metrics: trajectory alignment, ATE, RPE, association and
//! occupancy-map accuracy. Everything here works on plain data so it can be
//! used from tests, benchmarks and report generators alike.
//!
//! Typical trajectory evaluation:
//! ```
//! use cartographer_sim::metrics::*;
//! use cartographer_sim::Trajectory;
//!
//! let traj = Trajectory::corridor_loops(1.0, 1.5);
//! let reference = TimedPose::from_states(&traj.sample(100.0));
//! // ... `estimate` would come from SLAM; here a copy of the ground truth.
//! let estimate = TimedPose::from_states(&traj.sample(10.0));
//! let pairs = associate(&reference, &estimate, Association::Interpolate { max_gap: 0.05 });
//! let a = ate(&pairs, Alignment::Rigid2d);
//! let r = rpe(&pairs, RpeDelta::Distance(1.0));
//! assert!(a.translation.rmse < 1e-6 && r.translation.rmse < 1e-6);
//! ```

mod align;
mod occupancy;
mod trajectory;

pub use align::{SimilarityTransform, align_2d, align_3d, umeyama};
pub use occupancy::{OccupancyMetrics, OccupancyMetricsConfig, occupancy_metrics};
pub use trajectory::{
    Alignment, Association, AteResult, ErrorStats, PosePair, RpeDelta, RpeResult, TimedPose,
    associate, ate, interpolate_pose, rpe,
};
