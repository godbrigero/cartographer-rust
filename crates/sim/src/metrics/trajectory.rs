//! Trajectory metrics: association, ATE and RPE.

use nalgebra::{Isometry2, Isometry3, Translation3, UnitQuaternion, Vector3};

use super::align::{SimilarityTransform, align_2d, align_3d};
use crate::trajectory::TrajectoryState;

/// A time-stamped pose (`world_from_body`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedPose {
    /// Time (s).
    pub time: f64,
    /// Pose.
    pub pose: Isometry3<f64>,
}

impl TimedPose {
    /// Creates a timed pose.
    pub fn new(time: f64, pose: Isometry3<f64>) -> Self {
        Self { time, pose }
    }

    /// Lifts a planar pose `(x, y, yaw)` to 3D (z = 0).
    pub fn from_2d(time: f64, pose: &Isometry2<f64>) -> Self {
        Self {
            time,
            pose: Isometry3::from_parts(
                Translation3::new(pose.translation.x, pose.translation.y, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, pose.rotation.angle()),
            ),
        }
    }

    /// Converts ground-truth states.
    pub fn from_states(states: &[TrajectoryState]) -> Vec<Self> {
        states.iter().map(|s| Self::new(s.time, s.pose)).collect()
    }
}

/// Linear/spherical interpolation between two poses (`f` in `[0, 1]`).
pub fn interpolate_pose(a: &Isometry3<f64>, b: &Isometry3<f64>, f: f64) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(a.translation.vector.lerp(&b.translation.vector, f)),
        a.rotation.slerp(&b.rotation, f),
    )
}

/// How estimated poses are matched to reference poses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Association {
    /// Pair each estimate with the reference sample nearest in time if
    /// `|dt| <= max_dt`.
    Nearest {
        /// Maximum time difference (s).
        max_dt: f64,
    },
    /// Interpolate the reference at each estimate's time between the two
    /// bracketing samples, if they are at most `max_gap` apart. An exact
    /// time match always succeeds.
    Interpolate {
        /// Maximum gap between bracketing reference samples (s).
        max_gap: f64,
    },
}

/// A matched pair of reference (ground truth) and estimated poses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosePair {
    /// Estimate time (s).
    pub time: f64,
    /// Reference pose.
    pub reference: Isometry3<f64>,
    /// Estimated pose.
    pub estimate: Isometry3<f64>,
}

/// Associates estimates with reference poses. Both inputs must be sorted by
/// time; estimates that cannot be associated are skipped.
pub fn associate(
    reference: &[TimedPose],
    estimate: &[TimedPose],
    mode: Association,
) -> Vec<PosePair> {
    let mut out = Vec::with_capacity(estimate.len());
    if reference.is_empty() {
        return out;
    }
    for e in estimate {
        // First reference index with time > e.time.
        let hi = reference.partition_point(|r| r.time <= e.time);
        let reference_pose = match mode {
            Association::Nearest { max_dt } => {
                let candidates = [hi.checked_sub(1), (hi < reference.len()).then_some(hi)];
                candidates
                    .into_iter()
                    .flatten()
                    .map(|i| &reference[i])
                    .min_by(|a, b| (a.time - e.time).abs().total_cmp(&(b.time - e.time).abs()))
                    .filter(|r| (r.time - e.time).abs() <= max_dt)
                    .map(|r| r.pose)
            }
            Association::Interpolate { max_gap } => {
                if hi == 0 {
                    None
                } else if reference[hi - 1].time == e.time {
                    Some(reference[hi - 1].pose)
                } else if hi < reference.len() {
                    let (a, b) = (&reference[hi - 1], &reference[hi]);
                    (b.time - a.time <= max_gap).then(|| {
                        interpolate_pose(&a.pose, &b.pose, (e.time - a.time) / (b.time - a.time))
                    })
                } else {
                    None
                }
            }
        };
        if let Some(r) = reference_pose {
            out.push(PosePair {
                time: e.time,
                reference: r,
                estimate: e.pose,
            });
        }
    }
    out
}

/// Summary statistics of a list of errors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorStats {
    /// Number of samples.
    pub count: usize,
    /// Root mean square.
    pub rmse: f64,
    /// Mean.
    pub mean: f64,
    /// Median.
    pub median: f64,
    /// Standard deviation (population).
    pub std: f64,
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
}

impl ErrorStats {
    /// Computes statistics (all zero for an empty slice).
    pub fn from_errors(errors: &[f64]) -> Self {
        let n = errors.len();
        if n == 0 {
            return Self::default();
        }
        let nf = n as f64;
        let mean = errors.iter().sum::<f64>() / nf;
        let ms = errors.iter().map(|e| e * e).sum::<f64>() / nf;
        let mut sorted = errors.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        };
        Self {
            count: n,
            rmse: ms.sqrt(),
            mean,
            median,
            std: (ms - mean * mean).max(0.0).sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

/// Alignment applied to the estimate before computing ATE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alignment {
    /// No alignment (estimate assumed to be in the reference frame).
    None,
    /// Rotation about z + x-y translation (for 2D SLAM).
    Rigid2d,
    /// Full 6-DoF rigid alignment.
    Rigid3d,
    /// 6-DoF rigid plus uniform scale.
    Similarity3d,
}

/// Absolute trajectory error.
#[derive(Clone, Debug, PartialEq)]
pub struct AteResult {
    /// Statistics of translation errors (m) after alignment.
    pub translation: ErrorStats,
    /// Statistics of rotation errors (rad) after alignment.
    pub rotation: ErrorStats,
    /// Transform applied to estimates (`reference_from_estimate`).
    pub alignment: SimilarityTransform,
    /// Per-pair translation errors (m), in input order.
    pub translation_errors: Vec<f64>,
}

/// Computes the ATE of associated pairs after the requested alignment
/// (estimated from positions only, as in the TUM/evo convention).
pub fn ate(pairs: &[PosePair], alignment: Alignment) -> AteResult {
    let src: Vec<Vector3<f64>> = pairs
        .iter()
        .map(|p| p.estimate.translation.vector)
        .collect();
    let dst: Vec<Vector3<f64>> = pairs
        .iter()
        .map(|p| p.reference.translation.vector)
        .collect();
    let align = match alignment {
        Alignment::None => SimilarityTransform::identity(),
        Alignment::Rigid2d => align_2d(&src, &dst, false),
        Alignment::Rigid3d => align_3d(&src, &dst, false),
        Alignment::Similarity3d => align_3d(&src, &dst, true),
    };
    let mut t_err = Vec::with_capacity(pairs.len());
    let mut r_err = Vec::with_capacity(pairs.len());
    for p in pairs {
        let e = align.transform_pose(&p.estimate);
        t_err.push((e.translation.vector - p.reference.translation.vector).norm());
        r_err.push(e.rotation.angle_to(&p.reference.rotation));
    }
    AteResult {
        translation: ErrorStats::from_errors(&t_err),
        rotation: ErrorStats::from_errors(&r_err),
        alignment: align,
        translation_errors: t_err,
    }
}

/// Spacing between the two poses of an RPE pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RpeDelta {
    /// Second pose is the first one at least `dt` seconds later.
    Time(f64),
    /// Second pose is the first one at least this far along the reference
    /// path (m).
    Distance(f64),
    /// Second pose is `n` pairs later.
    Frames(usize),
}

/// Relative pose error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RpeResult {
    /// Statistics of relative translation errors (m).
    pub translation: ErrorStats,
    /// Statistics of relative rotation errors (rad).
    pub rotation: ErrorStats,
}

/// Computes the RPE: for each pair `i` and its partner `j` chosen by
/// `delta`, the error is `E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j)` (`Q` reference,
/// `P` estimate); translation error `|E.t|`, rotation error `angle(E.R)`.
/// Pairs must be sorted by time.
pub fn rpe(pairs: &[PosePair], delta: RpeDelta) -> RpeResult {
    let mut cum = Vec::with_capacity(pairs.len());
    let mut acc = 0.0;
    for (k, p) in pairs.iter().enumerate() {
        if k > 0 {
            acc +=
                (p.reference.translation.vector - pairs[k - 1].reference.translation.vector).norm();
        }
        cum.push(acc);
    }
    let mut t_err = Vec::new();
    let mut r_err = Vec::new();
    let mut j = 0;
    for i in 0..pairs.len() {
        let partner = match delta {
            RpeDelta::Frames(n) => (i + n < pairs.len() && n > 0).then_some(i + n),
            RpeDelta::Time(dt) => {
                j = j.max(i + 1);
                while j < pairs.len() && pairs[j].time - pairs[i].time < dt - 1e-9 {
                    j += 1;
                }
                (j < pairs.len()).then_some(j)
            }
            RpeDelta::Distance(d) => {
                j = j.max(i + 1);
                while j < pairs.len() && cum[j] - cum[i] < d - 1e-9 {
                    j += 1;
                }
                (j < pairs.len()).then_some(j)
            }
        };
        let Some(j) = partner else { continue };
        let (a, b) = (&pairs[i], &pairs[j]);
        let dq = a.reference.inverse() * b.reference;
        let dp = a.estimate.inverse() * b.estimate;
        let e = dq.inverse() * dp;
        t_err.push(e.translation.vector.norm());
        r_err.push(e.rotation.angle());
    }
    RpeResult {
        translation: ErrorStats::from_errors(&t_err),
        rotation: ErrorStats::from_errors(&r_err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Trajectory;

    fn gt() -> Vec<TimedPose> {
        TimedPose::from_states(&Trajectory::room_wander_3d(1.0, 0.8).sample(20.0))
    }

    #[test]
    fn ate_of_rigidly_offset_copy() {
        let reference = gt();
        let offset = Isometry3::from_parts(
            Translation3::new(5.0, -3.0, 0.7),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 1.3),
        );
        let est: Vec<_> = reference
            .iter()
            .map(|r| TimedPose::new(r.time, offset * r.pose))
            .collect();
        let pairs = associate(&reference, &est, Association::Nearest { max_dt: 1e-6 });
        assert_eq!(pairs.len(), reference.len());
        let a = ate(&pairs, Alignment::Rigid3d);
        assert!(a.translation.rmse < 1e-9, "{}", a.translation.rmse);
        assert!(a.rotation.max < 1e-9);
        assert!(ate(&pairs, Alignment::None).translation.rmse > 1.0);
        // Similarity alignment of a scaled copy.
        let est: Vec<_> = reference
            .iter()
            .map(|r| {
                let mut p = offset * r.pose;
                p.translation.vector *= 2.0;
                TimedPose::new(r.time, p)
            })
            .collect();
        let pairs = associate(&reference, &est, Association::Nearest { max_dt: 1e-6 });
        let a = ate(&pairs, Alignment::Similarity3d);
        assert!(a.translation.rmse < 1e-9);
        assert!((a.alignment.scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ate_pure_translation_without_alignment() {
        let reference = gt();
        let est: Vec<_> = reference
            .iter()
            .map(|r| {
                let mut p = r.pose;
                p.translation.vector += Vector3::new(0.3, 0.4, 0.0);
                TimedPose::new(r.time, p)
            })
            .collect();
        let pairs = associate(&reference, &est, Association::Interpolate { max_gap: 0.1 });
        let a = ate(&pairs, Alignment::None);
        assert!((a.translation.rmse - 0.5).abs() < 1e-12);
        assert!((a.translation.median - 0.5).abs() < 1e-12);
        assert!(a.translation.std < 1e-9);
    }

    #[test]
    fn ate_planar_alignment() {
        let traj = Trajectory::corridor_loops(1.0, 1.5);
        let reference: Vec<_> = traj
            .sample(10.0)
            .iter()
            .map(|s| TimedPose::from_2d(s.time, &s.pose_2d()))
            .collect();
        let off = Isometry2::new(nalgebra::Vector2::new(-2.0, 7.0), 0.8);
        let est: Vec<_> = traj
            .sample(10.0)
            .iter()
            .map(|s| TimedPose::from_2d(s.time, &(off * s.pose_2d())))
            .collect();
        let pairs = associate(&reference, &est, Association::Nearest { max_dt: 0.01 });
        let a = ate(&pairs, Alignment::Rigid2d);
        assert!(a.translation.rmse < 1e-9);
        assert!(a.rotation.max < 1e-9);
    }

    #[test]
    fn association_modes() {
        let reference: Vec<_> = (0..11)
            .map(|k| TimedPose::new(k as f64, Isometry3::translation(k as f64, 0.0, 0.0)))
            .collect();
        let est = vec![
            TimedPose::new(-1.0, Isometry3::identity()),
            TimedPose::new(2.25, Isometry3::identity()),
            TimedPose::new(5.0, Isometry3::identity()),
            TimedPose::new(10.5, Isometry3::identity()),
        ];
        let near = associate(&reference, &est, Association::Nearest { max_dt: 0.3 });
        let times: Vec<_> = near.iter().map(|p| p.time).collect();
        assert_eq!(times, vec![2.25, 5.0]);
        assert_eq!(near[0].reference.translation.x, 2.0);
        let interp = associate(&reference, &est, Association::Interpolate { max_gap: 1.0 });
        let times: Vec<_> = interp.iter().map(|p| p.time).collect();
        assert_eq!(times, vec![2.25, 5.0]);
        assert!((interp[0].reference.translation.x - 2.25).abs() < 1e-12);
        assert!(associate(&reference, &est, Association::Interpolate { max_gap: 0.5 }).len() == 1);
    }

    #[test]
    fn rpe_known_drift() {
        // Straight line at 1 m/s sampled at 10 Hz; estimate has a 1 % scale
        // error and a body-frame yaw drift of 0.01 rad/s.
        let reference: Vec<_> = (0..201)
            .map(|k| {
                let t = k as f64 * 0.1;
                TimedPose::new(t, Isometry3::translation(t, 0.0, 0.0))
            })
            .collect();
        let scaled: Vec<_> = reference
            .iter()
            .map(|r| TimedPose::new(r.time, Isometry3::translation(1.01 * r.time, 0.0, 0.0)))
            .collect();
        let pairs = associate(&reference, &scaled, Association::Nearest { max_dt: 1e-6 });
        let r = rpe(&pairs, RpeDelta::Distance(1.0));
        assert_eq!(r.translation.count, 191);
        assert!((r.translation.rmse - 0.01).abs() < 1e-9);
        assert!((r.translation.max - 0.01).abs() < 1e-9);
        assert!(r.rotation.max < 1e-12);
        let r = rpe(&pairs, RpeDelta::Time(2.0));
        assert!((r.translation.mean - 0.02).abs() < 1e-9);
        let r = rpe(&pairs, RpeDelta::Frames(5));
        assert!((r.translation.mean - 0.005).abs() < 1e-9);

        let yawed: Vec<_> = reference
            .iter()
            .map(|r| {
                TimedPose::new(
                    r.time,
                    r.pose * UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01 * r.time),
                )
            })
            .collect();
        let pairs = associate(&reference, &yawed, Association::Nearest { max_dt: 1e-6 });
        let r = rpe(&pairs, RpeDelta::Time(1.0));
        assert!((r.rotation.rmse - 0.01).abs() < 1e-9);
        assert!((r.rotation.min - 0.01).abs() < 1e-9);
    }

    #[test]
    fn stats() {
        let s = ErrorStats::from_errors(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.count, 4);
        assert!((s.mean - 2.5).abs() < 1e-12);
        assert!((s.median - 2.5).abs() < 1e-12);
        assert!((s.rmse - 7.5f64.sqrt()).abs() < 1e-12);
        assert!((s.std - 1.25f64.sqrt()).abs() < 1e-12);
        assert_eq!((s.min, s.max), (1.0, 4.0));
        assert_eq!(ErrorStats::from_errors(&[]).count, 0);
    }
}
