//! Port of `cartographer/mapping/internal/3d/scan_matching/rotational_scan_matcher` (.h, .cc).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use nalgebra::{DVector, Vector3};

use crate::common::math::{atan2, clamp};
use crate::common::port::round_to_int;
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::rangefinder_point::RangefinderPoint;

const MIN_DISTANCE: f32 = 0.2;
const MAX_DISTANCE: f32 = 0.9;
const SLICE_HEIGHT: f32 = 0.2;

const PI_F32: f32 = PI as f32;

fn add_value_to_histogram(mut angle: f32, value: f32, histogram: &mut DVector<f32>) {
    // Map the angle to [0, pi), i.e. a vector and its inverse are considered
    // to represent the same angle.
    while angle > PI_F32 {
        angle -= PI_F32;
    }
    while angle < 0.0 {
        angle += PI_F32;
    }
    let zero_to_one = angle / PI_F32;
    let size = histogram.len() as i32;
    // C++: `histogram->size() * zero_to_one - 0.5f` (long * float -> float).
    let bucket = clamp(
        round_to_int(histogram.len() as f32 * zero_to_one - 0.5),
        0,
        size - 1,
    );
    histogram[bucket as usize] += value;
}

fn compute_centroid(slice: &[RangefinderPoint]) -> Vector3<f32> {
    assert!(!slice.is_empty(), "Check failed: !slice.empty()");
    let mut sum = Vector3::<f32>::zeros();
    for point in slice {
        sum += point.position;
    }
    sum / slice.len() as f32
}

fn add_point_cloud_slice_to_histogram(slice: &[RangefinderPoint], histogram: &mut DVector<f32>) {
    if slice.is_empty() {
        return;
    }
    // We compute the angle of the ray from a point to the centroid of the
    // whole point cloud. If it is orthogonal to the angle we compute between
    // points, we will add the angle between points to the histogram with the
    // maximum weight. This is to reject, e.g., the angles observed on the
    // ceiling and floor.
    //
    // Note: the centroid is recomputed here on the already filtered slice, and
    // 'last_point_position' only advances after a gap larger than
    // MAX_DISTANCE (both as in the C++).
    let centroid = compute_centroid(slice);
    let mut last_point_position = slice[0].position;
    for point in slice {
        let delta = (point.position - last_point_position).xy();
        let direction = (point.position - centroid).xy();
        let distance = delta.norm();
        if distance < MIN_DISTANCE || direction.norm() < MIN_DISTANCE {
            continue;
        }
        if distance > MAX_DISTANCE {
            last_point_position = point.position;
            continue;
        }
        let angle = atan2(&delta);
        let value = 0.0f32.max(1.0 - delta.normalize().dot(&direction.normalize()).abs());
        add_value_to_histogram(angle, value, histogram);
    }
}

/// Sorts the points in each slice by angle around the centroid. This is
/// because the returns from different rangefinders are interleaved in the
/// data. Points closer than `MIN_DISTANCE` (in xy) to the centroid are
/// dropped.
// RUST PORT NOTE: C++ uses the unstable `std::sort`; we use a stable sort.
// The order can only differ for points with exactly equal angles.
fn sort_slice(slice: &[RangefinderPoint]) -> Vec<RangefinderPoint> {
    let centroid = compute_centroid(slice);
    let mut by_angle: Vec<(f32, RangefinderPoint)> = Vec::with_capacity(slice.len());
    for point in slice {
        let delta = (point.position - centroid).xy();
        if delta.norm() < MIN_DISTANCE {
            continue;
        }
        by_angle.push((atan2(&delta), *point));
    }
    by_angle.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    by_angle.into_iter().map(|(_, point)| point).collect()
}

fn dot(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn match_histograms(submap_histogram: &DVector<f32>, scan_histogram: &DVector<f32>) -> f32 {
    // We compute the dot product of normalized histograms as a measure of
    // similarity.
    // RUST PORT NOTE: Eigen may use a vectorized (reordered) reduction for
    // `norm()`/`dot()`; we sum sequentially. Differences are float rounding.
    let scan_histogram_norm = dot(scan_histogram, scan_histogram).sqrt();
    let submap_histogram_norm = dot(submap_histogram, submap_histogram).sqrt();
    let normalization = scan_histogram_norm * submap_histogram_norm;
    if normalization < 1e-3 {
        return 1.0;
    }
    dot(submap_histogram, scan_histogram) / normalization
}

/// Scores rotations of a scan histogram against a reference (submap)
/// histogram.
///
/// RUST PORT NOTE: the C++ stores a raw `const Eigen::VectorXf*` that must
/// outlive the matcher; here it is a borrow with lifetime `'a`.
#[derive(Debug, Clone, Copy)]
pub struct RotationalScanMatcher<'a> {
    histogram: &'a DVector<f32>,
}

impl<'a> RotationalScanMatcher<'a> {
    /// Creates a matcher against `histogram`, which must outlive it.
    pub fn new(histogram: &'a DVector<f32>) -> Self {
        Self { histogram }
    }

    /// Rotates the given `histogram` by the given `angle`. This might lead to
    /// rotations of a fractional bucket which is handled by linearly
    /// interpolating. A positive angle moves content to higher bucket
    /// indices; `histogram.len()` buckets correspond to pi.
    pub fn rotate_histogram(histogram: &DVector<f32>, angle: f32) -> DVector<f32> {
        if histogram.is_empty() {
            return histogram.clone();
        }
        let n = histogram.len();
        // C++: `-angle * histogram.size() / M_PI` is (float * long) -> float,
        // then float / double -> double, stored as float.
        let rotate_by_buckets = (f64::from(-angle * n as f32) / PI) as f32;
        let mut full_buckets = round_to_int(rotate_by_buckets - 0.5);
        let fraction = rotate_by_buckets - full_buckets as f32;
        while full_buckets < 0 {
            full_buckets += n as i32;
        }
        let full_buckets = full_buckets as usize;
        DVector::from_fn(n, |i, _| {
            let rotated_0 = histogram[(i + full_buckets) % n];
            let rotated_1 = histogram[(i + 1 + full_buckets) % n];
            fraction * rotated_1 + (1.0 - fraction) * rotated_0
        })
    }

    /// Computes the histogram for a gravity aligned `point_cloud`.
    pub fn compute_histogram(point_cloud: &PointCloud, histogram_size: i32) -> DVector<f32> {
        let mut histogram = DVector::<f32>::zeros(histogram_size as usize);
        let mut slices: BTreeMap<i32, Vec<RangefinderPoint>> = BTreeMap::new();
        for point in point_cloud {
            slices
                .entry(round_to_int(point.position.z / SLICE_HEIGHT))
                .or_default()
                .push(*point);
        }
        for slice in slices.values() {
            add_point_cloud_slice_to_histogram(&sort_slice(slice), &mut histogram);
        }
        histogram
    }

    /// Scores how well `histogram` rotated by `initial_angle` can be
    /// understood as further rotated by certain `angles` relative to the
    /// reference histogram. Returns one score in [0, 1] per angle
    /// (1 if either histogram is (nearly) empty).
    pub fn match_(&self, histogram: &DVector<f32>, initial_angle: f32, angles: &[f32]) -> Vec<f32> {
        angles
            .iter()
            .map(|&angle| {
                let scan_histogram = Self::rotate_histogram(histogram, initial_angle + angle);
                match_histograms(self.histogram, &scan_histogram)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    mod rotational_scan_matcher_3d_test {
        use nalgebra::DVector;

        use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::*;

        fn near(a: f32, b: f32, eps: f32) {
            assert!((a - b).abs() <= eps, "expected {a} near {b} (eps {eps})");
        }

        fn unit(n: usize, i: usize) -> DVector<f32> {
            let mut v = DVector::zeros(n);
            v[i] = 1.0;
            v
        }

        #[test]
        fn only_same_histogram_is_score_one() {
            let histogram = DVector::from_vec(vec![1.0f32, 43.0, 0.5, 0.3123, 23.0, 42.0, 0.0]);
            let matcher = RotationalScanMatcher::new(&histogram);
            let scores = matcher.match_(&histogram, 0.0, &[0.0, 1.0]);
            assert_eq!(2, scores.len());
            near(1.0, scores[0], 1e-6);
            assert!(1.0 > scores[1]);
        }

        #[test]
        fn interpolates_as_expected() {
            const NUM_BUCKETS: usize = 10;
            // C++ `constexpr float kAnglePerBucket = M_PI / kNumBuckets;`
            const ANGLE_PER_BUCKET: f32 = (std::f64::consts::PI / NUM_BUCKETS as f64) as f32;
            const NO_INITIAL_ROTATION: f32 = 0.0;
            let histogram_no_initial_rotation = unit(NUM_BUCKETS, 3);
            let matcher = RotationalScanMatcher::new(&histogram_no_initial_rotation);
            let mut t = 0.0f32;
            while t < 1.0 {
                // 't' is the fraction of overlap and we have to divide by the
                // norm of the histogram to get the expected score.
                let expected_score = t / t.hypot(1.0 - t);
                // We rotate the 't'-th fraction of a bucket into the
                // matcher's histogram.
                let scores = matcher.match_(
                    &unit(NUM_BUCKETS, 2),
                    NO_INITIAL_ROTATION,
                    &[t * ANGLE_PER_BUCKET],
                );
                assert_eq!(1, scores.len());
                near(expected_score, scores[0], 1e-6);
                // Also verify rotating out of a bucket.
                let scores = matcher.match_(
                    &unit(NUM_BUCKETS, 2),
                    NO_INITIAL_ROTATION,
                    &[(2.0 - t) * ANGLE_PER_BUCKET],
                );
                assert_eq!(1, scores.len());
                near(expected_score, scores[0], 1e-6);
                // And into and out of a bucket with negative angle.
                let scores = matcher.match_(
                    &unit(NUM_BUCKETS, 4),
                    NO_INITIAL_ROTATION,
                    &[-t * ANGLE_PER_BUCKET, (t - 2.0) * ANGLE_PER_BUCKET],
                );
                assert_eq!(2, scores.len());
                near(expected_score, scores[0], 1e-6);
                near(expected_score, scores[1], 1e-6);
                t += 0.1;
            }
        }
    }

    /// Additional tests (no C++ counterpart).
    mod extra {
        use std::f32::consts::PI;

        use nalgebra::{DVector, Vector3};

        use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::*;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        fn point(x: f32, y: f32, z: f32) -> RangefinderPoint {
            RangefinderPoint {
                position: Vector3::new(x, y, z),
            }
        }

        /// An irregular pentagonal room sampled every 0.3 m along the walls,
        /// in several 0.2 m slices.
        fn room_cloud(yaw: f32) -> PointCloud {
            let corners = [
                (4.0f32, 0.5f32),
                (1.0, 3.5),
                (-3.0, 2.0),
                (-2.5, -2.5),
                (2.0, -3.0),
            ];
            let (s, c) = yaw.sin_cos();
            let mut points = Vec::new();
            for k in 0..5 {
                let z = -0.4 + 0.2 * k as f32;
                for i in 0..corners.len() {
                    let (x0, y0) = corners[i];
                    let (x1, y1) = corners[(i + 1) % corners.len()];
                    let length = (x1 - x0).hypot(y1 - y0);
                    let num = (length / 0.3) as usize;
                    for j in 0..num {
                        let f = j as f32 / num as f32;
                        let x = x0 + f * (x1 - x0);
                        let y = y0 + f * (y1 - y0);
                        points.push(point(c * x - s * y, s * x + c * y, z));
                    }
                }
            }
            PointCloud::from_points(points)
        }

        fn cosine(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
            a.dot(b) / (a.norm() * b.norm())
        }

        #[test]
        fn hand_computed_slice_keeps_reference_point() {
            // Sorted by angle around the centroid (0, 0): (0,-.5), (.5,0),
            // (0,.5), (-.5,0). The reference point only advances after a gap
            // > 0.9 m, so (.5,0) and (-.5,0) each contribute
            // 1 - cos(45 deg) at angle pi/4 -> bucket round(8 * 0.25 - 0.5) = 2.
            let cloud = PointCloud::from_points(vec![
                point(0.5, 0.0, 0.0),
                point(0.0, 0.5, 0.0),
                point(-0.5, 0.0, 0.0),
                point(0.0, -0.5, 0.0),
                // Dropped by `sort_slice` (too close to the centroid).
                point(0.0, 0.0, 0.0),
            ]);
            let histogram = RotationalScanMatcher::compute_histogram(&cloud, 8);
            let expected_value = 2.0 * (1.0 - std::f32::consts::FRAC_1_SQRT_2);
            for i in 0..8 {
                let expected = if i == 2 { expected_value } else { 0.0 };
                assert!(
                    (histogram[i] - expected).abs() < 1e-5,
                    "bucket {i}: {} vs {expected}",
                    histogram[i]
                );
            }
        }

        #[test]
        fn empty_histograms_match_with_score_one() {
            let empty = DVector::<f32>::zeros(10);
            let matcher = RotationalScanMatcher::new(&empty);
            let scores = matcher.match_(&DVector::from_element(10, 1.0), 0.3, &[0.0, 0.5]);
            assert_eq!(scores, vec![1.0, 1.0]);
            assert_eq!(
                RotationalScanMatcher::rotate_histogram(&DVector::zeros(0), 1.0).len(),
                0
            );
        }

        #[test]
        fn rotate_histogram_has_period_pi() {
            let h = DVector::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let rotated = RotationalScanMatcher::rotate_histogram(&h, PI);
            for i in 0..6 {
                assert!((rotated[i] - h[i]).abs() < 1e-4);
            }
            // One bucket (pi / 6) shifts content to the next index.
            let rotated = RotationalScanMatcher::rotate_histogram(&h, PI / 6.0);
            for i in 0..6 {
                assert!((rotated[(i + 1) % 6] - h[i]).abs() < 1e-4);
            }
        }

        #[test]
        fn rotating_the_cloud_shifts_the_histogram() {
            const SIZE: i32 = 120;
            let h0 = RotationalScanMatcher::compute_histogram(&room_cloud(0.0), SIZE);
            assert!(h0.norm() > 1.0);
            // Rotate by an exact number of buckets and by a fraction.
            for &yaw in &[20.0 * PI / SIZE as f32, 0.37, -0.81] {
                let h1 = RotationalScanMatcher::compute_histogram(&room_cloud(yaw), SIZE);
                let predicted = RotationalScanMatcher::rotate_histogram(&h0, yaw);
                let similarity = cosine(&predicted, &h1);
                assert!(similarity > 0.9, "yaw {yaw}: similarity {similarity}");
                // The unrotated histogram is a poor fit.
                assert!(cosine(&h0, &h1) < similarity);
            }
        }

        #[test]
        fn match_scores_highest_at_true_angle() {
            const SIZE: i32 = 120;
            let yaw = 0.42f32;
            let scan = RotationalScanMatcher::compute_histogram(&room_cloud(0.0), SIZE);
            let submap = RotationalScanMatcher::compute_histogram(&room_cloud(yaw), SIZE);
            let matcher = RotationalScanMatcher::new(&submap);
            let angles: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.01).collect();
            let initial_angle = 0.1f32;
            let scores = matcher.match_(&scan, initial_angle, &angles);
            let (best, best_score) = scores.iter().enumerate().fold(
                (0, f32::MIN),
                |acc, (i, &s)| if s > acc.1 { (i, s) } else { acc },
            );
            let best_angle = initial_angle + angles[best];
            assert!(
                (best_angle - yaw).abs() <= 0.03,
                "best angle {best_angle} vs {yaw}"
            );
            assert!(best_score > 0.9, "best score {best_score}");
            for &s in &scores {
                assert!((0.0..=1.0 + 1e-6).contains(&s));
            }
        }
    }
}
