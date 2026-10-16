//! Port of `cartographer/mapping/internal/2d/scan_matching/correlative_scan_matcher_2d` (.h, .cc).
//!
//! Shared pieces of the correlative scan matchers (real-time and fast):
//! search windows, rotated and discretized scans, and candidates.

use nalgebra::{Vector2, Vector3};

use crate::common::math::pow2;
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
use crate::sensor::point_cloud::{PointCloud, transform_point_cloud};
use crate::transform::rigid_transform::{AngleAxis, Rigid3f};

/// A scan discretized into cell indices of a grid (C++
/// `typedef std::vector<Eigen::Array2i> DiscreteScan2D`).
pub type DiscreteScan2D = Vec<Array2i>;

/// Linear search window in pixel offsets; bounds are inclusive (C++
/// `SearchParameters::LinearBounds`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearBounds {
    /// Minimum x offset (inclusive).
    pub min_x: i32,
    /// Maximum x offset (inclusive).
    pub max_x: i32,
    /// Minimum y offset (inclusive).
    pub min_y: i32,
    /// Maximum y offset (inclusive).
    pub max_y: i32,
}

/// Search window of a correlative scan matcher.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchParameters {
    /// Number of angular steps on each side of the initial orientation.
    pub num_angular_perturbations: i32,
    /// Angular step in radians.
    pub angular_perturbation_step_size: f64,
    /// Grid resolution in meters.
    pub resolution: f64,
    /// `2 * num_angular_perturbations + 1`.
    pub num_scans: i32,
    /// Per rotated scans.
    pub linear_bounds: Vec<LinearBounds>,
}

impl SearchParameters {
    /// Computes the search parameters for the given linear (meters) and
    /// angular (radians) search windows. The angular step is chosen such
    /// that the farthest point of `point_cloud` moves by at most about one
    /// `resolution` per step.
    pub fn new(
        linear_search_window: f64,
        angular_search_window: f64,
        point_cloud: &PointCloud,
        resolution: f64,
    ) -> Self {
        // We set this value to something on the order of resolution to make
        // sure that the acos() below is defined.
        let mut max_scan_range = 3.0_f32 * resolution as f32;
        for point in point_cloud.iter() {
            let range = point.position.xy().norm();
            max_scan_range = range.max(max_scan_range);
        }
        const SAFETY_MARGIN: f64 = 1.0 - 1e-3;
        // `common::Pow2(max_scan_range)` is evaluated in float, like C++.
        let angular_perturbation_step_size = SAFETY_MARGIN
            * (1.0 - pow2(resolution) / (2.0 * f64::from(pow2(max_scan_range)))).acos();
        let num_angular_perturbations =
            (angular_search_window / angular_perturbation_step_size).ceil() as i32;
        let num_scans = 2 * num_angular_perturbations + 1;
        let num_linear_perturbations = (linear_search_window / resolution).ceil() as i32;
        let linear_bounds = vec![
            LinearBounds {
                min_x: -num_linear_perturbations,
                max_x: num_linear_perturbations,
                min_y: -num_linear_perturbations,
                max_y: num_linear_perturbations,
            };
            num_scans as usize
        ];
        Self {
            num_angular_perturbations,
            angular_perturbation_step_size,
            resolution,
            num_scans,
            linear_bounds,
        }
    }

    /// For testing: explicit perturbation counts and step size.
    pub fn new_for_testing(
        num_linear_perturbations: i32,
        num_angular_perturbations: i32,
        angular_perturbation_step_size: f64,
        resolution: f64,
    ) -> Self {
        let num_scans = 2 * num_angular_perturbations + 1;
        let linear_bounds = vec![
            LinearBounds {
                min_x: -num_linear_perturbations,
                max_x: num_linear_perturbations,
                min_y: -num_linear_perturbations,
                max_y: num_linear_perturbations,
            };
            num_scans as usize
        ];
        Self {
            num_angular_perturbations,
            angular_perturbation_step_size,
            resolution,
            num_scans,
            linear_bounds,
        }
    }

    /// Tightens the search window as much as possible: every offset keeps at
    /// least one point of the scan inside `cell_limits`.
    pub fn shrink_to_fit(&mut self, scans: &[DiscreteScan2D], cell_limits: &CellLimits) {
        assert_eq!(
            scans.len(),
            self.num_scans as usize,
            "Check failed: scans.size() == num_scans"
        );
        assert_eq!(
            self.linear_bounds.len(),
            self.num_scans as usize,
            "Check failed: linear_bounds.size() == num_scans"
        );
        let max_index = Array2i::new(cell_limits.num_x_cells - 1, cell_limits.num_y_cells - 1);
        for (scan, bounds) in scans.iter().zip(self.linear_bounds.iter_mut()) {
            let mut min_bound = Array2i::zeros();
            let mut max_bound = Array2i::zeros();
            for xy_index in scan {
                min_bound = min_bound.inf(&(-xy_index));
                max_bound = max_bound.sup(&(max_index - xy_index));
            }
            bounds.min_x = bounds.min_x.max(min_bound.x);
            bounds.max_x = bounds.max_x.min(max_bound.x);
            bounds.min_y = bounds.min_y.max(min_bound.y);
            bounds.max_y = bounds.max_y.min(max_bound.y);
        }
    }
}

/// Generates `search_parameters.num_scans` rotated scans: scan `k` is
/// `point_cloud` rotated about Z by `(k - num_angular_perturbations) *
/// angular_perturbation_step_size`.
pub fn generate_rotated_scans(
    point_cloud: &PointCloud,
    search_parameters: &SearchParameters,
) -> Vec<PointCloud> {
    let mut rotated_scans = Vec::with_capacity(search_parameters.num_scans.max(0) as usize);
    // Accumulated by repeated addition in double, like C++.
    let mut delta_theta = -f64::from(search_parameters.num_angular_perturbations)
        * search_parameters.angular_perturbation_step_size;
    for _ in 0..search_parameters.num_scans {
        rotated_scans.push(transform_point_cloud(
            point_cloud,
            &Rigid3f::from_angle_axis(AngleAxis::new(delta_theta as f32, Vector3::z())),
        ));
        delta_theta += search_parameters.angular_perturbation_step_size;
    }
    rotated_scans
}

/// Translates each scan by `initial_translation` and converts its points to
/// cell indices of `map_limits`.
pub fn discretize_scans(
    map_limits: &MapLimits,
    scans: &[PointCloud],
    initial_translation: &Vector2<f32>,
) -> Vec<DiscreteScan2D> {
    scans
        .iter()
        .map(|scan| {
            scan.iter()
                .map(|point| {
                    let translated_point = point.position.xy() + initial_translation;
                    map_limits.get_cell_index(&translated_point)
                })
                .collect()
        })
        .collect()
}

/// A candidate pose of a correlative scan matcher, relative to the initial
/// pose.
///
/// RUST PORT NOTE: C++ defines `operator<` / `operator>` comparing only
/// `score`; Rust code compares `score` explicitly instead of implementing
/// `PartialOrd` with a non-structural equality.
#[derive(Clone, Copy, Debug)]
pub struct Candidate2D {
    /// Index into the rotated scans vector.
    pub scan_index: i32,
    /// Linear offset from the initial pose, in cells.
    pub x_index_offset: i32,
    /// Linear offset from the initial pose, in cells.
    pub y_index_offset: i32,
    /// Pose of this candidate relative to the initial pose (meters).
    pub x: f64,
    /// Pose of this candidate relative to the initial pose (meters).
    pub y: f64,
    /// Pose of this candidate relative to the initial pose (radians).
    pub orientation: f64,
    /// Score, higher is better.
    pub score: f32,
}

impl Candidate2D {
    /// Creates a candidate with score 0. Note the axis swap: cell x indices
    /// follow world -y and cell y indices world -x.
    pub fn new(
        scan_index: i32,
        x_index_offset: i32,
        y_index_offset: i32,
        search_parameters: &SearchParameters,
    ) -> Self {
        Self {
            scan_index,
            x_index_offset,
            y_index_offset,
            x: -f64::from(y_index_offset) * search_parameters.resolution,
            y: -f64::from(x_index_offset) * search_parameters.resolution,
            orientation: f64::from(scan_index - search_parameters.num_angular_perturbations)
                * search_parameters.angular_perturbation_step_size,
            score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    fn point(x: f32, y: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, y, 0.0),
        }
    }

    mod search_parameters {
        use super::*;

        #[test]
        fn construction() {
            let search_parameters = SearchParameters::new_for_testing(4, 5, 0.03, 0.05);
            assert_eq!(5, search_parameters.num_angular_perturbations);
            assert!((0.03 - search_parameters.angular_perturbation_step_size).abs() <= 1e-9);
            assert!((0.05 - search_parameters.resolution).abs() <= 1e-9);
            assert_eq!(11, search_parameters.num_scans);
            assert_eq!(11, search_parameters.linear_bounds.len());
            for linear_bounds in &search_parameters.linear_bounds {
                assert_eq!(-4, linear_bounds.min_x);
                assert_eq!(4, linear_bounds.max_x);
                assert_eq!(-4, linear_bounds.min_y);
                assert_eq!(4, linear_bounds.max_y);
            }
        }

        // Not in C++: checks the window computation from a point cloud.
        #[test]
        fn construction_from_point_cloud() {
            let point_cloud = PointCloud::from_points(vec![point(10.0, 0.0)]);
            let search_parameters = SearchParameters::new(0.1, 0.2, &point_cloud, 0.05);
            let expected_step = (1.0 - 1e-3) * (1.0 - 0.0025 / 200.0_f64).acos();
            assert!(
                (search_parameters.angular_perturbation_step_size - expected_step).abs() < 1e-9
            );
            assert_eq!(
                search_parameters.num_angular_perturbations,
                (0.2 / expected_step).ceil() as i32
            );
            assert_eq!(search_parameters.linear_bounds[0].max_x, 2);
        }

        // Not in C++: ShrinkToFit keeps at least one point inside the grid.
        #[test]
        fn shrink_to_fit() {
            let mut search_parameters = SearchParameters::new_for_testing(10, 0, 0.0, 1.0);
            let scans = vec![vec![Array2i::new(2, 3), Array2i::new(4, 1)]];
            search_parameters.shrink_to_fit(&scans, &CellLimits::new(6, 5));
            assert_eq!(
                search_parameters.linear_bounds[0],
                LinearBounds {
                    min_x: -4,
                    max_x: 3,
                    min_y: -3,
                    max_y: 3
                }
            );
        }
    }

    mod candidate {
        use super::*;

        #[test]
        fn construction() {
            let search_parameters = SearchParameters::new_for_testing(4, 5, 0.03, 0.05);
            let candidate = Candidate2D::new(3, 4, -5, &search_parameters);
            assert_eq!(3, candidate.scan_index);
            assert_eq!(4, candidate.x_index_offset);
            assert_eq!(-5, candidate.y_index_offset);
            assert!((0.25 - candidate.x).abs() <= 1e-9);
            assert!((-0.2 - candidate.y).abs() <= 1e-9);
            assert!((-0.06 - candidate.orientation).abs() <= 1e-9);
            assert!((0.0 - f64::from(candidate.score)).abs() <= 1e-9);

            let mut bigger_candidate = Candidate2D::new(3, 4, 5, &search_parameters);
            bigger_candidate.score = 1.0;
            assert!(candidate.score < bigger_candidate.score);
        }
    }

    mod generate_rotated_scans {
        use super::*;

        #[test]
        fn generate_rotated_scans() {
            let point_cloud = PointCloud::from_points(vec![point(-1.0, 1.0)]);
            let scans = super::super::generate_rotated_scans(
                &point_cloud,
                &SearchParameters::new_for_testing(0, 1, std::f64::consts::FRAC_PI_2, 0.0),
            );
            assert_eq!(3, scans.len());
            let near = |a: f32, b: f32| (a - b).abs() <= 1e-6;
            assert!(near(1.0, scans[0][0].position.x));
            assert!(near(1.0, scans[0][0].position.y));
            assert!(near(-1.0, scans[1][0].position.x));
            assert!(near(1.0, scans[1][0].position.y));
            assert!(near(-1.0, scans[2][0].position.x));
            assert!(near(-1.0, scans[2][0].position.y));
        }
    }

    mod discretize_scans {
        use super::*;

        #[test]
        fn discretize_scans() {
            let point_cloud = PointCloud::from_points(vec![
                point(0.025, 0.175),
                point(-0.025, 0.175),
                point(-0.075, 0.175),
                point(-0.125, 0.175),
                point(-0.125, 0.125),
                point(-0.125, 0.075),
                point(-0.125, 0.025),
            ]);
            let map_limits = MapLimits::new(0.05, Vector2::new(0.05, 0.25), CellLimits::new(6, 6));
            let scans = generate_rotated_scans(
                &point_cloud,
                &SearchParameters::new_for_testing(0, 0, 0.0, 0.0),
            );
            let discrete_scans =
                super::super::discretize_scans(&map_limits, &scans, &Vector2::zeros());
            assert_eq!(1, discrete_scans.len());
            assert_eq!(7, discrete_scans[0].len());
            assert_eq!(Array2i::new(1, 0), discrete_scans[0][0]);
            assert_eq!(Array2i::new(1, 1), discrete_scans[0][1]);
            assert_eq!(Array2i::new(1, 2), discrete_scans[0][2]);
            assert_eq!(Array2i::new(1, 3), discrete_scans[0][3]);
            assert_eq!(Array2i::new(2, 3), discrete_scans[0][4]);
            assert_eq!(Array2i::new(3, 3), discrete_scans[0][5]);
            assert_eq!(Array2i::new(4, 3), discrete_scans[0][6]);
        }
    }
}
