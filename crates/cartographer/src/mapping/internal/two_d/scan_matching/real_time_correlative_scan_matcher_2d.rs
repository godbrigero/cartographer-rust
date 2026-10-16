//! Port of `cartographer/mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d` (.h, .cc).
//!
//! This is an implementation of the algorithm described in "Real-Time
//! Correlative Scan Matching" by Olson.
//!
//! The correlative scan matching algorithm is exhaustively evaluating the
//! scan matching search space: the matcher performs an exhaustive search over all translations and
//! rotations in the search window and scores every candidate with the grid
//! (mean probability, or normalized TSD for TSDFs), damped by
//! `exp(-(|t| * w_t + |theta| * w_r)^2)`.

use nalgebra::{Vector2, Vector3};

use crate::common::math::pow2;
use crate::mapping::internal::two_d::scan_matching::correlative_scan_matcher_2d::{
    Candidate2D, DiscreteScan2D, SearchParameters, discretize_scans, generate_rotated_scans,
};
use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::mapping::scan_matching::proto;
use crate::mapping::two_d::grid_2d::{Grid2DTrait, GridType};
use crate::mapping::two_d::probability_grid::ProbabilityGrid;
use crate::mapping::two_d::xy_index::Array2i;
use crate::sensor::point_cloud::{PointCloud, transform_point_cloud};
use crate::transform::rigid_transform::{AngleAxis, Rigid2d, Rigid3f, Rotation2D};

fn compute_candidate_score_tsdf(
    tsdf: &TSDF2D,
    discrete_scan: &DiscreteScan2D,
    x_index_offset: i32,
    y_index_offset: i32,
) -> f32 {
    let max_correspondence_cost = tsdf.get_max_correspondence_cost();
    let mut candidate_score = 0.0_f32;
    let mut summed_weight = 0.0_f32;
    for xy_index in discrete_scan {
        let proposed_xy_index =
            Array2i::new(xy_index.x + x_index_offset, xy_index.y + y_index_offset);
        let (tsd, weight) = tsdf.get_tsd_and_weight(&proposed_xy_index);
        let normalized_tsd_score = (max_correspondence_cost - tsd.abs()) / max_correspondence_cost;
        candidate_score += normalized_tsd_score * weight;
        summed_weight += weight;
    }
    if summed_weight == 0.0 {
        return 0.0;
    }
    candidate_score /= summed_weight;
    assert!(
        candidate_score >= 0.0,
        "Check failed: candidate_score >= 0.f ({candidate_score} vs. 0)"
    );
    candidate_score
}

fn compute_candidate_score_probability(
    probability_grid: &ProbabilityGrid,
    discrete_scan: &DiscreteScan2D,
    x_index_offset: i32,
    y_index_offset: i32,
) -> f32 {
    let mut candidate_score = 0.0_f32;
    for xy_index in discrete_scan {
        let proposed_xy_index =
            Array2i::new(xy_index.x + x_index_offset, xy_index.y + y_index_offset);
        candidate_score += probability_grid.get_probability(&proposed_xy_index);
    }
    candidate_score /= discrete_scan.len() as f32;
    assert!(
        candidate_score > 0.0,
        "Check failed: candidate_score > 0.f ({candidate_score} vs. 0)"
    );
    candidate_score
}

/// An implementation of "Real-Time Correlative Scan Matching" by Olson.
#[derive(Clone, Debug)]
pub struct RealTimeCorrelativeScanMatcher2D {
    options: proto::RealTimeCorrelativeScanMatcherOptions,
}

impl RealTimeCorrelativeScanMatcher2D {
    /// Creates a matcher with the given search window and delta cost weights.
    pub fn new(options: &proto::RealTimeCorrelativeScanMatcherOptions) -> Self {
        Self { options: *options }
    }

    /// Aligns `point_cloud` within the `grid` given an
    /// `initial_pose_estimate` and returns `(score, pose_estimate)`.
    ///
    /// RUST PORT NOTE: C++ writes the pose to an out-parameter and returns
    /// the score.
    pub fn match_(
        &self,
        initial_pose_estimate: &Rigid2d,
        point_cloud: &PointCloud,
        grid: &dyn Grid2DTrait,
    ) -> (f64, Rigid2d) {
        let initial_rotation = initial_pose_estimate.rotation();
        let rotated_point_cloud = transform_point_cloud(
            point_cloud,
            &Rigid3f::from_angle_axis(AngleAxis::new(
                initial_rotation.angle() as f32,
                Vector3::z(),
            )),
        );
        let search_parameters = SearchParameters::new(
            self.options.linear_search_window,
            self.options.angular_search_window,
            &rotated_point_cloud,
            grid.limits().resolution(),
        );

        let rotated_scans = generate_rotated_scans(&rotated_point_cloud, &search_parameters);
        let initial_translation = initial_pose_estimate.translation();
        let discrete_scans = discretize_scans(
            grid.limits(),
            &rotated_scans,
            &Vector2::new(initial_translation.x as f32, initial_translation.y as f32),
        );
        let mut candidates = self.generate_exhaustive_search_candidates(&search_parameters);
        self.score_candidates(grid, &discrete_scans, &search_parameters, &mut candidates);

        // `std::max_element`: the first maximum wins on ties.
        let mut best_candidate = &candidates[0];
        for candidate in &candidates[1..] {
            if best_candidate.score < candidate.score {
                best_candidate = candidate;
            }
        }
        let pose_estimate = Rigid2d::from_parts(
            Vector2::new(
                initial_translation.x + best_candidate.x,
                initial_translation.y + best_candidate.y,
            ),
            initial_rotation * Rotation2D::new(best_candidate.orientation),
        );
        (f64::from(best_candidate.score), pose_estimate)
    }

    /// Computes the score for each candidate in a collection. The cost is
    /// computed as the sum of probabilities or normalized TSD values,
    /// different from the Ceres cost functions.
    ///
    /// Visible for testing.
    pub fn score_candidates(
        &self,
        grid: &dyn Grid2DTrait,
        discrete_scans: &[DiscreteScan2D],
        _search_parameters: &SearchParameters,
        candidates: &mut [Candidate2D],
    ) {
        // RUST PORT NOTE: the grid type is dispatched once, not per
        // candidate.
        match grid.get_grid_type() {
            GridType::ProbabilityGrid => {
                let probability_grid = grid
                    .downcast_ref::<ProbabilityGrid>()
                    .expect("PROBABILITY_GRID must be a ProbabilityGrid");
                for candidate in candidates.iter_mut() {
                    candidate.score = compute_candidate_score_probability(
                        probability_grid,
                        &discrete_scans[candidate.scan_index as usize],
                        candidate.x_index_offset,
                        candidate.y_index_offset,
                    );
                    self.apply_delta_cost(candidate);
                }
            }
            GridType::Tsdf => {
                let tsdf = grid
                    .downcast_ref::<TSDF2D>()
                    .expect("TSDF must be a TSDF2D");
                for candidate in candidates.iter_mut() {
                    candidate.score = compute_candidate_score_tsdf(
                        tsdf,
                        &discrete_scans[candidate.scan_index as usize],
                        candidate.x_index_offset,
                        candidate.y_index_offset,
                    );
                    self.apply_delta_cost(candidate);
                }
            }
        }
    }

    fn apply_delta_cost(&self, candidate: &mut Candidate2D) {
        // `float *= double` is computed in double, like C++.
        let factor = (-pow2(
            candidate.x.hypot(candidate.y) * self.options.translation_delta_cost_weight
                + candidate.orientation.abs() * self.options.rotation_delta_cost_weight,
        ))
        .exp();
        candidate.score = (f64::from(candidate.score) * factor) as f32;
    }

    fn generate_exhaustive_search_candidates(
        &self,
        search_parameters: &SearchParameters,
    ) -> Vec<Candidate2D> {
        let num_candidates: usize = search_parameters
            .linear_bounds
            .iter()
            .map(|b| ((b.max_x - b.min_x + 1) * (b.max_y - b.min_y + 1)) as usize)
            .sum();
        let mut candidates = Vec::with_capacity(num_candidates);
        for scan_index in 0..search_parameters.num_scans {
            let bounds = &search_parameters.linear_bounds[scan_index as usize];
            for x_index_offset in bounds.min_x..=bounds.max_x {
                for y_index_offset in bounds.min_y..=bounds.max_y {
                    candidates.push(Candidate2D::new(
                        scan_index,
                        x_index_offset,
                        y_index_offset,
                        search_parameters,
                    ));
                }
            }
        }
        assert_eq!(
            candidates.len(),
            num_candidates,
            "Check failed: candidates.size() == num_candidates"
        );
        candidates
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::{Vector2, Vector3};

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::mapping::internal::scan_matching::real_time_correlative_scan_matcher::create_real_time_correlative_scan_matcher_options;
    use crate::mapping::internal::two_d::tsdf_range_data_inserter_2d::{
        TSDFRangeDataInserter2D, create_tsdf_range_data_inserter_options_2d,
    };
    use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::probability_grid_range_data_inserter_2d::{
        ProbabilityGridRangeDataInserter2D, create_probability_grid_range_data_inserter_options_2d,
    };
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::range_data::RangeData;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    fn create_real_time_correlative_scan_matcher_test_options_2d()
    -> proto::RealTimeCorrelativeScanMatcherOptions {
        let parameter_dictionary = make_dictionary(
            "return {\
             linear_search_window = 0.6, \
             angular_search_window = 0.16, \
             translation_delta_cost_weight = 0., \
             rotation_delta_cost_weight = 0., \
             }",
        );
        create_real_time_correlative_scan_matcher_options(&parameter_dictionary)
    }

    /// C++ fixture `RealTimeCorrelativeScanMatcherTest`.
    struct Fixture {
        grid: Box<dyn Grid2DTrait>,
        point_cloud: PointCloud,
        real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher2D,
    }

    fn point_cloud() -> PointCloud {
        let p = |x: f32, y: f32| RangefinderPoint {
            position: Vector3::new(x, y, 0.0),
        };
        PointCloud::from_points(vec![
            p(0.025, 0.175),
            p(-0.025, 0.175),
            p(-0.075, 0.175),
            p(-0.125, 0.175),
            p(-0.125, 0.125),
            p(-0.125, 0.075),
            p(-0.125, 0.025),
        ])
    }

    impl Fixture {
        fn set_up_tsdf() -> Self {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut grid: Box<dyn Grid2DTrait> = Box::new(TSDF2D::new(
                MapLimits::new(0.05, Vector2::new(0.3, 0.5), CellLimits::new(20, 20)),
                0.3,
                1.0,
                &conversion_tables,
            ));
            let parameter_dictionary = make_dictionary(
                r"
      return {
        truncation_distance = 0.3,
        maximum_weight = 10.,
        update_free_space = false,
        normal_estimation_options = {
          num_normal_samples = 4,
          sample_radius = 0.5,
        },
        project_sdf_distance_to_scan_normal = true,
        update_weight_range_exponent = 0,
        update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0.5,
        update_weight_distance_cell_to_hit_kernel_bandwidth = 0.5,
      }",
            );
            let range_data_inserter = TSDFRangeDataInserter2D::new(
                &create_tsdf_range_data_inserter_options_2d(&parameter_dictionary),
            );
            let point_cloud = point_cloud();
            range_data_inserter.insert(
                &RangeData {
                    origin: Vector3::new(0.5, -0.5, 0.0),
                    returns: point_cloud.clone(),
                    misses: PointCloud::new(),
                },
                grid.as_mut(),
            );
            grid.finish_update();
            Self::new(grid, point_cloud)
        }

        fn set_up_probability_grid() -> Self {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut grid: Box<dyn Grid2DTrait> = Box::new(ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(0.05, 0.25), CellLimits::new(6, 6)),
                &conversion_tables,
            ));
            let parameter_dictionary = make_dictionary(
                "return { \
                 insert_free_space = true, \
                 hit_probability = 0.7, \
                 miss_probability = 0.4, \
                 }",
            );
            let range_data_inserter = ProbabilityGridRangeDataInserter2D::new(
                &create_probability_grid_range_data_inserter_options_2d(&parameter_dictionary),
            );
            let point_cloud = point_cloud();
            range_data_inserter.insert(
                &RangeData {
                    origin: Vector3::zeros(),
                    returns: point_cloud.clone(),
                    misses: PointCloud::new(),
                },
                grid.as_mut(),
            );
            grid.finish_update();
            Self::new(grid, point_cloud)
        }

        fn new(grid: Box<dyn Grid2DTrait>, point_cloud: PointCloud) -> Self {
            Self {
                grid,
                point_cloud,
                real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher2D::new(
                    &create_real_time_correlative_scan_matcher_test_options_2d(),
                ),
            }
        }

        fn score_single_candidate(&self, x_index_offset: i32, y_index_offset: i32) -> Candidate2D {
            let search_parameters = SearchParameters::new_for_testing(0, 0, 0.0, 0.0);
            let scans = generate_rotated_scans(&self.point_cloud, &search_parameters);
            let discrete_scans = discretize_scans(self.grid.limits(), &scans, &Vector2::zeros());
            let mut candidates = vec![Candidate2D::new(
                0,
                x_index_offset,
                y_index_offset,
                &search_parameters,
            )];
            self.real_time_correlative_scan_matcher.score_candidates(
                self.grid.as_ref(),
                &discrete_scans,
                &search_parameters,
                &mut candidates,
            );
            candidates[0]
        }
    }

    mod real_time_correlative_scan_matcher_test {
        use super::*;

        #[test]
        fn score_perfect_high_resolution_candidate_probability_grid() {
            let fixture = Fixture::set_up_probability_grid();
            let candidate = fixture.score_single_candidate(0, 0);
            assert_eq!(0, candidate.scan_index);
            assert_eq!(0, candidate.x_index_offset);
            assert_eq!(0, candidate.y_index_offset);
            // Every point should align perfectly.
            assert!((0.7 - candidate.score).abs() <= 1e-2, "{}", candidate.score);
        }

        #[test]
        fn score_perfect_high_resolution_candidate_tsdf() {
            let fixture = Fixture::set_up_tsdf();
            let candidate = fixture.score_single_candidate(0, 0);
            assert_eq!(0, candidate.scan_index);
            assert_eq!(0, candidate.x_index_offset);
            assert_eq!(0, candidate.y_index_offset);
            // Every point should align perfectly.
            assert!((1.0 - candidate.score).abs() <= 1e-1, "{}", candidate.score);
            assert!(0.95 < candidate.score, "{}", candidate.score);
        }

        #[test]
        fn score_partially_correct_high_resolution_candidate_probability_grid() {
            let fixture = Fixture::set_up_probability_grid();
            let candidate = fixture.score_single_candidate(0, 1);
            assert_eq!(0, candidate.scan_index);
            assert_eq!(0, candidate.x_index_offset);
            assert_eq!(1, candidate.y_index_offset);
            // 3 points should align perfectly.
            let score = f64::from(candidate.score);
            assert!(0.7 * 3.0 / 7.0 < score, "{score}");
            assert!(0.7 > score, "{score}");
        }

        #[test]
        fn score_partially_correct_high_resolution_candidate_tsdf() {
            let fixture = Fixture::set_up_tsdf();
            let candidate = fixture.score_single_candidate(0, 1);
            assert_eq!(0, candidate.scan_index);
            assert_eq!(0, candidate.x_index_offset);
            assert_eq!(1, candidate.y_index_offset);
            // 3 points should align perfectly.
            let score = f64::from(candidate.score);
            assert!(1.0 - 4.0 / (7.0 * 6.0) < score, "{score}");
            assert!(1.0 > score, "{score}");
        }

        // Not in C++: a full match recovers a small offset of the scan.
        #[test]
        fn match_recovers_offset() {
            let fixture = Fixture::set_up_probability_grid();
            let initial = Rigid2d::new(Vector2::new(0.05, 0.0), 0.0);
            let (score, pose) = fixture.real_time_correlative_scan_matcher.match_(
                &initial,
                &fixture.point_cloud,
                fixture.grid.as_ref(),
            );
            assert!((score - 0.7).abs() <= 1e-2, "{score}");
            assert!(pose.translation().norm() < 1e-9, "{pose}");
            assert!(pose.rotation().angle().abs() < 1e-9, "{pose}");
        }
    }
}
