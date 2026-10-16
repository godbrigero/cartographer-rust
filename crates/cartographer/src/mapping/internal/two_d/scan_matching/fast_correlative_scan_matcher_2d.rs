//! Port of `cartographer/mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d` (.h, .cc).
//!
//! This is an implementation of the algorithm described in "Real-Time
//! Correlative Scan Matching" by Olson.
//!
//! It is similar to the RealTimeCorrelativeScanMatcher but has a different
//! trade-off: Scan matching is faster because more effort is put into the
//! precomputation done for a given map. However, this map is immutable after
//! construction.

use std::cmp::Ordering;
use std::collections::VecDeque;

use nalgebra::{Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::port::round_to_int;
use crate::mapping::internal::two_d::scan_matching::correlative_scan_matcher_2d::{
    Candidate2D, DiscreteScan2D, SearchParameters, discretize_scans, generate_rotated_scans,
};
use crate::mapping::scan_matching::proto;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
use crate::sensor::point_cloud::{PointCloud, transform_point_cloud};
use crate::transform::rigid_transform::{AngleAxis, Rigid2d, Rigid3f, Rotation2D};

/// A collection of values which can be added and later removed, and the
/// maximum of the current values in the collection can be retrieved.
/// All of it in (amortized) O(1).
#[derive(Debug, Default)]
struct SlidingWindowMaximum {
    // Maximum of the current sliding window at the front. Then the maximum of
    // the remaining window that came after this values first occurrence, and
    // so on.
    non_ascending_maxima: VecDeque<f32>,
}

impl SlidingWindowMaximum {
    fn add_value(&mut self, value: f32) {
        while self
            .non_ascending_maxima
            .back()
            .is_some_and(|&back| value > back)
        {
            self.non_ascending_maxima.pop_back();
        }
        self.non_ascending_maxima.push_back(value);
    }

    fn remove_value(&mut self, value: f32) {
        // debug_assert for performance, since this is done for every value in
        // the precomputation grid.
        debug_assert!(!self.non_ascending_maxima.is_empty());
        debug_assert!(value <= self.non_ascending_maxima[0]);
        if self.non_ascending_maxima.front() == Some(&value) {
            self.non_ascending_maxima.pop_front();
        }
    }

    fn get_maximum(&self) -> f32 {
        // debug_assert for performance, since this is done for every value in
        // the precomputation grid.
        debug_assert!(!self.non_ascending_maxima.is_empty());
        self.non_ascending_maxima[0]
    }

    fn check_is_empty(&self) {
        assert_eq!(
            self.non_ascending_maxima.len(),
            0,
            "Check failed: non_ascending_maxima_.size() == 0"
        );
    }

    fn clear(&mut self) {
        self.non_ascending_maxima.clear();
    }
}

/// Reads `FastCorrelativeScanMatcherOptions2D` from `parameter_dictionary`.
pub fn create_fast_correlative_scan_matcher_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::FastCorrelativeScanMatcherOptions2D {
    proto::FastCorrelativeScanMatcherOptions2D {
        linear_search_window: parameter_dictionary.get_double("linear_search_window"),
        angular_search_window: parameter_dictionary.get_double("angular_search_window"),
        branch_and_bound_depth: parameter_dictionary.get_int("branch_and_bound_depth"),
    }
}

/// A precomputed grid that contains in each cell (x0, y0) the maximum
/// probability in the width x width area defined by x0 <= x < x0 + width and
/// y0 <= y < y0 + width.
#[derive(Clone, Debug)]
pub struct PrecomputationGrid2D {
    // Offset of the precomputation grid in relation to the 'grid' including
    // the additional 'width' - 1 cells.
    offset: Array2i,
    // Size of the precomputation grid.
    wide_limits: CellLimits,
    min_score: f32,
    max_score: f32,
    // Probabilites mapped to 0 to 255.
    cells: Vec<u8>,
}

impl PrecomputationGrid2D {
    /// Precomputes the `width` x `width` sliding window maxima of
    /// `1 - |correspondence cost|` of `grid` (restricted to `limits`).
    /// `reusable_intermediate_grid` is scratch space shared between grids.
    pub fn new(
        grid: &dyn Grid2DTrait,
        limits: &CellLimits,
        width: i32,
        reusable_intermediate_grid: &mut Vec<f32>,
    ) -> Self {
        assert!(width >= 1, "Check failed: width >= 1 ({width} vs. 1)");
        assert!(
            limits.num_x_cells >= 1,
            "Check failed: limits.num_x_cells >= 1 ({} vs. 1)",
            limits.num_x_cells
        );
        assert!(
            limits.num_y_cells >= 1,
            "Check failed: limits.num_y_cells >= 1 ({} vs. 1)",
            limits.num_y_cells
        );
        let wide_limits = CellLimits::new(
            limits.num_x_cells + width - 1,
            limits.num_y_cells + width - 1,
        );
        let mut result = Self {
            offset: Array2i::new(-width + 1, -width + 1),
            wide_limits,
            min_score: 1.0 - grid.get_max_correspondence_cost(),
            max_score: 1.0 - grid.get_min_correspondence_cost(),
            cells: vec![0; wide_limits.num_x_cells as usize * wide_limits.num_y_cells as usize],
        };
        let grid = grid.grid_2d();
        let num_x_cells = limits.num_x_cells;
        let num_y_cells = limits.num_y_cells;
        let stride = wide_limits.num_x_cells;
        // First we compute the maximum probability for each (x0, y) achieved
        // in the span defined by x0 <= x < x0 + width.
        let intermediate = reusable_intermediate_grid;
        intermediate.resize(stride as usize * num_y_cells as usize, 0.0);
        // RUST PORT NOTE: the C++ code recomputes `1 - |cost|` of each cell
        // when adding and when removing it; the values of one row are cached
        // here instead (identical results, `RemoveValue` compares exactly).
        let mut row = vec![0.0_f32; num_x_cells as usize];
        let mut current_values = SlidingWindowMaximum::default();
        for y in 0..num_y_cells {
            for (x, value) in row.iter_mut().enumerate() {
                *value = 1.0
                    - grid
                        .get_correspondence_cost(&Array2i::new(x as i32, y))
                        .abs();
            }
            let value = |x: i32| row[x as usize];
            let index = |x: i32| (x + width - 1 + y * stride) as usize;
            current_values.clear();
            current_values.add_value(value(0));
            for x in (-width + 1)..0 {
                intermediate[index(x)] = current_values.get_maximum();
                if x + width < num_x_cells {
                    current_values.add_value(value(x + width));
                }
            }
            for x in 0..(num_x_cells - width) {
                intermediate[index(x)] = current_values.get_maximum();
                current_values.remove_value(value(x));
                current_values.add_value(value(x + width));
            }
            for x in (num_x_cells - width).max(0)..num_x_cells {
                intermediate[index(x)] = current_values.get_maximum();
                current_values.remove_value(value(x));
            }
            current_values.check_is_empty();
        }
        // For each (x, y), we compute the maximum probability in the width x
        // width region starting at each (x, y) and precompute the resulting
        // bound on the score.
        for x in 0..wide_limits.num_x_cells {
            let value = |y: i32| intermediate[(x + y * stride) as usize];
            let index = |y: i32| (x + (y + width - 1) * stride) as usize;
            current_values.clear();
            current_values.add_value(value(0));
            for y in (-width + 1)..0 {
                result.cells[index(y)] = result.compute_cell_value(current_values.get_maximum());
                if y + width < num_y_cells {
                    current_values.add_value(value(y + width));
                }
            }
            for y in 0..(num_y_cells - width) {
                result.cells[index(y)] = result.compute_cell_value(current_values.get_maximum());
                current_values.remove_value(value(y));
                current_values.add_value(value(y + width));
            }
            for y in (num_y_cells - width).max(0)..num_y_cells {
                result.cells[index(y)] = result.compute_cell_value(current_values.get_maximum());
                current_values.remove_value(value(y));
            }
            current_values.check_is_empty();
        }
        result
    }

    /// Returns a value between 0 and 255 to represent probabilities between
    /// min_score and max_score.
    #[inline]
    pub fn get_value(&self, xy_index: &Array2i) -> i32 {
        let local_xy_index = xy_index - self.offset;
        // The `as u32` is for performance to check with 2 comparisons
        // xy_index.x() < offset_.x() || xy_index.y() < offset_.y() ||
        // local_xy_index.x() >= wide_limits_.num_x_cells ||
        // local_xy_index.y() >= wide_limits_.num_y_cells
        // instead of using 4 comparisons.
        if local_xy_index.x as u32 >= self.wide_limits.num_x_cells as u32
            || local_xy_index.y as u32 >= self.wide_limits.num_y_cells as u32
        {
            return 0;
        }
        let stride = self.wide_limits.num_x_cells;
        i32::from(self.cells[(local_xy_index.x + local_xy_index.y * stride) as usize])
    }

    /// Maps values from [0, 255] to [min_score, max_score].
    #[inline]
    pub fn to_score(&self, value: f32) -> f32 {
        self.min_score + value * ((self.max_score - self.min_score) / 255.0)
    }

    fn compute_cell_value(&self, probability: f32) -> u8 {
        let cell_value = round_to_int(
            (probability - self.min_score) * (255.0_f32 / (self.max_score - self.min_score)),
        );
        assert!(
            (0..=255).contains(&cell_value),
            "Check failed: 0 <= cell_value <= 255 ({cell_value})"
        );
        cell_value as u8
    }
}

/// The precomputation grids of widths `1, 2, 4, ..., 2^(depth - 1)`.
#[derive(Clone, Debug)]
pub struct PrecomputationGridStack2D {
    precomputation_grids: Vec<PrecomputationGrid2D>,
}

impl PrecomputationGridStack2D {
    /// Precomputes `options.branch_and_bound_depth` grids (must be >= 1).
    pub fn new(
        grid: &dyn Grid2DTrait,
        options: &proto::FastCorrelativeScanMatcherOptions2D,
    ) -> Self {
        assert!(
            options.branch_and_bound_depth >= 1,
            "Check failed: options.branch_and_bound_depth() >= 1 ({} vs. 1)",
            options.branch_and_bound_depth
        );
        let max_width = 1 << (options.branch_and_bound_depth - 1);
        let limits = *grid.limits().cell_limits();
        let mut reusable_intermediate_grid = Vec::with_capacity(
            (limits.num_x_cells + max_width - 1) as usize * limits.num_y_cells as usize,
        );
        let precomputation_grids = (0..options.branch_and_bound_depth)
            .map(|i| {
                let width = 1 << i;
                PrecomputationGrid2D::new(grid, &limits, width, &mut reusable_intermediate_grid)
            })
            .collect();
        Self {
            precomputation_grids,
        }
    }

    /// The grid of width `1 << index`.
    pub fn get(&self, index: i32) -> &PrecomputationGrid2D {
        &self.precomputation_grids[index as usize]
    }

    /// Index of the coarsest grid.
    pub fn max_depth(&self) -> i32 {
        self.precomputation_grids.len() as i32 - 1
    }
}

/// An implementation of "Real-Time Correlative Scan Matching" by Olson
/// using branch and bound over precomputed grids. Immutable after
/// construction (the grid is not retained), so it can be shared between
/// threads.
#[derive(Clone, Debug)]
pub struct FastCorrelativeScanMatcher2D {
    options: proto::FastCorrelativeScanMatcherOptions2D,
    limits: MapLimits,
    precomputation_grid_stack: PrecomputationGridStack2D,
}

impl FastCorrelativeScanMatcher2D {
    /// Precomputes the grids for matching against `grid`.
    pub fn new(
        grid: &dyn Grid2DTrait,
        options: &proto::FastCorrelativeScanMatcherOptions2D,
    ) -> Self {
        Self {
            options: *options,
            limits: *grid.limits(),
            precomputation_grid_stack: PrecomputationGridStack2D::new(grid, options),
        }
    }

    /// Aligns `point_cloud` within the grid given an
    /// `initial_pose_estimate`. If a score above `min_score` (excluding
    /// equality) is possible, returns `(score, pose_estimate)`.
    ///
    /// RUST PORT NOTE: C++ returns `bool` and writes score and pose to
    /// out-parameters.
    pub fn match_(
        &self,
        initial_pose_estimate: &Rigid2d,
        point_cloud: &PointCloud,
        min_score: f32,
    ) -> Option<(f32, Rigid2d)> {
        let search_parameters = SearchParameters::new(
            self.options.linear_search_window,
            self.options.angular_search_window,
            point_cloud,
            self.limits.resolution(),
        );
        self.match_with_search_parameters(
            search_parameters,
            initial_pose_estimate,
            point_cloud,
            min_score,
        )
    }

    /// Aligns `point_cloud` within the full grid, i.e., not restricted to the
    /// configured search window. If a score above `min_score` (excluding
    /// equality) is possible, returns `(score, pose_estimate)`.
    pub fn match_full_submap(
        &self,
        point_cloud: &PointCloud,
        min_score: f32,
    ) -> Option<(f32, Rigid2d)> {
        // Compute a search window around the center of the submap that
        // includes it fully.
        let search_parameters = SearchParameters::new(
            1e6 * self.limits.resolution(), // Linear search window, 1e6 cells/direction.
            std::f64::consts::PI, // Angular search window, 180 degrees in both directions.
            point_cloud,
            self.limits.resolution(),
        );
        let cell_limits = self.limits.cell_limits();
        let center = Rigid2d::from_translation(
            self.limits.max()
                - 0.5
                    * self.limits.resolution()
                    * Vector2::new(
                        f64::from(cell_limits.num_y_cells),
                        f64::from(cell_limits.num_x_cells),
                    ),
        );
        self.match_with_search_parameters(search_parameters, &center, point_cloud, min_score)
    }

    // The actual implementation of the scan matcher, called by match_() and
    // match_full_submap() with appropriate 'initial_pose_estimate' and
    // 'search_parameters'.
    fn match_with_search_parameters(
        &self,
        mut search_parameters: SearchParameters,
        initial_pose_estimate: &Rigid2d,
        point_cloud: &PointCloud,
        min_score: f32,
    ) -> Option<(f32, Rigid2d)> {
        let initial_rotation = initial_pose_estimate.rotation();
        let rotated_point_cloud = transform_point_cloud(
            point_cloud,
            &Rigid3f::from_angle_axis(AngleAxis::new(
                initial_rotation.angle() as f32,
                Vector3::z(),
            )),
        );
        let rotated_scans = generate_rotated_scans(&rotated_point_cloud, &search_parameters);
        let initial_translation = initial_pose_estimate.translation();
        let discrete_scans = discretize_scans(
            &self.limits,
            &rotated_scans,
            &Vector2::new(initial_translation.x as f32, initial_translation.y as f32),
        );
        search_parameters.shrink_to_fit(&discrete_scans, self.limits.cell_limits());

        let lowest_resolution_candidates =
            self.compute_lowest_resolution_candidates(&discrete_scans, &search_parameters);
        let best_candidate = self.branch_and_bound(
            &discrete_scans,
            &search_parameters,
            &lowest_resolution_candidates,
            self.precomputation_grid_stack.max_depth(),
            min_score,
        );
        if best_candidate.score > min_score {
            return Some((
                best_candidate.score,
                Rigid2d::from_parts(
                    Vector2::new(
                        initial_translation.x + best_candidate.x,
                        initial_translation.y + best_candidate.y,
                    ),
                    initial_rotation * Rotation2D::new(best_candidate.orientation),
                ),
            ));
        }
        None
    }

    fn compute_lowest_resolution_candidates(
        &self,
        discrete_scans: &[DiscreteScan2D],
        search_parameters: &SearchParameters,
    ) -> Vec<Candidate2D> {
        let mut lowest_resolution_candidates =
            self.generate_lowest_resolution_candidates(search_parameters);
        Self::score_candidates(
            self.precomputation_grid_stack
                .get(self.precomputation_grid_stack.max_depth()),
            discrete_scans,
            &mut lowest_resolution_candidates,
        );
        lowest_resolution_candidates
    }

    fn generate_lowest_resolution_candidates(
        &self,
        search_parameters: &SearchParameters,
    ) -> Vec<Candidate2D> {
        let linear_step_size = 1 << self.precomputation_grid_stack.max_depth();
        let num_candidates: i32 = search_parameters
            .linear_bounds
            .iter()
            .map(|b| {
                let num_lowest_resolution_linear_x_candidates =
                    (b.max_x - b.min_x + linear_step_size) / linear_step_size;
                let num_lowest_resolution_linear_y_candidates =
                    (b.max_y - b.min_y + linear_step_size) / linear_step_size;
                num_lowest_resolution_linear_x_candidates
                    * num_lowest_resolution_linear_y_candidates
            })
            .sum();
        let mut candidates = Vec::with_capacity(num_candidates.max(0) as usize);
        for scan_index in 0..search_parameters.num_scans {
            let bounds = &search_parameters.linear_bounds[scan_index as usize];
            let mut x_index_offset = bounds.min_x;
            while x_index_offset <= bounds.max_x {
                let mut y_index_offset = bounds.min_y;
                while y_index_offset <= bounds.max_y {
                    candidates.push(Candidate2D::new(
                        scan_index,
                        x_index_offset,
                        y_index_offset,
                        search_parameters,
                    ));
                    y_index_offset += linear_step_size;
                }
                x_index_offset += linear_step_size;
            }
        }
        assert_eq!(
            candidates.len() as i32,
            num_candidates,
            "Check failed: candidates.size() == num_candidates"
        );
        candidates
    }

    fn score_candidates(
        precomputation_grid: &PrecomputationGrid2D,
        discrete_scans: &[DiscreteScan2D],
        candidates: &mut [Candidate2D],
    ) {
        for candidate in candidates.iter_mut() {
            let discrete_scan = &discrete_scans[candidate.scan_index as usize];
            let mut sum = 0_i32;
            for xy_index in discrete_scan {
                let proposed_xy_index = Array2i::new(
                    xy_index.x + candidate.x_index_offset,
                    xy_index.y + candidate.y_index_offset,
                );
                sum += precomputation_grid.get_value(&proposed_xy_index);
            }
            candidate.score = precomputation_grid.to_score(sum as f32 / discrete_scan.len() as f32);
        }
        // RUST PORT NOTE: C++ uses the unstable `std::sort` with
        // `std::greater`; a stable sort is used here (identical for the small
        // child lists, deterministic tie order otherwise).
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    }

    fn branch_and_bound(
        &self,
        discrete_scans: &[DiscreteScan2D],
        search_parameters: &SearchParameters,
        candidates: &[Candidate2D],
        candidate_depth: i32,
        min_score: f32,
    ) -> Candidate2D {
        if candidate_depth == 0 {
            // Return the best candidate.
            // RUST PORT NOTE: an empty list (only possible if the search
            // window is empty) is UB in C++; here it yields a candidate that
            // does not beat `min_score`.
            return candidates.first().copied().unwrap_or_else(|| {
                let mut candidate = Candidate2D::new(0, 0, 0, search_parameters);
                candidate.score = min_score;
                candidate
            });
        }

        let mut best_high_resolution_candidate = Candidate2D::new(0, 0, 0, search_parameters);
        best_high_resolution_candidate.score = min_score;
        let half_width = 1 << (candidate_depth - 1);
        for candidate in candidates {
            if candidate.score <= min_score {
                break;
            }
            // RUST PORT NOTE: at most 4 children, kept on the stack instead
            // of in a `std::vector`.
            let mut higher_resolution_candidates = [*candidate; 4];
            let mut num_children = 0;
            let bounds = &search_parameters.linear_bounds[candidate.scan_index as usize];
            'x: for x_offset in [0, half_width] {
                if candidate.x_index_offset + x_offset > bounds.max_x {
                    break 'x;
                }
                for y_offset in [0, half_width] {
                    if candidate.y_index_offset + y_offset > bounds.max_y {
                        break;
                    }
                    higher_resolution_candidates[num_children] = Candidate2D::new(
                        candidate.scan_index,
                        candidate.x_index_offset + x_offset,
                        candidate.y_index_offset + y_offset,
                        search_parameters,
                    );
                    num_children += 1;
                }
            }
            let children = &mut higher_resolution_candidates[..num_children];
            Self::score_candidates(
                self.precomputation_grid_stack.get(candidate_depth - 1),
                discrete_scans,
                children,
            );
            let child_best = self.branch_and_bound(
                discrete_scans,
                search_parameters,
                children,
                candidate_depth - 1,
                best_high_resolution_candidate.score,
            );
            // `std::max` keeps the first argument on ties.
            if best_high_resolution_candidate.score < child_best.score {
                best_high_resolution_candidate = child_best;
            }
        }
        best_high_resolution_candidate
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::mapping::proto as mapping_proto;
    use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::probability_grid_range_data_inserter_2d::{
        ProbabilityGridRangeDataInserter2D, create_probability_grid_range_data_inserter_options_2d,
    };
    use crate::mapping::two_d::xy_index::XYIndexRangeIterator;
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::sensor::range_data::RangeData;
    use crate::sensor::rangefinder_point::RangefinderPoint;
    use crate::transform::rigid_transform::Rigid2f;
    use crate::transform::rigid_transform_test_helpers::assert_is_nearly;
    use crate::transform::transform::embed_3d;

    fn check_precomputation_grid(probability_grid: &ProbabilityGrid, widths: &[i32]) {
        let mut reusable_intermediate_grid = Vec::new();
        for &width in widths {
            let precomputation_grid = PrecomputationGrid2D::new(
                probability_grid,
                probability_grid.limits().cell_limits(),
                width,
                &mut reusable_intermediate_grid,
            );
            for xy_index in
                XYIndexRangeIterator::from_cell_limits(probability_grid.limits().cell_limits())
            {
                let mut max_score = f32::NEG_INFINITY;
                for y in 0..width {
                    for x in 0..width {
                        max_score = max_score.max(
                            probability_grid.get_probability(&(xy_index + Array2i::new(x, y))),
                        );
                    }
                }
                let value =
                    precomputation_grid.to_score(precomputation_grid.get_value(&xy_index) as f32);
                assert!(
                    (max_score - value).abs() <= 1e-4,
                    "width {width} at {xy_index:?}: {max_score} vs. {value}"
                );
            }
        }
    }

    mod precomputation_grid_test {
        use super::*;

        #[test]
        fn correct_values() {
            // Create a probability grid with random values that can be
            // exactly represented by uint8 values.
            let mut prng = StdRng::seed_from_u64(42);
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(5.0, 5.0), CellLimits::new(250, 250)),
                &conversion_tables,
            );
            let mut reusable_intermediate_grid = Vec::new();
            let precomputation_grid_dummy = PrecomputationGrid2D::new(
                &probability_grid,
                probability_grid.limits().cell_limits(),
                1,
                &mut reusable_intermediate_grid,
            );
            for xy_index in XYIndexRangeIterator::new(Array2i::new(50, 50), Array2i::new(249, 249))
            {
                let value = prng.random_range(0..=255) as f32;
                probability_grid
                    .set_probability(&xy_index, precomputation_grid_dummy.to_score(value));
            }
            check_precomputation_grid(&probability_grid, &[1, 2, 3, 8]);
        }

        #[test]
        fn tiny_probability_grid() {
            let mut prng = StdRng::seed_from_u64(42);
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(0.1, 0.1), CellLimits::new(4, 4)),
                &conversion_tables,
            );
            let mut reusable_intermediate_grid = Vec::new();
            let precomputation_grid_dummy = PrecomputationGrid2D::new(
                &probability_grid,
                probability_grid.limits().cell_limits(),
                1,
                &mut reusable_intermediate_grid,
            );
            for xy_index in
                XYIndexRangeIterator::from_cell_limits(probability_grid.limits().cell_limits())
            {
                let value = prng.random_range(0..=255) as f32;
                probability_grid
                    .set_probability(&xy_index, precomputation_grid_dummy.to_score(value));
            }
            check_precomputation_grid(&probability_grid, &[1, 2, 3, 8, 200]);
        }
    }

    fn create_fast_correlative_scan_matcher_test_options_2d(
        branch_and_bound_depth: i32,
    ) -> proto::FastCorrelativeScanMatcherOptions2D {
        let parameter_dictionary = make_dictionary(&format!(
            r"
      return {{
         linear_search_window = 3.,
         angular_search_window = 1.,
         branch_and_bound_depth = {branch_and_bound_depth}}}"
        ));
        create_fast_correlative_scan_matcher_options_2d(&parameter_dictionary)
    }

    fn create_range_data_inserter_test_options_2d()
    -> mapping_proto::ProbabilityGridRangeDataInserterOptions2D {
        let parameter_dictionary = make_dictionary(
            r"
      return {
        insert_free_space = true,
        hit_probability = 0.7,
        miss_probability = 0.4,
      }",
        );
        create_probability_grid_range_data_inserter_options_2d(&parameter_dictionary)
    }

    fn point_cloud(points: &[(f32, f32)]) -> PointCloud {
        PointCloud::from_points(
            points
                .iter()
                .map(|&(x, y)| RangefinderPoint {
                    position: Vector3::new(x, y, 0.0),
                })
                .collect(),
        )
    }

    fn new_probability_grid() -> ProbabilityGrid {
        let conversion_tables = Arc::new(ValueConversionTables::new());
        ProbabilityGrid::new(
            MapLimits::new(0.05, Vector2::new(5.0, 5.0), CellLimits::new(200, 200)),
            &conversion_tables,
        )
    }

    mod fast_correlative_scan_matcher_test {
        use super::*;

        #[test]
        fn correct_pose() {
            let mut prng = StdRng::seed_from_u64(42);
            let mut distribution = move || prng.random_range(-1.0_f32..1.0_f32);
            let range_data_inserter = ProbabilityGridRangeDataInserter2D::new(
                &create_range_data_inserter_test_options_2d(),
            );
            const MIN_SCORE: f32 = 0.1;
            let options = create_fast_correlative_scan_matcher_test_options_2d(3);

            let point_cloud = point_cloud(&[
                (-2.5, 0.5),
                (-2.0, 0.5),
                (0.0, -0.5),
                (0.5, -1.6),
                (2.5, 0.5),
                (2.5, 1.7),
            ]);

            for _ in 0..50 {
                let x = 2.0 * distribution();
                let y = 2.0 * distribution();
                let theta = 0.5 * distribution();
                let expected_pose = Rigid2f::new(Vector2::new(x, y), theta);

                let mut probability_grid = new_probability_grid();
                range_data_inserter.insert(
                    &RangeData {
                        origin: Vector3::new(
                            expected_pose.translation().x,
                            expected_pose.translation().y,
                            0.0,
                        ),
                        returns: transform_point_cloud(&point_cloud, &embed_3d(&expected_pose)),
                        misses: PointCloud::new(),
                    },
                    &mut probability_grid,
                );
                probability_grid.finish_update();

                let fast_correlative_scan_matcher =
                    FastCorrelativeScanMatcher2D::new(&probability_grid, &options);
                let (score, pose_estimate) = fast_correlative_scan_matcher
                    .match_(&Rigid2d::identity(), &point_cloud, MIN_SCORE)
                    .expect("match failed");
                assert!(MIN_SCORE < score);
                assert_is_nearly(&expected_pose, &pose_estimate.cast::<f32>(), 0.03);
            }
        }

        #[test]
        fn full_submap_matching() {
            let mut prng = StdRng::seed_from_u64(42);
            let mut distribution = move || prng.random_range(-1.0_f32..1.0_f32);
            let range_data_inserter = ProbabilityGridRangeDataInserter2D::new(
                &create_range_data_inserter_test_options_2d(),
            );
            const MIN_SCORE: f32 = 0.1;
            let options = create_fast_correlative_scan_matcher_test_options_2d(6);

            let unperturbed_point_cloud = point_cloud(&[
                (-2.5, 0.5),
                (-2.25, 0.5),
                (0.0, 0.5),
                (0.25, 1.6),
                (2.5, 0.5),
                (2.0, 1.8),
            ]);

            for _ in 0..20 {
                let x = 10.0 * distribution();
                let y = 10.0 * distribution();
                let theta = 1.6 * distribution();
                let perturbation = Rigid2f::new(Vector2::new(x, y), theta);
                let point_cloud =
                    transform_point_cloud(&unperturbed_point_cloud, &embed_3d(&perturbation));
                let x = 2.0 * distribution();
                let y = 2.0 * distribution();
                let theta = 0.5 * distribution();
                let expected_pose =
                    Rigid2f::new(Vector2::new(x, y), theta) * perturbation.inverse();

                let mut probability_grid = new_probability_grid();
                range_data_inserter.insert(
                    &RangeData {
                        origin: *embed_3d(&(expected_pose * perturbation)).translation(),
                        returns: transform_point_cloud(&point_cloud, &embed_3d(&expected_pose)),
                        misses: PointCloud::new(),
                    },
                    &mut probability_grid,
                );
                probability_grid.finish_update();

                let fast_correlative_scan_matcher =
                    FastCorrelativeScanMatcher2D::new(&probability_grid, &options);
                let (score, pose_estimate) = fast_correlative_scan_matcher
                    .match_full_submap(&point_cloud, MIN_SCORE)
                    .expect("match failed");
                assert!(MIN_SCORE < score);
                assert_is_nearly(&expected_pose, &pose_estimate.cast::<f32>(), 0.03);
            }
        }

        // Not in C++: no match above an unreachable minimum score.
        #[test]
        fn no_match_above_max_score() {
            let options = create_fast_correlative_scan_matcher_test_options_2d(3);
            let probability_grid = new_probability_grid();
            let matcher = FastCorrelativeScanMatcher2D::new(&probability_grid, &options);
            let cloud = point_cloud(&[(1.0, 0.0), (0.0, 1.0)]);
            assert!(matcher.match_(&Rigid2d::identity(), &cloud, 0.95).is_none());
        }
    }

    mod sliding_window_maximum {
        use super::*;

        #[test]
        fn tracks_maximum() {
            let mut window = SlidingWindowMaximum::default();
            window.add_value(0.3);
            window.add_value(0.5);
            window.add_value(0.4);
            assert_eq!(window.get_maximum(), 0.5);
            window.remove_value(0.3);
            assert_eq!(window.get_maximum(), 0.5);
            window.remove_value(0.5);
            assert_eq!(window.get_maximum(), 0.4);
            window.remove_value(0.4);
            window.check_is_empty();
        }
    }
}
