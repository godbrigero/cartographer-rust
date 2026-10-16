//! Port of `cartographer/mapping/internal/3d/scan_matching/fast_correlative_scan_matcher_3d` (.h, .cc).
//!
//! This is an implementation of a 3D branch-and-bound algorithm similar to
//! `FastCorrelativeScanMatcher2D`.

use std::borrow::Borrow;
use std::cmp::Ordering;

use nalgebra::{DVector, Quaternion, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::math::{pow2, power};
use crate::common::port::round_to_int;
use crate::mapping::internal::three_d::scan_matching::low_resolution_matcher::{
    MatchingFunction, create_low_resolution_matcher,
};
use crate::mapping::internal::three_d::scan_matching::precomputation_grid_3d::{
    PrecomputationGrid3D, convert_to_precomputation_grid, precompute_grid,
};
use crate::mapping::internal::three_d::scan_matching::rotational_scan_matcher::RotationalScanMatcher;
use crate::mapping::scan_matching::proto;
use crate::mapping::three_d::hybrid_grid::{Array3i, HybridGrid};
use crate::mapping::trajectory_node::TrajectoryNodeData;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::{
    Rigid3d, Rigid3f, quaternion_cast, quaternion_conjugate, quaternion_mul,
    quaternion_squared_norm,
};
use crate::transform::transform::{angle_axis_vector_to_rotation_quaternion, get_yaw_quaternion};

/// Reads `FastCorrelativeScanMatcherOptions3D` from `parameter_dictionary`.
pub fn create_fast_correlative_scan_matcher_options_3d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::FastCorrelativeScanMatcherOptions3D {
    proto::FastCorrelativeScanMatcherOptions3D {
        branch_and_bound_depth: parameter_dictionary.get_int("branch_and_bound_depth"),
        full_resolution_depth: parameter_dictionary.get_int("full_resolution_depth"),
        min_rotational_score: parameter_dictionary.get_double("min_rotational_score"),
        min_low_resolution_score: parameter_dictionary.get_double("min_low_resolution_score"),
        linear_xy_search_window: parameter_dictionary.get_double("linear_xy_search_window"),
        linear_z_search_window: parameter_dictionary.get_double("linear_z_search_window"),
        angular_search_window: parameter_dictionary.get_double("angular_search_window"),
    }
}

/// The stack of precomputation grids used by the branch-and-bound search,
/// from the original grid (depth 0) to the coarsest one.
pub struct PrecomputationGridStack3D {
    precomputation_grids: Vec<PrecomputationGrid3D>,
}

impl PrecomputationGridStack3D {
    /// Builds `options.branch_and_bound_depth` grids from `hybrid_grid`.
    pub fn new(
        hybrid_grid: &HybridGrid,
        options: &proto::FastCorrelativeScanMatcherOptions3D,
    ) -> Self {
        assert!(
            options.branch_and_bound_depth >= 1,
            "Check failed: options.branch_and_bound_depth() >= 1"
        );
        assert!(
            options.full_resolution_depth >= 1,
            "Check failed: options.full_resolution_depth() >= 1"
        );
        let mut precomputation_grids = Vec::with_capacity(options.branch_and_bound_depth as usize);
        precomputation_grids.push(convert_to_precomputation_grid(hybrid_grid));
        let mut last_width = 1;
        for depth in 1..options.branch_and_bound_depth {
            let half_resolution = depth >= options.full_resolution_depth;
            let next_width = 1 << depth;
            let full_voxels_per_high_resolution_voxel =
                1 << 0.max(depth - options.full_resolution_depth);
            let shift = Array3i::repeat(
                (next_width - last_width + (full_voxels_per_high_resolution_voxel - 1))
                    / full_voxels_per_high_resolution_voxel,
            );
            let next = precompute_grid(
                precomputation_grids.last().expect("not empty"),
                half_resolution,
                &shift,
            );
            precomputation_grids.push(next);
            last_width = next_width;
        }
        Self {
            precomputation_grids,
        }
    }

    /// The grid at `depth` (panics if out of range, like `std::vector::at`).
    #[inline]
    pub fn get(&self, depth: i32) -> &PrecomputationGrid3D {
        &self.precomputation_grids[depth as usize]
    }

    /// The largest depth (number of grids - 1).
    #[inline]
    pub fn max_depth(&self) -> i32 {
        self.precomputation_grids.len() as i32 - 1
    }
}

/// A scan at one rotation, discretized for all depths.
///
/// RUST PORT NOTE: C++ stores `cell_indices_per_depth` with
/// `min(full_resolution_depth, branch_and_bound_depth)` identical copies of
/// the full resolution indices; here they are stored once
/// (`full_resolution_cell_indices`) and [`DiscreteScan3D::cell_indices`]
/// maps the depth.
#[derive(Clone, Debug)]
pub struct DiscreteScan3D {
    /// Pose of the scan in the submap frame.
    pub pose: Rigid3f,
    /// Full resolution cell indices (depths `0..full_resolution_depth`).
    full_resolution_cell_indices: Vec<Array3i>,
    /// Number of depths using `full_resolution_cell_indices`.
    full_resolution_depth: usize,
    /// Cell indices for the depths `full_resolution_depth..`.
    low_resolution_cell_indices: Vec<Vec<Array3i>>,
    /// Score of the rotational scan matcher for this rotation.
    pub rotational_score: f32,
}

impl DiscreteScan3D {
    /// Number of depths (C++ `cell_indices_per_depth.size()`).
    pub fn num_depths(&self) -> usize {
        self.full_resolution_depth + self.low_resolution_cell_indices.len()
    }

    /// The discretized scan at `depth` (C++ `cell_indices_per_depth[depth]`).
    #[inline]
    pub fn cell_indices(&self, depth: usize) -> &[Array3i] {
        assert!(
            depth < self.num_depths(),
            "Check failed: depth < discrete_scan.cell_indices_per_depth.size()"
        );
        if depth < self.full_resolution_depth {
            &self.full_resolution_cell_indices
        } else {
            &self.low_resolution_cell_indices[depth - self.full_resolution_depth]
        }
    }
}

/// A search candidate.
#[derive(Clone, Copy, Debug)]
pub struct Candidate3D {
    /// Index into the discrete scans vectors.
    pub scan_index: usize,
    /// Linear offset from the initial pose in cell indices. For lower
    /// resolution candidates this is the lowest offset of the 2^depth x
    /// 2^depth x 2^depth block of possibilities.
    pub offset: Array3i,
    /// Score, higher is better.
    pub score: f32,
    /// Score of the low resolution matcher.
    pub low_resolution_score: f32,
}

impl Candidate3D {
    /// Creates an unscored candidate.
    pub fn new(scan_index: usize, offset: Array3i) -> Self {
        Self {
            scan_index,
            offset,
            score: f32::NEG_INFINITY,
            low_resolution_score: 0.,
        }
    }

    /// C++ `Candidate3D::Unsuccessful()`.
    pub fn unsuccessful() -> Self {
        Self::new(0, Array3i::zeros())
    }
}

/// Result of a successful match (C++ `FastCorrelativeScanMatcher3D::Result`).
#[derive(Clone, Copy, Debug)]
pub struct FastCorrelativeScanMatcher3DResult {
    /// Branch-and-bound score.
    pub score: f32,
    /// Pose of the node in the submap frame.
    pub pose_estimate: Rigid3d,
    /// Score of the rotational scan matcher.
    pub rotational_score: f32,
    /// Score of the low resolution matcher.
    pub low_resolution_score: f32,
}

struct SearchParameters<'a, 'b> {
    linear_xy_window_size: i32, // voxels
    linear_z_window_size: i32,  // voxels
    angular_search_window: f64, // radians
    low_resolution_matcher: &'b MatchingFunction<'a>,
}

/// Eigen `Quaternion::inverse()`: conjugate divided by the squared norm (zero
/// if the norm is zero).
fn quaternion_inverse(q: &Quaternion<f32>) -> Quaternion<f32> {
    let n2 = quaternion_squared_norm(q);
    if n2 > 0. {
        quaternion_conjugate(q) / n2
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

/// 3D branch-and-bound scan matcher against the high resolution grid of a
/// submap, verified by a low resolution matcher and pre-filtered by the
/// rotational scan matcher.
///
/// RUST PORT NOTE: C++ keeps raw pointers to the low resolution grid and the
/// rotational histogram, which must outlive the matcher. Here the low
/// resolution grid is any `G: Borrow<HybridGrid>` (e.g. `&HybridGrid`,
/// `Arc<HybridGrid>` or a handle to a shared submap) and the histogram (a
/// small vector) is copied. The precomputation grids are owned. The matcher
/// is immutable after construction, `Sync` if `G` is, and all matching
/// methods take `&self`, so it can be shared between threads like in C++.
pub struct FastCorrelativeScanMatcher3D<G: Borrow<HybridGrid>> {
    options: proto::FastCorrelativeScanMatcherOptions3D,
    resolution: f32,
    width_in_voxels: i32,
    precomputation_grid_stack: PrecomputationGridStack3D,
    low_resolution_hybrid_grid: G,
    rotational_scan_matcher_histogram: DVector<f32>,
}

impl<G: Borrow<HybridGrid>> FastCorrelativeScanMatcher3D<G> {
    /// Creates the matcher, precomputing the grid stack for `hybrid_grid`.
    pub fn new(
        hybrid_grid: &HybridGrid,
        low_resolution_hybrid_grid: G,
        rotational_scan_matcher_histogram: &DVector<f32>,
        options: &proto::FastCorrelativeScanMatcherOptions3D,
    ) -> Self {
        Self {
            options: *options,
            resolution: hybrid_grid.resolution(),
            width_in_voxels: hybrid_grid.grid_size(),
            precomputation_grid_stack: PrecomputationGridStack3D::new(hybrid_grid, options),
            low_resolution_hybrid_grid,
            rotational_scan_matcher_histogram: rotational_scan_matcher_histogram.clone(),
        }
    }

    /// Aligns the node with the given `constant_data` within the
    /// `hybrid_grid` given `global_node_pose` and `global_submap_pose`. A
    /// result is only returned if a score above `min_score` (excluding
    /// equality) is possible.
    pub fn match_(
        &self,
        global_node_pose: &Rigid3d,
        global_submap_pose: &Rigid3d,
        constant_data: &TrajectoryNodeData,
        min_score: f32,
    ) -> Option<FastCorrelativeScanMatcher3DResult> {
        let low_resolution_matcher = create_low_resolution_matcher(
            self.low_resolution_hybrid_grid.borrow(),
            &constant_data.low_resolution_point_cloud,
        );
        let search_parameters = SearchParameters {
            linear_xy_window_size: round_to_int(
                self.options.linear_xy_search_window / f64::from(self.resolution),
            ),
            linear_z_window_size: round_to_int(
                self.options.linear_z_search_window / f64::from(self.resolution),
            ),
            angular_search_window: self.options.angular_search_window,
            low_resolution_matcher: &*low_resolution_matcher,
        };
        self.match_with_search_parameters(
            &search_parameters,
            &global_node_pose.cast::<f32>(),
            &global_submap_pose.cast::<f32>(),
            &constant_data.high_resolution_point_cloud,
            &constant_data.rotational_scan_matcher_histogram,
            &constant_data.gravity_alignment,
            min_score,
        )
    }

    /// Aligns the node with the given `constant_data` within the
    /// `hybrid_grid` given rotations which are expected to be approximately
    /// gravity aligned. A result is only returned if a score above
    /// `min_score` (excluding equality) is possible.
    pub fn match_full_submap(
        &self,
        global_node_rotation: &Quaternion<f64>,
        global_submap_rotation: &Quaternion<f64>,
        constant_data: &TrajectoryNodeData,
        min_score: f32,
    ) -> Option<FastCorrelativeScanMatcher3DResult> {
        let mut max_point_distance = 0.0f32;
        for point in &constant_data.high_resolution_point_cloud {
            max_point_distance = max_point_distance.max(point.position.norm());
        }
        let linear_window_size = (self.width_in_voxels + 1) / 2
            + round_to_int(max_point_distance / self.resolution + 0.5);
        let low_resolution_matcher = create_low_resolution_matcher(
            self.low_resolution_hybrid_grid.borrow(),
            &constant_data.low_resolution_point_cloud,
        );
        let search_parameters = SearchParameters {
            linear_xy_window_size: linear_window_size,
            linear_z_window_size: linear_window_size,
            angular_search_window: std::f64::consts::PI,
            low_resolution_matcher: &*low_resolution_matcher,
        };
        self.match_with_search_parameters(
            &search_parameters,
            &Rigid3f::from_rotation(quaternion_cast(global_node_rotation)),
            &Rigid3f::from_rotation(quaternion_cast(global_submap_rotation)),
            &constant_data.high_resolution_point_cloud,
            &constant_data.rotational_scan_matcher_histogram,
            &constant_data.gravity_alignment,
            min_score,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn match_with_search_parameters(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        global_node_pose: &Rigid3f,
        global_submap_pose: &Rigid3f,
        point_cloud: &PointCloud,
        rotational_scan_matcher_histogram: &DVector<f32>,
        gravity_alignment: &Quaternion<f64>,
        min_score: f32,
    ) -> Option<FastCorrelativeScanMatcher3DResult> {
        let discrete_scans = self.generate_discrete_scans(
            search_parameters,
            point_cloud,
            rotational_scan_matcher_histogram,
            gravity_alignment,
            global_node_pose,
            global_submap_pose,
        );

        let mut lowest_resolution_candidates =
            self.compute_lowest_resolution_candidates(search_parameters, &discrete_scans);

        let best_candidate = self.branch_and_bound(
            search_parameters,
            &discrete_scans,
            &mut lowest_resolution_candidates,
            self.precomputation_grid_stack.max_depth(),
            min_score,
        );
        if best_candidate.score > min_score {
            return Some(FastCorrelativeScanMatcher3DResult {
                score: best_candidate.score,
                pose_estimate: self
                    .get_pose_from_candidate(&discrete_scans, &best_candidate)
                    .cast::<f64>(),
                rotational_score: discrete_scans[best_candidate.scan_index].rotational_score,
                low_resolution_score: best_candidate.low_resolution_score,
            });
        }
        None
    }

    fn discretize_scan(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        point_cloud: &PointCloud,
        pose: &Rigid3f,
        rotational_score: f32,
    ) -> DiscreteScan3D {
        let original_grid = self.precomputation_grid_stack.get(0);
        let full_resolution_cell_indices: Vec<Array3i> = point_cloud
            .iter()
            .map(|point| original_grid.get_cell_index(&(pose * &point.position)))
            .collect();
        let full_resolution_depth = self
            .options
            .full_resolution_depth
            .min(self.options.branch_and_bound_depth);
        assert!(
            full_resolution_depth >= 1,
            "Check failed: full_resolution_depth >= 1"
        );
        let low_resolution_depth = self.options.branch_and_bound_depth - full_resolution_depth;
        assert!(
            low_resolution_depth >= 0,
            "Check failed: low_resolution_depth >= 0"
        );
        let search_window_start = Array3i::new(
            -search_parameters.linear_xy_window_size,
            -search_parameters.linear_xy_window_size,
            -search_parameters.linear_z_window_size,
        );
        let mut low_resolution_cell_indices = Vec::with_capacity(low_resolution_depth as usize);
        for i in 0..low_resolution_depth {
            let reduction_exponent = i + 1;
            let low_resolution_search_window_start =
                search_window_start.map(|c| c >> reduction_exponent);
            low_resolution_cell_indices.push(
                full_resolution_cell_indices
                    .iter()
                    .map(|cell_index| {
                        let cell_at_start = cell_index + search_window_start;
                        let low_resolution_cell_at_start =
                            cell_at_start.map(|c| c >> reduction_exponent);
                        low_resolution_cell_at_start - low_resolution_search_window_start
                    })
                    .collect(),
            );
        }
        DiscreteScan3D {
            pose: *pose,
            full_resolution_cell_indices,
            full_resolution_depth: full_resolution_depth as usize,
            low_resolution_cell_indices,
            rotational_score,
        }
    }

    fn generate_discrete_scans(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        point_cloud: &PointCloud,
        rotational_scan_matcher_histogram: &DVector<f32>,
        gravity_alignment: &Quaternion<f64>,
        global_node_pose: &Rigid3f,
        global_submap_pose: &Rigid3f,
    ) -> Vec<DiscreteScan3D> {
        // We set this value to something on the order of resolution to make
        // sure that the std::acos() below is defined.
        let mut max_scan_range = 3.0f32 * self.resolution;
        for point in point_cloud {
            let range = point.position.norm();
            max_scan_range = range.max(max_scan_range);
        }
        const SAFETY_MARGIN: f32 = 1.0 - 1e-2;
        let angular_step_size =
            SAFETY_MARGIN * (1.0f32 - pow2(self.resolution) / (2.0 * pow2(max_scan_range))).acos();
        let angular_window_size =
            round_to_int(search_parameters.angular_search_window / f64::from(angular_step_size));
        let angles: Vec<f32> = (-angular_window_size..=angular_window_size)
            .map(|rz| rz as f32 * angular_step_size)
            .collect();
        let node_to_submap = global_submap_pose.inverse() * *global_node_pose;
        let gravity_alignment_inverse =
            quaternion_cast::<f64, f32>(&quaternion_inverse_f64(gravity_alignment));
        let rotational_scan_matcher =
            RotationalScanMatcher::new(&self.rotational_scan_matcher_histogram);
        let scores = rotational_scan_matcher.match_(
            rotational_scan_matcher_histogram,
            get_yaw_quaternion(&quaternion_mul(
                node_to_submap.rotation(),
                &gravity_alignment_inverse,
            )),
            &angles,
        );
        let global_submap_rotation_inverse = quaternion_inverse(global_submap_pose.rotation());
        let mut result = Vec::new();
        for (&angle, &score) in angles.iter().zip(&scores) {
            if f64::from(score) < self.options.min_rotational_score {
                continue;
            }
            let angle_axis = Vector3::new(0., 0., angle);
            // It's important to apply the 'angle_axis' rotation between the
            // translation and rotation of the 'initial_pose', so that the
            // rotation is around the origin of the range data, and yaw is in
            // map frame.
            let pose = Rigid3f::new(
                *node_to_submap.translation(),
                quaternion_mul(
                    &quaternion_mul(
                        &global_submap_rotation_inverse,
                        &angle_axis_vector_to_rotation_quaternion(&angle_axis),
                    ),
                    global_node_pose.rotation(),
                ),
            );
            result.push(self.discretize_scan(search_parameters, point_cloud, &pose, score));
        }
        result
    }

    fn generate_lowest_resolution_candidates(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        num_discrete_scans: usize,
    ) -> Vec<Candidate3D> {
        let linear_step_size = 1 << self.precomputation_grid_stack.max_depth();
        let num_lowest_resolution_linear_xy_candidates =
            (2 * search_parameters.linear_xy_window_size + linear_step_size) / linear_step_size;
        let num_lowest_resolution_linear_z_candidates =
            (2 * search_parameters.linear_z_window_size + linear_step_size) / linear_step_size;
        let num_candidates = num_discrete_scans
            * power(num_lowest_resolution_linear_xy_candidates as usize, 2)
            * num_lowest_resolution_linear_z_candidates as usize;
        let mut candidates = Vec::with_capacity(num_candidates);
        for scan_index in 0..num_discrete_scans {
            let mut z = -search_parameters.linear_z_window_size;
            while z <= search_parameters.linear_z_window_size {
                let mut y = -search_parameters.linear_xy_window_size;
                while y <= search_parameters.linear_xy_window_size {
                    let mut x = -search_parameters.linear_xy_window_size;
                    while x <= search_parameters.linear_xy_window_size {
                        candidates.push(Candidate3D::new(scan_index, Array3i::new(x, y, z)));
                        x += linear_step_size;
                    }
                    y += linear_step_size;
                }
                z += linear_step_size;
            }
        }
        assert_eq!(
            candidates.len(),
            num_candidates,
            "Check failed: candidates.size() == num_candidates"
        );
        candidates
    }

    fn score_candidates(
        &self,
        depth: i32,
        discrete_scans: &[DiscreteScan3D],
        candidates: &mut [Candidate3D],
    ) {
        let reduction_exponent = 0.max(depth - self.options.full_resolution_depth + 1);
        let grid = self.precomputation_grid_stack.get(depth);
        for candidate in candidates.iter_mut() {
            let discrete_scan = &discrete_scans[candidate.scan_index];
            let offset = candidate.offset.map(|c| c >> reduction_exponent);
            let cell_indices = discrete_scan.cell_indices(depth as usize);
            let mut sum: i32 = 0;
            for cell_index in cell_indices {
                let proposed_cell_index = cell_index + offset;
                sum += i32::from(grid.value(&proposed_cell_index));
            }
            candidate.score =
                PrecomputationGrid3D::to_probability(sum as f32 / cell_indices.len() as f32);
        }
        // RUST PORT NOTE: C++ `std::sort` with `std::greater` (unstable;
        // the order of equal scores is implementation-defined).
        candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    }

    fn compute_lowest_resolution_candidates(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        discrete_scans: &[DiscreteScan3D],
    ) -> Vec<Candidate3D> {
        let mut lowest_resolution_candidates =
            self.generate_lowest_resolution_candidates(search_parameters, discrete_scans.len());
        self.score_candidates(
            self.precomputation_grid_stack.max_depth(),
            discrete_scans,
            &mut lowest_resolution_candidates,
        );
        lowest_resolution_candidates
    }

    fn get_pose_from_candidate(
        &self,
        discrete_scans: &[DiscreteScan3D],
        candidate: &Candidate3D,
    ) -> Rigid3f {
        Rigid3f::from_translation(candidate.offset.cast::<f32>() * self.resolution)
            * discrete_scans[candidate.scan_index].pose
    }

    fn branch_and_bound(
        &self,
        search_parameters: &SearchParameters<'_, '_>,
        discrete_scans: &[DiscreteScan3D],
        candidates: &mut [Candidate3D],
        candidate_depth: i32,
        min_score: f32,
    ) -> Candidate3D {
        if candidate_depth == 0 {
            for candidate in candidates.iter() {
                if candidate.score <= min_score {
                    // Return if the candidate is bad because the following
                    // candidate will not have better score.
                    return Candidate3D::unsuccessful();
                }
                let low_resolution_score = (search_parameters.low_resolution_matcher)(
                    &self.get_pose_from_candidate(discrete_scans, candidate),
                );
                if f64::from(low_resolution_score) >= self.options.min_low_resolution_score {
                    // We found the best candidate that passes the matching
                    // function.
                    let mut best_candidate = *candidate;
                    best_candidate.low_resolution_score = low_resolution_score;
                    return best_candidate;
                }
            }

            // All candidates have good scores but none passes the matching
            // function.
            return Candidate3D::unsuccessful();
        }

        let mut best_high_resolution_candidate = Candidate3D::unsuccessful();
        best_high_resolution_candidate.score = min_score;
        let half_width = 1 << (candidate_depth - 1);
        for candidate in candidates.iter() {
            if candidate.score <= min_score {
                break;
            }
            // RUST PORT NOTE: at most 8 children; stored on the stack instead
            // of a heap-allocated `std::vector`.
            let mut children = [Candidate3D::unsuccessful(); 8];
            let mut num_children = 0;
            'z: for z in [0, half_width] {
                if candidate.offset.z + z > search_parameters.linear_z_window_size {
                    break 'z;
                }
                for y in [0, half_width] {
                    if candidate.offset.y + y > search_parameters.linear_xy_window_size {
                        break;
                    }
                    for x in [0, half_width] {
                        if candidate.offset.x + x > search_parameters.linear_xy_window_size {
                            break;
                        }
                        children[num_children] = Candidate3D::new(
                            candidate.scan_index,
                            candidate.offset + Array3i::new(x, y, z),
                        );
                        num_children += 1;
                    }
                }
            }
            let higher_resolution_candidates = &mut children[..num_children];
            self.score_candidates(
                candidate_depth - 1,
                discrete_scans,
                higher_resolution_candidates,
            );
            let child = self.branch_and_bound(
                search_parameters,
                discrete_scans,
                higher_resolution_candidates,
                candidate_depth - 1,
                best_high_resolution_candidate.score,
            );
            // C++ `std::max(best, child)`: keeps `best` unless `best < child`.
            if best_high_resolution_candidate
                .score
                .partial_cmp(&child.score)
                == Some(Ordering::Less)
            {
                best_high_resolution_candidate = child;
            }
        }
        best_high_resolution_candidate
    }
}

/// Eigen `Quaterniond::inverse()`.
fn quaternion_inverse_f64(q: &Quaternion<f64>) -> Quaternion<f64> {
    let n2 = quaternion_squared_norm(q);
    if n2 > 0. {
        quaternion_conjugate(q) / n2
    } else {
        Quaternion::new(0., 0., 0., 0.)
    }
}

#[cfg(test)]
mod tests {
    mod fast_correlative_scan_matcher_3d_test {
        use nalgebra::{DVector, Quaternion, Vector3};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::common::time::from_universal;
        use crate::mapping::internal::three_d::scan_matching::fast_correlative_scan_matcher_3d::*;
        use crate::mapping::proto as mapping_proto;
        use crate::mapping::three_d::range_data_inserter_3d::{
            RangeDataInserter3D, create_range_data_inserter_options_3d,
        };
        use crate::sensor::point_cloud::transform_point_cloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;
        use crate::transform::rigid_transform::{AngleAxis, Rigid3f};
        use crate::transform::rigid_transform_test_helpers::assert_is_nearly;

        const MIN_SCORE: f32 = 0.1;

        fn create_fast_correlative_scan_matcher_3d_test_options_3d(
            branch_and_bound_depth: i32,
        ) -> proto::FastCorrelativeScanMatcherOptions3D {
            let parameter_dictionary = make_dictionary(&format!(
                "return {{\
                 branch_and_bound_depth = {branch_and_bound_depth}, \
                 full_resolution_depth = {branch_and_bound_depth}, \
                 min_rotational_score = 0.1, \
                 min_low_resolution_score = 0.15, \
                 linear_xy_search_window = 0.8, \
                 linear_z_search_window = 0.8, \
                 angular_search_window = 0.3, \
                 }}"
            ));
            // Unknown space has kMinProbability = 0.1, so we need to make
            // sure here to pick a larger number for min_low_resolution_score,
            // otherwise we always find matches.
            create_fast_correlative_scan_matcher_options_3d(&parameter_dictionary)
        }

        fn create_range_data_inserter_test_options_3d() -> mapping_proto::RangeDataInserterOptions3D
        {
            let parameter_dictionary = make_dictionary(
                "return { \
                 hit_probability = 0.7, \
                 miss_probability = 0.4, \
                 num_free_space_voxels = 5, \
                 intensity_threshold = 100.0, \
                 }",
            );
            create_range_data_inserter_options_3d(&parameter_dictionary)
        }

        struct Fixture {
            // C++ uses std::mt19937(42) with uniform_real_distribution(-1, 1);
            // exact samples are not required.
            prng: StdRng,
            range_data_inserter: RangeDataInserter3D,
            options: proto::FastCorrelativeScanMatcherOptions3D,
            point_cloud: PointCloud,
            rotational_scan_matcher_histogram: DVector<f32>,
        }

        impl Fixture {
            fn new() -> Self {
                let point = |x: f32, y: f32, z: f32| RangefinderPoint {
                    position: Vector3::new(x, y, z),
                };
                Self {
                    prng: StdRng::seed_from_u64(42),
                    range_data_inserter: RangeDataInserter3D::new(
                        &create_range_data_inserter_test_options_3d(),
                    ),
                    options: create_fast_correlative_scan_matcher_3d_test_options_3d(6),
                    point_cloud: PointCloud::from_points(vec![
                        point(4., 0., 0.),
                        point(4.5, 0., 0.),
                        point(5., 0., 0.),
                        point(5.5, 0., 0.),
                        point(0., 4., 0.),
                        point(0., 4.5, 0.),
                        point(0., 5., 0.),
                        point(0., 5.5, 0.),
                        point(0., 0., 4.),
                        point(0., 0., 4.5),
                        point(0., 0., 5.),
                        point(0., 0., 5.5),
                    ]),
                    rotational_scan_matcher_histogram: DVector::zeros(10),
                }
            }

            fn distribution(&mut self) -> f32 {
                self.prng.random_range(-1.0f32..1.0)
            }

            fn get_random_pose(&mut self) -> Rigid3f {
                let x = 0.7 * self.distribution();
                let y = 0.7 * self.distribution();
                let z = 0.7 * self.distribution();
                let theta = 0.2 * self.distribution();
                Rigid3f::from_translation(Vector3::new(x, y, z))
                    * Rigid3f::from_angle_axis(AngleAxis::new(theta, Vector3::z()))
            }

            fn get_hybrid_grid(&self, pose: &Rigid3f) -> HybridGrid {
                let mut hybrid_grid = HybridGrid::new(0.05);
                self.range_data_inserter.insert(
                    &RangeData {
                        origin: *pose.translation(),
                        returns: transform_point_cloud(&self.point_cloud, pose),
                        misses: PointCloud::new(),
                    },
                    &mut hybrid_grid,
                    /*intensity_hybrid_grid=*/ None,
                );
                hybrid_grid.finish_update();
                hybrid_grid
            }

            fn create_constant_data(
                &self,
                low_resolution_point_cloud: &PointCloud,
            ) -> TrajectoryNodeData {
                TrajectoryNodeData {
                    time: from_universal(0),
                    gravity_alignment: Quaternion::identity(),
                    filtered_gravity_aligned_point_cloud: PointCloud::new(),
                    high_resolution_point_cloud: self.point_cloud.clone(),
                    low_resolution_point_cloud: low_resolution_point_cloud.clone(),
                    rotational_scan_matcher_histogram: self
                        .rotational_scan_matcher_histogram
                        .clone(),
                    local_pose: Rigid3d::identity(),
                }
            }
        }

        fn far_away_cloud() -> PointCloud {
            PointCloud::from_points(vec![RangefinderPoint {
                position: Vector3::new(42., 42., 42.),
            }])
        }

        fn check_result(result: &FastCorrelativeScanMatcher3DResult, expected_pose: &Rigid3f) {
            assert!(MIN_SCORE < result.score, "{}", result.score);
            assert!(0.09 < result.rotational_score);
            assert!(0.14 < result.low_resolution_score);
            assert_is_nearly(expected_pose, &result.pose_estimate.cast::<f32>(), 0.05);
        }

        #[test]
        fn correct_pose_for_match() {
            let mut fixture = Fixture::new();
            for _ in 0..20 {
                let expected_pose = fixture.get_random_pose();
                let hybrid_grid = fixture.get_hybrid_grid(&expected_pose);
                let fast_correlative_scan_matcher = FastCorrelativeScanMatcher3D::new(
                    &hybrid_grid,
                    &hybrid_grid,
                    &fixture.rotational_scan_matcher_histogram,
                    &fixture.options,
                );

                let result = fast_correlative_scan_matcher
                    .match_(
                        &Rigid3d::identity(),
                        &Rigid3d::identity(),
                        &fixture.create_constant_data(&fixture.point_cloud),
                        MIN_SCORE,
                    )
                    .expect("expected a match");
                check_result(&result, &expected_pose);

                let low_resolution_result = fast_correlative_scan_matcher.match_(
                    &Rigid3d::identity(),
                    &Rigid3d::identity(),
                    &fixture.create_constant_data(&far_away_cloud()),
                    MIN_SCORE,
                );
                assert!(
                    low_resolution_result.is_none(),
                    "{:?}",
                    low_resolution_result.map(|r| r.low_resolution_score)
                );
            }
        }

        #[test]
        fn correct_pose_for_match_full_submap() {
            let mut fixture = Fixture::new();
            let expected_pose = fixture.get_random_pose();
            let hybrid_grid = fixture.get_hybrid_grid(&expected_pose);
            let fast_correlative_scan_matcher = FastCorrelativeScanMatcher3D::new(
                &hybrid_grid,
                &hybrid_grid,
                &fixture.rotational_scan_matcher_histogram,
                &fixture.options,
            );

            let result = fast_correlative_scan_matcher
                .match_full_submap(
                    &Quaternion::identity(),
                    &Quaternion::identity(),
                    &fixture.create_constant_data(&fixture.point_cloud),
                    MIN_SCORE,
                )
                .expect("expected a match");
            check_result(&result, &expected_pose);

            let low_resolution_result = fast_correlative_scan_matcher.match_full_submap(
                &Quaternion::identity(),
                &Quaternion::identity(),
                &fixture.create_constant_data(&far_away_cloud()),
                MIN_SCORE,
            );
            assert!(
                low_resolution_result.is_none(),
                "{:?}",
                low_resolution_result.map(|r| r.low_resolution_score)
            );
        }

        #[test]
        fn options_and_grid_stack() {
            let options = create_fast_correlative_scan_matcher_options_3d(&make_dictionary(
                "return { branch_and_bound_depth = 8, full_resolution_depth = 3, \
                 min_rotational_score = 0.77, min_low_resolution_score = 0.55, \
                 linear_xy_search_window = 5., linear_z_search_window = 1., \
                 angular_search_window = math.rad(15.) }",
            ));
            assert_eq!(options.branch_and_bound_depth, 8);
            assert_eq!(options.full_resolution_depth, 3);
            assert_eq!(options.min_rotational_score, 0.77);
            assert_eq!(options.min_low_resolution_score, 0.55);
            assert_eq!(options.linear_xy_search_window, 5.);
            assert_eq!(options.linear_z_search_window, 1.);
            assert!((options.angular_search_window - 15f64.to_radians()).abs() < 1e-12);

            let fixture = Fixture::new();
            let hybrid_grid = fixture.get_hybrid_grid(&Rigid3f::identity());
            let stack = PrecomputationGridStack3D::new(&hybrid_grid, &options);
            assert_eq!(stack.max_depth(), 7);
            for depth in 0..=stack.max_depth() {
                assert_eq!(stack.get(depth).resolution(), 0.05);
            }
        }

        #[test]
        fn shared_grid_handles_work() {
            // The low resolution grid can be owned through an `Arc`.
            let fixture = Fixture::new();
            let expected_pose = Rigid3f::from_translation(Vector3::new(0.3, -0.2, 0.1));
            let hybrid_grid = fixture.get_hybrid_grid(&expected_pose);
            let shared = std::sync::Arc::new(hybrid_grid.clone());
            let matcher = FastCorrelativeScanMatcher3D::new(
                &hybrid_grid,
                shared,
                &fixture.rotational_scan_matcher_histogram,
                &fixture.options,
            );
            fn assert_sync<T: Sync + Send>(_: &T) {}
            assert_sync(&matcher);
            let result = matcher
                .match_(
                    &Rigid3d::identity(),
                    &Rigid3d::identity(),
                    &fixture.create_constant_data(&fixture.point_cloud),
                    MIN_SCORE,
                )
                .expect("expected a match");
            check_result(&result, &expected_pose);
        }

        /// The C++ random stream (mt19937) is not reproduced, so check the
        /// fixture over several seeds, and with half resolution levels
        /// (full_resolution_depth < branch_and_bound_depth).
        #[test]
        fn correct_pose_for_match_many_seeds_and_half_resolution_levels() {
            for seed in 0..8 {
                for full_resolution_depth in [6, 3, 1] {
                    let mut fixture = Fixture::new();
                    fixture.prng = StdRng::seed_from_u64(seed);
                    fixture.options.full_resolution_depth = full_resolution_depth;
                    for _ in 0..3 {
                        let expected_pose = fixture.get_random_pose();
                        let hybrid_grid = fixture.get_hybrid_grid(&expected_pose);
                        let matcher = FastCorrelativeScanMatcher3D::new(
                            &hybrid_grid,
                            &hybrid_grid,
                            &fixture.rotational_scan_matcher_histogram,
                            &fixture.options,
                        );
                        let result = matcher
                            .match_(
                                &Rigid3d::identity(),
                                &Rigid3d::identity(),
                                &fixture.create_constant_data(&fixture.point_cloud),
                                MIN_SCORE,
                            )
                            .unwrap_or_else(|| {
                                panic!("seed {seed} full_res {full_resolution_depth}")
                            });
                        check_result(&result, &expected_pose);
                    }
                }
            }
        }

        #[test]
        fn respects_min_score_and_node_pose() {
            let mut fixture = Fixture::new();
            let expected_pose = fixture.get_random_pose();
            let hybrid_grid = fixture.get_hybrid_grid(&expected_pose);
            let matcher = FastCorrelativeScanMatcher3D::new(
                &hybrid_grid,
                &hybrid_grid,
                &fixture.rotational_scan_matcher_histogram,
                &fixture.options,
            );
            let constant_data = fixture.create_constant_data(&fixture.point_cloud);
            let result = matcher
                .match_(
                    &Rigid3d::identity(),
                    &Rigid3d::identity(),
                    &constant_data,
                    MIN_SCORE,
                )
                .unwrap();
            // A min score equal to the best score is not reached.
            assert!(
                matcher
                    .match_(
                        &Rigid3d::identity(),
                        &Rigid3d::identity(),
                        &constant_data,
                        result.score
                    )
                    .is_none()
            );
            // The search is relative to the node pose in the submap frame: a
            // node pose estimate offset within the window still finds the
            // same pose, and a submap pose offset is undone.
            let offset = Rigid3d::from_translation(Vector3::new(0.3, -0.2, 0.1));
            let shifted = matcher
                .match_(&offset, &Rigid3d::identity(), &constant_data, MIN_SCORE)
                .unwrap();
            assert_is_nearly(&expected_pose, &shifted.pose_estimate.cast::<f32>(), 0.05);
            let in_submap = matcher
                .match_(&offset, &offset, &constant_data, MIN_SCORE)
                .unwrap();
            assert_is_nearly(&expected_pose, &in_submap.pose_estimate.cast::<f32>(), 0.05);
        }
    }
}
