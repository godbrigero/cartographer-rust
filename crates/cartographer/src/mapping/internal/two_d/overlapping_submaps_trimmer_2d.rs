//! Port of `cartographer/mapping/internal/2d/overlapping_submaps_trimmer_2d` (.h, .cc).

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, PoisonError, RwLock};

use nalgebra::{Vector2, Vector3};

use crate::common::math::pow2;
use crate::common::port::round_to_int64;
use crate::common::time::Time;
use crate::mapping::id::{MapById, NodeId, SubmapId};
use crate::mapping::pose_graph_interface::{Constraint, SubmapData, Tag};
use crate::mapping::pose_graph_trimmer::{PoseGraphTrimmer, Trimmable};
use crate::mapping::submaps::Submap;
use crate::mapping::trajectory_node::TrajectoryNode;
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::submap_2d::Submap2D;
use crate::mapping::two_d::xy_index::XYIndexRangeIterator;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::project_2d;

/// Runs `f` on the `Submap2D` behind `submap` (C++
/// `std::static_pointer_cast<const Submap2D>`). Panics if the submap is not a
/// `Submap2D`.
fn with_submap_2d<R>(submap: &Arc<RwLock<dyn Submap>>, f: impl FnOnce(&Submap2D) -> R) -> R {
    let guard = submap.read().unwrap_or_else(PoisonError::into_inner);
    let any: &dyn Any = &*guard;
    f(any
        .downcast_ref::<Submap2D>()
        .expect("OverlappingSubmapsTrimmer2D requires Submap2D submaps"))
}

fn submap_of(submap_data: &SubmapData) -> &Arc<RwLock<dyn Submap>> {
    submap_data
        .submap
        .as_ref()
        .expect("SubmapData without submap")
}

// (x cells, y cells).
type CellId = (i64, i64);
type StoredType = Vec<(SubmapId, Time)>;

struct SubmapCoverageGrid2D {
    offset: Vector2<f64>,
    resolution: f64,
    cells: BTreeMap<CellId, StoredType>,
}

impl SubmapCoverageGrid2D {
    fn new(map_limits: &MapLimits) -> Self {
        Self {
            offset: *map_limits.max(),
            resolution: map_limits.resolution(),
            cells: BTreeMap::new(),
        }
    }

    fn add_point(&mut self, point: &Vector2<f64>, submap_id: &SubmapId, time: Time) {
        let cell_id: CellId = (
            round_to_int64((self.offset[0] - point[0]) / self.resolution),
            round_to_int64((self.offset[1] - point[1]) / self.resolution),
        );
        self.cells
            .entry(cell_id)
            .or_default()
            .push((*submap_id, time));
    }

    fn cells(&self) -> &BTreeMap<CellId, StoredType> {
        &self.cells
    }

    fn resolution(&self) -> f64 {
        self.resolution
    }
}

// Iterates over every cell in a submap, transforms the center of the cell to
// the global frame and then adds the submap id and the timestamp of the most
// recent range data insertion into the global grid.
fn add_submaps_to_submap_coverage_grid_2d(
    submap_freshness: &BTreeMap<SubmapId, Time>,
    submap_data: &MapById<SubmapId, SubmapData>,
    coverage_grid: &mut SubmapCoverageGrid2D,
) -> BTreeSet<SubmapId> {
    let mut all_submap_ids = BTreeSet::new();

    for submap in submap_data {
        let Some(&freshness) = submap_freshness.get(&submap.id) else {
            continue;
        };
        let submap_handle = submap_of(submap.data);
        with_submap_2d(submap_handle, |submap_2d| {
            if !submap_2d.insertion_finished() {
                return;
            }
            all_submap_ids.insert(submap.id);
            let grid = submap_2d.grid();
            // Iterate over every cell in a submap.
            let (offset, cell_limits) = grid.compute_cropped_limits();
            if cell_limits.num_x_cells == 0 || cell_limits.num_y_cells == 0 {
                log::warn!("Empty grid found in submap ID = {}", submap.id);
                return;
            }

            let global_frame_from_submap_frame = &submap.data.pose;
            let submap_frame_from_local_frame = submap_2d.local_pose().inverse();
            for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
                let index = xy_index + offset;
                if !grid.is_known(&index) {
                    continue;
                }

                let center_of_cell_in_local_frame = Rigid3d::from_translation(Vector3::new(
                    grid.limits().max().x - grid.limits().resolution() * (f64::from(index.y) + 0.5),
                    grid.limits().max().y - grid.limits().resolution() * (f64::from(index.x) + 0.5),
                    0.,
                ));

                let center_of_cell_in_global_frame = project_2d(
                    &(*global_frame_from_submap_frame
                        * submap_frame_from_local_frame
                        * center_of_cell_in_local_frame),
                );
                coverage_grid.add_point(
                    center_of_cell_in_global_frame.translation(),
                    &submap.id,
                    freshness,
                );
            }
        });
    }
    all_submap_ids
}

// Uses intra-submap constraints and trajectory node timestamps to identify
// time of the last range data insertion to the submap.
fn compute_submap_freshness(
    submap_data: &MapById<SubmapId, SubmapData>,
    trajectory_nodes: &MapById<NodeId, TrajectoryNode>,
    constraints: &[Constraint],
) -> BTreeMap<SubmapId, Time> {
    let mut submap_freshness = BTreeMap::new();

    // Find the node with the largest NodeId per SubmapId.
    let mut submap_to_latest_node: BTreeMap<SubmapId, NodeId> = BTreeMap::new();
    for constraint in constraints {
        if constraint.tag != Tag::IntraSubmap {
            continue;
        }
        submap_to_latest_node
            .entry(constraint.submap_id)
            .and_modify(|node_id| *node_id = (*node_id).max(constraint.node_id))
            .or_insert(constraint.node_id);
    }

    // Find timestamp of every latest node.
    for (submap_id, node_id) in &submap_to_latest_node {
        if !submap_data.contains(submap_id) {
            log::warn!(
                "Intra-submap constraint between SubmapID = {submap_id} and NodeID {node_id} is \
                 missing submap data"
            );
            continue;
        }
        let Some(latest_node) = trajectory_nodes.get(node_id) else {
            continue;
        };
        submap_freshness.insert(*submap_id, latest_node.time());
    }
    submap_freshness
}

// Returns IDs of submaps that have less than 'min_covered_cells_count' cells
// not overlapped by at least 'fresh_submaps_count' submaps.
fn find_submap_ids_to_trim(
    coverage_grid: &SubmapCoverageGrid2D,
    all_submap_ids: &BTreeSet<SubmapId>,
    fresh_submaps_count: u16,
    min_covered_cells_count: u16,
) -> Vec<SubmapId> {
    let mut submap_to_covered_cells_count: BTreeMap<SubmapId, u16> = BTreeMap::new();
    for cell in coverage_grid.cells().values() {
        let mut submaps_per_cell = cell.clone();

        // In case there are several submaps covering the cell, only the
        // freshest submaps are kept.
        if submaps_per_cell.len() > usize::from(fresh_submaps_count) {
            // Sort by time in descending order.
            // RUST PORT NOTE: C++ uses the unstable `std::sort`, so the order
            // of equally fresh submaps is unspecified; this stable sort is
            // one of the allowed outcomes.
            submaps_per_cell.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            submaps_per_cell.truncate(usize::from(fresh_submaps_count));
        }
        for (submap_id, _) in &submaps_per_cell {
            let count = submap_to_covered_cells_count.entry(*submap_id).or_insert(0);
            *count = count.wrapping_add(1);
        }
    }
    let submap_ids_to_keep: BTreeSet<SubmapId> = submap_to_covered_cells_count
        .iter()
        .filter(|&(_, &count)| count >= min_covered_cells_count)
        .map(|(id, _)| *id)
        .collect();

    all_submap_ids
        .difference(&submap_ids_to_keep)
        .copied()
        .collect()
}

/// Trims submaps that have less than `min_covered_cells_count` cells not
/// overlapped by at least `fresh_submaps_count` submaps.
#[derive(Debug)]
pub struct OverlappingSubmapsTrimmer2D {
    // Number of the most recent submaps to keep.
    fresh_submaps_count: u16,
    // Minimum area of covered space to keep submap from trimming measured in
    // m^2.
    min_covered_area: f64,
    // Number of added submaps before the trimmer is invoked.
    min_added_submaps_count: u16,
    // Current finished submap count.
    current_submap_count: u16,

    finished: bool,
}

impl OverlappingSubmapsTrimmer2D {
    /// Creates the trimmer.
    pub fn new(
        fresh_submaps_count: u16,
        min_covered_area: f64,
        min_added_submaps_count: u16,
    ) -> Self {
        Self {
            fresh_submaps_count,
            min_covered_area,
            min_added_submaps_count,
            current_submap_count: 0,
            finished: false,
        }
    }
}

impl PoseGraphTrimmer for OverlappingSubmapsTrimmer2D {
    fn trim(&mut self, pose_graph: &mut dyn Trimmable) {
        let submap_data = pose_graph.get_optimized_submap_data();
        // RUST PORT NOTE: `size_t - uint16` wraps around in C++.
        if submap_data
            .size()
            .wrapping_sub(usize::from(self.current_submap_count))
            <= usize::from(self.min_added_submaps_count)
        {
            return;
        }

        let first_submap_map_limits =
            with_submap_2d(submap_of(submap_data.begin().data()), |submap_2d| {
                *submap_2d.grid().limits()
            });
        let mut coverage_grid = SubmapCoverageGrid2D::new(&first_submap_map_limits);
        let submap_freshness = compute_submap_freshness(
            &submap_data,
            pose_graph.get_trajectory_nodes(),
            pose_graph.get_constraints(),
        );
        let all_submap_ids = add_submaps_to_submap_coverage_grid_2d(
            &submap_freshness,
            &submap_data,
            &mut coverage_grid,
        );
        // RUST PORT NOTE: the C++ implicitly converts the double to `uint16`
        // (truncation); `as` truncates as well and saturates out of range.
        let min_covered_cells_count =
            (self.min_covered_area / pow2(coverage_grid.resolution())) as u16;
        let submap_ids_to_remove = find_submap_ids_to_trim(
            &coverage_grid,
            &all_submap_ids,
            self.fresh_submaps_count,
            min_covered_cells_count,
        );
        self.current_submap_count = (submap_data.size() - submap_ids_to_remove.len()) as u16;
        for id in &submap_ids_to_remove {
            pose_graph.trim_submap(id);
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    mod overlapping_submaps_trimmer_2d_test {
        use std::sync::{Arc, RwLock};

        use nalgebra::{DVector, Vector2};

        use crate::common::time::from_universal;
        use crate::mapping::id::{NodeId, SubmapId};
        use crate::mapping::internal::testing::fake_trimmable::FakeTrimmable;
        use crate::mapping::internal::two_d::overlapping_submaps_trimmer_2d::*;
        use crate::mapping::pose_graph_interface::{Constraint, Pose, SubmapData, Tag};
        use crate::mapping::proto;
        use crate::mapping::trajectory_node::{TrajectoryNode, TrajectoryNodeData};
        use crate::mapping::two_d::submap_2d::Submap2D;
        use crate::mapping::value_conversion_tables::ValueConversionTables;
        use crate::sensor::point_cloud::PointCloud;
        use crate::transform::rigid_transform::{Rigid2d, Rigid3d, quaternion_identity};
        use crate::transform::transform::{embed_3d, to_proto};

        struct Fixture {
            conversion_tables: Arc<ValueConversionTables>,
            fake_pose_graph: FakeTrimmable,
        }

        impl Fixture {
            fn new() -> Self {
                Self {
                    conversion_tables: Arc::new(ValueConversionTables::new()),
                    fake_pose_graph: FakeTrimmable::new(),
                }
            }

            // Creates a submap with num_cells x num_cells grid.
            fn add_square_submap(
                &mut self,
                global_to_submap_frame: &Rigid2d,
                local_to_submap_frame: &Rigid2d,
                submap_corner: &Vector2<f64>,
                submap_index: i32,
                num_cells: i32,
                is_finished: bool,
            ) {
                let grid = proto::Grid2D {
                    cells: vec![1; (num_cells * num_cells) as usize],
                    limits: Some(proto::MapLimits {
                        resolution: 1.0,
                        max: Some(to_proto(submap_corner)),
                        cell_limits: Some(proto::CellLimits {
                            num_x_cells: num_cells,
                            num_y_cells: num_cells,
                        }),
                    }),
                    known_cells_box: Some(proto::grid2_d::CellBox {
                        min_x: 0,
                        min_y: 0,
                        max_x: num_cells - 1,
                        max_y: num_cells - 1,
                    }),
                    grid: Some(proto::grid2_d::Grid::ProbabilityGrid2d(
                        proto::ProbabilityGrid::default(),
                    )),
                    ..Default::default()
                };
                let submap_2d = proto::Submap2D {
                    num_range_data: 1,
                    finished: is_finished,
                    local_pose: Some(to_proto(&embed_3d(local_to_submap_frame))),
                    grid: Some(grid),
                };
                self.fake_pose_graph.mutable_submap_data().insert(
                    &SubmapId::new(0 /* trajectory_id */, submap_index),
                    SubmapData {
                        submap: Some(Arc::new(RwLock::new(Submap2D::from_proto(
                            &submap_2d,
                            &self.conversion_tables,
                        )))),
                        pose: embed_3d(global_to_submap_frame),
                    },
                );
            }

            fn add_trajectory_node(&mut self, node_index: i32, timestamp: i64) {
                // C++ default-constructs `TrajectoryNode::Data` and sets the
                // time only.
                let data = TrajectoryNodeData {
                    time: from_universal(timestamp),
                    gravity_alignment: quaternion_identity(),
                    filtered_gravity_aligned_point_cloud: PointCloud::new(),
                    high_resolution_point_cloud: PointCloud::new(),
                    low_resolution_point_cloud: PointCloud::new(),
                    rotational_scan_matcher_histogram: DVector::zeros(0),
                    local_pose: Rigid3d::identity(),
                };
                self.fake_pose_graph.mutable_trajectory_nodes().insert(
                    &NodeId::new(0 /* trajectory_id */, node_index),
                    TrajectoryNode::new(Arc::new(data), Rigid3d::default() /* pose */),
                );
            }

            fn add_constraint(
                &mut self,
                submap_index: i32,
                node_index: i32,
                is_intra_submap: bool,
            ) {
                self.fake_pose_graph.mutable_constraints().push(Constraint {
                    submap_id: SubmapId::new(0 /* trajectory_id */, submap_index),
                    node_id: NodeId::new(0 /* trajectory_id */, node_index),
                    pose: Pose {
                        zbar_ij: Rigid3d::default(),
                        translation_weight: 0.,
                        rotation_weight: 0.,
                    },
                    tag: if is_intra_submap {
                        Tag::IntraSubmap
                    } else {
                        Tag::InterSubmap
                    },
                });
            }

            fn add_two_unit_submaps_and_nodes(&mut self) {
                self.add_square_submap(
                    &Rigid2d::identity(),  /* global_from_submap_frame */
                    &Rigid2d::identity(),  /* local_from_submap_frame */
                    &Vector2::new(1., 1.), /* submap corner */
                    0,                     /* submap_index */
                    1,                     /* num_cells */
                    true,                  /* is_finished */
                );
                self.add_square_submap(
                    &Rigid2d::identity(),
                    &Rigid2d::identity(),
                    &Vector2::new(1., 1.),
                    1,
                    1,
                    true,
                );
                self.add_trajectory_node(0, 1000);
                self.add_trajectory_node(1, 2000);
                self.add_constraint(0, 0, true);
                self.add_constraint(1, 1, true);
            }
        }

        #[test]
        fn empty_pose_graph() {
            let mut f = Fixture::new();
            let mut trimmer = OverlappingSubmapsTrimmer2D::new(
                1,  /* fresh_submaps_count */
                0., /* min_covered_area */
                0,  /* min_added_submaps_count */
            );
            trimmer.trim(&mut f.fake_pose_graph);
            assert!(f.fake_pose_graph.trimmed_submaps().is_empty());
        }

        #[test]
        fn trim_one_of_two_overlapping_submaps() {
            let mut f = Fixture::new();
            f.add_two_unit_submaps_and_nodes();

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(1, 0., 0);
            trimmer.trim(&mut f.fake_pose_graph);
            assert_eq!(
                f.fake_pose_graph.trimmed_submaps(),
                vec![SubmapId::new(0, 0)]
            );
        }

        #[test]
        fn test_min_added_submaps_count_param() {
            let mut f = Fixture::new();
            f.add_two_unit_submaps_and_nodes();

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(1, 0., 2);
            trimmer.trim(&mut f.fake_pose_graph);
            assert!(f.fake_pose_graph.trimmed_submaps().is_empty());

            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                2,
                1,
                true,
            );
            f.add_trajectory_node(2, 3000);
            f.add_constraint(2, 2, true);
            trimmer.trim(&mut f.fake_pose_graph);
            assert_eq!(
                f.fake_pose_graph.trimmed_submaps(),
                vec![SubmapId::new(0, 0), SubmapId::new(0, 1)]
            );
        }

        #[test]
        fn do_not_trim_unfinished_submap() {
            let mut f = Fixture::new();
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                0,
                1,
                false, /* is_finished */
            );
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                1,
                1,
                true,
            );
            f.add_trajectory_node(0, 1000);
            f.add_trajectory_node(1, 2000);
            f.add_constraint(0, 0, true);
            f.add_constraint(1, 1, true);

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(1, 0., 0);
            trimmer.trim(&mut f.fake_pose_graph);
            assert!(f.fake_pose_graph.trimmed_submaps().is_empty());
        }

        #[test]
        fn use_only_intra_submaps_to_compute_freshness() {
            let mut f = Fixture::new();
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                0,
                1,
                true,
            );
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                1,
                1,
                true,
            );
            f.add_trajectory_node(0, 1000);
            f.add_trajectory_node(1, 2000);
            f.add_trajectory_node(2, 3000);
            f.add_constraint(0, 0, false);
            f.add_constraint(0, 2, true);
            f.add_constraint(1, 1, true);

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(1, 0., 0);
            trimmer.trim(&mut f.fake_pose_graph);
            assert_eq!(
                f.fake_pose_graph.trimmed_submaps(),
                vec![SubmapId::new(0, 1)]
            );
        }

        // This test covers two 4-cells submaps that overlap each other
        // displaced like:
        //    ___
        //  _|   |
        // | |_ _|
        // |___|
        //
        // The background submap should be trimmed, since it has only 3 cells
        // not-covered by another submap.
        #[test]
        fn trim_submap_with_low_covered_cells_count() {
            let mut f = Fixture::new();
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                0,
                2,
                true,
            );
            f.add_square_submap(
                &Rigid2d::from_translation(Vector2::new(1., 1.)),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                1,
                2,
                true,
            );
            f.add_trajectory_node(0, 1000);
            f.add_trajectory_node(1, 2000);
            f.add_constraint(0, 0, true);
            f.add_constraint(1, 1, true);

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(
                1,  /* fresh_submaps_count */
                4., /* min_covered_cells_count */
                0,  /* min_added_submaps_count */
            );
            trimmer.trim(&mut f.fake_pose_graph);
            assert_eq!(
                f.fake_pose_graph.trimmed_submaps(),
                vec![SubmapId::new(0, 0)]
            );
        }

        #[test]
        fn test_transformations() {
            let mut f = Fixture::new();
            f.add_square_submap(
                &Rigid2d::identity(),
                &Rigid2d::identity(),
                &Vector2::new(1., 1.),
                0,
                1,
                true,
            );
            let transform = Rigid2d::new(Vector2::new(1., 1.), std::f64::consts::FRAC_PI_2);
            f.add_square_submap(
                &transform, /* global_from_submap_frame */
                &transform, /* local_from_submap_frame */
                &Vector2::new(1., 1.),
                1,
                1,
                true,
            );
            f.add_trajectory_node(0, 1000);
            f.add_trajectory_node(1, 2000);
            f.add_constraint(0, 0, true);
            f.add_constraint(1, 1, true);

            let mut trimmer = OverlappingSubmapsTrimmer2D::new(1, 0., 0);
            trimmer.trim(&mut f.fake_pose_graph);
            assert_eq!(
                f.fake_pose_graph.trimmed_submaps(),
                vec![SubmapId::new(0, 0)]
            );
        }
    }
}
