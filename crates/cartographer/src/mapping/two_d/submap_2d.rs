//! Port of `cartographer/mapping/2d/submap_2d` (.h, .cc).
//!
//! RUST PORT NOTE: ownership. C++ `Submap2D` holds a
//! `std::unique_ptr<Grid2D>` and a raw `ValueConversionTables*` owned by
//! `ActiveSubmaps2D`; here the grid is a `Box<dyn Grid2DTrait>` and the
//! tables are shared via `Arc<ValueConversionTables>`. `ActiveSubmaps2D`
//! hands out [`SharedSubmap<Submap2D>`] (`Arc<RwLock<Submap2D>>`) handles
//! instead of `std::shared_ptr<const Submap2D>`; it takes a write lock to
//! insert, readers take read locks (see `mapping/submaps.rs`).

use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use nalgebra::{Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::mapping::internal::two_d::tsdf_range_data_inserter_2d::TSDFRangeDataInserter2D;
use crate::mapping::proto;
use crate::mapping::proto::grid_options2_d::GridType as ProtoGridType;
use crate::mapping::proto::range_data_inserter_options::RangeDataInserterType;
use crate::mapping::proto::submap_query::response::SubmapTexture;
use crate::mapping::range_data_inserter_interface::{
    RangeDataInserterInterface, create_range_data_inserter_options,
};
use crate::mapping::submaps::{SharedSubmap, Submap, SubmapBase};
use crate::mapping::two_d::grid_2d::{Grid2DTrait, create_grid_options_2d};
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::probability_grid::ProbabilityGrid;
use crate::mapping::two_d::probability_grid_range_data_inserter_2d::ProbabilityGridRangeDataInserter2D;
use crate::mapping::two_d::xy_index::CellLimits;
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::sensor::range_data::RangeData;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::{to_proto, to_rigid3};

/// Reads `SubmapsOptions2D` from `parameter_dictionary`. Panics unless
/// `num_range_data > 0` and the grid type matches the range data inserter
/// type (`PROBABILITY_GRID` + `PROBABILITY_GRID_INSERTER_2D` or `TSDF` +
/// `TSDF_INSERTER_2D`).
pub fn create_submaps_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::SubmapsOptions2D {
    let options = proto::SubmapsOptions2D {
        num_range_data: parameter_dictionary.get_non_negative_int("num_range_data"),
        grid_options_2d: Some(create_grid_options_2d(
            &parameter_dictionary.get_dictionary("grid_options_2d"),
        )),
        range_data_inserter_options: Some(create_range_data_inserter_options(
            &parameter_dictionary.get_dictionary("range_data_inserter"),
        )),
    };

    let grid_type = options.grid_options_2d.unwrap_or_default().grid_type();
    let range_data_inserter_type = options
        .range_data_inserter_options
        .unwrap_or_default()
        .range_data_inserter_type();
    let valid_range_data_inserter_grid_combination = matches!(
        (grid_type, range_data_inserter_type),
        (
            ProtoGridType::ProbabilityGrid,
            RangeDataInserterType::ProbabilityGridInserter2d
        ) | (ProtoGridType::Tsdf, RangeDataInserterType::TsdfInserter2d)
    );
    assert!(
        valid_range_data_inserter_grid_combination,
        "Invalid combination grid_type {} with range_data_inserter_type {}",
        grid_type.as_str_name(),
        range_data_inserter_type.as_str_name()
    );
    assert!(
        options.num_range_data > 0,
        "Check failed: options.num_range_data() > 0 ({} vs. 0)",
        options.num_range_data
    );
    options
}

/// Creates the grid stored in a `proto::Grid2D` (probability grid or TSDF).
fn grid_from_proto(
    grid: &proto::Grid2D,
    conversion_tables: &Arc<ValueConversionTables>,
) -> Box<dyn Grid2DTrait> {
    match &grid.grid {
        Some(proto::grid2_d::Grid::ProbabilityGrid2d(_)) => {
            Box::new(ProbabilityGrid::from_proto(grid, conversion_tables))
        }
        Some(proto::grid2_d::Grid::Tsdf2d(_)) => {
            Box::new(TSDF2D::from_proto(grid, conversion_tables))
        }
        None => panic!("proto::Submap2D has grid with unknown type."),
    }
}

/// A 2D submap: a grid (probability grid or TSDF) plus the common submap
/// state (local pose, number of inserted range data, finished flag).
pub struct Submap2D {
    base: SubmapBase,
    // `None` only for submaps deserialized without grid data (C++ null
    // `grid_`).
    grid: Option<Box<dyn Grid2DTrait>>,
    conversion_tables: Arc<ValueConversionTables>,
}

impl fmt::Debug for Submap2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submap2D")
            .field("base", &self.base)
            .field("grid_type", &self.grid.as_ref().map(|g| g.get_grid_type()))
            .field("limits", &self.grid.as_ref().map(|g| *g.limits()))
            .finish_non_exhaustive()
    }
}

impl Submap2D {
    /// Creates a submap at `origin` (local pose = translation
    /// `(origin.x, origin.y, 0)`) with the given (empty) grid.
    pub fn new(
        origin: &Vector2<f32>,
        grid: Box<dyn Grid2DTrait>,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Self {
        Self {
            base: SubmapBase::new(Rigid3d::from_translation(Vector3::new(
                f64::from(origin.x),
                f64::from(origin.y),
                0.0,
            ))),
            grid: Some(grid),
            conversion_tables: Arc::clone(conversion_tables),
        }
    }

    /// Creates a submap from a proto. The grid is only created if present in
    /// the proto. Panics on grids of unknown type.
    pub fn from_proto(
        proto: &proto::Submap2D,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Self {
        let grid = proto
            .grid
            .as_ref()
            .map(|grid| grid_from_proto(grid, conversion_tables));
        let mut submap = Self {
            base: SubmapBase::new(to_rigid3(&proto.local_pose.unwrap_or_default())),
            grid,
            conversion_tables: Arc::clone(conversion_tables),
        };
        submap.set_num_range_data(proto.num_range_data);
        submap.set_insertion_finished(proto.finished);
        submap
    }

    /// The grid. Panics if the submap has no grid (C++ dereferences a null
    /// `grid()` in that case); see [`try_grid`](Self::try_grid).
    pub fn grid(&self) -> &dyn Grid2DTrait {
        self.grid
            .as_deref()
            .expect("Submap2D has no grid (deserialized without grid data)")
    }

    /// The grid, or `None` if the submap was deserialized without grid data
    /// (C++ `grid()` returning `nullptr`).
    pub fn try_grid(&self) -> Option<&dyn Grid2DTrait> {
        self.grid.as_deref()
    }

    /// The conversion tables shared with the grid.
    pub fn conversion_tables(&self) -> &Arc<ValueConversionTables> {
        &self.conversion_tables
    }

    /// Inserts `range_data` into this submap using `range_data_inserter`.
    /// The submap must not be finished yet.
    pub fn insert_range_data(
        &mut self,
        range_data: &RangeData,
        range_data_inserter: &dyn RangeDataInserterInterface,
    ) {
        assert!(self.grid.is_some(), "Check failed: grid_");
        assert!(
            !self.insertion_finished(),
            "Check failed: !insertion_finished()"
        );
        let grid = self.grid.as_deref_mut().expect("Check failed: grid_");
        range_data_inserter.insert(range_data, grid);
        let num_range_data = self.num_range_data() + 1;
        self.set_num_range_data(num_range_data);
    }

    /// Finishes the submap: replaces the grid by its cropped version and
    /// marks insertion as finished.
    pub fn finish(&mut self) {
        let grid = self.grid.as_deref().expect("Check failed: grid_");
        assert!(
            !self.insertion_finished(),
            "Check failed: !insertion_finished()"
        );
        self.grid = Some(grid.compute_cropped_grid());
        self.set_insertion_finished(true);
    }
}

impl Submap for Submap2D {
    fn base(&self) -> &SubmapBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut SubmapBase {
        &mut self.base
    }

    fn to_proto(&self, include_grid_data: bool) -> proto::Submap {
        let mut submap_2d = proto::Submap2D {
            local_pose: Some(to_proto(&self.local_pose())),
            num_range_data: self.num_range_data(),
            finished: self.insertion_finished(),
            grid: None,
        };
        if include_grid_data {
            let grid = self.grid.as_deref().expect("Check failed: grid_");
            submap_2d.grid = Some(grid.to_proto());
        }
        proto::Submap {
            submap_2d: Some(submap_2d),
            ..Default::default()
        }
    }

    fn update_from_proto(&mut self, proto: &proto::Submap) {
        let submap_2d = proto
            .submap_2d
            .as_ref()
            .expect("Check failed: proto.has_submap_2d()");
        self.set_num_range_data(submap_2d.num_range_data);
        self.set_insertion_finished(submap_2d.finished);
        if let Some(grid) = &submap_2d.grid {
            self.grid = Some(grid_from_proto(grid, &self.conversion_tables));
        }
    }

    fn to_response_proto(
        &self,
        _global_submap_pose: &Rigid3d,
        response: &mut proto::submap_query::Response,
    ) {
        let Some(grid) = self.grid.as_deref() else {
            return;
        };
        response.submap_version = self.num_range_data();
        response.textures.push(SubmapTexture::default());
        let texture = response
            .textures
            .last_mut()
            .expect("a texture was just added");
        grid.draw_to_submap_texture(texture, self.local_pose());
    }
}

/// Initial number of cells along each axis of a new submap grid.
const INITIAL_SUBMAP_SIZE: i32 = 100;

/// The first active submap will be created on the insertion of the first
/// range data. Except during this initialization when no or only one single
/// submap exists, there are always two submaps into which range data is
/// inserted: an old submap that is used for matching, and a new one, which
/// will be used for matching next, that is being initialized.
///
/// Once a certain number of range data have been inserted, the new submap is
/// considered initialized: the old submap is no longer changed, the "new"
/// submap is now the "old" submap and is used for scan-to-map matching.
/// Moreover, a "new" submap gets created. The "old" submap is forgotten by
/// this object.
pub struct ActiveSubmaps2D {
    options: proto::SubmapsOptions2D,
    submaps: Vec<SharedSubmap<Submap2D>>,
    range_data_inserter: Box<dyn RangeDataInserterInterface>,
    // RUST PORT NOTE: owned by value in C++ (grids keep raw pointers into
    // it); shared with the grids via `Arc` here.
    conversion_tables: Arc<ValueConversionTables>,
}

impl fmt::Debug for ActiveSubmaps2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActiveSubmaps2D")
            .field("options", &self.options)
            .field("num_submaps", &self.submaps.len())
            .finish_non_exhaustive()
    }
}

impl ActiveSubmaps2D {
    /// Creates an empty set of active submaps.
    pub fn new(options: &proto::SubmapsOptions2D) -> Self {
        Self {
            options: *options,
            submaps: Vec::new(),
            range_data_inserter: Self::create_range_data_inserter(options),
            conversion_tables: Arc::new(ValueConversionTables::new()),
        }
    }

    /// Inserts `range_data` into the submap collection and returns the
    /// active submaps (front = older submap used for matching).
    pub fn insert_range_data(&mut self, range_data: &RangeData) -> Vec<SharedSubmap<Submap2D>> {
        let num_range_data = self.options.num_range_data;
        let needs_new_submap = match self.submaps.last() {
            None => true,
            Some(back) => read(back).num_range_data() == num_range_data,
        };
        if needs_new_submap {
            self.add_submap(&range_data.origin.xy());
        }
        for submap in &self.submaps {
            write(submap).insert_range_data(range_data, self.range_data_inserter.as_ref());
        }
        let front = &self.submaps[0];
        let mut front = write(front);
        if front.num_range_data() == 2 * num_range_data {
            front.finish();
        }
        drop(front);
        self.submaps()
    }

    /// The active submaps (0, 1 or 2; front = older submap used for
    /// matching).
    pub fn submaps(&self) -> Vec<SharedSubmap<Submap2D>> {
        self.submaps.clone()
    }

    /// The conversion tables shared by all grids created by this object.
    pub fn conversion_tables(&self) -> &Arc<ValueConversionTables> {
        &self.conversion_tables
    }

    fn create_range_data_inserter(
        options: &proto::SubmapsOptions2D,
    ) -> Box<dyn RangeDataInserterInterface> {
        let range_data_inserter_options = options.range_data_inserter_options.unwrap_or_default();
        match range_data_inserter_options.range_data_inserter_type() {
            RangeDataInserterType::ProbabilityGridInserter2d => {
                Box::new(ProbabilityGridRangeDataInserter2D::new(
                    &range_data_inserter_options
                        .probability_grid_range_data_inserter_options_2d
                        .unwrap_or_default(),
                ))
            }
            RangeDataInserterType::TsdfInserter2d => Box::new(TSDFRangeDataInserter2D::new(
                &range_data_inserter_options
                    .tsdf_range_data_inserter_options_2d
                    .unwrap_or_default(),
            )),
            RangeDataInserterType::InvalidInserter => panic!("Unknown RangeDataInserterType."),
        }
    }

    fn create_grid(&self, origin: &Vector2<f32>) -> Box<dyn Grid2DTrait> {
        let grid_options_2d = self.options.grid_options_2d.unwrap_or_default();
        let resolution: f32 = grid_options_2d.resolution;
        let limits = MapLimits::new(
            f64::from(resolution),
            origin.cast::<f64>()
                + 0.5
                    * f64::from(INITIAL_SUBMAP_SIZE)
                    * f64::from(resolution)
                    * Vector2::<f64>::repeat(1.0),
            CellLimits::new(INITIAL_SUBMAP_SIZE, INITIAL_SUBMAP_SIZE),
        );
        match grid_options_2d.grid_type() {
            ProtoGridType::ProbabilityGrid => {
                Box::new(ProbabilityGrid::new(limits, &self.conversion_tables))
            }
            ProtoGridType::Tsdf => {
                let tsdf_options = self
                    .options
                    .range_data_inserter_options
                    .unwrap_or_default()
                    .tsdf_range_data_inserter_options_2d
                    .unwrap_or_default();
                Box::new(TSDF2D::new(
                    limits,
                    tsdf_options.truncation_distance as f32,
                    tsdf_options.maximum_weight as f32,
                    &self.conversion_tables,
                ))
            }
            ProtoGridType::InvalidGrid => panic!("Unknown GridType."),
        }
    }

    fn add_submap(&mut self, origin: &Vector2<f32>) {
        if self.submaps.len() >= 2 {
            // This will crop the finished Submap before inserting a new Submap
            // to reduce peak memory usage a bit.
            assert!(
                read(&self.submaps[0]).insertion_finished(),
                "Check failed: submaps_.front()->insertion_finished()"
            );
            self.submaps.remove(0);
        }
        let grid = self.create_grid(origin);
        self.submaps.push(Arc::new(RwLock::new(Submap2D::new(
            origin,
            grid,
            &self.conversion_tables,
        ))));
    }
}

#[inline]
fn read(submap: &SharedSubmap<Submap2D>) -> std::sync::RwLockReadGuard<'_, Submap2D> {
    submap.read().unwrap_or_else(PoisonError::into_inner)
}

#[inline]
fn write(submap: &SharedSubmap<Submap2D>) -> std::sync::RwLockWriteGuard<'_, Submap2D> {
    submap.write().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use std::sync::Arc;

    use nalgebra::{Vector2, Vector3};

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
    use crate::mapping::submaps::Submap;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::sensor::rangefinder_point::RangefinderPoint;

    fn submaps_options_lua(num_range_data: i32, grid_type: &str, inserter_type: &str) -> String {
        format!(
            "return {{\
             num_range_data = {num_range_data}, \
             grid_options_2d = {{\
             grid_type = \"{grid_type}\",\
             resolution = 0.05, \
             }},\
             range_data_inserter = {{\
             range_data_inserter_type = \"{inserter_type}\",\
             probability_grid_range_data_inserter = {{\
             insert_free_space = true, \
             hit_probability = 0.53, \
             miss_probability = 0.495, \
             }},\
             tsdf_range_data_inserter = {{ \
             truncation_distance = 0.3,\
             maximum_weight = 10.,\
             update_free_space = false,\
             normal_estimation_options = {{\
             num_normal_samples = 2,\
             sample_radius = 10.,\
             }},\
             project_sdf_distance_to_scan_normal = false,\
             update_weight_range_exponent = 0,\
             update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0,\
             update_weight_distance_cell_to_hit_kernel_bandwidth = 0,\
             }},\
             }},\
             }}"
        )
    }

    fn probability_grid_options(num_range_data: i32) -> proto::SubmapsOptions2D {
        create_submaps_options_2d(&make_dictionary(&submaps_options_lua(
            num_range_data,
            "PROBABILITY_GRID",
            "PROBABILITY_GRID_INSERTER_2D",
        )))
    }

    mod submap_2d_test {
        use super::*;

        #[test]
        fn the_right_number_of_range_data_are_inserted() {
            const NUM_RANGE_DATA: i32 = 10;
            // C++ uses truncation_distance = 2.0 in the (unused) TSDF block.
            let mut submaps = ActiveSubmaps2D::new(&probability_grid_options(NUM_RANGE_DATA));
            // C++ `std::set<std::shared_ptr<const Submap2D>>` (pointer
            // identity).
            let mut all_submaps: Vec<SharedSubmap<Submap2D>> = Vec::new();
            for _ in 0..1000 {
                let insertion_submaps = submaps.insert_range_data(&RangeData {
                    origin: Vector3::zeros(),
                    ..Default::default()
                });
                // Except for the first, maps should only be returned after
                // enough range data.
                for submap in insertion_submaps {
                    if !all_submaps.iter().any(|s| Arc::ptr_eq(s, &submap)) {
                        all_submaps.push(submap);
                    }
                }
                if submaps.submaps().len() > 1 {
                    assert!(NUM_RANGE_DATA <= read(&submaps.submaps()[0]).num_range_data());
                }
            }
            assert_eq!(2, submaps.submaps().len());
            let mut correct_num_finished_submaps = 0;
            let mut num_unfinished_submaps = 0;
            for submap in &all_submaps {
                let num_range_data = read(submap).num_range_data();
                if num_range_data == NUM_RANGE_DATA * 2 {
                    correct_num_finished_submaps += 1;
                } else {
                    assert_eq!(NUM_RANGE_DATA, num_range_data);
                    num_unfinished_submaps += 1;
                }
            }
            // Submaps should not be left without the right number of range
            // data in them.
            assert_eq!(correct_num_finished_submaps, all_submaps.len() - 1);
            assert_eq!(1, num_unfinished_submaps);
        }

        #[test]
        fn to_from_proto() {
            let expected_map_limits =
                MapLimits::new(1.0, Vector2::new(2.0, 3.0), CellLimits::new(100, 110));
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let expected = Submap2D::new(
                &Vector2::new(4.0, 5.0),
                Box::new(ProbabilityGrid::new(
                    expected_map_limits,
                    &conversion_tables,
                )),
                &conversion_tables,
            );
            let proto = expected.to_proto(true /* include_probability_grid_data */);
            assert!(proto.submap_2d.is_some());
            assert!(proto.submap_3d.is_none());
            let actual =
                Submap2D::from_proto(proto.submap_2d.as_ref().unwrap(), &conversion_tables);
            // Eigen `isApprox(other, prec)`: |a - b| <= prec * min(|a|, |b|).
            let is_approx_3 = |a: &Vector3<f64>, b: &Vector3<f64>| {
                (a - b).norm() <= 1e-6 * a.norm().min(b.norm())
            };
            assert!(is_approx_3(
                expected.local_pose().translation(),
                actual.local_pose().translation()
            ));
            let (er, ar) = (
                expected.local_pose().rotation().coords,
                actual.local_pose().rotation().coords,
            );
            assert!((er - ar).norm() <= 1e-6 * er.norm().min(ar.norm()));
            assert_eq!(expected.num_range_data(), actual.num_range_data());
            assert_eq!(expected.insertion_finished(), actual.insertion_finished());
            assert!(
                (expected.grid().limits().resolution() - actual.grid().limits().resolution()).abs()
                    <= 1e-6
            );
            let (em, am) = (expected.grid().limits().max(), actual.grid().limits().max());
            assert!((em - am).norm() <= 1e-6 * em.norm().min(am.norm()));
            assert_eq!(
                expected.grid().limits().cell_limits().num_x_cells,
                actual.grid().limits().cell_limits().num_x_cells
            );
        }
    }

    // Additional tests (not in C++).

    /// One scan of a square room with walls at `x, y = ±half_size` around the
    /// origin `(0, 0)`, one ray per `angle_step`.
    fn square_room_scan(half_size: f32, num_rays: usize) -> RangeData {
        let mut range_data = RangeData::default();
        for i in 0..num_rays {
            let angle = -PI + 2.0 * PI * i as f32 / num_rays as f32;
            let direction = Vector2::new(angle.cos(), angle.sin());
            let range = half_size / direction.x.abs().max(direction.y.abs());
            let hit = direction * range;
            range_data.returns.push_back(RangefinderPoint {
                position: Vector3::new(hit.x, hit.y, 0.0),
            });
        }
        range_data
    }

    fn probability_at(submap: &Submap2D, x: f32, y: f32) -> Option<f32> {
        let grid = submap.grid().downcast_ref::<ProbabilityGrid>().unwrap();
        let cell = grid.limits().get_cell_index(&Vector2::new(x, y));
        grid.is_known(&cell).then(|| grid.get_probability(&cell))
    }

    #[test]
    fn square_room_has_occupied_walls_and_free_interior() {
        const HALF_SIZE: f32 = 2.0;
        let options = create_submaps_options_2d(&make_dictionary(
            &submaps_options_lua(5, "PROBABILITY_GRID", "PROBABILITY_GRID_INSERTER_2D")
                .replace("hit_probability = 0.53", "hit_probability = 0.55")
                .replace("miss_probability = 0.495", "miss_probability = 0.49"),
        ));
        let mut active_submaps = ActiveSubmaps2D::new(&options);
        let scan = square_room_scan(HALF_SIZE, 720);
        let mut insertion_submaps = Vec::new();
        for _ in 0..5 {
            insertion_submaps = active_submaps.insert_range_data(&scan);
        }
        assert_eq!(insertion_submaps.len(), 1);
        let submap = read(&insertion_submaps[0]);
        assert_eq!(submap.num_range_data(), 5);
        // 100 cells of 0.05 m cover only [-2.5, 2.5]; the walls are inside.
        assert_eq!(submap.grid().limits().cell_limits().num_x_cells, 100);

        // Every hit cell is occupied: 5 hits at 0.55 -> odds 1.2222^5.
        for hit in scan.returns.iter() {
            let p = probability_at(&submap, hit.position.x, hit.position.y)
                .expect("wall cell must be known");
            assert!(p > 0.7, "wall at {:?}: {p}", hit.position);
        }
        // Interior cells along the rays are free: 5 misses at 0.49.
        for i in 0..72 {
            let angle = -PI + 2.0 * PI * i as f32 / 72.0;
            for r in [0.1_f32, 0.5, 1.0, 1.5, 1.8] {
                let p = probability_at(&submap, r * angle.cos(), r * angle.sin())
                    .expect("interior cell must be known");
                assert!(p < 0.47, "interior at angle {angle} r {r}: {p}");
            }
        }
        // Beyond the walls nothing is known.
        for (x, y) in [(2.3, 0.0), (-2.3, 0.0), (0.0, 2.3), (0.0, -2.3)] {
            assert!(probability_at(&submap, x, y).is_none(), "({x}, {y})");
        }
    }

    #[test]
    fn square_room_tsdf_has_zero_crossing_at_walls() {
        let options = create_submaps_options_2d(&make_dictionary(&submaps_options_lua(
            5,
            "TSDF",
            "TSDF_INSERTER_2D",
        )));
        let mut active_submaps = ActiveSubmaps2D::new(&options);
        let scan = square_room_scan(2.0, 720);
        let submaps = active_submaps.insert_range_data(&scan);
        let submap = read(&submaps[0]);
        let tsdf = submap.grid().downcast_ref::<TSDF2D>().unwrap();
        // Cell centers next to the wall x = 2 along the x axis (5 cm cells).
        let tsd_at = |x: f32, y: f32| {
            let cell = tsdf.limits().get_cell_index(&Vector2::new(x, y));
            assert!(tsdf.is_known(&cell), "({x}, {y})");
            tsdf.get_tsd(&cell)
        };
        assert!(tsd_at(1.975, 0.025).abs() < 0.05);
        assert!((tsd_at(1.825, 0.025) - 0.175).abs() < 0.05);
        assert!((tsd_at(2.175, 0.025) + 0.175).abs() < 0.05);
    }

    #[test]
    fn submap_2d_proto_round_trip_with_data() {
        let conversion_tables = Arc::new(ValueConversionTables::new());
        let options = proto::ProbabilityGridRangeDataInserterOptions2D {
            hit_probability: 0.55,
            miss_probability: 0.49,
            insert_free_space: true,
        };
        let inserter = ProbabilityGridRangeDataInserter2D::new(&options);
        let mut submap = Submap2D::new(
            &Vector2::new(0.0, 0.0),
            Box::new(ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(1.0, 1.0), CellLimits::new(40, 40)),
                &conversion_tables,
            )),
            &conversion_tables,
        );
        let scan = square_room_scan(2.0, 360);
        submap.insert_range_data(&scan, &inserter);
        submap.insert_range_data(&scan, &inserter);
        let unfinished_proto = submap.to_proto(true);
        submap.finish();
        assert!(submap.insertion_finished());
        let proto = submap.to_proto(true);

        let actual = Submap2D::from_proto(proto.submap_2d.as_ref().unwrap(), &conversion_tables);
        assert_eq!(actual.num_range_data(), 2);
        assert!(actual.insertion_finished());
        assert_eq!(actual.to_proto(true), proto);
        assert_eq!(actual.grid().to_proto(), submap.grid().to_proto());

        // Without grid data.
        let no_grid = Submap2D::from_proto(
            submap.to_proto(false).submap_2d.as_ref().unwrap(),
            &conversion_tables,
        );
        assert!(no_grid.try_grid().is_none());
        let mut response = proto::submap_query::Response::default();
        no_grid.to_response_proto(&Rigid3d::identity(), &mut response);
        assert!(response.textures.is_empty());

        // `update_from_proto` replaces counters and grid.
        let mut updated = Submap2D::from_proto(
            unfinished_proto.submap_2d.as_ref().unwrap(),
            &conversion_tables,
        );
        assert!(!updated.insertion_finished());
        updated.update_from_proto(&proto);
        assert!(updated.insertion_finished());
        assert_eq!(updated.to_proto(true), proto);

        // Response texture.
        let mut response = proto::submap_query::Response::default();
        submap.to_response_proto(&Rigid3d::identity(), &mut response);
        assert_eq!(response.submap_version, 2);
        assert_eq!(response.textures.len(), 1);
        let (_, cropped) = submap.grid().compute_cropped_limits();
        assert_eq!(response.textures[0].width, cropped.num_x_cells);
        assert_eq!(response.textures[0].height, cropped.num_y_cells);
    }

    #[test]
    fn active_submaps_lifecycle() {
        const N: i32 = 3;
        let mut active_submaps = ActiveSubmaps2D::new(&probability_grid_options(N));
        assert!(active_submaps.submaps().is_empty());
        let scan = square_room_scan(1.0, 36);
        // (num_range_data, finished) of the active submaps after each insert.
        let expected: Vec<Vec<(i32, bool)>> = vec![
            vec![(1, false)],
            vec![(2, false)],
            vec![(3, false)],
            vec![(4, false), (1, false)],
            vec![(5, false), (2, false)],
            vec![(6, true), (3, false)],
            vec![(4, false), (1, false)],
            vec![(5, false), (2, false)],
            vec![(6, true), (3, false)],
            vec![(4, false), (1, false)],
        ];
        let mut previous: Vec<SharedSubmap<Submap2D>> = Vec::new();
        for (i, expected_state) in expected.iter().enumerate() {
            let returned = active_submaps.insert_range_data(&scan);
            let current = active_submaps.submaps();
            assert_eq!(returned.len(), current.len());
            for (a, b) in returned.iter().zip(&current) {
                assert!(Arc::ptr_eq(a, b));
            }
            let state: Vec<(i32, bool)> = returned
                .iter()
                .map(|s| {
                    let s = read(s);
                    (s.num_range_data(), s.insertion_finished())
                })
                .collect();
            assert_eq!(&state, expected_state, "after insert {}", i + 1);
            // Finished submaps are cropped (the room is about 2 m wide).
            for submap in &returned {
                let submap = read(submap);
                let num_x_cells = submap.grid().limits().cell_limits().num_x_cells;
                if submap.insertion_finished() {
                    assert!(num_x_cells < 50, "{num_x_cells}");
                } else {
                    assert_eq!(num_x_cells, 100);
                }
            }
            // When a new submap is added after two exist, the old back becomes
            // the front.
            if previous.len() == 2 && expected_state[0].0 == N + 1 {
                assert!(Arc::ptr_eq(&previous[1], &returned[0]));
            }
            // The handles coerce to the polymorphic C++ `Submap` interface.
            let as_dyn: Arc<std::sync::RwLock<dyn Submap>> = returned[0].clone();
            assert_eq!(as_dyn.read().unwrap().num_range_data(), expected_state[0].0);
            previous = returned;
        }
    }

    #[test]
    #[should_panic(expected = "Invalid combination grid_type")]
    fn invalid_grid_inserter_combination() {
        let _ = create_submaps_options_2d(&make_dictionary(&submaps_options_lua(
            10,
            "TSDF",
            "PROBABILITY_GRID_INSERTER_2D",
        )));
    }

    #[test]
    #[should_panic(expected = "Check failed: !insertion_finished()")]
    fn insert_into_finished_submap_panics() {
        let conversion_tables = Arc::new(ValueConversionTables::new());
        let mut submap = Submap2D::new(
            &Vector2::new(0.0, 0.0),
            Box::new(ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(1.0, 1.0), CellLimits::new(40, 40)),
                &conversion_tables,
            )),
            &conversion_tables,
        );
        submap.finish();
        let inserter = ProbabilityGridRangeDataInserter2D::new(
            &proto::ProbabilityGridRangeDataInserterOptions2D {
                hit_probability: 0.55,
                miss_probability: 0.49,
                insert_free_space: true,
            },
        );
        submap.insert_range_data(&RangeData::default(), &inserter);
    }
}
