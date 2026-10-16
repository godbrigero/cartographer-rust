//! Port of `cartographer/mapping/2d/probability_grid` (.h, .cc).

use std::sync::Arc;

use crate::mapping::grid_interface::GridInterface;
use crate::mapping::probability_values::{
    MAX_CORRESPONDENCE_COST, MIN_CORRESPONDENCE_COST, MIN_PROBABILITY, UNKNOWN_PROBABILITY_VALUE,
    UPDATE_MARKER, correspondence_cost_to_probability, correspondence_cost_to_value,
    probability_to_correspondence_cost, value_to_correspondence_cost,
};
use crate::mapping::proto;
use crate::mapping::proto::submap_query::response::SubmapTexture;
use crate::mapping::submaps::probability_to_log_odds_integer;
use crate::mapping::two_d::grid_2d::{Grid2D, Grid2DTrait, GridType, fill_submap_texture};
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::xy_index::{Array2i, XYIndexRangeIterator};
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::transform::rigid_transform::Rigid3d;

/// Represents a 2D grid of probabilities.
///
/// The cells store *correspondence cost* values (`1 - probability`) in
/// `[MIN_CORRESPONDENCE_COST, MAX_CORRESPONDENCE_COST]`.
#[derive(Debug, Clone)]
pub struct ProbabilityGrid {
    grid: Grid2D,
    conversion_tables: Arc<ValueConversionTables>,
}

impl ProbabilityGrid {
    /// Creates an all-unknown grid.
    pub fn new(limits: MapLimits, conversion_tables: &Arc<ValueConversionTables>) -> Self {
        Self {
            grid: Grid2D::new(
                limits,
                MIN_CORRESPONDENCE_COST,
                MAX_CORRESPONDENCE_COST,
                conversion_tables,
            ),
            conversion_tables: Arc::clone(conversion_tables),
        }
    }

    /// Creates a grid from a proto. Panics unless the `probability_grid_2d`
    /// oneof is set.
    pub fn from_proto(
        proto: &proto::Grid2D,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Self {
        let grid = Grid2D::from_proto(proto, conversion_tables);
        assert!(
            matches!(proto.grid, Some(proto::grid2_d::Grid::ProbabilityGrid2d(_))),
            "Check failed: proto.has_probability_grid_2d()"
        );
        Self {
            grid,
            conversion_tables: Arc::clone(conversion_tables),
        }
    }

    /// Sets the probability of the cell at `cell_index` to the given
    /// `probability`. Only allowed if the cell was unknown before.
    pub fn set_probability(&mut self, cell_index: &Array2i, probability: f32) {
        let flat_index = self.grid.to_flat_index(cell_index);
        let cell = &mut self.grid.mutable_correspondence_cost_cells()[flat_index];
        assert_eq!(
            *cell, UNKNOWN_PROBABILITY_VALUE,
            "Check failed: cell == kUnknownProbabilityValue"
        );
        *cell = correspondence_cost_to_value(probability_to_correspondence_cost(probability));
        self.grid.mutable_known_cells_box().extend(cell_index);
    }

    /// Applies the `odds` specified when calling
    /// `compute_lookup_table_to_apply_correspondence_cost_odds()` to the
    /// probability of the cell at `cell_index` if the cell has not already
    /// been updated. Multiple updates of the same cell will be ignored until
    /// `finish_update()` is called. Returns true if the cell was updated.
    ///
    /// If this is the first call for the specified cell, its value will be
    /// set to the probability corresponding to `odds`.
    pub fn apply_lookup_table(&mut self, cell_index: &Array2i, table: &[u16]) -> bool {
        debug_assert_eq!(table.len(), UPDATE_MARKER as usize);
        let flat_index = self.grid.to_flat_index(cell_index);
        let cell = self.grid.correspondence_cost_cells()[flat_index];
        if cell >= UPDATE_MARKER {
            return false;
        }
        self.grid.mutable_update_indices().push(flat_index);
        let new_value = table[cell as usize];
        debug_assert!(new_value >= UPDATE_MARKER);
        self.grid.mutable_correspondence_cost_cells()[flat_index] = new_value;
        self.grid.mutable_known_cells_box().extend(cell_index);
        true
    }

    /// Returns the probability of the cell with `cell_index`
    /// (`MIN_PROBABILITY` outside the limits). Uses the global conversion
    /// table, like C++.
    pub fn get_probability(&self, cell_index: &Array2i) -> f32 {
        if !self.grid.limits().contains(cell_index) {
            return MIN_PROBABILITY;
        }
        correspondence_cost_to_probability(value_to_correspondence_cost(
            self.grid.correspondence_cost_cells()[self.grid.to_flat_index(cell_index)],
        ))
    }

    /// The conversion tables this grid was created with.
    pub fn conversion_tables(&self) -> &Arc<ValueConversionTables> {
        &self.conversion_tables
    }

    /// Concrete-typed [`Grid2DTrait::compute_cropped_grid`].
    pub fn compute_cropped_probability_grid(&self) -> ProbabilityGrid {
        let (offset, cell_limits) = self.grid.compute_cropped_limits();
        let resolution = self.grid.limits().resolution();
        let max = self.grid.limits().max()
            - resolution * nalgebra::Vector2::new(f64::from(offset.y), f64::from(offset.x));
        let mut cropped_grid = ProbabilityGrid::new(
            MapLimits::new(resolution, max, cell_limits),
            &self.conversion_tables,
        );
        for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
            if !self.grid.is_known(&(xy_index + offset)) {
                continue;
            }
            cropped_grid.set_probability(&xy_index, self.get_probability(&(xy_index + offset)));
        }
        cropped_grid
    }
}

impl GridInterface for ProbabilityGrid {}

impl Grid2DTrait for ProbabilityGrid {
    fn grid_2d(&self) -> &Grid2D {
        &self.grid
    }

    fn grid_2d_mut(&mut self) -> &mut Grid2D {
        &mut self.grid
    }

    fn get_grid_type(&self) -> GridType {
        GridType::ProbabilityGrid
    }

    fn compute_cropped_grid(&self) -> Box<dyn Grid2DTrait> {
        Box::new(self.compute_cropped_probability_grid())
    }

    fn to_proto(&self) -> proto::Grid2D {
        let mut result = self.grid.to_proto();
        result.grid = Some(proto::grid2_d::Grid::ProbabilityGrid2d(
            proto::ProbabilityGrid::default(),
        ));
        result
    }

    fn draw_to_submap_texture(&self, texture: &mut SubmapTexture, local_pose: Rigid3d) -> bool {
        let (offset, cell_limits) = self.grid.compute_cropped_limits();
        let mut cells = Vec::new();
        for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
            if !self.grid.is_known(&(xy_index + offset)) {
                cells.push(0); // unknown log odds value
                cells.push(0); // alpha
                continue;
            }
            // We would like to add 'delta' but this is not possible using a
            // value and alpha. We use premultiplied alpha, so when 'delta' is
            // positive we can add it by setting 'alpha' to zero. If it is
            // negative, we set 'value' to zero, and use 'alpha' to subtract.
            // This is only correct when the pixel is currently white, so walls
            // will look too gray. This should be hard to detect visually for
            // the user, though.
            let delta = 128
                - i32::from(probability_to_log_odds_integer(
                    self.get_probability(&(xy_index + offset)),
                ));
            let alpha: u8 = if delta > 0 { 0 } else { (-delta) as u8 };
            let value: u8 = if delta > 0 { delta as u8 } else { 0 };
            cells.push(value);
            cells.push(if value != 0 || alpha != 0 { alpha } else { 1 });
        }
        fill_submap_texture(
            texture,
            &cells,
            &cell_limits,
            self.grid.limits(),
            &offset,
            &local_pose,
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nalgebra::Vector2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::mapping::probability_values::*;
    use crate::mapping::proto;
    use crate::mapping::two_d::grid_2d::{Grid2DTrait, GridType};
    use crate::mapping::two_d::map_limits::{self, MapLimits};
    use crate::mapping::two_d::probability_grid::ProbabilityGrid;
    use crate::mapping::two_d::xy_index::{Array2i, CellLimits, XYIndexRangeIterator};
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::transform::rigid_transform::Rigid3d;

    fn tables() -> Arc<ValueConversionTables> {
        Arc::new(ValueConversionTables::new())
    }

    mod probability_grid_test {
        use super::*;

        #[test]
        fn proto_constructor() {
            let limits = MapLimits::new(1.0, Vector2::new(2.0, 3.0), CellLimits::new(4, 5));
            let proto = proto::Grid2D {
                limits: Some(map_limits::to_proto(&limits)),
                cells: (6..12).collect(),
                known_cells_box: Some(proto::grid2_d::CellBox {
                    max_x: 19,
                    max_y: 20,
                    min_x: 21,
                    min_y: 22,
                }),
                grid: Some(proto::grid2_d::Grid::ProbabilityGrid2d(
                    proto::ProbabilityGrid::default(),
                )),
                ..Default::default()
            };
            let conversion_tables = tables();
            let grid = ProbabilityGrid::from_proto(&proto, &conversion_tables);
            assert_eq!(proto.limits.unwrap(), map_limits::to_proto(grid.limits()));
            assert_eq!(grid.get_grid_type(), GridType::ProbabilityGrid);
        }

        #[test]
        fn constructor_grid_type() {
            let conversion_tables = tables();
            let probability_grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                &conversion_tables,
            );
            assert_eq!(probability_grid.get_grid_type(), GridType::ProbabilityGrid);
        }

        #[test]
        fn to_proto() {
            let conversion_tables = tables();
            let probability_grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                &conversion_tables,
            );
            let proto = probability_grid.to_proto();
            assert_eq!(
                map_limits::to_proto(probability_grid.limits()),
                proto.limits.unwrap()
            );
        }

        #[test]
        fn apply_odds() {
            let conversion_tables = tables();
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                &conversion_tables,
            );
            let limits = *probability_grid.limits();
            assert!(limits.contains(&Array2i::new(0, 0)));
            assert!(limits.contains(&Array2i::new(0, 1)));
            assert!(limits.contains(&Array2i::new(1, 0)));
            assert!(limits.contains(&Array2i::new(1, 1)));
            assert!(!probability_grid.is_known(&Array2i::new(0, 0)));
            assert!(!probability_grid.is_known(&Array2i::new(0, 1)));
            assert!(!probability_grid.is_known(&Array2i::new(1, 0)));
            assert!(!probability_grid.is_known(&Array2i::new(1, 1)));

            probability_grid.set_probability(&Array2i::new(1, 0), 0.5);
            probability_grid.apply_lookup_table(
                &Array2i::new(1, 0),
                &compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.9)),
            );
            probability_grid.finish_update();
            assert!(probability_grid.get_probability(&Array2i::new(1, 0)) > 0.5);

            probability_grid.set_probability(&Array2i::new(0, 1), 0.5);
            probability_grid.apply_lookup_table(
                &Array2i::new(0, 1),
                &compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.1)),
            );
            probability_grid.finish_update();
            assert!(probability_grid.get_probability(&Array2i::new(0, 1)) < 0.5);

            // Tests adding odds to an unknown cell.
            probability_grid.apply_lookup_table(
                &Array2i::new(1, 1),
                &compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.42)),
            );
            assert!((probability_grid.get_probability(&Array2i::new(1, 1)) - 0.42).abs() <= 1e-4);

            // Tests that further updates are ignored if FinishUpdate() isn't
            // called.
            probability_grid.apply_lookup_table(
                &Array2i::new(1, 1),
                &compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.9)),
            );
            assert!((probability_grid.get_probability(&Array2i::new(1, 1)) - 0.42).abs() <= 1e-4);
            probability_grid.finish_update();
            probability_grid.apply_lookup_table(
                &Array2i::new(1, 1),
                &compute_lookup_table_to_apply_correspondence_cost_odds(odds(0.9)),
            );
            assert!(probability_grid.get_probability(&Array2i::new(1, 1)) > 0.42);
        }

        #[test]
        fn get_probability() {
            let conversion_tables = tables();
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(1.0, Vector2::new(1.0, 2.0), CellLimits::new(2, 2)),
                &conversion_tables,
            );
            let limits = *probability_grid.limits();
            assert_eq!(1.0, limits.max().x);
            assert_eq!(2.0, limits.max().y);
            let cell_limits = limits.cell_limits();
            assert_eq!(2, cell_limits.num_x_cells);
            assert_eq!(2, cell_limits.num_y_cells);

            probability_grid.set_probability(
                &limits.get_cell_index(&Vector2::new(-0.5, 0.5)),
                MAX_PROBABILITY,
            );
            assert!(
                (probability_grid
                    .get_probability(&limits.get_cell_index(&Vector2::new(-0.5, 0.5)))
                    - MAX_PROBABILITY)
                    .abs()
                    <= 1e-6
            );
            for xy_index in [
                limits.get_cell_index(&Vector2::new(-0.5, 1.5)),
                limits.get_cell_index(&Vector2::new(0.5, 0.5)),
                limits.get_cell_index(&Vector2::new(0.5, 1.5)),
            ] {
                assert!(limits.contains(&xy_index));
                assert!(!probability_grid.is_known(&xy_index));
            }
        }

        #[test]
        fn get_cell_index() {
            let conversion_tables = tables();
            let probability_grid = ProbabilityGrid::new(
                MapLimits::new(2.0, Vector2::new(8.0, 14.0), CellLimits::new(14, 8)),
                &conversion_tables,
            );
            let limits = probability_grid.limits();
            let cell_limits = limits.cell_limits();
            assert_eq!(14, cell_limits.num_x_cells);
            assert_eq!(8, cell_limits.num_y_cells);
            let check = |expected: (i32, i32), x: f32, y: f32| {
                assert_eq!(
                    Array2i::new(expected.0, expected.1),
                    limits.get_cell_index(&Vector2::new(x, y)),
                    "point ({x}, {y})"
                );
            };
            check((0, 0), 7.0, 13.0);
            check((13, 0), 7.0, -13.0);
            check((0, 7), -7.0, 13.0);
            check((13, 7), -7.0, -13.0);
            // Check around the origin.
            check((6, 3), 0.5, 0.5);
            check((6, 3), 1.5, 1.5);
            check((7, 3), 0.5, -0.5);
            check((6, 4), -0.5, 0.5);
            check((7, 4), -0.5, -0.5);
        }

        #[test]
        fn correct_cropping() {
            // Create a probability grid with random values.
            let mut rng = StdRng::seed_from_u64(42);
            let conversion_tables = tables();
            let mut probability_grid = ProbabilityGrid::new(
                MapLimits::new(0.05, Vector2::new(10.0, 10.0), CellLimits::new(400, 400)),
                &conversion_tables,
            );
            for xy_index in
                XYIndexRangeIterator::new(Array2i::new(100, 100), Array2i::new(299, 299))
            {
                probability_grid.set_probability(
                    &xy_index,
                    rng.random_range(MIN_PROBABILITY..MAX_PROBABILITY),
                );
            }
            let (offset, limits) = probability_grid.compute_cropped_limits();
            assert_eq!(offset, Array2i::new(100, 100));
            assert_eq!(limits.num_x_cells, 200);
            assert_eq!(limits.num_y_cells, 200);
        }
    }

    mod additional {
        use super::*;
        use crate::common::port::fast_gunzip_string;

        fn random_grid() -> ProbabilityGrid {
            let mut rng = StdRng::seed_from_u64(7);
            let mut grid = ProbabilityGrid::new(
                MapLimits::new(0.1, Vector2::new(1.0, 2.0), CellLimits::new(30, 20)),
                &tables(),
            );
            for xy_index in XYIndexRangeIterator::new(Array2i::new(5, 3), Array2i::new(17, 12)) {
                if rng.random_bool(0.7) {
                    grid.set_probability(
                        &xy_index,
                        rng.random_range(MIN_PROBABILITY..MAX_PROBABILITY),
                    );
                }
            }
            grid
        }

        #[test]
        fn proto_round_trip() {
            let grid = random_grid();
            let proto = grid.to_proto();
            let restored = ProbabilityGrid::from_proto(&proto, grid.conversion_tables());
            assert_eq!(restored.to_proto(), proto);
            assert_eq!(
                restored.grid_2d().correspondence_cost_cells(),
                grid.grid_2d().correspondence_cost_cells()
            );
            assert_eq!(
                restored.grid_2d().known_cells_box(),
                grid.grid_2d().known_cells_box()
            );
            assert_eq!(restored.limits(), grid.limits());
        }

        #[test]
        #[should_panic(expected = "has_probability_grid_2d")]
        fn proto_without_probability_grid_panics() {
            let grid = random_grid();
            let mut proto = grid.to_proto();
            proto.grid = None;
            let _ = ProbabilityGrid::from_proto(&proto, grid.conversion_tables());
        }

        #[test]
        fn cropped_grid_preserves_known_cells() {
            let grid = random_grid();
            let cropped_box = grid.compute_cropped_grid();
            let cropped = cropped_box.downcast_ref::<ProbabilityGrid>().unwrap();
            let (offset, cell_limits) = grid.compute_cropped_limits();
            assert_eq!(cropped.limits().cell_limits(), &cell_limits);
            for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
                let source = xy_index + offset;
                assert_eq!(cropped.is_known(&xy_index), grid.is_known(&source));
                assert_eq!(
                    cropped.get_probability(&xy_index),
                    grid.get_probability(&source)
                );
                // Same world position.
                assert_eq!(
                    cropped.limits().get_cell_center(&xy_index),
                    grid.limits().get_cell_center(&source)
                );
            }
        }

        #[test]
        fn draw_to_submap_texture() {
            let mut grid = ProbabilityGrid::new(
                MapLimits::new(0.5, Vector2::new(2.0, 3.0), CellLimits::new(4, 4)),
                &tables(),
            );
            grid.set_probability(&Array2i::new(1, 1), MAX_PROBABILITY);
            grid.set_probability(&Array2i::new(2, 1), MIN_PROBABILITY);
            grid.set_probability(&Array2i::new(2, 2), 0.5);
            let mut texture = proto::submap_query::response::SubmapTexture::default();
            assert!(grid.draw_to_submap_texture(&mut texture, Rigid3d::identity()));
            assert_eq!(texture.width, 2);
            assert_eq!(texture.height, 2);
            assert_eq!(texture.resolution, 0.5);
            let mut cells = Vec::new();
            fast_gunzip_string(&texture.cells, &mut cells);
            // Row-major over the cropped 2x2 region starting at (1, 1):
            // (1,1) p=0.9 -> delta=-127; (2,1) p=0.1 -> delta=127;
            // (1,2) unknown; (2,2) p=0.5 -> delta=0 -> (0, 1).
            assert_eq!(cells, vec![0, 127, 127, 0, 0, 0, 0, 1]);
            let slice_pose = texture.slice_pose.unwrap();
            let t = slice_pose.translation.unwrap();
            assert_eq!((t.x, t.y, t.z), (1.5, 2.5, 0.0));
        }
    }
}
