//! Port of `cartographer/mapping/internal/2d/tsdf_2d` (.h, .cc).

use std::sync::Arc;

use nalgebra::Vector2;

use crate::mapping::grid_interface::GridInterface;
use crate::mapping::internal::two_d::tsd_value_converter::TSDValueConverter;
use crate::mapping::proto;
use crate::mapping::proto::submap_query::response::SubmapTexture;
use crate::mapping::two_d::grid_2d::{Grid2D, Grid2DTrait, GridType, fill_submap_texture};
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::xy_index::{Array2i, XYIndexRangeIterator};
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::transform::rigid_transform::Rigid3d;

/// Represents a 2D grid of truncated signed distances and weights.
///
/// The TSD values live in the base [`Grid2D`] cells (which carry the update
/// marker); the base is created with correspondence cost bounds
/// `[-truncation_distance, truncation_distance]`, so
/// `get_correspondence_cost` of unknown / out-of-bounds cells is
/// `+truncation_distance` whereas [`get_tsd`](Self::get_tsd) returns
/// `-truncation_distance` there (like C++; the two tables differ).
#[derive(Debug, Clone)]
pub struct TSDF2D {
    grid: Grid2D,
    conversion_tables: Arc<ValueConversionTables>,
    // RUST PORT NOTE: `std::unique_ptr<TSDValueConverter>` in C++.
    value_converter: TSDValueConverter,
    // Weights never carry the update marker (it lives in the TSD cell).
    weight_cells: Vec<u16>,
}

impl TSDF2D {
    /// Creates an all-unknown TSDF.
    pub fn new(
        limits: MapLimits,
        truncation_distance: f32,
        max_weight: f32,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Self {
        let num_cells =
            limits.cell_limits().num_x_cells as usize * limits.cell_limits().num_y_cells as usize;
        let grid = Grid2D::new(
            limits,
            -truncation_distance,
            truncation_distance,
            conversion_tables,
        );
        let value_converter =
            TSDValueConverter::new(truncation_distance, max_weight, conversion_tables);
        Self {
            grid,
            conversion_tables: Arc::clone(conversion_tables),
            value_converter,
            weight_cells: vec![TSDValueConverter::get_unknown_weight_value(); num_cells],
        }
    }

    /// Creates a TSDF from a proto. Panics unless the `tsdf_2d` oneof is set.
    pub fn from_proto(
        proto: &proto::Grid2D,
        conversion_tables: &Arc<ValueConversionTables>,
    ) -> Self {
        let grid = Grid2D::from_proto(proto, conversion_tables);
        let Some(proto::grid2_d::Grid::Tsdf2d(tsdf_2d)) = &proto.grid else {
            panic!("Check failed: proto.has_tsdf_2d()");
        };
        let value_converter = TSDValueConverter::new(
            tsdf_2d.truncation_distance,
            tsdf_2d.max_weight,
            conversion_tables,
        );
        let weight_cells = tsdf_2d
            .weight_cells
            .iter()
            .map(|&cell| {
                assert!(
                    cell <= i32::from(u16::MAX),
                    "Check failed: cell <= std::numeric_limits<uint16>::max() ({cell} vs. 65535)"
                );
                cell as u16
            })
            .collect();
        Self {
            grid,
            conversion_tables: Arc::clone(conversion_tables),
            value_converter,
            weight_cells,
        }
    }

    /// True if the cell was updated since the last `finish_update`.
    pub fn cell_is_updated(&self, cell_index: &Array2i) -> bool {
        let flat_index = self.grid.to_flat_index(cell_index);
        let tsdf_cell = self.grid.correspondence_cost_cells()[flat_index];
        tsdf_cell >= TSDValueConverter::get_update_marker()
    }

    /// Sets TSD and weight of the cell. Silently ignored if the cell was
    /// already updated since the last `finish_update`.
    pub fn set_cell(&mut self, cell_index: &Array2i, tsd: f32, weight: f32) {
        let flat_index = self.grid.to_flat_index(cell_index);
        if self.grid.correspondence_cost_cells()[flat_index]
            >= TSDValueConverter::get_update_marker()
        {
            return;
        }
        self.grid.mutable_update_indices().push(flat_index);
        self.grid.mutable_known_cells_box().extend(cell_index);
        self.grid.mutable_correspondence_cost_cells()[flat_index] =
            self.value_converter.tsd_to_value(tsd) + TSDValueConverter::get_update_marker();
        self.weight_cells[flat_index] = self.value_converter.weight_to_value(weight);
    }

    /// Returns the TSD of the cell (`-truncation_distance` outside the limits
    /// or for unknown cells).
    pub fn get_tsd(&self, cell_index: &Array2i) -> f32 {
        if self.grid.limits().contains(cell_index) {
            return self.value_converter.value_to_tsd(
                self.grid.correspondence_cost_cells()[self.grid.to_flat_index(cell_index)],
            );
        }
        self.value_converter.get_min_tsd()
    }

    /// Returns the weight of the cell (0 outside the limits or for unknown
    /// cells).
    pub fn get_weight(&self, cell_index: &Array2i) -> f32 {
        if self.grid.limits().contains(cell_index) {
            return self
                .value_converter
                .value_to_weight(self.weight_cells[self.grid.to_flat_index(cell_index)]);
        }
        self.value_converter.get_min_weight()
    }

    /// Returns `(tsd, weight)` of the cell.
    pub fn get_tsd_and_weight(&self, cell_index: &Array2i) -> (f32, f32) {
        if self.grid.limits().contains(cell_index) {
            let flat_index = self.grid.to_flat_index(cell_index);
            return (
                self.value_converter
                    .value_to_tsd(self.grid.correspondence_cost_cells()[flat_index]),
                self.value_converter
                    .value_to_weight(self.weight_cells[flat_index]),
            );
        }
        (
            self.value_converter.get_min_tsd(),
            self.value_converter.get_min_weight(),
        )
    }

    /// The value converter (C++ has no accessor; exposed for the TSDF
    /// inserter / scan matchers that need truncation distance and max weight).
    pub fn value_converter(&self) -> &TSDValueConverter {
        &self.value_converter
    }

    /// The conversion tables this grid was created with.
    pub fn conversion_tables(&self) -> &Arc<ValueConversionTables> {
        &self.conversion_tables
    }

    /// The raw weight cells.
    pub fn weight_cells(&self) -> &[u16] {
        &self.weight_cells
    }

    /// Concrete-typed [`Grid2DTrait::compute_cropped_grid`].
    pub fn compute_cropped_tsdf(&self) -> TSDF2D {
        let (offset, cell_limits) = self.grid.compute_cropped_limits();
        let resolution = self.grid.limits().resolution();
        let max = self.grid.limits().max()
            - resolution * Vector2::new(f64::from(offset.y), f64::from(offset.x));
        let mut cropped_grid = TSDF2D::new(
            MapLimits::new(resolution, max, cell_limits),
            self.value_converter.get_max_tsd(),
            self.value_converter.get_max_weight(),
            &self.conversion_tables,
        );
        for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
            if !self.grid.is_known(&(xy_index + offset)) {
                continue;
            }
            cropped_grid.set_cell(
                &xy_index,
                self.get_tsd(&(xy_index + offset)),
                self.get_weight(&(xy_index + offset)),
            );
        }
        cropped_grid.grid.finish_update();
        cropped_grid
    }
}

impl GridInterface for TSDF2D {}

impl Grid2DTrait for TSDF2D {
    fn grid_2d(&self) -> &Grid2D {
        &self.grid
    }

    fn grid_2d_mut(&mut self) -> &mut Grid2D {
        &mut self.grid
    }

    fn get_grid_type(&self) -> GridType {
        GridType::Tsdf
    }

    /// Grows both the TSD and the weight cells.
    fn grow_limits(&mut self, point: &Vector2<f32>) {
        self.grid.grow_limits_with(
            point,
            &mut [(
                &mut self.weight_cells,
                TSDValueConverter::get_unknown_weight_value(),
            )],
        );
    }

    fn compute_cropped_grid(&self) -> Box<dyn Grid2DTrait> {
        Box::new(self.compute_cropped_tsdf())
    }

    fn to_proto(&self) -> proto::Grid2D {
        let mut result = self.grid.to_proto();
        result.grid = Some(proto::grid2_d::Grid::Tsdf2d(proto::Tsdf2d {
            truncation_distance: self.value_converter.get_max_tsd(),
            max_weight: self.value_converter.get_max_weight(),
            weight_cells: self.weight_cells.iter().map(|&c| i32::from(c)).collect(),
        }));
        result
    }

    fn draw_to_submap_texture(&self, texture: &mut SubmapTexture, local_pose: Rigid3d) -> bool {
        let (offset, cell_limits) = self.grid.compute_cropped_limits();
        let mut cells = Vec::new();
        for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
            if !self.grid.is_known(&(xy_index + offset)) {
                cells.push(0); // value
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
            let mut normalized_tsdf = self.get_tsd(&(xy_index + offset)).abs();
            normalized_tsdf = (normalized_tsdf / self.value_converter.get_max_tsd()).powf(0.5);
            let normalized_weight =
                self.get_weight(&(xy_index + offset)) / self.value_converter.get_max_weight();
            let delta = (f64::from(normalized_weight)
                * (f64::from(normalized_tsdf) * 255.0 - 128.0))
                .round() as i32;
            // `uint8` conversions wrap like C++.
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

    use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
    use crate::mapping::proto;
    use crate::mapping::two_d::grid_2d::{Grid2DTrait, GridType};
    use crate::mapping::two_d::map_limits::{self, MapLimits};
    use crate::mapping::two_d::xy_index::{Array2i, CellLimits, XYIndexRangeIterator};
    use crate::mapping::value_conversion_tables::ValueConversionTables;
    use crate::transform::rigid_transform::Rigid3d;

    fn tables() -> Arc<ValueConversionTables> {
        Arc::new(ValueConversionTables::new())
    }

    mod tsdf_2d_test {
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
                grid: Some(proto::grid2_d::Grid::Tsdf2d(proto::Tsdf2d {
                    max_weight: 10.0,
                    truncation_distance: 1.0,
                    weight_cells: (6..12).collect(),
                })),
                max_correspondence_cost: 1.0,
                min_correspondence_cost: -1.0,
            };
            let conversion_tables = tables();
            let grid = TSDF2D::from_proto(&proto, &conversion_tables);
            assert_eq!(grid.get_grid_type(), GridType::Tsdf);
            assert_eq!(proto.limits.unwrap(), map_limits::to_proto(grid.limits()));
        }

        #[test]
        fn constructor_grid_type() {
            let conversion_tables = tables();
            let tsdf = TSDF2D::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                1.0,
                10.0,
                &conversion_tables,
            );
            assert_eq!(tsdf.get_grid_type(), GridType::Tsdf);
        }

        #[test]
        fn to_proto() {
            let conversion_tables = tables();
            let tsdf = TSDF2D::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                1.0,
                10.0,
                &conversion_tables,
            );
            let proto = tsdf.to_proto();
            assert_eq!(map_limits::to_proto(tsdf.limits()), proto.limits.unwrap());
        }

        #[test]
        fn get_cell_index() {
            let conversion_tables = tables();
            let tsdf = TSDF2D::new(
                MapLimits::new(2.0, Vector2::new(8.0, 14.0), CellLimits::new(14, 8)),
                1.0,
                10.0,
                &conversion_tables,
            );
            let limits = tsdf.limits();
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
        fn write_read() {
            let truncation_distance = 1.0_f32;
            let max_weight = 10.0_f32;
            let conversion_tables = tables();
            let mut tsdf = TSDF2D::new(
                MapLimits::new(1.0, Vector2::new(1.0, 2.0), CellLimits::new(2, 2)),
                truncation_distance,
                max_weight,
                &conversion_tables,
            );
            let limits = *tsdf.limits();
            assert_eq!(1.0, limits.max().x);
            assert_eq!(2.0, limits.max().y);
            let cell_limits = limits.cell_limits();
            assert_eq!(2, cell_limits.num_x_cells);
            assert_eq!(2, cell_limits.num_y_cells);

            let mut rng = StdRng::seed_from_u64(42);
            let tsd_tolerance = 2.0 * truncation_distance / 32768.0;
            let weight_tolerance = max_weight / 32768.0;
            let cell = limits.get_cell_index(&Vector2::new(-0.5, 0.5));
            for _ in 0..1 {
                let tsd: f32 = rng.random_range(-truncation_distance..truncation_distance);
                let weight: f32 = rng.random_range(0.0..max_weight);
                tsdf.set_cell(&cell, tsd, weight);
                assert!((tsdf.get_tsd_and_weight(&cell).0 - tsd).abs() <= tsd_tolerance);
                assert!((tsdf.get_tsd_and_weight(&cell).1 - weight).abs() <= weight_tolerance);
                assert!((tsdf.get_tsd(&cell) - tsd).abs() <= tsd_tolerance);
                assert!((tsdf.get_weight(&cell) - weight).abs() <= weight_tolerance);
            }
            for xy_index in [
                limits.get_cell_index(&Vector2::new(-0.5, 1.5)),
                limits.get_cell_index(&Vector2::new(0.5, 0.5)),
                limits.get_cell_index(&Vector2::new(0.5, 1.5)),
            ] {
                assert!(limits.contains(&xy_index));
                assert!(!tsdf.is_known(&xy_index));
            }
        }

        #[test]
        fn correct_cropping() {
            // Create a TSDF with random values.
            let truncation_distance = 1.0_f32;
            let max_weight = 10.0_f32;
            let mut rng = StdRng::seed_from_u64(42);
            let conversion_tables = tables();
            let mut tsdf = TSDF2D::new(
                MapLimits::new(0.05, Vector2::new(10.0, 10.0), CellLimits::new(400, 400)),
                truncation_distance,
                max_weight,
                &conversion_tables,
            );
            for xy_index in
                XYIndexRangeIterator::new(Array2i::new(100, 100), Array2i::new(299, 299))
            {
                let tsd = rng.random_range(-truncation_distance..truncation_distance);
                let weight = rng.random_range(0.0..max_weight);
                tsdf.set_cell(&xy_index, tsd, weight);
            }
            let (offset, limits) = tsdf.compute_cropped_limits();
            assert_eq!(offset, Array2i::new(100, 100));
            assert_eq!(limits.num_x_cells, 200);
            assert_eq!(limits.num_y_cells, 200);
        }
    }

    mod additional {
        use super::*;
        use crate::common::port::fast_gunzip_string;

        fn random_tsdf() -> TSDF2D {
            let mut rng = StdRng::seed_from_u64(3);
            let mut tsdf = TSDF2D::new(
                MapLimits::new(0.1, Vector2::new(1.0, 2.0), CellLimits::new(30, 20)),
                0.3,
                10.0,
                &tables(),
            );
            for xy_index in XYIndexRangeIterator::new(Array2i::new(4, 2), Array2i::new(20, 15)) {
                if rng.random_bool(0.6) {
                    let tsd = rng.random_range(-0.3..0.3);
                    let weight = rng.random_range(0.0..10.0);
                    tsdf.set_cell(&xy_index, tsd, weight);
                }
            }
            tsdf.finish_update();
            tsdf
        }

        #[test]
        fn unknown_and_out_of_bounds_values() {
            let tsdf = random_tsdf();
            let outside = Array2i::new(-1, 0);
            assert_eq!(tsdf.get_tsd(&outside), -0.3);
            assert_eq!(tsdf.get_weight(&outside), 0.0);
            assert_eq!(tsdf.get_tsd_and_weight(&outside), (-0.3, 0.0));
            // The base table maps unknown to +truncation.
            assert_eq!(tsdf.get_correspondence_cost(&outside), 0.3);
            let unknown = Array2i::new(0, 0);
            assert!(!tsdf.is_known(&unknown));
            assert_eq!(tsdf.get_tsd(&unknown), -0.3);
            assert_eq!(tsdf.get_correspondence_cost(&unknown), 0.3);
        }

        #[test]
        fn set_cell_ignores_updated_cells() {
            let mut tsdf = TSDF2D::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(2, 2)),
                1.0,
                10.0,
                &tables(),
            );
            let cell = Array2i::new(1, 1);
            tsdf.set_cell(&cell, 0.5, 2.0);
            assert!(tsdf.cell_is_updated(&cell));
            tsdf.set_cell(&cell, -0.5, 7.0);
            assert!((tsdf.get_tsd(&cell) - 0.5).abs() < 1e-4);
            assert!((tsdf.get_weight(&cell) - 2.0).abs() < 1e-3);
            tsdf.finish_update();
            assert!(!tsdf.cell_is_updated(&cell));
            tsdf.set_cell(&cell, -0.5, 7.0);
            assert!((tsdf.get_tsd(&cell) + 0.5).abs() < 1e-4);
        }

        #[test]
        fn proto_round_trip() {
            let tsdf = random_tsdf();
            let proto = tsdf.to_proto();
            let restored = TSDF2D::from_proto(&proto, tsdf.conversion_tables());
            assert_eq!(restored.to_proto(), proto);
            assert_eq!(restored.weight_cells(), tsdf.weight_cells());
            assert_eq!(
                restored.grid_2d().correspondence_cost_cells(),
                tsdf.grid_2d().correspondence_cost_cells()
            );
            assert_eq!(restored.value_converter().get_max_tsd(), 0.3);
            assert_eq!(restored.value_converter().get_max_weight(), 10.0);
        }

        #[test]
        #[should_panic(expected = "has_tsdf_2d")]
        fn proto_without_tsdf_panics() {
            let tsdf = random_tsdf();
            let mut proto = tsdf.to_proto();
            proto.grid = Some(proto::grid2_d::Grid::ProbabilityGrid2d(
                proto::ProbabilityGrid::default(),
            ));
            let _ = TSDF2D::from_proto(&proto, tsdf.conversion_tables());
        }

        #[test]
        fn grow_limits_grows_weights() {
            let mut tsdf = random_tsdf();
            let cell = Array2i::new(10, 10);
            let world = tsdf.limits().get_cell_center(&cell);
            let before = tsdf.get_tsd_and_weight(&cell);
            let known_before = tsdf.is_known(&cell);
            let mut grid: Box<dyn Grid2DTrait> = Box::new(tsdf.clone());
            grid.grow_limits(&Vector2::new(20.0, -20.0));
            let grown = grid.downcast_ref::<TSDF2D>().unwrap();
            assert!(grown.limits().cell_limits().num_x_cells > 30);
            assert_eq!(
                grown.weight_cells().len(),
                grown.grid_2d().correspondence_cost_cells().len()
            );
            let new_cell = grown.limits().get_cell_index(&world);
            assert_eq!(grown.get_tsd_and_weight(&new_cell), before);
            assert_eq!(grown.is_known(&new_cell), known_before);
            tsdf.grow_limits(&Vector2::new(20.0, -20.0));
            assert_eq!(tsdf.to_proto(), grown.to_proto());
        }

        #[test]
        fn cropped_grid_preserves_known_cells() {
            let tsdf = random_tsdf();
            let cropped_box = tsdf.compute_cropped_grid();
            assert_eq!(cropped_box.get_grid_type(), GridType::Tsdf);
            let cropped = cropped_box.downcast_ref::<TSDF2D>().unwrap();
            assert!(cropped.grid_2d().update_indices().is_empty());
            let (offset, cell_limits) = tsdf.compute_cropped_limits();
            for xy_index in XYIndexRangeIterator::from_cell_limits(&cell_limits) {
                let source = xy_index + offset;
                assert_eq!(cropped.is_known(&xy_index), tsdf.is_known(&source));
                assert_eq!(
                    cropped.get_tsd_and_weight(&xy_index),
                    tsdf.get_tsd_and_weight(&source)
                );
            }
        }

        #[test]
        fn draw_to_submap_texture() {
            let mut tsdf = TSDF2D::new(
                MapLimits::new(1.0, Vector2::new(1.0, 1.0), CellLimits::new(3, 1)),
                1.0,
                10.0,
                &tables(),
            );
            tsdf.set_cell(&Array2i::new(0, 0), 1.0, 10.0);
            tsdf.set_cell(&Array2i::new(1, 0), 0.0, 10.0);
            tsdf.finish_update();
            let mut texture = proto::submap_query::response::SubmapTexture::default();
            assert!(tsdf.draw_to_submap_texture(&mut texture, Rigid3d::identity()));
            assert_eq!((texture.width, texture.height), (2, 1));
            let mut cells = Vec::new();
            fast_gunzip_string(&texture.cells, &mut cells);
            // |tsd| = 1, w = 1 -> delta = 127; tsd = 0 -> delta = -128.
            assert_eq!(cells[0], 127);
            assert_eq!(cells[1], 0);
            assert_eq!(cells[2], 0);
            assert_eq!(cells[3], 128);
        }
    }
}
