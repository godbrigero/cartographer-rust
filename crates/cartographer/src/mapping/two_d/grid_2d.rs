//! Port of `cartographer/mapping/2d/grid_2d` (.h, .cc).
//!
//! RUST PORT NOTE: the C++ abstract class `Grid2D : GridInterface` is split
//! into
//! * the struct [`Grid2D`], holding the shared state (limits, correspondence
//!   cost cells, update indices, known cells box, conversion table) and the
//!   non-virtual methods, and
//! * the object-safe trait [`Grid2DTrait`] for the virtual methods
//!   (`GetGridType` -> [`get_grid_type`](Grid2DTrait::get_grid_type),
//!   `GrowLimits` -> [`grow_limits`](Grid2DTrait::grow_limits),
//!   `ComputeCroppedGrid` -> [`compute_cropped_grid`](Grid2DTrait::compute_cropped_grid),
//!   `ToProto` -> [`to_proto`](Grid2DTrait::to_proto),
//!   `DrawToSubmapTexture` -> [`draw_to_submap_texture`](Grid2DTrait::draw_to_submap_texture)).
//!   The trait also forwards the non-virtual public `Grid2D` API (`limits`,
//!   `finish_update`, `get_correspondence_cost`, `is_known`, ...) so that a
//!   `&dyn Grid2DTrait` reads like a C++ `const Grid2D&`.
//!
//! `ProbabilityGrid` and `Tsdf2D` contain a `Grid2D` and implement the trait.
//! Code that holds a C++ `std::unique_ptr<Grid2D>` holds a
//! `Box<dyn Grid2DTrait>` and uses `downcast_ref`/`downcast_mut` where C++
//! uses `static_cast`.

use std::fmt;
use std::sync::Arc;

use nalgebra::{Vector2, Vector3};

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::grid_interface::GridInterface;
use crate::mapping::probability_values::{
    MAX_CORRESPONDENCE_COST, MIN_CORRESPONDENCE_COST, UNKNOWN_CORRESPONDENCE_VALUE, UPDATE_MARKER,
};
use crate::mapping::proto;
use crate::mapping::proto::submap_query::response::SubmapTexture;
use crate::mapping::two_d::map_limits::{self, MapLimits};
use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
use crate::mapping::value_conversion_tables::ValueConversionTables;
use crate::transform::rigid_transform::Rigid3d;

/// Reads `GridOptions2D` from `parameter_dictionary` (`grid_type` string,
/// `resolution`).
pub fn create_grid_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::GridOptions2D {
    let grid_type_string = parameter_dictionary.get_string("grid_type");
    let grid_type = proto::grid_options2_d::GridType::from_str_name(&grid_type_string)
        .unwrap_or_else(|| panic!("Unknown GridOptions2D_GridType kind: {grid_type_string}"));
    let mut options = proto::GridOptions2D::default();
    options.set_grid_type(grid_type);
    // The proto field is a float; C++ narrows the double the same way.
    options.resolution = parameter_dictionary.get_double("resolution") as f32;
    options
}

/// The kind of a 2D grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridType {
    /// `GridType::PROBABILITY_GRID`.
    ProbabilityGrid,
    /// `GridType::TSDF`.
    Tsdf,
}

/// Axis-aligned inclusive integer box (C++ `Eigen::AlignedBox2i`).
///
/// The default box is empty (`min = i32::MAX`, `max = i32::MIN`, like Eigen);
/// a box is empty if `min > max` in any coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignedBox2i {
    min: Vector2<i32>,
    max: Vector2<i32>,
}

impl Default for AlignedBox2i {
    fn default() -> Self {
        Self {
            min: Vector2::new(i32::MAX, i32::MAX),
            max: Vector2::new(i32::MIN, i32::MIN),
        }
    }
}

impl AlignedBox2i {
    /// Creates the box `[min, max]` (may be empty if `min > max`).
    pub fn new(min: Vector2<i32>, max: Vector2<i32>) -> Self {
        Self { min, max }
    }

    /// Minimum corner.
    pub fn min(&self) -> &Vector2<i32> {
        &self.min
    }

    /// Maximum corner.
    pub fn max(&self) -> &Vector2<i32> {
        &self.max
    }

    /// True if the box contains no point.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }

    /// Extends the box to contain `p`.
    pub fn extend(&mut self, p: &Vector2<i32>) {
        self.min = self.min.inf(p);
        self.max = self.max.sup(p);
    }

    /// `max - min`.
    pub fn sizes(&self) -> Vector2<i32> {
        self.max - self.min
    }

    /// Translates the box by `t`.
    pub fn translate(&mut self, t: &Vector2<i32>) {
        self.min += t;
        self.max += t;
    }
}

/// Shared state and non-virtual methods of 2D grids (see module docs).
#[derive(Clone)]
pub struct Grid2D {
    limits: MapLimits,
    correspondence_cost_cells: Vec<u16>,
    min_correspondence_cost: f32,
    max_correspondence_cost: f32,
    // RUST PORT NOTE: `std::vector<int>` in C++; flat indices are `usize` here.
    update_indices: Vec<usize>,
    // Bounding box of known cells to efficiently compute cropping limits.
    known_cells_box: AlignedBox2i,
    value_to_correspondence_cost_table: Arc<[f32]>,
}

impl fmt::Debug for Grid2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Grid2D")
            .field("limits", &self.limits)
            .field("num_cells", &self.correspondence_cost_cells.len())
            .field("min_correspondence_cost", &self.min_correspondence_cost)
            .field("max_correspondence_cost", &self.max_correspondence_cost)
            .field("num_update_indices", &self.update_indices.len())
            .field("known_cells_box", &self.known_cells_box)
            .finish_non_exhaustive()
    }
}

fn min_correspondence_cost_from_proto(proto: &proto::Grid2D) -> f32 {
    if proto.min_correspondence_cost == 0.0 && proto.max_correspondence_cost == 0.0 {
        log::warn!(
            "proto::Grid2D: min_correspondence_cost is initialized with 0 indicating an older \
             version of the protobuf format. Loading default values."
        );
        MIN_CORRESPONDENCE_COST
    } else {
        proto.min_correspondence_cost
    }
}

fn max_correspondence_cost_from_proto(proto: &proto::Grid2D) -> f32 {
    if proto.min_correspondence_cost == 0.0 && proto.max_correspondence_cost == 0.0 {
        log::warn!(
            "proto::Grid2D: max_correspondence_cost is initialized with 0 indicating an older \
             version of the protobuf format. Loading default values."
        );
        MAX_CORRESPONDENCE_COST
    } else {
        proto.max_correspondence_cost
    }
}

impl Grid2D {
    /// Creates an all-unknown grid. The conversion table maps unknown cells
    /// to `max_correspondence_cost`.
    pub fn new(
        limits: MapLimits,
        min_correspondence_cost: f32,
        max_correspondence_cost: f32,
        conversion_tables: &ValueConversionTables,
    ) -> Self {
        let num_cells =
            limits.cell_limits().num_x_cells as usize * limits.cell_limits().num_y_cells as usize;
        let grid = Self {
            limits,
            correspondence_cost_cells: vec![UNKNOWN_CORRESPONDENCE_VALUE; num_cells],
            min_correspondence_cost,
            max_correspondence_cost,
            update_indices: Vec::new(),
            known_cells_box: AlignedBox2i::default(),
            value_to_correspondence_cost_table: conversion_tables.get_conversion_table(
                max_correspondence_cost,
                min_correspondence_cost,
                max_correspondence_cost,
            ),
        };
        grid.check_correspondence_cost_bounds();
        grid
    }

    /// Creates a grid from a proto. Legacy protos with both min and max
    /// correspondence cost equal to 0 get the default bounds. The number of
    /// cells is not validated against the limits (like C++).
    pub fn from_proto(proto: &proto::Grid2D, conversion_tables: &ValueConversionTables) -> Self {
        let min_correspondence_cost = min_correspondence_cost_from_proto(proto);
        let max_correspondence_cost = max_correspondence_cost_from_proto(proto);
        let known_cells_box = match &proto.known_cells_box {
            Some(b) => AlignedBox2i::new(
                Vector2::new(b.min_x, b.min_y),
                Vector2::new(b.max_x, b.max_y),
            ),
            None => AlignedBox2i::default(),
        };
        let correspondence_cost_cells = proto
            .cells
            .iter()
            .map(|&cell| {
                assert!(
                    cell <= i32::from(u16::MAX),
                    "Check failed: cell <= std::numeric_limits<uint16>::max() ({cell} vs. 65535)"
                );
                // Truncating like the implicit int32 -> uint16 conversion in C++.
                cell as u16
            })
            .collect();
        let grid = Self {
            limits: MapLimits::from_proto(&proto.limits.unwrap_or_default()),
            correspondence_cost_cells,
            min_correspondence_cost,
            max_correspondence_cost,
            update_indices: Vec::new(),
            known_cells_box,
            value_to_correspondence_cost_table: conversion_tables.get_conversion_table(
                max_correspondence_cost,
                min_correspondence_cost,
                max_correspondence_cost,
            ),
        };
        grid.check_correspondence_cost_bounds();
        grid
    }

    fn check_correspondence_cost_bounds(&self) {
        assert!(
            self.min_correspondence_cost < self.max_correspondence_cost,
            "Check failed: min_correspondence_cost_ < max_correspondence_cost_ ({} vs. {})",
            self.min_correspondence_cost,
            self.max_correspondence_cost
        );
    }

    /// Returns the limits of this grid.
    #[inline]
    pub fn limits(&self) -> &MapLimits {
        &self.limits
    }

    /// Finishes the update sequence: removes the update marker from all
    /// cells updated since the last call.
    pub fn finish_update(&mut self) {
        while let Some(index) = self.update_indices.pop() {
            debug_assert!(self.correspondence_cost_cells[index] >= UPDATE_MARKER);
            self.correspondence_cost_cells[index] -= UPDATE_MARKER;
        }
    }

    /// Returns the correspondence cost of the cell with `cell_index`
    /// (`max_correspondence_cost` outside the limits; the update marker is
    /// tolerated).
    #[inline]
    pub fn get_correspondence_cost(&self, cell_index: &Array2i) -> f32 {
        if !self.limits.contains(cell_index) {
            return self.max_correspondence_cost;
        }
        self.value_to_correspondence_cost_table
            [self.correspondence_cost_cells[self.to_flat_index(cell_index)] as usize]
    }

    /// Returns the minimum possible correspondence cost.
    #[inline]
    pub fn get_min_correspondence_cost(&self) -> f32 {
        self.min_correspondence_cost
    }

    /// Returns the maximum possible correspondence cost.
    #[inline]
    pub fn get_max_correspondence_cost(&self) -> f32 {
        self.max_correspondence_cost
    }

    /// Returns true if the probability at the specified index is known.
    #[inline]
    pub fn is_known(&self, cell_index: &Array2i) -> bool {
        self.limits.contains(cell_index)
            && self.correspondence_cost_cells[self.to_flat_index(cell_index)]
                != UNKNOWN_CORRESPONDENCE_VALUE
    }

    /// Returns `(offset, limits)` defining a subregion that contains all
    /// known cells (C++ fills two out-parameters). Without known cells this is
    /// `((0, 0), 1x1)`.
    pub fn compute_cropped_limits(&self) -> (Array2i, CellLimits) {
        if self.known_cells_box.is_empty() {
            return (Array2i::zeros(), CellLimits::new(1, 1));
        }
        let sizes = self.known_cells_box.sizes();
        (
            *self.known_cells_box.min(),
            CellLimits::new(sizes.x + 1, sizes.y + 1),
        )
    }

    /// Grows the map as necessary to include `point`, growing only the
    /// correspondence cost cells (C++ `Grid2D::GrowLimits(point)`, the base
    /// implementation of the virtual method). This changes the meaning of
    /// these coordinates going forward. This method must be called
    /// immediately after `finish_update`, before any calls to
    /// `apply_lookup_table`.
    ///
    /// Prefer [`Grid2DTrait::grow_limits`], which is overridden by grids
    /// holding additional per-cell arrays (TSDF weights).
    pub fn grow_limits(&mut self, point: &Vector2<f32>) {
        self.grow_limits_with(point, &mut []);
    }

    /// Protected C++ `GrowLimits(point, grids, grids_unknown_cell_values)`.
    ///
    /// RUST PORT NOTE: the correspondence cost cells (unknown value
    /// `UNKNOWN_CORRESPONDENCE_VALUE`) are always grown; `extra_grids` lists
    /// additional per-cell arrays with their unknown value (C++ passes the
    /// own cells explicitly as the first entry).
    pub fn grow_limits_with(
        &mut self,
        point: &Vector2<f32>,
        extra_grids: &mut [(&mut Vec<u16>, u16)],
    ) {
        assert!(
            self.update_indices.is_empty(),
            "Check failed: update_indices_.empty()"
        );
        while !self.limits.contains(&self.limits.get_cell_index(point)) {
            let num_x_cells = self.limits.cell_limits().num_x_cells;
            let num_y_cells = self.limits.cell_limits().num_y_cells;
            let x_offset = num_x_cells / 2;
            let y_offset = num_y_cells / 2;
            let new_limits = MapLimits::new(
                self.limits.resolution(),
                self.limits.max()
                    + self.limits.resolution()
                        * Vector2::new(f64::from(y_offset), f64::from(x_offset)),
                CellLimits::new(2 * num_x_cells, 2 * num_y_cells),
            );
            let stride = new_limits.cell_limits().num_x_cells as usize;
            let offset = x_offset as usize + stride * y_offset as usize;
            let new_size = new_limits.cell_limits().num_x_cells as usize
                * new_limits.cell_limits().num_y_cells as usize;
            let grow = |cells: &mut Vec<u16>, unknown: u16| {
                let mut new_cells = vec![unknown; new_size];
                for i in 0..num_y_cells as usize {
                    for j in 0..num_x_cells as usize {
                        new_cells[offset + j + i * stride] = cells[j + i * num_x_cells as usize];
                    }
                }
                *cells = new_cells;
            };
            grow(
                &mut self.correspondence_cost_cells,
                UNKNOWN_CORRESPONDENCE_VALUE,
            );
            for (cells, unknown) in extra_grids.iter_mut() {
                grow(cells, *unknown);
            }
            self.limits = new_limits;
            if !self.known_cells_box.is_empty() {
                self.known_cells_box
                    .translate(&Vector2::new(x_offset, y_offset));
            }
        }
    }

    /// Serializes the shared grid state (C++ `Grid2D::ToProto`; the `grid`
    /// oneof is left unset). Panics during an unfinished update.
    pub fn to_proto(&self) -> proto::Grid2D {
        assert!(
            self.update_indices.is_empty(),
            "Serializing a grid during an update is not supported. Finish the update first."
        );
        let known_cells_box = if self.known_cells_box.is_empty() {
            None
        } else {
            Some(proto::grid2_d::CellBox {
                max_x: self.known_cells_box.max().x,
                max_y: self.known_cells_box.max().y,
                min_x: self.known_cells_box.min().x,
                min_y: self.known_cells_box.min().y,
            })
        };
        proto::Grid2D {
            limits: Some(map_limits::to_proto(&self.limits)),
            cells: self
                .correspondence_cost_cells
                .iter()
                .map(|&c| i32::from(c))
                .collect(),
            known_cells_box,
            min_correspondence_cost: self.min_correspondence_cost,
            max_correspondence_cost: self.max_correspondence_cost,
            grid: None,
        }
    }

    /// The raw cell values (protected in C++).
    #[inline]
    pub fn correspondence_cost_cells(&self) -> &[u16] {
        &self.correspondence_cost_cells
    }

    /// Flat indices of cells updated since the last `finish_update`
    /// (protected in C++).
    #[inline]
    pub fn update_indices(&self) -> &[usize] {
        &self.update_indices
    }

    /// Bounding box of known cells (protected in C++).
    #[inline]
    pub fn known_cells_box(&self) -> &AlignedBox2i {
        &self.known_cells_box
    }

    /// Mutable raw cell values (protected in C++; for grid implementations).
    #[inline]
    pub fn mutable_correspondence_cost_cells(&mut self) -> &mut Vec<u16> {
        &mut self.correspondence_cost_cells
    }

    /// Mutable update indices (protected in C++; for grid implementations).
    #[inline]
    pub fn mutable_update_indices(&mut self) -> &mut Vec<usize> {
        &mut self.update_indices
    }

    /// Mutable known cells box (protected in C++; for grid implementations).
    #[inline]
    pub fn mutable_known_cells_box(&mut self) -> &mut AlignedBox2i {
        &mut self.known_cells_box
    }

    /// Converts a `cell_index` into an index into the cell arrays. Panics if
    /// the cell is outside the limits (C++ `CHECK`, always on).
    #[inline]
    pub fn to_flat_index(&self, cell_index: &Array2i) -> usize {
        assert!(
            self.limits.contains(cell_index),
            "Check failed: limits_.Contains(cell_index) [{}, {}]",
            cell_index.x,
            cell_index.y
        );
        (self.limits.cell_limits().num_x_cells * cell_index.y + cell_index.x) as usize
    }
}

/// Computes the texture `slice_pose` shared by the `draw_to_submap_texture`
/// implementations: `local_pose⁻¹ * Translation(max_x, max_y, 0)` where
/// `max` is the corner of the cropped region starting at `offset`.
pub(crate) fn submap_texture_slice_pose(
    limits: &MapLimits,
    offset: &Array2i,
    local_pose: &Rigid3d,
) -> crate::transform::proto::Rigid3d {
    let resolution = limits.resolution();
    let max_x = limits.max().x - resolution * f64::from(offset.y);
    let max_y = limits.max().y - resolution * f64::from(offset.x);
    crate::transform::transform::to_proto(
        &(local_pose.inverse() * Rigid3d::from_translation(Vector3::new(max_x, max_y, 0.0))),
    )
}

/// Fills `texture` from per-cell `(value, alpha)` bytes (shared tail of the
/// `draw_to_submap_texture` implementations).
pub(crate) fn fill_submap_texture(
    texture: &mut SubmapTexture,
    cells: &[u8],
    cell_limits: &CellLimits,
    limits: &MapLimits,
    offset: &Array2i,
    local_pose: &Rigid3d,
) {
    texture.cells.clear();
    crate::common::port::fast_gzip_string(cells, &mut texture.cells);
    texture.width = cell_limits.num_x_cells;
    texture.height = cell_limits.num_y_cells;
    texture.resolution = limits.resolution();
    texture.slice_pose = Some(submap_texture_slice_pose(limits, offset, local_pose));
}

/// Virtual interface of 2D grids (see module docs). Object safe; use
/// `Box<dyn Grid2DTrait>` where C++ uses `std::unique_ptr<Grid2D>`.
pub trait Grid2DTrait: GridInterface {
    /// The shared grid state.
    fn grid_2d(&self) -> &Grid2D;

    /// The shared grid state, mutably. Beware that
    /// `grid_2d_mut().grow_limits()` bypasses overrides of
    /// [`grow_limits`](Self::grow_limits).
    fn grid_2d_mut(&mut self) -> &mut Grid2D;

    /// C++ `GetGridType`.
    fn get_grid_type(&self) -> GridType;

    /// Grows the map as necessary to include `point` (C++ virtual
    /// `GrowLimits`). Must be called after `finish_update`.
    fn grow_limits(&mut self, point: &Vector2<f32>) {
        self.grid_2d_mut().grow_limits(point);
    }

    /// Returns a copy of the grid cropped to the known cells.
    fn compute_cropped_grid(&self) -> Box<dyn Grid2DTrait>;

    /// Serializes the grid, including the `grid` oneof.
    fn to_proto(&self) -> proto::Grid2D;

    /// Renders the grid into `texture` (gzipped `(value, alpha)` pairs).
    fn draw_to_submap_texture(&self, texture: &mut SubmapTexture, local_pose: Rigid3d) -> bool;

    // Non-virtual `Grid2D` API, forwarded for convenience.

    /// See [`Grid2D::limits`].
    fn limits(&self) -> &MapLimits {
        self.grid_2d().limits()
    }

    /// See [`Grid2D::finish_update`].
    fn finish_update(&mut self) {
        self.grid_2d_mut().finish_update();
    }

    /// See [`Grid2D::get_correspondence_cost`].
    fn get_correspondence_cost(&self, cell_index: &Array2i) -> f32 {
        self.grid_2d().get_correspondence_cost(cell_index)
    }

    /// See [`Grid2D::get_min_correspondence_cost`].
    fn get_min_correspondence_cost(&self) -> f32 {
        self.grid_2d().get_min_correspondence_cost()
    }

    /// See [`Grid2D::get_max_correspondence_cost`].
    fn get_max_correspondence_cost(&self) -> f32 {
        self.grid_2d().get_max_correspondence_cost()
    }

    /// See [`Grid2D::is_known`].
    fn is_known(&self, cell_index: &Array2i) -> bool {
        self.grid_2d().is_known(cell_index)
    }

    /// See [`Grid2D::compute_cropped_limits`].
    fn compute_cropped_limits(&self) -> (Array2i, CellLimits) {
        self.grid_2d().compute_cropped_limits()
    }
}

impl dyn Grid2DTrait {
    /// Returns the concrete grid if it is of type `T` (C++ `static_cast`).
    pub fn downcast_ref<T: Grid2DTrait>(&self) -> Option<&T> {
        (self as &dyn std::any::Any).downcast_ref::<T>()
    }

    /// Returns the concrete grid if it is of type `T` (C++ `static_cast`).
    pub fn downcast_mut<T: Grid2DTrait>(&mut self) -> Option<&mut T> {
        (self as &mut dyn std::any::Any).downcast_mut::<T>()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector2;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    #[test]
    fn create_grid_options_2d_test() {
        let dict = make_dictionary("return { grid_type = \"TSDF\", resolution = 0.05 }");
        let options = create_grid_options_2d(&dict);
        assert_eq!(options.grid_type(), proto::grid_options2_d::GridType::Tsdf);
        assert_eq!(options.resolution, 0.05_f32);
    }

    #[test]
    #[should_panic(expected = "Unknown GridOptions2D_GridType kind")]
    fn create_grid_options_2d_unknown_type() {
        let dict = make_dictionary("return { grid_type = \"FOO\", resolution = 0.05 }");
        let _ = create_grid_options_2d(&dict);
    }

    #[test]
    fn aligned_box() {
        let mut b = AlignedBox2i::default();
        assert!(b.is_empty());
        b.extend(&Vector2::new(3, 4));
        assert!(!b.is_empty());
        assert_eq!(b.sizes(), Vector2::new(0, 0));
        b.extend(&Vector2::new(1, 7));
        assert_eq!(*b.min(), Vector2::new(1, 4));
        assert_eq!(*b.max(), Vector2::new(3, 7));
        b.translate(&Vector2::new(1, -1));
        assert_eq!(*b.min(), Vector2::new(2, 3));
        assert!(AlignedBox2i::new(Vector2::new(21, 22), Vector2::new(19, 20)).is_empty());
    }

    #[test]
    fn grow_limits_keeps_content_centered() {
        let tables = ValueConversionTables::new();
        let mut grid = Grid2D::new(
            MapLimits::new(1.0, Vector2::new(2.0, 2.0), CellLimits::new(4, 4)),
            MIN_CORRESPONDENCE_COST,
            MAX_CORRESPONDENCE_COST,
            &tables,
        );
        let cell = Array2i::new(1, 2);
        let flat = grid.to_flat_index(&cell);
        grid.mutable_correspondence_cost_cells()[flat] = 1234;
        grid.mutable_known_cells_box().extend(&cell);
        let world = grid.limits().get_cell_center(&cell);
        grid.grow_limits(&Vector2::new(10.0, 10.0));
        assert!(
            grid.limits()
                .contains(&grid.limits().get_cell_index(&Vector2::new(10.0, 10.0)))
        );
        let new_cell = grid.limits().get_cell_index(&world);
        assert_eq!(
            grid.correspondence_cost_cells()[grid.to_flat_index(&new_cell)],
            1234
        );
        assert_eq!(grid.compute_cropped_limits().0, new_cell);
        let known = grid
            .correspondence_cost_cells()
            .iter()
            .filter(|&&c| c != 0)
            .count();
        assert_eq!(known, 1);
    }

    #[test]
    #[should_panic(expected = "update_indices_.empty()")]
    fn grow_limits_during_update_panics() {
        let tables = ValueConversionTables::new();
        let mut grid = Grid2D::new(
            MapLimits::new(1.0, Vector2::new(2.0, 2.0), CellLimits::new(4, 4)),
            MIN_CORRESPONDENCE_COST,
            MAX_CORRESPONDENCE_COST,
            &tables,
        );
        grid.mutable_update_indices().push(0);
        grid.grow_limits(&Vector2::new(10.0, 10.0));
    }

    #[test]
    fn legacy_proto_gets_default_bounds() {
        let tables = ValueConversionTables::new();
        let grid = Grid2D::new(
            MapLimits::new(1.0, Vector2::new(2.0, 2.0), CellLimits::new(2, 2)),
            MIN_CORRESPONDENCE_COST,
            MAX_CORRESPONDENCE_COST,
            &tables,
        );
        let mut proto = grid.to_proto();
        proto.min_correspondence_cost = 0.0;
        proto.max_correspondence_cost = 0.0;
        let restored = Grid2D::from_proto(&proto, &tables);
        assert_eq!(
            restored.get_min_correspondence_cost(),
            MIN_CORRESPONDENCE_COST
        );
        assert_eq!(
            restored.get_max_correspondence_cost(),
            MAX_CORRESPONDENCE_COST
        );
    }
}
