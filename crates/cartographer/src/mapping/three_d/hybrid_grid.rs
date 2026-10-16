//! Port of `cartographer/mapping/3d/hybrid_grid` (.h).
//!
//! A 3D grid represented as a wide, shallow tree:
//! [`DynamicGrid`] (growable, signed indices) of [`NestedGrid`]s (lazily
//! allocated) of [`FlatGrid`]s (contiguous voxel arrays).
//!
//! RUST PORT NOTE: C++ class inheritance (`HybridGrid : HybridGridBase<uint16>
//! : DynamicGrid<...>`) is modelled with composition plus `Deref`/`DerefMut`,
//! so e.g. `hybrid_grid.mutable_value(..)`, `hybrid_grid.grid_size()` and
//! `hybrid_grid.iter()` resolve to the base classes like in C++.

use std::ops::{Deref, DerefMut};

use nalgebra::Vector3;

use crate::common::port::round_to_int;
use crate::mapping::proto;

use self::probability_values_shim::{UPDATE_MARKER, probability_to_value, value_to_probability};

// TEMPORARY: replace with crate::mapping::probability_values after merge.
//
// Minimal, exact copy of the parts of `mapping/probability_values.{h,cc}`
// needed by the hybrid grid (and its tests). `mapping/probability_values.rs`
// is being ported in parallel; swap the `use` above (and in the tests) for
// `crate::mapping::probability_values::*` and delete this module at merge.
#[allow(dead_code)]
mod probability_values_shim {
    use crate::common::math::clamp;
    use crate::common::port::round_to_int;

    const VALUE_COUNT: i32 = 32768;

    pub const MIN_PROBABILITY: f32 = 0.1;
    pub const MAX_PROBABILITY: f32 = 1.0 - MIN_PROBABILITY;
    pub const UNKNOWN_PROBABILITY_VALUE: u16 = 0;
    pub const UPDATE_MARKER: u16 = 1 << 15;

    pub fn odds(probability: f32) -> f32 {
        probability / (1.0 - probability)
    }

    pub fn probability_from_odds(odds: f32) -> f32 {
        odds / (odds + 1.0)
    }

    #[inline]
    fn bounded_float_to_value(float_value: f32, lower_bound: f32, upper_bound: f32) -> u16 {
        let value = round_to_int(
            (clamp(float_value, lower_bound, upper_bound) - lower_bound)
                * (32766.0f32 / (upper_bound - lower_bound)),
        ) + 1;
        debug_assert!(value >= 1);
        debug_assert!(value <= 32767);
        value as u16
    }

    #[inline]
    pub fn probability_to_value(probability: f32) -> u16 {
        bounded_float_to_value(probability, MIN_PROBABILITY, MAX_PROBABILITY)
    }

    /// Same as C++ `SlowValueToBoundedFloat` for the probability table; the
    /// C++ table repeats twice, so the update marker bit is ignored.
    #[inline]
    pub fn value_to_probability(value: u16) -> f32 {
        let value = value & (UPDATE_MARKER - 1);
        if value == UNKNOWN_PROBABILITY_VALUE {
            return MIN_PROBABILITY;
        }
        let scale = (MAX_PROBABILITY - MIN_PROBABILITY) / (VALUE_COUNT as f32 - 2.0);
        f32::from(value) * scale + (MIN_PROBABILITY - scale)
    }

    pub fn compute_lookup_table_to_apply_odds(odds_value: f32) -> Vec<u16> {
        let mut result = Vec::with_capacity(VALUE_COUNT as usize);
        result.push(probability_to_value(probability_from_odds(odds_value)) + UPDATE_MARKER);
        for cell in 1..VALUE_COUNT {
            result.push(
                probability_to_value(probability_from_odds(
                    odds_value * odds(value_to_probability(cell as u16)),
                )) + UPDATE_MARKER,
            );
        }
        result
    }
}

/// `Eigen::Array3i`.
pub type Array3i = Vector3<i32>;

/// Converts an `index` with each dimension from 0 to 2^`bits` - 1 to a flat
/// z-major index.
#[inline]
pub fn to_flat_index(index: &Array3i, bits: i32) -> i32 {
    debug_assert!(
        index.iter().all(|&c| c >= 0 && c < (1 << bits)),
        "{index:?}"
    );
    (((index.z << bits) + index.y) << bits) + index.x
}

/// Converts a flat z-major `index` to a 3-dimensional index with each
/// dimension from 0 to 2^`bits` - 1.
#[inline]
pub fn to_3d_index(index: i32, bits: i32) -> Array3i {
    debug_assert!(index < 1 << (3 * bits));
    let mask = (1 << bits) - 1;
    Array3i::new(
        index & mask,
        (index >> bits) & mask,
        (index >> bits) >> bits,
    )
}

/// Value types storable in the grids. `is_default_value` replaces the C++
/// `IsDefaultValue` function template (and its `std::vector` specialization):
/// iterators skip cells for which it returns true.
pub trait GridValue: Default + Clone {
    /// Returns true if this compares equal to the default value.
    fn is_default_value(&self) -> bool;
}

/// C++ `IsDefaultValue(v)`.
#[inline]
pub fn is_default_value<T: GridValue>(v: &T) -> bool {
    v.is_default_value()
}

macro_rules! impl_grid_value_for_primitive {
    ($($t:ty),*) => {
        $(impl GridValue for $t {
            #[inline]
            fn is_default_value(&self) -> bool {
                *self == <$t>::default()
            }
        })*
    };
}
impl_grid_value_for_primitive!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<E: Clone> GridValue for Vec<E> {
    #[inline]
    fn is_default_value(&self) -> bool {
        self.is_empty()
    }
}

/// Common iterator interface of the grid iterators (the C++ nested
/// `Iterator` classes). Iterates over all values not comparing equal to the
/// default value.
///
/// RUST PORT NOTE: C++ `Next()` is `advance()` here, so that the grid
/// iterators can also implement `std::iter::Iterator` (yielding
/// `(cell_index, &value)`, like the C++ range-based for loop).
pub trait GridIterator<'a> {
    /// Value type of the iterated grid.
    type ValueType: 'a;
    /// C++ `Done()`.
    fn done(&self) -> bool;
    /// C++ `Next()`.
    fn advance(&mut self);
    /// C++ `GetCellIndex()`.
    fn get_cell_index(&self) -> Array3i;
    /// C++ `GetValue()`.
    fn get_value(&self) -> &'a Self::ValueType;
}

/// A grid that can be wrapped by [`NestedGrid`] / [`DynamicGrid`] (the C++
/// `WrappedGrid` template parameter concept). `Default` constructs a grid
/// with all values default constructed.
pub trait WrappedGrid: Default {
    /// Value type stored in the grid.
    type ValueType: GridValue;
    /// Iterator type.
    type Iter<'a>: GridIterator<'a, ValueType = Self::ValueType>
    where
        Self: 'a;
    /// Number of voxels per dimension.
    fn grid_size() -> i32;
    /// Returns a reference to the value at `index` (each dimension between 0
    /// and `grid_size() - 1`), or `None` if the containing storage has not
    /// been allocated yet (meaning the value is the default value).
    fn get(&self, index: &Array3i) -> Option<&Self::ValueType>;
    /// Returns a mutable reference to the value at `index`, allocating
    /// storage as necessary.
    fn mutable_value(&mut self, index: &Array3i) -> &mut Self::ValueType;
    /// Returns an iterator over all non-default values.
    fn iter(&self) -> Self::Iter<'_>;
}

// ---------------------------------------------------------------------------
// FlatGrid
// ---------------------------------------------------------------------------

/// A flat grid of `2^BITS` x `2^BITS` x `2^BITS` voxels storing values of
/// type `T` in contiguous memory. Indices in each dimension are 0-based.
#[derive(Clone)]
pub struct FlatGrid<T: GridValue, const BITS: i32> {
    // RUST PORT NOTE: `std::array` member -> boxed slice so that a
    // `Option<FlatGrid>` in the parent is a single (nullable) pointer, like
    // the C++ `std::unique_ptr<FlatGrid>`.
    cells: Box<[T]>,
}

impl<T: GridValue, const BITS: i32> Default for FlatGrid<T, BITS> {
    /// Creates a new flat grid with all values being default constructed.
    fn default() -> Self {
        Self {
            cells: vec![T::default(); 1usize << (3 * BITS)].into_boxed_slice(),
        }
    }
}

impl<T: GridValue, const BITS: i32> FlatGrid<T, BITS> {
    /// Creates a new flat grid with all values being default constructed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of voxels per dimension.
    #[inline]
    pub fn grid_size() -> i32 {
        1 << BITS
    }

    /// Returns the value stored at `index`, each dimension of `index` being
    /// between 0 and `grid_size() - 1`.
    #[inline]
    pub fn value(&self, index: &Array3i) -> T {
        self.cells[to_flat_index(index, BITS) as usize].clone()
    }

    /// Returns a mutable reference to the value at `index`.
    #[inline]
    pub fn mutable_value(&mut self, index: &Array3i) -> &mut T {
        &mut self.cells[to_flat_index(index, BITS) as usize]
    }

    /// Returns an iterator over all non-default values.
    pub fn iter(&self) -> FlatGridIterator<'_, T, BITS> {
        FlatGridIterator::new(self)
    }
}

impl<T: GridValue, const BITS: i32> WrappedGrid for FlatGrid<T, BITS> {
    type ValueType = T;
    type Iter<'a>
        = FlatGridIterator<'a, T, BITS>
    where
        Self: 'a;

    #[inline]
    fn grid_size() -> i32 {
        1 << BITS
    }

    #[inline]
    fn get(&self, index: &Array3i) -> Option<&T> {
        Some(&self.cells[to_flat_index(index, BITS) as usize])
    }

    #[inline]
    fn mutable_value(&mut self, index: &Array3i) -> &mut T {
        &mut self.cells[to_flat_index(index, BITS) as usize]
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        FlatGridIterator::new(self)
    }
}

/// Iterator over all values of a [`FlatGrid`] not comparing equal to the
/// default value, in flat z-major order.
pub struct FlatGridIterator<'a, T: GridValue, const BITS: i32> {
    cells: &'a [T],
    current: usize,
}

impl<'a, T: GridValue, const BITS: i32> FlatGridIterator<'a, T, BITS> {
    /// C++ `Iterator(const FlatGrid&)`.
    pub fn new(flat_grid: &'a FlatGrid<T, BITS>) -> Self {
        let mut it = Self {
            cells: &flat_grid.cells,
            current: 0,
        };
        while !it.done() && it.cells[it.current].is_default_value() {
            it.current += 1;
        }
        it
    }
}

impl<'a, T: GridValue, const BITS: i32> GridIterator<'a> for FlatGridIterator<'a, T, BITS> {
    type ValueType = T;

    #[inline]
    fn done(&self) -> bool {
        self.current == self.cells.len()
    }

    #[inline]
    fn advance(&mut self) {
        debug_assert!(!self.done());
        loop {
            self.current += 1;
            if self.done() || !self.cells[self.current].is_default_value() {
                break;
            }
        }
    }

    #[inline]
    fn get_cell_index(&self) -> Array3i {
        debug_assert!(!self.done());
        to_3d_index(self.current as i32, BITS)
    }

    #[inline]
    fn get_value(&self) -> &'a T {
        debug_assert!(!self.done());
        &self.cells[self.current]
    }
}

// ---------------------------------------------------------------------------
// NestedGrid
// ---------------------------------------------------------------------------

/// A grid consisting of `2^BITS` x `2^BITS` x `2^BITS` grids of type `W`.
/// Wrapped grids are constructed on first access via `mutable_value()`.
#[derive(Clone)]
pub struct NestedGrid<W: WrappedGrid, const BITS: i32> {
    meta_cells: Box<[Option<W>]>,
}

impl<W: WrappedGrid, const BITS: i32> Default for NestedGrid<W, BITS> {
    fn default() -> Self {
        Self {
            meta_cells: (0..1usize << (3 * BITS)).map(|_| None).collect(),
        }
    }
}

impl<W: WrappedGrid, const BITS: i32> NestedGrid<W, BITS> {
    /// Creates an empty nested grid.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of voxels per dimension.
    #[inline]
    pub fn grid_size() -> i32 {
        W::grid_size() << BITS
    }

    /// Returns the value stored at `index`, each dimension of `index` being
    /// between 0 and `grid_size() - 1`.
    #[inline]
    pub fn value(&self, index: &Array3i) -> W::ValueType {
        <Self as WrappedGrid>::get(self, index)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns a mutable reference to the value at `index`. If necessary a
    /// new wrapped grid is constructed to contain that value.
    #[inline]
    pub fn mutable_value(&mut self, index: &Array3i) -> &mut W::ValueType {
        <Self as WrappedGrid>::mutable_value(self, index)
    }

    /// Returns an iterator over all non-default values.
    pub fn iter(&self) -> NestedGridIterator<'_, W, BITS> {
        NestedGridIterator::new(self)
    }

    /// Returns the (meta) index of the meta cell containing `index`.
    #[inline]
    fn get_meta_index(index: &Array3i) -> Array3i {
        debug_assert!(index.iter().all(|&c| c >= 0), "{index:?}");
        // Indices are non-negative, so unsigned division equals the C++
        // truncating division (and compiles to a shift).
        let size = W::grid_size() as u32;
        let meta_index = index.map(|c| (c as u32 / size) as i32);
        debug_assert!(meta_index.iter().all(|&c| c < (1 << BITS)), "{index:?}");
        meta_index
    }
}

impl<W: WrappedGrid, const BITS: i32> WrappedGrid for NestedGrid<W, BITS> {
    type ValueType = W::ValueType;
    type Iter<'a>
        = NestedGridIterator<'a, W, BITS>
    where
        Self: 'a;

    #[inline]
    fn grid_size() -> i32 {
        W::grid_size() << BITS
    }

    #[inline]
    fn get(&self, index: &Array3i) -> Option<&W::ValueType> {
        let meta_index = Self::get_meta_index(index);
        let meta_cell = self.meta_cells[to_flat_index(&meta_index, BITS) as usize].as_ref()?;
        let inner_index = index - meta_index * W::grid_size();
        meta_cell.get(&inner_index)
    }

    #[inline]
    fn mutable_value(&mut self, index: &Array3i) -> &mut W::ValueType {
        let meta_index = Self::get_meta_index(index);
        let meta_cell = self.meta_cells[to_flat_index(&meta_index, BITS) as usize]
            .get_or_insert_with(W::default);
        let inner_index = index - meta_index * W::grid_size();
        meta_cell.mutable_value(&inner_index)
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        NestedGridIterator::new(self)
    }
}

/// Iterator over all values of a [`NestedGrid`] not comparing equal to the
/// default value: meta cells in flat z-major order, and within each meta cell
/// in the wrapped grid's order.
pub struct NestedGridIterator<'a, W: WrappedGrid + 'a, const BITS: i32> {
    meta_cells: &'a [Option<W>],
    current: usize,
    // RUST PORT NOTE: C++ default-constructs the nested iterator; `None`
    // here (it is always `Some` while `!done()`).
    nested_iterator: Option<W::Iter<'a>>,
}

impl<'a, W: WrappedGrid + 'a, const BITS: i32> NestedGridIterator<'a, W, BITS> {
    /// C++ `Iterator(const NestedGrid&)`.
    pub fn new(nested_grid: &'a NestedGrid<W, BITS>) -> Self {
        let mut it = Self {
            meta_cells: &nested_grid.meta_cells,
            current: 0,
            nested_iterator: None,
        };
        it.advance_to_valid_nested_iterator();
        it
    }

    fn advance_to_valid_nested_iterator(&mut self) {
        let meta_cells = self.meta_cells;
        while !self.done() {
            if let Some(meta_cell) = &meta_cells[self.current] {
                let nested_iterator = meta_cell.iter();
                if !nested_iterator.done() {
                    self.nested_iterator = Some(nested_iterator);
                    break;
                }
            }
            self.current += 1;
        }
    }

    #[inline]
    fn nested(&self) -> &W::Iter<'a> {
        self.nested_iterator
            .as_ref()
            .expect("nested iterator must be valid while not done")
    }
}

impl<'a, W: WrappedGrid + 'a, const BITS: i32> GridIterator<'a>
    for NestedGridIterator<'a, W, BITS>
{
    type ValueType = W::ValueType;

    #[inline]
    fn done(&self) -> bool {
        self.current == self.meta_cells.len()
    }

    #[inline]
    fn advance(&mut self) {
        debug_assert!(!self.done());
        let nested_iterator = self
            .nested_iterator
            .as_mut()
            .expect("nested iterator must be valid while not done");
        nested_iterator.advance();
        if !nested_iterator.done() {
            return;
        }
        self.current += 1;
        self.advance_to_valid_nested_iterator();
    }

    #[inline]
    fn get_cell_index(&self) -> Array3i {
        debug_assert!(!self.done());
        to_3d_index(self.current as i32, BITS) * W::grid_size() + self.nested().get_cell_index()
    }

    #[inline]
    fn get_value(&self) -> &'a W::ValueType {
        debug_assert!(!self.done());
        self.nested().get_value()
    }
}

// ---------------------------------------------------------------------------
// DynamicGrid
// ---------------------------------------------------------------------------

/// A grid consisting of 2x2x2 grids of type `W` initially. Wrapped grids are
/// constructed on first access via `mutable_value()`. If necessary, the grid
/// grows to twice the size in each dimension. The range of indices is
/// (almost) symmetric around the origin, i.e. negative indices are allowed.
#[derive(Clone)]
pub struct DynamicGrid<W: WrappedGrid> {
    bits: i32,
    meta_cells: Vec<Option<W>>,
}

impl<W: WrappedGrid> Default for DynamicGrid<W> {
    fn default() -> Self {
        Self {
            bits: 1,
            meta_cells: (0..8).map(|_| None).collect(),
        }
    }
}

impl<W: WrappedGrid> DynamicGrid<W> {
    /// Creates an empty grid (2x2x2 unallocated wrapped grids).
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current number of voxels per dimension.
    #[inline]
    pub fn grid_size(&self) -> i32 {
        W::grid_size() << self.bits
    }

    /// Returns `index` shifted into `[0, grid_size())`, or `None` if it is
    /// outside the current grid.
    #[inline]
    fn shifted_index(&self, index: &Array3i) -> Option<Array3i> {
        let grid_size = self.grid_size();
        let half = grid_size >> 1;
        let shifted_index = index.map(|c| c.wrapping_add(half));
        // The cast to unsigned is for performance to check with 3 comparisons
        // shifted_index.[xyz] >= 0 and shifted_index.[xyz] < grid_size.
        if shifted_index.iter().any(|&c| c as u32 >= grid_size as u32) {
            None
        } else {
            Some(shifted_index)
        }
    }

    /// Returns a reference to the value stored at `index`, or `None` if it
    /// lies in unallocated storage (i.e. it is the default value). Never
    /// allocates.
    ///
    /// RUST PORT NOTE: added so non-`Copy` values (e.g. `Vec`) can be read
    /// without cloning; [`DynamicGrid::value`] is the C++ API.
    #[inline]
    pub fn get(&self, index: &Array3i) -> Option<&W::ValueType> {
        let shifted_index = self.shifted_index(index)?;
        let meta_index = self.get_meta_index(&shifted_index);
        let meta_cell = self.meta_cells[to_flat_index(&meta_index, self.bits) as usize].as_ref()?;
        let inner_index = shifted_index - meta_index * W::grid_size();
        meta_cell.get(&inner_index)
    }

    /// Returns the value stored at `index` (the default value if outside the
    /// grid or not yet allocated).
    #[inline]
    pub fn value(&self, index: &Array3i) -> W::ValueType {
        self.get(index).cloned().unwrap_or_default()
    }

    /// Returns a mutable reference to the value at `index`, dynamically
    /// growing the grid and constructing new wrapped grids as needed.
    #[inline]
    pub fn mutable_value(&mut self, index: &Array3i) -> &mut W::ValueType {
        let shifted_index = loop {
            match self.shifted_index(index) {
                Some(shifted_index) => break shifted_index,
                None => self.grow(),
            }
        };
        let meta_index = self.get_meta_index(&shifted_index);
        let meta_cell = self.meta_cells[to_flat_index(&meta_index, self.bits) as usize]
            .get_or_insert_with(W::default);
        let inner_index = shifted_index - meta_index * W::grid_size();
        meta_cell.mutable_value(&inner_index)
    }

    /// Returns an iterator over all non-default values (C++
    /// `Iterator(*this)` / `begin()`).
    pub fn iter(&self) -> DynamicGridIterator<'_, W> {
        DynamicGridIterator::new(self)
    }

    /// Returns the (meta) index of the meta cell containing the shifted
    /// `index`.
    #[inline]
    fn get_meta_index(&self, index: &Array3i) -> Array3i {
        debug_assert!(index.iter().all(|&c| c >= 0), "{index:?}");
        let size = W::grid_size() as u32;
        let meta_index = index.map(|c| (c as u32 / size) as i32);
        debug_assert!(
            meta_index.iter().all(|&c| c < (1 << self.bits)),
            "{index:?}"
        );
        meta_index
    }

    /// Grows this grid by a factor of 2 in each of the 3 dimensions.
    #[cold]
    fn grow(&mut self) {
        let new_bits = self.bits + 1;
        assert!(
            new_bits <= 8,
            "Check failed: new_bits <= 8 ({new_bits} vs. 8): index out of the DynamicGrid range"
        );
        let mut new_meta_cells: Vec<Option<W>> =
            (0..8 * self.meta_cells.len()).map(|_| None).collect();
        let n = 1 << self.bits;
        let offset = 1 << (self.bits - 1);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let original_meta_index = Array3i::new(x, y, z);
                    let new_meta_index = original_meta_index.add_scalar(offset);
                    new_meta_cells[to_flat_index(&new_meta_index, new_bits) as usize] =
                        self.meta_cells[to_flat_index(&original_meta_index, self.bits) as usize]
                            .take();
                }
            }
        }
        self.meta_cells = new_meta_cells;
        self.bits = new_bits;
    }
}

impl<'a, W: WrappedGrid> IntoIterator for &'a DynamicGrid<W> {
    type Item = (Array3i, &'a W::ValueType);
    type IntoIter = DynamicGridIterator<'a, W>;
    fn into_iter(self) -> Self::IntoIter {
        DynamicGridIterator::new(self)
    }
}

/// Iterator over all values of a [`DynamicGrid`] not comparing equal to the
/// default value. As a `std::iter::Iterator` it yields `(cell_index, &value)`
/// (the C++ range-based for loop yields `std::pair<Array3i, ValueType>`).
///
/// Order is hierarchical (root meta cells in flat z-major order, then the
/// wrapped grid's order); it is load-bearing for `HybridGrid::to_proto`.
pub struct DynamicGridIterator<'a, W: WrappedGrid + 'a> {
    bits: i32,
    meta_cells: &'a [Option<W>],
    current: usize,
    nested_iterator: Option<W::Iter<'a>>,
}

impl<'a, W: WrappedGrid + 'a> DynamicGridIterator<'a, W> {
    /// C++ `Iterator(const DynamicGrid&)`.
    pub fn new(dynamic_grid: &'a DynamicGrid<W>) -> Self {
        let mut it = Self {
            bits: dynamic_grid.bits,
            meta_cells: &dynamic_grid.meta_cells,
            current: 0,
            nested_iterator: None,
        };
        it.advance_to_valid_nested_iterator();
        it
    }

    /// C++ `AdvanceToEnd()`.
    pub fn advance_to_end(&mut self) {
        self.current = self.meta_cells.len();
    }

    fn advance_to_valid_nested_iterator(&mut self) {
        let meta_cells = self.meta_cells;
        while !self.done() {
            if let Some(meta_cell) = &meta_cells[self.current] {
                let nested_iterator = meta_cell.iter();
                if !nested_iterator.done() {
                    self.nested_iterator = Some(nested_iterator);
                    break;
                }
            }
            self.current += 1;
        }
    }

    #[inline]
    fn nested(&self) -> &W::Iter<'a> {
        self.nested_iterator
            .as_ref()
            .expect("nested iterator must be valid while not done")
    }
}

impl<'a, W: WrappedGrid + 'a> GridIterator<'a> for DynamicGridIterator<'a, W> {
    type ValueType = W::ValueType;

    #[inline]
    fn done(&self) -> bool {
        self.current == self.meta_cells.len()
    }

    #[inline]
    fn advance(&mut self) {
        debug_assert!(!self.done());
        let nested_iterator = self
            .nested_iterator
            .as_mut()
            .expect("nested iterator must be valid while not done");
        nested_iterator.advance();
        if !nested_iterator.done() {
            return;
        }
        self.current += 1;
        self.advance_to_valid_nested_iterator();
    }

    #[inline]
    fn get_cell_index(&self) -> Array3i {
        debug_assert!(!self.done());
        let shifted_index = to_3d_index(self.current as i32, self.bits) * W::grid_size()
            + self.nested().get_cell_index();
        shifted_index.add_scalar(-((1 << (self.bits - 1)) * W::grid_size()))
    }

    #[inline]
    fn get_value(&self) -> &'a W::ValueType {
        debug_assert!(!self.done());
        self.nested().get_value()
    }
}

macro_rules! impl_std_iterator {
    ($ty:ty, [$($generics:tt)*]) => {
        impl<$($generics)*> Iterator for $ty {
            type Item = (Array3i, &'a <Self as GridIterator<'a>>::ValueType);

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if self.done() {
                    return None;
                }
                let item = (self.get_cell_index(), self.get_value());
                self.advance();
                Some(item)
            }
        }
    };
}
impl_std_iterator!(FlatGridIterator<'a, T, BITS>, ['a, T: GridValue, const BITS: i32]);
impl_std_iterator!(NestedGridIterator<'a, W, BITS>, ['a, W: WrappedGrid + 'a, const BITS: i32]);
impl_std_iterator!(DynamicGridIterator<'a, W>, ['a, W: WrappedGrid + 'a]);

// ---------------------------------------------------------------------------
// HybridGridBase
// ---------------------------------------------------------------------------

/// C++ `GridBase<ValueType>`.
pub type GridBase<T> = DynamicGrid<NestedGrid<FlatGrid<T, 3>, 3>>;

/// C++ `HybridGridBase<ValueType>::Iterator`.
pub type HybridGridIterator<'a, T> = DynamicGridIterator<'a, NestedGrid<FlatGrid<T, 3>, 3>>;

/// Represents a 3D grid as a wide, shallow tree.
#[derive(Clone)]
pub struct HybridGridBase<T: GridValue> {
    grid: GridBase<T>,
    // Edge length of each voxel.
    resolution: f32,
}

impl<T: GridValue> HybridGridBase<T> {
    /// Creates a new tree-based probability grid with voxels having edge
    /// length `resolution` around the origin which becomes the center of the
    /// cell at index (0, 0, 0).
    pub fn new(resolution: f32) -> Self {
        Self {
            grid: GridBase::default(),
            resolution,
        }
    }

    /// Edge length of each voxel.
    #[inline]
    pub fn resolution(&self) -> f32 {
        self.resolution
    }

    /// Returns the index of the cell containing the `point`. Indices are
    /// integer vectors identifying cells, for this the coordinates are rounded
    /// to the next multiple of the resolution.
    #[inline]
    pub fn get_cell_index(&self, point: &Vector3<f32>) -> Array3i {
        let index = point / self.resolution;
        Array3i::new(
            round_to_int(index.x),
            round_to_int(index.y),
            round_to_int(index.z),
        )
    }

    /// Returns one of the octants, (0, 0, 0), (1, 0, 0), ..., (1, 1, 1).
    #[inline]
    pub fn get_octant(i: i32) -> Array3i {
        debug_assert!(i >= 0);
        debug_assert!(i < 8);
        Array3i::new(
            i32::from(i & 1 != 0),
            i32::from(i & 2 != 0),
            i32::from(i & 4 != 0),
        )
    }

    /// Returns the center of the cell at `index`.
    #[inline]
    pub fn get_center_of_cell(&self, index: &Array3i) -> Vector3<f32> {
        index.cast::<f32>() * self.resolution
    }

    /// C++ `begin()`. RUST PORT NOTE: `end()` is replaced by the
    /// `std::iter::Iterator` protocol (`for (index, value) in &grid`).
    pub fn begin(&self) -> HybridGridIterator<'_, T> {
        self.grid.iter()
    }
}

impl<T: GridValue> Deref for HybridGridBase<T> {
    type Target = GridBase<T>;
    fn deref(&self) -> &GridBase<T> {
        &self.grid
    }
}

impl<T: GridValue> DerefMut for HybridGridBase<T> {
    fn deref_mut(&mut self) -> &mut GridBase<T> {
        &mut self.grid
    }
}

impl<'a, T: GridValue> IntoIterator for &'a HybridGridBase<T> {
    type Item = (Array3i, &'a T);
    type IntoIter = HybridGridIterator<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.grid.iter()
    }
}

// ---------------------------------------------------------------------------
// HybridGrid
// ---------------------------------------------------------------------------

/// A grid containing probability values stored using 15 bits, and an update
/// marker per voxel.
/// Points are expected to be close to the origin. Points far from the origin
/// require the grid to grow dynamically. For centimeter resolution, points
/// can only be tens of meters from the origin.
/// The hard limit of cell indexes is +/- 8192 around the origin.
#[derive(Clone)]
pub struct HybridGrid {
    base: HybridGridBase<u16>,
    // Markers at changed cells.
    // RUST PORT NOTE: C++ stores raw `uint16*` into the leaf arrays. Without
    // `unsafe` we store the (signed, grow-invariant) cell indices instead and
    // re-resolve them in `finish_update()`.
    update_indices: Vec<Array3i>,
}

impl HybridGrid {
    /// Creates an empty grid with voxel edge length `resolution`.
    pub fn new(resolution: f32) -> Self {
        Self {
            base: HybridGridBase::new(resolution),
            update_indices: Vec::new(),
        }
    }

    /// C++ `HybridGrid(const proto::HybridGrid&)`.
    pub fn from_proto(proto: &proto::HybridGrid) -> Self {
        let mut grid = Self::new(proto.resolution);
        assert_eq!(
            proto.values.len(),
            proto.x_indices.len(),
            "Check failed: proto.values_size() == proto.x_indices_size()"
        );
        assert_eq!(
            proto.values.len(),
            proto.y_indices.len(),
            "Check failed: proto.values_size() == proto.y_indices_size()"
        );
        assert_eq!(
            proto.values.len(),
            proto.z_indices.len(),
            "Check failed: proto.values_size() == proto.z_indices_size()"
        );
        for i in 0..proto.values.len() {
            // set_probability does some error checking for us.
            grid.set_probability(
                &Array3i::new(proto.x_indices[i], proto.y_indices[i], proto.z_indices[i]),
                // C++ implicit int32 -> uint16 conversion (truncating).
                value_to_probability(proto.values[i] as u16),
            );
        }
        grid
    }

    /// Sets the probability of the cell at `index` to the given
    /// `probability`.
    #[inline]
    pub fn set_probability(&mut self, index: &Array3i, probability: f32) {
        *self.base.mutable_value(index) = probability_to_value(probability);
    }

    /// Finishes the update sequence.
    pub fn finish_update(&mut self) {
        while let Some(index) = self.update_indices.pop() {
            let cell = self.base.mutable_value(&index);
            debug_assert!(*cell >= UPDATE_MARKER);
            *cell -= UPDATE_MARKER;
        }
    }

    /// Applies the `odds` specified when calling
    /// `compute_lookup_table_to_apply_odds()` to the probability of the cell
    /// at `index` if the cell has not already been updated. Multiple updates
    /// of the same cell will be ignored until `finish_update()` is called.
    /// Returns true if the cell was updated.
    ///
    /// If this is the first call to `apply_odds()` for the specified cell, its
    /// value will be set to probability corresponding to `odds`.
    #[inline]
    pub fn apply_lookup_table(&mut self, index: &Array3i, table: &[u16]) -> bool {
        debug_assert_eq!(table.len(), UPDATE_MARKER as usize);
        let cell = self.base.mutable_value(index);
        if *cell >= UPDATE_MARKER {
            return false;
        }
        *cell = table[*cell as usize];
        debug_assert!(*cell >= UPDATE_MARKER);
        self.update_indices.push(*index);
        true
    }

    /// Returns the probability of the cell with `index`.
    #[inline]
    pub fn get_probability(&self, index: &Array3i) -> f32 {
        value_to_probability(self.base.value(index))
    }

    /// Returns true if the probability at the specified `index` is known.
    #[inline]
    pub fn is_known(&self, index: &Array3i) -> bool {
        self.base.value(index) != 0
    }

    /// Serializes the grid; cells are emitted in iteration order.
    pub fn to_proto(&self) -> proto::HybridGrid {
        assert!(
            self.update_indices.is_empty(),
            "Serializing a grid during an update is not supported. Finish the update first."
        );
        let mut result = proto::HybridGrid {
            resolution: self.resolution(),
            ..Default::default()
        };
        for (index, &value) in self {
            result.x_indices.push(index.x);
            result.y_indices.push(index.y);
            result.z_indices.push(index.z);
            result.values.push(i32::from(value));
        }
        result
    }
}

impl Deref for HybridGrid {
    type Target = HybridGridBase<u16>;
    fn deref(&self) -> &HybridGridBase<u16> {
        &self.base
    }
}

impl DerefMut for HybridGrid {
    fn deref_mut(&mut self) -> &mut HybridGridBase<u16> {
        &mut self.base
    }
}

impl<'a> IntoIterator for &'a HybridGrid {
    type Item = (Array3i, &'a u16);
    type IntoIter = HybridGridIterator<'a, u16>;
    fn into_iter(self) -> Self::IntoIter {
        self.base.iter()
    }
}

// ---------------------------------------------------------------------------
// IntensityHybridGrid
// ---------------------------------------------------------------------------

/// Accumulated intensity of a voxel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AverageIntensityData {
    /// Sum of all intensities added.
    pub sum: f32,
    /// Number of intensities added.
    pub count: i32,
}

// RUST PORT NOTE: C++ `AverageIntensityData` has no `operator==`, so its
// iterator cannot be instantiated there; here a cell is "default" if it equals
// `{0, 0}`.
impl GridValue for AverageIntensityData {
    #[inline]
    fn is_default_value(&self) -> bool {
        *self == Self::default()
    }
}

/// A grid of averaged intensities.
#[derive(Clone)]
pub struct IntensityHybridGrid {
    base: HybridGridBase<AverageIntensityData>,
}

impl IntensityHybridGrid {
    /// Creates an empty grid with voxel edge length `resolution`.
    pub fn new(resolution: f32) -> Self {
        Self {
            base: HybridGridBase::new(resolution),
        }
    }

    /// Adds `intensity` to the average of the cell at `index`.
    #[inline]
    pub fn add_intensity(&mut self, index: &Array3i, intensity: f32) {
        let cell = self.base.mutable_value(index);
        cell.count += 1;
        cell.sum += intensity;
    }

    /// Returns the average intensity of the cell at `index` (0 if unknown).
    #[inline]
    pub fn get_intensity(&self, index: &Array3i) -> f32 {
        match self.base.get(index) {
            Some(cell) if cell.count != 0 => cell.sum / cell.count as f32,
            _ => 0.0,
        }
    }
}

impl Deref for IntensityHybridGrid {
    type Target = HybridGridBase<AverageIntensityData>;
    fn deref(&self) -> &HybridGridBase<AverageIntensityData> {
        &self.base
    }
}

impl DerefMut for IntensityHybridGrid {
    fn deref_mut(&mut self) -> &mut HybridGridBase<AverageIntensityData> {
        &mut self.base
    }
}

impl<'a> IntoIterator for &'a IntensityHybridGrid {
    type Item = (Array3i, &'a AverageIntensityData);
    type IntoIter = HybridGridIterator<'a, AverageIntensityData>;
    fn into_iter(self) -> Self::IntoIter {
        self.base.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::probability_values_shim::{
        MAX_PROBABILITY, MIN_PROBABILITY, compute_lookup_table_to_apply_odds, odds,
    };
    use super::*;
    use std::collections::BTreeMap;

    fn a(x: i32, y: i32, z: i32) -> Array3i {
        Array3i::new(x, y, z)
    }

    fn v(x: f32, y: f32, z: f32) -> Vector3<f32> {
        Vector3::new(x, y, z)
    }

    fn assert_near(actual: f32, expected: f32, eps: f32) {
        assert!(
            (actual - expected).abs() <= eps,
            "{actual} vs {expected} (eps {eps})"
        );
    }

    mod hybrid_grid_test {
        use super::*;

        #[test]
        fn apply_odds() {
            let mut hybrid_grid = HybridGrid::new(1.0);

            assert!(!hybrid_grid.is_known(&a(0, 0, 0)));
            assert!(!hybrid_grid.is_known(&a(0, 1, 0)));
            assert!(!hybrid_grid.is_known(&a(1, 0, 0)));
            assert!(!hybrid_grid.is_known(&a(1, 1, 0)));
            assert!(!hybrid_grid.is_known(&a(0, 0, 1)));
            assert!(!hybrid_grid.is_known(&a(0, 1, 1)));
            assert!(!hybrid_grid.is_known(&a(1, 0, 1)));
            assert!(!hybrid_grid.is_known(&a(1, 1, 1)));

            hybrid_grid.set_probability(&a(1, 0, 1), 0.5);

            hybrid_grid
                .apply_lookup_table(&a(1, 0, 1), &compute_lookup_table_to_apply_odds(odds(0.9)));
            hybrid_grid.finish_update();
            assert!(hybrid_grid.get_probability(&a(1, 0, 1)) > 0.5);

            hybrid_grid.set_probability(&a(0, 1, 0), 0.5);

            hybrid_grid
                .apply_lookup_table(&a(0, 1, 0), &compute_lookup_table_to_apply_odds(odds(0.1)));
            hybrid_grid.finish_update();
            assert!(hybrid_grid.get_probability(&a(0, 1, 0)) < 0.5);

            // Tests adding odds to an unknown cell.
            hybrid_grid
                .apply_lookup_table(&a(1, 1, 1), &compute_lookup_table_to_apply_odds(odds(0.42)));
            assert_near(hybrid_grid.get_probability(&a(1, 1, 1)), 0.42, 1e-4);

            // Tests that further updates are ignored if finish_update() isn't
            // called.
            hybrid_grid
                .apply_lookup_table(&a(1, 1, 1), &compute_lookup_table_to_apply_odds(odds(0.9)));
            assert_near(hybrid_grid.get_probability(&a(1, 1, 1)), 0.42, 1e-4);
            hybrid_grid.finish_update();
            hybrid_grid
                .apply_lookup_table(&a(1, 1, 1), &compute_lookup_table_to_apply_odds(odds(0.9)));
            assert!(hybrid_grid.get_probability(&a(1, 1, 1)) > 0.42);
        }

        #[test]
        fn get_probability() {
            let mut hybrid_grid = HybridGrid::new(1.0);

            let index = hybrid_grid.get_cell_index(&v(0.0, 1.0, 1.0));
            hybrid_grid.set_probability(&index, MAX_PROBABILITY);
            assert_near(
                hybrid_grid.get_probability(&hybrid_grid.get_cell_index(&v(0.0, 1.0, 1.0))),
                MAX_PROBABILITY,
                1e-6,
            );
            for index in [
                hybrid_grid.get_cell_index(&v(0.0, 2.0, 1.0)),
                hybrid_grid.get_cell_index(&v(1.0, 1.0, 1.0)),
                hybrid_grid.get_cell_index(&v(1.0, 2.0, 1.0)),
            ] {
                assert!(!hybrid_grid.is_known(&index));
            }
        }

        #[test]
        fn get_intensity() {
            let mut hybrid_grid = IntensityHybridGrid::new(1.0);
            let cell_index = hybrid_grid.get_cell_index(&v(0.0, 1.0, 1.0));
            let intensity = 58.0f32;

            assert_near(hybrid_grid.get_intensity(&cell_index), 0.0, 1e-9);
            hybrid_grid.add_intensity(&cell_index, intensity);
            assert_near(hybrid_grid.get_intensity(&cell_index), intensity, 1e-9);
            for index in [
                hybrid_grid.get_cell_index(&v(0.0, 2.0, 1.0)),
                hybrid_grid.get_cell_index(&v(1.0, 1.0, 1.0)),
                hybrid_grid.get_cell_index(&v(1.0, 2.0, 1.0)),
            ] {
                assert_near(hybrid_grid.get_intensity(&index), 0.0, 1e-9);
            }
        }

        #[test]
        fn get_cell_index() {
            let hybrid_grid = HybridGrid::new(2.0);

            assert_eq!(hybrid_grid.get_cell_index(&v(0.0, 0.0, 0.0)), a(0, 0, 0));
            assert_eq!(hybrid_grid.get_cell_index(&v(0.0, 26.0, 10.0)), a(0, 13, 5));
            assert_eq!(hybrid_grid.get_cell_index(&v(14.0, 0.0, 10.0)), a(7, 0, 5));
            assert_eq!(hybrid_grid.get_cell_index(&v(14.0, 26.0, 0.0)), a(7, 13, 0));

            // Check around the origin.
            assert_eq!(hybrid_grid.get_cell_index(&v(8.5, 11.5, 0.5)), a(4, 6, 0));
            assert_eq!(hybrid_grid.get_cell_index(&v(7.5, 12.5, 1.5)), a(4, 6, 1));
            assert_eq!(hybrid_grid.get_cell_index(&v(6.5, 14.5, 2.5)), a(3, 7, 1));
            assert_eq!(hybrid_grid.get_cell_index(&v(5.5, 13.5, 3.5)), a(3, 7, 2));
        }

        #[test]
        fn get_center_of_cell() {
            let hybrid_grid = HybridGrid::new(2.0);

            let index = a(3, 2, 1);
            let center = hybrid_grid.get_center_of_cell(&index);
            assert_near(6.0, center.x, 1e-6);
            assert_near(4.0, center.y, 1e-6);
            assert_near(2.0, center.z, 1e-6);
            assert_eq!(hybrid_grid.get_cell_index(&center), index);
        }
    }

    mod random_hybrid_grid_test {
        use super::*;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        type ValueMap = BTreeMap<(i32, i32, i32), f32>;

        /// Fixture of the C++ `RandomHybridGridTest`.
        struct RandomHybridGridTest {
            hybrid_grid: HybridGrid,
            values: ValueMap,
        }

        impl RandomHybridGridTest {
            fn new() -> Self {
                let mut hybrid_grid = HybridGrid::new(2.0);
                let mut values = ValueMap::new();
                let mut rng = StdRng::seed_from_u64(1285120005);
                for _ in 0..10000 {
                    let x = rng.random_range(-3000..=2999);
                    let y = rng.random_range(-3000..=2999);
                    let z = rng.random_range(-3000..=2999);
                    let value = rng.random_range(MIN_PROBABILITY..MAX_PROBABILITY);
                    // C++ `std::map::emplace` keeps the first value.
                    values.entry((x, y, z)).or_insert(value);
                }
                for (&(x, y, z), &probability) in &values {
                    hybrid_grid.set_probability(&a(x, y, z), probability);
                }
                Self {
                    hybrid_grid,
                    values,
                }
            }
        }

        #[test]
        fn test_iteration() {
            let RandomHybridGridTest {
                hybrid_grid,
                mut values,
            } = RandomHybridGridTest::new();
            let mut it = HybridGridIterator::new(&hybrid_grid);
            while !it.done() {
                let cell_index = it.get_cell_index();
                let iterator_probability = value_to_probability(*it.get_value());
                assert_eq!(
                    iterator_probability,
                    hybrid_grid.get_probability(&cell_index)
                );
                let key = (cell_index[0], cell_index[1], cell_index[2]);
                let expected = values
                    .remove(&key)
                    .unwrap_or_else(|| panic!("unexpected cell {cell_index:?}"));
                assert_near(expected, iterator_probability, 1e-4);
                it.advance();
            }

            // Test that range based loop is equivalent to using the iterator.
            let mut it = HybridGridIterator::new(&hybrid_grid);
            for (index, value) in &hybrid_grid {
                assert!(!it.done());
                assert_eq!(index, it.get_cell_index());
                assert_eq!(*value, *it.get_value());
                it.advance();
            }
            assert!(it.done());

            // Now 'values' must not contain values.
            assert!(values.is_empty(), "cells not visited: {values:?}");
        }

        #[test]
        fn to_proto() {
            let RandomHybridGridTest { hybrid_grid, .. } = RandomHybridGridTest::new();
            let proto = hybrid_grid.to_proto();
            assert_eq!(hybrid_grid.resolution(), proto.resolution);
            assert_eq!(proto.x_indices.len(), proto.y_indices.len());
            assert_eq!(proto.x_indices.len(), proto.z_indices.len());
            assert_eq!(proto.x_indices.len(), proto.values.len());

            let mut proto_map = ValueMap::new();
            for i in 0..proto.x_indices.len() {
                proto_map.insert(
                    (proto.x_indices[i], proto.y_indices[i], proto.z_indices[i]),
                    proto.values[i] as f32,
                );
            }

            // Get hybrid_grid into the same format.
            let mut hybrid_grid_map = ValueMap::new();
            for (index, &value) in &hybrid_grid {
                hybrid_grid_map.insert((index.x, index.y, index.z), f32::from(value));
            }

            assert_eq!(proto_map, hybrid_grid_map);
        }

        #[test]
        fn from_proto() {
            let RandomHybridGridTest { hybrid_grid, .. } = RandomHybridGridTest::new();
            let constructed_grid = HybridGrid::from_proto(&hybrid_grid.to_proto());

            let member_map: BTreeMap<(i32, i32, i32), u16> = hybrid_grid
                .iter()
                .map(|(i, &v)| ((i.x, i.y, i.z), v))
                .collect();
            let constructed_map: BTreeMap<(i32, i32, i32), u16> = constructed_grid
                .iter()
                .map(|(i, &v)| ((i.x, i.y, i.z), v))
                .collect();

            assert_eq!(member_map, constructed_map);
        }
    }

    // Additional tests (not in the C++ test file).
    mod extra {
        use super::*;

        #[test]
        fn flat_index_round_trip() {
            for bits in 1..=4 {
                for i in 0..(1 << (3 * bits)) {
                    assert_eq!(to_flat_index(&to_3d_index(i, bits), bits), i);
                }
            }
            assert_eq!(to_flat_index(&a(1, 2, 3), 3), (3 * 8 + 2) * 8 + 1);
        }

        #[test]
        fn get_octant() {
            let expected = [
                a(0, 0, 0),
                a(1, 0, 0),
                a(0, 1, 0),
                a(1, 1, 0),
                a(0, 0, 1),
                a(1, 0, 1),
                a(0, 1, 1),
                a(1, 1, 1),
            ];
            for (i, e) in expected.iter().enumerate() {
                assert_eq!(HybridGridBase::<u16>::get_octant(i as i32), *e);
            }
        }

        #[test]
        fn get_cell_index_rounds_half_away_from_zero() {
            let grid = HybridGrid::new(1.0);
            assert_eq!(grid.get_cell_index(&v(-0.5, 0.5, -1.5)), a(-1, 1, -2));
            assert_eq!(grid.get_cell_index(&v(-0.49, 0.49, 2.5)), a(0, 0, 3));
        }

        #[test]
        fn grows_and_keeps_values() {
            let mut grid = HybridGrid::new(1.0);
            assert_eq!(grid.grid_size(), 128);
            grid.set_probability(&a(-64, 63, 0), 0.7);
            assert_eq!(grid.grid_size(), 128);
            grid.set_probability(&a(64, 0, 0), 0.3);
            assert_eq!(grid.grid_size(), 256);
            grid.set_probability(&a(-8192, 8191, 0), 0.6);
            assert_eq!(grid.grid_size(), 16384);
            assert_near(grid.get_probability(&a(-64, 63, 0)), 0.7, 1e-4);
            assert_near(grid.get_probability(&a(64, 0, 0)), 0.3, 1e-4);
            assert_near(grid.get_probability(&a(-8192, 8191, 0)), 0.6, 1e-4);
            // Reads outside never grow and return the default.
            assert!(!grid.is_known(&a(8192, 0, 0)));
            assert!(!grid.is_known(&a(i32::MAX, i32::MIN, 0)));
            assert_eq!(grid.grid_size(), 16384);
        }

        #[test]
        #[should_panic(expected = "new_bits <= 8")]
        fn growing_beyond_limit_panics() {
            let mut grid = HybridGrid::new(1.0);
            grid.set_probability(&a(8192, 0, 0), 0.5);
        }

        #[test]
        fn iteration_order_is_hierarchical() {
            // Within one root meta cell and nested meta cell: z-major order.
            // Across nested meta cells (8 voxels apart): nested order first.
            let mut grid = HybridGrid::new(1.0);
            // Root bits = 1, offset 64. Shifted = index + 64.
            let cells = [
                a(-64, -64, -64), // root (0,0,0), nested (0,0,0), flat 0
                a(-63, -64, -64), // root (0,0,0), nested (0,0,0), flat 1
                a(-64, -63, -64), // flat (0,1,0)
                a(-56, -64, -64), // nested (1,0,0)
                a(-64, -64, -56), // nested (0,0,1)
                a(-64, -64, -63), // flat (0,0,1) in nested (0,0,0)
                a(0, -64, -64),   // root (1,0,0)
                a(-64, 0, -64),   // root (0,1,0)
            ];
            for c in &cells {
                grid.set_probability(c, 0.5);
            }
            let order: Vec<Array3i> = grid.iter().map(|(i, _)| i).collect();
            assert_eq!(
                order,
                vec![
                    a(-64, -64, -64),
                    a(-63, -64, -64),
                    a(-64, -63, -64),
                    a(-64, -64, -63),
                    a(-56, -64, -64),
                    a(-64, -64, -56),
                    a(0, -64, -64),
                    a(-64, 0, -64),
                ]
            );
        }

        #[test]
        fn finish_update_survives_grow() {
            let mut grid = HybridGrid::new(1.0);
            let table = compute_lookup_table_to_apply_odds(odds(0.7));
            assert!(grid.apply_lookup_table(&a(1, 2, 3), &table));
            // Forces a grow while an update is pending.
            assert!(grid.apply_lookup_table(&a(1000, -1000, 5), &table));
            assert!(!grid.apply_lookup_table(&a(1, 2, 3), &table));
            grid.finish_update();
            for (_, &value) in &grid {
                assert!(value < UPDATE_MARKER);
            }
            assert_near(grid.get_probability(&a(1, 2, 3)), 0.7, 1e-4);
            assert_near(grid.get_probability(&a(1000, -1000, 5)), 0.7, 1e-4);
            // Proto is allowed again.
            assert_eq!(grid.to_proto().values.len(), 2);
        }

        #[test]
        #[should_panic(expected = "Finish the update first")]
        fn to_proto_during_update_panics() {
            let mut grid = HybridGrid::new(1.0);
            let table = compute_lookup_table_to_apply_odds(odds(0.7));
            grid.apply_lookup_table(&a(0, 0, 0), &table);
            let _ = grid.to_proto();
        }

        #[test]
        fn vec_values_and_flat_nested_grids() {
            let mut grid = HybridGridBase::<Vec<i32>>::new(0.5);
            grid.mutable_value(&a(-3, 4, 5)).push(7);
            grid.mutable_value(&a(2, 2, 2)); // Allocated but empty: skipped.
            assert_eq!(grid.get(&a(-3, 4, 5)), Some(&vec![7]));
            assert!(grid.value(&a(2, 2, 2)).is_empty());
            let cells: Vec<_> = grid.iter().collect();
            assert_eq!(cells, vec![(a(-3, 4, 5), &vec![7])]);

            let mut flat = FlatGrid::<u8, 2>::new();
            *flat.mutable_value(&a(3, 0, 1)) = 4;
            *flat.mutable_value(&a(1, 1, 0)) = 2;
            assert_eq!(flat.value(&a(3, 0, 1)), 4);
            let cells: Vec<_> = flat.iter().map(|(i, &v)| (i, v)).collect();
            assert_eq!(cells, vec![(a(1, 1, 0), 2), (a(3, 0, 1), 4)]);

            let mut nested = NestedGrid::<FlatGrid<u8, 1>, 1>::new();
            assert_eq!(NestedGrid::<FlatGrid<u8, 1>, 1>::grid_size(), 4);
            *nested.mutable_value(&a(3, 0, 0)) = 1;
            *nested.mutable_value(&a(0, 1, 0)) = 2;
            assert_eq!(nested.value(&a(3, 3, 3)), 0);
            let cells: Vec<_> = nested.iter().map(|(i, &v)| (i, v)).collect();
            // Nested meta cell (0,0,0) first, then (1,0,0).
            assert_eq!(cells, vec![(a(0, 1, 0), 2), (a(3, 0, 0), 1)]);
        }

        #[test]
        fn intensity_average() {
            let mut grid = IntensityHybridGrid::new(0.1);
            grid.add_intensity(&a(-1, -1, -1), 10.0);
            grid.add_intensity(&a(-1, -1, -1), 20.0);
            assert_near(grid.get_intensity(&a(-1, -1, -1)), 15.0, 1e-6);
            let cells: Vec<_> = grid.iter().collect();
            assert_eq!(cells.len(), 1);
            assert_eq!(cells[0].1.count, 2);
        }
    }

    mod proptests {
        use super::*;
        use proptest::collection::btree_map;
        use proptest::prelude::*;

        fn index_strategy() -> impl Strategy<Value = (i32, i32, i32)> {
            (-8192i32..8192, -8192i32..8192, -8192i32..8192)
        }

        /// Reference iteration key per C++ order: (root flat index, nested
        /// flat index, flat index), given the final root bits.
        fn iteration_key(index: &Array3i, grid_size: i32) -> (i32, i32, i32) {
            let root_bits = (grid_size / 64).trailing_zeros() as i32;
            let shifted = index.add_scalar(grid_size >> 1);
            let root = shifted.map(|c| c / 64);
            let nested = shifted.map(|c| (c % 64) / 8);
            let flat = shifted.map(|c| c % 8);
            (
                to_flat_index(&root, root_bits),
                to_flat_index(&nested, 3),
                to_flat_index(&flat, 3),
            )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn set_get_round_trip(
                cells in btree_map(index_strategy(), 1u16..u16::MAX, 0..200),
            ) {
                let mut grid = HybridGridBase::<u16>::new(0.1);
                for (&(x, y, z), &value) in &cells {
                    *grid.mutable_value(&a(x, y, z)) = value;
                }
                for (&(x, y, z), &value) in &cells {
                    prop_assert_eq!(grid.value(&a(x, y, z)), value);
                }
            }

            #[test]
            fn iteration_visits_exactly_set_cells_in_cpp_order(
                cells in btree_map(index_strategy(), 0u16..u16::MAX, 0..200),
            ) {
                let mut grid = HybridGridBase::<u16>::new(0.1);
                for (&(x, y, z), &value) in &cells {
                    *grid.mutable_value(&a(x, y, z)) = value;
                }
                let visited: Vec<(Array3i, u16)> =
                    grid.iter().map(|(i, &v)| (i, v)).collect();
                let mut expected: Vec<(Array3i, u16)> = cells
                    .iter()
                    .filter(|&(_, &v)| v != 0)
                    .map(|(&(x, y, z), &v)| (a(x, y, z), v))
                    .collect();
                let grid_size = grid.grid_size();
                expected.sort_by_key(|(i, _)| iteration_key(i, grid_size));
                prop_assert_eq!(visited, expected);
            }

            #[test]
            fn proto_round_trip(
                cells in btree_map(index_strategy(), 0.0f32..1.0, 0..200),
                resolution in 0.01f32..1.0,
            ) {
                let mut grid = HybridGrid::new(resolution);
                for (&(x, y, z), &p) in &cells {
                    grid.set_probability(&a(x, y, z), p);
                }
                let proto = grid.to_proto();
                prop_assert_eq!(proto.values.len(), cells.len());
                let constructed = HybridGrid::from_proto(&proto);
                prop_assert_eq!(constructed.resolution(), resolution);
                let original: Vec<_> = grid.iter().map(|(i, &v)| (i, v)).collect();
                let round_tripped: Vec<_> =
                    constructed.iter().map(|(i, &v)| (i, v)).collect();
                prop_assert_eq!(&original, &round_tripped);
                prop_assert_eq!(constructed.to_proto(), proto);
            }
        }
    }
}
