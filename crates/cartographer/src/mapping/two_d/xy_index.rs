//! Port of `cartographer/mapping/2d/xy_index` (.h).

use nalgebra::Vector2;

use crate::mapping::proto;

/// 2D integer cell index (C++ `Eigen::Array2i`): `x` is the column (fastest
/// varying in flat storage), `y` the row. Componentwise arithmetic and
/// integer division (truncating toward zero, like Eigen) are provided by
/// nalgebra.
pub type Array2i = Vector2<i32>;

/// Size of a grid in cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct CellLimits {
    /// Number of cells along x.
    pub num_x_cells: i32,
    /// Number of cells along y.
    pub num_y_cells: i32,
}

impl CellLimits {
    /// Creates cell limits.
    pub fn new(num_x_cells: i32, num_y_cells: i32) -> Self {
        Self {
            num_x_cells,
            num_y_cells,
        }
    }

    /// Creates cell limits from a proto (C++ explicit proto constructor).
    pub fn from_proto(cell_limits: &proto::CellLimits) -> Self {
        Self::new(cell_limits.num_x_cells, cell_limits.num_y_cells)
    }
}

/// Converts `cell_limits` to a proto.
pub fn to_proto(cell_limits: &CellLimits) -> proto::CellLimits {
    proto::CellLimits {
        num_x_cells: cell_limits.num_x_cells,
        num_y_cells: cell_limits.num_y_cells,
    }
}

/// Iterates in row-major order (x fastest) through an inclusive range of
/// xy-indices.
///
/// The C++ begin/end/`operator*` API is kept ([`begin`](Self::begin),
/// [`end`](Self::end), [`current`](Self::current)) and the type additionally
/// implements [`Iterator`], which yields every index from the current
/// position up to (excluding) `end()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYIndexRangeIterator {
    min_xy_index: Array2i,
    max_xy_index: Array2i,
    xy_index: Array2i,
}

impl XYIndexRangeIterator {
    /// Constructs a new iterator for the specified (inclusive) range.
    pub fn new(min_xy_index: Array2i, max_xy_index: Array2i) -> Self {
        Self {
            min_xy_index,
            max_xy_index,
            xy_index: min_xy_index,
        }
    }

    /// Constructs a new iterator for everything contained in `cell_limits`.
    pub fn from_cell_limits(cell_limits: &CellLimits) -> Self {
        Self::new(
            Array2i::zeros(),
            Array2i::new(cell_limits.num_x_cells - 1, cell_limits.num_y_cells - 1),
        )
    }

    /// Advances to the next index (C++ `operator++`).
    pub fn advance(&mut self) {
        // This is a necessary evil. Bounds checking is very expensive and
        // needs to be avoided in production.
        debug_assert!(*self != self.end());
        if self.xy_index.x < self.max_xy_index.x {
            self.xy_index.x += 1;
        } else {
            self.xy_index.x = self.min_xy_index.x;
            self.xy_index.y += 1;
        }
    }

    /// The current index (C++ `operator*`).
    pub fn current(&self) -> Array2i {
        self.xy_index
    }

    /// An iterator positioned at the first index.
    pub fn begin(&self) -> Self {
        Self::new(self.min_xy_index, self.max_xy_index)
    }

    /// An iterator positioned one past the last index, i.e. at
    /// `(min.x, max.y + 1)`.
    pub fn end(&self) -> Self {
        let mut it = self.begin();
        it.xy_index = Array2i::new(self.min_xy_index.x, self.max_xy_index.y + 1);
        it
    }
}

impl Iterator for XYIndexRangeIterator {
    type Item = Array2i;

    fn next(&mut self) -> Option<Array2i> {
        if self.xy_index == Array2i::new(self.min_xy_index.x, self.max_xy_index.y + 1) {
            return None;
        }
        let result = self.xy_index;
        self.advance();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    mod xy_index_test {
        use crate::mapping::proto;
        use crate::mapping::two_d::xy_index::*;

        #[test]
        fn cell_limits_to_proto() {
            let limits = CellLimits::new(1, 2);
            let proto = to_proto(&limits);
            assert_eq!(limits.num_x_cells, proto.num_x_cells);
            assert_eq!(limits.num_y_cells, proto.num_y_cells);
        }

        #[test]
        fn cell_limits_proto_constructor() {
            let limits = proto::CellLimits {
                num_x_cells: 1,
                num_y_cells: 2,
            };
            let native = CellLimits::from_proto(&limits);
            assert_eq!(limits.num_x_cells, native.num_x_cells);
            assert_eq!(limits.num_y_cells, native.num_y_cells);
        }

        #[test]
        fn xy_index_range_iterator() {
            let min = Array2i::new(1, 2);
            let max = Array2i::new(3, 4);
            let it = XYIndexRangeIterator::new(min, max);
            assert_eq!(min, it.begin().current());
            assert_eq!(Array2i::new(1, 5), it.end().current());
            assert_eq!(min, it.current());
            let mut num_indices = 0;
            for xy_index in XYIndexRangeIterator::new(min, max) {
                assert!(xy_index.x >= min.x && xy_index.y >= min.y);
                assert!(xy_index.x <= max.x && xy_index.y <= max.y);
                num_indices += 1;
            }
            assert_eq!(9, num_indices);
        }
    }

    mod proptests {
        use proptest::prelude::*;

        use crate::mapping::two_d::xy_index::*;

        proptest! {
            #[test]
            fn iteration_is_row_major_and_complete(
                min_x in -20i32..20, min_y in -20i32..20, w in 1i32..15, h in 1i32..15) {
                let min = Array2i::new(min_x, min_y);
                let max = Array2i::new(min_x + w - 1, min_y + h - 1);
                let indices: Vec<Array2i> = XYIndexRangeIterator::new(min, max).collect();
                let mut expected = Vec::new();
                for y in min.y..=max.y {
                    for x in min.x..=max.x {
                        expected.push(Array2i::new(x, y));
                    }
                }
                prop_assert_eq!(indices, expected);
            }

            #[test]
            fn cell_limits_iteration_count(nx in 1i32..30, ny in 1i32..30) {
                let limits = CellLimits::new(nx, ny);
                prop_assert_eq!(
                    XYIndexRangeIterator::from_cell_limits(&limits).count() as i32, nx * ny);
                prop_assert_eq!(CellLimits::from_proto(&to_proto(&limits)), limits);
            }
        }
    }
}
