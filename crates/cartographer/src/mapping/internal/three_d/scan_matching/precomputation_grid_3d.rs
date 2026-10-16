//! Port of `cartographer/mapping/internal/3d/scan_matching/precomputation_grid_3d` (.h, .cc).

use nalgebra::Vector3;

use crate::common::port::round_to_int;
use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY, value_to_probability};
use crate::mapping::three_d::hybrid_grid::{Array3i, HybridGrid, HybridGridBase};

const BLOCK_BITS: i32 = 3;
const BLOCK_MASK: i32 = (1 << BLOCK_BITS) - 1;
const BLOCK_CELLS: usize = 1 << (3 * BLOCK_BITS);
const NO_BLOCK: u32 = u32::MAX;

/// A grid of `u8` values used by the 3D branch-and-bound scan matcher, where
/// 0 is unknown (probability `MIN_PROBABILITY`) and 255 is
/// `MAX_PROBABILITY`.
///
/// RUST PORT NOTE: C++ derives from `HybridGridBase<uint8>` (a three level
/// `DynamicGrid`/`NestedGrid`/`FlatGrid` tree of heap allocations). Lookups
/// into these grids are the innermost loop of `FastCorrelativeScanMatcher3D`,
/// so this port stores the data in flat arrays instead: a dense table over
/// the bounding box (in 8x8x8 blocks) maps each block to an offset into one
/// contiguous `Vec<u8>` of allocated blocks. A lookup is two array reads
/// with unsigned bounds checks. The grid does not grow: its bounds are fixed
/// at construction (both factory functions compute them up front), and
/// `mutable_value` panics outside of them. Values (and therefore all
/// matching results) are identical to the C++ grid; only the iteration order
/// of [`PrecomputationGrid3D::iter`] differs, which no caller depends on
/// (all consumers take maxima).
#[derive(Clone, Debug)]
pub struct PrecomputationGrid3D {
    resolution: f32,
    /// Block coordinates of the first block of the table.
    block_origin: Array3i,
    /// Number of blocks of the table in x, y and z.
    num_blocks: [i32; 3],
    /// Dense table over the bounding box: index into `block_indices` /
    /// `cells` (in blocks), or `NO_BLOCK`.
    block_table: Vec<u32>,
    /// Block coordinates of each allocated block.
    block_indices: Vec<Array3i>,
    /// Cell values of all allocated blocks, `BLOCK_CELLS` per block (x
    /// fastest, then y, then z).
    cells: Vec<u8>,
}

impl PrecomputationGrid3D {
    /// Creates an empty grid that can hold cells in the inclusive index range
    /// `[min_index, max_index]` (empty if any `min > max`).
    pub fn new(resolution: f32, min_index: &Array3i, max_index: &Array3i) -> Self {
        let block_origin = min_index.map(|c| c >> BLOCK_BITS);
        let block_max = max_index.map(|c| c >> BLOCK_BITS);
        let empty = (0..3).any(|i| min_index[i] > max_index[i]);
        let num_blocks = if empty {
            [0; 3]
        } else {
            [
                block_max.x - block_origin.x + 1,
                block_max.y - block_origin.y + 1,
                block_max.z - block_origin.z + 1,
            ]
        };
        let table_size = num_blocks.iter().map(|&n| n as usize).product();
        Self {
            resolution,
            block_origin,
            num_blocks,
            block_table: vec![NO_BLOCK; table_size],
            block_indices: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Edge length of each voxel.
    #[inline]
    pub fn resolution(&self) -> f32 {
        self.resolution
    }

    /// Index of the voxel containing `point` (C++
    /// `HybridGridBase::GetCellIndex`).
    #[inline]
    pub fn get_cell_index(&self, point: &Vector3<f32>) -> Array3i {
        let index = point / self.resolution;
        Array3i::new(
            round_to_int(index.x),
            round_to_int(index.y),
            round_to_int(index.z),
        )
    }

    /// C++ `HybridGridBase::GetOctant`.
    #[inline]
    pub fn get_octant(i: i32) -> Array3i {
        HybridGridBase::<u8>::get_octant(i)
    }

    /// Maps values from [0, 255] to [`MIN_PROBABILITY`, `MAX_PROBABILITY`].
    #[inline]
    pub fn to_probability(value: f32) -> f32 {
        MIN_PROBABILITY + value * ((MAX_PROBABILITY - MIN_PROBABILITY) / 255.0)
    }

    #[inline]
    fn table_index(&self, index: &Array3i) -> Option<usize> {
        let bx = (index.x >> BLOCK_BITS).wrapping_sub(self.block_origin.x);
        let by = (index.y >> BLOCK_BITS).wrapping_sub(self.block_origin.y);
        let bz = (index.z >> BLOCK_BITS).wrapping_sub(self.block_origin.z);
        let [nx, ny, nz] = self.num_blocks;
        // Unsigned comparison also rejects negative values.
        if (bx as u32) >= (nx as u32) || (by as u32) >= (ny as u32) || (bz as u32) >= (nz as u32) {
            return None;
        }
        Some(((bz * ny + by) * nx + bx) as usize)
    }

    #[inline]
    fn offset_in_block(index: &Array3i) -> usize {
        (((index.z & BLOCK_MASK) << (2 * BLOCK_BITS))
            | ((index.y & BLOCK_MASK) << BLOCK_BITS)
            | (index.x & BLOCK_MASK)) as usize
    }

    /// Returns the value at `index` (0 if unknown or outside the grid).
    #[inline]
    pub fn value(&self, index: &Array3i) -> u8 {
        match self.table_index(index) {
            Some(table_index) => {
                let block = self.block_table[table_index];
                if block == NO_BLOCK {
                    0
                } else {
                    self.cells[block as usize * BLOCK_CELLS + Self::offset_in_block(index)]
                }
            }
            None => 0,
        }
    }

    /// Returns a mutable reference to the value at `index`, allocating its
    /// block if needed. Panics if `index` is outside the bounds given at
    /// construction.
    #[inline]
    pub fn mutable_value(&mut self, index: &Array3i) -> &mut u8 {
        let table_index = self
            .table_index(index)
            .unwrap_or_else(|| panic!("Cell index {index:?} outside of the PrecomputationGrid3D"));
        let mut block = self.block_table[table_index];
        if block == NO_BLOCK {
            block = self.block_indices.len() as u32;
            self.block_table[table_index] = block;
            self.block_indices.push(index.map(|c| c >> BLOCK_BITS));
            self.cells.resize(self.cells.len() + BLOCK_CELLS, 0);
        }
        &mut self.cells[block as usize * BLOCK_CELLS + Self::offset_in_block(index)]
    }

    /// Iterates over all non-zero cells as `(cell_index, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (Array3i, u8)> + '_ {
        self.block_indices
            .iter()
            .enumerate()
            .flat_map(move |(block, block_index)| {
                let base = block_index.map(|c| c << BLOCK_BITS);
                let cells = &self.cells[block * BLOCK_CELLS..(block + 1) * BLOCK_CELLS];
                cells
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0)
                    .map(move |(i, &v)| {
                        let i = i as i32;
                        (
                            base + Array3i::new(
                                i & BLOCK_MASK,
                                (i >> BLOCK_BITS) & BLOCK_MASK,
                                i >> (2 * BLOCK_BITS),
                            ),
                            v,
                        )
                    })
            })
    }

    /// Bounding box `(min, max)` of the non-zero cells, or `None` if all are
    /// zero.
    fn bounds(&self) -> Option<(Array3i, Array3i)> {
        bounds_of(self.iter().map(|(index, _)| index))
    }
}

fn bounds_of(indices: impl Iterator<Item = Array3i>) -> Option<(Array3i, Array3i)> {
    let mut result: Option<(Array3i, Array3i)> = None;
    for index in indices {
        result = Some(match result {
            None => (index, index),
            Some((min, max)) => (min.inf(&index), max.sup(&index)),
        });
    }
    result
}

fn empty_bounds() -> (Array3i, Array3i) {
    (Array3i::new(0, 0, 0), Array3i::new(-1, -1, -1))
}

/// Computes the half resolution index corresponding to the full resolution
/// `cell_index`.
///
/// C++11 defines that integer division rounds towards zero. For index math,
/// we actually need it to round towards negative infinity. Luckily bit
/// shifts have that property.
#[inline]
fn cell_index_at_half_resolution(cell_index: &Array3i) -> Array3i {
    cell_index.map(|c| c >> 1)
}

/// Converts a [`HybridGrid`] to a [`PrecomputationGrid3D`] representing the
/// same data, but only using 8 bit instead of 2 x 16 bit.
pub fn convert_to_precomputation_grid(hybrid_grid: &HybridGrid) -> PrecomputationGrid3D {
    let (min, max) =
        bounds_of(hybrid_grid.iter().map(|(index, _)| index)).unwrap_or_else(empty_bounds);
    let mut result = PrecomputationGrid3D::new(hybrid_grid.resolution(), &min, &max);
    for (index, &value) in hybrid_grid {
        let cell_value = round_to_int(
            (value_to_probability(value) - MIN_PROBABILITY)
                * (255.0 / (MAX_PROBABILITY - MIN_PROBABILITY)),
        );
        assert!(
            cell_value >= 0,
            "Check failed: cell_value >= 0 ({cell_value} vs. 0)"
        );
        assert!(
            cell_value <= 255,
            "Check failed: cell_value <= 255 ({cell_value} vs. 255)"
        );
        *result.mutable_value(&index) = cell_value as u8;
    }
    result
}

/// Returns a grid of the same resolution containing the maximum value of
/// original voxels in `grid`. This maximum is over the 8 voxels that have
/// any combination of index components optionally increased by `shift`.
/// If `shift` is 2 ** (depth - 1), where depth 0 is the original grid, and
/// this is using the precomputed grid of one depth before, this results in
/// precomputation grids analogous to the 2D case.
pub fn precompute_grid(
    grid: &PrecomputationGrid3D,
    half_resolution: bool,
    shift: &Array3i,
) -> PrecomputationGrid3D {
    let (min, max) = match grid.bounds() {
        Some((min, max)) => {
            let target_min = min - shift.map(|c| c.max(0));
            let target_max = max - shift.map(|c| c.min(0));
            if half_resolution {
                (
                    cell_index_at_half_resolution(&target_min),
                    cell_index_at_half_resolution(&target_max),
                )
            } else {
                (target_min, target_max)
            }
        }
        None => empty_bounds(),
    };
    let mut result = PrecomputationGrid3D::new(grid.resolution(), &min, &max);
    for (index, value) in grid.iter() {
        for i in 0..8 {
            // We use this value to update 8 values in the resulting grid, at
            // position (x - {0, 'shift'}, y - {0, 'shift'}, z - {0, 'shift'}).
            // If 'shift' is 2 ** (depth - 1), where depth 0 is the original
            // grid, this results in precomputation grids analogous to the 2D
            // case.
            let cell_index = index - shift.component_mul(&PrecomputationGrid3D::get_octant(i));
            let target = if half_resolution {
                cell_index_at_half_resolution(&cell_index)
            } else {
                cell_index
            };
            let cell_value = result.mutable_value(&target);
            *cell_value = value.max(*cell_value);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    mod precomputed_grid_generator_3d_test {
        use super::*;

        #[test]
        fn test_against_naive_algorithm() {
            let mut hybrid_grid = HybridGrid::new(2.);

            // C++ uses std::mt19937(23847); exact samples are not required.
            let mut rng = StdRng::seed_from_u64(23847);
            for _ in 0..1000 {
                let x = rng.random_range(-50..=49);
                let y = rng.random_range(-50..=49);
                let z = rng.random_range(-50..=49);
                let cell_index = Array3i::new(x, y, z);
                hybrid_grid.set_probability(
                    &cell_index,
                    rng.random_range(MIN_PROBABILITY..MAX_PROBABILITY),
                );
            }

            let mut precomputed_grids: Vec<PrecomputationGrid3D> = Vec::new();
            for depth in 0..=3 {
                if depth == 0 {
                    precomputed_grids.push(convert_to_precomputation_grid(&hybrid_grid));
                } else {
                    let next = precompute_grid(
                        precomputed_grids.last().unwrap(),
                        false,
                        &Array3i::repeat(1 << (depth - 1)),
                    );
                    precomputed_grids.push(next);
                }
                let width = 1 << depth;
                for _ in 0..100 {
                    let x = rng.random_range(-50..=49);
                    let y = rng.random_range(-50..=49);
                    let z = rng.random_range(-50..=49);
                    let mut max_probability = 0.0f32;
                    for dx in 0..width {
                        for dy in 0..width {
                            for dz in 0..width {
                                max_probability = max_probability.max(
                                    hybrid_grid.get_probability(&Array3i::new(
                                        x + dx,
                                        y + dy,
                                        z + dz,
                                    )),
                                );
                            }
                        }
                    }

                    let actual = PrecomputationGrid3D::to_probability(f32::from(
                        precomputed_grids
                            .last()
                            .unwrap()
                            .value(&Array3i::new(x, y, z)),
                    ));
                    assert!(
                        (max_probability - actual).abs() <= 1e-2,
                        "depth {depth} ({x}, {y}, {z}): {max_probability} vs {actual}"
                    );
                }
            }
        }
    }

    mod additional {
        use super::*;
        use std::collections::HashMap;

        /// The C++ grid semantics on a `HashMap` (default 0).
        fn naive_precompute(
            grid: &HashMap<Array3i, u8>,
            half_resolution: bool,
            shift: &Array3i,
        ) -> HashMap<Array3i, u8> {
            let mut result = HashMap::new();
            for (index, &value) in grid {
                if value == 0 {
                    continue;
                }
                for i in 0..8 {
                    let cell_index =
                        index - shift.component_mul(&PrecomputationGrid3D::get_octant(i));
                    let target = if half_resolution {
                        cell_index.map(|c| c.div_euclid(2))
                    } else {
                        cell_index
                    };
                    let entry = result.entry(target).or_insert(0u8);
                    *entry = (*entry).max(value);
                }
            }
            result
        }

        #[test]
        fn matches_naive_hybrid_grid_semantics_including_half_resolution() {
            let mut rng = StdRng::seed_from_u64(99);
            let mut naive = HashMap::new();
            let mut grid = PrecomputationGrid3D::new(
                0.1,
                &Array3i::new(-40, -40, -40),
                &Array3i::new(40, 40, 40),
            );
            for _ in 0..800 {
                let index = Array3i::new(
                    rng.random_range(-40..=40),
                    rng.random_range(-40..=40),
                    rng.random_range(-40..=40),
                );
                let value: u8 = rng.random_range(1..=255);
                naive.insert(index, value);
                *grid.mutable_value(&index) = value;
            }
            let mut grids = vec![(grid, naive)];
            let full_resolution_depth = 2;
            let mut last_width = 1;
            for depth in 1..6 {
                let half_resolution = depth >= full_resolution_depth;
                let next_width = 1 << depth;
                let k = 1 << 0.max(depth - full_resolution_depth);
                let shift = Array3i::repeat((next_width - last_width + (k - 1)) / k);
                let (previous, previous_naive) = grids.last().unwrap();
                let next = precompute_grid(previous, half_resolution, &shift);
                let next_naive = naive_precompute(previous_naive, half_resolution, &shift);
                for z in -50..=50 {
                    for y in -50..=50 {
                        for x in -50..=50 {
                            let index = Array3i::new(x, y, z);
                            assert_eq!(
                                next.value(&index),
                                next_naive.get(&index).copied().unwrap_or(0),
                                "depth {depth} {index:?}"
                            );
                        }
                    }
                }
                grids.push((next, next_naive));
                last_width = next_width;
            }
        }

        #[test]
        fn convert_round_trips_probabilities_and_iterates_non_zero_cells() {
            let mut hybrid_grid = HybridGrid::new(0.05);
            hybrid_grid.set_probability(&Array3i::new(-3, 7, 100), 0.9);
            hybrid_grid.set_probability(&Array3i::new(5, -9, -2), 0.5);
            hybrid_grid.set_probability(&Array3i::new(0, 0, 0), 0.1);
            let grid = convert_to_precomputation_grid(&hybrid_grid);
            assert_eq!(grid.resolution(), 0.05);
            assert_eq!(grid.value(&Array3i::new(-3, 7, 100)), 255);
            assert_eq!(grid.value(&Array3i::new(0, 0, 0)), 0);
            assert!(
                (PrecomputationGrid3D::to_probability(f32::from(
                    grid.value(&Array3i::new(5, -9, -2))
                )) - 0.5)
                    .abs()
                    < 2e-3
            );
            assert_eq!(grid.value(&Array3i::new(1000, 0, 0)), 0);
            let mut cells: Vec<_> = grid.iter().collect();
            cells.sort_by_key(|(index, _)| (index.x, index.y, index.z));
            assert_eq!(cells.len(), 2);
            assert_eq!(cells[0].0, Array3i::new(-3, 7, 100));
            assert_eq!(cells[1].0, Array3i::new(5, -9, -2));

            let empty = convert_to_precomputation_grid(&HybridGrid::new(0.1));
            assert_eq!(empty.value(&Array3i::new(0, 0, 0)), 0);
            assert_eq!(
                precompute_grid(&empty, true, &Array3i::repeat(1))
                    .iter()
                    .count(),
                0
            );
        }
    }
}
