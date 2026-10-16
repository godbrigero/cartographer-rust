//! Regular 2D grids: layout helper, ground-truth occupancy raster and a plain
//! probability grid for estimated maps.
//!
//! Layout convention (used by every grid in this crate): cell `(ix, iy)`
//! covers `[origin.x + ix * res, origin.x + (ix + 1) * res) x
//! [origin.y + iy * res, origin.y + (iy + 1) * res)`; `origin` is the **min
//! corner** of cell `(0, 0)`; storage is row-major with index
//! `iy * width + ix`. (This differs from Cartographer's `MapLimits`, which
//! stores the max corner and indexes x/y reversed; convert by sampling
//! through [`ProbabilityGrid::from_fn`] or a lookup closure.)

use nalgebra::Point2;

use crate::world2d::World2d;

/// Geometry of a regular 2D grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLayout {
    /// Min corner of cell `(0, 0)` in world coordinates.
    pub origin: Point2<f64>,
    /// Cell edge length in metres.
    pub resolution: f64,
    /// Number of cells along x.
    pub width: usize,
    /// Number of cells along y.
    pub height: usize,
}

impl GridLayout {
    /// Number of cells.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Whether the grid has no cells.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major linear index of `(ix, iy)`.
    pub fn index(&self, ix: usize, iy: usize) -> usize {
        iy * self.width + ix
    }

    /// World coordinates of the centre of cell `(ix, iy)`.
    pub fn cell_center(&self, ix: usize, iy: usize) -> Point2<f64> {
        Point2::new(
            self.origin.x + (ix as f64 + 0.5) * self.resolution,
            self.origin.y + (iy as f64 + 0.5) * self.resolution,
        )
    }

    /// Cell containing world point `p`, if inside the grid.
    pub fn cell_of(&self, p: &Point2<f64>) -> Option<(usize, usize)> {
        let fx = ((p.x - self.origin.x) / self.resolution).floor();
        let fy = ((p.y - self.origin.y) / self.resolution).floor();
        if fx < 0.0 || fy < 0.0 || fx >= self.width as f64 || fy >= self.height as f64 {
            return None;
        }
        Some((fx as usize, fy as usize))
    }
}

/// Ground-truth state of a grid cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CellState {
    /// A wall passes through the cell.
    Occupied,
    /// The cell centre is in walkable space and no wall touches the cell.
    Free,
    /// Inside an obstacle or outside the walkable area.
    Unknown,
}

/// Ground-truth occupancy raster of a [`World2d`].
#[derive(Clone, Debug, PartialEq)]
pub struct GroundTruthGrid {
    /// Grid geometry.
    pub layout: GridLayout,
    /// Row-major cell states (`layout.len()` entries).
    pub cells: Vec<CellState>,
}

impl GroundTruthGrid {
    /// Rasterises `world` on the given grid. Cells touched by any wall
    /// segment are [`CellState::Occupied`]; remaining cells whose centre is
    /// walkable are [`CellState::Free`]; everything else is
    /// [`CellState::Unknown`].
    pub fn from_world(
        world: &World2d,
        origin: Point2<f64>,
        resolution: f64,
        width: usize,
        height: usize,
    ) -> Self {
        let layout = GridLayout {
            origin,
            resolution,
            width,
            height,
        };
        let mut cells = vec![CellState::Unknown; layout.len()];
        for iy in 0..height {
            for ix in 0..width {
                if world.is_free(&layout.cell_center(ix, iy)) {
                    cells[layout.index(ix, iy)] = CellState::Free;
                }
            }
        }
        for s in &world.segments {
            let step = resolution * 0.25;
            let n = (s.length() / step).ceil().max(1.0) as usize;
            for k in 0..=n {
                let p = s.a + (s.b - s.a) * (k as f64 / n as f64);
                if let Some((ix, iy)) = layout.cell_of(&p) {
                    cells[layout.index(ix, iy)] = CellState::Occupied;
                }
            }
        }
        Self { layout, cells }
    }

    /// State of cell `(ix, iy)`.
    pub fn get(&self, ix: usize, iy: usize) -> CellState {
        self.cells[self.layout.index(ix, iy)]
    }

    /// Number of cells in the given state.
    pub fn count(&self, state: CellState) -> usize {
        self.cells.iter().filter(|&&c| c == state).count()
    }
}

/// A plain probability grid (e.g. an estimated SLAM map resampled into world
/// coordinates). `None` means unknown / never observed.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbabilityGrid {
    /// Grid geometry.
    pub layout: GridLayout,
    /// Row-major occupancy probabilities.
    pub cells: Vec<Option<f32>>,
}

impl ProbabilityGrid {
    /// Builds a grid by evaluating `f` at every cell centre.
    pub fn from_fn(layout: GridLayout, mut f: impl FnMut(Point2<f64>) -> Option<f32>) -> Self {
        let mut cells = Vec::with_capacity(layout.len());
        for iy in 0..layout.height {
            for ix in 0..layout.width {
                cells.push(f(layout.cell_center(ix, iy)));
            }
        }
        Self { layout, cells }
    }

    /// Probability at world point `p` (`None` outside the grid or unknown).
    pub fn lookup(&self, p: &Point2<f64>) -> Option<f32> {
        let (ix, iy) = self.layout.cell_of(p)?;
        self.cells[self.layout.index(ix, iy)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world2d::Polygon2;

    #[test]
    fn rasterize_room() {
        let mut w = World2d::new();
        w.add_enclosure(Polygon2::rect(0.0, 0.0, 4.0, 2.0));
        // Offset by half a cell so walls fall in cell interiors.
        let g = GroundTruthGrid::from_world(&w, Point2::new(-0.55, -0.55), 0.1, 52, 32);
        let occ = g.count(CellState::Occupied);
        // Perimeter: 2 * (40 + 20) cells plus/minus corners.
        assert!((118..=124).contains(&occ), "occupied = {occ}");
        let (ix, iy) = g.layout.cell_of(&Point2::new(2.0, 1.0)).unwrap();
        assert_eq!(g.get(ix, iy), CellState::Free);
        let (ix, iy) = g.layout.cell_of(&Point2::new(0.0, 1.0)).unwrap();
        assert_eq!(g.get(ix, iy), CellState::Occupied);
        let (ix, iy) = g.layout.cell_of(&Point2::new(-0.3, 1.0)).unwrap();
        assert_eq!(g.get(ix, iy), CellState::Unknown);
        // Interior: 39 x 19 free cells.
        assert_eq!(g.count(CellState::Free), 39 * 19);
    }

    #[test]
    fn layout_roundtrip() {
        let l = GridLayout {
            origin: Point2::new(-1.0, 2.0),
            resolution: 0.05,
            width: 10,
            height: 20,
        };
        let c = l.cell_center(3, 7);
        assert_eq!(l.cell_of(&c), Some((3, 7)));
        assert_eq!(l.cell_of(&Point2::new(-1.01, 2.5)), None);
    }

    #[test]
    fn world_rasterize_padding() {
        let g = World2d::loop_corridor().rasterize(0.1, 1.0);
        assert!(g.count(CellState::Occupied) > 1000);
        assert!(g.count(CellState::Free) > 10_000);
    }
}
