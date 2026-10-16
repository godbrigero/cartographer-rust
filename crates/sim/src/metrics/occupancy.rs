//! Occupancy-map accuracy against a [`GroundTruthGrid`].

use nalgebra::Point2;

use crate::grid::{CellState, GroundTruthGrid};

/// Parameters of [`occupancy_metrics`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyMetricsConfig {
    /// A cell is estimated occupied if its probability is `>=` this.
    pub occupied_threshold: f32,
    /// Distance tolerance (m) for matching occupied cells (rounded down to
    /// whole cells, Euclidean disk).
    pub tolerance: f64,
}

impl Default for OccupancyMetricsConfig {
    /// Threshold 0.65 (clearly occupied in Cartographer's probability
    /// range), tolerance 0.1 m.
    fn default() -> Self {
        Self {
            occupied_threshold: 0.65,
            tolerance: 0.1,
        }
    }
}

/// Result of [`occupancy_metrics`]. All ratios are in `[0, 1]` and are `0`
/// when their denominator is empty.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OccupancyMetrics {
    /// Fraction of estimated-occupied cells within `tolerance` of a GT
    /// occupied cell.
    pub precision: f64,
    /// Fraction of GT occupied cells within `tolerance` of an
    /// estimated-occupied cell.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// Among GT free cells farther than `tolerance` from every GT occupied
    /// cell and observed (known) in the estimate, the fraction marked
    /// occupied.
    pub free_marked_occupied: f64,
    /// Fraction of GT free cells that are known in the estimate.
    pub free_coverage: f64,
    /// Number of estimated-occupied cells (on the GT grid).
    pub estimated_occupied: usize,
    /// Number of GT occupied cells.
    pub gt_occupied: usize,
    /// Number of GT free cells.
    pub gt_free: usize,
}

/// Compares an estimated map with the ground truth on the GT grid.
///
/// `estimate` returns the occupancy probability at a **world** point (GT
/// frame), or `None` if unknown; it is evaluated at every GT cell centre.
/// If the estimated map lives in another frame (e.g. Cartographer's map
/// frame anchored at the first tracking pose), transform the query point
/// inside the closure.
pub fn occupancy_metrics(
    gt: &GroundTruthGrid,
    estimate: impl Fn(&Point2<f64>) -> Option<f32>,
    config: &OccupancyMetricsConfig,
) -> OccupancyMetrics {
    let l = gt.layout;
    let n = l.len();
    let mut est_prob = Vec::with_capacity(n);
    for iy in 0..l.height {
        for ix in 0..l.width {
            est_prob.push(estimate(&l.cell_center(ix, iy)));
        }
    }
    let est_occ: Vec<bool> = est_prob
        .iter()
        .map(|p| p.is_some_and(|p| p >= config.occupied_threshold))
        .collect();
    let gt_occ: Vec<bool> = gt.cells.iter().map(|&c| c == CellState::Occupied).collect();
    let radius = (config.tolerance / l.resolution + 1e-9).floor().max(0.0) as i64;
    let gt_dil = dilate(&gt_occ, l.width, l.height, radius);
    let est_dil = dilate(&est_occ, l.width, l.height, radius);

    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    let n_est = est_occ.iter().filter(|&&b| b).count();
    let n_gt = gt_occ.iter().filter(|&&b| b).count();
    let tp_est = (0..n).filter(|&i| est_occ[i] && gt_dil[i]).count();
    let tp_gt = (0..n).filter(|&i| gt_occ[i] && est_dil[i]).count();
    let precision = ratio(tp_est, n_est);
    let recall = ratio(tp_gt, n_gt);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let gt_free: Vec<usize> = (0..n).filter(|&i| gt.cells[i] == CellState::Free).collect();
    let observed_free = gt_free.iter().filter(|&&i| est_prob[i].is_some()).count();
    let clear: Vec<usize> = gt_free
        .iter()
        .copied()
        .filter(|&i| !gt_dil[i] && est_prob[i].is_some())
        .collect();
    let false_occ = clear.iter().filter(|&&i| est_occ[i]).count();
    OccupancyMetrics {
        precision,
        recall,
        f1,
        free_marked_occupied: ratio(false_occ, clear.len()),
        free_coverage: ratio(observed_free, gt_free.len()),
        estimated_occupied: n_est,
        gt_occupied: n_gt,
        gt_free: gt_free.len(),
    }
}

/// Binary dilation with a Euclidean disk of `radius` cells.
fn dilate(mask: &[bool], width: usize, height: usize, radius: i64) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let offsets: Vec<(i64, i64)> = (-radius..=radius)
        .flat_map(|dy| (-radius..=radius).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| dx * dx + dy * dy <= radius * radius)
        .collect();
    let mut out = vec![false; mask.len()];
    for iy in 0..height as i64 {
        for ix in 0..width as i64 {
            if !mask[(iy * width as i64 + ix) as usize] {
                continue;
            }
            for (dx, dy) in &offsets {
                let (x, y) = (ix + dx, iy + dy);
                if x >= 0 && y >= 0 && x < width as i64 && y < height as i64 {
                    out[(y * width as i64 + x) as usize] = true;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::ProbabilityGrid;
    use crate::world2d::World2d;

    fn gt_as_estimate(gt: &GroundTruthGrid, shift: (f64, f64)) -> ProbabilityGrid {
        ProbabilityGrid::from_fn(gt.layout, |p| {
            let q = Point2::new(p.x - shift.0, p.y - shift.1);
            let (ix, iy) = gt.layout.cell_of(&q)?;
            match gt.get(ix, iy) {
                CellState::Occupied => Some(0.9),
                CellState::Free => Some(0.1),
                CellState::Unknown => None,
            }
        })
    }

    #[test]
    fn perfect_map() {
        let gt = World2d::rect_room().rasterize(0.05, 0.5);
        let est = gt_as_estimate(&gt, (0.0, 0.0));
        let m = occupancy_metrics(&gt, |p| est.lookup(p), &OccupancyMetricsConfig::default());
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1, 1.0);
        assert_eq!(m.free_marked_occupied, 0.0);
        assert_eq!(m.free_coverage, 1.0);
        assert_eq!(m.estimated_occupied, m.gt_occupied);
    }

    #[test]
    fn shifted_map_and_tolerance() {
        let gt = World2d::rect_room().rasterize(0.05, 0.5);
        let est = gt_as_estimate(&gt, (0.1, 0.0)); // 2 cells
        let strict = OccupancyMetricsConfig {
            tolerance: 0.0,
            ..Default::default()
        };
        let m = occupancy_metrics(&gt, |p| est.lookup(p), &strict);
        assert!(m.precision < 0.7 && m.recall < 0.7, "{m:?}");
        let loose = OccupancyMetricsConfig {
            tolerance: 0.1,
            ..Default::default()
        };
        let m = occupancy_metrics(&gt, |p| est.lookup(p), &loose);
        assert!(m.precision > 0.999 && m.recall > 0.999, "{m:?}");
    }

    #[test]
    fn spurious_occupied_free_cells() {
        let gt = World2d::rect_room().rasterize(0.1, 0.5);
        let l = gt.layout;
        // Mark every 10th GT free cell occupied.
        let mut k = 0;
        let mut marked = 0;
        let mut est = gt_as_estimate(&gt, (0.0, 0.0));
        for i in 0..l.len() {
            if gt.cells[i] == CellState::Free {
                if k % 10 == 0 {
                    est.cells[i] = Some(0.95);
                    marked += 1;
                }
                k += 1;
            }
        }
        assert!(marked > 0);
        let cfg = OccupancyMetricsConfig {
            tolerance: 0.0,
            ..Default::default()
        };
        let m = occupancy_metrics(&gt, |p| est.lookup(p), &cfg);
        assert!((m.free_marked_occupied - 0.1).abs() < 0.01, "{m:?}");
        assert_eq!(m.recall, 1.0);
        assert!(m.precision < 1.0);
        // Unknown estimate: zero coverage, zero recall.
        let m = occupancy_metrics(&gt, |_| None, &cfg);
        assert_eq!((m.recall, m.free_coverage, m.precision), (0.0, 0.0, 0.0));
    }
}
