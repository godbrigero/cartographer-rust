//! Port of `cartographer/mapping/2d/map_limits` (.h).

use nalgebra::Vector2;

use crate::common::port::round_to_int;
use crate::mapping::proto;
use crate::mapping::two_d::xy_index::{self, Array2i, CellLimits};
use crate::transform::transform::{ToEigen, ToProto};

/// Defines the limits of a grid map.
///
/// Cell indices are transposed and flipped relative to the world frame: cell
/// `x` follows world `-y` and cell `y` follows world `-x`; cell `(0, 0)`
/// contains the corner `max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapLimits {
    resolution: f64,
    max: Vector2<f64>,
    cell_limits: CellLimits,
}

impl MapLimits {
    /// Creates limits. Panics unless `resolution > 0` and both cell counts
    /// are positive.
    pub fn new(resolution: f64, max: Vector2<f64>, cell_limits: CellLimits) -> Self {
        assert!(
            resolution > 0.0,
            "Check failed: resolution_ > 0. ({resolution} vs. 0)"
        );
        assert!(
            cell_limits.num_x_cells > 0,
            "Check failed: cell_limits.num_x_cells > 0. ({} vs. 0)",
            cell_limits.num_x_cells
        );
        assert!(
            cell_limits.num_y_cells > 0,
            "Check failed: cell_limits.num_y_cells > 0. ({} vs. 0)",
            cell_limits.num_y_cells
        );
        Self {
            resolution,
            max,
            cell_limits,
        }
    }

    /// Creates limits from a proto. Like C++, this does not validate the
    /// values (missing submessages are treated as all-zero defaults).
    pub fn from_proto(map_limits: &proto::MapLimits) -> Self {
        Self {
            resolution: map_limits.resolution,
            max: map_limits.max.unwrap_or_default().to_eigen(),
            cell_limits: CellLimits::from_proto(&map_limits.cell_limits.unwrap_or_default()),
        }
    }

    /// Returns the cell size in meters. All cells are square and the
    /// resolution is the length of one side.
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Returns the corner of the limits, i.e., all pixels have positions with
    /// smaller coordinates.
    pub fn max(&self) -> &Vector2<f64> {
        &self.max
    }

    /// Returns the limits of the grid in number of cells.
    pub fn cell_limits(&self) -> &CellLimits {
        &self.cell_limits
    }

    /// Returns the index of the cell containing the `point` which may be
    /// outside the map, i.e., negative or too large indices that will return
    /// false for [`contains`](Self::contains).
    ///
    /// The `f32` point is promoted to `f64` before the arithmetic, as in C++.
    #[inline]
    pub fn get_cell_index(&self, point: &Vector2<f32>) -> Array2i {
        // Index values are row major and the top left has Array2i::zeros()
        // and contains (centered_max_x, centered_max_y). We need to flip and
        // rotate.
        Array2i::new(
            round_to_int((self.max.y - f64::from(point.y)) / self.resolution - 0.5),
            round_to_int((self.max.x - f64::from(point.x)) / self.resolution - 0.5),
        )
    }

    /// Returns the center of the cell at `cell_index` (computed in `f64`,
    /// returned as `f32`).
    #[inline]
    pub fn get_cell_center(&self, cell_index: &Array2i) -> Vector2<f32> {
        Vector2::new(
            (self.max.x - self.resolution * (f64::from(cell_index.y) + 0.5)) as f32,
            (self.max.y - self.resolution * (f64::from(cell_index.x) + 0.5)) as f32,
        )
    }

    /// Returns true if the grid contains `cell_index`.
    #[inline]
    pub fn contains(&self, cell_index: &Array2i) -> bool {
        cell_index.x >= 0
            && cell_index.y >= 0
            && cell_index.x < self.cell_limits.num_x_cells
            && cell_index.y < self.cell_limits.num_y_cells
    }
}

/// Converts `map_limits` to a proto.
pub fn to_proto(map_limits: &MapLimits) -> proto::MapLimits {
    proto::MapLimits {
        resolution: map_limits.resolution(),
        max: Some(map_limits.max().to_proto()),
        cell_limits: Some(xy_index::to_proto(map_limits.cell_limits())),
    }
}

#[cfg(test)]
mod tests {
    mod map_limits_test {
        use nalgebra::Vector2;

        use crate::mapping::proto;
        use crate::mapping::two_d::map_limits::*;
        use crate::mapping::two_d::xy_index;
        use crate::transform::proto as transform_proto;

        #[test]
        fn to_proto_test() {
            let limits = MapLimits::new(42.0, Vector2::new(3.0, 0.0), CellLimits::new(2, 3));
            let proto = to_proto(&limits);
            assert_eq!(limits.resolution(), proto.resolution);
            assert_eq!(limits.max().x, proto.max.unwrap().x);
            assert_eq!(limits.max().y, proto.max.unwrap().y);
            assert_eq!(
                xy_index::to_proto(limits.cell_limits()),
                proto.cell_limits.unwrap()
            );
        }

        #[test]
        fn proto_constructor() {
            let limits = proto::MapLimits {
                resolution: 1.0,
                max: Some(transform_proto::Vector2d { x: 2.0, y: 3.0 }),
                cell_limits: Some(proto::CellLimits {
                    num_x_cells: 4,
                    num_y_cells: 5,
                }),
            };
            let native = MapLimits::from_proto(&limits);
            assert_eq!(limits.resolution, native.resolution());
            assert_eq!(limits.max.unwrap().x, native.max().x);
            assert_eq!(limits.max.unwrap().y, native.max().y);
            assert_eq!(
                limits.cell_limits.unwrap(),
                xy_index::to_proto(native.cell_limits())
            );
        }

        #[test]
        fn construct_and_get() {
            let limits = MapLimits::new(42.0, Vector2::new(3.0, 0.0), CellLimits::new(2, 3));
            let cell_limits = limits.cell_limits();
            assert_eq!(2, cell_limits.num_x_cells);
            assert_eq!(3, cell_limits.num_y_cells);
            let max = limits.max();
            assert_eq!(3.0, max.x);
            assert_eq!(0.0, max.y);
            assert_eq!(42.0, limits.resolution());
        }

        #[test]
        fn proto_round_trip() {
            let limits = MapLimits::new(0.05, Vector2::new(-1.5, 7.25), CellLimits::new(17, 9));
            assert_eq!(MapLimits::from_proto(&to_proto(&limits)), limits);
        }

        #[test]
        #[should_panic(expected = "resolution_ > 0.")]
        fn non_positive_resolution_panics() {
            let _ = MapLimits::new(0.0, Vector2::zeros(), CellLimits::new(1, 1));
        }

        #[test]
        #[should_panic(expected = "num_y_cells > 0.")]
        fn empty_cell_limits_panic() {
            let _ = MapLimits::new(1.0, Vector2::zeros(), CellLimits::new(1, 0));
        }
    }

    mod proptests {
        use nalgebra::Vector2;
        use proptest::prelude::*;

        use crate::mapping::two_d::map_limits::*;

        proptest! {
            #[test]
            fn cell_center_maps_back_to_cell_index(
                resolution in prop::sample::select(vec![0.025_f64, 0.05, 0.1, 0.5, 1.0, 2.0]),
                max_x in -50.0_f64..50.0, max_y in -50.0_f64..50.0,
                nx in 1i32..200, ny in 1i32..200, fx in 0.0_f64..1.0, fy in 0.0_f64..1.0) {
                let limits = MapLimits::new(
                    resolution, Vector2::new(max_x, max_y), CellLimits::new(nx, ny));
                let index = Array2i::new(
                    ((f64::from(nx) * fx) as i32).min(nx - 1),
                    ((f64::from(ny) * fy) as i32).min(ny - 1));
                prop_assert!(limits.contains(&index));
                let center = limits.get_cell_center(&index);
                prop_assert_eq!(limits.get_cell_index(&center), index);
            }

            #[test]
            fn point_lies_within_its_cell(
                resolution in prop::sample::select(vec![0.05_f64, 0.1, 1.0]),
                px in -20.0_f32..20.0, py in -20.0_f32..20.0) {
                let limits = MapLimits::new(
                    resolution, Vector2::new(25.0, 25.0), CellLimits::new(1000, 1000));
                let index = limits.get_cell_index(&Vector2::new(px, py));
                prop_assert!(limits.contains(&index));
                let center = limits.get_cell_center(&index);
                // Half a cell plus f32 rounding slack.
                let tolerance = 0.5 * resolution + 1e-4;
                prop_assert!(f64::from((center.x - px).abs()) <= tolerance);
                prop_assert!(f64::from((center.y - py).abs()) <= tolerance);
            }
        }
    }
}
