//! Port of `cartographer/mapping/internal/2d/scan_matching/interpolated_tsdf_2d` (.h).

use ceres::Scalar;
use nalgebra::Vector2;

use crate::mapping::internal::two_d::tsdf_2d::TSDF2D;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::xy_index::Array2i;

/// Interpolates between [`TSDF2D`] pixels with bilinear interpolation.
///
/// Generic over [`Scalar`] to work with Ceres autodiff: the interpolation
/// cell is chosen at the scalar part, the corner values are constants and
/// the gradient flows through `(x, y)` only.
#[derive(Clone, Copy, Debug)]
pub struct InterpolatedTSDF2D<'a> {
    tsdf: &'a TSDF2D,
}

impl<'a> InterpolatedTSDF2D<'a> {
    /// Wraps `grid`.
    pub fn new(grid: &'a TSDF2D) -> Self {
        Self { tsdf: grid }
    }

    /// Returns the interpolated correspondence cost at (x,y). Cells with at
    /// least one 'unknown' interpolation point result in
    /// "MaxCorrespondenceCost()" with zero gradient.
    pub fn get_correspondence_cost<T: Scalar>(&self, x: T, y: T) -> T {
        let (x1, y1, x2, y2) = self.compute_interpolation_data_points(&x, &y);

        let index1 = self.tsdf.limits().get_cell_index(&Vector2::new(x1, y1));

        let w11 = self.tsdf.get_weight(&index1);
        let w12 = self.tsdf.get_weight(&(index1 + Array2i::new(-1, 0)));
        let w21 = self.tsdf.get_weight(&(index1 + Array2i::new(0, -1)));
        let w22 = self.tsdf.get_weight(&(index1 + Array2i::new(-1, -1)));
        if w11 == 0.0 || w12 == 0.0 || w21 == 0.0 || w22 == 0.0 {
            return T::constant(f64::from(self.tsdf.get_max_correspondence_cost()));
        }

        let q11 = self.tsdf.get_correspondence_cost(&index1);
        let q12 = self
            .tsdf
            .get_correspondence_cost(&(index1 + Array2i::new(-1, 0)));
        let q21 = self
            .tsdf
            .get_correspondence_cost(&(index1 + Array2i::new(0, -1)));
        let q22 = self
            .tsdf
            .get_correspondence_cost(&(index1 + Array2i::new(-1, -1)));

        interpolate_bilinear(x, y, x1, y1, x2, y2, q11, q12, q21, q22)
    }

    /// Returns the interpolated weight at (x,y).
    pub fn get_weight<T: Scalar>(&self, x: T, y: T) -> T {
        let (x1, y1, x2, y2) = self.compute_interpolation_data_points(&x, &y);

        let index1 = self.tsdf.limits().get_cell_index(&Vector2::new(x1, y1));
        let q11 = self.tsdf.get_weight(&index1);
        let q12 = self.tsdf.get_weight(&(index1 + Array2i::new(-1, 0)));
        let q21 = self.tsdf.get_weight(&(index1 + Array2i::new(0, -1)));
        let q22 = self.tsdf.get_weight(&(index1 + Array2i::new(-1, -1)));

        interpolate_bilinear(x, y, x1, y1, x2, y2, q11, q12, q21, q22)
    }

    /// Returns `(x1, y1, x2, y2)`: the center of the lower pixel and the
    /// center of the diagonally upper pixel.
    fn compute_interpolation_data_points<T: Scalar>(&self, x: &T, y: &T) -> (f32, f32, f32, f32) {
        // Uses the scalar part of a Ceres Jet.
        let lower = self.center_of_lower_pixel(x.scalar_part(), y.scalar_part());
        let resolution = self.tsdf.limits().resolution() as f32;
        (lower.x, lower.y, lower.x + resolution, lower.y + resolution)
    }

    /// Center of the next lower pixel, i.e., not necessarily the pixel
    /// containing (x, y). For each dimension, the largest pixel index so that
    /// the corresponding center is at most the given coordinate.
    fn center_of_lower_pixel(&self, x: f64, y: f64) -> Vector2<f32> {
        let limits = self.tsdf.limits();
        // Center of the cell containing (x, y).
        let mut center =
            limits.get_cell_center(&limits.get_cell_index(&Vector2::new(x as f32, y as f32)));
        // Move to the next lower pixel center (float -= double, computed in
        // double like C++).
        if f64::from(center.x) > x {
            center.x = (f64::from(center.x) - limits.resolution()) as f32;
        }
        if f64::from(center.y) > y {
            center.y = (f64::from(center.y) - limits.resolution()) as f32;
        }
        center
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_bilinear<T: Scalar>(
    x: T,
    y: T,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    q11: f32,
    q12: f32,
    q21: f32,
    q22: f32,
) -> T {
    let c = |v: f32| T::constant(f64::from(v));
    let normalized_x = (x - c(x1)) / c(x2 - x1);
    let normalized_y = (y - c(y1)) / c(y2 - y1);
    let q1 = c(q12 - q11) * normalized_y + c(q11);
    let q2 = c(q22 - q21) * normalized_y + c(q21);
    (q2 - q1) * normalized_x + q1
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::mapping::two_d::map_limits::MapLimits;
    use crate::mapping::two_d::xy_index::CellLimits;
    use crate::mapping::value_conversion_tables::ValueConversionTables;

    /// C++ fixture `InterpolatedTSDF2DTest`.
    struct Fixture {
        tsdf: TSDF2D,
    }

    impl Fixture {
        fn new() -> Self {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            Self {
                tsdf: TSDF2D::new(
                    MapLimits::new(1.0, Vector2::new(5.5, 5.5), CellLimits::new(10, 10)),
                    1.0,
                    10.0,
                    &conversion_tables,
                ),
            }
        }

        fn get_uninterpolated_correspondence_cost(&self, x: f32, y: f32) -> f32 {
            self.tsdf
                .get_correspondence_cost(&self.tsdf.limits().get_cell_index(&Vector2::new(x, y)))
        }

        fn get_uninterpolated_weight(&self, x: f32, y: f32) -> f32 {
            self.tsdf
                .get_weight(&self.tsdf.limits().get_cell_index(&Vector2::new(x, y)))
        }

        fn set_cell(&mut self, x: f32, y: f32, tsd: f32, weight: f32) {
            let cell = self.tsdf.limits().get_cell_index(&Vector2::new(x, y));
            self.tsdf.set_cell(&cell, tsd, weight);
        }
    }

    mod interpolated_tsdf_2d_test {
        use super::*;

        #[test]
        fn interpolates_grid_points() {
            let mut fixture = Fixture::new();
            let inner_points = [(1.0_f32, 1.0_f32), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)];
            for &(x, y) in &inner_points {
                fixture.set_cell(x, y, 0.1, 1.0);
            }
            // Insert surrounding points.
            for x in 0..4 {
                fixture.set_cell(x as f32, 0.0, 0.1, 1.0);
                fixture.set_cell(x as f32, 3.0, 0.1, 1.0);
            }
            for y in 1..3 {
                fixture.set_cell(0.0, y as f32, 0.1, 1.0);
                fixture.set_cell(3.0, y as f32, 0.1, 1.0);
            }
            let interpolated_tsdf = InterpolatedTSDF2D::new(&fixture.tsdf);
            for &(x, y) in &inner_points {
                let expected = f64::from(fixture.get_uninterpolated_correspondence_cost(x, y));
                let actual = interpolated_tsdf.get_correspondence_cost(f64::from(x), f64::from(y));
                assert!((expected - actual).abs() <= 1e-4, "{expected} vs. {actual}");
                let expected = f64::from(fixture.get_uninterpolated_weight(x, y));
                let actual = interpolated_tsdf.get_weight(f64::from(x), f64::from(y));
                assert!((expected - actual).abs() <= 1e-4, "{expected} vs. {actual}");
            }
            // Check unknown cell.
            let expected = f64::from(fixture.tsdf.get_max_correspondence_cost());
            let actual = interpolated_tsdf.get_correspondence_cost(3.0, 2.0);
            assert!((expected - actual).abs() <= 1e-4, "{expected} vs. {actual}");
            let expected = f64::from(fixture.get_uninterpolated_weight(3.0, 2.0));
            let actual = interpolated_tsdf.get_weight(3.0, 2.0);
            assert!((expected - actual).abs() <= 1e-4, "{expected} vs. {actual}");
        }

        #[test]
        fn interpolates_within_cell() {
            let mut fixture = Fixture::new();
            let tsd_00 = 0.1_f32;
            let tsd_01 = 0.2_f32;
            let tsd_10 = 0.3_f32;
            let tsd_11 = 0.4_f32;
            let w_00 = 1.0_f32;
            let w_01 = 2.0_f32;
            let w_10 = 3.0_f32;
            let w_11 = 4.0_f32;

            fixture.set_cell(0.0, 0.0, tsd_00, w_00);
            fixture.set_cell(0.0, 1.0, tsd_01, w_01);
            fixture.set_cell(1.0, 0.0, tsd_10, w_10);
            fixture.set_cell(1.0, 1.0, tsd_11, w_11);

            let interpolated_tsdf = InterpolatedTSDF2D::new(&fixture.tsdf);
            let resolution = fixture.tsdf.limits().resolution();
            let sample_step = resolution / 100.0;
            let mut num_samples = 0;
            let mut x = 0.0 + sample_step;
            while x < 1.0 {
                let mut y = 0.0 + sample_step;
                while y < 1.0 {
                    // `x`, `y` are doubles, so the C++ expression is
                    // evaluated in double.
                    let (xf, yf) = (x, y);
                    let tsd_expected = (xf * f64::from(tsd_10) + (1.0 - xf) * f64::from(tsd_00))
                        * (1.0 - yf)
                        + (xf * f64::from(tsd_11) + (1.0 - xf) * f64::from(tsd_01)) * yf;
                    let actual = interpolated_tsdf.get_correspondence_cost(x, y);
                    assert!(
                        (actual - tsd_expected).abs() <= 1e-3,
                        "{actual} vs. {tsd_expected}"
                    );
                    let w_expected = (xf * f64::from(w_10) + (1.0 - xf) * f64::from(w_00))
                        * (1.0 - yf)
                        + (xf * f64::from(w_11) + (1.0 - xf) * f64::from(w_01)) * yf;
                    let actual = interpolated_tsdf.get_weight(x, y);
                    assert!(
                        (actual - w_expected).abs() <= 1e-3,
                        "{actual} vs. {w_expected}"
                    );
                    num_samples += 1;
                    y += resolution;
                }
                x += resolution;
            }
            // Like in C++, the loops sample only (0.01, 0.01).
            assert_eq!(num_samples, 1);
        }

        // Not in C++: the gradient of a jet flows through (x, y).
        #[test]
        fn jet_gradient() {
            let mut fixture = Fixture::new();
            fixture.set_cell(0.0, 0.0, 0.1, 1.0);
            fixture.set_cell(0.0, 1.0, 0.2, 1.0);
            fixture.set_cell(1.0, 0.0, 0.3, 1.0);
            fixture.set_cell(1.0, 1.0, 0.4, 1.0);
            let interpolated_tsdf = InterpolatedTSDF2D::new(&fixture.tsdf);
            let x = ceres::Jet::<2>::variable(0.5, 0);
            let y = ceres::Jet::<2>::variable(0.5, 1);
            let cost = interpolated_tsdf.get_correspondence_cost(x, y);
            // d/dx = (tsd_1y - tsd_0y) averaged = 0.2, d/dy = 0.1.
            assert!((cost.v[0] - 0.2).abs() < 1e-3, "{:?}", cost.v);
            assert!((cost.v[1] - 0.1).abs() < 1e-3, "{:?}", cost.v);
        }
    }
}
