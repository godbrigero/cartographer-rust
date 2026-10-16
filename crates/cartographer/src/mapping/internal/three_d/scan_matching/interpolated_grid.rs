//! Port of `cartographer/mapping/internal/3d/scan_matching/interpolated_grid` (.h).
//!
//! Piecewise tricubic interpolation (with vanishing derivatives at the voxel
//! centers) over a [`HybridGrid`] or an [`IntensityHybridGrid`], generic over
//! the scalar type so it can be used inside Ceres cost functors with jets.

use ceres::Scalar;
use nalgebra::Vector3;

use crate::mapping::three_d::hybrid_grid::{
    Array3i, HybridGrid, HybridGridBase, IntensityHybridGrid,
};

/// A 3D grid that can be interpolated by [`InterpolatedGrid`].
///
/// RUST PORT NOTE: replaces the C++ template parameter `HybridGridType` and the
/// overloaded static `GetValue` functions.
pub trait InterpolationSource {
    /// Edge length of each voxel.
    fn resolution(&self) -> f32;
    /// Index of the voxel containing `point`.
    fn get_cell_index(&self, point: &Vector3<f32>) -> Array3i;
    /// The value at `index` (C++ `GetValue`).
    fn get_value(&self, index: &Array3i) -> f32;
}

impl InterpolationSource for HybridGrid {
    #[inline]
    fn resolution(&self) -> f32 {
        HybridGridBase::resolution(&**self)
    }
    #[inline]
    fn get_cell_index(&self, point: &Vector3<f32>) -> Array3i {
        HybridGridBase::get_cell_index(&**self, point)
    }
    #[inline]
    fn get_value(&self, index: &Array3i) -> f32 {
        self.get_probability(index)
    }
}

impl InterpolationSource for IntensityHybridGrid {
    #[inline]
    fn resolution(&self) -> f32 {
        HybridGridBase::resolution(&**self)
    }
    #[inline]
    fn get_cell_index(&self, point: &Vector3<f32>) -> Array3i {
        HybridGridBase::get_cell_index(&**self, point)
    }
    #[inline]
    fn get_value(&self, index: &Array3i) -> f32 {
        self.get_intensity(index)
    }
}

/// Interpolates between [`HybridGrid`] (or [`IntensityHybridGrid`]) voxel
/// values. Holds a borrow of the grid (the C++ class holds a reference).
#[derive(Clone, Copy)]
pub struct InterpolatedGrid<'a, G: InterpolationSource> {
    hybrid_grid: &'a G,
}

/// C++ `InterpolatedIntensityGrid`.
pub type InterpolatedIntensityGrid<'a> = InterpolatedGrid<'a, IntensityHybridGrid>;
/// C++ `InterpolatedProbabilityGrid`.
pub type InterpolatedProbabilityGrid<'a> = InterpolatedGrid<'a, HybridGrid>;

impl<'a, G: InterpolationSource> InterpolatedGrid<'a, G> {
    /// Creates an interpolator over `hybrid_grid`.
    pub fn new(hybrid_grid: &'a G) -> Self {
        Self { hybrid_grid }
    }

    /// Returns the interpolated value at (x, y, z) of the grid used to
    /// perform the interpolation.
    ///
    /// This is a piecewise, continuously differentiable function. We use the
    /// scalar part of Jet parameters to select our interval below. It is the
    /// tensor product volume of piecewise cubic polynomials that interpolate
    /// the values, and have vanishing derivative at the interval boundaries.
    #[inline]
    pub fn get_interpolated_value<T: Scalar>(&self, x: T, y: T, z: T) -> T {
        let (x1, y1, z1, x2, y2, z2) = self.compute_interpolation_data_points(
            x.scalar_part(),
            y.scalar_part(),
            z.scalar_part(),
        );

        let grid = self.hybrid_grid;
        let index1 = grid.get_cell_index(&Vector3::new(x1 as f32, y1 as f32, z1 as f32));
        let value = |dx: i32, dy: i32, dz: i32| -> f64 {
            f64::from(grid.get_value(&(index1 + Array3i::new(dx, dy, dz))))
        };
        let q111 = value(0, 0, 0);
        let q112 = value(0, 0, 1);
        let q121 = value(0, 1, 0);
        let q122 = value(0, 1, 1);
        let q211 = value(1, 0, 0);
        let q212 = value(1, 0, 1);
        let q221 = value(1, 1, 0);
        let q222 = value(1, 1, 1);

        let normalized_x = (x - T::constant(x1)) / T::constant(x2 - x1);
        let normalized_y = (y - T::constant(y1)) / T::constant(y2 - y1);
        let normalized_z = (z - T::constant(z1)) / T::constant(z2 - z1);

        // Compute pow(..., 2) and pow(..., 3). Using pow() here is very
        // expensive.
        let normalized_xx = normalized_x * normalized_x;
        let normalized_xxx = normalized_x * normalized_xx;
        let normalized_yy = normalized_y * normalized_y;
        let normalized_yyy = normalized_y * normalized_yy;
        let normalized_zz = normalized_z * normalized_z;
        let normalized_zzz = normalized_z * normalized_zz;

        // We first interpolate in z, then y, then x. All 7 times this uses the
        // same scheme: A * (2t^3 - 3t^2 + 1) + B * (-2t^3 + 3t^2).
        // The first polynomial is 1 at t=0, 0 at t=1, the second polynomial is
        // 0 at t=0, 1 at t=1. Both polynomials have derivative zero at t=0 and
        // t=1.
        let two = T::constant(2.);
        let three = T::constant(3.);
        // Values that are constants (not jets) in C++ (doubles).
        let blend_constant = |a: f64, b: f64, tt: T, ttt: T| -> T {
            T::constant(a - b) * ttt * two + T::constant(b - a) * tt * three + T::constant(a)
        };
        let blend =
            |a: T, b: T, tt: T, ttt: T| -> T { (a - b) * ttt * two + (b - a) * tt * three + a };
        let q11 = blend_constant(q111, q112, normalized_zz, normalized_zzz);
        let q12 = blend_constant(q121, q122, normalized_zz, normalized_zzz);
        let q21 = blend_constant(q211, q212, normalized_zz, normalized_zzz);
        let q22 = blend_constant(q221, q222, normalized_zz, normalized_zzz);
        let q1 = blend(q11, q12, normalized_yy, normalized_yyy);
        let q2 = blend(q21, q22, normalized_yy, normalized_yyy);
        blend(q1, q2, normalized_xx, normalized_xxx)
    }

    #[inline]
    fn compute_interpolation_data_points(
        &self,
        x: f64,
        y: f64,
        z: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let lower = self.center_of_lower_voxel(x, y, z);
        let resolution = self.hybrid_grid.resolution();
        (
            f64::from(lower.x),
            f64::from(lower.y),
            f64::from(lower.z),
            f64::from(lower.x + resolution),
            f64::from(lower.y + resolution),
            f64::from(lower.z + resolution),
        )
    }

    /// Center of the next lower voxel, i.e., not necessarily the voxel
    /// containing (x, y, z). For each dimension, the largest voxel index so
    /// that the corresponding center is at most the given coordinate.
    ///
    /// RUST PORT NOTE: the C++ Jet overload forwards the scalar parts; here
    /// the caller passes `scalar_part()` directly.
    #[inline]
    fn center_of_lower_voxel(&self, x: f64, y: f64, z: f64) -> Vector3<f32> {
        let grid = self.hybrid_grid;
        let resolution = grid.resolution();
        // Center of the cell containing (x, y, z).
        let index = grid.get_cell_index(&Vector3::new(x as f32, y as f32, z as f32));
        // C++ `GetCenterOfCell`: `index.cast<float>() * resolution`.
        let mut center = index.cast::<f32>() * resolution;
        // Move to the next lower voxel center.
        if f64::from(center.x) > x {
            center.x -= resolution;
        }
        if f64::from(center.y) > y {
            center.y -= resolution;
        }
        if f64::from(center.z) > z {
            center.z -= resolution;
        }
        center
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod interpolated_grid_test {
        use super::*;

        fn make_grid() -> HybridGrid {
            let mut hybrid_grid = HybridGrid::new(0.1);
            for point in [
                Vector3::new(-3.0f32, 2., 0.),
                Vector3::new(-4., 2., 0.),
                Vector3::new(-5., 2., 0.),
                Vector3::new(-6., 2., 0.),
                Vector3::new(-6., 3., 1.),
                Vector3::new(-6., 4., 2.),
                Vector3::new(-7., 3., 1.),
            ] {
                let index = hybrid_grid.get_cell_index(&point);
                hybrid_grid.set_probability(&index, 1.);
            }
            hybrid_grid
        }

        fn get_hybrid_grid_probability(grid: &HybridGrid, x: f32, y: f32, z: f32) -> f32 {
            grid.get_probability(&grid.get_cell_index(&Vector3::new(x, y, z)))
        }

        /// Calls `f(x, y, z)` for the C++ test loops (double accumulation of
        /// the float resolution).
        fn for_each_sample(resolution: f64, mut f: impl FnMut(f64, f64, f64)) {
            let mut z = -1.;
            while z < 3. {
                let mut y = 1.;
                while y < 5. {
                    let mut x = -8.;
                    while x < -2. {
                        f(x, y, z);
                        x += resolution;
                    }
                    y += resolution;
                }
                z += resolution;
            }
        }

        #[test]
        fn interpolates_grid_points() {
            let hybrid_grid = make_grid();
            let interpolated_grid = InterpolatedProbabilityGrid::new(&hybrid_grid);
            let resolution = f64::from(hybrid_grid.resolution());
            for_each_sample(resolution, |x, y, z| {
                let expected = f64::from(get_hybrid_grid_probability(
                    &hybrid_grid,
                    x as f32,
                    y as f32,
                    z as f32,
                ));
                let actual = interpolated_grid.get_interpolated_value(x, y, z);
                assert!(
                    (expected - actual).abs() <= 1e-6,
                    "({x}, {y}, {z}): {expected} vs {actual}"
                );
            });
        }

        #[test]
        fn monotonic_behavior_between_grid_points_in_x() {
            let hybrid_grid = make_grid();
            let interpolated_grid = InterpolatedProbabilityGrid::new(&hybrid_grid);
            let resolution = f64::from(hybrid_grid.resolution());
            let sample_step = resolution / 10.;
            let mut num_checked = 0;
            for_each_sample(resolution, |x, y, z| {
                let start_probability =
                    get_hybrid_grid_probability(&hybrid_grid, x as f32, y as f32, z as f32);
                let next_probability = get_hybrid_grid_probability(
                    &hybrid_grid,
                    (x + resolution) as f32,
                    y as f32,
                    z as f32,
                );
                let grid_difference = next_probability - start_probability;
                if grid_difference.abs() < 1e-6 {
                    return;
                }
                let mut sample = sample_step;
                while sample < resolution - 2. * sample_step {
                    let delta =
                        interpolated_grid.get_interpolated_value(x + sample + sample_step, y, z)
                            - interpolated_grid.get_interpolated_value(x + sample, y, z);
                    assert!(
                        0. < f64::from(grid_difference) * delta,
                        "({x}, {y}, {z}) sample {sample}"
                    );
                    num_checked += 1;
                    sample += sample_step;
                }
            });
            assert!(num_checked > 0);
        }
    }

    mod additional {
        use super::*;
        use ceres::Jet;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        fn random_grid(resolution: f32, seed: u64) -> HybridGrid {
            let mut grid = HybridGrid::new(resolution);
            let mut rng = StdRng::seed_from_u64(seed);
            for z in -3..=3 {
                for y in -3..=3 {
                    for x in -3..=3 {
                        if rng.random_bool(0.7) {
                            grid.set_probability(
                                &Array3i::new(x, y, z),
                                rng.random_range(0.1..0.9),
                            );
                        }
                    }
                }
            }
            grid
        }

        #[test]
        fn matches_hybrid_grid_at_cell_centers() {
            let grid = random_grid(0.25, 7);
            let interpolated = InterpolatedProbabilityGrid::new(&grid);
            for z in -4..=4 {
                for y in -4..=4 {
                    for x in -4..=4 {
                        let index = Array3i::new(x, y, z);
                        let center = grid.get_center_of_cell(&index).cast::<f64>();
                        let value =
                            interpolated.get_interpolated_value(center.x, center.y, center.z);
                        assert!(
                            (value - f64::from(grid.get_probability(&index))).abs() < 1e-6,
                            "{index:?}"
                        );
                    }
                }
            }
        }

        #[test]
        fn intensity_grid_interpolates_cell_centers() {
            let mut grid = IntensityHybridGrid::new(0.5);
            grid.add_intensity(&Array3i::new(1, 0, 0), 40.);
            grid.add_intensity(&Array3i::new(1, 0, 0), 60.);
            let interpolated = InterpolatedIntensityGrid::new(&grid);
            assert!((interpolated.get_interpolated_value(0.5f64, 0., 0.) - 50.).abs() < 1e-9);
            assert!(interpolated.get_interpolated_value(0f64, 0., 0.).abs() < 1e-9);
            // Half way between, the Hermite blend gives exactly the mean.
            assert!((interpolated.get_interpolated_value(0.25f64, 0., 0.) - 25.).abs() < 1e-5);
        }

        #[test]
        fn jet_derivatives_match_finite_differences() {
            let grid = random_grid(0.3, 11);
            let interpolated = InterpolatedProbabilityGrid::new(&grid);
            let mut rng = StdRng::seed_from_u64(3);
            let h = 1e-6;
            for _ in 0..500 {
                let p = [
                    rng.random_range(-0.8..0.8),
                    rng.random_range(-0.8..0.8),
                    rng.random_range(-0.8..0.8),
                ];
                let jet = interpolated.get_interpolated_value(
                    Jet::<3>::variable(p[0], 0),
                    Jet::<3>::variable(p[1], 1),
                    Jet::<3>::variable(p[2], 2),
                );
                let value = interpolated.get_interpolated_value(p[0], p[1], p[2]);
                // Jet division multiplies by the reciprocal (as in C++), so the
                // scalar part may differ from the f64 path by an ulp.
                assert!((jet.a - value).abs() < 1e-12);
                for k in 0..3 {
                    let mut plus = p;
                    let mut minus = p;
                    plus[k] += h;
                    minus[k] -= h;
                    let numeric = (interpolated.get_interpolated_value(plus[0], plus[1], plus[2])
                        - interpolated.get_interpolated_value(minus[0], minus[1], minus[2]))
                        / (2. * h);
                    // The function is C1 (also across voxel boundaries), so
                    // central differences are accurate to O(h).
                    assert!(
                        (numeric - jet.v[k]).abs() < 1e-4,
                        "{p:?} axis {k}: numeric {numeric} vs jet {}",
                        jet.v[k]
                    );
                }
            }
        }

        #[test]
        fn gradient_vanishes_at_cell_centers() {
            let grid = random_grid(0.3, 5);
            let interpolated = InterpolatedProbabilityGrid::new(&grid);
            let center = grid
                .get_center_of_cell(&Array3i::new(1, -1, 2))
                .cast::<f64>();
            let jet = interpolated.get_interpolated_value(
                Jet::<3>::variable(center.x, 0),
                Jet::<3>::variable(center.y, 1),
                Jet::<3>::variable(center.z, 2),
            );
            for k in 0..3 {
                assert!(jet.v[k].abs() < 1e-9, "{:?}", jet.v);
            }
        }
    }
}
