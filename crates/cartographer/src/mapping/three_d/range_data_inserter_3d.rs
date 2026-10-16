//! Port of `cartographer/mapping/3d/range_data_inserter_3d` (.h, .cc).

use nalgebra::Vector3;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::probability_values::{compute_lookup_table_to_apply_odds, odds};
use crate::mapping::proto;
use crate::mapping::three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
use crate::sensor::point_cloud::PointCloud;
use crate::sensor::range_data::RangeData;

fn insert_misses_into_grid(
    miss_table: &[u16],
    origin: &Vector3<f32>,
    returns: &PointCloud,
    hybrid_grid: &mut HybridGrid,
    num_free_space_voxels: i32,
) {
    let origin_cell = hybrid_grid.get_cell_index(origin);
    for hit in returns {
        let hit_cell = hybrid_grid.get_cell_index(&hit.position);

        let delta = hit_cell - origin_cell;
        let num_samples = delta.x.abs().max(delta.y.abs()).max(delta.z.abs());
        assert!(
            num_samples < (1 << 15),
            "Check failed: num_samples < 1 << 15 ({num_samples} vs. 32768)"
        );
        // 'num_samples' is the number of samples we equi-distantly place on the
        // line between 'origin' and 'hit'. (including a fractional part for sub-
        // voxels) It is chosen so that between two samples we change from one
        // voxel to the next on the fastest changing dimension.
        //
        // Only the last 'num_free_space_voxels' are updated for performance.
        for position in 0.max(num_samples - num_free_space_voxels)..num_samples {
            // Integer arithmetic: `/` truncates toward zero like C++.
            let miss_cell = origin_cell + (delta * position) / num_samples;
            hybrid_grid.apply_lookup_table(&miss_cell, miss_table);
        }
    }
}

fn insert_intensities_into_grid(
    returns: &PointCloud,
    intensity_hybrid_grid: &mut IntensityHybridGrid,
    intensity_threshold: f32,
) {
    let intensities = returns.intensities();
    if !intensities.is_empty() {
        for (point, &intensity) in returns.points().iter().zip(intensities) {
            if intensity > intensity_threshold {
                continue;
            }
            let hit_cell = intensity_hybrid_grid.get_cell_index(&point.position);
            intensity_hybrid_grid.add_intensity(&hit_cell, intensity);
        }
    }
}

/// C++ `CreateRangeDataInserterOptions3D`.
pub fn create_range_data_inserter_options_3d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::RangeDataInserterOptions3D {
    let options = proto::RangeDataInserterOptions3D {
        hit_probability: parameter_dictionary.get_double("hit_probability"),
        miss_probability: parameter_dictionary.get_double("miss_probability"),
        num_free_space_voxels: parameter_dictionary.get_int("num_free_space_voxels"),
        intensity_threshold: parameter_dictionary.get_double("intensity_threshold") as f32,
    };
    assert!(
        options.hit_probability > 0.5,
        "Check failed: options.hit_probability() > 0.5 ({} vs. 0.5)",
        options.hit_probability
    );
    assert!(
        options.miss_probability < 0.5,
        "Check failed: options.miss_probability() < 0.5 ({} vs. 0.5)",
        options.miss_probability
    );
    options
}

/// Inserts range data into a [`HybridGrid`] (and optionally an
/// [`IntensityHybridGrid`]) using hit/miss odds lookup tables.
#[derive(Debug, Clone)]
pub struct RangeDataInserter3D {
    options: proto::RangeDataInserterOptions3D,
    hit_table: Vec<u16>,
    miss_table: Vec<u16>,
}

impl RangeDataInserter3D {
    /// Creates an inserter, precomputing the hit and miss lookup tables.
    pub fn new(options: &proto::RangeDataInserterOptions3D) -> Self {
        Self {
            options: *options,
            hit_table: compute_lookup_table_to_apply_odds(odds(options.hit_probability as f32)),
            miss_table: compute_lookup_table_to_apply_odds(odds(options.miss_probability as f32)),
        }
    }

    /// Inserts `range_data` into `hybrid_grid` and optionally into
    /// `intensity_hybrid_grid`.
    ///
    /// Hits are applied first, then misses along the rays (sampled on the
    /// dominant axis, only the last `num_free_space_voxels` before each hit),
    /// then intensities; finally the update is finished. `range_data.misses`
    /// is ignored.
    pub fn insert(
        &self,
        range_data: &RangeData,
        hybrid_grid: &mut HybridGrid,
        intensity_hybrid_grid: Option<&mut IntensityHybridGrid>,
    ) {
        for hit in &range_data.returns {
            let hit_cell = hybrid_grid.get_cell_index(&hit.position);
            hybrid_grid.apply_lookup_table(&hit_cell, &self.hit_table);
        }

        // By not starting a new update after hits are inserted, we give hits
        // priority (i.e. no hits will be ignored because of a miss in the same
        // cell).
        insert_misses_into_grid(
            &self.miss_table,
            &range_data.origin,
            &range_data.returns,
            hybrid_grid,
            self.options.num_free_space_voxels,
        );
        if let Some(intensity_hybrid_grid) = intensity_hybrid_grid {
            insert_intensities_into_grid(
                &range_data.returns,
                intensity_hybrid_grid,
                self.options.intensity_threshold,
            );
        }
        hybrid_grid.finish_update();
    }
}

#[cfg(test)]
mod tests {
    mod range_data_inserter_3d_test {
        use nalgebra::Vector3;

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY};
        use crate::mapping::proto;
        use crate::mapping::three_d::hybrid_grid::{HybridGrid, IntensityHybridGrid};
        use crate::mapping::three_d::range_data_inserter_3d::*;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        fn near(a: f64, b: f64, eps: f64) {
            assert!((a - b).abs() <= eps, "expected {a} near {b} (eps {eps})");
        }

        struct Fixture {
            hybrid_grid: HybridGrid,
            intensity_hybrid_grid: IntensityHybridGrid,
            range_data_inserter: RangeDataInserter3D,
            options: proto::RangeDataInserterOptions3D,
        }

        fn returns() -> Vec<RangefinderPoint> {
            [
                Vector3::new(-3.0, -1.0, 4.0),
                Vector3::new(-2.0, 0.0, 4.0),
                Vector3::new(-1.0, 1.0, 4.0),
                Vector3::new(0.0, 2.0, 4.0),
            ]
            .into_iter()
            .map(|position| RangefinderPoint { position })
            .collect()
        }

        impl Fixture {
            fn new() -> Self {
                let parameter_dictionary = make_dictionary(
                    "return { \
                     hit_probability = 0.7, \
                     miss_probability = 0.4, \
                     num_free_space_voxels = 1000, \
                     intensity_threshold = 100, \
                     }",
                );
                let options = create_range_data_inserter_options_3d(&parameter_dictionary);
                Self {
                    hybrid_grid: HybridGrid::new(1.0),
                    intensity_hybrid_grid: IntensityHybridGrid::new(1.0),
                    range_data_inserter: RangeDataInserter3D::new(&options),
                    options,
                }
            }

            fn insert_point_cloud(&mut self) {
                let range_data = RangeData {
                    origin: Vector3::new(0.0, 0.0, -4.0),
                    returns: PointCloud::from_points(returns()),
                    misses: PointCloud::new(),
                };
                self.range_data_inserter
                    .insert(&range_data, &mut self.hybrid_grid, None);
            }

            fn insert_point_cloud_with_intensities(&mut self) {
                let range_data = RangeData {
                    origin: Vector3::new(0.0, 0.0, -4.0),
                    returns: PointCloud::with_intensities(returns(), vec![7.0, 8.0, 9.0, 10.0]),
                    misses: PointCloud::new(),
                };
                self.range_data_inserter.insert(
                    &range_data,
                    &mut self.hybrid_grid,
                    Some(&mut self.intensity_hybrid_grid),
                );
            }

            fn get_probability(&self, x: f32, y: f32, z: f32) -> f64 {
                f64::from(
                    self.hybrid_grid
                        .get_probability(&self.hybrid_grid.get_cell_index(&Vector3::new(x, y, z))),
                )
            }

            fn get_intensity(&self, x: f32, y: f32, z: f32) -> f64 {
                f64::from(
                    self.intensity_hybrid_grid.get_intensity(
                        &self
                            .intensity_hybrid_grid
                            .get_cell_index(&Vector3::new(x, y, z)),
                    ),
                )
            }

            fn is_known(&self, x: f32, y: f32, z: f32) -> bool {
                self.hybrid_grid
                    .is_known(&self.hybrid_grid.get_cell_index(&Vector3::new(x, y, z)))
            }

            fn check_misses_along_ray(&self) {
                let miss = self.options.miss_probability;
                near(miss, self.get_probability(0.0, 0.0, -4.0), 1e-4);
                near(miss, self.get_probability(0.0, 0.0, -3.0), 1e-4);
                near(miss, self.get_probability(0.0, 0.0, -2.0), 1e-4);
            }
        }

        #[test]
        fn insert_point_cloud() {
            let mut f = Fixture::new();
            f.insert_point_cloud();
            f.check_misses_along_ray();
            for x in -4..=4 {
                for y in -4..=4 {
                    let (xf, yf) = (x as f32, y as f32);
                    if x < -3 || x > 0 || y != x + 2 {
                        assert!(!f.is_known(xf, yf, 4.0));
                    } else {
                        near(
                            f.options.hit_probability,
                            f.get_probability(xf, yf, 4.0),
                            1e-4,
                        );
                    }
                }
            }
        }

        #[test]
        fn insert_point_cloud_with_intensities() {
            let mut f = Fixture::new();
            f.insert_point_cloud_with_intensities();
            f.check_misses_along_ray();
            for x in -4..=4 {
                for y in -4..=4 {
                    let (xf, yf) = (x as f32, y as f32);
                    if x < -3 || x > 0 || y != x + 2 {
                        assert!(!f.is_known(xf, yf, 4.0));
                        near(0.0, f.get_intensity(xf, yf, 4.0), 1e-6);
                    } else {
                        near(
                            f.options.hit_probability,
                            f.get_probability(xf, yf, 4.0),
                            1e-4,
                        );
                        near(f64::from(10 + x), f.get_intensity(xf, yf, 4.0), 1e-6);
                    }
                }
            }
        }

        #[test]
        fn probability_progression() {
            let mut f = Fixture::new();
            f.insert_point_cloud();
            near(
                f.options.hit_probability,
                f.get_probability(-2.0, 0.0, 4.0),
                1e-4,
            );
            near(
                f.options.miss_probability,
                f.get_probability(-2.0, 0.0, 3.0),
                1e-4,
            );
            near(
                f.options.miss_probability,
                f.get_probability(0.0, 0.0, -3.0),
                1e-4,
            );

            for _ in 0..1000 {
                f.insert_point_cloud();
            }
            near(
                f64::from(MAX_PROBABILITY),
                f.get_probability(-2.0, 0.0, 4.0),
                1e-3,
            );
            near(
                f64::from(MIN_PROBABILITY),
                f.get_probability(-2.0, 0.0, 3.0),
                1e-3,
            );
            near(
                f64::from(MIN_PROBABILITY),
                f.get_probability(0.0, 0.0, -3.0),
                1e-3,
            );
        }
    }

    /// Additional tests (no C++ counterpart).
    mod extra {
        use nalgebra::Vector3;

        use crate::mapping::proto;
        use crate::mapping::three_d::hybrid_grid::{Array3i, HybridGrid, IntensityHybridGrid};
        use crate::mapping::three_d::range_data_inserter_3d::*;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        fn options(num_free_space_voxels: i32) -> proto::RangeDataInserterOptions3D {
            proto::RangeDataInserterOptions3D {
                hit_probability: 0.55,
                miss_probability: 0.49,
                num_free_space_voxels,
                intensity_threshold: 100.0,
            }
        }

        /// Rays to every wall cell of a 4 m box room, sensor in the center.
        fn box_room_returns() -> Vec<RangefinderPoint> {
            let mut points = Vec::new();
            for a in -20..=20 {
                for b in -20..=20 {
                    let (u, v) = (a as f32 * 0.1, b as f32 * 0.1);
                    for &w in &[-2.0f32, 2.0] {
                        points.push(RangefinderPoint {
                            position: Vector3::new(w, u, v),
                        });
                        points.push(RangefinderPoint {
                            position: Vector3::new(u, w, v),
                        });
                        points.push(RangefinderPoint {
                            position: Vector3::new(u, v, w),
                        });
                    }
                }
            }
            points
        }

        #[test]
        fn box_room_walls_occupied_interior_free() {
            let inserter = RangeDataInserter3D::new(&options(1000));
            let mut grid = HybridGrid::new(0.1);
            let mut intensity_grid = IntensityHybridGrid::new(0.1);
            let range_data = RangeData {
                origin: Vector3::zeros(),
                returns: PointCloud::from_points(box_room_returns()),
                misses: PointCloud::new(),
            };
            for _ in 0..50 {
                inserter.insert(&range_data, &mut grid, Some(&mut intensity_grid));
            }
            // Walls are occupied.
            for &wall in &[
                Vector3::new(2.0f32, 0.0, 0.0),
                Vector3::new(-2.0, 1.0, -1.5),
                Vector3::new(0.3, 2.0, 0.7),
                Vector3::new(1.0, -1.0, -2.0),
                Vector3::new(-1.7, 1.2, 2.0),
            ] {
                let p = grid.get_probability(&grid.get_cell_index(&wall));
                assert!(p > 0.8, "wall {wall:?} has probability {p}");
            }
            // Interior is free.
            for &free in &[
                Vector3::new(0.0f32, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, -1.0, 1.0),
                Vector3::new(1.5, 1.5, 1.5),
            ] {
                let index = grid.get_cell_index(&free);
                assert!(grid.is_known(&index));
                let p = grid.get_probability(&index);
                assert!(p < 0.5, "interior {free:?} has probability {p}");
            }
            // Outside is unknown.
            assert!(!grid.is_known(&grid.get_cell_index(&Vector3::new(3.0, 0.0, 0.0))));
            // No update markers left behind.
            for (_, &value) in &grid {
                assert!(value < crate::mapping::probability_values::UPDATE_MARKER);
            }
            // No intensities were given.
            assert_eq!(
                0.0,
                intensity_grid.get_intensity(&grid.get_cell_index(&Vector3::new(2.0, 0.0, 0.0)))
            );
        }

        #[test]
        fn miss_samples_truncate_toward_zero() {
            // origin cell (0,0,0), hit cell (-3, 2, 7): 7 samples, positions
            // 0..7, miss cells = delta * position / 7 truncated toward zero.
            let inserter = RangeDataInserter3D::new(&options(1000));
            let mut grid = HybridGrid::new(1.0);
            let range_data = RangeData {
                origin: Vector3::zeros(),
                returns: PointCloud::from_points(vec![RangefinderPoint {
                    position: Vector3::new(-3.0, 2.0, 7.0),
                }]),
                misses: PointCloud::new(),
            };
            inserter.insert(&range_data, &mut grid, None);
            let delta = Array3i::new(-3, 2, 7);
            let mut expected_known = Vec::new();
            for position in 0..7 {
                expected_known.push(delta * position / 7);
            }
            expected_known.push(delta);
            let mut known: Vec<Array3i> = (&grid).into_iter().map(|(index, _)| index).collect();
            known.sort_by_key(|i| (i.x, i.y, i.z));
            expected_known.sort_by_key(|i| (i.x, i.y, i.z));
            expected_known.dedup();
            assert_eq!(known, expected_known);
            // e.g. position 1: (-3/7, 2/7, 1) = (0, 0, 1), not (-1, 0, 1).
            assert!(grid.is_known(&Array3i::new(0, 0, 1)));
            assert!(!grid.is_known(&Array3i::new(-1, 0, 1)));
        }

        #[test]
        fn num_free_space_voxels_limits_misses() {
            let inserter = RangeDataInserter3D::new(&options(2));
            let mut grid = HybridGrid::new(1.0);
            let range_data = RangeData {
                origin: Vector3::zeros(),
                returns: PointCloud::from_points(vec![RangefinderPoint {
                    position: Vector3::new(10.0, 0.0, 0.0),
                }]),
                misses: PointCloud::new(),
            };
            inserter.insert(&range_data, &mut grid, None);
            let known: Vec<i32> = (&grid).into_iter().map(|(index, _)| index.x).collect();
            let mut known = known;
            known.sort_unstable();
            assert_eq!(known, vec![8, 9, 10]);

            // Zero free space voxels: only the hit.
            let inserter = RangeDataInserter3D::new(&options(0));
            let mut grid = HybridGrid::new(1.0);
            inserter.insert(&range_data, &mut grid, None);
            assert_eq!((&grid).into_iter().count(), 1);
        }

        #[test]
        fn intensity_threshold_is_respected() {
            let inserter = RangeDataInserter3D::new(&options(0));
            let mut grid = HybridGrid::new(1.0);
            let mut intensity_grid = IntensityHybridGrid::new(1.0);
            let points = vec![
                RangefinderPoint {
                    position: Vector3::new(1.0, 0.0, 0.0),
                },
                RangefinderPoint {
                    position: Vector3::new(2.0, 0.0, 0.0),
                },
                RangefinderPoint {
                    position: Vector3::new(2.0, 0.0, 0.0),
                },
            ];
            let range_data = RangeData {
                origin: Vector3::zeros(),
                returns: PointCloud::with_intensities(points, vec![100.0, 101.0, 50.0]),
                misses: PointCloud::new(),
            };
            inserter.insert(&range_data, &mut grid, Some(&mut intensity_grid));
            assert_eq!(100.0, intensity_grid.get_intensity(&Array3i::new(1, 0, 0)));
            // 101 is above the threshold and skipped.
            assert_eq!(50.0, intensity_grid.get_intensity(&Array3i::new(2, 0, 0)));
        }
    }
}
