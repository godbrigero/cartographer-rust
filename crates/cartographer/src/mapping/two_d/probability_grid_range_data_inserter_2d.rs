//! Port of `cartographer/mapping/2d/probability_grid_range_data_inserter_2d` (.h, .cc).

use nalgebra::Vector2;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::grid_interface::GridInterface;
use crate::mapping::internal::two_d::ray_to_pixel_mask::ray_to_pixel_mask;
use crate::mapping::probability_values::{
    compute_lookup_table_to_apply_correspondence_cost_odds, odds,
};
use crate::mapping::proto;
use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
use crate::mapping::two_d::grid_2d::Grid2DTrait;
use crate::mapping::two_d::map_limits::MapLimits;
use crate::mapping::two_d::probability_grid::ProbabilityGrid;
use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
use crate::sensor::range_data::RangeData;

/// Factor for subpixel accuracy of start and end point for ray casts.
pub(crate) const SUBPIXEL_SCALE: i32 = 1000;

/// Padding around the bounding box to avoid numerical issues at cell
/// boundaries.
const PADDING: f32 = 1e-6;

/// Reads `ProbabilityGridRangeDataInserterOptions2D` from
/// `parameter_dictionary`. `insert_free_space` defaults to `true` if absent.
pub fn create_probability_grid_range_data_inserter_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::ProbabilityGridRangeDataInserterOptions2D {
    let options = proto::ProbabilityGridRangeDataInserterOptions2D {
        hit_probability: parameter_dictionary.get_double("hit_probability"),
        miss_probability: parameter_dictionary.get_double("miss_probability"),
        insert_free_space: if parameter_dictionary.has_key("insert_free_space") {
            parameter_dictionary.get_bool("insert_free_space")
        } else {
            true
        },
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

/// Componentwise bounding box of 2D float points (C++ `Eigen::AlignedBox2f`
/// initialized with one point).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundingBox2f {
    pub(crate) min: Vector2<f32>,
    pub(crate) max: Vector2<f32>,
}

impl BoundingBox2f {
    #[inline]
    pub(crate) fn new(point: Vector2<f32>) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    #[inline]
    pub(crate) fn extend(&mut self, point: &Vector2<f32>) {
        // Eigen uses `cwiseMin`/`cwiseMax`, i.e. `std::min`/`std::max`.
        self.min.x = if point.x < self.min.x {
            point.x
        } else {
            self.min.x
        };
        self.min.y = if point.y < self.min.y {
            point.y
        } else {
            self.min.y
        };
        self.max.x = if self.max.x < point.x {
            point.x
        } else {
            self.max.x
        };
        self.max.y = if self.max.y < point.y {
            point.y
        } else {
            self.max.y
        };
    }

    /// Grows `grid` so that the padded box is contained.
    pub(crate) fn grow(&self, grid: &mut dyn Grid2DTrait) {
        let padding = Vector2::new(PADDING, PADDING);
        grid.grow_limits(&(self.min - padding));
        grid.grow_limits(&(self.max + padding));
    }
}

/// Returns `limits` with each cell subdivided `SUBPIXEL_SCALE` times along
/// both axes (same `max`).
#[inline]
pub(crate) fn superscaled_limits(limits: &MapLimits) -> MapLimits {
    let superscaled_resolution = limits.resolution() / f64::from(SUBPIXEL_SCALE);
    MapLimits::new(
        superscaled_resolution,
        *limits.max(),
        CellLimits::new(
            limits.cell_limits().num_x_cells * SUBPIXEL_SCALE,
            limits.cell_limits().num_y_cells * SUBPIXEL_SCALE,
        ),
    )
}

fn grow_as_needed(range_data: &RangeData, probability_grid: &mut ProbabilityGrid) {
    let mut bounding_box = BoundingBox2f::new(range_data.origin.xy());
    for hit in range_data.returns.iter() {
        bounding_box.extend(&hit.position.xy());
    }
    for miss in range_data.misses.iter() {
        bounding_box.extend(&miss.position.xy());
    }
    bounding_box.grow(probability_grid);
}

fn cast_rays(
    range_data: &RangeData,
    hit_table: &[u16],
    miss_table: &[u16],
    insert_free_space: bool,
    probability_grid: &mut ProbabilityGrid,
) {
    grow_as_needed(range_data, probability_grid);

    let superscaled_limits = superscaled_limits(probability_grid.limits());
    let begin = superscaled_limits.get_cell_index(&range_data.origin.xy());
    // Compute and add the end points.
    let mut ends: Vec<Array2i> = Vec::with_capacity(range_data.returns.size());
    for hit in range_data.returns.iter() {
        let end = superscaled_limits.get_cell_index(&hit.position.xy());
        ends.push(end);
        // Integer division truncates toward zero, like Eigen's `Array2i / int`.
        probability_grid.apply_lookup_table(&end.map(|c| c / SUBPIXEL_SCALE), hit_table);
    }

    if !insert_free_space {
        return;
    }

    // Now add the misses.
    // RUST PORT NOTE: `ray_to_pixel_mask` returns a fresh `Vec` per ray, like
    // the C++ `RayToPixelMask` returns a `std::vector`.
    for end in &ends {
        for cell_index in ray_to_pixel_mask(&begin, end, SUBPIXEL_SCALE) {
            probability_grid.apply_lookup_table(&cell_index, miss_table);
        }
    }

    // Finally, compute and add empty rays based on misses in the range data.
    for missing_echo in range_data.misses.iter() {
        let end = superscaled_limits.get_cell_index(&missing_echo.position.xy());
        for cell_index in ray_to_pixel_mask(&begin, &end, SUBPIXEL_SCALE) {
            probability_grid.apply_lookup_table(&cell_index, miss_table);
        }
    }
}

/// Inserts range data into a [`ProbabilityGrid`] by casting exact subpixel
/// rays: hit cells get the hit odds update, the cells traversed by rays to
/// returns and misses get the miss odds update (hits have priority).
#[derive(Debug, Clone)]
pub struct ProbabilityGridRangeDataInserter2D {
    options: proto::ProbabilityGridRangeDataInserterOptions2D,
    hit_table: Vec<u16>,
    miss_table: Vec<u16>,
}

impl ProbabilityGridRangeDataInserter2D {
    /// Creates the inserter and precomputes the hit and miss lookup tables.
    pub fn new(options: &proto::ProbabilityGridRangeDataInserterOptions2D) -> Self {
        Self {
            options: *options,
            hit_table: compute_lookup_table_to_apply_correspondence_cost_odds(odds(
                options.hit_probability as f32,
            )),
            miss_table: compute_lookup_table_to_apply_correspondence_cost_odds(odds(
                options.miss_probability as f32,
            )),
        }
    }

    /// The options this inserter was created with.
    pub fn options(&self) -> &proto::ProbabilityGridRangeDataInserterOptions2D {
        &self.options
    }

    /// Inserts `range_data` into `probability_grid` (the concrete-typed
    /// version of [`RangeDataInserterInterface::insert`]).
    pub fn insert_into_probability_grid(
        &self,
        range_data: &RangeData,
        probability_grid: &mut ProbabilityGrid,
    ) {
        // By not finishing the update after hits are inserted, we give hits
        // priority (i.e. no hits will be ignored because of a miss in the same
        // cell).
        cast_rays(
            range_data,
            &self.hit_table,
            &self.miss_table,
            self.options.insert_free_space,
            probability_grid,
        );
        probability_grid.finish_update();
    }
}

impl RangeDataInserterInterface for ProbabilityGridRangeDataInserter2D {
    /// Inserts `range_data` into `grid`, which must be a [`ProbabilityGrid`]
    /// (panics otherwise; C++ uses an unchecked `static_cast`).
    fn insert(&self, range_data: &RangeData, grid: &mut dyn GridInterface) {
        let probability_grid = grid
            .downcast_mut::<ProbabilityGrid>()
            .expect("ProbabilityGridRangeDataInserter2D requires a ProbabilityGrid");
        self.insert_into_probability_grid(range_data, probability_grid);
    }
}

#[cfg(test)]
mod tests {
    mod range_data_inserter_test_2d {
        use std::sync::Arc;

        use nalgebra::{Vector2, Vector3};

        use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
        use crate::mapping::probability_values::{MAX_PROBABILITY, MIN_PROBABILITY};
        use crate::mapping::proto;
        use crate::mapping::range_data_inserter_interface::RangeDataInserterInterface;
        use crate::mapping::two_d::grid_2d::Grid2DTrait;
        use crate::mapping::two_d::map_limits::MapLimits;
        use crate::mapping::two_d::probability_grid::ProbabilityGrid;
        use crate::mapping::two_d::probability_grid_range_data_inserter_2d::*;
        use crate::mapping::two_d::xy_index::{Array2i, CellLimits};
        use crate::mapping::value_conversion_tables::ValueConversionTables;
        use crate::sensor::range_data::RangeData;
        use crate::sensor::rangefinder_point::RangefinderPoint;

        struct Fixture {
            probability_grid: ProbabilityGrid,
            range_data_inserter: ProbabilityGridRangeDataInserter2D,
            options: proto::ProbabilityGridRangeDataInserterOptions2D,
        }

        impl Fixture {
            fn new() -> Self {
                let conversion_tables = Arc::new(ValueConversionTables::new());
                let probability_grid = ProbabilityGrid::new(
                    MapLimits::new(1.0, Vector2::new(1.0, 5.0), CellLimits::new(5, 5)),
                    &conversion_tables,
                );
                let parameter_dictionary = make_dictionary(
                    "return { \
                     insert_free_space = true, \
                     hit_probability = 0.7, \
                     miss_probability = 0.4, \
                     }",
                );
                let options =
                    create_probability_grid_range_data_inserter_options_2d(&parameter_dictionary);
                let range_data_inserter = ProbabilityGridRangeDataInserter2D::new(&options);
                Self {
                    probability_grid,
                    range_data_inserter,
                    options,
                }
            }

            fn insert_point_cloud(&mut self) {
                let mut range_data = RangeData::default();
                for (x, y) in [(-3.5, 0.5), (-2.5, 1.5), (-1.5, 2.5), (-0.5, 3.5)] {
                    range_data.returns.push_back(RangefinderPoint {
                        position: Vector3::new(x, y, 0.0),
                    });
                }
                range_data.origin.x = -0.5;
                range_data.origin.y = 0.5;
                self.range_data_inserter
                    .insert(&range_data, &mut self.probability_grid);
                self.probability_grid.finish_update();
            }
        }

        #[derive(Clone, Copy)]
        enum State {
            Unknown,
            Miss,
            Hit,
        }

        #[test]
        fn insert_point_cloud() {
            let mut f = Fixture::new();
            f.insert_point_cloud();

            assert!((1.0 - f.probability_grid.limits().max().x).abs() <= 1e-9);
            assert!((5.0 - f.probability_grid.limits().max().y).abs() <= 1e-9);

            let cell_limits = *f.probability_grid.limits().cell_limits();
            assert_eq!(5, cell_limits.num_x_cells);
            assert_eq!(5, cell_limits.num_y_cells);

            use State::{Hit as H, Miss as M, Unknown as U};
            let expected_states: [[State; 5]; 5] = [
                [U, U, U, U, U],
                [U, H, M, M, M],
                [U, U, H, M, M],
                [U, U, U, H, M],
                [U, U, U, U, H],
            ];
            for row in 0..5 {
                for column in 0..5 {
                    let cell_index = Array2i::new(row, column);
                    assert!(f.probability_grid.limits().contains(&cell_index));
                    match expected_states[column as usize][row as usize] {
                        State::Unknown => assert!(!f.probability_grid.is_known(&cell_index)),
                        State::Miss => assert!(
                            (f.options.miss_probability
                                - f64::from(f.probability_grid.get_probability(&cell_index)))
                            .abs()
                                <= 1e-4
                        ),
                        State::Hit => assert!(
                            (f.options.hit_probability
                                - f64::from(f.probability_grid.get_probability(&cell_index)))
                            .abs()
                                <= 1e-4
                        ),
                    }
                }
            }
        }

        #[test]
        fn probability_progression() {
            let mut f = Fixture::new();
            f.insert_point_cloud();
            let probability_at = |f: &Fixture, x: f32, y: f32| {
                f.probability_grid.get_probability(
                    &f.probability_grid
                        .limits()
                        .get_cell_index(&Vector2::new(x, y)),
                )
            };
            assert!(
                (f.options.hit_probability - f64::from(probability_at(&f, -3.5, 0.5))).abs()
                    <= 1e-4
            );
            assert!(
                (f.options.miss_probability - f64::from(probability_at(&f, -2.5, 0.5))).abs()
                    <= 1e-4
            );

            for _ in 0..1000 {
                f.insert_point_cloud();
            }
            assert!((MAX_PROBABILITY - probability_at(&f, -3.5, 0.5)).abs() <= 1e-3);
            assert!((MIN_PROBABILITY - probability_at(&f, -2.5, 0.5)).abs() <= 1e-3);
        }

        // Additional tests (not in C++).

        #[test]
        fn options_default_insert_free_space() {
            let parameter_dictionary =
                make_dictionary("return { hit_probability = 0.55, miss_probability = 0.49 }");
            let options =
                create_probability_grid_range_data_inserter_options_2d(&parameter_dictionary);
            assert!(options.insert_free_space);
            assert_eq!(options.hit_probability, 0.55);
            assert_eq!(options.miss_probability, 0.49);
        }

        #[test]
        #[should_panic(expected = "hit_probability")]
        fn options_reject_low_hit_probability() {
            let parameter_dictionary =
                make_dictionary("return { hit_probability = 0.5, miss_probability = 0.49 }");
            let _ = create_probability_grid_range_data_inserter_options_2d(&parameter_dictionary);
        }

        #[test]
        fn insert_without_free_space_only_marks_hits() {
            let mut f = Fixture::new();
            f.options.insert_free_space = false;
            f.range_data_inserter = ProbabilityGridRangeDataInserter2D::new(&f.options);
            f.insert_point_cloud();
            let mut num_known = 0;
            for row in 0..5 {
                for column in 0..5 {
                    let cell_index = Array2i::new(row, column);
                    if f.probability_grid.is_known(&cell_index) {
                        num_known += 1;
                        assert!(
                            (0.7 - f.probability_grid.get_probability(&cell_index)).abs() <= 1e-4
                        );
                    }
                }
            }
            assert_eq!(num_known, 4);
        }

        #[test]
        fn misses_are_inserted_and_grid_grows() {
            let conversion_tables = Arc::new(ValueConversionTables::new());
            let mut grid = ProbabilityGrid::new(
                MapLimits::new(0.1, Vector2::new(0.5, 0.5), CellLimits::new(10, 10)),
                &conversion_tables,
            );
            let options = proto::ProbabilityGridRangeDataInserterOptions2D {
                hit_probability: 0.7,
                miss_probability: 0.4,
                insert_free_space: true,
            };
            let inserter = ProbabilityGridRangeDataInserter2D::new(&options);
            let mut range_data = RangeData::default();
            range_data.misses.push_back(RangefinderPoint {
                position: Vector3::new(3.05, 0.05, 0.0),
            });
            inserter.insert(&range_data, &mut grid);
            // The grid grew to contain the miss.
            let end = grid.limits().get_cell_index(&Vector2::new(3.05, 0.05));
            assert!(grid.limits().contains(&end));
            assert!(grid.grid_2d().update_indices().is_empty());
            // All cells along the ray, including the end cell, are misses.
            for i in 0..=30 {
                let x = 0.05 + 0.1 * i as f32;
                let cell = grid.limits().get_cell_index(&Vector2::new(x, 0.05));
                assert!((0.4 - grid.get_probability(&cell)).abs() <= 1e-4, "i = {i}");
            }
        }
    }
}
