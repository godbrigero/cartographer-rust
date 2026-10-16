//! Port of `cartographer/mapping/range_data_inserter_interface` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::grid_interface::GridInterface;
use crate::mapping::internal::two_d::tsdf_range_data_inserter_2d::create_tsdf_range_data_inserter_options_2d;
use crate::mapping::proto;
use crate::mapping::two_d::probability_grid_range_data_inserter_2d::create_probability_grid_range_data_inserter_options_2d;
use crate::sensor::range_data::RangeData;

/// Inserts range data into a grid (C++ `RangeDataInserterInterface`).
pub trait RangeDataInserterInterface: Send + Sync {
    /// Inserts 'range_data' into 'grid'.
    fn insert(&self, range_data: &RangeData, grid: &mut dyn GridInterface);
}

/// Reads `RangeDataInserterOptions` from `parameter_dictionary`: the
/// `range_data_inserter_type` (panics on unknown names) and always both the
/// `probability_grid_range_data_inserter` and `tsdf_range_data_inserter`
/// sub-dictionaries.
pub fn create_range_data_inserter_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::RangeDataInserterOptions {
    let range_data_inserter_type_string =
        parameter_dictionary.get_string("range_data_inserter_type");
    let range_data_inserter_type =
        proto::range_data_inserter_options::RangeDataInserterType::from_str_name(
            &range_data_inserter_type_string,
        )
        .unwrap_or_else(|| {
            panic!(
                "Unknown RangeDataInserterOptions_RangeDataInserterType kind: \
                 {range_data_inserter_type_string}"
            )
        });
    let mut options = proto::RangeDataInserterOptions::default();
    options.set_range_data_inserter_type(range_data_inserter_type);
    options.probability_grid_range_data_inserter_options_2d =
        Some(create_probability_grid_range_data_inserter_options_2d(
            &parameter_dictionary.get_dictionary("probability_grid_range_data_inserter"),
        ));
    options.tsdf_range_data_inserter_options_2d = Some(create_tsdf_range_data_inserter_options_2d(
        &parameter_dictionary.get_dictionary("tsdf_range_data_inserter"),
    ));
    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    const INSERTER_OPTIONS: &str = "probability_grid_range_data_inserter = {
          insert_free_space = true,
          hit_probability = 0.55,
          miss_probability = 0.49,
        },
        tsdf_range_data_inserter = {
          truncation_distance = 0.3,
          maximum_weight = 10.,
          update_free_space = false,
          normal_estimation_options = {
            num_normal_samples = 4,
            sample_radius = 0.5,
          },
          project_sdf_distance_to_scan_normal = true,
          update_weight_range_exponent = 0,
          update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0.5,
          update_weight_distance_cell_to_hit_kernel_bandwidth = 0.5,
        },";

    #[test]
    fn create_options_reads_both_inserters() {
        let dict = make_dictionary(&format!(
            "return {{ range_data_inserter_type = \"TSDF_INSERTER_2D\", {INSERTER_OPTIONS} }}"
        ));
        let options = create_range_data_inserter_options(&dict);
        assert_eq!(
            options.range_data_inserter_type(),
            proto::range_data_inserter_options::RangeDataInserterType::TsdfInserter2d
        );
        let probability = options
            .probability_grid_range_data_inserter_options_2d
            .unwrap();
        assert_eq!(probability.hit_probability, 0.55);
        assert_eq!(probability.miss_probability, 0.49);
        assert!(probability.insert_free_space);
        let tsdf = options.tsdf_range_data_inserter_options_2d.unwrap();
        assert_eq!(tsdf.truncation_distance, 0.3);
        assert_eq!(tsdf.maximum_weight, 10.0);
        assert!(!tsdf.update_free_space);
        assert!(tsdf.project_sdf_distance_to_scan_normal);
        let normal_options = tsdf.normal_estimation_options.unwrap();
        assert_eq!(normal_options.num_normal_samples, 4);
        assert_eq!(normal_options.sample_radius, 0.5_f32);
        assert_eq!(
            tsdf.update_weight_angle_scan_normal_to_ray_kernel_bandwidth,
            0.5
        );
        assert_eq!(
            tsdf.update_weight_distance_cell_to_hit_kernel_bandwidth,
            0.5
        );
    }

    #[test]
    #[should_panic(expected = "Unknown RangeDataInserterOptions_RangeDataInserterType kind")]
    fn create_options_rejects_unknown_type() {
        let dict = make_dictionary(&format!(
            "return {{ range_data_inserter_type = \"FOO\", {INSERTER_OPTIONS} }}"
        ));
        let _ = create_range_data_inserter_options(&dict);
    }
}
