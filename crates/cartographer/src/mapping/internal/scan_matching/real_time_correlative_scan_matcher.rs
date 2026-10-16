//! Port of `cartographer/mapping/internal/scan_matching/real_time_correlative_scan_matcher` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::scan_matching::proto;

/// Reads `RealTimeCorrelativeScanMatcherOptions` from
/// `parameter_dictionary` (shared by 2D and 3D). Panics if a delta cost
/// weight is negative.
pub fn create_real_time_correlative_scan_matcher_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::RealTimeCorrelativeScanMatcherOptions {
    let options = proto::RealTimeCorrelativeScanMatcherOptions {
        linear_search_window: parameter_dictionary.get_double("linear_search_window"),
        angular_search_window: parameter_dictionary.get_double("angular_search_window"),
        translation_delta_cost_weight: parameter_dictionary
            .get_double("translation_delta_cost_weight"),
        rotation_delta_cost_weight: parameter_dictionary.get_double("rotation_delta_cost_weight"),
    };
    assert!(
        options.translation_delta_cost_weight >= 0.0,
        "Check failed: options.translation_delta_cost_weight() >= 0. ({} vs. 0)",
        options.translation_delta_cost_weight
    );
    assert!(
        options.rotation_delta_cost_weight >= 0.0,
        "Check failed: options.rotation_delta_cost_weight() >= 0. ({} vs. 0)",
        options.rotation_delta_cost_weight
    );
    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    #[test]
    fn create_options() {
        let dict = make_dictionary(
            "return { linear_search_window = 0.6, angular_search_window = 0.16, \
             translation_delta_cost_weight = 0.5, rotation_delta_cost_weight = 0.25 }",
        );
        let options = create_real_time_correlative_scan_matcher_options(&dict);
        assert_eq!(options.linear_search_window, 0.6);
        assert_eq!(options.angular_search_window, 0.16);
        assert_eq!(options.translation_delta_cost_weight, 0.5);
        assert_eq!(options.rotation_delta_cost_weight, 0.25);
    }

    #[test]
    #[should_panic(expected = "rotation_delta_cost_weight() >= 0.")]
    fn negative_weight_panics() {
        let dict = make_dictionary(
            "return { linear_search_window = 0.6, angular_search_window = 0.16, \
             translation_delta_cost_weight = 0., rotation_delta_cost_weight = -1. }",
        );
        create_real_time_correlative_scan_matcher_options(&dict);
    }
}
