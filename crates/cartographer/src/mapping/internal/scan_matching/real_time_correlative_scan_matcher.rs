//! Port of `cartographer/mapping/internal/scan_matching/real_time_correlative_scan_matcher` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::scan_matching::proto;

/// Reads `RealTimeCorrelativeScanMatcherOptions` from `parameter_dictionary`
/// (C++ `CreateRealTimeCorrelativeScanMatcherOptions`).
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
        options.translation_delta_cost_weight >= 0.,
        "Check failed: options.translation_delta_cost_weight() >= 0. ({} vs. 0)",
        options.translation_delta_cost_weight
    );
    assert!(
        options.rotation_delta_cost_weight >= 0.,
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
    fn reads_all_keys() {
        let options = create_real_time_correlative_scan_matcher_options(&make_dictionary(
            r#"return {
                linear_search_window = 0.2,
                angular_search_window = 0.5,
                translation_delta_cost_weight = 1e-1,
                rotation_delta_cost_weight = 2.,
            }"#,
        ));
        assert_eq!(options.linear_search_window, 0.2);
        assert_eq!(options.angular_search_window, 0.5);
        assert_eq!(options.translation_delta_cost_weight, 1e-1);
        assert_eq!(options.rotation_delta_cost_weight, 2.);
    }

    #[test]
    #[should_panic(expected = "translation_delta_cost_weight() >= 0.")]
    fn negative_weight_panics() {
        create_real_time_correlative_scan_matcher_options(&make_dictionary(
            r#"return {
                linear_search_window = 0.2,
                angular_search_window = 0.5,
                translation_delta_cost_weight = -1.,
                rotation_delta_cost_weight = 2.,
            }"#,
        ));
    }
}
