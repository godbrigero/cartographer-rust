//! Port of `cartographer/mapping/internal/constraints/constraint_builder` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::constraints::proto;
use crate::mapping::internal::three_d::scan_matching::ceres_scan_matcher_3d::create_ceres_scan_matcher_options_3d;
use crate::mapping::internal::three_d::scan_matching::fast_correlative_scan_matcher_3d::create_fast_correlative_scan_matcher_options_3d;
use crate::mapping::internal::two_d::scan_matching::ceres_scan_matcher_2d::create_ceres_scan_matcher_options_2d;
use crate::mapping::internal::two_d::scan_matching::fast_correlative_scan_matcher_2d::create_fast_correlative_scan_matcher_options_2d;

/// Reads `ConstraintBuilderOptions` from `parameter_dictionary`.
pub fn create_constraint_builder_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::ConstraintBuilderOptions {
    proto::ConstraintBuilderOptions {
        sampling_ratio: parameter_dictionary.get_double("sampling_ratio"),
        max_constraint_distance: parameter_dictionary.get_double("max_constraint_distance"),
        min_score: parameter_dictionary.get_double("min_score"),
        global_localization_min_score: parameter_dictionary
            .get_double("global_localization_min_score"),
        loop_closure_translation_weight: parameter_dictionary
            .get_double("loop_closure_translation_weight"),
        loop_closure_rotation_weight: parameter_dictionary
            .get_double("loop_closure_rotation_weight"),
        log_matches: parameter_dictionary.get_bool("log_matches"),
        fast_correlative_scan_matcher_options: Some(
            create_fast_correlative_scan_matcher_options_2d(
                &parameter_dictionary.get_dictionary("fast_correlative_scan_matcher"),
            ),
        ),
        ceres_scan_matcher_options: Some(create_ceres_scan_matcher_options_2d(
            &parameter_dictionary.get_dictionary("ceres_scan_matcher"),
        )),
        fast_correlative_scan_matcher_options_3d: Some(
            create_fast_correlative_scan_matcher_options_3d(
                &parameter_dictionary.get_dictionary("fast_correlative_scan_matcher_3d"),
            ),
        ),
        ceres_scan_matcher_options_3d: Some(create_ceres_scan_matcher_options_3d(
            &parameter_dictionary.get_dictionary("ceres_scan_matcher_3d"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::internal::testing::test_helpers::resolve_lua_parameters;

    // No C++ test; checks the options of `configuration_files/pose_graph.lua`.
    #[test]
    fn creates_options_from_pose_graph_lua() {
        let parameters = resolve_lua_parameters(
            r#"
            include "pose_graph.lua"
            return POSE_GRAPH.constraint_builder"#,
        );
        let options = create_constraint_builder_options(&parameters);
        assert_eq!(options.sampling_ratio, 0.3);
        assert_eq!(options.max_constraint_distance, 15.);
        assert_eq!(options.min_score, 0.55);
        assert_eq!(options.global_localization_min_score, 0.6);
        assert_eq!(options.loop_closure_translation_weight, 1.1e4);
        assert_eq!(options.loop_closure_rotation_weight, 1e5);
        assert!(options.log_matches);
        let fast_2d = options.fast_correlative_scan_matcher_options.unwrap();
        assert_eq!(fast_2d.linear_search_window, 7.);
        assert_eq!(fast_2d.branch_and_bound_depth, 7);
        let fast_3d = options.fast_correlative_scan_matcher_options_3d.unwrap();
        assert_eq!(fast_3d.branch_and_bound_depth, 8);
        assert_eq!(fast_3d.full_resolution_depth, 3);
        assert_eq!(fast_3d.min_rotational_score, 0.77);
        assert_eq!(fast_3d.min_low_resolution_score, 0.55);
        let ceres_3d = options.ceres_scan_matcher_options_3d.unwrap();
        assert_eq!(ceres_3d.occupied_space_weight, vec![5., 30.]);
        assert!(!ceres_3d.only_optimize_yaw);
    }
}
