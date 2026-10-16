//! Port of `cartographer/common/internal/ceres_solver_options` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::proto;

/// Reads `CeresSolverOptions` from `parameter_dictionary`.
pub fn create_ceres_solver_options_proto(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::CeresSolverOptions {
    let proto = proto::CeresSolverOptions {
        use_nonmonotonic_steps: parameter_dictionary.get_bool("use_nonmonotonic_steps"),
        max_num_iterations: parameter_dictionary.get_non_negative_int("max_num_iterations"),
        num_threads: parameter_dictionary.get_non_negative_int("num_threads"),
    };
    assert!(
        proto.max_num_iterations > 0,
        "Check failed: proto.max_num_iterations() > 0 ({} vs. 0)",
        proto.max_num_iterations
    );
    assert!(
        proto.num_threads > 0,
        "Check failed: proto.num_threads() > 0 ({} vs. 0)",
        proto.num_threads
    );
    proto
}

/// Builds Ceres solver options from `proto`. All other fields keep the Ceres
/// defaults, as in C++.
pub fn create_ceres_solver_options(proto: &proto::CeresSolverOptions) -> ceres::SolverOptions {
    ceres::SolverOptions {
        use_nonmonotonic_steps: proto.use_nonmonotonic_steps,
        max_num_iterations: proto.max_num_iterations,
        num_threads: proto.num_threads,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    #[test]
    fn create_proto() {
        let dict = make_dictionary(
            "return { use_nonmonotonic_steps = true, max_num_iterations = 10, num_threads = 2 }",
        );
        let proto = create_ceres_solver_options_proto(&dict);
        assert!(proto.use_nonmonotonic_steps);
        assert_eq!(proto.max_num_iterations, 10);
        assert_eq!(proto.num_threads, 2);
        let options = create_ceres_solver_options(&proto);
        assert!(options.use_nonmonotonic_steps);
        assert_eq!(options.max_num_iterations, 10);
        assert_eq!(options.num_threads, 2);
    }

    #[test]
    #[should_panic(expected = "num_threads() > 0")]
    fn zero_threads_panics() {
        let dict = make_dictionary(
            "return { use_nonmonotonic_steps = true, max_num_iterations = 10, num_threads = 0 }",
        );
        create_ceres_solver_options_proto(&dict);
    }
}
