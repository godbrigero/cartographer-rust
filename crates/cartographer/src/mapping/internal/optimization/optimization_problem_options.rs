//! Port of `cartographer/mapping/internal/optimization/optimization_problem_options` (.h, .cc).

use crate::common::internal::ceres_solver_options::create_ceres_solver_options_proto;
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::optimization::proto;

/// Reads `OptimizationProblemOptions` from `parameter_dictionary`.
pub fn create_optimization_problem_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::OptimizationProblemOptions {
    proto::OptimizationProblemOptions {
        huber_scale: parameter_dictionary.get_double("huber_scale"),
        acceleration_weight: parameter_dictionary.get_double("acceleration_weight"),
        rotation_weight: parameter_dictionary.get_double("rotation_weight"),
        odometry_translation_weight: parameter_dictionary.get_double("odometry_translation_weight"),
        odometry_rotation_weight: parameter_dictionary.get_double("odometry_rotation_weight"),
        local_slam_pose_translation_weight: parameter_dictionary
            .get_double("local_slam_pose_translation_weight"),
        local_slam_pose_rotation_weight: parameter_dictionary
            .get_double("local_slam_pose_rotation_weight"),
        fixed_frame_pose_translation_weight: parameter_dictionary
            .get_double("fixed_frame_pose_translation_weight"),
        fixed_frame_pose_rotation_weight: parameter_dictionary
            .get_double("fixed_frame_pose_rotation_weight"),
        fixed_frame_pose_use_tolerant_loss: parameter_dictionary
            .get_bool("fixed_frame_pose_use_tolerant_loss"),
        fixed_frame_pose_tolerant_loss_param_a: parameter_dictionary
            .get_double("fixed_frame_pose_tolerant_loss_param_a"),
        fixed_frame_pose_tolerant_loss_param_b: parameter_dictionary
            .get_double("fixed_frame_pose_tolerant_loss_param_b"),
        log_solver_summary: parameter_dictionary.get_bool("log_solver_summary"),
        use_online_imu_extrinsics_in_3d: parameter_dictionary
            .get_bool("use_online_imu_extrinsics_in_3d"),
        fix_z_in_3d: parameter_dictionary.get_bool("fix_z_in_3d"),
        ceres_solver_options: Some(create_ceres_solver_options_proto(
            &parameter_dictionary.get_dictionary("ceres_solver_options"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;

    #[test]
    fn reads_all_options() {
        let dictionary = make_dictionary(
            r#"
            return {
              huber_scale = 1e1,
              acceleration_weight = 1.1e2,
              rotation_weight = 1.6e4,
              local_slam_pose_translation_weight = 1e5,
              local_slam_pose_rotation_weight = 2e5,
              odometry_translation_weight = 3e5,
              odometry_rotation_weight = 4e5,
              fixed_frame_pose_translation_weight = 1e1,
              fixed_frame_pose_rotation_weight = 1e2,
              fixed_frame_pose_use_tolerant_loss = true,
              fixed_frame_pose_tolerant_loss_param_a = 2,
              fixed_frame_pose_tolerant_loss_param_b = 3,
              log_solver_summary = false,
              use_online_imu_extrinsics_in_3d = true,
              fix_z_in_3d = true,
              ceres_solver_options = {
                use_nonmonotonic_steps = false,
                max_num_iterations = 50,
                num_threads = 7,
              },
            }"#,
        );
        let options = create_optimization_problem_options(&dictionary);
        assert_eq!(options.huber_scale, 1e1);
        assert_eq!(options.acceleration_weight, 1.1e2);
        assert_eq!(options.rotation_weight, 1.6e4);
        assert_eq!(options.local_slam_pose_translation_weight, 1e5);
        assert_eq!(options.local_slam_pose_rotation_weight, 2e5);
        assert_eq!(options.odometry_translation_weight, 3e5);
        assert_eq!(options.odometry_rotation_weight, 4e5);
        assert_eq!(options.fixed_frame_pose_translation_weight, 1e1);
        assert_eq!(options.fixed_frame_pose_rotation_weight, 1e2);
        assert!(options.fixed_frame_pose_use_tolerant_loss);
        assert_eq!(options.fixed_frame_pose_tolerant_loss_param_a, 2.);
        assert_eq!(options.fixed_frame_pose_tolerant_loss_param_b, 3.);
        assert!(!options.log_solver_summary);
        assert!(options.use_online_imu_extrinsics_in_3d);
        assert!(options.fix_z_in_3d);
        let ceres = options.ceres_solver_options.unwrap();
        assert!(!ceres.use_nonmonotonic_steps);
        assert_eq!(ceres.max_num_iterations, 50);
        assert_eq!(ceres.num_threads, 7);
    }
}
