//! Port of `cartographer/mapping/pose_extrapolator_interface` (.h, .cc).

use nalgebra::{Quaternion, Vector3};

use crate::common::internal::ceres_solver_options::create_ceres_solver_options_proto;
use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::{Time, from_seconds};
use crate::mapping::pose_extrapolator::PoseExtrapolator;
use crate::mapping::proto;
use crate::sensor::imu_data::ImuData;
use crate::sensor::odometry_data::OdometryData;
use crate::transform::rigid_transform::{Rigid3d, Rigid3f};
use crate::transform::timestamped_transform::TimestampedTransform;

fn create_constant_velocity_pose_extrapolator_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::ConstantVelocityPoseExtrapolatorOptions {
    proto::ConstantVelocityPoseExtrapolatorOptions {
        pose_queue_duration: parameter_dictionary.get_double("pose_queue_duration"),
        imu_gravity_time_constant: parameter_dictionary.get_double("imu_gravity_time_constant"),
    }
}

fn create_imu_based_pose_extrapolator_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::ImuBasedPoseExtrapolatorOptions {
    // Keys are read in the C++ order.
    let pose_queue_duration = parameter_dictionary.get_double("pose_queue_duration");
    let gravity_constant = parameter_dictionary.get_double("gravity_constant");
    let pose_translation_weight = parameter_dictionary.get_double("pose_translation_weight");
    let pose_rotation_weight = parameter_dictionary.get_double("pose_rotation_weight");
    let imu_acceleration_weight = parameter_dictionary.get_double("imu_acceleration_weight");
    let imu_rotation_weight = parameter_dictionary.get_double("imu_rotation_weight");
    let odometry_rotation_weight = parameter_dictionary.get_double("odometry_rotation_weight");
    let odometry_translation_weight =
        parameter_dictionary.get_double("odometry_translation_weight");
    let solver_options =
        create_ceres_solver_options_proto(&parameter_dictionary.get_dictionary("solver_options"));
    proto::ImuBasedPoseExtrapolatorOptions {
        pose_queue_duration,
        gravity_constant,
        pose_translation_weight,
        pose_rotation_weight,
        imu_acceleration_weight,
        imu_rotation_weight,
        solver_options: Some(solver_options),
        odometry_translation_weight,
        odometry_rotation_weight,
    }
}

/// Reads `PoseExtrapolatorOptions` from `parameter_dictionary`
/// (C++ `CreatePoseExtrapolatorOptions`).
pub fn create_pose_extrapolator_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::PoseExtrapolatorOptions {
    let use_imu_based = parameter_dictionary.get_bool("use_imu_based");
    let constant_velocity = create_constant_velocity_pose_extrapolator_options(
        &parameter_dictionary.get_dictionary("constant_velocity"),
    );
    let imu_based = create_imu_based_pose_extrapolator_options(
        &parameter_dictionary.get_dictionary("imu_based"),
    );
    proto::PoseExtrapolatorOptions {
        use_imu_based,
        constant_velocity: Some(constant_velocity),
        imu_based: Some(imu_based),
    }
}

/// Result of [`PoseExtrapolatorInterface::extrapolate_poses_with_gravity`]
/// (C++ `PoseExtrapolatorInterface::ExtrapolationResult`).
#[derive(Clone, Debug)]
pub struct ExtrapolationResult {
    /// The poses for the requested times at index 0 to N-1.
    pub previous_poses: Vec<Rigid3f>,
    /// The pose for the requested time at index N.
    pub current_pose: Rigid3d,
    /// The current linear velocity estimate.
    pub current_velocity: Vector3<f64>,
    /// Rotation from the tracking frame into a gravity aligned frame at the
    /// last requested time.
    pub gravity_from_tracking: Quaternion<f64>,
}

/// Extrapolates poses from past poses, IMU and odometry data
/// (C++ abstract class `PoseExtrapolatorInterface`).
pub trait PoseExtrapolatorInterface: Send {
    /// Returns the time of the last added pose or `Time::min()` if no pose was
    /// added yet.
    fn get_last_pose_time(&self) -> Time;

    /// Returns the time of the last extrapolation or `Time::min()` if there
    /// was none yet.
    fn get_last_extrapolated_time(&self) -> Time;

    /// Adds a pose estimate at `time`.
    fn add_pose(&mut self, time: Time, pose: &Rigid3d);

    /// Adds an IMU measurement.
    fn add_imu_data(&mut self, imu_data: &ImuData);

    /// Adds an odometry measurement.
    fn add_odometry_data(&mut self, odometry_data: &OdometryData);

    /// Extrapolates the pose to `time`.
    fn extrapolate_pose(&mut self, time: Time) -> Rigid3d;

    /// Extrapolates the poses for all `times` (which must be non-empty) and
    /// estimates gravity at the last one.
    fn extrapolate_poses_with_gravity(&mut self, times: &[Time]) -> ExtrapolationResult;

    /// Returns the current gravity alignment estimate as a rotation from
    /// the tracking frame into a gravity aligned frame.
    fn estimate_gravity_orientation(&mut self, time: Time) -> Quaternion<f64>;
}

impl dyn PoseExtrapolatorInterface {
    /// C++ static `PoseExtrapolatorInterface::CreateWithImuData`; see
    /// [`create_with_imu_data`].
    pub fn create_with_imu_data(
        options: &proto::PoseExtrapolatorOptions,
        imu_data: &[ImuData],
        initial_poses: &[TimestampedTransform],
    ) -> Box<dyn PoseExtrapolatorInterface> {
        create_with_imu_data(options, imu_data, initial_poses)
    }
}

/// Creates the extrapolator selected by `options`, initialized from
/// `imu_data` (which must be non-empty) and, for the IMU-based extrapolator,
/// `initial_poses`.
pub fn create_with_imu_data(
    options: &proto::PoseExtrapolatorOptions,
    imu_data: &[ImuData],
    initial_poses: &[TimestampedTransform],
) -> Box<dyn PoseExtrapolatorInterface> {
    assert!(!imu_data.is_empty(), "Check failed: !imu_data.empty()");
    if options.use_imu_based {
        // PENDING(imu_based_pose_extrapolator): call
        // `ImuBasedPoseExtrapolator::initialize_with_imu(
        //     &options.imu_based.unwrap_or_default(), imu_data, initial_poses)`
        // once `mapping/internal/imu_based_pose_extrapolator` is ported.
        let _ = initial_poses;
        panic!("ImuBasedPoseExtrapolator not ported yet (pose_extrapolator.use_imu_based = true)");
    } else {
        let constant_velocity = options.constant_velocity.unwrap_or_default();
        Box::new(PoseExtrapolator::initialize_with_imu(
            from_seconds(constant_velocity.pose_queue_duration),
            constant_velocity.imu_gravity_time_constant,
            imu_data.last().expect("imu_data is not empty"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::common::time::from_universal;
    use crate::transform::rigid_transform::quaternion_rotate;

    const OPTIONS_LUA: &str = r#"
        return {
          use_imu_based = false,
          constant_velocity = {
            imu_gravity_time_constant = 10.,
            pose_queue_duration = 0.001,
          },
          imu_based = {
            pose_queue_duration = 5.,
            gravity_constant = 9.806,
            pose_translation_weight = 1.,
            pose_rotation_weight = 2.,
            imu_acceleration_weight = 3.,
            imu_rotation_weight = 4.,
            odometry_translation_weight = 5.,
            odometry_rotation_weight = 6.,
            solver_options = {
              use_nonmonotonic_steps = false;
              max_num_iterations = 10;
              num_threads = 1;
            },
          },
        }"#;

    #[test]
    fn create_pose_extrapolator_options_reads_all_keys() {
        let options = create_pose_extrapolator_options(&make_dictionary(OPTIONS_LUA));
        assert!(!options.use_imu_based);
        let cv = options.constant_velocity.unwrap();
        assert_eq!(cv.imu_gravity_time_constant, 10.);
        assert_eq!(cv.pose_queue_duration, 0.001);
        let ib = options.imu_based.unwrap();
        assert_eq!(ib.pose_queue_duration, 5.);
        assert_eq!(ib.gravity_constant, 9.806);
        assert_eq!(ib.pose_translation_weight, 1.);
        assert_eq!(ib.pose_rotation_weight, 2.);
        assert_eq!(ib.imu_acceleration_weight, 3.);
        assert_eq!(ib.imu_rotation_weight, 4.);
        assert_eq!(ib.odometry_translation_weight, 5.);
        assert_eq!(ib.odometry_rotation_weight, 6.);
        assert_eq!(ib.solver_options.unwrap().max_num_iterations, 10);
    }

    #[test]
    fn create_with_imu_data_constant_velocity() {
        let options = create_pose_extrapolator_options(&make_dictionary(OPTIONS_LUA));
        let time = from_universal(1000);
        let imu = [
            ImuData {
                time: from_universal(500),
                linear_acceleration: Vector3::new(0., 0., 9.8),
                angular_velocity: Vector3::zeros(),
            },
            ImuData {
                time,
                linear_acceleration: Vector3::new(1., 0., 9.8),
                angular_velocity: Vector3::zeros(),
            },
        ];
        let mut extrapolator =
            <dyn PoseExtrapolatorInterface>::create_with_imu_data(&options, &imu, &[]);
        // Initialized from the last IMU sample.
        assert_eq!(extrapolator.get_last_pose_time(), time);
        let gravity = extrapolator.estimate_gravity_orientation(time);
        let up = quaternion_rotate(&gravity, &Vector3::new(1., 0., 9.8).normalize());
        assert!((up - Vector3::z()).norm() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "!imu_data.empty()")]
    fn create_with_imu_data_requires_imu() {
        let options = create_pose_extrapolator_options(&make_dictionary(OPTIONS_LUA));
        create_with_imu_data(&options, &[], &[]);
    }

    #[test]
    #[should_panic(expected = "ImuBasedPoseExtrapolator not ported yet")]
    fn create_with_imu_data_imu_based_is_pending() {
        let mut options = create_pose_extrapolator_options(&make_dictionary(OPTIONS_LUA));
        options.use_imu_based = true;
        create_with_imu_data(&options, &[ImuData::default()], &[]);
    }
}
