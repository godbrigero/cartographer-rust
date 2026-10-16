//! Port of `cartographer/mapping/internal/motion_filter` (.h, .cc).

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::common::time::{Time, from_seconds};
use crate::mapping::proto;
use crate::transform::rigid_transform::Rigid3d;
use crate::transform::transform::get_angle;

/// Reads `MotionFilterOptions` from `parameter_dictionary`
/// (C++ `CreateMotionFilterOptions`).
pub fn create_motion_filter_options(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::MotionFilterOptions {
    proto::MotionFilterOptions {
        max_time_seconds: parameter_dictionary.get_double("max_time_seconds"),
        max_distance_meters: parameter_dictionary.get_double("max_distance_meters"),
        max_angle_radians: parameter_dictionary.get_double("max_angle_radians"),
    }
}

/// Takes poses as input and filters them to get fewer poses.
#[derive(Clone, Debug)]
pub struct MotionFilter {
    options: proto::MotionFilterOptions,
    num_total: i32,
    num_different: i32,
    last_time: Time,
    last_pose: Rigid3d,
}

impl MotionFilter {
    /// Creates a filter with the given thresholds.
    pub fn new(options: &proto::MotionFilterOptions) -> Self {
        MotionFilter {
            options: *options,
            num_total: 0,
            num_different: 0,
            last_time: Time::default(),
            last_pose: Rigid3d::identity(),
        }
    }

    /// If the accumulated motion (linear, rotational, or time) is above the
    /// threshold, returns false. Otherwise the relative motion is accumulated
    /// and true is returned.
    pub fn is_similar(&mut self, time: Time, pose: &Rigid3d) -> bool {
        // C++ `LOG_IF_EVERY_N(INFO, num_total_ >= 500, 500)`: logs on the
        // 1st, 501st, ... call where the condition holds, i.e. when
        // `num_total_` is 500, 1000, ...
        // RUST PORT NOTE: glog's occurrence counter is a function-local static
        // shared by all filters; here it is derived per filter instance.
        if self.num_total >= 500 && self.num_total % 500 == 0 {
            log::info!(
                "Motion filter reduced the number of nodes to {}%.",
                100. * f64::from(self.num_different) / f64::from(self.num_total)
            );
        }
        self.num_total += 1;
        if self.num_total > 1
            && time - self.last_time <= from_seconds(self.options.max_time_seconds)
            && (pose.translation() - self.last_pose.translation()).norm()
                <= self.options.max_distance_meters
            && get_angle(&(pose.inverse() * self.last_pose)) <= self.options.max_angle_radians
        {
            return true;
        }
        self.last_time = time;
        self.last_pose = *pose;
        self.num_different += 1;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::common::time::from_universal;
    use crate::transform::rigid_transform::AngleAxis;
    use nalgebra::Vector3;

    mod motion_filter_test {
        use super::*;

        fn options() -> proto::MotionFilterOptions {
            let parameter_dictionary = make_dictionary(
                "return {\
                 max_time_seconds = 0.5, \
                 max_distance_meters = 0.2, \
                 max_angle_radians = 2., \
                 }",
            );
            create_motion_filter_options(&parameter_dictionary)
        }

        fn seconds_since_epoch(seconds: i64) -> Time {
            from_universal(seconds * 10_000_000)
        }

        fn rotation_y(angle: f64) -> Rigid3d {
            Rigid3d::from_angle_axis(AngleAxis::new(angle, Vector3::y()))
        }

        #[test]
        fn not_initialized() {
            let mut motion_filter = MotionFilter::new(&options());
            assert!(!motion_filter.is_similar(Time::default(), &Rigid3d::identity()));
        }

        #[test]
        fn no_change() {
            let mut motion_filter = MotionFilter::new(&options());
            assert!(!motion_filter.is_similar(seconds_since_epoch(42), &Rigid3d::identity()));
            assert!(motion_filter.is_similar(seconds_since_epoch(42), &Rigid3d::identity()));
        }

        #[test]
        fn time_elapsed() {
            let mut motion_filter = MotionFilter::new(&options());
            assert!(!motion_filter.is_similar(seconds_since_epoch(42), &Rigid3d::identity()));
            assert!(!motion_filter.is_similar(seconds_since_epoch(43), &Rigid3d::identity()));
            assert!(motion_filter.is_similar(seconds_since_epoch(43), &Rigid3d::identity()));
        }

        #[test]
        fn linear_motion() {
            let mut motion_filter = MotionFilter::new(&options());
            let t = seconds_since_epoch(42);
            assert!(!motion_filter.is_similar(t, &Rigid3d::identity()));
            assert!(
                !motion_filter.is_similar(t, &Rigid3d::from_translation(Vector3::new(0.3, 0., 0.)))
            );
            assert!(
                motion_filter.is_similar(t, &Rigid3d::from_translation(Vector3::new(0.45, 0., 0.)))
            );
            assert!(
                !motion_filter.is_similar(t, &Rigid3d::from_translation(Vector3::new(0.6, 0., 0.)))
            );
            assert!(
                motion_filter
                    .is_similar(t, &Rigid3d::from_translation(Vector3::new(0.6, 0.15, 0.)))
            );
        }

        #[test]
        fn rotational_motion() {
            let mut motion_filter = MotionFilter::new(&options());
            let t = seconds_since_epoch(42);
            assert!(!motion_filter.is_similar(t, &Rigid3d::identity()));
            assert!(motion_filter.is_similar(t, &rotation_y(1.9)));
            assert!(!motion_filter.is_similar(t, &rotation_y(2.1)));
            assert!(motion_filter.is_similar(t, &rotation_y(4.)));
            assert!(!motion_filter.is_similar(t, &rotation_y(5.9)));
            assert!(motion_filter.is_similar(t, &Rigid3d::identity()));
        }
    }
}
