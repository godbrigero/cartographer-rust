//! Port of `cartographer/mapping/internal/2d/normal_estimation_2d` (.h, .cc).

use nalgebra::Vector3;

use crate::common::lua_parameter_dictionary::LuaParameterDictionary;
use crate::mapping::proto;
use crate::sensor::range_data::RangeData;
use crate::sensor::rangefinder_point::RangefinderPoint;

/// Minimum length of a sample normal; shorter ones are skipped.
const MIN_NORMAL_LENGTH: f32 = 1e-6;

/// Reads `NormalEstimationOptions2D` from `parameter_dictionary`. Panics
/// unless `num_normal_samples > 0` and `sample_radius > 0`.
pub fn create_normal_estimation_options_2d(
    parameter_dictionary: &LuaParameterDictionary,
) -> proto::NormalEstimationOptions2D {
    let options = proto::NormalEstimationOptions2D {
        num_normal_samples: parameter_dictionary.get_int("num_normal_samples"),
        // The proto field is a float; C++ narrows the double the same way.
        sample_radius: parameter_dictionary.get_double("sample_radius") as f32,
    };
    assert!(
        options.num_normal_samples > 0,
        "Check failed: options.num_normal_samples() > 0 ({} vs. 0)",
        options.num_normal_samples
    );
    assert!(
        options.sample_radius > 0.0,
        "Check failed: options.sample_radius() > 0.0 ({} vs. 0)",
        options.sample_radius
    );
    options
}

#[inline]
fn normal_to_2d_angle(v: &Vector3<f32>) -> f32 {
    v.y.atan2(v.x)
}

/// Estimates the normal of an estimation point as the arithmetic mean of the
/// normals of the vectors from the estimation point to each point in the
/// sample window `[sample_window_begin, sample_window_end)`.
fn estimate_normal(
    returns: &[RangefinderPoint],
    estimation_point_index: usize,
    sample_window_begin: usize,
    sample_window_end: usize,
    sensor_origin: &Vector3<f32>,
) -> f32 {
    let estimation_point = &returns[estimation_point_index].position;
    if sample_window_end - sample_window_begin < 2 {
        return normal_to_2d_angle(&(sensor_origin - estimation_point));
    }
    let mut mean_normal = Vector3::<f32>::zeros();
    let estimation_point_to_observation = sensor_origin - estimation_point;
    for (sample_point_index, sample) in returns
        .iter()
        .enumerate()
        .take(sample_window_end)
        .skip(sample_window_begin)
    {
        if sample_point_index == estimation_point_index {
            continue;
        }
        let sample_point = &sample.position;
        let tangent = estimation_point - sample_point;
        let mut sample_normal = Vector3::new(-tangent.y, tangent.x, 0.0);
        let norm = sample_normal.norm();
        if norm < MIN_NORMAL_LENGTH {
            continue;
        }
        // Ensure sample_normal points towards 'sensor_origin'.
        if sample_normal.dot(&estimation_point_to_observation) < 0.0 {
            sample_normal = -sample_normal;
        }
        // `norm >= 1e-6 > 0`, so this matches Eigen's `normalize()`.
        sample_normal /= norm;
        mean_normal += sample_normal;
    }
    normal_to_2d_angle(&mean_normal)
}

/// Estimates the normal (as an angle in radians) for each return in
/// `range_data`. Assumes the returns are sorted with respect to the
/// orientation of the vector from `origin` to the return.
pub fn estimate_normals(
    range_data: &RangeData,
    normal_estimation_options: &proto::NormalEstimationOptions2D,
) -> Vec<f32> {
    estimate_normals_for_points(
        range_data.returns.points(),
        &range_data.origin,
        normal_estimation_options,
    )
}

/// [`estimate_normals`] on a slice of returns with the given sensor origin
/// (avoids building a `RangeData` for sorted copies).
pub fn estimate_normals_for_points(
    returns: &[RangefinderPoint],
    origin: &Vector3<f32>,
    normal_estimation_options: &proto::NormalEstimationOptions2D,
) -> Vec<f32> {
    let mut normals = Vec::with_capacity(returns.len());
    // The options are checked to be positive on creation.
    let max_num_samples = normal_estimation_options.num_normal_samples as usize;
    let sample_radius = normal_estimation_options.sample_radius;
    let max_forward_samples = (max_num_samples as f64 / 2.0).ceil() + 1.0;
    for (current_point, hit) in returns.iter().enumerate() {
        let hit = &hit.position;
        let mut sample_window_begin = current_point;
        while sample_window_begin > 0
            && current_point - sample_window_begin < max_num_samples / 2
            && (hit - returns[sample_window_begin - 1].position).norm() < sample_radius
        {
            sample_window_begin -= 1;
        }
        let mut sample_window_end = current_point;
        while sample_window_end < returns.len()
            && ((sample_window_end - current_point) as f64) < max_forward_samples
            && (hit - returns[sample_window_end].position).norm() < sample_radius
        {
            sample_window_end += 1;
        }
        normals.push(estimate_normal(
            returns,
            current_point,
            sample_window_begin,
            sample_window_end,
            origin,
        ));
    }
    normals
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use nalgebra::Vector3;

    use super::*;
    use crate::common::internal::testing::lua_parameter_dictionary_test_helpers::make_dictionary;
    use crate::common::math::normalize_angle_difference;
    use crate::sensor::point_cloud::PointCloud;

    fn point(x: f32, y: f32) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(x, y, 0.0),
        }
    }

    fn circle_point(angle: f64) -> RangefinderPoint {
        RangefinderPoint {
            position: Vector3::new(angle.cos(), angle.sin(), 0.0).cast::<f32>(),
        }
    }

    fn options(num_normal_samples: i32) -> proto::NormalEstimationOptions2D {
        let parameter_dictionary = make_dictionary(&format!(
            "return {{ num_normal_samples = {num_normal_samples}, sample_radius = 10.0, }}"
        ));
        create_normal_estimation_options_2d(&parameter_dictionary)
    }

    mod normal_estimation_2d_test {
        use super::*;

        #[test]
        fn single_point() {
            let options = options(2);
            let mut range_data = RangeData::default();
            let num_angles = 100_usize;
            range_data.origin.x = 0.0;
            range_data.origin.y = 0.0;
            for angle_idx in 0..num_angles {
                let angle = angle_idx as f64 / num_angles as f64 * 2.0 * PI - PI;
                range_data.returns = PointCloud::from_points(vec![circle_point(angle)]);
                let normals = estimate_normals(&range_data, &options);
                let error = normalize_angle_difference(angle - f64::from(normals[0]) - PI);
                assert!(
                    error.abs() <= 2.0 * PI / num_angles as f64 + 1e-4,
                    "angle {angle}: normal {}",
                    normals[0]
                );
            }
        }

        #[test]
        fn straight_line_geometry() {
            let options = options(2);
            let mut range_data = RangeData::default();
            range_data.returns.push_back(point(-1.0, 1.0));
            range_data.returns.push_back(point(0.0, 1.0));
            range_data.returns.push_back(point(1.0, 1.0));
            range_data.origin.x = 0.0;
            range_data.origin.y = 0.0;
            let normals = estimate_normals(&range_data, &options);
            for normal in normals {
                assert!((f64::from(normal) - -FRAC_PI_2).abs() <= 1e-4);
            }

            range_data.returns =
                PointCloud::from_points(vec![point(1.0, 1.0), point(1.0, 0.0), point(1.0, -1.0)]);
            let normals = estimate_normals(&range_data, &options);
            for normal in normals {
                assert!((f64::from(normal.abs()) - PI).abs() <= 1e-4);
            }

            range_data.returns = PointCloud::from_points(vec![
                point(1.0, -1.0),
                point(0.0, -1.0),
                point(-1.0, -1.0),
            ]);
            let normals = estimate_normals(&range_data, &options);
            for normal in normals {
                assert!((f64::from(normal) - FRAC_PI_2).abs() <= 1e-4);
            }

            range_data.returns = PointCloud::from_points(vec![
                point(-1.0, -1.0),
                point(-1.0, 0.0),
                point(-1.0, 1.0),
            ]);
            let normals = estimate_normals(&range_data, &options);
            for normal in normals {
                assert!(f64::from(normal).abs() <= 1e-4);
            }
        }
    }

    mod circular_geometry_2d_test {
        use super::*;

        // C++ parameterized test `NumSamplesPerNormal` instantiated with
        // `Values(1, 2, 4, 5, 8)`.
        fn num_samples_per_normal(param: i32) {
            let options = options(param);
            let mut range_data = RangeData::default();
            let num_angles = 100_usize;
            for angle_idx in 0..num_angles {
                let angle = angle_idx as f64 / num_angles as f64 * 2.0 * PI - PI;
                range_data.returns.push_back(circle_point(angle));
            }
            range_data.origin.x = 0.0;
            range_data.origin.y = 0.0;
            let normals = estimate_normals(&range_data, &options);
            for (angle_idx, normal) in normals.iter().enumerate() {
                let angle = angle_idx as f64 / num_angles as f64 * 2.0 * PI;
                let error = normalize_angle_difference(f64::from(*normal) - angle);
                assert!(
                    error.abs() <= 2.0 * PI / num_angles as f64 * f64::from(param) / 2.0 + 1e-4,
                    "param {param}, angle_idx {angle_idx}: error {error}"
                );
            }
        }

        #[test]
        fn num_samples_per_normal_1() {
            num_samples_per_normal(1);
        }

        #[test]
        fn num_samples_per_normal_2() {
            num_samples_per_normal(2);
        }

        #[test]
        fn num_samples_per_normal_4() {
            num_samples_per_normal(4);
        }

        #[test]
        fn num_samples_per_normal_5() {
            num_samples_per_normal(5);
        }

        #[test]
        fn num_samples_per_normal_8() {
            num_samples_per_normal(8);
        }
    }

    #[test]
    #[should_panic(expected = "num_normal_samples")]
    fn options_reject_zero_samples() {
        let _ = options(0);
    }
}
