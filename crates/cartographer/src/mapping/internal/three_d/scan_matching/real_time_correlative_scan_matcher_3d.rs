//! Port of `cartographer/mapping/internal/3d/scan_matching/real_time_correlative_scan_matcher_3d` (.h, .cc).

use nalgebra::Vector3;

use crate::common::math::pow2;
use crate::common::port::round_to_int;
use crate::mapping::scan_matching::proto;
use crate::mapping::three_d::hybrid_grid::HybridGrid;
use crate::sensor::point_cloud::PointCloud;
use crate::transform::rigid_transform::{Rigid3d, Rigid3f};
use crate::transform::transform::{angle_axis_vector_to_rotation_quaternion, get_angle};

/// A voxel accurate scan matcher, exhaustively evaluating the scan matching
/// search space.
pub struct RealTimeCorrelativeScanMatcher3D {
    options: proto::RealTimeCorrelativeScanMatcherOptions,
}

impl RealTimeCorrelativeScanMatcher3D {
    /// Creates the matcher.
    pub fn new(options: &proto::RealTimeCorrelativeScanMatcherOptions) -> Self {
        Self { options: *options }
    }

    /// Aligns `point_cloud` within the `hybrid_grid` given an
    /// `initial_pose_estimate` then updates `pose_estimate` with the result
    /// and returns the score.
    pub fn match_(
        &self,
        initial_pose_estimate: &Rigid3d,
        point_cloud: &PointCloud,
        hybrid_grid: &HybridGrid,
        pose_estimate: &mut Rigid3d,
    ) -> f32 {
        let mut best_score = -1.0f32;
        let initial_pose_estimate = initial_pose_estimate.cast::<f32>();
        for transform in
            self.generate_exhaustive_search_transforms(hybrid_grid.resolution(), point_cloud)
        {
            let candidate = initial_pose_estimate * transform;
            let score = self.score_candidate(hybrid_grid, point_cloud, &candidate, &transform);
            if score > best_score {
                best_score = score;
                *pose_estimate = candidate.cast::<f64>();
            }
        }
        best_score
    }

    fn generate_exhaustive_search_transforms(
        &self,
        resolution: f32,
        point_cloud: &PointCloud,
    ) -> Vec<Rigid3f> {
        let linear_window_size =
            round_to_int(self.options.linear_search_window / f64::from(resolution));
        // We set this value to something on the order of resolution to make
        // sure that the std::acos() below is defined.
        let mut max_scan_range = 3.0f32 * resolution;
        for point in point_cloud {
            let range = point.position.norm();
            max_scan_range = range.max(max_scan_range);
        }
        const SAFETY_MARGIN: f32 = 1.0 - 1e-3;
        let angular_step_size =
            SAFETY_MARGIN * (1.0f32 - pow2(resolution) / (2.0 * pow2(max_scan_range))).acos();
        let angular_window_size =
            round_to_int(self.options.angular_search_window / f64::from(angular_step_size));
        let linear_count = (2 * linear_window_size + 1) as usize;
        let angular_count = (2 * angular_window_size + 1) as usize;
        let mut result = Vec::with_capacity(linear_count.pow(3) * angular_count.pow(3));
        for z in -linear_window_size..=linear_window_size {
            for y in -linear_window_size..=linear_window_size {
                for x in -linear_window_size..=linear_window_size {
                    for rz in -angular_window_size..=angular_window_size {
                        for ry in -angular_window_size..=angular_window_size {
                            for rx in -angular_window_size..=angular_window_size {
                                let angle_axis = Vector3::new(
                                    rx as f32 * angular_step_size,
                                    ry as f32 * angular_step_size,
                                    rz as f32 * angular_step_size,
                                );
                                result.push(Rigid3f::new(
                                    Vector3::new(
                                        x as f32 * resolution,
                                        y as f32 * resolution,
                                        z as f32 * resolution,
                                    ),
                                    angle_axis_vector_to_rotation_quaternion(&angle_axis),
                                ));
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// RUST PORT NOTE: C++ materializes `TransformPointCloud(point_cloud,
    /// candidate)` per candidate; here each point is transformed on the fly
    /// (same float operations, no allocation).
    fn score_candidate(
        &self,
        hybrid_grid: &HybridGrid,
        point_cloud: &PointCloud,
        candidate: &Rigid3f,
        transform: &Rigid3f,
    ) -> f32 {
        let mut score = 0.0f32;
        for point in point_cloud {
            let world = candidate * &point.position;
            score += hybrid_grid.get_probability(&hybrid_grid.get_cell_index(&world));
        }
        score /= point_cloud.size() as f32;
        let angle = get_angle(transform);
        // The penalty is computed in double precision (float * double option
        // weights in C++), then multiplied into the float score.
        score = (f64::from(score)
            * (-pow2(
                f64::from(transform.translation().norm())
                    * self.options.translation_delta_cost_weight
                    + f64::from(angle) * self.options.rotation_delta_cost_weight,
            ))
            .exp()) as f32;
        assert!(score > 0.0, "Check failed: score > 0.f ({score} vs. 0)");
        score
    }
}

#[cfg(test)]
mod tests {
    mod real_time_correlative_scan_matcher_3d_test {
        use nalgebra::Vector3;

        use crate::mapping::internal::three_d::scan_matching::real_time_correlative_scan_matcher_3d::RealTimeCorrelativeScanMatcher3D;
        use crate::mapping::scan_matching::proto;
        use crate::mapping::three_d::hybrid_grid::HybridGrid;
        use crate::sensor::point_cloud::PointCloud;
        use crate::sensor::rangefinder_point::RangefinderPoint;
        use crate::transform::rigid_transform::{AngleAxis, Rigid3d};
        use crate::transform::rigid_transform_test_helpers::assert_is_nearly;

        struct Fixture {
            hybrid_grid: HybridGrid,
            expected_pose: Rigid3d,
            point_cloud: PointCloud,
            real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher3D,
        }

        impl Fixture {
            fn new() -> Self {
                let mut hybrid_grid = HybridGrid::new(0.1);
                let expected_pose = Rigid3d::from_translation(Vector3::new(-1., 0., 0.));
                let mut point_cloud = PointCloud::new();
                for point in [
                    Vector3::new(-3.0f32, 2., 0.),
                    Vector3::new(-4., 2., 0.),
                    Vector3::new(-5., 2., 0.),
                    Vector3::new(-6., 2., 0.),
                    Vector3::new(-6., 3., 1.),
                    Vector3::new(-6., 4., 2.),
                    Vector3::new(-7., 3., 1.),
                ] {
                    point_cloud.push_back(RangefinderPoint { position: point });
                    let index = hybrid_grid.get_cell_index(&(expected_pose.cast::<f32>() * point));
                    hybrid_grid.set_probability(&index, 1.);
                }
                // RUST PORT NOTE: C++ reads these options from Lua with
                // `CreateRealTimeCorrelativeScanMatcherOptions` (owned by the
                // shared `internal/scan_matching` module):
                // linear_search_window = 0.3,
                // angular_search_window = math.rad(1.),
                // translation_delta_cost_weight = 1e-1,
                // rotation_delta_cost_weight = 1.
                let options = proto::RealTimeCorrelativeScanMatcherOptions {
                    linear_search_window: 0.3,
                    angular_search_window: 1.0f64.to_radians(),
                    translation_delta_cost_weight: 1e-1,
                    rotation_delta_cost_weight: 1.,
                };
                Self {
                    hybrid_grid,
                    expected_pose,
                    point_cloud,
                    real_time_correlative_scan_matcher: RealTimeCorrelativeScanMatcher3D::new(
                        &options,
                    ),
                }
            }

            fn test_from_initial_pose(&self, initial_pose: &Rigid3d) {
                let mut pose = Rigid3d::identity();
                let score = self.real_time_correlative_scan_matcher.match_(
                    initial_pose,
                    &self.point_cloud,
                    &self.hybrid_grid,
                    &mut pose,
                );
                log::info!("Score: {score}");
                assert_is_nearly(&pose, &self.expected_pose, 1e-3);
            }
        }

        fn rotated(axis: Vector3<f64>) -> Rigid3d {
            // C++ `Eigen::AngleAxisd(angle, axis)` with an unnormalized axis
            // builds the quaternion `(cos(a/2), sin(a/2) * axis)`; replicated
            // here (AngleAxis::to_quaternion does not normalize either).
            Rigid3d::with_angle_axis(
                Vector3::new(-1., 0., 0.),
                AngleAxis::new(0.8 / 180. * std::f64::consts::PI, axis),
            )
        }

        #[test]
        fn perfect_estimate() {
            Fixture::new()
                .test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-1., 0., 0.)));
        }

        #[test]
        fn along_x() {
            Fixture::new()
                .test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-0.8, 0., 0.)));
        }

        #[test]
        fn along_z() {
            Fixture::new()
                .test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-1., 0., -0.2)));
        }

        #[test]
        fn along_xyz() {
            Fixture::new()
                .test_from_initial_pose(&Rigid3d::from_translation(Vector3::new(-0.9, -0.2, 0.2)));
        }

        #[test]
        fn rotation_around_x() {
            Fixture::new().test_from_initial_pose(&rotated(Vector3::new(1., 0., 0.)));
        }

        #[test]
        fn rotation_around_y() {
            Fixture::new().test_from_initial_pose(&rotated(Vector3::new(0., 1., 0.)));
        }

        #[test]
        fn rotation_around_yz() {
            Fixture::new().test_from_initial_pose(&rotated(Vector3::new(0., 1., 1.)));
        }
    }
}
