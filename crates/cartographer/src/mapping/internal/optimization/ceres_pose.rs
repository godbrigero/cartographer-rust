//! Port of `cartographer/mapping/internal/optimization/ceres_pose` (.h, .cc).
//!
//! RUST PORT NOTE: in C++ `CeresPose` owns the parameter memory
//! (`std::shared_ptr<Data>`) that the `ceres::Problem` optimizes in place, and
//! `translation()`/`rotation()` return the raw `double*` used to add residual
//! blocks. The Rust `ceres::Problem` is handle-based and owns the parameter
//! values, so `CeresPose` stores the two [`ParameterBlockId`]s instead:
//! [`CeresPose::translation`]/[`CeresPose::rotation`] return the handles, and
//! [`CeresPose::to_rigid`]/[`CeresPose::data`] read the current values back
//! from the problem. Like the C++ class (which shares its `Data`), the type is
//! cheap to copy.

use ceres::{LocalParameterization, ParameterBlockId, Problem};

use crate::transform::rigid_transform::Rigid3d;

/// The parameter values of a [`CeresPose`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Data {
    /// Translation `[x, y, z]`.
    pub translation: [f64; 3],
    /// Rotation quaternion as `(w, x, y, z)`.
    pub rotation: [f64; 4],
}

/// Converts a pose into parameter values (C++ `FromPose`).
pub fn from_pose(pose: &Rigid3d) -> Data {
    let t = pose.translation();
    let r = pose.rotation();
    Data {
        translation: [t.x, t.y, t.z],
        rotation: [r.w, r.i, r.j, r.k],
    }
}

/// A 3D pose in a `ceres::Problem`, as a translation parameter block (size 3)
/// and a rotation parameter block (size 4, `(w, x, y, z)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeresPose {
    translation: ParameterBlockId,
    rotation: ParameterBlockId,
}

impl CeresPose {
    /// Adds the parameter blocks for `rigid` to `problem`, with optional local
    /// parameterizations (C++ `nullptr` -> `None`).
    pub fn new<'a>(
        rigid: &Rigid3d,
        translation_parametrization: Option<Box<dyn LocalParameterization + 'a>>,
        rotation_parametrization: Option<Box<dyn LocalParameterization + 'a>>,
        problem: &mut Problem<'a>,
    ) -> Self {
        let data = from_pose(rigid);
        let translation = problem.add_parameter_block_with_parameterization(
            &data.translation,
            translation_parametrization,
        );
        let rotation = problem
            .add_parameter_block_with_parameterization(&data.rotation, rotation_parametrization);
        CeresPose {
            translation,
            rotation,
        }
    }

    /// The current pose in `problem`.
    pub fn to_rigid(&self, problem: &Problem<'_>) -> Rigid3d {
        let data = self.data(problem);
        Rigid3d::from_arrays(&data.rotation, &data.translation)
    }

    /// Handle of the translation parameter block.
    pub fn translation(&self) -> ParameterBlockId {
        self.translation
    }

    /// Handle of the rotation parameter block.
    pub fn rotation(&self) -> ParameterBlockId {
        self.rotation
    }

    /// The current parameter values in `problem`.
    pub fn data(&self, problem: &Problem<'_>) -> Data {
        let mut data = Data {
            translation: [0.; 3],
            rotation: [0.; 4],
        };
        data.translation
            .copy_from_slice(problem.parameter_block(self.translation));
        data.rotation
            .copy_from_slice(problem.parameter_block(self.rotation));
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ceres::QuaternionParameterization;
    use nalgebra::{Quaternion, Vector3};

    #[test]
    fn round_trip_through_problem() {
        let pose = Rigid3d::new(Vector3::new(1., 2., 3.), Quaternion::new(0.6, 0., 0.8, 0.));
        let mut problem = Problem::new();
        let ceres_pose = CeresPose::new(
            &pose,
            None,
            Some(Box::new(QuaternionParameterization)),
            &mut problem,
        );
        assert_eq!(problem.num_parameter_blocks(), 2);
        assert_eq!(problem.parameter_block_size(ceres_pose.translation()), 3);
        assert_eq!(problem.parameter_block_size(ceres_pose.rotation()), 4);
        assert_eq!(problem.parameter_block_local_size(ceres_pose.rotation()), 3);
        assert_eq!(ceres_pose.to_rigid(&problem), pose);
        assert_eq!(
            ceres_pose.data(&problem),
            Data {
                translation: [1., 2., 3.],
                rotation: [0.6, 0., 0.8, 0.],
            }
        );
        problem.set_parameter_block_values(ceres_pose.translation(), &[4., 5., 6.]);
        assert_eq!(
            *ceres_pose.to_rigid(&problem).translation(),
            Vector3::new(4., 5., 6.)
        );
    }
}
