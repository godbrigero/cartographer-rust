// Ported from Ceres Solver 2.0.0: include/ceres/cost_function.h and
// include/ceres/sized_cost_function.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/cost_function.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sameeragarwal@google.com (Sameer Agarwal),
// keir@google.com (Keir Mierle)

//! The [`CostFunction`] interface and the [`SizedCostFunction`] helper.

use std::sync::Arc;

/// This is the interface through which the least squares solver accesses
/// the residual and Jacobian of the least squares problem. Users are
/// expected to implement this trait (or use
/// [`AutoDiffCostFunction`](crate::AutoDiffCostFunction)).
///
/// To create a new cost function, implement [`CostFunction::evaluate`],
/// [`CostFunction::num_residuals`] and
/// [`CostFunction::parameter_block_sizes`]. The size of the residual vector
/// and the sizes of the parameter blocks must stay constant for the lifetime
/// of the object.
///
/// RUST PORT NOTE: C++ requires `Send + Sync` implicitly (Ceres may evaluate
/// residual blocks from several threads); here it is explicit.
pub trait CostFunction: Send + Sync {
    /// Compute the residual vector and the Jacobian matrices.
    ///
    /// `parameters` is an array of slices to arrays containing the various
    /// parameter blocks. `parameters` has the same number of elements as
    /// `parameter_block_sizes()`, and `parameters[i]` has
    /// `parameter_block_sizes()[i]` elements.
    ///
    /// `residuals` is an array of size `num_residuals()`.
    ///
    /// `jacobians` is an array of size `parameter_block_sizes().len()`
    /// containing optional slices to storage for Jacobian matrices
    /// corresponding to each parameter block. The Jacobian matrices are in
    /// the same order as `parameter_block_sizes()`, i.e. `jacobians[i]` is
    /// the Jacobian of the residual vector with respect to `parameters[i]`.
    /// `jacobians[i]` is stored in **row-major** order, i.e.
    ///
    /// ```text
    ///   jacobians[i][r * parameter_block_size_[i] + c] =
    ///                              d residual[r] / d parameters[i][c]
    /// ```
    ///
    /// If `jacobians` is `None`, then no derivatives are returned; this is
    /// the case when computing cost only. If `jacobians[i]` is `None`, then
    /// the Jacobian block corresponding to the i'th parameter block must not
    /// be returned.
    ///
    /// The return value indicates whether the computation of the residuals
    /// and/or jacobians was successful or not. This can be used to
    /// communicate numerical failures in Jacobian computations for
    /// instance.
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool;

    /// Number of residuals.
    fn num_residuals(&self) -> usize;

    /// Sizes of the parameter blocks.
    fn parameter_block_sizes(&self) -> &[usize];
}

impl<C: CostFunction + ?Sized> CostFunction for Arc<C> {
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        (**self).evaluate(parameters, residuals, jacobians)
    }
    fn num_residuals(&self) -> usize {
        (**self).num_residuals()
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        (**self).parameter_block_sizes()
    }
}

/// A convenience helper for cost functions which know all parameter sizes
/// and the number of residuals at compile time (C++
/// `SizedCostFunction<kNumResiduals, N0, N1, ...>`).
///
/// Every `SizedCostFunction` is a [`CostFunction`] (blanket impl).
///
/// ```
/// use ceres::{CostFunction, SizedCostFunction};
///
/// struct Linear;
/// impl SizedCostFunction for Linear {
///     const NUM_RESIDUALS: usize = 1;
///     const PARAMETER_BLOCK_SIZES: &'static [usize] = &[2];
///     fn evaluate(
///         &self,
///         p: &[&[f64]],
///         r: &mut [f64],
///         j: Option<&mut [Option<&mut [f64]>]>,
///     ) -> bool {
///         r[0] = p[0][0] + 2.0 * p[0][1];
///         if let Some(Some(j0)) = j.map(|j| j[0].as_deref_mut()) {
///             j0.copy_from_slice(&[1.0, 2.0]);
///         }
///         true
///     }
/// }
/// let f: &dyn CostFunction = &Linear;
/// assert_eq!(f.parameter_block_sizes(), &[2]);
/// ```
pub trait SizedCostFunction: Send + Sync {
    /// Number of residuals (`kNumResiduals`).
    const NUM_RESIDUALS: usize;
    /// Sizes of the parameter blocks (`Ns...`).
    const PARAMETER_BLOCK_SIZES: &'static [usize];

    /// See [`CostFunction::evaluate`].
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool;
}

impl<S: SizedCostFunction> CostFunction for S {
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        SizedCostFunction::evaluate(self, parameters, residuals, jacobians)
    }
    fn num_residuals(&self) -> usize {
        S::NUM_RESIDUALS
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        S::PARAMETER_BLOCK_SIZES
    }
}
