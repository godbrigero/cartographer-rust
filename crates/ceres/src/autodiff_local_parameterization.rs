// Ported from Ceres Solver 2.0.0: include/ceres/autodiff_local_parameterization.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/autodiff_local_parameterization.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sergey.vfx@gmail.com (Sergey Sharybin),
// mierle@gmail.com (Keir Mierle), sameeragarwal@google.com (Sameer Agarwal)

//! Create local parameterization with Jacobians computed via automatic
//! differentiation.
//!
//! To get an auto differentiated local parameterization, you must define a
//! type with a generic `plus` method ([`PlusFunctor`]) that computes
//!
//! ```text
//!   x_plus_delta = Plus(x, delta);
//! ```
//!
//! the template parameter `T`. The autodiff framework substitutes
//! appropriate [`Jet`] objects for `T` in order to compute the derivative
//! when necessary, but this is hidden, and you should write the function as
//! if `T` were a scalar type (e.g. `f64`).
//!
//! For example, Cartographer's `YawOnlyQuaternionPlus`:
//!
//! ```
//! use ceres::{AutoDiffLocalParameterization, LocalParameterization, PlusFunctor, Scalar};
//! use ceres::rotation::quaternion_product;
//!
//! struct YawOnlyQuaternionPlus;
//! impl PlusFunctor for YawOnlyQuaternionPlus {
//!     fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool {
//!         let clamped_delta = delta[0].clamp(T::constant(-0.5), T::constant(0.5));
//!         let q_delta = [
//!             (T::constant(1.0) - clamped_delta * clamped_delta).sqrt(),
//!             T::constant(0.0),
//!             T::constant(0.0),
//!             clamped_delta,
//!         ];
//!         quaternion_product(&q_delta, x, x_plus_delta);
//!         true
//!     }
//! }
//! let p = AutoDiffLocalParameterization::<_, 4, 1>::new(YawOnlyQuaternionPlus);
//! let mut jacobian = [0.0; 4];
//! assert!(p.compute_jacobian(&[1.0, 0.0, 0.0, 0.0], &mut jacobian));
//! assert_eq!(jacobian, [0.0, 0.0, 0.0, 1.0]);
//! ```
//!
//! RUST PORT NOTE: the C++ version differentiates w.r.t. `x` and `delta`
//! (jets of dimension `kGlobalSize + kLocalSize`) and discards the `x` part;
//! here only `delta` is seeded (jets of dimension `LOCAL`), which yields the
//! identical Jacobian at lower cost (Rust cannot form `GLOBAL + LOCAL` as a
//! const generic).

use crate::jet::Jet;
use crate::local_parameterization::LocalParameterization;
use crate::scalar::Scalar;

/// The `Plus` operation of an [`AutoDiffLocalParameterization`], generic over
/// the scalar type.
pub trait PlusFunctor: Send + Sync {
    /// `x_plus_delta = Plus(x, delta)`; returns `false` on failure.
    fn plus<T: Scalar>(&self, x: &[T], delta: &[T], x_plus_delta: &mut [T]) -> bool;
}

/// A [`LocalParameterization`] whose Jacobian is computed by automatic
/// differentiation of a [`PlusFunctor`].
pub struct AutoDiffLocalParameterization<F, const GLOBAL: usize, const LOCAL: usize> {
    functor: F,
}

impl<F: PlusFunctor, const GLOBAL: usize, const LOCAL: usize>
    AutoDiffLocalParameterization<F, GLOBAL, LOCAL>
{
    /// Takes ownership of functor.
    pub fn new(functor: F) -> Self {
        Self { functor }
    }

    /// The wrapped functor.
    pub fn functor(&self) -> &F {
        &self.functor
    }
}

impl<F: PlusFunctor + Default, const GLOBAL: usize, const LOCAL: usize> Default
    for AutoDiffLocalParameterization<F, GLOBAL, LOCAL>
{
    fn default() -> Self {
        Self::new(F::default())
    }
}

impl<F: PlusFunctor, const GLOBAL: usize, const LOCAL: usize> LocalParameterization
    for AutoDiffLocalParameterization<F, GLOBAL, LOCAL>
{
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        self.functor
            .plus::<f64>(&x[..GLOBAL], &delta[..LOCAL], &mut x_plus_delta[..GLOBAL])
    }

    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        let mut x_jets = [Jet::<LOCAL>::constant(0.0); GLOBAL];
        for (j, &xi) in x_jets.iter_mut().zip(&x[..GLOBAL]) {
            *j = Jet::constant(xi);
        }
        let mut delta_jets = [Jet::<LOCAL>::constant(0.0); LOCAL];
        for (i, d) in delta_jets.iter_mut().enumerate() {
            *d = Jet::variable(0.0, i);
        }
        let impossible = Jet::<LOCAL>::new(
            crate::autodiff_cost_function::K_IMPOSSIBLE_VALUE,
            [crate::autodiff_cost_function::K_IMPOSSIBLE_VALUE; LOCAL],
        );
        let mut x_plus_delta = [impossible; GLOBAL];
        if !self
            .functor
            .plus::<Jet<LOCAL>>(&x_jets, &delta_jets, &mut x_plus_delta)
        {
            return false;
        }
        for (r, out) in x_plus_delta.iter().enumerate() {
            jacobian[r * LOCAL..(r + 1) * LOCAL].copy_from_slice(&out.v);
        }
        true
    }

    fn global_size(&self) -> usize {
        GLOBAL
    }

    fn local_size(&self) -> usize {
        LOCAL
    }
}
