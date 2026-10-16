// Ported from Ceres Solver 2.0.0: include/ceres/autodiff_cost_function.h and
// include/ceres/internal/autodiff.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/autodiff_cost_function.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Create [`CostFunction`]s as needed by the least squares framework, with
//! Jacobians computed via automatic differentiation.
//!
//! To get an auto differentiated cost function, you must define a type with
//! a generic `evaluate` method (a functor, [`CostFunctor`]) that computes the
//! cost function in terms of the generic scalar type `T`. The autodiff
//! framework substitutes appropriate [`Jet`] objects for `T` in order to
//! compute the derivative when necessary, but this is hidden, and you should
//! write the function as if `T` were a scalar type (e.g. `f64`).
//!
//! ```
//! use ceres::{AutoDiffCostFunction, CostFunction, CostFunctor, Scalar};
//!
//! // residual = k - x'y, with x and y 2-vectors.
//! struct MyScalarCostFunctor { k: f64 }
//! impl CostFunctor for MyScalarCostFunctor {
//!     fn evaluate<T: Scalar>(&self, p: &[&[T]], e: &mut [T]) -> bool {
//!         let (x, y) = (p[0], p[1]);
//!         e[0] = T::constant(self.k) - x[0] * y[0] - x[1] * y[1];
//!         true
//!     }
//! }
//! // 1 residual, two parameter blocks of size 2 (N = 2 + 2 = 4 in total).
//! let f = AutoDiffCostFunction::<_, 4>::new(MyScalarCostFunctor { k: 1.0 }, 1, &[2, 2]);
//! let mut r = [0.0];
//! let (mut jx, mut jy) = ([0.0; 2], [0.0; 2]);
//! let mut jac = [Some(&mut jx[..]), Some(&mut jy[..])];
//! assert!(f.evaluate(&[&[1.0, 2.0], &[3.0, 4.0]], &mut r, Some(&mut jac)));
//! assert_eq!(r[0], 1.0 - 11.0);
//! assert_eq!(jx, [-3.0, -4.0]);
//! assert_eq!(jy, [-1.0, -2.0]);
//! ```
//!
//! RUST PORT NOTE: C++ encodes the residual count and the parameter block
//! sizes as template arguments (`AutoDiffCostFunction<F, kNumResiduals, N0,
//! N1, ...>`). Rust cannot sum a const-generic list, so the total number of
//! parameters `N = N0 + N1 + ...` (the jet dimension) is the const generic,
//! and the residual count and block sizes are runtime constructor arguments
//! (validated against `N`). `ceres::DYNAMIC` residuals are therefore just a
//! runtime `num_residuals`.

use std::sync::Mutex;

use crate::cost_function::CostFunction;
use crate::jet::Jet;
use crate::scalar::Scalar;

/// A cost functor: computes residuals from parameter blocks for any scalar
/// type `T` (`f64` or [`Jet`]).
///
/// The `evaluate` method receives one slice per parameter block and must
/// write all residuals; it returns `false` to signal that the residuals are
/// not computable at this point.
pub trait CostFunctor: Send + Sync {
    /// Computes the residuals (C++ `template <typename T> bool
    /// operator()(const T* const x0, ..., T* residuals) const`).
    fn evaluate<T: Scalar>(&self, parameters: &[&[T]], residuals: &mut [T]) -> bool;
}

/// Sentinel written into the residual jets before calling the functor, as in
/// C++ (`kImpossibleValue`), so unwritten residuals are detected downstream.
pub(crate) const K_IMPOSSIBLE_VALUE: f64 = 1e302;

/// Number of residuals evaluated with stack storage (C++
/// `CERES_AUTODIFF_MAX_RESIDUALS_ON_STACK`).
const MAX_RESIDUALS_ON_STACK: usize = 20;
/// Number of parameter blocks handled with stack storage.
const MAX_PARAMETER_BLOCKS_ON_STACK: usize = 10;

/// A [`CostFunction`] whose Jacobians are computed by automatic
/// differentiation of a [`CostFunctor`]; `N` is the total number of
/// parameters (sum of all parameter block sizes).
pub struct AutoDiffCostFunction<F, const N: usize> {
    functor: F,
    num_residuals: usize,
    parameter_block_sizes: Vec<usize>,
    // Scratch space for the residual jets when there are many residuals (e.g.
    // scan matching), to keep the hot path allocation-free.
    scratch: Mutex<Vec<Jet<N>>>,
}

impl<F: CostFunctor, const N: usize> AutoDiffCostFunction<F, N> {
    /// Creates the cost function; `num_residuals` may be chosen at runtime
    /// (C++ `ceres::DYNAMIC`). Panics if the parameter block sizes do not
    /// sum to `N` or a block size is zero.
    pub fn new(functor: F, num_residuals: usize, parameter_block_sizes: &[usize]) -> Self {
        assert!(
            !parameter_block_sizes.is_empty(),
            "AutoDiffCostFunction needs at least one parameter block"
        );
        assert!(
            parameter_block_sizes.iter().all(|&s| s > 0),
            "Parameter block sizes must be positive: {parameter_block_sizes:?}"
        );
        let total: usize = parameter_block_sizes.iter().sum();
        assert_eq!(
            total, N,
            "Sum of parameter block sizes {parameter_block_sizes:?} must equal the jet dimension N = {N}"
        );
        Self {
            functor,
            num_residuals,
            parameter_block_sizes: parameter_block_sizes.to_vec(),
            scratch: Mutex::new(Vec::new()),
        }
    }

    /// The wrapped functor.
    pub fn functor(&self) -> &F {
        &self.functor
    }

    fn autodiff(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: &mut [Option<&mut [f64]>],
    ) -> bool {
        let num_blocks = self.parameter_block_sizes.len();

        // Make1stOrderPerturbations: x[k] = parameters[b][j] + t_k.
        let mut x = [Jet::<N>::constant(0.0); N];
        let mut k = 0;
        for (b, &size) in self.parameter_block_sizes.iter().enumerate() {
            let p = parameters[b];
            for &pj in &p[..size] {
                x[k] = Jet::variable(pj, k);
                k += 1;
            }
        }

        // Unpack the jets into one slice per parameter block.
        let mut stack_blocks: [&[Jet<N>]; MAX_PARAMETER_BLOCKS_ON_STACK] =
            [&[]; MAX_PARAMETER_BLOCKS_ON_STACK];
        let mut heap_blocks: Vec<&[Jet<N>]>;
        let blocks: &mut [&[Jet<N>]] = if num_blocks <= MAX_PARAMETER_BLOCKS_ON_STACK {
            &mut stack_blocks[..num_blocks]
        } else {
            heap_blocks = vec![&[]; num_blocks];
            &mut heap_blocks[..]
        };
        let mut rest: &[Jet<N>] = &x;
        for (b, &size) in self.parameter_block_sizes.iter().enumerate() {
            let (head, tail) = rest.split_at(size);
            blocks[b] = head;
            rest = tail;
        }

        let num_outputs = self.num_residuals;
        let impossible = Jet::<N>::new(K_IMPOSSIBLE_VALUE, [K_IMPOSSIBLE_VALUE; N]);
        let mut stack_residuals = [impossible; MAX_RESIDUALS_ON_STACK];
        let mut heap_residuals: Vec<Jet<N>>;
        let mut guard;
        let residual_jets: &mut [Jet<N>] = if num_outputs <= MAX_RESIDUALS_ON_STACK {
            &mut stack_residuals[..num_outputs]
        } else if let Ok(g) = self.scratch.try_lock() {
            guard = g;
            guard.clear();
            guard.resize(num_outputs, impossible);
            &mut guard[..]
        } else {
            heap_residuals = vec![impossible; num_outputs];
            &mut heap_residuals[..]
        };

        if !self.functor.evaluate::<Jet<N>>(blocks, residual_jets) {
            return false;
        }

        // Take0thOrderPart.
        for (r, j) in residuals.iter_mut().zip(residual_jets.iter()) {
            *r = j.a;
        }

        // Take1stOrderParts.
        let mut offset = 0;
        for (b, &size) in self.parameter_block_sizes.iter().enumerate() {
            if let Some(jac) = jacobians[b].as_deref_mut() {
                for (i, rj) in residual_jets.iter().enumerate() {
                    jac[i * size..(i + 1) * size].copy_from_slice(&rj.v[offset..offset + size]);
                }
            }
            offset += size;
        }
        true
    }
}

impl<F: CostFunctor, const N: usize> CostFunction for AutoDiffCostFunction<F, N> {
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        match jacobians {
            None => self.functor.evaluate::<f64>(parameters, residuals),
            Some(jacobians) => self.autodiff(parameters, residuals, jacobians),
        }
    }

    fn num_residuals(&self) -> usize {
        self.num_residuals
    }

    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }
}
