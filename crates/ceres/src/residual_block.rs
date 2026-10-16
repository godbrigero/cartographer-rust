// Ported from Ceres Solver 2.0.0: internal/ceres/residual_block.cc,
// internal/ceres/residual_block_utils.cc and internal/ceres/array_utils.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/residual_block.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sameeragarwal@google.com (Sameer Agarwal),
// keir@google.com (Keir Mierle)

//! Evaluation of a single residual block: cost function, local
//! parameterization Jacobians and loss function correction.

use crate::autodiff_cost_function::K_IMPOSSIBLE_VALUE;
use crate::corrector::Corrector;
use crate::cost_function::CostFunction;
use crate::loss_function::LossFunction;

/// Number of parameter blocks per residual handled without heap allocation
/// (C++ `FixedArray<..., 8>`).
pub(crate) const MAX_ARGS_ON_STACK: usize = 8;

/// `IsArrayValid`: all values are finite and none is the sentinel
/// `kImpossibleValue`.
#[inline]
pub(crate) fn is_array_valid(x: &[f64]) -> bool {
    x.iter().all(|&v| v.is_finite() && v != K_IMPOSSIBLE_VALUE)
}

/// `InvalidateArray`: fills `x` with `kImpossibleValue`.
#[inline]
pub(crate) fn invalidate_array(x: &mut [f64]) {
    x.fill(K_IMPOSSIBLE_VALUE);
}

/// Inputs describing one parameter block argument of a residual block.
#[derive(Clone, Copy)]
pub(crate) struct ArgInput<'s> {
    /// Current values of the parameter block.
    pub values: &'s [f64],
    /// Row-major `size x local_size` local parameterization Jacobian, if the
    /// block has a local parameterization.
    pub local_parameterization_jacobian: Option<&'s [f64]>,
    /// Local size of the block.
    pub local_size: usize,
}

/// Evaluates a residual block (C++ `ResidualBlock::Evaluate`).
///
/// Evaluates the residual term, storing the scalar cost in `cost`, the
/// residuals in `residuals` (if `Some`), and the jacobians between the
/// parameters and residuals in `jacobians[i]` (row-major,
/// `num_residuals x local_size`), if requested. If `jacobians[i]` is `None`
/// (e.g. the block is constant), that jacobian is not computed. If
/// `apply_loss_function` is true, the loss function is applied (via the
/// corrector) to the residuals and jacobians.
///
/// The returned jacobians have the local parameterizations applied.
/// `scratch` is grown as needed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_residual_block(
    cost_function: &dyn CostFunction,
    loss_function: Option<&dyn LossFunction>,
    args: &[ArgInput<'_>],
    apply_loss_function: bool,
    cost: &mut f64,
    residuals: Option<&mut [f64]>,
    mut jacobians: Option<&mut [Option<&mut [f64]>]>,
    scratch: &mut Vec<f64>,
) -> bool {
    let num_parameter_blocks = args.len();
    let num_residuals = cost_function.num_residuals();
    let sizes = cost_function.parameter_block_sizes();

    // Compute the amount of scratch space needed to store the full-sized
    // jacobians. For parameters that have no local parameterization no
    // storage is needed and the passed-in jacobian array is used directly.
    // Also include space to store the residuals, which is needed for
    // cost-only evaluations.
    let mut jacobian_scratch_size = 0;
    if let Some(jacs) = jacobians.as_deref() {
        for i in 0..num_parameter_blocks {
            if jacs[i].is_some() && args[i].local_parameterization_jacobian.is_some() {
                jacobian_scratch_size += num_residuals * sizes[i];
            }
        }
    }
    let needed = jacobian_scratch_size + num_residuals;
    if scratch.len() < needed {
        scratch.resize(needed, 0.0);
    }
    let (jacobian_scratch, residual_scratch) =
        scratch[..needed].split_at_mut(jacobian_scratch_size);

    // If the caller didn't request residuals, use the scratch space for them.
    let outputting_residuals = residuals.is_some();
    let residuals: &mut [f64] = match residuals {
        Some(r) => &mut r[..num_residuals],
        None => residual_scratch,
    };

    // Collect the parameters from their blocks. This will rarely allocate,
    // since residuals taking more than 8 parameter block arguments are rare.
    let mut stack_parameters: [&[f64]; MAX_ARGS_ON_STACK] = [&[]; MAX_ARGS_ON_STACK];
    let mut heap_parameters: Vec<&[f64]>;
    let parameters: &mut [&[f64]] = if num_parameter_blocks <= MAX_ARGS_ON_STACK {
        &mut stack_parameters[..num_parameter_blocks]
    } else {
        heap_parameters = vec![&[]; num_parameter_blocks];
        &mut heap_parameters[..]
    };
    for (p, a) in parameters.iter_mut().zip(args) {
        *p = a.values;
    }

    // Invalidate the evaluation buffers so that we can check them after the
    // CostFunction::Evaluate call, to see if all the return values that were
    // required were written to and that they are finite.
    invalidate_array(residuals);

    {
        // Put pointers into the scratch space into global_jacobians as
        // appropriate.
        let mut stack_global: [Option<&mut [f64]>; MAX_ARGS_ON_STACK] =
            [const { None }; MAX_ARGS_ON_STACK];
        let mut heap_global: Vec<Option<&mut [f64]>>;
        let ok = match jacobians.as_deref_mut() {
            None => {
                if !cost_function.evaluate(parameters, residuals, None) {
                    return false;
                }
                true
            }
            Some(jacs) => {
                let global: &mut [Option<&mut [f64]>] = if num_parameter_blocks <= MAX_ARGS_ON_STACK
                {
                    &mut stack_global[..num_parameter_blocks]
                } else {
                    heap_global = (0..num_parameter_blocks).map(|_| None).collect();
                    &mut heap_global[..]
                };
                let mut rest: &mut [f64] = &mut *jacobian_scratch;
                for (i, (g, j)) in global.iter_mut().zip(jacs.iter_mut()).enumerate() {
                    if let Some(j) = j.as_deref_mut() {
                        let n = num_residuals * sizes[i];
                        if args[i].local_parameterization_jacobian.is_some() {
                            let (head, tail) = std::mem::take(&mut rest).split_at_mut(n);
                            rest = tail;
                            invalidate_array(head);
                            *g = Some(head);
                        } else {
                            let j = &mut j[..n];
                            invalidate_array(j);
                            *g = Some(j);
                        }
                    }
                }
                if !cost_function.evaluate(parameters, residuals, Some(&mut *global)) {
                    return false;
                }
                global
                    .iter()
                    .all(|g| g.as_deref().is_none_or(is_array_valid))
            }
        };
        if !ok {
            log::warn!(
                "\n\nError in evaluating the ResidualBlock: the CostFunction did not fill all \
                 requested jacobians or produced a non-finite jacobian value.\nParameters: {:?}",
                parameters
            );
            return false;
        }
    }

    if !is_array_valid(residuals) {
        log::warn!(
            "\n\nError in evaluating the ResidualBlock.\n\n\
             There are two possible reasons. Either the CostFunction did not evaluate and fill all\n\
             residual and jacobians that were requested or there was a non-finite value (nan/infinite)\n\
             generated during the or jacobian computation.\n\n\
             Parameters: {:?}\nResiduals: {:?}",
            parameters,
            residuals
        );
        return false;
    }

    let squared_norm: f64 = residuals.iter().map(|r| r * r).sum();

    // Update the jacobians with the local parameterizations.
    if let Some(jacs) = jacobians.as_deref_mut() {
        let mut rest: &[f64] = &*jacobian_scratch;
        for (i, j) in jacs.iter_mut().enumerate() {
            let Some(j) = j.as_deref_mut() else { continue };
            let Some(lp) = args[i].local_parameterization_jacobian else {
                continue;
            };
            let size = sizes[i];
            let local_size = args[i].local_size;
            let (global, tail) = rest.split_at(num_residuals * size);
            rest = tail;
            // jacobians[i] = global_jacobians[i] * global_to_local_jacobian.
            for r in 0..num_residuals {
                for c in 0..local_size {
                    let mut sum = 0.0;
                    for k in 0..size {
                        sum += global[r * size + k] * lp[k * local_size + c];
                    }
                    j[r * local_size + c] = sum;
                }
            }
        }
    }

    let loss_function = match loss_function {
        Some(l) if apply_loss_function => l,
        _ => {
            *cost = 0.5 * squared_norm;
            return true;
        }
    };

    let mut rho = [0.0; 3];
    loss_function.evaluate(squared_norm, &mut rho);
    *cost = 0.5 * rho[0];

    // No jacobians and not outputting residuals? All done. Doing an early
    // exit here avoids constructing the "Corrector" object below in a common
    // case.
    if jacobians.is_none() && !outputting_residuals {
        return true;
    }

    // Correct for the effects of the loss function. The jacobians need to be
    // corrected before the residuals, since they use the uncorrected
    // residuals.
    let correct = Corrector::new(squared_norm, &rho);
    if let Some(jacs) = jacobians {
        for (i, j) in jacs.iter_mut().enumerate() {
            if let Some(j) = j.as_deref_mut() {
                // Correct the jacobians for the loss function.
                correct.correct_jacobian(num_residuals, args[i].local_size, residuals, j);
            }
        }
    }

    // Correct the residuals with the loss function.
    if outputting_residuals {
        correct.correct_residuals(residuals);
    }
    true
}
