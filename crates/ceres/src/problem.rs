// Ported from Ceres Solver 2.0.0: include/ceres/problem.h,
// internal/ceres/problem.cc, internal/ceres/problem_impl.cc,
// internal/ceres/parameter_block.h and include/ceres/crs_matrix.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/problem.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: sameeragarwal@google.com (Sameer Agarwal),
// keir@google.com (Keir Mierle)

//! A non-linear least squares problem.
//!
//! The cost function takes the form
//!
//! ```text
//!    N    1
//!   SUM  --- loss( || r_i1, r_i2,..., r_ik ||^2  ),
//!   i=1   2
//! ```
//!
//! where `r_ij` is residual number i, component j; the residual is a
//! function of some subset of the parameters x1...xk, and `loss(y)` is the
//! loss function; for example, squared error or Huber L1 loss. If
//! `loss(y) = y`, then the cost function is non-robustified least squares.
//!
//! RUST PORT NOTE: Ceres identifies parameter blocks by the user's `double*`
//! and optimizes the user's memory in place. Rust cannot share mutable raw
//! memory that way, so this port uses a handle-based API: the problem owns
//! a copy of the values of each parameter block
//! ([`Problem::add_parameter_block`] returns a [`ParameterBlockId`]), residual
//! blocks refer to parameter blocks by id, and after
//! [`solve`](crate::solve) the optimized values are read back with
//! [`Problem::parameter_block`]. Since there is no pointer identity, adding
//! the same values twice creates two different blocks, and parameter blocks
//! must be added before the residual blocks that use them. Ownership flags
//! (`Problem::Options::*_ownership`) are replaced by Rust ownership (`Box`;
//! wrap shared objects in `Arc`). The lifetime `'a` allows cost functions to
//! borrow data (e.g. a point cloud or grid) for the duration of the problem.

use crate::cost_function::CostFunction;
use crate::local_parameterization::LocalParameterization;
use crate::loss_function::LossFunction;
use crate::program::{BlockJacobian, Evaluator, Program};
use crate::residual_block::{ArgInput, evaluate_residual_block};

/// Opaque handle of a parameter block in a [`Problem`] (C++ `double*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterBlockId(pub(crate) usize);

/// Opaque handle of a residual block in a [`Problem`] (C++
/// `ResidualBlockId`), e.g. to remove it later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidualBlockId(pub(crate) usize);

/// Options of a [`Problem`] (C++ `Problem::Options`).
///
/// RUST PORT NOTE: the ownership options and the evaluation callback / context
/// are not needed in Rust and therefore absent.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProblemOptions {
    /// Kept for API compatibility; removal is always supported here.
    pub enable_fast_removal: bool,
    /// By default, Ceres performs a variety of safety checks when constructing
    /// the problem. If set, the duplicate parameter block check in
    /// `add_residual_block` is skipped (size checks are always performed).
    pub disable_all_safety_checks: bool,
}

/// A parameter block stored in the problem (C++ `ParameterBlock`).
pub(crate) struct ParameterBlock<'a> {
    pub values: Vec<f64>,
    pub constant: bool,
    pub parameterization: Option<Box<dyn LocalParameterization + 'a>>,
    /// Upper and lower bounds for the parameter block. `set_upper_bound`
    /// and `set_lower_bound` lazily initialize them; `None` means
    /// unbounded (`±f64::MAX`).
    pub upper_bounds: Option<Vec<f64>>,
    pub lower_bounds: Option<Vec<f64>>,
}

impl ParameterBlock<'_> {
    /// C++ `ParameterBlock::SetUpperBound`.
    pub(crate) fn set_upper_bound(&mut self, index: usize, upper_bound: f64) {
        let size = self.size();
        assert!(
            index < size,
            "Check failed: index < size ({index} vs. {size})"
        );

        if upper_bound >= f64::MAX && self.upper_bounds.is_none() {
            return;
        }

        self.upper_bounds
            .get_or_insert_with(|| vec![f64::MAX; size])[index] = upper_bound;
    }

    /// C++ `ParameterBlock::SetLowerBound`.
    pub(crate) fn set_lower_bound(&mut self, index: usize, lower_bound: f64) {
        let size = self.size();
        assert!(
            index < size,
            "Check failed: index < size ({index} vs. {size})"
        );

        if lower_bound <= -f64::MAX && self.lower_bounds.is_none() {
            return;
        }

        self.lower_bounds
            .get_or_insert_with(|| vec![-f64::MAX; size])[index] = lower_bound;
    }

    /// C++ `ParameterBlock::UpperBoundForParameter` (also `UpperBound`).
    pub(crate) fn upper_bound_for_parameter(&self, index: usize) -> f64 {
        self.upper_bounds.as_ref().map_or(f64::MAX, |b| b[index])
    }

    /// C++ `ParameterBlock::LowerBoundForParameter` (also `LowerBound`).
    pub(crate) fn lower_bound_for_parameter(&self, index: usize) -> f64 {
        self.lower_bounds.as_ref().map_or(-f64::MAX, |b| b[index])
    }

    /// Generalization of the addition operation. This is the same as
    /// `LocalParameterization::plus()` followed by projection onto the hyper
    /// cube implied by the bounds constraints (C++ `ParameterBlock::Plus`).
    pub(crate) fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let size = self.size();
        match &self.parameterization {
            Some(p) => {
                if !p.plus(x, delta, x_plus_delta) {
                    return false;
                }
            }
            None => {
                for i in 0..size {
                    x_plus_delta[i] = x[i] + delta[i];
                }
            }
        }

        // Project onto the box constraints.
        if let Some(lower_bounds) = &self.lower_bounds {
            for i in 0..size {
                x_plus_delta[i] = x_plus_delta[i].max(lower_bounds[i]);
            }
        }

        if let Some(upper_bounds) = &self.upper_bounds {
            for i in 0..size {
                x_plus_delta[i] = x_plus_delta[i].min(upper_bounds[i]);
            }
        }

        true
    }

    pub(crate) fn size(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn local_size(&self) -> usize {
        self.parameterization
            .as_ref()
            .map_or(self.values.len(), |p| p.local_size())
    }

    /// A block is constant if set constant or if its tangent space is empty.
    pub(crate) fn is_constant(&self) -> bool {
        self.constant || self.local_size() == 0
    }
}

/// A residual block stored in the problem (C++ `ResidualBlock`).
pub(crate) struct ResidualBlock<'a> {
    pub cost_function: Box<dyn CostFunction + 'a>,
    pub loss_function: Option<Box<dyn LossFunction + 'a>>,
    pub parameter_blocks: Vec<ParameterBlockId>,
}

/// Options for [`Problem::evaluate`] (C++ `Problem::EvaluateOptions`).
#[derive(Clone, Debug)]
pub struct EvaluateOptions {
    /// The set of parameter blocks for which evaluation should be performed.
    /// This vector determines the order that parameter blocks occur in the
    /// gradient vector and in the columns of the jacobian matrix. If empty,
    /// then it is assumed to be equal to a vector containing ALL the
    /// parameter blocks (in the order in which they were added). Blocks not
    /// listed are treated as constant.
    pub parameter_blocks: Vec<ParameterBlockId>,
    /// The set of residual blocks to evaluate. This vector determines the
    /// order in which the residuals occur, and how the rows of the jacobian
    /// are ordered. If empty, then it is assumed to be equal to the vector
    /// containing ALL the residual blocks.
    pub residual_blocks: Vec<ResidualBlockId>,
    /// Even though the residual blocks in the problem may contain loss
    /// functions, setting apply_loss_function to false will turn off the
    /// application of the loss function to the output of the cost function.
    pub apply_loss_function: bool,
    /// Number of threads used for the evaluation.
    pub num_threads: usize,
}

impl Default for EvaluateOptions {
    fn default() -> Self {
        Self {
            parameter_blocks: Vec::new(),
            residual_blocks: Vec::new(),
            apply_loss_function: true,
            num_threads: 1,
        }
    }
}

/// A compressed row sparse matrix used primarily for communicating the
/// Jacobian matrix to the user (C++ `CRSMatrix`).
///
/// `rows` is a `num_rows + 1` sized array that points into the `cols` and
/// `values` array. For each row i: `cols[rows[i]] ... cols[rows[i + 1] - 1]`
/// are the indices of the non-zero columns of row i, and
/// `values[rows[i]] .. values[rows[i + 1] - 1]` are the values of the
/// corresponding entries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrsMatrix {
    /// Number of rows.
    pub num_rows: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// Row pointers (`num_rows + 1` entries).
    pub rows: Vec<usize>,
    /// Column indices.
    pub cols: Vec<usize>,
    /// Values.
    pub values: Vec<f64>,
}

impl CrsMatrix {
    /// Converts to a dense matrix (handy in tests).
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut m = nalgebra::DMatrix::zeros(self.num_rows, self.num_cols);
        for r in 0..self.num_rows {
            for k in self.rows[r]..self.rows[r + 1] {
                m[(r, self.cols[k])] = self.values[k];
            }
        }
        m
    }
}

/// A non-linear least squares problem (C++ `ceres::Problem`).
///
/// ```
/// use ceres::{AutoDiffCostFunction, CostFunctor, Problem, Scalar};
///
/// struct Residual;
/// impl CostFunctor for Residual {
///     fn evaluate<T: Scalar>(&self, p: &[&[T]], r: &mut [T]) -> bool {
///         r[0] = T::constant(10.0) - p[0][0];
///         true
///     }
/// }
/// let mut problem = Problem::new();
/// let x = problem.add_parameter_block(&[0.5]);
/// problem.add_residual_block(
///     Box::new(AutoDiffCostFunction::<_, 1>::new(Residual, 1, &[1])),
///     None,
///     &[x],
/// );
/// assert_eq!(problem.num_residuals(), 1);
/// ```
pub struct Problem<'a> {
    options: ProblemOptions,
    parameter_blocks: Vec<Option<ParameterBlock<'a>>>,
    residual_blocks: Vec<Option<ResidualBlock<'a>>>,
}

impl Default for Problem<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Problem<'a> {
    /// Creates an empty problem with default options.
    pub fn new() -> Self {
        Self::with_options(ProblemOptions::default())
    }

    /// Creates an empty problem.
    pub fn with_options(options: ProblemOptions) -> Self {
        Self {
            options,
            parameter_blocks: Vec::new(),
            residual_blocks: Vec::new(),
        }
    }

    // --- internal accessors --------------------------------------------------

    pub(crate) fn parameter_block_capacity(&self) -> usize {
        self.parameter_blocks.len()
    }

    pub(crate) fn block(&self, index: usize) -> &ParameterBlock<'a> {
        self.parameter_blocks[index]
            .as_ref()
            .expect("Parameter block not found (it was removed).")
    }

    pub(crate) fn block_mut(&mut self, index: usize) -> &mut ParameterBlock<'a> {
        self.parameter_blocks[index]
            .as_mut()
            .expect("Parameter block not found (it was removed).")
    }

    pub(crate) fn residual(&self, index: usize) -> &ResidualBlock<'a> {
        self.residual_blocks[index]
            .as_ref()
            .expect("Residual block not found (it was removed).")
    }

    pub(crate) fn parameter_blocks_iter(
        &self,
    ) -> impl Iterator<Item = (usize, &ParameterBlock<'a>)> {
        self.parameter_blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|b| (i, b)))
    }

    pub(crate) fn residual_blocks_iter(&self) -> impl Iterator<Item = (usize, &ResidualBlock<'a>)> {
        self.residual_blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|b| (i, b)))
    }

    fn checked_block(&self, id: ParameterBlockId) -> &ParameterBlock<'a> {
        match self.parameter_blocks.get(id.0).and_then(Option::as_ref) {
            Some(b) => b,
            None => panic!("Parameter block not found: {id:?}"),
        }
    }

    fn checked_block_mut(&mut self, id: ParameterBlockId) -> &mut ParameterBlock<'a> {
        match self.parameter_blocks.get_mut(id.0).and_then(Option::as_mut) {
            Some(b) => b,
            None => panic!("Parameter block not found: {id:?}"),
        }
    }

    fn checked_residual(&self, id: ResidualBlockId) -> &ResidualBlock<'a> {
        match self.residual_blocks.get(id.0).and_then(Option::as_ref) {
            Some(b) => b,
            None => panic!("Residual block not found: {id:?}"),
        }
    }

    // --- construction --------------------------------------------------------

    /// Add a parameter block with the given initial values to the problem
    /// and return its handle (C++ `AddParameterBlock(double* values, int
    /// size)`). The size of the block is `values.len()`.
    pub fn add_parameter_block(&mut self, values: &[f64]) -> ParameterBlockId {
        self.add_parameter_block_with_parameterization(values, None)
    }

    /// Add a parameter block with a local parameterization (C++
    /// `AddParameterBlock(double* values, int size, LocalParameterization*)`).
    /// `None` means no parameterization.
    pub fn add_parameter_block_with_parameterization(
        &mut self,
        values: &[f64],
        local_parameterization: Option<Box<dyn LocalParameterization + 'a>>,
    ) -> ParameterBlockId {
        assert!(
            !values.is_empty(),
            "Tried adding a parameter block of size zero."
        );
        let id = ParameterBlockId(self.parameter_blocks.len());
        self.parameter_blocks.push(Some(ParameterBlock {
            values: values.to_vec(),
            constant: false,
            parameterization: None,
            upper_bounds: None,
            lower_bounds: None,
        }));
        if local_parameterization.is_some() {
            self.set_parameterization(id, local_parameterization);
        }
        id
    }

    /// Add a residual block to the overall cost function. The cost function
    /// carries with it information about the sizes of the parameter blocks
    /// it expects. The function checks that these match the sizes of the
    /// parameter blocks listed in `parameter_blocks`. The program aborts if
    /// a mismatch is detected. `loss_function` can be `None`, in which case
    /// the cost of the term is just the squared norm of the residuals.
    ///
    /// The user has the option of explicitly adding the parameter blocks
    /// using `add_parameter_block`. RUST PORT NOTE: here this is required,
    /// since parameter blocks are referred to by handle.
    ///
    /// Duplicate parameter blocks in a residual are not allowed.
    pub fn add_residual_block(
        &mut self,
        cost_function: Box<dyn CostFunction + 'a>,
        loss_function: Option<Box<dyn LossFunction + 'a>>,
        parameter_blocks: &[ParameterBlockId],
    ) -> ResidualBlockId {
        let sizes = cost_function.parameter_block_sizes();
        assert_eq!(
            sizes.len(),
            parameter_blocks.len(),
            "Number of blocks input is different than the number of blocks that the cost function expects."
        );
        if !self.options.disable_all_safety_checks {
            // Check for duplicate parameter blocks.
            let mut sorted = parameter_blocks.to_vec();
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                panic!(
                    "Duplicate parameter blocks in a residual parameter are not allowed. \
                     Parameter blocks: {parameter_blocks:?}"
                );
            }
        }
        // Check that the block sizes match the block sizes expected by the
        // cost_function.
        for (i, &id) in parameter_blocks.iter().enumerate() {
            let size = self.checked_block(id).size();
            assert_eq!(
                sizes[i], size,
                "The cost function expects parameter block {i} of size {} but was given a block of size {size}",
                sizes[i]
            );
        }
        let id = ResidualBlockId(self.residual_blocks.len());
        self.residual_blocks.push(Some(ResidualBlock {
            cost_function,
            loss_function,
            parameter_blocks: parameter_blocks.to_vec(),
        }));
        id
    }

    /// Remove a residual block from the problem. Any parameters that the
    /// residual block depends on are not removed.
    pub fn remove_residual_block(&mut self, residual_block: ResidualBlockId) {
        self.checked_residual(residual_block);
        self.residual_blocks[residual_block.0] = None;
    }

    /// Remove a parameter block from the problem. Any residual terms
    /// depending on the parameter block are also removed.
    pub fn remove_parameter_block(&mut self, parameter_block: ParameterBlockId) {
        self.checked_block(parameter_block);
        for slot in &mut self.residual_blocks {
            if slot
                .as_ref()
                .is_some_and(|r| r.parameter_blocks.contains(&parameter_block))
            {
                *slot = None;
            }
        }
        self.parameter_blocks[parameter_block.0] = None;
    }

    /// Hold the indicated parameter block constant during optimization.
    pub fn set_parameter_block_constant(&mut self, id: ParameterBlockId) {
        self.checked_block_mut(id).constant = true;
    }

    /// Allow the indicated parameter block to vary during optimization.
    pub fn set_parameter_block_variable(&mut self, id: ParameterBlockId) {
        self.checked_block_mut(id).constant = false;
    }

    /// Returns true if a parameter block is set constant, and false
    /// otherwise. A parameter block may be set constant in two ways: either
    /// by calling `set_parameter_block_constant` or by associating a local
    /// parameterization with a zero dimensional tangent space with it.
    pub fn is_parameter_block_constant(&self, id: ParameterBlockId) -> bool {
        self.checked_block(id).is_constant()
    }

    /// Set the local parameterization for one of the parameter blocks
    /// (`None` removes it). Panics if the global size of the
    /// parameterization does not match the block size.
    pub fn set_parameterization(
        &mut self,
        id: ParameterBlockId,
        local_parameterization: Option<Box<dyn LocalParameterization + 'a>>,
    ) {
        let block = self.checked_block_mut(id);
        if let Some(p) = &local_parameterization {
            assert_eq!(
                p.global_size(),
                block.size(),
                "Invalid parameterization for parameter block. The parameter block has size {} \
                 while the parameterization has a global size of {}. Did you accidentally use \
                 the wrong parameter block or parameterization?",
                block.size(),
                p.global_size()
            );
            let mut jacobian = vec![0.0; p.global_size() * p.local_size()];
            assert!(
                p.compute_jacobian(&block.values, &mut jacobian),
                "Local parameterization Jacobian computation failed for x: {:?}",
                block.values
            );
        }
        block.parameterization = local_parameterization;
    }

    /// Get the local parameterization object associated with this parameter
    /// block. If there is no parameterization object associated then `None`
    /// is returned.
    pub fn get_parameterization(
        &self,
        id: ParameterBlockId,
    ) -> Option<&(dyn LocalParameterization + 'a)> {
        self.checked_block(id).parameterization.as_deref()
    }

    /// Set the lower bound for the parameter at position `index` in the
    /// parameter block corresponding to `id`. By default the lower bound is
    /// `-f64::MAX`, which is treated by the solver as the same as -infinity.
    /// Panics if the block is not in the problem or `index` is out of range.
    pub fn set_parameter_lower_bound(
        &mut self,
        id: ParameterBlockId,
        index: usize,
        lower_bound: f64,
    ) {
        self.checked_block_mut(id)
            .set_lower_bound(index, lower_bound);
    }

    /// Set the upper bound for the parameter at position `index` in the
    /// parameter block corresponding to `id`. By default the value is
    /// `f64::MAX`, which is treated by the solver as the same as infinity.
    pub fn set_parameter_upper_bound(
        &mut self,
        id: ParameterBlockId,
        index: usize,
        upper_bound: f64,
    ) {
        self.checked_block_mut(id)
            .set_upper_bound(index, upper_bound);
    }

    /// Get the lower bound for the parameter with position `index`. If the
    /// parameter is not bounded by the user, then its lower bound is
    /// `-f64::MAX`.
    pub fn get_parameter_lower_bound(&self, id: ParameterBlockId, index: usize) -> f64 {
        self.checked_block(id).lower_bound_for_parameter(index)
    }

    /// Get the upper bound for the parameter with position `index`. If the
    /// parameter is not bounded by the user, then its upper bound is
    /// `f64::MAX`.
    pub fn get_parameter_upper_bound(&self, id: ParameterBlockId, index: usize) -> f64 {
        self.checked_block(id).upper_bound_for_parameter(index)
    }

    /// The current values of a parameter block (after `solve`, the optimized
    /// values).
    pub fn parameter_block(&self, id: ParameterBlockId) -> &[f64] {
        &self.checked_block(id).values
    }

    /// Mutable access to the values of a parameter block.
    pub fn parameter_block_mut(&mut self, id: ParameterBlockId) -> &mut [f64] {
        &mut self.checked_block_mut(id).values
    }

    /// Overwrites the values of a parameter block. Panics on size mismatch.
    pub fn set_parameter_block_values(&mut self, id: ParameterBlockId, values: &[f64]) {
        self.checked_block_mut(id).values.copy_from_slice(values);
    }

    /// The size of the parameter block.
    pub fn parameter_block_size(&self, id: ParameterBlockId) -> usize {
        self.checked_block(id).size()
    }

    /// The size of local parameterization for the parameter block. If there
    /// is no local parameterization associated with this parameter block,
    /// then `parameter_block_local_size == parameter_block_size`.
    pub fn parameter_block_local_size(&self, id: ParameterBlockId) -> usize {
        self.checked_block(id).local_size()
    }

    /// Is the given parameter block present in this problem or not?
    pub fn has_parameter_block(&self, id: ParameterBlockId) -> bool {
        self.parameter_blocks.get(id.0).is_some_and(Option::is_some)
    }

    /// Number of parameter blocks in the problem. Always equals
    /// `parameter_blocks().len()`.
    pub fn num_parameter_blocks(&self) -> usize {
        self.parameter_blocks.iter().flatten().count()
    }

    /// The size of the parameter vector obtained by summing over the sizes
    /// of all the parameter blocks.
    pub fn num_parameters(&self) -> usize {
        self.parameter_blocks
            .iter()
            .flatten()
            .map(ParameterBlock::size)
            .sum()
    }

    /// Number of residual blocks in the problem.
    pub fn num_residual_blocks(&self) -> usize {
        self.residual_blocks.iter().flatten().count()
    }

    /// The size of the residual vector obtained by summing over the sizes of
    /// all of the residual blocks.
    pub fn num_residuals(&self) -> usize {
        self.residual_blocks
            .iter()
            .flatten()
            .map(|r| r.cost_function.num_residuals())
            .sum()
    }

    /// All the parameter blocks in the problem, in insertion order.
    pub fn parameter_blocks(&self) -> Vec<ParameterBlockId> {
        self.parameter_blocks_iter()
            .map(|(i, _)| ParameterBlockId(i))
            .collect()
    }

    /// All the residual blocks in the problem, in insertion order.
    pub fn residual_blocks(&self) -> Vec<ResidualBlockId> {
        self.residual_blocks_iter()
            .map(|(i, _)| ResidualBlockId(i))
            .collect()
    }

    /// The parameter blocks of a residual block.
    pub fn parameter_blocks_for_residual_block(&self, id: ResidualBlockId) -> &[ParameterBlockId] {
        &self.checked_residual(id).parameter_blocks
    }

    /// The residual blocks that depend on a parameter block.
    pub fn residual_blocks_for_parameter_block(
        &self,
        id: ParameterBlockId,
    ) -> Vec<ResidualBlockId> {
        self.checked_block(id);
        self.residual_blocks_iter()
            .filter(|(_, r)| r.parameter_blocks.contains(&id))
            .map(|(i, _)| ResidualBlockId(i))
            .collect()
    }

    /// The cost function of a residual block.
    pub fn cost_function_for_residual_block(
        &self,
        id: ResidualBlockId,
    ) -> &(dyn CostFunction + 'a) {
        self.checked_residual(id).cost_function.as_ref()
    }

    /// The loss function of a residual block (`None` for squared loss).
    pub fn loss_function_for_residual_block(
        &self,
        id: ResidualBlockId,
    ) -> Option<&(dyn LossFunction + 'a)> {
        self.checked_residual(id).loss_function.as_deref()
    }

    /// Evaluates the residual block, storing the scalar cost in `cost`, the
    /// residual components in `residuals`, and the jacobians between the
    /// parameters and residuals in `jacobians[i]`, in row-major order.
    ///
    /// If `residuals` is `None`, the residuals are not computed. If
    /// `jacobians` is `None`, no Jacobians are computed. If `jacobians[i]`
    /// is `None`, then the Jacobian for that parameter block is not
    /// computed. It is not okay to request the Jacobian w.r.t a parameter
    /// block that is constant.
    ///
    /// The returned cost and jacobians have had robustification and local
    /// parameterizations applied already; for example, the jacobian for a
    /// 4-dimensional quaternion parameter using the
    /// `QuaternionParameterization` is `num_residuals x 3` instead of
    /// `num_residuals x 4`.
    pub fn evaluate_residual_block(
        &self,
        id: ResidualBlockId,
        apply_loss_function: bool,
        cost: Option<&mut f64>,
        residuals: Option<&mut [f64]>,
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let rb = self.checked_residual(id);
        let mut lp_storage: Vec<Vec<f64>> = Vec::with_capacity(rb.parameter_blocks.len());
        for (i, &pb) in rb.parameter_blocks.iter().enumerate() {
            let b = self.block(pb.0);
            let wants_jacobian = jacobians.as_deref().is_some_and(|j| j[i].is_some());
            if b.is_constant() {
                if wants_jacobian {
                    log::error!(
                        "Jacobian requested for parameter block : {i}. But the parameter block is marked constant."
                    );
                    return false;
                }
                lp_storage.push(Vec::new());
            } else if let Some(p) = &b.parameterization {
                let mut jac = vec![0.0; b.size() * b.local_size()];
                assert!(
                    p.compute_jacobian(&b.values, &mut jac),
                    "Congratulations, you found a Ceres bug! Please report this error to the developers."
                );
                lp_storage.push(jac);
            } else {
                lp_storage.push(Vec::new());
            }
        }
        let args: Vec<ArgInput<'_>> = rb
            .parameter_blocks
            .iter()
            .zip(&lp_storage)
            .map(|(&pb, lp)| {
                let b = self.block(pb.0);
                ArgInput {
                    values: &b.values,
                    local_parameterization_jacobian: (!lp.is_empty()).then_some(&lp[..]),
                    local_size: b.local_size(),
                }
            })
            .collect();
        let mut dummy_cost = 0.0;
        let mut scratch = Vec::new();
        evaluate_residual_block(
            rb.cost_function.as_ref(),
            rb.loss_function.as_deref(),
            &args,
            apply_loss_function,
            cost.unwrap_or(&mut dummy_cost),
            residuals,
            jacobians,
            &mut scratch,
        )
    }

    /// Evaluate a Problem. Any of the output arguments can be `None`. Which
    /// residual blocks and parameter blocks are used is controlled by the
    /// [`EvaluateOptions`].
    ///
    /// The gradient has one entry per tangent space dimension of the
    /// selected parameter blocks (zero for constant blocks); the Jacobian
    /// has one row per residual and one column per tangent space dimension.
    ///
    /// Note: the loss function (if any) is applied to the residuals and the
    /// jacobian via the Triggs correction, unless
    /// `options.apply_loss_function` is false.
    pub fn evaluate(
        &self,
        options: &EvaluateOptions,
        cost: Option<&mut f64>,
        residuals: Option<&mut Vec<f64>>,
        gradient: Option<&mut Vec<f64>>,
        jacobian: Option<&mut CrsMatrix>,
    ) -> bool {
        if cost.is_none() && residuals.is_none() && gradient.is_none() && jacobian.is_none() {
            return true;
        }

        // If the user supplied residual blocks, then use them, otherwise take
        // the residual blocks from the underlying program.
        let residual_blocks: Vec<usize> = if options.residual_blocks.is_empty() {
            self.residual_blocks_iter().map(|(i, _)| i).collect()
        } else {
            options
                .residual_blocks
                .iter()
                .map(|&id| {
                    self.checked_residual(id);
                    id.0
                })
                .collect()
        };
        let parameter_blocks: Vec<usize> = if options.parameter_blocks.is_empty() {
            self.parameter_blocks_iter().map(|(i, _)| i).collect()
        } else {
            options
                .parameter_blocks
                .iter()
                .map(|&id| {
                    assert!(
                        self.has_parameter_block(id),
                        "No known parameter block for Problem::Evaluate::Options.parameter_blocks: {id:?}"
                    );
                    id.0
                })
                .collect()
        };

        let program = Program::new(self, &parameter_blocks, &residual_blocks);
        let mut state = vec![0.0; program.num_parameters];
        program.parameter_blocks_to_state_vector(self, &mut state);
        let num_residuals = program.num_residuals;
        let num_effective_parameters = program.num_effective_parameters;

        let mut evaluator = Evaluator::new(self, program, options.num_threads);
        let mut tmp_residuals = residuals.as_ref().map(|_| vec![0.0; num_residuals]);
        let mut tmp_gradient = gradient
            .as_ref()
            .map(|_| vec![0.0; num_effective_parameters]);
        let mut tmp_jacobian = jacobian.as_ref().map(|_| evaluator.create_jacobian());
        let mut tmp_cost = 0.0;
        let status = evaluator.evaluate(
            &state,
            options.apply_loss_function,
            &mut tmp_cost,
            tmp_residuals.as_deref_mut(),
            tmp_gradient.as_deref_mut(),
            tmp_jacobian.as_mut(),
        );
        if !status {
            return false;
        }
        if let Some(cost) = cost {
            *cost = tmp_cost;
        }
        if let (Some(out), Some(r)) = (residuals, tmp_residuals) {
            *out = r;
        }
        if let (Some(out), Some(g)) = (gradient, tmp_gradient) {
            *out = g;
        }
        if let (Some(out), Some(j)) = (jacobian, tmp_jacobian) {
            *out = block_jacobian_to_crs(&j);
        }
        true
    }
}

/// Converts the block sparse Jacobian to a compressed row matrix, with the
/// columns of each row sorted (C++ `CompressedRowJacobianWriter` +
/// `ToCRSMatrix`).
fn block_jacobian_to_crs(j: &BlockJacobian) -> CrsMatrix {
    let mut out = CrsMatrix {
        num_rows: j.num_rows,
        num_cols: j.num_cols,
        rows: Vec::with_capacity(j.num_rows + 1),
        cols: Vec::new(),
        values: Vec::new(),
    };
    out.rows.push(0);
    for row in &j.rows {
        let mut cells: Vec<_> = j.cells[row.cells.clone()].to_vec();
        cells.sort_by_key(|c| c.col_offset);
        for r in 0..row.num_rows {
            for cell in &cells {
                for c in 0..cell.num_cols {
                    out.cols.push(cell.col_offset + c);
                    out.values
                        .push(j.values[cell.value_offset + r * cell.num_cols + c]);
                }
            }
            out.rows.push(out.cols.len());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! Ported from Ceres Solver 2.0.0 internal/ceres/parameter_block_test.cc
    //! (the bounds tests; the others are covered through the public API).
    use super::*;
    use crate::local_parameterization::SubsetParameterization;

    #[test]
    fn plus_with_bounds_constraints() {
        let mut problem = Problem::new();
        let id = problem.add_parameter_block(&[1.0, 0.0]);
        problem.set_parameter_upper_bound(id, 0, 2.0);
        problem.set_parameter_lower_bound(id, 1, -1.0);
        let mut x_plus_delta = [0.0; 2];
        assert!(
            problem
                .block(id.0)
                .plus(&[1.0, 0.0], &[2.0, -10.0], &mut x_plus_delta)
        );
        assert_eq!(x_plus_delta[0], 2.0);
        assert_eq!(x_plus_delta[1], -1.0);
    }

    // RUST PORT NOTE: not upstream. The projection is applied after the
    // local parameterization's Plus.
    #[test]
    fn plus_with_parameterization_projects_after_plus() {
        let mut problem = Problem::new();
        let id = problem.add_parameter_block_with_parameterization(
            &[1.0, 5.0],
            Some(Box::new(SubsetParameterization::new(2, &[1]))),
        );
        problem.set_parameter_upper_bound(id, 0, 3.0);
        problem.set_parameter_upper_bound(id, 1, 3.0);
        let mut x_plus_delta = [0.0; 2];
        // Only coordinate 0 is in the tangent space (index 1 is held
        // constant by the subset parameterization), but both are projected.
        assert!(
            problem
                .block(id.0)
                .plus(&[1.0, 5.0], &[10.0], &mut x_plus_delta)
        );
        assert_eq!(x_plus_delta, [3.0, 3.0]);
    }

    #[test]
    fn setting_default_bound_does_not_allocate() {
        let mut problem = Problem::new();
        let id = problem.add_parameter_block(&[1.0, 0.0]);
        problem.set_parameter_upper_bound(id, 0, f64::MAX);
        problem.set_parameter_lower_bound(id, 1, -f64::MAX);
        assert!(problem.block(id.0).upper_bounds.is_none());
        assert!(problem.block(id.0).lower_bounds.is_none());
        problem.set_parameter_lower_bound(id, 1, 0.0);
        assert_eq!(
            problem.block(id.0).lower_bounds.as_deref(),
            Some(&[-f64::MAX, 0.0][..])
        );
    }
}
