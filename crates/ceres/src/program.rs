// Ported from Ceres Solver 2.0.0: internal/ceres/program.cc,
// internal/ceres/program_evaluator.h, internal/ceres/block_sparse_matrix.cc
// and internal/ceres/block_jacobian_writer.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/program.cc).
//
// Copyright 2015 Google Inc. All rights reserved. (New BSD license.)
// Original author: keir@google.com (Keir Mierle)

//! The internal `Program` (an ordered view of parameter and residual
//! blocks), the block-sparse Jacobian and the `Evaluator` computing cost,
//! residuals, gradient and Jacobian at a state vector.
//!
//! RUST PORT NOTE: Ceres' `ProgramEvaluator` sums per-thread costs and
//! gradients, which makes results depend on the thread schedule. Here each
//! thread writes per-residual-block costs into disjoint slots, and costs and
//! the gradient (`J^T r`) are reduced sequentially afterwards, so results are
//! bit-for-bit independent of `num_threads`.

use std::ops::Range;
use std::time::Instant;

use crate::problem::Problem;
use crate::residual_block::{
    ArgInput, MAX_ARGS_ON_STACK, evaluate_residual_block, invalidate_array, is_array_valid,
};

/// A parameter block as seen by a [`Program`].
#[derive(Clone, Debug)]
pub(crate) struct ProgramBlock {
    /// Index of the block in the problem.
    pub problem_index: usize,
    /// Global size.
    pub size: usize,
    /// Tangent space size.
    pub local_size: usize,
    /// Offset in the state vector.
    pub state_offset: usize,
    /// Offset in the delta (tangent) vector / Jacobian columns.
    pub delta_offset: usize,
    /// Whether the block varies (has Jacobian columns).
    pub variable: bool,
    /// Offset of the local parameterization Jacobian (if variable and
    /// parameterized).
    pub lp_offset: Option<usize>,
}

/// A parameter block argument of a residual block in a [`Program`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProgramArg {
    /// Index of the block in the problem.
    pub problem_index: usize,
    /// Position of the block in the program if it is variable.
    pub position: Option<usize>,
}

/// A residual block as seen by a [`Program`].
#[derive(Clone, Debug)]
pub(crate) struct ProgramResidual {
    /// Index of the block in the problem.
    pub problem_index: usize,
    pub num_residuals: usize,
    pub residual_offset: usize,
    pub args: Range<usize>,
    /// Range of the Jacobian values (cells of this row block, in argument
    /// order).
    pub cell_values: Range<usize>,
}

/// An ordered set of parameter and residual blocks (C++ `Program`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Program {
    pub blocks: Vec<ProgramBlock>,
    pub residuals: Vec<ProgramResidual>,
    pub args: Vec<ProgramArg>,
    pub num_parameters: usize,
    pub num_effective_parameters: usize,
    pub num_residuals: usize,
    pub num_jacobian_values: usize,
    pub lp_storage_size: usize,
}

impl Program {
    /// Builds a program with the given problem parameter blocks (in column
    /// order) and residual blocks (in row order). Problem parameter blocks
    /// not in `parameter_blocks` are treated as constant.
    pub(crate) fn new(
        problem: &Problem<'_>,
        parameter_blocks: &[usize],
        residual_blocks: &[usize],
    ) -> Self {
        let mut position = vec![None; problem.parameter_block_capacity()];
        let mut blocks = Vec::with_capacity(parameter_blocks.len());
        let mut state_offset = 0;
        let mut delta_offset = 0;
        let mut lp_offset = 0;
        for (p, &index) in parameter_blocks.iter().enumerate() {
            let pb = problem.block(index);
            let size = pb.size();
            let local_size = pb.local_size();
            let variable = !pb.is_constant();
            let lp = if variable && pb.parameterization.is_some() {
                let o = lp_offset;
                lp_offset += size * local_size;
                Some(o)
            } else {
                None
            };
            if variable {
                position[index] = Some(p);
            }
            blocks.push(ProgramBlock {
                problem_index: index,
                size,
                local_size,
                state_offset,
                delta_offset,
                variable,
                lp_offset: lp,
            });
            state_offset += size;
            delta_offset += local_size;
        }

        let mut residuals = Vec::with_capacity(residual_blocks.len());
        let mut args = Vec::new();
        let mut residual_offset = 0;
        let mut value_offset = 0;
        for &index in residual_blocks {
            let rb = problem.residual(index);
            let num_residuals = rb.cost_function.num_residuals();
            let args_start = args.len();
            let cell_start = value_offset;
            for &pb in &rb.parameter_blocks {
                let pos = position[pb.0];
                if let Some(p) = pos {
                    value_offset += num_residuals * blocks[p].local_size;
                }
                args.push(ProgramArg {
                    problem_index: pb.0,
                    position: pos,
                });
            }
            residuals.push(ProgramResidual {
                problem_index: index,
                num_residuals,
                residual_offset,
                args: args_start..args.len(),
                cell_values: cell_start..value_offset,
            });
            residual_offset += num_residuals;
        }

        Self {
            blocks,
            residuals,
            args,
            num_parameters: state_offset,
            num_effective_parameters: delta_offset,
            num_residuals: residual_offset,
            num_jacobian_values: value_offset,
            lp_storage_size: lp_offset,
        }
    }

    /// The reduced program used by the solver (C++
    /// `Program::CreateReducedProgram` / `RemoveFixedBlocks`): residual
    /// blocks whose parameter blocks are all constant are removed (their cost
    /// is returned as the fixed cost), and parameter blocks that are constant
    /// or not used by any remaining residual block are removed.
    ///
    /// Returns `Err(message)` if a removed residual block fails to evaluate.
    pub(crate) fn create_reduced_program(problem: &Problem<'_>) -> Result<(Self, f64), String> {
        let mut used = vec![false; problem.parameter_block_capacity()];
        let mut active_residuals = Vec::new();
        let mut fixed_cost = 0.0;
        let mut scratch = Vec::new();
        for (i, rb) in problem.residual_blocks_iter() {
            let mut all_constant = true;
            for &pb in &rb.parameter_blocks {
                if !problem.block(pb.0).is_constant() {
                    all_constant = false;
                    used[pb.0] = true;
                }
            }
            if !all_constant {
                active_residuals.push(i);
                continue;
            }

            // The residual is constant and will be removed, so its cost is
            // added to the variable fixed_cost.
            let mut cost = 0.0;
            let args: Vec<ArgInput<'_>> = rb
                .parameter_blocks
                .iter()
                .map(|pb| {
                    let b = problem.block(pb.0);
                    ArgInput {
                        values: &b.values,
                        local_parameterization_jacobian: None,
                        local_size: b.local_size(),
                    }
                })
                .collect();
            if !evaluate_residual_block(
                rb.cost_function.as_ref(),
                rb.loss_function.as_deref(),
                &args,
                true,
                &mut cost,
                None,
                None,
                &mut scratch,
            ) {
                return Err(format!(
                    "Evaluation of the residual {} failed during removal of fixed residual blocks.",
                    active_residuals.len()
                ));
            }
            fixed_cost += cost;
        }
        let parameter_blocks: Vec<usize> = problem
            .parameter_blocks_iter()
            .map(|(i, _)| i)
            .filter(|&i| used[i])
            .collect();
        Ok((
            Self::new(problem, &parameter_blocks, &active_residuals),
            fixed_cost,
        ))
    }

    /// Number of parameter blocks.
    pub(crate) fn num_parameter_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Number of residual blocks.
    pub(crate) fn num_residual_blocks(&self) -> usize {
        self.residuals.len()
    }

    /// Copies the problem's values of the program parameter blocks into a
    /// state vector (C++ `ParameterBlocksToStateVector`).
    pub(crate) fn parameter_blocks_to_state_vector(
        &self,
        problem: &Problem<'_>,
        state: &mut [f64],
    ) {
        for b in &self.blocks {
            state[b.state_offset..b.state_offset + b.size]
                .copy_from_slice(&problem.block(b.problem_index).values);
        }
    }

    /// Copies a state vector back into the problem's variable parameter
    /// blocks (C++ `StateVectorToParameterBlocks` +
    /// `CopyParameterBlockStateToUserState`).
    pub(crate) fn state_vector_to_parameter_blocks(
        &self,
        problem: &mut Problem<'_>,
        state: &[f64],
    ) {
        for b in &self.blocks {
            if b.variable {
                problem
                    .block_mut(b.problem_index)
                    .values
                    .copy_from_slice(&state[b.state_offset..b.state_offset + b.size]);
            }
        }
    }

    /// `state_plus_delta = Plus(state, delta)` for every parameter block
    /// (C++ `Program::Plus`).
    pub(crate) fn plus(
        &self,
        problem: &Problem<'_>,
        state: &[f64],
        delta: &[f64],
        state_plus_delta: &mut [f64],
    ) -> bool {
        for b in &self.blocks {
            let x = &state[b.state_offset..b.state_offset + b.size];
            let d = &delta[b.delta_offset..b.delta_offset + b.local_size];
            let out = &mut state_plus_delta[b.state_offset..b.state_offset + b.size];
            match &problem.block(b.problem_index).parameterization {
                Some(p) => {
                    if !p.plus(x, d, out) {
                        return false;
                    }
                }
                None => {
                    for i in 0..b.size {
                        out[i] = x[i] + d[i];
                    }
                }
            }
        }
        true
    }
}

/// A cell of a [`BlockJacobian`]: a dense row-major block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JacobianCell {
    /// Column block (program parameter block position).
    pub col_block: usize,
    /// First column.
    pub col_offset: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// Offset of the values.
    pub value_offset: usize,
}

/// A row block of a [`BlockJacobian`].
#[derive(Clone, Debug)]
pub(crate) struct JacobianRow {
    pub row_offset: usize,
    pub num_rows: usize,
    pub cells: Range<usize>,
}

/// Block sparse Jacobian (C++ `BlockSparseMatrix` as written by the
/// `BlockJacobianWriter`): one row block per residual block, one dense
/// row-major cell per variable parameter block of the residual block.
#[derive(Clone, Debug)]
pub(crate) struct BlockJacobian {
    pub num_rows: usize,
    pub num_cols: usize,
    pub rows: Vec<JacobianRow>,
    pub cells: Vec<JacobianCell>,
    /// Column block sizes and offsets.
    pub col_blocks: Vec<(usize, usize)>,
    pub values: Vec<f64>,
}

impl BlockJacobian {
    /// Creates the (zero) Jacobian with the structure of `program`.
    pub(crate) fn new(program: &Program) -> Self {
        let mut rows = Vec::with_capacity(program.residuals.len());
        let mut cells = Vec::new();
        for r in &program.residuals {
            let start = cells.len();
            let mut value_offset = r.cell_values.start;
            for arg in &program.args[r.args.clone()] {
                if let Some(p) = arg.position {
                    let b = &program.blocks[p];
                    cells.push(JacobianCell {
                        col_block: p,
                        col_offset: b.delta_offset,
                        num_cols: b.local_size,
                        value_offset,
                    });
                    value_offset += r.num_residuals * b.local_size;
                }
            }
            rows.push(JacobianRow {
                row_offset: r.residual_offset,
                num_rows: r.num_residuals,
                cells: start..cells.len(),
            });
        }
        Self {
            num_rows: program.num_residuals,
            num_cols: program.num_effective_parameters,
            rows,
            cells,
            col_blocks: program
                .blocks
                .iter()
                .map(|b| (b.delta_offset, b.local_size))
                .collect(),
            values: vec![0.0; program.num_jacobian_values],
        }
    }

    /// `x = sum_i J(i, :)^2` (C++ `SquaredColumnNorm`).
    pub(crate) fn squared_column_norm(&self, x: &mut [f64]) {
        x[..self.num_cols].fill(0.0);
        for row in &self.rows {
            for cell in &self.cells[row.cells.clone()] {
                let v = &self.values
                    [cell.value_offset..cell.value_offset + row.num_rows * cell.num_cols];
                for r in 0..row.num_rows {
                    for c in 0..cell.num_cols {
                        let m = v[r * cell.num_cols + c];
                        x[cell.col_offset + c] += m * m;
                    }
                }
            }
        }
    }

    /// `J = J * diag(scale)` (C++ `ScaleColumns`).
    pub(crate) fn scale_columns(&mut self, scale: &[f64]) {
        for row in &self.rows {
            for cell in &self.cells[row.cells.clone()] {
                let v = &mut self.values
                    [cell.value_offset..cell.value_offset + row.num_rows * cell.num_cols];
                for r in 0..row.num_rows {
                    for c in 0..cell.num_cols {
                        v[r * cell.num_cols + c] *= scale[cell.col_offset + c];
                    }
                }
            }
        }
    }

    /// `y += J x` (C++ `RightMultiply`).
    pub(crate) fn right_multiply(&self, x: &[f64], y: &mut [f64]) {
        for row in &self.rows {
            for cell in &self.cells[row.cells.clone()] {
                let v = &self.values
                    [cell.value_offset..cell.value_offset + row.num_rows * cell.num_cols];
                for r in 0..row.num_rows {
                    let mut sum = 0.0;
                    for c in 0..cell.num_cols {
                        sum += v[r * cell.num_cols + c] * x[cell.col_offset + c];
                    }
                    y[row.row_offset + r] += sum;
                }
            }
        }
    }

    /// `y += J^T x` (C++ `LeftMultiply`).
    pub(crate) fn left_multiply(&self, x: &[f64], y: &mut [f64]) {
        for row in &self.rows {
            for cell in &self.cells[row.cells.clone()] {
                let v = &self.values
                    [cell.value_offset..cell.value_offset + row.num_rows * cell.num_cols];
                for c in 0..cell.num_cols {
                    let mut sum = 0.0;
                    for r in 0..row.num_rows {
                        sum += v[r * cell.num_cols + c] * x[row.row_offset + r];
                    }
                    y[cell.col_offset + c] += sum;
                }
            }
        }
    }

    /// Writes the Jacobian into the top `num_rows` rows of the dense matrix
    /// `dense` (which must be zero there).
    pub(crate) fn to_dense(&self, dense: &mut nalgebra::DMatrix<f64>) {
        for row in &self.rows {
            for cell in &self.cells[row.cells.clone()] {
                let v = &self.values
                    [cell.value_offset..cell.value_offset + row.num_rows * cell.num_cols];
                for r in 0..row.num_rows {
                    for c in 0..cell.num_cols {
                        dense[(row.row_offset + r, cell.col_offset + c)] = v[r * cell.num_cols + c];
                    }
                }
            }
        }
    }
}

/// Call statistics of the evaluator (C++ `ExecutionSummary` entries
/// "Evaluator::Residual" and "Evaluator::Jacobian").
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EvaluatorStatistics {
    pub residual_calls: usize,
    pub residual_time: f64,
    pub jacobian_calls: usize,
    pub jacobian_time: f64,
}

/// Evaluates a [`Program`] of a [`Problem`] (C++ `ProgramEvaluator` with a
/// block sparse jacobian writer).
pub(crate) struct Evaluator<'p, 'a> {
    problem: &'p Problem<'a>,
    pub program: Program,
    chunks: Vec<Range<usize>>,
    lp_jacobians: Vec<f64>,
    block_costs: Vec<f64>,
    residual_scratch: Vec<f64>,
    jacobian_scratch: Option<BlockJacobian>,
    thread_scratch: Vec<Vec<f64>>,
    pub statistics: EvaluatorStatistics,
}

impl<'p, 'a> Evaluator<'p, 'a> {
    /// Creates an evaluator using up to `num_threads` threads.
    pub(crate) fn new(problem: &'p Problem<'a>, program: Program, num_threads: usize) -> Self {
        let chunks = split_into_chunks(&program, num_threads.max(1));
        Self {
            problem,
            lp_jacobians: vec![0.0; program.lp_storage_size],
            block_costs: vec![0.0; program.residuals.len()],
            residual_scratch: Vec::new(),
            jacobian_scratch: None,
            thread_scratch: vec![Vec::new(); chunks.len().max(1)],
            chunks,
            program,
            statistics: EvaluatorStatistics::default(),
        }
    }

    /// A zero Jacobian with the structure of the program.
    pub(crate) fn create_jacobian(&self) -> BlockJacobian {
        BlockJacobian::new(&self.program)
    }

    /// `state_plus_delta = Plus(state, delta)`.
    pub(crate) fn plus(&self, state: &[f64], delta: &[f64], state_plus_delta: &mut [f64]) -> bool {
        self.program
            .plus(self.problem, state, delta, state_plus_delta)
    }

    /// Evaluate the cost, residuals, gradient (in the tangent space) and
    /// Jacobian at `state`. Any output may be `None`. Returns `false` if any
    /// residual block or local parameterization fails to evaluate.
    pub(crate) fn evaluate(
        &mut self,
        state: &[f64],
        apply_loss_function: bool,
        cost: &mut f64,
        residuals: Option<&mut [f64]>,
        gradient: Option<&mut [f64]>,
        jacobian: Option<&mut BlockJacobian>,
    ) -> bool {
        let start = Instant::now();
        let is_jacobian_call = gradient.is_some() || jacobian.is_some();
        let ok = self.evaluate_impl(
            state,
            apply_loss_function,
            cost,
            residuals,
            gradient,
            jacobian,
        );
        let elapsed = start.elapsed().as_secs_f64();
        if is_jacobian_call {
            self.statistics.jacobian_calls += 1;
            self.statistics.jacobian_time += elapsed;
        } else {
            self.statistics.residual_calls += 1;
            self.statistics.residual_time += elapsed;
        }
        ok
    }

    fn evaluate_impl(
        &mut self,
        state: &[f64],
        apply_loss_function: bool,
        cost: &mut f64,
        residuals: Option<&mut [f64]>,
        gradient: Option<&mut [f64]>,
        jacobian: Option<&mut BlockJacobian>,
    ) -> bool {
        let problem = self.problem;
        let program = &self.program;

        // The parameters are stateful, so set the state before evaluating
        // (this updates the local parameterization Jacobians).
        for b in &program.blocks {
            let Some(offset) = b.lp_offset else { continue };
            let pb = problem.block(b.problem_index);
            let parameterization = pb.parameterization.as_ref().expect("parameterized block");
            let x = &state[b.state_offset..b.state_offset + b.size];
            let jac = &mut self.lp_jacobians[offset..offset + b.size * b.local_size];
            invalidate_array(jac);
            if !parameterization.compute_jacobian(x, jac) {
                log::warn!("Local parameterization Jacobian computation failed for x: {x:?}");
                return false;
            }
            if !is_array_valid(jac) {
                log::warn!(
                    "Local parameterization Jacobian computation returned an invalid matrix for x: {x:?}\n Jacobian matrix : {jac:?}"
                );
                return false;
            }
        }

        let need_gradient = gradient.is_some();
        // Residuals are needed internally for the gradient.
        if residuals.is_none() && need_gradient {
            self.residual_scratch.resize(program.num_residuals, 0.0);
        }
        let residuals_out: Option<&mut [f64]> = match residuals {
            Some(r) => Some(r),
            None if need_gradient => Some(&mut self.residual_scratch[..]),
            None => None,
        };

        // Jacobian storage is needed internally for the gradient.
        if jacobian.is_none() && need_gradient && self.jacobian_scratch.is_none() {
            self.jacobian_scratch = Some(BlockJacobian::new(program));
        }
        let jacobian_out: Option<&mut BlockJacobian> = match jacobian {
            Some(j) => Some(j),
            None if need_gradient => self.jacobian_scratch.as_mut(),
            None => None,
        };

        let mut residuals_out = residuals_out;
        let mut jacobian_out = jacobian_out;
        if let Some(r) = residuals_out.as_deref_mut() {
            r[..program.num_residuals].fill(0.0);
        }
        if let Some(j) = jacobian_out.as_deref_mut() {
            j.values.fill(0.0);
        }

        let lp_jacobians = &self.lp_jacobians;
        let ok = if self.chunks.len() <= 1 {
            evaluate_chunk(
                problem,
                program,
                state,
                lp_jacobians,
                apply_loss_function,
                0..program.residuals.len(),
                residuals_out.as_deref_mut(),
                jacobian_out.as_deref_mut().map(|j| &mut j.values[..]),
                &mut self.block_costs,
                &mut self.thread_scratch[0],
            )
        } else {
            // Split the outputs into disjoint per-chunk slices.
            let mut residual_rest = residuals_out.as_deref_mut();
            let mut value_rest = jacobian_out.as_deref_mut().map(|j| &mut j.values[..]);
            let mut cost_rest = &mut self.block_costs[..];
            let mut work = Vec::with_capacity(self.chunks.len());
            let mut residual_pos = 0;
            let mut value_pos = 0;
            for (chunk, scratch) in self.chunks.iter().zip(self.thread_scratch.iter_mut()) {
                let last = &program.residuals[chunk.end - 1];
                let residual_end = last.residual_offset + last.num_residuals;
                let value_end = last.cell_values.end;
                let r = residual_rest.take().map(|r| {
                    let (head, tail) = r.split_at_mut(residual_end - residual_pos);
                    residual_rest = Some(tail);
                    head
                });
                let v = value_rest.take().map(|v| {
                    let (head, tail) = v.split_at_mut(value_end - value_pos);
                    value_rest = Some(tail);
                    head
                });
                residual_pos = residual_end;
                value_pos = value_end;
                let (c, tail) = std::mem::take(&mut cost_rest).split_at_mut(chunk.len());
                cost_rest = tail;
                work.push((chunk.clone(), r, v, c, scratch));
            }
            std::thread::scope(|s| {
                let handles: Vec<_> = work
                    .into_iter()
                    .map(|(range, r, v, c, scratch)| {
                        s.spawn(move || {
                            evaluate_chunk(
                                problem,
                                program,
                                state,
                                lp_jacobians,
                                apply_loss_function,
                                range,
                                r,
                                v,
                                c,
                                scratch,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("evaluation thread panicked"))
                    .fold(true, |all_ok, ok| all_ok & ok)
            })
        };
        if !ok {
            return false;
        }

        // Sum the cost (sequentially, for determinism).
        *cost = self.block_costs.iter().sum();

        // Compute the gradient, g = J^T r, if it was requested.
        if let Some(gradient) = gradient {
            let g = &mut gradient[..program.num_effective_parameters];
            g.fill(0.0);
            let residuals = residuals_out.expect("residuals for gradient");
            let jacobian = jacobian_out.expect("jacobian for gradient");
            jacobian.left_multiply(residuals, g);
        }
        true
    }
}

/// Splits the residual blocks into at most `num_threads` contiguous chunks
/// of roughly equal numbers of residuals.
fn split_into_chunks(program: &Program, num_threads: usize) -> Vec<Range<usize>> {
    let n = program.residuals.len();
    if n == 0 {
        return Vec::new();
    }
    let num_chunks = num_threads.min(n);
    if num_chunks <= 1 {
        return std::iter::once(0..n).collect();
    }
    let total = program.num_residuals.max(1);
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut start = 0;
    let mut accumulated = 0;
    for (i, r) in program.residuals.iter().enumerate() {
        accumulated += r.num_residuals;
        let target = total * (chunks.len() + 1) / num_chunks;
        if accumulated >= target && chunks.len() + 1 < num_chunks {
            chunks.push(start..i + 1);
            start = i + 1;
        }
    }
    if start < n {
        chunks.push(start..n);
    }
    chunks
}

/// Evaluates the residual blocks `range` of `program`. `residuals`, `values`
/// and `costs` are the slices of the outputs covering exactly `range`.
#[allow(clippy::too_many_arguments)]
fn evaluate_chunk(
    problem: &Problem<'_>,
    program: &Program,
    state: &[f64],
    lp_jacobians: &[f64],
    apply_loss_function: bool,
    range: Range<usize>,
    mut residuals: Option<&mut [f64]>,
    mut values: Option<&mut [f64]>,
    costs: &mut [f64],
    scratch: &mut Vec<f64>,
) -> bool {
    if range.is_empty() {
        return true;
    }
    let residual_base = program.residuals[range.start].residual_offset;
    let value_base = program.residuals[range.start].cell_values.start;
    for i in range.clone() {
        let pr = &program.residuals[i];
        let rb = problem.residual(pr.problem_index);
        let args = &program.args[pr.args.clone()];
        let num_args = args.len();

        let empty = ArgInput {
            values: &[],
            local_parameterization_jacobian: None,
            local_size: 0,
        };
        let mut stack_inputs = [empty; MAX_ARGS_ON_STACK];
        let mut heap_inputs: Vec<ArgInput<'_>>;
        let inputs: &mut [ArgInput<'_>] = if num_args <= MAX_ARGS_ON_STACK {
            &mut stack_inputs[..num_args]
        } else {
            heap_inputs = vec![empty; num_args];
            &mut heap_inputs[..]
        };
        for (input, arg) in inputs.iter_mut().zip(args) {
            *input = match arg.position {
                Some(p) => {
                    let b = &program.blocks[p];
                    ArgInput {
                        values: &state[b.state_offset..b.state_offset + b.size],
                        local_parameterization_jacobian: b
                            .lp_offset
                            .map(|o| &lp_jacobians[o..o + b.size * b.local_size]),
                        local_size: b.local_size,
                    }
                }
                None => {
                    let b = problem.block(arg.problem_index);
                    ArgInput {
                        values: &b.values,
                        local_parameterization_jacobian: None,
                        local_size: b.local_size(),
                    }
                }
            };
        }

        let block_residuals = residuals.as_deref_mut().map(|r| {
            let start = pr.residual_offset - residual_base;
            &mut r[start..start + pr.num_residuals]
        });

        let mut block_cost = 0.0;
        let ok = match values.as_deref_mut() {
            None => evaluate_residual_block(
                rb.cost_function.as_ref(),
                rb.loss_function.as_deref(),
                inputs,
                apply_loss_function,
                &mut block_cost,
                block_residuals,
                None,
                scratch,
            ),
            Some(values) => {
                let mut cells =
                    &mut values[pr.cell_values.start - value_base..pr.cell_values.end - value_base];
                let mut stack_jacobians: [Option<&mut [f64]>; MAX_ARGS_ON_STACK] =
                    [const { None }; MAX_ARGS_ON_STACK];
                let mut heap_jacobians: Vec<Option<&mut [f64]>>;
                let jacobians: &mut [Option<&mut [f64]>] = if num_args <= MAX_ARGS_ON_STACK {
                    &mut stack_jacobians[..num_args]
                } else {
                    heap_jacobians = (0..num_args).map(|_| None).collect();
                    &mut heap_jacobians[..]
                };
                for (j, arg) in jacobians.iter_mut().zip(args) {
                    if let Some(p) = arg.position {
                        let n = pr.num_residuals * program.blocks[p].local_size;
                        let (head, tail) = std::mem::take(&mut cells).split_at_mut(n);
                        cells = tail;
                        *j = Some(head);
                    }
                }
                evaluate_residual_block(
                    rb.cost_function.as_ref(),
                    rb.loss_function.as_deref(),
                    inputs,
                    apply_loss_function,
                    &mut block_cost,
                    block_residuals,
                    Some(jacobians),
                    scratch,
                )
            }
        };
        if !ok {
            return false;
        }
        costs[i - range.start] = block_cost;
    }
    true
}
