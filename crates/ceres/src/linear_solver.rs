// Ported from Ceres Solver 2.0.0: internal/ceres/linear_solver.h,
// internal/ceres/dense_qr_solver.cc,
// internal/ceres/dense_normal_cholesky_solver.cc,
// internal/ceres/sparse_normal_cholesky_solver.cc and
// internal/ceres/inner_product_computer.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/internal/ceres/linear_solver.h).
//
// Copyright 2017 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Linear solvers for the regularized linear least squares problem
//!
//! ```text
//!   min_x |A x - b|^2 + |D x|^2
//! ```
//!
//! used to compute the Levenberg-Marquardt step.
//!
//! RUST PORT NOTE: `DENSE_QR` and `DENSE_NORMAL_CHOLESKY` use nalgebra (C++:
//! Eigen). `SPARSE_NORMAL_CHOLESKY` forms the upper triangle of
//! `A^T A + D^T D` directly from the block structure of the Jacobian (never
//! densified) and factors it with faer's sparse Cholesky (AMD fill-reducing
//! ordering + supernodal/simplicial LLT; C++: SuiteSparse CHOLMOD). The
//! symbolic analysis is computed once per solve and reused.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, SymbolicCholesky, SymmetricOrdering, factorize_symbolic_cholesky,
};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};
use nalgebra::{DMatrix, DVector};

use crate::program::BlockJacobian;
use crate::types::{LinearSolverTerminationType, LinearSolverType};

/// Result of a linear solve (C++ `LinearSolver::Summary`).
#[derive(Clone, Debug)]
pub(crate) struct LinearSolverSummary {
    pub termination_type: LinearSolverTerminationType,
    pub num_iterations: i32,
    pub message: String,
}

/// A linear solver for `[A; D] x = [b; 0]` in the least squares sense.
pub(crate) enum LinearSolver {
    DenseQr,
    DenseNormalCholesky,
    SparseNormalCholesky(Box<SparseNormalCholeskySolver>),
}

impl LinearSolver {
    /// Creates a solver of the given type (C++ `LinearSolver::Create`).
    pub(crate) fn new(linear_solver_type: LinearSolverType) -> Self {
        match linear_solver_type {
            LinearSolverType::DenseQr => Self::DenseQr,
            LinearSolverType::DenseNormalCholesky => Self::DenseNormalCholesky,
            LinearSolverType::SparseNormalCholesky => Self::SparseNormalCholesky(Box::default()),
        }
    }

    /// Solves `min |A x - b|^2 + |diag(d) x|^2`.
    pub(crate) fn solve(
        &mut self,
        a: &BlockJacobian,
        b: &[f64],
        d: Option<&[f64]>,
        x: &mut [f64],
    ) -> LinearSolverSummary {
        match self {
            Self::DenseQr => dense_qr_solve(a, b, d, x),
            Self::DenseNormalCholesky => dense_normal_cholesky_solve(a, b, d, x),
            Self::SparseNormalCholesky(s) => s.solve(a, b, d, x),
        }
    }
}

/// C++ `DenseQRSolver::SolveUsingEigen`: Householder QR of `[A; D]`.
fn dense_qr_solve(
    a: &BlockJacobian,
    b: &[f64],
    d: Option<&[f64]>,
    x: &mut [f64],
) -> LinearSolverSummary {
    let num_rows = a.num_rows;
    let num_cols = a.num_cols;
    let augmented_num_rows = num_rows + if d.is_some() { num_cols } else { 0 };

    // Temporarily append a diagonal block to the A matrix.
    let mut lhs = DMatrix::<f64>::zeros(augmented_num_rows, num_cols);
    a.to_dense(&mut lhs);
    if let Some(d) = d {
        for i in 0..num_cols {
            lhs[(num_rows + i, i)] = d[i];
        }
    }

    // rhs = [b;0] to account for the additional rows in the lhs.
    let mut rhs = DVector::<f64>::zeros(augmented_num_rows);
    rhs.rows_mut(0, num_rows).copy_from_slice(&b[..num_rows]);

    // Solve the system.
    let qr = lhs.qr();
    qr.q_tr_mul(&mut rhs);
    let r = qr.r();
    let k = r.nrows().min(num_cols);
    let solution = r
        .view((0, 0), (k, num_cols))
        .solve_upper_triangular(&rhs.rows(0, k).into_owned());
    match solution {
        Some(s) if k == num_cols => x[..num_cols].copy_from_slice(s.as_slice()),
        // A zero on the diagonal of R: Eigen would produce non-finite values;
        // the caller detects them and treats the step as a failure.
        _ => x[..num_cols].fill(f64::NAN),
    }

    // We always succeed, since the QR solver returns the best solution it
    // can. It is the job of the caller to determine if the solution is good
    // enough or not.
    LinearSolverSummary {
        termination_type: LinearSolverTerminationType::Success,
        num_iterations: 1,
        message: "Success.".to_string(),
    }
}

/// C++ `DenseNormalCholeskySolver::SolveUsingEigen`.
fn dense_normal_cholesky_solve(
    a: &BlockJacobian,
    b: &[f64],
    d: Option<&[f64]>,
    x: &mut [f64],
) -> LinearSolverSummary {
    let num_rows = a.num_rows;
    let num_cols = a.num_cols;
    let mut dense = DMatrix::<f64>::zeros(num_rows, num_cols);
    a.to_dense(&mut dense);

    //   lhs += A'A
    let mut lhs = dense.tr_mul(&dense);
    //   rhs = A'b
    let rhs = dense.tr_mul(&DVector::from_column_slice(&b[..num_rows]));

    if let Some(d) = d {
        for i in 0..num_cols {
            lhs[(i, i)] += d[i] * d[i];
        }
    }

    match nalgebra::Cholesky::new(lhs) {
        Some(llt) => {
            let s = llt.solve(&rhs);
            x[..num_cols].copy_from_slice(s.as_slice());
            LinearSolverSummary {
                termination_type: LinearSolverTerminationType::Success,
                num_iterations: 1,
                message: "Success.".to_string(),
            }
        }
        None => LinearSolverSummary {
            termination_type: LinearSolverTerminationType::Failure,
            num_iterations: 1,
            message: "Eigen LLT decomposition failed.".to_string(),
        },
    }
}

/// Symbolic structure of the upper triangle of `A^T A` (CSC), and the
/// faer symbolic factorization.
struct NormalEquationsStructure {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// For each column block, the sorted row blocks (<= column block) of the
    /// upper triangle.
    col_block_rows: Vec<Vec<usize>>,
    /// For each column block, the prefix sums of the sizes of
    /// `col_block_rows` (offset of each row block within a column).
    col_block_row_offsets: Vec<Vec<usize>>,
    symbolic: SymbolicCholesky<usize>,
}

/// C++ `SparseNormalCholeskySolver` (with `InnerProductComputer`).
#[derive(Default)]
pub(crate) struct SparseNormalCholeskySolver {
    structure: Option<NormalEquationsStructure>,
    values: Vec<f64>,
    l_values: Vec<f64>,
    rhs: Vec<f64>,
    mem: Option<MemBuffer>,
}

impl SparseNormalCholeskySolver {
    fn analyze(a: &BlockJacobian) -> Result<NormalEquationsStructure, String> {
        let num_col_blocks = a.col_blocks.len();
        let n = a.num_cols;

        // Block sparsity pattern of the upper triangle of A^T A.
        let mut col_block_rows: Vec<Vec<usize>> = (0..num_col_blocks).map(|c| vec![c]).collect();
        for row in &a.rows {
            let cells = &a.cells[row.cells.clone()];
            for (i, ci) in cells.iter().enumerate() {
                for cj in &cells[i + 1..] {
                    let (r, c) = if ci.col_block < cj.col_block {
                        (ci.col_block, cj.col_block)
                    } else {
                        (cj.col_block, ci.col_block)
                    };
                    col_block_rows[c].push(r);
                }
            }
        }
        let mut col_block_row_offsets = Vec::with_capacity(num_col_blocks);
        for rows in &mut col_block_rows {
            rows.sort_unstable();
            rows.dedup();
        }
        for rows in &col_block_rows {
            let mut offsets = Vec::with_capacity(rows.len());
            let mut acc = 0;
            for &r in rows {
                offsets.push(acc);
                acc += a.col_blocks[r].1;
            }
            col_block_row_offsets.push(offsets);
        }

        // Scalar CSC structure (upper triangle, sorted row indices).
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for (c, rows) in col_block_rows.iter().enumerate() {
            let (_, size) = a.col_blocks[c];
            for b in 0..size {
                for &r in rows {
                    let (r_offset, r_size) = a.col_blocks[r];
                    let count = if r == c { b + 1 } else { r_size };
                    row_idx.extend(r_offset..r_offset + count);
                }
                col_ptr.push(row_idx.len());
            }
        }

        let symbolic_a = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
        let symbolic = factorize_symbolic_cholesky(
            symbolic_a,
            Side::Upper,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|e| format!("Sparse Cholesky symbolic factorization failed: {e:?}"))?;

        Ok(NormalEquationsStructure {
            n,
            col_ptr,
            row_idx,
            col_block_rows,
            col_block_row_offsets,
            symbolic,
        })
    }

    fn solve(
        &mut self,
        a: &BlockJacobian,
        b: &[f64],
        d: Option<&[f64]>,
        x: &mut [f64],
    ) -> LinearSolverSummary {
        let num_cols = a.num_cols;
        x[..num_cols].fill(0.0);

        if self.structure.is_none() {
            match Self::analyze(a) {
                Ok(s) => self.structure = Some(s),
                Err(message) => {
                    return LinearSolverSummary {
                        termination_type: LinearSolverTerminationType::FatalError,
                        num_iterations: 1,
                        message,
                    };
                }
            }
        }
        let s = self.structure.as_ref().expect("structure");

        // rhs = A^T b.
        self.rhs.clear();
        self.rhs.resize(num_cols, 0.0);
        a.left_multiply(b, &mut self.rhs);

        // values = upper triangle of A^T A (+ D^T D).
        self.values.clear();
        self.values.resize(s.row_idx.len(), 0.0);
        for row in &a.rows {
            let cells = &a.cells[row.cells.clone()];
            let nr = row.num_rows;
            for (i, ci) in cells.iter().enumerate() {
                for cj in &cells[i..] {
                    let (lo, hi) = if ci.col_block <= cj.col_block {
                        (ci, cj)
                    } else {
                        (cj, ci)
                    };
                    let c = hi.col_block;
                    let k = s.col_block_rows[c]
                        .binary_search(&lo.col_block)
                        .expect("block in structure");
                    let row_offset_in_col = s.col_block_row_offsets[c][k];
                    let lo_vals = &a.values[lo.value_offset..lo.value_offset + nr * lo.num_cols];
                    let hi_vals = &a.values[hi.value_offset..hi.value_offset + nr * hi.num_cols];
                    let diagonal = lo.col_block == hi.col_block;
                    for bcol in 0..hi.num_cols {
                        let col_start = s.col_ptr[hi.col_offset + bcol] + row_offset_in_col;
                        let arows = if diagonal { bcol + 1 } else { lo.num_cols };
                        for arow in 0..arows {
                            let mut sum = 0.0;
                            for r in 0..nr {
                                sum += lo_vals[r * lo.num_cols + arow]
                                    * hi_vals[r * hi.num_cols + bcol];
                            }
                            self.values[col_start + arow] += sum;
                        }
                    }
                }
            }
        }
        if let Some(d) = d {
            for j in 0..num_cols {
                // The diagonal entry is the last one of each column.
                self.values[s.col_ptr[j + 1] - 1] += d[j] * d[j];
            }
        }

        // Factor and solve.
        let par = Par::Seq;
        let req = StackReq::any_of(&[
            s.symbolic
                .factorize_numeric_llt_scratch::<f64>(par, Default::default()),
            s.symbolic.solve_in_place_scratch::<f64>(1, par),
        ]);
        let mem = match &mut self.mem {
            Some(mem) => mem,
            None => self.mem.insert(MemBuffer::new(req)),
        };
        // `MemBuffer` cannot be resized; it is created once per solve
        // structure, so the requirement never changes.
        let stack = MemStack::new(mem);
        self.l_values.clear();
        self.l_values.resize(s.symbolic.len_val(), 0.0);
        let a_upper = SparseColMatRef::new(
            SymbolicSparseColMatRef::new_checked(s.n, s.n, &s.col_ptr, None, &s.row_idx),
            &self.values,
        );
        let llt = match s.symbolic.factorize_numeric_llt(
            &mut self.l_values,
            a_upper,
            Side::Upper,
            LltRegularization::default(),
            par,
            &mut *stack,
            Default::default(),
        ) {
            Ok(llt) => llt,
            Err(_) => {
                return LinearSolverSummary {
                    termination_type: LinearSolverTerminationType::Failure,
                    num_iterations: 1,
                    message: "Sparse Cholesky factorization failed: matrix not positive definite."
                        .to_string(),
                };
            }
        };
        x[..num_cols].copy_from_slice(&self.rhs);
        llt.solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(&mut x[..num_cols], num_cols, 1),
            par,
            stack,
        );
        LinearSolverSummary {
            termination_type: LinearSolverTerminationType::Success,
            num_iterations: 1,
            message: "Success.".to_string(),
        }
    }
}
