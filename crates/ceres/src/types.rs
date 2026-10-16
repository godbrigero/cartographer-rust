// Ported from Ceres Solver 2.0.0: include/ceres/types.h and
// internal/ceres/types.cc
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/types.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original author: sameeragarwal@google.com (Sameer Agarwal)

//! Enums shared across the solver, with their Ceres string names
//! (`Display` gives e.g. `"DENSE_QR"`, like `LinearSolverTypeToString`).
//!
//! RUST PORT NOTE: only the variants implemented by this port are present
//! (e.g. no Schur or iterative linear solvers, no line search minimizer, no
//! dogleg). C++ `SCREAMING_CASE` enumerators become CamelCase variants.

use std::fmt;

macro_rules! ceres_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident { $($(#[$vmeta:meta])* $variant:ident => $s:literal),* $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant),*
        }

        impl $name {
            /// The Ceres name of the enumerator (e.g. `"DENSE_QR"`).
            pub fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $s),*
                }
            }

            /// Parses the Ceres name of the enumerator (C++
            /// `StringTo<Enum>`); case insensitive.
            pub fn from_ceres_str(s: &str) -> Option<Self> {
                let upper = s.to_ascii_uppercase();
                match upper.as_str() {
                    $($s => Some($name::$variant),)*
                    _ => None,
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

ceres_enum! {
    /// Linear solvers used to compute the trust region step.
    pub enum LinearSolverType {
        /// Solve the normal equations using a dense Cholesky solver.
        DenseNormalCholesky => "DENSE_NORMAL_CHOLESKY",
        /// Solve the normal equations using a dense QR solver.
        DenseQr => "DENSE_QR",
        /// Solve the normal equations using a sparse Cholesky solver.
        SparseNormalCholesky => "SPARSE_NORMAL_CHOLESKY",
    }
}

ceres_enum! {
    /// Sparse linear algebra backends.
    ///
    /// RUST PORT NOTE: the only available backend is the pure-Rust `faer`
    /// sparse Cholesky (with AMD ordering); the C++ names are kept so that
    /// reports look familiar, but selecting them is an invalid configuration.
    pub enum SparseLinearAlgebraLibraryType {
        /// Pure-Rust `faer` sparse Cholesky (default in this port).
        Faer => "FAER",
        /// High performance sparse Cholesky factorization (not available).
        SuiteSparse => "SUITE_SPARSE",
        /// A lightweight replacement for SuiteSparse (not available).
        CxSparse => "CX_SPARSE",
        /// Eigen's sparse linear algebra routines (not available).
        EigenSparse => "EIGEN_SPARSE",
        /// Apple's Accelerate framework (not available).
        AccelerateSparse => "ACCELERATE_SPARSE",
        /// No sparse linear solver should be used.
        NoSparse => "NO_SPARSE",
    }
}

ceres_enum! {
    /// Dense linear algebra backends.
    pub enum DenseLinearAlgebraLibraryType {
        /// RUST PORT NOTE: `EIGEN` maps to `nalgebra`.
        Eigen => "EIGEN",
        /// System LAPACK (not available in this port).
        Lapack => "LAPACK",
    }
}

ceres_enum! {
    /// Minimizer type. Only the trust region minimizer is ported.
    pub enum MinimizerType {
        /// Trust region minimizer.
        TrustRegion => "TRUST_REGION",
    }
}

ceres_enum! {
    /// Trust region strategy. Only Levenberg-Marquardt is ported.
    pub enum TrustRegionStrategyType {
        /// Levenberg-Marquardt.
        LevenbergMarquardt => "LEVENBERG_MARQUARDT",
    }
}

ceres_enum! {
    /// How the solver logs progress.
    pub enum LoggingType {
        /// No logging.
        Silent => "SILENT",
        /// Log each iteration (`log::debug!`, or stdout if
        /// `minimizer_progress_to_stdout`).
        PerMinimizerIteration => "PER_MINIMIZER_ITERATION",
    }
}

ceres_enum! {
    /// Termination status of the solver.
    pub enum TerminationType {
        /// Minimizer terminated because one of the convergence criterion set
        /// by the user was satisfied.
        ///
        /// 1.  (new_cost - old_cost) < function_tolerance * old_cost;
        /// 2.  max_i |gradient_i| < gradient_tolerance
        /// 3.  |step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)
        ///
        /// The user's parameter blocks will be updated with the solution.
        Convergence => "CONVERGENCE",
        /// The solver ran for maximum number of iterations or maximum amount
        /// of time specified by the user, but none of the convergence
        /// criterion specified by the user were met. The user's parameter
        /// blocks will be updated with the solution found so far.
        NoConvergence => "NO_CONVERGENCE",
        /// The minimizer terminated because of an error. The user's parameter
        /// blocks will not be updated.
        Failure => "FAILURE",
        /// Using an IterationCallback object, user code can control the
        /// minimizer. The following enums indicate that the user code was
        /// responsible for termination.
        ///
        /// Minimizer terminated successfully because a user IterationCallback
        /// returned SOLVER_TERMINATE_SUCCESSFULLY.
        ///
        /// The user's parameter blocks will be updated with the solution.
        UserSuccess => "USER_SUCCESS",
        /// Minimizer terminated because because a user IterationCallback
        /// returned SOLVER_ABORT.
        ///
        /// The user's parameter blocks will not be updated.
        UserFailure => "USER_FAILURE",
    }
}

ceres_enum! {
    /// Enums used by the IterationCallback instances to indicate to the
    /// solver whether it should continue solving, the user detected an error
    /// or the solution is good enough and the solver should terminate.
    pub enum CallbackReturnType {
        /// Continue solving to next iteration.
        SolverContinue => "SOLVER_CONTINUE",
        /// Terminate solver, and do not update the parameter blocks upon
        /// return. Unless the user has set
        /// Solver:Options:::update_state_every_iteration, in which case the
        /// state would have been updated every iteration anyways.
        /// Solver::Summary::termination_type is set to USER_ABORT.
        SolverAbort => "SOLVER_ABORT",
        /// Terminate solver, update state and return
        /// Solver::Summary::termination_type = USER_SUCCESS.
        SolverTerminateSuccessfully => "SOLVER_TERMINATE_SUCCESSFULLY",
    }
}

/// Status of a linear solve (internal/ceres/linear_solver.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearSolverTerminationType {
    /// Termination criterion was met.
    Success,
    /// Solver ran for max_num_iterations and terminated before the
    /// termination tolerance could be satisfied.
    #[allow(dead_code)]
    NoConvergence,
    /// Solver was terminated due to numerical problems, generally due to the
    /// linear system being poorly conditioned.
    Failure,
    /// Solver failed with a fatal error that cannot be recovered from, e.g.
    /// CHOLMOD ran out of memory when computing the symbolic or numeric
    /// factorization or an underlying library was called with the wrong
    /// arguments.
    FatalError,
}
