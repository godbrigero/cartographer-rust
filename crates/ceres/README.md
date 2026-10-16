# cartographer-ceres

A focused, pure-Rust port of the subset of [Ceres Solver] 2.0.0 used by
[Google Cartographer]. It is the optimization backend of the
[`cartographer`](https://crates.io/crates/cartographer) crate, but has no
dependency on it and can be used on its own. The library name is `ceres`.

- `Jet`: dual numbers for forward-mode automatic differentiation, usable with
  `nalgebra` and generic `T: Scalar` code.
- `rotation` and `cubic_interpolation` helpers.
- Cost functions (`CostFunction`, `SizedCostFunction`,
  `AutoDiffCostFunction`), loss functions (`HuberLoss`, `TolerantLoss`, ...)
  and local parameterizations (`QuaternionParameterization`, ...).
- `Problem` + `solve`: trust-region Levenberg-Marquardt with `DENSE_QR`,
  `DENSE_NORMAL_CHOLESKY` or `SPARSE_NORMAL_CHOLESKY` (via `faer`) linear
  solvers and parameter bounds.

Only what Cartographer needs is implemented (no Schur or iterative solvers,
no line-search minimizer, no dogleg). No C/C++ code, no `unsafe`.

```toml
[dependencies]
ceres = { package = "cartographer-ceres", version = "0.1" }
```

See the [API documentation](https://docs.rs/cartographer-ceres) for a
curve-fitting example and `DESIGN.md` for the design notes.

## License

Code ported from Ceres Solver is under the New BSD license (`LICENSE-CERES`,
Copyright Google Inc.); the rest is under the Apache License 2.0 (`LICENSE`),
like Cartographer. See `NOTICE`.

[Ceres Solver]: http://ceres-solver.org/
[Google Cartographer]: https://github.com/cartographer-project/cartographer
