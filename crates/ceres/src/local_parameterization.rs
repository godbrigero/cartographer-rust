// Ported from Ceres Solver 2.0.0: include/ceres/local_parameterization.h,
// internal/ceres/local_parameterization.cc and
// include/ceres/internal/householder_vector.h
// (https://github.com/ceres-solver/ceres-solver/blob/2.0.0/include/ceres/local_parameterization.h).
//
// Copyright 2019 Google Inc. All rights reserved. (New BSD license.)
// Original authors: keir@google.com (Keir Mierle), sameeragarwal@google.com
// (Sameer Agarwal)

//! Local parameterizations (manifolds) of parameter blocks.
//!
//! Sometimes the parameters `x` can overparameterize a problem. In that case
//! it is desirable to choose a parameterization for the block itself to
//! remove the null directions of the cost. More generally, if `x` lies on a
//! manifold of a smaller dimension than the ambient space that it is
//! embedded in, then it is numerically and computationally more effective to
//! optimize it using a parameterization that lives in the tangent space of
//! that manifold at each point.
//!
//! We define a function
//!
//! ```text
//!   x_plus_delta = Plus(x, delta),
//! ```
//!
//! where `x_plus_delta` has the same size as `x`, and `delta` is of size
//! less than or equal to `x`. The function `Plus` generalizes the definition
//! of vector addition. Thus it satisfies the identity `Plus(x, 0) = x`.
//!
//! An example that occurs commonly in Structure from Motion problems is when
//! camera rotations are parameterized using Quaternion. There, it is useful
//! to only make updates orthogonal to that 4-vector defining the
//! quaternion. One way to do this is to let delta be a 3 dimensional vector
//! and define Plus to be
//!
//! ```text
//!   Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x
//! ```
//!
//! The multiplication between the two 4-vectors on the RHS is the standard
//! quaternion product.
//!
//! RUST PORT NOTE: `LineParameterization` (internal/line_parameterization.h)
//! is not ported (unused by Cartographer).

use std::sync::Arc;

use crate::rotation::quaternion_product;

/// Defines the function `Plus` and its Jacobian which is needed to compute
/// the Jacobian of f w.r.t delta.
pub trait LocalParameterization: Send + Sync {
    /// Generalization of the addition operation,
    ///
    /// ```text
    ///   x_plus_delta = Plus(x, delta)
    /// ```
    ///
    /// with the condition that `Plus(x, 0) = x`.
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool;

    /// The jacobian of `Plus(x, delta)` w.r.t delta at `delta = 0`.
    ///
    /// `jacobian` is a row-major `global_size() x local_size()` matrix.
    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool;

    /// `local_matrix = global_matrix * jacobian`
    ///
    /// `global_matrix` is a `num_rows x global_size()` row major matrix.
    /// `local_matrix` is a `num_rows x local_size()` row major matrix.
    /// `jacobian(x)` is the matrix returned by `compute_jacobian` at `x`.
    fn multiply_by_jacobian(
        &self,
        x: &[f64],
        num_rows: usize,
        global_matrix: &[f64],
        local_matrix: &mut [f64],
    ) -> bool {
        let local_size = self.local_size();
        if local_size == 0 {
            return true;
        }
        let global_size = self.global_size();
        let mut jacobian = vec![0.0; global_size * local_size];
        if !self.compute_jacobian(x, &mut jacobian) {
            return false;
        }
        for r in 0..num_rows {
            for c in 0..local_size {
                let mut sum = 0.0;
                for k in 0..global_size {
                    sum += global_matrix[r * global_size + k] * jacobian[k * local_size + c];
                }
                local_matrix[r * local_size + c] = sum;
            }
        }
        true
    }

    /// Size of x.
    fn global_size(&self) -> usize;

    /// Size of delta.
    fn local_size(&self) -> usize;
}

impl<P: LocalParameterization + ?Sized> LocalParameterization for Arc<P> {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        (**self).plus(x, delta, x_plus_delta)
    }
    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        (**self).compute_jacobian(x, jacobian)
    }
    fn multiply_by_jacobian(
        &self,
        x: &[f64],
        num_rows: usize,
        global_matrix: &[f64],
        local_matrix: &mut [f64],
    ) -> bool {
        (**self).multiply_by_jacobian(x, num_rows, global_matrix, local_matrix)
    }
    fn global_size(&self) -> usize {
        (**self).global_size()
    }
    fn local_size(&self) -> usize {
        (**self).local_size()
    }
}

impl<P: LocalParameterization + ?Sized> LocalParameterization for Box<P> {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        (**self).plus(x, delta, x_plus_delta)
    }
    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        (**self).compute_jacobian(x, jacobian)
    }
    fn multiply_by_jacobian(
        &self,
        x: &[f64],
        num_rows: usize,
        global_matrix: &[f64],
        local_matrix: &mut [f64],
    ) -> bool {
        (**self).multiply_by_jacobian(x, num_rows, global_matrix, local_matrix)
    }
    fn global_size(&self) -> usize {
        (**self).global_size()
    }
    fn local_size(&self) -> usize {
        (**self).local_size()
    }
}

/// Identity Parameterization: `Plus(x, delta) = x + delta`.
#[derive(Clone, Copy, Debug)]
pub struct IdentityParameterization {
    size: usize,
}

impl IdentityParameterization {
    /// Creates the identity parameterization of dimension `size > 0`.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Check failed: size > 0");
        Self { size }
    }
}

impl LocalParameterization for IdentityParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        for i in 0..self.size {
            x_plus_delta[i] = x[i] + delta[i];
        }
        true
    }
    fn compute_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        let n = self.size;
        for r in 0..n {
            for c in 0..n {
                jacobian[r * n + c] = if r == c { 1.0 } else { 0.0 };
            }
        }
        true
    }
    fn multiply_by_jacobian(
        &self,
        _x: &[f64],
        num_cols: usize,
        global_matrix: &[f64],
        local_matrix: &mut [f64],
    ) -> bool {
        let n = num_cols * self.global_size();
        local_matrix[..n].copy_from_slice(&global_matrix[..n]);
        true
    }
    fn global_size(&self) -> usize {
        self.size
    }
    fn local_size(&self) -> usize {
        self.size
    }
}

/// Hold a subset of the parameters inside a parameter block constant.
#[derive(Clone, Debug)]
pub struct SubsetParameterization {
    local_size: usize,
    constancy_mask: Vec<bool>,
}

impl SubsetParameterization {
    /// Creates the parameterization of a block of `size` parameters, where
    /// the parameters at the indices in `constant_parameters` are held
    /// constant. Panics on out of range or duplicate indices.
    pub fn new(size: usize, constant_parameters: &[usize]) -> Self {
        let mut constancy_mask = vec![false; size];
        if constant_parameters.is_empty() {
            return Self {
                local_size: size,
                constancy_mask,
            };
        }
        let mut constant = constant_parameters.to_vec();
        constant.sort_unstable();
        assert!(
            *constant.last().expect("non-empty") < size,
            "Indices indicating constant parameter must be less than the size of the parameter block."
        );
        assert!(
            constant.windows(2).all(|w| w[0] != w[1]),
            "The set of constant parameters cannot contain duplicates"
        );
        for &i in constant_parameters {
            constancy_mask[i] = true;
        }
        Self {
            local_size: size - constant_parameters.len(),
            constancy_mask,
        }
    }
}

impl LocalParameterization for SubsetParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let mut j = 0;
        for (i, &constant) in self.constancy_mask.iter().enumerate() {
            if constant {
                x_plus_delta[i] = x[i];
            } else {
                x_plus_delta[i] = x[i] + delta[j];
                j += 1;
            }
        }
        true
    }
    fn compute_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        if self.local_size == 0 {
            return true;
        }
        let global_size = self.global_size();
        jacobian[..global_size * self.local_size].fill(0.0);
        let mut j = 0;
        for (i, &constant) in self.constancy_mask.iter().enumerate() {
            if !constant {
                jacobian[i * self.local_size + j] = 1.0;
                j += 1;
            }
        }
        true
    }
    fn multiply_by_jacobian(
        &self,
        _x: &[f64],
        num_cols: usize,
        global_matrix: &[f64],
        local_matrix: &mut [f64],
    ) -> bool {
        if self.local_size == 0 {
            return true;
        }
        let global_size = self.global_size();
        for col in 0..num_cols {
            let mut j = 0;
            for (i, &constant) in self.constancy_mask.iter().enumerate() {
                if !constant {
                    local_matrix[col * self.local_size + j] = global_matrix[col * global_size + i];
                    j += 1;
                }
            }
        }
        true
    }
    fn global_size(&self) -> usize {
        self.constancy_mask.len()
    }
    fn local_size(&self) -> usize {
        self.local_size
    }
}

/// `Plus(x, delta) = [cos(|delta|), sin(|delta|) delta / |delta|] * x` with
/// `*` being the quaternion multiplication operator. Here we assume that the
/// first element of the quaternion vector is the real (cos theta) part.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuaternionParameterization;

impl LocalParameterization for QuaternionParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let norm_delta = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if norm_delta > 0.0 {
            let sin_delta_by_delta = norm_delta.sin() / norm_delta;
            let q_delta = [
                norm_delta.cos(),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2],
            ];
            quaternion_product(&q_delta, x, x_plus_delta);
        } else {
            x_plus_delta[..4].copy_from_slice(&x[..4]);
        }
        true
    }
    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[0] = -x[1];
        jacobian[1] = -x[2];
        jacobian[2] = -x[3];
        jacobian[3] = x[0];
        jacobian[4] = x[3];
        jacobian[5] = -x[2];
        jacobian[6] = -x[3];
        jacobian[7] = x[0];
        jacobian[8] = x[1];
        jacobian[9] = x[2];
        jacobian[10] = -x[1];
        jacobian[11] = x[0];
        true
    }
    fn global_size(&self) -> usize {
        4
    }
    fn local_size(&self) -> usize {
        3
    }
}

/// Implements the quaternion local parameterization for Eigen's
/// representation of the quaternion. Eigen uses a different internal memory
/// layout for the elements of the quaternion than what is commonly used.
/// Specifically, Eigen stores the elements in memory as `[x, y, z, w]` where
/// the real part is last whereas it is typically stored first. (nalgebra's
/// `Quaternion::coords` uses the same `[x, y, z, w]` layout.)
///
/// `Plus(x, delta) = [sin(|delta|) delta / |delta|, cos(|delta|)] * x` with
/// `*` being the quaternion multiplication operator.
#[derive(Clone, Copy, Debug, Default)]
pub struct EigenQuaternionParameterization;

impl LocalParameterization for EigenQuaternionParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let norm_delta = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if norm_delta > 0.0 {
            let sin_delta_by_delta = norm_delta.sin() / norm_delta;
            // Note, in the constructor w is first.
            let (aw, ax, ay, az) = (
                norm_delta.cos(),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2],
            );
            let (bx, by, bz, bw) = (x[0], x[1], x[2], x[3]);
            // x_plus_delta = delta_q * x (Eigen's quat_product).
            x_plus_delta[3] = aw * bw - ax * bx - ay * by - az * bz;
            x_plus_delta[0] = aw * bx + ax * bw + ay * bz - az * by;
            x_plus_delta[1] = aw * by + ay * bw + az * bx - ax * bz;
            x_plus_delta[2] = aw * bz + az * bw + ax * by - ay * bx;
        } else {
            x_plus_delta[..4].copy_from_slice(&x[..4]);
        }
        true
    }
    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[0] = x[3];
        jacobian[1] = x[2];
        jacobian[2] = -x[1];
        jacobian[3] = -x[2];
        jacobian[4] = x[3];
        jacobian[5] = x[0];
        jacobian[6] = x[1];
        jacobian[7] = -x[0];
        jacobian[8] = x[3];
        jacobian[9] = -x[0];
        jacobian[10] = -x[1];
        jacobian[11] = -x[2];
        true
    }
    fn global_size(&self) -> usize {
        4
    }
    fn local_size(&self) -> usize {
        3
    }
}

/// Algorithm 5.1.1 from 'Matrix Computations' by Golub et al. (Johns Hopkins
/// Studies in Mathematical Sciences) but using the nth element of the input
/// vector as pivot instead of first. This computes the vector v with
/// v(n) = 1 and beta such that H = I - beta * v * v^T is orthogonal and
/// H * x = ||x||_2 * e_n.
fn compute_householder_vector(x: &[f64], v: &mut [f64]) -> f64 {
    let n = x.len();
    assert!(n > 1);
    assert_eq!(n, v.len());

    let sigma: f64 = x[..n - 1].iter().map(|a| a * a).sum();
    v.copy_from_slice(x);
    v[n - 1] = 1.0;

    let mut beta = 0.0;
    let x_pivot = x[n - 1];

    if sigma <= f64::EPSILON {
        if x_pivot < 0.0 {
            beta = 2.0;
        }
        return beta;
    }

    let mu = (x_pivot * x_pivot + sigma).sqrt();
    let v_pivot = if x_pivot <= 0.0 {
        x_pivot - mu
    } else {
        -sigma / (x_pivot + mu)
    };

    beta = 2.0 * v_pivot * v_pivot / (sigma + v_pivot * v_pivot);

    for vi in &mut v[..n - 1] {
        *vi /= v_pivot;
    }
    beta
}

/// This provides a parameterization for homogeneous vectors which are
/// commonly used in Structure for Motion problems. One example where they are
/// used is in representing points whose triangulation is ill-conditioned.
/// Here it is advantageous to use an over-parameterization since homogeneous
/// vectors can represent points at infinity.
///
/// The plus operator is defined as
/// `Plus(x, delta) = [sin(0.5 * |delta|) * delta / |delta|, cos(0.5 * |delta|)] * x`
/// with `*` defined as an operator which applies the update orthogonal to x
/// to remain on the sphere. We assume that the last element of x is the
/// scalar component. The size of the homogeneous vector is required to be
/// greater than 1.
#[derive(Clone, Copy, Debug)]
pub struct HomogeneousVectorParameterization {
    size: usize,
}

impl HomogeneousVectorParameterization {
    /// Creates the parameterization for vectors of dimension `size > 1`.
    pub fn new(size: usize) -> Self {
        assert!(
            size > 1,
            "The size of the homogeneous vector needs to be greater than 1."
        );
        Self { size }
    }
}

impl LocalParameterization for HomogeneousVectorParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let n = self.size;
        let x = &x[..n];
        let delta = &delta[..n - 1];
        let norm_delta = delta.iter().map(|d| d * d).sum::<f64>().sqrt();

        if norm_delta == 0.0 {
            x_plus_delta[..n].copy_from_slice(x);
            return true;
        }

        // Map the delta from the minimum representation to the over
        // parameterized homogeneous vector. See section A6.9.2 on page 624 of
        // Hartley & Zisserman (2nd Edition) for a detailed description. Note
        // there is a typo on Page 625, line 4 so check the book errata.
        let norm_delta_div_2 = 0.5 * norm_delta;
        let sin_delta_by_delta = norm_delta_div_2.sin() / norm_delta_div_2;

        let mut y = vec![0.0; n];
        for i in 0..n - 1 {
            y[i] = 0.5 * sin_delta_by_delta * delta[i];
        }
        y[n - 1] = norm_delta_div_2.cos();

        let mut v = vec![0.0; n];
        let beta = compute_householder_vector(x, &mut v);

        // Apply the delta update to remain on the unit sphere. See section
        // A6.9.3 on page 625 of Hartley & Zisserman (2nd Edition) for a
        // detailed description.
        let x_norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let vty: f64 = v.iter().zip(&y).map(|(a, b)| a * b).sum();
        for i in 0..n {
            x_plus_delta[i] = x_norm * (y[i] - v[i] * (beta * vty));
        }
        true
    }

    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        let n = self.size;
        let x = &x[..n];
        let mut v = vec![0.0; n];
        let beta = compute_householder_vector(x, &mut v);

        // The Jacobian is equal to J = 0.5 * H.leftCols(size_ - 1) where H is
        // the Householder matrix (H = I - beta * v * v').
        let x_norm = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let local = n - 1;
        for i in 0..local {
            for r in 0..n {
                let mut value = -0.5 * beta * v[i] * v[r];
                if r == i {
                    value += 0.5;
                }
                jacobian[r * local + i] = value * x_norm;
            }
        }
        true
    }

    fn global_size(&self) -> usize {
        self.size
    }
    fn local_size(&self) -> usize {
        self.size - 1
    }
}

/// Construct a local parameterization by taking the Cartesian product of a
/// number of other local parameterizations. This is useful, when a parameter
/// block is the cartesian product of two or more manifolds. For example the
/// parameters of a camera consist of a rotation and a translation, i.e.,
/// SO(3) x R^3.
///
/// ```
/// use ceres::{IdentityParameterization, LocalParameterization, ProductParameterization,
///             QuaternionParameterization};
/// let product = ProductParameterization::new(vec![
///     Box::new(QuaternionParameterization),
///     Box::new(IdentityParameterization::new(3)),
/// ]);
/// assert_eq!((product.global_size(), product.local_size()), (7, 6));
/// ```
pub struct ProductParameterization {
    local_params: Vec<Box<dyn LocalParameterization>>,
    local_size: usize,
    global_size: usize,
    buffer_size: usize,
}

impl ProductParameterization {
    /// Creates the product of at least two parameterizations.
    pub fn new(local_params: Vec<Box<dyn LocalParameterization>>) -> Self {
        assert!(
            local_params.len() >= 2,
            "At least two local parameterizations must be specified."
        );
        let mut local_size = 0;
        let mut global_size = 0;
        let mut buffer_size = 0;
        for param in &local_params {
            buffer_size = buffer_size.max(param.local_size() * param.global_size());
            global_size += param.global_size();
            local_size += param.local_size();
        }
        Self {
            local_params,
            local_size,
            global_size,
            buffer_size,
        }
    }
}

impl LocalParameterization for ProductParameterization {
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let mut x_cursor = 0;
        let mut delta_cursor = 0;
        for param in &self.local_params {
            if !param.plus(
                &x[x_cursor..],
                &delta[delta_cursor..],
                &mut x_plus_delta[x_cursor..],
            ) {
                return false;
            }
            delta_cursor += param.local_size();
            x_cursor += param.global_size();
        }
        true
    }

    fn compute_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        let total_local = self.local_size;
        jacobian[..self.global_size * total_local].fill(0.0);
        let mut buffer = vec![0.0; self.buffer_size];

        let mut x_cursor = 0;
        let mut delta_cursor = 0;
        for param in &self.local_params {
            let local_size = param.local_size();
            let global_size = param.global_size();

            if !param.compute_jacobian(&x[x_cursor..], &mut buffer) {
                return false;
            }
            for r in 0..global_size {
                for c in 0..local_size {
                    jacobian[(x_cursor + r) * total_local + delta_cursor + c] =
                        buffer[r * local_size + c];
                }
            }
            delta_cursor += local_size;
            x_cursor += global_size;
        }
        true
    }

    fn global_size(&self) -> usize {
        self.global_size
    }
    fn local_size(&self) -> usize {
        self.local_size
    }
}
