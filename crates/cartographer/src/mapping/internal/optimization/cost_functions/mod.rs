//! Port of `cartographer/mapping/internal/optimization/cost_functions/`.

pub mod acceleration_cost_function_3d;
pub mod cost_helpers;
pub mod cost_helpers_impl;
pub mod landmark_cost_function_2d;
pub mod landmark_cost_function_3d;
pub mod rotation_cost_function_3d;
pub mod spa_cost_function_2d;
pub mod spa_cost_function_3d;

/// Test helpers shared by the cost function tests.
#[cfg(test)]
pub(crate) mod testing {
    use ceres::CostFunction;
    use rand::Rng;

    /// A random unit quaternion `[w, x, y, z]`.
    pub(crate) fn random_unit_quaternion(rng: &mut impl Rng) -> Vec<f64> {
        loop {
            let q: Vec<f64> = (0..4).map(|_| rng.random_range(-1.0..1.0)).collect();
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n > 0.1 {
                return q.iter().map(|x| x / n).collect();
            }
        }
    }

    /// A random vector with `size` entries in `[-scale, scale)`.
    pub(crate) fn random_vector(rng: &mut impl Rng, size: usize, scale: f64) -> Vec<f64> {
        (0..size).map(|_| rng.random_range(-scale..scale)).collect()
    }

    /// Evaluates `cost_function` at `parameters`, returning the residuals and
    /// the row-major Jacobian of every parameter block.
    pub(crate) fn evaluate(
        cost_function: &dyn CostFunction,
        parameters: &[&[f64]],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let num_residuals = cost_function.num_residuals();
        let mut residuals = vec![0.; num_residuals];
        let mut jacobians: Vec<Vec<f64>> = cost_function
            .parameter_block_sizes()
            .iter()
            .map(|&size| vec![0.; size * num_residuals])
            .collect();
        {
            let mut jacobian_refs: Vec<Option<&mut [f64]>> = jacobians
                .iter_mut()
                .map(|jacobian| Some(jacobian.as_mut_slice()))
                .collect();
            assert!(cost_function.evaluate(parameters, &mut residuals, Some(&mut jacobian_refs)));
        }
        (residuals, jacobians)
    }

    /// Checks the Jacobians of `cost_function` at `parameters` against central
    /// finite differences: `|analytic - numeric| <= tolerance * max(1, |numeric|)`.
    pub(crate) fn check_jacobians_numerically(
        cost_function: &dyn CostFunction,
        parameters: &[Vec<f64>],
        tolerance: f64,
    ) {
        const STEP: f64 = 1e-6;
        let refs: Vec<&[f64]> = parameters.iter().map(Vec::as_slice).collect();
        let (_, jacobians) = evaluate(cost_function, &refs);
        let num_residuals = cost_function.num_residuals();
        for (block, block_values) in parameters.iter().enumerate() {
            let size = block_values.len();
            for column in 0..size {
                let mut plus = parameters.to_vec();
                plus[block][column] += STEP;
                let mut minus = parameters.to_vec();
                minus[block][column] -= STEP;
                let plus_refs: Vec<&[f64]> = plus.iter().map(Vec::as_slice).collect();
                let minus_refs: Vec<&[f64]> = minus.iter().map(Vec::as_slice).collect();
                let mut r_plus = vec![0.; num_residuals];
                let mut r_minus = vec![0.; num_residuals];
                assert!(cost_function.evaluate(&plus_refs, &mut r_plus, None));
                assert!(cost_function.evaluate(&minus_refs, &mut r_minus, None));
                for row in 0..num_residuals {
                    let numeric = (r_plus[row] - r_minus[row]) / (2. * STEP);
                    let analytic = jacobians[block][row * size + column];
                    assert!(
                        (analytic - numeric).abs() <= tolerance * numeric.abs().max(1.),
                        "block {block} d r[{row}] / d p[{column}]: analytic {analytic} vs \
                         numeric {numeric}"
                    );
                }
            }
        }
    }
}
