// Ported from Ceres Solver 2.0.0: internal/ceres/loss_function_test.cc

use std::sync::Arc;

use ceres::{
    ArctanLoss, CauchyLoss, ComposedLoss, HuberLoss, LossFunction, LossFunctionWrapper, ScaledLoss,
    SoftLOneLoss, TolerantLoss, TrivialLoss, TukeyLoss,
};

#[track_caller]
fn assert_near(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} to be within {tol} of {expected} (diff {})",
        (actual - expected).abs()
    );
}

/// Helper function for testing a `LossFunction` callback.
///
/// Compares the values of rho'(s) and rho''(s) computed by the callback with
/// estimates obtained by symmetric finite differencing of rho(s).
#[track_caller]
fn assert_loss_function_is_valid(loss: &dyn LossFunction, s: f64) {
    assert!(s > 0.0);

    // Evaluate rho(s), rho'(s) and rho''(s).
    let mut rho = [0.0; 3];
    loss.evaluate(s, &mut rho);

    // Use symmetric finite differencing to estimate rho'(s) and rho''(s).
    const K_H: f64 = 1e-4;
    // Values at s + kH.
    let mut fwd = [0.0; 3];
    // Values at s - kH.
    let mut bwd = [0.0; 3];
    loss.evaluate(s + K_H, &mut fwd);
    loss.evaluate(s - K_H, &mut bwd);

    // First derivative.
    let fd_1 = (fwd[0] - bwd[0]) / (2.0 * K_H);
    assert_near(fd_1, rho[1], 1e-6);

    // Second derivative.
    let fd_2 = (fwd[0] - 2.0 * rho[0] + bwd[0]) / (K_H * K_H);
    assert_near(fd_2, rho[2], 1e-6);
}

// Try two values of the scaling a = 0.7 and 1.3 (where scaling makes sense)
// and of the squared norm s = 0.357 and 1.792
//
// Note that for the Huber loss the test exercises both code paths (i.e. both
// small and large values of s).

#[test]
fn trivial_loss() {
    assert_loss_function_is_valid(&TrivialLoss, 0.357);
    assert_loss_function_is_valid(&TrivialLoss, 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    let mut rho = [0.0; 3];
    TrivialLoss.evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], 0.0, 1e-6);
}

#[test]
fn huber_loss() {
    assert_loss_function_is_valid(&HuberLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&HuberLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&HuberLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&HuberLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    let mut rho = [0.0; 3];
    HuberLoss::new(0.7).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], 0.0, 1e-6);
}

#[test]
fn soft_l_one_loss() {
    assert_loss_function_is_valid(&SoftLOneLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&SoftLOneLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&SoftLOneLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&SoftLOneLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -1 / (2 * a^2)].
    let mut rho = [0.0; 3];
    SoftLOneLoss::new(0.7).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], -0.5 / (0.7 * 0.7), 1e-6);
}

#[test]
fn cauchy_loss() {
    assert_loss_function_is_valid(&CauchyLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&CauchyLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&CauchyLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&CauchyLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -1 / a^2].
    let mut rho = [0.0; 3];
    CauchyLoss::new(0.7).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], -1.0 / (0.7 * 0.7), 1e-6);
}

#[test]
fn arctan_loss() {
    assert_loss_function_is_valid(&ArctanLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&ArctanLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&ArctanLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&ArctanLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    let mut rho = [0.0; 3];
    ArctanLoss::new(0.7).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], 0.0, 1e-6);
}

#[test]
fn tolerant_loss() {
    assert_loss_function_is_valid(&TolerantLoss::new(0.7, 0.4), 0.357);
    assert_loss_function_is_valid(&TolerantLoss::new(0.7, 0.4), 1.792);
    assert_loss_function_is_valid(&TolerantLoss::new(0.7, 0.4), 55.5);
    assert_loss_function_is_valid(&TolerantLoss::new(1.3, 0.1), 0.357);
    assert_loss_function_is_valid(&TolerantLoss::new(1.3, 0.1), 1.792);
    assert_loss_function_is_valid(&TolerantLoss::new(1.3, 0.1), 55.5);
    // Check the value at zero is actually zero.
    let mut rho = [0.0; 3];
    TolerantLoss::new(0.7, 0.4).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    // Check that loss before and after the approximation threshold are good.
    // A threshold of 36.7 is used by the implementation.
    assert_loss_function_is_valid(&TolerantLoss::new(20.0, 1.0), 20.0 + 36.6);
    assert_loss_function_is_valid(&TolerantLoss::new(20.0, 1.0), 20.0 + 36.7);
    assert_loss_function_is_valid(&TolerantLoss::new(20.0, 1.0), 20.0 + 36.8);
    assert_loss_function_is_valid(&TolerantLoss::new(20.0, 1.0), 20.0 + 1000.0);
}

#[test]
fn tukey_loss() {
    assert_loss_function_is_valid(&TukeyLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&TukeyLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&TukeyLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&TukeyLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -2 / a^2].
    let mut rho = [0.0; 3];
    TukeyLoss::new(0.7).evaluate(0.0, &mut rho);
    assert_near(rho[0], 0.0, 1e-6);
    assert_near(rho[1], 1.0, 1e-6);
    assert_near(rho[2], -2.0 / (0.7 * 0.7), 1e-6);
}

#[test]
fn composed_loss() {
    // C++ uses DO_NOT_TAKE_OWNERSHIP of stack objects; here the components are
    // shared via Arc.
    {
        let f = Arc::new(HuberLoss::new(0.7));
        let g = Arc::new(CauchyLoss::new(1.3));
        let c = ComposedLoss::new(Box::new(Arc::clone(&f)), Box::new(Arc::clone(&g)));
        assert_loss_function_is_valid(&c, 0.357);
        assert_loss_function_is_valid(&c, 1.792);
    }
    {
        let f = Arc::new(CauchyLoss::new(0.7));
        let g = Arc::new(HuberLoss::new(1.3));
        let c = ComposedLoss::new(Box::new(Arc::clone(&f)), Box::new(Arc::clone(&g)));
        assert_loss_function_is_valid(&c, 0.357);
        assert_loss_function_is_valid(&c, 1.792);
    }
}

#[test]
fn scaled_loss() {
    // Wrap a few loss functions, and a few scale factors.
    {
        let scaled_loss = ScaledLoss::new(None, 6.0);
        assert_loss_function_is_valid(&scaled_loss, 0.323);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(TrivialLoss)), 10.0);
        assert_loss_function_is_valid(&scaled_loss, 0.357);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(HuberLoss::new(0.7))), 0.1);
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(SoftLOneLoss::new(1.3))), 0.1);
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(CauchyLoss::new(1.3))), 10.0);
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(ArctanLoss::new(1.3))), 10.0);
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
    {
        let scaled_loss = ScaledLoss::new(Some(Box::new(TolerantLoss::new(1.3, 0.1))), 10.0);
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
    {
        let scaled_loss = ScaledLoss::new(
            Some(Box::new(ComposedLoss::new(
                Box::new(HuberLoss::new(0.8)),
                Box::new(TolerantLoss::new(1.3, 0.5)),
            ))),
            10.0,
        );
        assert_loss_function_is_valid(&scaled_loss, 1.792);
    }
}

#[test]
fn loss_function_wrapper() {
    // Initialization
    let loss_function1 = HuberLoss::new(1.0);
    let loss_function_wrapper = LossFunctionWrapper::new(Some(Box::new(HuberLoss::new(1.0))));

    let s = 0.862;
    let mut rho_gold = [0.0; 3];
    let mut rho = [0.0; 3];
    loss_function1.evaluate(s, &mut rho_gold);
    loss_function_wrapper.evaluate(s, &mut rho);
    for i in 0..3 {
        assert_near(rho[i], rho_gold[i], 1e-12);
    }

    // Resetting
    let loss_function2 = HuberLoss::new(0.5);
    loss_function_wrapper.reset(Some(Box::new(HuberLoss::new(0.5))));
    loss_function_wrapper.evaluate(s, &mut rho);
    loss_function2.evaluate(s, &mut rho_gold);
    for i in 0..3 {
        assert_near(rho[i], rho_gold[i], 1e-12);
    }

    // Not taking ownership (shared via Arc).
    let loss_function3 = Arc::new(HuberLoss::new(0.3));
    loss_function_wrapper.reset(Some(Box::new(Arc::clone(&loss_function3))));
    loss_function_wrapper.evaluate(s, &mut rho);
    loss_function3.evaluate(s, &mut rho_gold);
    for i in 0..3 {
        assert_near(rho[i], rho_gold[i], 1e-12);
    }

    // Set to NULL
    let loss_function4 = TrivialLoss;
    loss_function_wrapper.reset(None);
    loss_function_wrapper.evaluate(s, &mut rho);
    loss_function4.evaluate(s, &mut rho_gold);
    for i in 0..3 {
        assert_near(rho[i], rho_gold[i], 1e-12);
    }

    // Set to NULL, not taking ownership (no ownership distinction in Rust).
    loss_function_wrapper.reset(None);
    loss_function_wrapper.evaluate(s, &mut rho);
    loss_function4.evaluate(s, &mut rho_gold);
    for i in 0..3 {
        assert_near(rho[i], rho_gold[i], 1e-12);
    }
}
