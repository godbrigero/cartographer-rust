//! The parts of the C++ `async_grpc` library that `cartographer/cloud` relies
//! on, rebuilt on `tonic` + `tokio`: channel creation and
//! `WaitForConnected`, per-call deadlines, the retry strategies
//! (`CreateLimitedBackoffStrategy`, `CreateUnlimitedConstantDelayStrategy`)
//! and an owned runtime used to drive async gRPC from the synchronous
//! Cartographer interfaces.
//!
//! RUST PORT NOTE: new file (no single C++ counterpart; replaces the external
//! `async_grpc` dependency).

use std::future::Future;
use std::time::{Duration, Instant};

use tokio::runtime::{Handle, Runtime};
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};

/// Maximum gRPC message size used by server and clients (C++
/// `kMaxMessageSize = 100 MB` in `map_builder_server.cc`).
pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024;

/// A multi-threaded tokio runtime owned by a synchronous object (server,
/// stub, uploader).
///
/// Dropping it never blocks and is allowed from any thread (it uses
/// `Runtime::shutdown_background`), so owners can be dropped anywhere.
pub struct RuntimeHolder {
    runtime: Option<Runtime>,
}

impl std::fmt::Debug for RuntimeHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHolder").finish_non_exhaustive()
    }
}

impl RuntimeHolder {
    /// Creates a runtime with `worker_threads` (at least 1) worker threads
    /// named `thread_name`.
    pub fn new(thread_name: &str, worker_threads: usize) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads.max(1))
            .thread_name(thread_name)
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("Failed to create tokio runtime: {e}"));
        Self {
            runtime: Some(runtime),
        }
    }

    /// The runtime handle.
    pub fn handle(&self) -> &Handle {
        self.runtime
            .as_ref()
            .expect("runtime is only taken on drop")
            .handle()
    }

    /// Runs `future` to completion on this runtime, blocking the calling
    /// thread. Must not be called from within an async context (like all
    /// blocking Cartographer interface methods of the stubs).
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.handle().block_on(future)
    }
}

impl Drop for RuntimeHolder {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Builds a plaintext (`http://`) endpoint for `address` (`host:port`, the
/// gRPC C++ target syntax used in the Lua options).
pub fn create_endpoint(address: &str) -> Result<Endpoint, Status> {
    let uri = if address.contains("://") {
        address.to_string()
    } else {
        format!("http://{address}")
    };
    Endpoint::from_shared(uri)
        .map(|endpoint| endpoint.connect_timeout(Duration::from_secs(10)))
        .map_err(|e| Status::invalid_argument(format!("Invalid address '{address}': {e}")))
}

/// Creates a lazily connecting channel (C++ `grpc::CreateChannel`); it
/// (re)connects on demand. Must be called inside the runtime's context.
pub fn create_channel(handle: &Handle, endpoint: &Endpoint) -> Channel {
    let _guard = handle.enter();
    endpoint.connect_lazy()
}

/// Waits until a connection to `endpoint` can be established or `deadline`
/// passes (C++ `Channel::WaitForConnected`). Returns whether it connected.
pub async fn wait_for_connected(endpoint: &Endpoint, deadline: Instant) -> bool {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        if let Ok(Ok(_channel)) = tokio::time::timeout(deadline - now, endpoint.connect()).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// When and how often a failed unary call is retried (C++
/// `async_grpc::RetryStrategy`).
#[derive(Clone, Debug, PartialEq)]
pub enum RetryStrategy {
    /// No retries (C++ default `nullptr` strategy).
    None,
    /// C++ `CreateLimitedBackoffStrategy(min_delay, backoff_factor,
    /// max_attempts)`: retry while `failed_attempts < max_attempts`, waiting
    /// `backoff_factor^(failed_attempts - 1) * min_delay`.
    LimitedBackoff {
        /// Delay after the first failure.
        min_delay: Duration,
        /// Multiplier per further failure.
        backoff_factor: f64,
        /// Maximum number of failed attempts.
        max_attempts: u32,
    },
    /// C++ `CreateUnlimitedConstantDelayStrategy(delay,
    /// unrecoverable_codes)`: retry forever with a constant delay unless the
    /// status code is one of `unrecoverable_codes`.
    UnlimitedConstantDelay {
        /// Delay between attempts.
        delay: Duration,
        /// Codes that are not retried.
        unrecoverable_codes: Vec<Code>,
    },
}

impl RetryStrategy {
    /// Returns the delay before the next attempt, or `None` to give up.
    fn next_delay(&self, failed_attempts: u32, status: &Status) -> Option<Duration> {
        match self {
            RetryStrategy::None => None,
            RetryStrategy::LimitedBackoff {
                min_delay,
                backoff_factor,
                max_attempts,
            } => (failed_attempts < *max_attempts)
                .then(|| min_delay.mul_f64(backoff_factor.powi(failed_attempts as i32 - 1))),
            RetryStrategy::UnlimitedConstantDelay {
                delay,
                unrecoverable_codes,
            } => (!unrecoverable_codes.contains(&status.code())).then_some(*delay),
        }
    }
}

/// Runs the unary call produced by `call` and retries it according to
/// `strategy` (C++ `async_grpc::RetryWithStrategy`). `stop` is checked
/// before every retry so that unlimited strategies terminate on shutdown.
///
/// RUST PORT NOTE: the `stop` predicate is an addition; the C++ unlimited
/// strategy can block shutdown forever.
pub async fn call_with_retry<T, F, Fut>(
    strategy: &RetryStrategy,
    stop: impl Fn() -> bool,
    mut call: F,
) -> Result<T, Status>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    let mut failed_attempts = 0;
    loop {
        match call().await {
            Ok(response) => return Ok(response),
            Err(status) => {
                failed_attempts += 1;
                let Some(delay) = strategy.next_delay(failed_attempts, &status) else {
                    return Err(status);
                };
                if stop() {
                    return Err(status);
                }
                log::info!(
                    "RPC failed ({}: {}), retrying in {:?}.",
                    status.code(),
                    status.message(),
                    delay
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Wraps `message` into a `tonic::Request` with an optional deadline (C++
/// `async_grpc::Client(channel, timeout)`).
pub fn request_with_timeout<T>(message: T, timeout: Option<Duration>) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    if let Some(timeout) = timeout {
        request.set_timeout(timeout);
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn limited_backoff_delays_and_attempts() {
        let strategy = RetryStrategy::LimitedBackoff {
            min_delay: Duration::from_millis(500),
            backoff_factor: 2.0,
            max_attempts: 5,
        };
        let status = Status::unavailable("x");
        assert_eq!(
            strategy.next_delay(1, &status),
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            strategy.next_delay(2, &status),
            Some(Duration::from_millis(1000))
        );
        assert_eq!(
            strategy.next_delay(4, &status),
            Some(Duration::from_millis(4000))
        );
        assert_eq!(strategy.next_delay(5, &status), None);
    }

    #[test]
    fn unlimited_constant_delay_stops_on_unrecoverable_codes() {
        let strategy = RetryStrategy::UnlimitedConstantDelay {
            delay: Duration::from_secs(1),
            unrecoverable_codes: vec![Code::NotFound],
        };
        assert_eq!(
            strategy.next_delay(100, &Status::internal("x")),
            Some(Duration::from_secs(1))
        );
        assert_eq!(strategy.next_delay(1, &Status::not_found("x")), None);
        assert_eq!(
            RetryStrategy::None.next_delay(1, &Status::internal("x")),
            None
        );
    }

    #[test]
    fn call_with_retry_retries_until_success() {
        let runtime = RuntimeHolder::new("retry-test", 1);
        let attempts = AtomicU32::new(0);
        let strategy = RetryStrategy::LimitedBackoff {
            min_delay: Duration::from_millis(1),
            backoff_factor: 2.0,
            max_attempts: 5,
        };
        let result = runtime.block_on(call_with_retry(
            &strategy,
            || false,
            || {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    if attempt < 2 {
                        Err(Status::unavailable("not yet"))
                    } else {
                        Ok(attempt)
                    }
                }
            },
        ));
        assert_eq!(result.unwrap(), 2);
        let result: Result<(), Status> = runtime.block_on(call_with_retry(
            &strategy,
            || false,
            || async { Err(Status::unavailable("never")) },
        ));
        assert_eq!(result.unwrap_err().code(), Code::Unavailable);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn endpoint_adds_scheme() {
        assert_eq!(
            create_endpoint("localhost:50051")
                .unwrap()
                .uri()
                .to_string(),
            "http://localhost:50051/"
        );
        assert!(create_endpoint("not a uri").is_err());
    }
}
