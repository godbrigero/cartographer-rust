//! A minimal HTTP endpoint serving `/metrics` (the C++ uses
//! `prometheus::Exposer` from prometheus-cpp).
//!
//! RUST PORT NOTE: new file. A tiny HTTP/1.1 responder on tokio instead of
//! an HTTP framework: it answers `GET /metrics` with the text exposition
//! format and everything else with 404, then closes the connection.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use prometheus_client::registry::Registry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

use super::family_factory::encode_registry;
use crate::internal::async_grpc::RuntimeHolder;

const MAX_REQUEST_HEADER_SIZE: usize = 8192;

/// Serves the metrics of a registry over HTTP until dropped.
pub struct Exposer {
    _runtime: RuntimeHolder,
    local_address: SocketAddr,
    shutdown_sender: Option<oneshot::Sender<()>>,
}

impl std::fmt::Debug for Exposer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exposer")
            .field("local_address", &self.local_address)
            .finish_non_exhaustive()
    }
}

impl Exposer {
    /// Binds `bind_address` (e.g. `"0.0.0.0:9100"`) and starts serving
    /// `registry` at `/metrics`.
    pub fn new(bind_address: &str, registry: Arc<Mutex<Registry>>) -> std::io::Result<Self> {
        let runtime = RuntimeHolder::new("cartographer-metrics", 1);
        let listener = runtime.block_on(tokio::net::TcpListener::bind(bind_address))?;
        let local_address = listener.local_addr()?;
        let (shutdown_sender, mut shutdown_receiver) = oneshot::channel::<()>();
        runtime.handle().spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_receiver => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { continue };
                        let registry = Arc::clone(&registry);
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, &registry).await {
                                log::warn!("Metrics request failed: {e}");
                            }
                        });
                    }
                }
            }
        });
        Ok(Self {
            _runtime: runtime,
            local_address,
            shutdown_sender: Some(shutdown_sender),
        })
    }

    /// The bound address.
    pub fn local_address(&self) -> SocketAddr {
        self.local_address
    }
}

impl Drop for Exposer {
    fn drop(&mut self) {
        if let Some(shutdown_sender) = self.shutdown_sender.take() {
            let _ = shutdown_sender.send(());
        }
    }
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    registry: &Mutex<Registry>,
) -> std::io::Result<()> {
    let mut request = Vec::new();
    let mut buffer = [0u8; 1024];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buffer).await?;
        if n == 0 || request.len() > MAX_REQUEST_HEADER_SIZE {
            break;
        }
        request.extend_from_slice(&buffer[..n]);
    }
    let request = String::from_utf8_lossy(&request);
    let mut request_line = request.lines().next().unwrap_or("").split_whitespace();
    let (method, path) = (request_line.next(), request_line.next());
    let (status, content_type, body) = if method == Some("GET")
        && path.is_some_and(|path| path == "/metrics" || path.starts_with("/metrics?"))
    {
        (
            "200 OK",
            "text/plain; version=0.0.4; charset=utf-8",
            encode_registry(registry),
        )
    } else {
        ("404 Not Found", "text/plain", "Not Found\n".to_string())
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.shutdown().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::prometheus::family_factory::FamilyFactory;
    use cartographer::metrics::family_factory::FamilyFactory as _;
    use std::io::{Read, Write};

    fn http_get(address: SocketAddr, path: &str) -> String {
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        write!(stream, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    }

    /// C++ `MetricsTest.RunExposerServer` (on an ephemeral localhost port
    /// instead of `0.0.0.0:9100`), plus an actual scrape.
    #[test]
    fn run_exposer_server() {
        let mut registry = FamilyFactory::new();
        let scores = registry
            .new_histogram_family(
                "/algorithm/scores",
                "Scores achieved",
                &cartographer::metrics::histogram::fixed_width(0.05, 20),
            )
            .add(
                &[("kind".to_string(), "score".to_string())]
                    .into_iter()
                    .collect(),
            );
        cartographer::metrics::register::register_all_metrics(&mut registry);
        let exposer = Exposer::new("127.0.0.1:0", registry.registry()).unwrap();
        for score in [-1., 0.11, 0.2, 0.5, 2.] {
            scores.observe(score);
        }
        let response = http_get(exposer.local_address(), "/metrics");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(
            response.contains("/algorithm/scores_count{kind=\"score\"} 5"),
            "{response}"
        );
        let response = http_get(exposer.local_address(), "/other");
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    }
}
