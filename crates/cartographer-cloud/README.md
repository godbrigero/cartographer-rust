# cartographer-cloud

Pure-Rust port of `cartographer/cloud` from [Google Cartographer], built on
[`cartographer`](https://crates.io/crates/cartographer), `tonic` and `tokio`:

- a gRPC `MapBuilderService` server (`MapBuilderServer`) and the
  `cartographer_grpc_server` binary,
- client stubs implementing Cartographer's `MapBuilderInterface`,
  `PoseGraphInterface` and `TrajectoryBuilderInterface` over gRPC,
- the edge-to-cloud local trajectory uploader,
- optional Prometheus metrics (`prometheus` feature, on by default).

The wire format and RPC paths are identical to the C++ implementation, so
Rust and C++ clients and servers interoperate. Plaintext HTTP/2 only (no TLS
dependencies).

```sh
cargo install cartographer-cloud
cartographer_grpc_server --configuration_directory <dir> \
    --configuration_basename map_builder_server.lua
```

## License

Apache License 2.0, like Cartographer (see `LICENSE` and `NOTICE`). This is
an independent port and is not affiliated with or endorsed by Google or the
Cartographer project.

[Google Cartographer]: https://github.com/cartographer-project/cartographer
