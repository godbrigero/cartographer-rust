//! The cloud `.proto` files must stay byte-identical to the C++ ones (checked
//! when the C++ source tree is next to the workspace).

use std::path::PathBuf;

#[test]
fn cloud_protos_are_byte_identical_to_cpp() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The C++ checkout is a sibling of the Rust workspace (or of one of its
    // ancestors, for worktrees).
    let Some(cpp_dir) = manifest_dir
        .ancestors()
        .map(|dir| dir.join("cartographer/cartographer/cloud/proto"))
        .find(|dir| dir.is_dir())
    else {
        eprintln!("C++ tree not found; skipping");
        return;
    };
    for name in [
        "map_builder_server_options.proto",
        "map_builder_service.proto",
    ] {
        let ours = std::fs::read(
            manifest_dir
                .join("proto/cartographer/cloud/proto")
                .join(name),
        )
        .unwrap();
        let cpp = std::fs::read(cpp_dir.join(name)).unwrap();
        assert_eq!(ours, cpp, "{name} differs from the C++ file");
    }
}
