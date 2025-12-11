use std::{env, path::Path};
use std::{fs, path::PathBuf};
use std::process::Command;

fn main() {
    let server_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inference_dir = server_dir.parent().unwrap().join("inference");
    let out_dir = env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir)
        .ancestors()
        .nth(4)  // climb back to the `target` directory
        .unwrap();
    
    let wasm_src = target_dir.join("wasm32-wasip1/release/inference.wasm");
    let wasm_dst = server_dir.join("inference.wasm");

    // Always rebuild if inference changes
    println!("cargo:rerun-if-changed={}", inference_dir.display());

    // Build the inference crate for wasm
    let status = Command::new("cargo")
        .args(["build", "-p", "inference", "--target", "wasm32-wasip1", "--release"])
        .status()
        .expect("failed to build inference wasm");

    assert!(status.success(), "inference crate failed to build");

    fs::copy(&wasm_src, &wasm_dst)
        .expect("failed to copy wasm into server build output");
}
