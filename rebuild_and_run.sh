#!/bin/bash
# Rebuilds the WASM object detection app and runs it with CPU and debug mode by default
set -e
cd "$(dirname "$0")/inference"
cargo build --release --target wasm32-wasip2
cd ..
wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y --dir ./models::/models target/wasm32-wasip2/release/inference.wasm --device gpu --debug --profile --udp
