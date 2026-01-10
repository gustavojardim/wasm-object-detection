#!/usr/bin/env bash
set -e

echo "=== 1) Wasmtime version ==="
wasmtime -V || { echo "ERROR: wasmtime not found"; exit 1; }

echo
echo "=== 2) Check if WASI-NN is supported ==="
# Modern Wasmtime exposes WASI-NN under -S help, NOT --help
if wasmtime -S help | grep -q "nn\[=y|n\]"; then
    echo "WASI-NN support: OK"
else
    echo "ERROR: WASI-NN support not detected in this Wasmtime build."
    exit 1
fi

echo
echo "=== 3) Test WASI-NN graph preloading support ==="
if wasmtime -S help | grep -q "nn-graph"; then
    echo "nn-graph option: OK"
else
    echo "ERROR: nn-graph flag missing. This Wasmtime build may lack PyTorch backend support."
    exit 1
fi

echo
echo "=== 4) libtorch detection ==="
if [ -d "$HOME/libtorch" ]; then
    echo "Found libtorch at: $HOME/libtorch"
    echo "libtorch found"
else
    echo "WARNING: libtorch not found at ~/libtorch"
fi

echo
echo "=== 5) CUDA + GPU check ==="
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | head -n 3
else
    echo "ERROR: nvcc not found"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -n 5
else
    echo "WARNING: nvidia-smi not available"
fi

echo
echo "=== 6) Checking model files ==="
MODEL_DIR="./models"
CPU_MODEL="$MODEL_DIR/yolov8n.torchscript"
CUDA_MODEL="$MODEL_DIR/yolov8n_cuda.torchscript"

if [ -f "$CPU_MODEL" ]; then
    echo "[OK] CPU model present"
else
    echo "[ERROR] CPU model missing: $CPU_MODEL"
    exit 1
fi

if [ -f "$CUDA_MODEL" ]; then
    echo "[OK] CUDA model present"
else
    echo "[WARN] CUDA model missing: $CUDA_MODEL"
fi

echo
echo "=== 7) Wasmtime --mapdir test ==="
TEST=$(wasmtime -S nn=y --mapdir /models=./models -V 2>&1 || true)
if echo "$TEST" | grep -q "wasmtime"; then
    echo "[OK] wasmtime accepts --mapdir and -S nn=y"
else
    echo "ERROR: wasmtime failed with mapdir or nn support"
    exit 1
fi

echo
echo "=== 8) Minimal nn-graph preloading test ==="
# This does not load a real model; it just verifies wasmtime accepts the flag.
if wasmtime -S nn -S 'nn-graph=torch::./models' -V >/dev/null 2>&1; then
    echo "[OK] Wasmtime accepts torch nn-graph preload"
else
    echo "ERROR: Wasmtime rejected the torch nn-graph preload."
    echo "Your Wasmtime likely lacks PyTorch backend support."
    exit 1
fi

echo
echo "=== Setup Verified Successfully ==="
exit 0