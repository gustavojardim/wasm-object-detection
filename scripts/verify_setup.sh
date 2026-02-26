#!/usr/bin/env bash
set -e

LIBTORCH_DIR=""
if [ -d "$HOME/libtorch" ]; then
    LIBTORCH_DIR="$HOME/libtorch"
elif [ -d "/opt/libtorch" ]; then
    LIBTORCH_DIR="/opt/libtorch"
fi

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
if [ -n "$LIBTORCH_DIR" ]; then
    echo "Found libtorch at: $LIBTORCH_DIR"
    if [ -f "$LIBTORCH_DIR/build-version" ]; then
        LIBTORCH_VERSION="$(cat "$LIBTORCH_DIR/build-version")"
        echo "libtorch version: $LIBTORCH_VERSION"
    else
        echo "WARNING: libtorch build-version file not found"
        LIBTORCH_VERSION="unknown"
    fi
else
    echo "WARNING: libtorch not found at ~/libtorch or /opt/libtorch"
    LIBTORCH_VERSION="missing"
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
    GPU_QUERY=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_QUERY" ]; then
        echo "GPU details: $GPU_QUERY"
        GPU_COMPUTE_CAP=$(echo "$GPU_QUERY" | head -n1 | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
    else
        echo "WARNING: could not query GPU compute capability"
        GPU_COMPUTE_CAP="unknown"
    fi
else
    echo "WARNING: nvidia-smi not available"
    GPU_COMPUTE_CAP="unknown"
fi

if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "WARNING: TORCH_CUDA_ARCH_LIST is set to '$TORCH_CUDA_ARCH_LIST'"
    echo "         This may force incompatible NVRTC architectures at runtime."
fi

echo
echo "=== 6) Checking model files ==="
MODEL_DIR="./models"
CPU_MODEL="$MODEL_DIR/yolov8n_cpu.torchscript"
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
echo "=== 9) Compatibility guardrails ==="
if [ "$GPU_COMPUTE_CAP" != "unknown" ] && [ "$LIBTORCH_VERSION" != "unknown" ] && [ "$LIBTORCH_VERSION" != "missing" ]; then
    GPU_MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d'.' -f1)
    if [ "$GPU_MAJOR" -ge 12 ] 2>/dev/null; then
        if ! echo "$LIBTORCH_VERSION" | grep -Eq '\+cu13[0-9]'; then
            echo "ERROR: GPU compute capability $GPU_COMPUTE_CAP detected, but libtorch is '$LIBTORCH_VERSION'."
            echo "       For SM 12.x GPUs, use a CUDA 13.x libtorch build (e.g., +cu130)."
            exit 1
        else
            echo "[OK] SM 12.x GPU with CUDA 13.x libtorch build detected"
        fi
    else
        echo "[OK] GPU/libtorch compatibility check passed for compute capability $GPU_COMPUTE_CAP"
    fi
else
    echo "[WARN] Skipping strict compatibility check (missing GPU or libtorch metadata)"
fi

echo
echo "=== Setup Verified Successfully ==="
exit 0