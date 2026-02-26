#!/usr/bin/env bash

echo "===== CUDA TOOLCHAIN ====="
which nvcc || { echo "nvcc not found"; exit 1; }
nvcc --version | grep "release"

echo
echo "===== CUDA SYMLINK ====="
ls -l /usr/local/cuda
ls -l /etc/alternatives/cuda

echo
echo "===== CUDA HEADERS ====="
if [ -f "$CUDA_HOME/include/cuda.h" ]; then
  echo "CUDA headers found"
else
  echo "CUDA headers NOT found"
fi

echo
echo "===== cuDNN HEADERS ====="
if [ -f "/usr/include/cudnn_version.h" ]; then
  grep CUDNN_MAJOR /usr/include/cudnn_version.h
  grep CUDNN_MINOR /usr/include/cudnn_version.h
else
  echo "cuDNN headers NOT found"
fi

echo
echo "===== cuDNN RUNTIME LIBS ====="
ls /usr/lib/aarch64-linux-gnu/libcudnn.so* 2>/dev/null || echo "cuDNN runtime libs missing"

echo
echo "===== ENVIRONMENT VARIABLES ====="
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS=$MAX_JOBS"
echo "USE_NCCL=$USE_NCCL"
echo "USE_DISTRIBUTED=$USE_DISTRIBUTED"
echo "BUILD_TEST=$BUILD_TEST"

echo
echo "===== PYTHON CHECK ====="
python3 - <<EOF
import os
print("Python sees CUDA_HOME:", os.environ.get("CUDA_HOME"))
try:
    import torch
    print("Torch already installed:", torch.__version__)
except:
    print("Torch not installed (expected before build)")
EOF

echo
echo "===== DONE ====="