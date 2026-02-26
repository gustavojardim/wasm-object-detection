# WASM Object Detection with YOLOv8 and wasi-nn

## Overview

This project provides a WebAssembly (WASM) TCP server for real-time object detection using YOLOv8 and the wasi-nn API. It supports both CPU and CUDA (GPU) inference, with easy model export scripts and Kubernetes deployment.

## Main GPU USed

- GPU: NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)
- Driver: 590.48.01
- Wasmtime: 43.0.0
- libtorch: 2.10.0+cu130
- Target: wasm32-wasip2

If your stack differs significantly (especially CUDA/libtorch generation), GPU graph load may work while GPU compute fails.

## How to

### 1. Export Model to .torchscript

- Export for CPU or CUDA:
	```
	python3 scripts/export_yolo.py [cpu|cuda]
	```
	Produces `models/yolov8n_[cpu|cuda].torchscript`.

### 2. Build and Run the WASM Server

- Build the Rust project for WASM:
	```
	cd inference
	cargo build --release --target wasm32-wasip2
	```
- Run with Wasmtime:
	```
	wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y --dir ./models::/models target/wasm32-wasip2/release/inference.wasm --device [cpu|gpu]
	```

### 3. Quick Validation Checklist

- Verify environment and model checks:
	```
	bash scripts/verify_setup.sh
	```
- Run one TCP image inference:
	```
	python3 test-scripts/test_tcp_client.py samples/image.png
	```
- Monitor GPU utilization live:
	```
	watch -n 1 'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader'
	```
	or
	```
	nvtop
	```

### 4. Test with Python Client

- Send an image:
	```
	python3 test_tcp_client.py samples/image.png
	```
- Send a video:
	```
	python3 test_tcp_video.py samples/walking_people_hd.mp4 --save output_detection.mp4
	```

## Model Selection

- The server loads `models/yolov8n_cpu.torchscript` for CPU and `models/yolov8n_cuda.torchscript` for CUDA, based on the `--device` argument.

## Deployment

- See `DEPLOYMENT.md` for Kubernetes and OCI image deployment instructions.
- Use `push-to-registry.sh` to package and push the WASM app and model.

## Troubleshooting

- Run `scripts/verify_setup.sh` to check your environment, dependencies, and model files.
- Ensure the correct model files exist in the `models/` directory before starting the server.
- You can add `--debug` as parameter when running the app.

### GPU Troubleshooting Matrix

| Symptom | Likely Cause | Action |
|---|---|---|
| `Failed while accessing backend` on `--device gpu` at graph load | Wasmtime/libtorch CUDA linkage issue | Rebuild Wasmtime and validate `ldd` shows `libtorch_cuda.so` and `libc10_cuda.so` |
| GPU graph loads but compute fails with NVRTC architecture error | CUDA/libtorch generation does not support your GPU compute capability | Upgrade libtorch CUDA build (for this project, `2.10.0+cu130` resolved it) |
| Falls back to CPU unexpectedly | Missing CUDA model or incompatible runtime stack | Confirm `models/yolov8n_cuda.torchscript` and rerun `scripts/verify_setup.sh` |
| App runs but you are unsure GPU is active | No runtime monitoring | Use `nvidia-smi` watch or `nvtop` while inference is running |

### Advanced Fallback (Older Environments)

Most environments should work with a normal Wasmtime build. If your setup still drops CUDA symbols at runtime, check and rebuild Wasmtime with explicit CUDA linkage:

Check wasmtime + cuda linkage, if no *cuda* .so files appear, they are not linked

```
ldd "$(which wasmtime)" | grep -Ei 'libtorch_cuda|libc10_cuda|libcudart|libcudnn|libcublas'
```

Rebuild

```
cd /path/to/wasmtime
export LIBTORCH=/home/$USER/libtorch
export LD_LIBRARY_PATH=/home/$USER/libtorch/lib:${LD_LIBRARY_PATH}
export RUSTFLAGS='-L native=/home/'"$USER"'/libtorch/lib -C link-arg=-Wl,--no-as-needed -C link-arg=-ltorch_cuda -C link-arg=-lc10_cuda -C link-arg=-Wl,--as-needed'
cargo build -p wasmtime-cli --release --features wasi-nn
```
