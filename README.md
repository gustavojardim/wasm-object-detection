# WASM Object Detection with YOLOv8 and wasi-nn

## Overview

This project provides a WebAssembly (WASM) TCP server for real-time object detection using YOLOv8 and the wasi-nn API. It supports both CPU and CUDA (GPU) inference, with easy model export scripts and Kubernetes deployment.

## How to

### 1. Export Model to .torchscript

- For CPU:
	```
	python3 export_yolo.py [cpu|gpu]
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

### 3. Test with Python Client

- Send an image:
	```
	python3 test_tcp_client.py samples/image.png
	```
- Send a video:
	```
	python3 test_tcp_video.py samples/walking_people_hd.mp4 --save output_detection.mp4
	```

## Model Selection

- The server loads `yolov8n_cpu.torchscript` for CPU and `yolov8n_cuda.torchscript` for CUDA, based on the `--device` argument.

## Deployment

- See `DEPLOYMENT.md` for Kubernetes and OCI image deployment instructions.
- Use `push-to-registry.sh` to package and push the WASM app and model.

## Troubleshooting

- Run `verify_setup.sh` to check your environment, dependencies, and model files.
- Ensure the correct model files exist in the `models/` directory before starting the server.
- You can add `--debug` as parameter when running the app
