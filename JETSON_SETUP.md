# Jetson Setup (Definitive, No Source Build)

This document is the canonical setup for Jetson devices running this project.

Validated target:
- JetPack 6.2.x (L4T R36.5)
- CUDA 12.6
- Python 3.10 (system Python, no venv)
- PyTorch 2.8.0 + torchvision 0.23.0 from Jetson AI Lab index

---

## 0) Confirm Platform

Run:

```bash
cat /etc/nv_tegra_release
apt-cache policy nvidia-jetpack | head -n 3
python3 --version
```

Expected:
- L4T `R36.5`
- JetPack candidate `6.2.x`
- Python `3.10.x`

---

## 1) Install System Packages

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  wget \
  python3-dev \
  python3-pip \
  libopenblas-dev \
  libblas-dev \
  liblapack-dev \
  libjpeg-dev \
  zlib1g-dev \
  libcudnn9 \
  libcudnn9-dev \
  python3-opencv
```

---

## 2) Install cuSPARSELT (CUDA 12.6)

Use the current helper script from PyTorch `main`:

```bash
wget -O install_cusparselt.sh https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh
chmod +x install_cusparselt.sh
export CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
sudo -E bash ./install_cusparselt.sh
```

Verify:

```bash
ls /usr/local/cuda/lib64/libcusparseLt.so*
```

If it says `tmp_cusparselt` already exists:

```bash
rm -rf tmp_cusparselt
sudo -E bash ./install_cusparselt.sh
```

---

## 3) Install PyTorch + TorchVision (System Python)

Remove conflicting packages first:

```bash
python3 -m pip uninstall -y numpy
sudo -H python3 -m pip uninstall -y torch torchvision torchaudio numpy opencv-python opencv-python-headless
```

Install pinned compatible versions:

```bash
sudo -H python3 -m pip install --force-reinstall "numpy==1.26.4"

sudo -H python3 -m pip install --no-cache --no-deps \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple \
  torch==2.8.0 torchvision==0.23.0
```

Verify:

```bash
python3 - <<'PY'
import numpy, torch, torchvision
print("numpy:", numpy.__version__, numpy.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
PY
```

Expected:
- NumPy `1.26.4`
- Torch `2.8.0`
- Torch CUDA version `12.6`
- CUDA available `True`

---

## 4) Install Ultralytics Runtime Dependencies

```bash
sudo -H python3 -m pip install --no-deps ultralytics
sudo -H python3 -m pip install --no-cache \
  pyyaml requests scipy matplotlib pandas seaborn pillow psutil py-cpuinfo tqdm polars ultralytics-thop
```

---

## 5) Export and Validate CUDA TorchScript

Export model:

```bash
python3 scripts/export_yolo.py cuda
```

Validate TorchScript directly in Python:

```bash
python3 - <<'PY'
import torch
m = torch.jit.load("models/yolov8n_cuda.torchscript", map_location="cuda")
x = torch.randn(1, 3, 640, 640, device="cuda")
y = m(x)
print("ok", y[0].shape if isinstance(y, (list, tuple)) else type(y))
PY
```

If this works, PyTorch/CUDA/model are correct.

---

## 6) Configure Runtime Environment in `.bashrc`

Add these lines to `~/.bashrc`:

```bash
# CUDA runtime
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Derive libtorch from installed Python torch package
export LIBTORCH="$(python3 - <<'PY'
import os, torch
print(os.path.dirname(torch.__file__))
PY
)"
export TORCH_LIB_DIR="$LIBTORCH/lib"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"

# Optional wasi-nn/libtorch compatibility flags
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

Apply now in current shell:

```bash
source ~/.bashrc
```

Quick checks:

```bash
echo "$LIBTORCH"
echo "$TORCH_LIB_DIR"
ls "$TORCH_LIB_DIR"/libtorch_cpu.so "$TORCH_LIB_DIR"/libtorch_cuda.so "$TORCH_LIB_DIR"/libc10_cuda.so
wasmtime --version
```

Run GPU server:

```bash
wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y \
  --dir ./models::/models \
  target/wasm32-wasip2/release/inference.wasm \
  --device gpu --udp
```

---

## 7) Definitive Troubleshooting

### `ImportError: libcudss.so.0`

Cause: torch build requiring cuDSS not present on image.

Action: use pinned `torch==2.8.0` + `torchvision==0.23.0` from Jetson AI Lab index in this guide.

### NumPy warning mentions `NumPy 2.x`

Cause: user-site or other location still providing NumPy 2.x.

Action:

```bash
python3 -m pip uninstall -y numpy
sudo -H python3 -m pip uninstall -y numpy
sudo -H python3 -m pip install --force-reinstall numpy==1.26.4
python3 - <<'PY'
import numpy
print(numpy.__version__, numpy.__file__)
PY
```

### Wasmtime starts but falls back to CPU with CUDA operator errors

Cause: runtime loaded wrong libtorch path / CPU-only symbols at runtime.

Action:
- Ensure `LIBTORCH`, `TORCH_LIB_DIR`, and `LD_LIBRARY_PATH` are set in `~/.bashrc`.
- Run `source ~/.bashrc` before starting wasmtime.

### `wasmtime: error while loading shared libraries: libtorch_cpu.so`

Cause: invalid `LD_LIBRARY_PATH` (often from empty or wrong `LIBTORCH`).

Action:

```bash
source ~/.bashrc
echo "$LIBTORCH"
echo "$TORCH_LIB_DIR"
wasmtime --version
```

---

## 8) Validation Commands

```bash
bash scripts/verify_setup.sh
python3 scripts/export_yolo.py cuda
wasmtime run -S cli=y -S nn=y -S inherit-network=y -S tcp=y \
  --dir ./models::/models \
  target/wasm32-wasip2/release/inference.wasm \
  --device gpu --udp
python3 test-scripts/test_udp_video.py samples/walking_people_uhd.mp4 --remote <JETSON_IP> --port 8081
```

---

## Final Notes

This is the definitive project setup for Jetson in this repository:
- no PyTorch source build required
- no Python virtual environment required
- pinned binary versions to prevent CUDA/runtime drift
- persistent `.bashrc` runtime environment for reproducible GPU execution
