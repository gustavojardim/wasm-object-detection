# Building PyTorch 2.8.0 from Source on Jetson (CUDA 12.6 + cuDNN 9)

This guide documents a clean and reproducible method to build **PyTorch 2.8.0** from source on:

- JetPack 6
- CUDA 12.6
- cuDNN 9.x
- Jetson AGX Orin (SM 8.7 recommended)
- Python virtual environment

---

# 1. Verify CUDA Installation

### Check nvcc

```bash
which nvcc
nvcc --version
```

Expected:
- `/usr/local/cuda/bin/nvcc`
- CUDA release 12.6

### Verify CUDA Symlink

```bash
ls -l /usr/local/cuda
ls -l /etc/alternatives/cuda
```

Expected:

```
/usr/local/cuda -> /etc/alternatives/cuda
/etc/alternatives/cuda -> /usr/local/cuda-12.6
```

---

# 2. Install Required System Packages

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  curl \
  libopenblas-dev \
  libblas-dev \
  liblapack-dev \
  libjpeg-dev \
  zlib1g-dev \
  libpython3-dev \
  python3-dev \
  python3-pip \
  python3-venv \
  pkg-config \
  libopenmpi-dev \
  libcudnn9 \
  libcudnn9-dev
```

---

# 3. Verify cuDNN Installation

### Check Headers

```bash
ls /usr/include/cudnn_version.h
grep CUDNN_MAJOR /usr/include/cudnn_version.h
grep CUDNN_MINOR /usr/include/cudnn_version.h
```

Expected:

```
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 3
```

### Check Runtime Libraries

```bash
ls /usr/lib/aarch64-linux-gnu/libcudnn.so*
```

Expected:

```
libcudnn.so.9
libcudnn.so.9.x.x
```

---

# 4. Create Python Virtual Environment

```bash
python3 -m venv ~/venvs/torch28
source ~/venvs/torch28/bin/activate
pip install --upgrade pip
```

Remove existing torch installations:

```bash
pip uninstall -y torch torchvision torchaudio
```

---

# 5. Clone PyTorch 2.8 Cleanly

```bash
cd ~
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.8.0
```

Reset and reinitialize submodules:

```bash
git submodule deinit -f --all
rm -rf .git/modules
git clean -xffd
git submodule sync
git submodule update --init --recursive
```

Verify:

```bash
git status
```

Expected:

```
working tree clean
```

---

# 6. Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="8.7"

export MAX_JOBS=2
export USE_NCCL=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
```

Notes:
- `8.7` is correct for AGX Orin.
- Do not set unnecessary LIBTORCH_* variables unless required.
- Avoid mixing system torch with custom builds.

---

# 7. Install Python Build Dependencies

Inside the PyTorch directory:

```bash
pip install -r requirements.txt
pip install ninja
```

---

# 8. Build PyTorch

```bash
python3 setup.py bdist_wheel
```

Expected:
- 1–2 hours build time on AGX Orin
- High CPU and memory usage

If out-of-memory:
```
export MAX_JOBS=1
```

---

# 9. Install Built Wheel

```bash
cd dist
pip install torch-2.8.0-*.whl
```

---

# 10. Validate CUDA and cuDNN

```python
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("Device:", torch.cuda.get_device_name(0))

x = torch.randn(2, 2, device="cuda")
print("Tensor allocation successful:", x)
```

Expected:
- CUDA available → True
- cuDNN version → 9xxx
- Tensor allocates without error

---

# Common Failure Causes

### Missing cuDNN Development Package

```bash
sudo apt install libcudnn9-dev
```

### CUDA Version Mismatch

Ensure:
- `nvcc --version` matches `/usr/local/cuda`
- CUDA_HOME is correct

### Incorrect GPU Architecture

For AGX Orin:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
```

### Old Torch Installed in Environment

```bash
pip uninstall -y torch torchvision torchaudio
```

---

# Optional: Environment Verification Script

Run `scripts/verify_jetson_setup.sh`

---

# Final Notes

This process ensures:

- Clean repository state
- Correct CUDA alignment
- cuDNN properly detected
- Proper GPU architecture targeting
- No environment contamination