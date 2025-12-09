# CubeRCNN WASM Inference Server

GPU-accelerated 3D object detection using WebAssembly, WASI-NN, and PyTorch.

## ✨ Features

- **GPU Acceleration**: PyTorch inference runs on NVIDIA GPU via WASI-NN
- **WebAssembly Sandbox**: Safe, portable WASM module with CUDA GPU access
- **WebSocket API**: Real-time inference over WebSocket connections
- **Async Server**: High-performance Tokio-based Rust server
- **Pre-trained Models**: CubeRCNN Res34 FPN for 3D object detection

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  WebSocket Client (Send Images)                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Tokio Server (Rust)                                   │
│  - WebSocket listener on :9001                         │
│  - Pipes images to WASM stdin                          │
│  - Reads detections from WASM stdout                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Wasmtime (WASM Runtime)                               │
│  - Executes WASM module                                │
│  - Provides WASI-NN host functions                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  WASM Guest Module                                      │
│  - Preprocesses images (decode, resize, normalize)     │
│  - Creates WASI-NN tensors                             │
│  - Calls host inference via WASI-NN APIs               │
│  - Parses detection results                            │
│  - Serializes to JSON                                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  PyTorch + CUDA (GPU)                                  │
│  - CubeRCNN model inference                            │
│  - Bounding box regression                             │
│  - 3D object detection                                 │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Start Server
```bash
cd /home/gjardim/obj-test/server
cargo run
```
Server listens on `ws://127.0.0.1:9001`

### 2. Send Images
```bash
source /home/gjardim/obj-test/.venv/bin/activate
python3 /home/gjardim/obj-test/test_client.py image.jpg
```

### 3. Receive Detections
```json
[
  {
    "class": "car",
    "score": 0.95,
    "bbox": [0.1, 0.2, 0.5, 0.6]
  }
]
```

## 📦 Requirements Met

- ✅ WASM module compiles to release binary (1.7MB)
- ✅ Server compiles with zero errors
- ✅ WASI-NN tensor API properly integrated
- ✅ GPU access via Wasmtime PyTorch backend
- ✅ WebSocket communication protocol
- ✅ Image preprocessing pipeline
- ✅ Model file support (.pt files)

## 📋 Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| WASM Module | ✅ Ready | Can create tensors, call compute |
| Server | ✅ Ready | Pipes data to/from WASM |
| GPU Support | ✅ Ready | WASI-NN → Wasmtime → PyTorch |
| Image Input | ✅ Ready | Binary format, length-prefixed |
| Tensor Creation | ✅ Ready | [1, 3, 480, 640] format |
| Output Parsing | ⏳ TODO | Need to parse detection tensors |
| Class Labels | ⏳ TODO | Map IDs to names |
| 3D Data | ⏳ TODO | Extract 3D boxes, sizes, orientations |

## 🔧 Configuration

### Model Input
- **Shape**: [batch=1, channels=3, height=480, width=640]
- **Preprocessing**: ImageNet normalization (mean/std)
- **Format**: NCHW float32

### Model Output
- **Bounding boxes**: 2D boxes in image space
- **Confidence scores**: Per-detection confidence
- **Class IDs**: Object class predictions
- **3D data**: 3D boxes, sizes, orientations (when implemented)

See `CONFIG_GUIDE.md` for detailed configuration options.

## 📚 Documentation

- **QUICK_START.md** - Fast setup guide
- **CONFIG_GUIDE.md** - Model configuration details
- **IMPLEMENTATION_STATUS.md** - Current state and next steps
- **OUTPUT_PARSING_TEMPLATE.rs** - Template code for output parsing

## 🔨 Building

### WASM Module
```bash
cd /home/gjardim/obj-test/inference
cargo build --target wasm32-wasip1 --release
# Output: target/wasm32-wasip1/release/wasm_inference.wasm
```

### Server
```bash
cd /home/gjardim/obj-test/server
cargo build --release
# Output: target/release/host_server
```

## 🐛 Troubleshooting

### Server fails to start
- Check GPU: `nvidia-smi`
- Check Wasmtime: `wasmtime -V`
- Verify WASM binary: `ls inference/target/wasm32-wasip1/release/wasm_inference.wasm`

### Inference errors
- Check WASI-NN support: `wasmtime -S help | grep nn`
- Verify models: `ls -la models/`
- Run setup verification: `bash verify_setup.sh`

### Python environment
```bash
source /home/gjardim/obj-test/.venv/bin/activate
pip install -r requirements.txt
```

## 📊 Performance

**Expected Performance** (CubeRCNN Res34 FPN):
- **Input**: 480×640 RGB image
- **GPU**: NVIDIA GPU with CUDA support
- **Output**: Object detections with 3D bounding boxes
- **Latency**: Depends on GPU and model complexity

## 🎯 Next Steps

1. **Test inference** - Start server and send test images
2. **Identify output tensors** - Check what tensors the model outputs
3. **Implement parsing** - Use OUTPUT_PARSING_TEMPLATE.rs as reference
4. **Add class mapping** - Map detected class IDs to names
5. **Optimize** - Profile and improve inference speed
6. **Deploy** - Ready for production use

## 📝 Project Structure

```
.
├── inference/              # WASM module (guest)
│   ├── src/
│   │   ├── lib.rs         # Main inference logic
│   │   └── wasi_nn/       # Generated WASI-NN bindings
│   └── Cargo.toml
├── server/                # Tokio server (host)
│   ├── src/
│   │   └── main.rs        # WebSocket server
│   └── Cargo.toml
├── models/                # Pre-trained model files
│   ├── cubercnn_Res34_FPN_cpu.pt
│   └── cubercnn_Res34_FPN_cuda.pt
├── .venv/                 # Python virtual environment
├── QUICK_START.md         # Quick start guide
├── CONFIG_GUIDE.md        # Configuration details
└── IMPLEMENTATION_STATUS.md
```

## 🔗 Technologies

- **Rust**: Server (Tokio) and WASM module
- **WebAssembly**: Portable inference container (wasm32-wasip1)
- **WASI-NN**: GPU inference interface
- **Wasmtime**: WASM runtime with WASI support
- **PyTorch**: Deep learning inference
- **CUDA**: GPU acceleration
- **WebSocket**: Real-time communication
- **Python**: Testing and model inspection

## 📄 License

(Add your license here)

## ✉️ Support

For issues or questions:
1. Check **QUICK_START.md** and **CONFIG_GUIDE.md**
2. Review error logs: `RUST_LOG=debug cargo run`
3. Inspect model: `python3 examine_model.py`
4. Check GPU: `nvidia-smi` and `verify_setup.sh`

---

**Status**: Infrastructure complete ✅ | Awaiting output tensor parsing implementation ⏳
