#!/usr/bin/env python3
"""
Export YOLOv8n to TorchScript format for CPU or CUDA (GPU).
Usage:
  python3 export_yolo.py cpu   # Exports to models/yolov8n_cpu.torchscript
  python3 export_yolo.py cuda  # Exports to models/yolov8n_cuda.torchscript
"""

import sys
import os
import torch
import shutil
import platform
from ultralytics import YOLO

def export_yolo(device_choice):
    if device_choice not in ("cpu", "cuda"):
        print("Usage: python3 export_yolo.py [cpu|cuda]")
        sys.exit(1)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Exporting for device: {device_choice}")

    if device_choice == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available on this system.")
            sys.exit(1)
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        torch_arch = f"{compute_cap[0]}.{compute_cap[1]}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = torch_arch
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {compute_cap}")
        print(f"TORCH_CUDA_ARCH_LIST set to: {torch_arch}")
        out_path = "models/yolov8n_cuda.torchscript"
    else:
        device = "cpu"
        out_path = "models/yolov8n_cpu.torchscript"

    print("\nLoading YOLOv8n model...")
    model = YOLO("models/yolov8n.pt")

    print(f"Exporting to TorchScript for device: {device} ...")
    model.export(
        format="torchscript",
        imgsz=640,
        optimize=False,
        half=False,
        device=device
    )

    # Find the exported file
    src_path1 = "models/yolov8n.torchscript"
    src_path2 = "yolov8n.torchscript"
    if os.path.exists(src_path1):
        shutil.copy(src_path1, out_path)
        print(f"✓ Model copied to: {out_path}")
        os.remove(src_path1)
    elif os.path.exists(src_path2):
        shutil.copy(src_path2, out_path)
        print(f"✓ Model copied to: {out_path}")
        os.remove(src_path2)
    else:
        print("ERROR: Export failed - output file not found")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 export_yolo.py [cpu|cuda]")
        sys.exit(1)
    export_yolo(sys.argv[1].lower())
