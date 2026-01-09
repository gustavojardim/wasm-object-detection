#!/usr/bin/env python3
"""
Export YOLOv8n to TorchScript format optimized for RTX 5060 Ti (compute capability 12.0).
This ensures the CUDA kernels are compiled for the correct GPU architecture.
"""

import torch
from ultralytics import YOLO
import os

def export_yolo_gpu():
    import platform
    import shutil
    device = 'cpu'
    arch_info = platform.machine()
    cuda_available = torch.cuda.is_available()
    
    print(f"Detected architecture: {arch_info}")
    if cuda_available:
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        torch_arch = f"{compute_cap[0]}.{compute_cap[1]}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = torch_arch
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {compute_cap}")
        print(f"TORCH_CUDA_ARCH_LIST set to: {torch_arch}")
    else:
        print("CUDA not available. Exporting for CPU.")
    
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
    # Determine output filename
    suffix = "cuda" if cuda_available else "cpu"
    out_path = f"models/yolov8n_{suffix}.torchscript"
    # Rename the exported file
    if os.path.exists("models/yolov8n.torchscript"):
        shutil.move("models/yolov8n.torchscript", out_path)
        print(f"✓ Model exported successfully to: {out_path}")
    elif os.path.exists("yolov8n.torchscript"):
        shutil.move("yolov8n.torchscript", out_path)
        print(f"✓ Model exported successfully to: {out_path}")
    else:
        print("ERROR: Export failed - output file not found")

if __name__ == "__main__":
    export_yolo_gpu()
