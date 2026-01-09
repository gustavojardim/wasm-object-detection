#!/usr/bin/env python3
"""
Export YOLOv8n to TorchScript format optimized for RTX 5060 Ti (compute capability 12.0).
This ensures the CUDA kernels are compiled for the correct GPU architecture.
"""

from ultralytics import YOLO

import os
import torch
import shutil
import platform

def export_yolo_gpu():
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
        out_path = "models/yolov8n_cuda.torchscript"
        if os.path.exists("models/yolov8n.torchscript"):
            shutil.move("models/yolov8n.torchscript", out_path)
            print(f"✓ CUDA model exported successfully to: {out_path}")
        elif os.path.exists("yolov8n.torchscript"):
            shutil.move("yolov8n.torchscript", out_path)
            print(f"✓ CUDA model exported successfully to: {out_path}")
        else:
            print("ERROR: CUDA export failed - output file not found")
    else:
        print("CUDA not available, no CUDA model exported. CPU model untouched.")

if __name__ == "__main__":
    export_yolo_gpu()
