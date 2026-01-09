#!/usr/bin/env python3
"""
Export YOLOv8n to CPU-compatible TorchScript format.
This creates a portable model that can run on any system without GPU-specific kernels.
"""

from ultralytics import YOLO

import os
import torch
import shutil

def export_yolo_cpu():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    print("\nLoading YOLOv8n model...")
    model = YOLO("models/yolov8n.pt")
    
    print("Exporting to TorchScript (CPU-compatible)...")
    # Export on CPU for maximum portability
    model.export(
        format="torchscript",
        imgsz=640,
        optimize=False,
        half=False,  # Use FP32
        device='cpu'  # CPU export
    )
    
    # Check if file was exported successfully
    out_path = "models/yolov8n_cpu.torchscript"
    src_path1 = "models/yolov8n.torchscript"
    src_path2 = "yolov8n.torchscript"
    if os.path.exists(src_path1):
        shutil.copy(src_path1, out_path)
        print(f"✓ CPU model copied to: {out_path}")
        print("  This CPU-compatible model can run on any system")
    elif os.path.exists(src_path2):
        shutil.copy(src_path2, out_path)
        print(f"✓ CPU model copied to: {out_path}")
        print("  This CPU-compatible model can run on any system")
    else:
        print("ERROR: Export failed - output file not found")

if __name__ == "__main__":
    export_yolo_cpu()
