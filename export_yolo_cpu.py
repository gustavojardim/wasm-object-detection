#!/usr/bin/env python3
"""
Export YOLOv8n to CPU-compatible TorchScript format.
This creates a portable model that can run on any system without GPU-specific kernels.
"""

import torch
from ultralytics import YOLO
import os

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
    import shutil
    if os.path.exists("models/yolov8n.torchscript"):
        print(f"✓ Model exported successfully to: models/yolov8n.torchscript")
        print("  This CPU-compatible model can run on any system")
    elif os.path.exists("yolov8n.torchscript"):
        shutil.move("yolov8n.torchscript", "models/yolov8n.torchscript")
        print(f"✓ Model exported successfully to: models/yolov8n.torchscript")
        print("  This CPU-compatible model can run on any system")
    else:
        print("ERROR: Export failed - output file not found")

if __name__ == "__main__":
    export_yolo_cpu()
