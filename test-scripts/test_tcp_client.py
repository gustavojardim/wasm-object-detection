#!/usr/bin/env python3
"""
TCP client for WASM Object Detection.
Sends image -> Receives JSON -> Draws Bounding Boxes -> Saves 'prediction.jpg'
"""

import socket
import struct
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Define colors for different classes
COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta",
    "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint"
]

def send_image(image_path: str, host: str = "127.0.0.1", port: int = 8080):
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Open image using PIL
    try:
        original_image = Image.open(image_file).convert("RGB")
        img_width, img_height = original_image.size
    except Exception as e:
        print(f"Error opening image file: {e}")
        sys.exit(1)

    # Read image as binary
    with open(image_file, 'rb') as f:
        image_bytes = f.read()
    
    print(f"Sending image: {image_file.name} ({img_width}x{img_height}) - {len(image_bytes)} bytes")
    
    try:
        # Connect to TCP server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print(f"Connected to {host}:{port}")
        
        # Send length-prefixed message (4-byte little-endian length + data)
        length = len(image_bytes)
        sock.sendall(struct.pack('<I', length))
        sock.sendall(image_bytes)
        print(f"Sent {length} bytes")
        
        # Receive response (length-prefixed)
        resp_len_bytes = sock.recv(4)
        if len(resp_len_bytes) < 4:
            print("Error: Connection closed before receiving response length")
            sock.close()
            return
        
        resp_len = struct.unpack('<I', resp_len_bytes)[0]
        print(f"Receiving {resp_len} bytes...")
        
        # Receive full response
        response = b''
        while len(response) < resp_len:
            chunk = sock.recv(min(4096, resp_len - len(response)))
            if not chunk:
                break
            response += chunk
        
        sock.close()
        
        # Parse JSON
        detections = json.loads(response.decode('utf-8'))
        print(f"\nReceived {len(detections)} detections:")
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(original_image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, det in enumerate(detections):
            class_name = det["class"]
            conf = det["confidence"]
            x1 = int(det["bbox"]["x1"] * img_width)
            y1 = int(det["bbox"]["y1"] * img_height)
            x2 = int(det["bbox"]["x2"] * img_width)
            y2 = int(det["bbox"]["y2"] * img_height)
            
            color = COLORS[det["class_id"] % len(COLORS)]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1), label, fill="white", font=font)
            
            print(f"  {class_name}: {conf:.2%} at ({x1},{y1})-({x2},{y2})")
        
        # Save result
        output_path = "prediction.jpg"
        original_image.save(output_path)
        print(f"\nSaved annotated image to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tcp_client.py <image_path> [host] [port]")
        print("Example: python test_tcp_client.py samples/image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
    
    send_image(image_path, host, port)
