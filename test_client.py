#!/usr/bin/env python3
"""
WebSocket client for WASM Object Detection.
Sends image -> Receives JSON -> Draws Bounding Boxes -> Saves 'prediction.jpg'
"""

import asyncio
import sys
import websockets
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Define a set of colors for different classes to make it look nice
COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta",
    "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint"
]

async def send_image(image_path: str, server_url: str = "ws://127.0.0.1:9001"):
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Open image using PIL to get dimensions and prepare for drawing
    try:
        original_image = Image.open(image_file).convert("RGB")
        img_width, img_height = original_image.size
    except Exception as e:
        print(f"Error opening image file: {e}")
        sys.exit(1)

    # Read image as binary for transmission
    with open(image_file, 'rb') as f:
        image_bytes = f.read()
    
    print(f"Sending image: {image_file.name} ({img_width}x{img_height}) - {len(image_bytes)} bytes")
    
    try:
        async with websockets.connect(server_url) as ws:
            # 1. Send Image
            await ws.send(image_bytes)
            print("Image sent, waiting for inference results...")
            
            # 2. Receive Result
            result = await ws.recv()
            
            # 3. Parse JSON
            detections = json.loads(result)
            print("\n--- Detections ---")
            print(json.dumps(detections, indent=2))
            
            # 4. Draw Boxes
            draw = ImageDraw.Draw(original_image)
            
            # Try to load a nice font, fallback to default if not found
            try:
                # Common path for Ubuntu/Debian
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except OSError:
                font = ImageFont.load_default()

            if isinstance(detections, list):
                print(f"\nFound {len(detections)} object(s). Drawing boxes...")
                
                for i, det in enumerate(detections):
                    # Get data
                    class_name = det.get('class_name', det.get('class', 'unknown'))
                    class_id = det.get('class_id', 0)
                    score = det.get('score', 0.0)
                    bbox = det.get('bbox', [0, 0, 0, 0]) # [x, y, w, h] normalized

                    # 5. Convert Normalized [x, y, w, h] -> Pixel [x1, y1, x2, y2]
                    # x, y are top-left coordinates 0.0-1.0
                    x_norm, y_norm, w_norm, h_norm = bbox
                    
                    x1 = x_norm * img_width
                    y1 = y_norm * img_height
                    w_pixel = w_norm * img_width
                    h_pixel = h_norm * img_height
                    x2 = x1 + w_pixel
                    y2 = y1 + h_pixel

                    # Pick a color based on class_id (or random if not available)
                    color = COLORS[class_id % len(COLORS)]

                    # Draw Box (width=3 makes it thicker)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # Draw Label Background (so text is readable)
                    label = f"{class_name} {score:.2f}"
                    
                    # Calculate text size to draw a background box for the text
                    # (left, top, right, bottom) of the text
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    
                    # Draw filled rectangle for text background (slightly above or inside top-left)
                    text_bg = [x1, y1 - 20 if y1 > 20 else y1, x1 + (text_bbox[2]-text_bbox[0]) + 10, y1]
                    if y1 > 20:
                        text_bg = [x1, y1 - 20, x1 + (text_bbox[2]-text_bbox[0]) + 8, y1]
                        text_pos = (x1 + 4, y1 - 20)
                    else:
                        # If box is at the very top, draw text inside
                        text_bg = [x1, y1, x1 + (text_bbox[2]-text_bbox[0]) + 8, y1 + 20]
                        text_pos = (x1 + 4, y1)

                    draw.rectangle(text_bg, fill=color)
                    draw.text(text_pos, label, fill="white", font=font)

                # 6. Save Result
                output_filename = f"prediction_{image_file.name}"
                original_image.save(output_filename)
                print(f"\n[SUCCESS] Saved visualized result to: {output_filename}")
                
            else:
                print("Received unexpected data format.")

    except ConnectionRefusedError:
        print("Error: Could not connect to server. Is it running on ws://127.0.0.1:9001?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_client_viz.py <image_path> [ws_url]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9001"
    
    asyncio.run(send_image(image_path, server_url))