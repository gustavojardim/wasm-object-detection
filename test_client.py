#!/usr/bin/env python3
"""
Simple WebSocket client to test the inference server.
Sends binary image data and receives JSON detections.
"""

import asyncio
import sys
import websockets
import json
from pathlib import Path

async def send_image(image_path: str, server_url: str = "ws://127.0.0.1:9001"):
    """
    Send an image to the inference server and get detections.
    
    Args:
        image_path: Path to image file
        server_url: WebSocket server URL
    """
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Read image as binary
    with open(image_file, 'rb') as f:
        image_data = f.read()
    
    print(f"Sending image: {image_file.name} ({len(image_data)} bytes)")
    
    try:
        async with websockets.connect(server_url) as ws:
            # Send binary image data
            await ws.send(image_data)
            print("Image sent, waiting for results...")
            
            # Receive JSON result
            result = await ws.recv()
            
            # Parse and display results
            detections = json.loads(result)
            print("\nDetections:")
            print(json.dumps(detections, indent=2))
            
            # Print summary
            if isinstance(detections, list):
                print(f"\nFound {len(detections)} object(s)")
                for i, det in enumerate(detections):
                    print(f"  [{i}] class={det.get('class')}, score={det.get('score'):.4f}, bbox={det.get('bbox')}")
    
    except ConnectionRefusedError:
        print("Error: Could not connect to server. Is it running on ws://127.0.0.1:9001?")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON response: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_client.py <image_path> [ws_url]")
        print("Example: python3 test_client.py test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:9001"
    
    asyncio.run(send_image(image_path, server_url))
