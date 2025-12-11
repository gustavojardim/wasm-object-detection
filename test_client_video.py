import cv2
import asyncio
import websockets
import json
import numpy as np
import sys

# Constants
SERVER_URL = "ws://127.0.0.1:9001"
CONFIDENCE_THRESHOLD = 0.5

# Colors for different classes
COLORS = np.random.uniform(0, 255, size=(80, 3))

async def process_video(source):
    # 0 for webcam, or path string for file
    video_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Connecting to {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("Connected! Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Prepare Frame
                # Resize to reduce network load/latency (optional, but good for speed)
                # But remember YOLOv8n expects 640x640 internal resizing anyway.
                height, width, _ = frame.shape
                
                # Encode frame to JPEG to send over network
                _, buffer = cv2.imencode('.jpg', frame)
                jpg_as_bytes = buffer.tobytes()

                # 2. Send to Server
                await ws.send(jpg_as_bytes)

                # 3. Receive Analysis
                response = await ws.recv()
                detections = json.loads(response)

                # 4. Draw Detections
                for det in detections:
                    score = det.get('score', 0)
                    if score < CONFIDENCE_THRESHOLD:
                        continue

                    # Get normalized coordinates
                    bbox = det.get('bbox') # [x, y, w, h]
                    class_id = det.get('class_id', 0)
                    class_name = det.get('class_name', 'Unknown')

                    # Convert to pixel coordinates
                    # x, y are top-left relative to image size
                    x = int(bbox[0] * width)
                    y = int(bbox[1] * height)
                    w = int(bbox[2] * width)
                    h = int(bbox[3] * height)

                    # Draw Box
                    color = COLORS[class_id % len(COLORS)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Draw Label
                    label = f"{class_name}: {score:.2f}"
                    # Label background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - 20), (x + text_w, y), color, -1)
                    # Label text
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 5. Display Result
                cv2.imshow('WASM Object Detection Stream', frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    asyncio.run(process_video(src))