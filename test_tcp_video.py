#!/usr/bin/env python3
"""
TCP Video client for WASM Object Detection.
Processes video file or webcam stream, sending frames to TCP server.
"""

import cv2
import socket
import struct
import json
import numpy as np
import sys
import time

# Constants
HOST = "127.0.0.1"
PORT = 8080
CONFIDENCE_THRESHOLD = 0.5

# Colors for different classes (80 COCO classes)
np.random.seed(42)  # For consistent colors
COLORS = np.random.uniform(0, 255, size=(80, 3))

def send_frame_get_detections(sock, frame_bytes):
    """Send frame to server and receive detections"""
    # Send length-prefixed frame
    length = len(frame_bytes)
    sock.sendall(struct.pack('<I', length))
    sock.sendall(frame_bytes)
    
    # Receive response length
    resp_len_bytes = sock.recv(4)
    if len(resp_len_bytes) < 4:
        return None
    
    resp_len = struct.unpack('<I', resp_len_bytes)[0]
    
    # Receive full response
    response = b''
    while len(response) < resp_len:
        chunk = sock.recv(min(4096, resp_len - len(response)))
        if not chunk:
            break
        response += chunk
    
    # Parse JSON
    return json.loads(response.decode('utf-8'))

def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    height, width = frame.shape[:2]
    
    for det in detections:
        conf = det.get('confidence', 0)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        # Get bbox coordinates (normalized 0-1)
        bbox = det['bbox']
        x1 = int(bbox['x1'] * width)
        y1 = int(bbox['y1'] * height)
        x2 = int(bbox['x2'] * width)
        y2 = int(bbox['y2'] * height)
        
        class_id = det['class_id']
        class_name = det['class']
        
        # Get color for this class
        color = tuple(int(c) for c in COLORS[class_id % len(COLORS)])
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def process_video(source, display=True, save_output=None):
    """Process video file or webcam stream"""
    # 0 for webcam, or path string for file
    video_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if saving output
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        print(f"Saving output to: {save_output}")
    
    # Connect to server
    print(f"Connecting to {HOST}:{PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Connected! Processing video...")
    
    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send and receive
            frame_start = time.time()
            detections = send_frame_get_detections(sock, frame_bytes)
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
            
            if detections is None:
                print("Connection closed by server")
                break
            
            # Draw detections on frame
            draw_detections(frame, detections)
            
            # Add stats overlay
            avg_inference = np.mean(inference_times[-30:]) if inference_times else 0
            fps_actual = 1.0 / avg_inference if avg_inference > 0 else 0
            stats_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | FPS: {fps_actual:.1f} | Latency: {inference_time*1000:.0f}ms"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            if display:
                cv2.imshow('WASM Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit")
                    break
            
            # Save frame
            if out:
                out.write(frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | Avg FPS: {fps_actual:.1f} | Elapsed: {elapsed:.1f}s")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        sock.close()
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference time: {avg_inference*1000:.0f}ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tcp_video.py <video_path|0> [--no-display] [--save output.mp4]")
        print("Examples:")
        print("  python test_tcp_video.py samples/walking_people_hd.mp4")
        print("  python test_tcp_video.py 0  # Use webcam")
        print("  python test_tcp_video.py samples/walking_people_hd.mp4 --save output.mp4")
        sys.exit(1)
    
    source = sys.argv[1]
    display = "--no-display" not in sys.argv
    save_output = None
    
    # Check for --save option
    if "--save" in sys.argv:
        save_idx = sys.argv.index("--save")
        if save_idx + 1 < len(sys.argv):
            save_output = sys.argv[save_idx + 1]
    
    process_video(source, display, save_output)
