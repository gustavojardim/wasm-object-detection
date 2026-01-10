#!/usr/bin/env python3
"""
UDP Video client for WASM Object Detection.
Processes video file or webcam stream, sending frames to UDP server.
"""

import cv2
import socket
import json
import numpy as np
import sys
import time

# Constants
HOST = "127.0.0.1"
PORT = 8081  # UDP port
CONFIDENCE_THRESHOLD = 0.5

# Colors for different classes (80 COCO classes)
np.random.seed(42)  # For consistent colors
COLORS = np.random.uniform(0, 255, size=(80, 3))


def send_frame_get_detections(sock, frame_bytes, server_addr, timeout=2.0):
    """Send frame to server and receive detections via UDP"""
    sock.sendto(frame_bytes, server_addr)
    sock.settimeout(timeout)
    try:
        data, _ = sock.recvfrom(1024 * 1024)  # 1MB buffer
        return json.loads(data.decode('utf-8'))
    except socket.timeout:
        print("[WARN] UDP response timeout")
        return None


def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    height, width = frame.shape[:2]
    for det in detections:
        conf = det.get('confidence', 0)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        bbox = det['bbox']
        x1 = int(bbox['x1'] * width)
        y1 = int(bbox['y1'] * height)
        x2 = int(bbox['x2'] * width)
        y2 = int(bbox['y2'] * height)
        class_id = det['class_id']
        class_name = det['class']
        color = tuple(int(c) for c in COLORS[class_id % len(COLORS)])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def process_video(source, display=True, save_output=None):
    """Process video file or webcam stream via UDP"""
    video_source = 0 if source == "0" else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        print(f"Saving output to: {save_output}")
    print(f"Connecting to {HOST}:{PORT} (UDP)...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (HOST, PORT)
    print("Connected! Processing video...")
    frame_count = 0
    start_time = time.time()
    inference_times = []
    try:
        udp_size = (480, 270)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Resize for UDP safety (send), but keep original for display/output
            frame_udp = cv2.resize(frame, udp_size)
            _, buffer = cv2.imencode('.jpg', frame_udp, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_bytes = buffer.tobytes()
            if len(frame_bytes) > 60000:
                print(f"[WARN] Skipping frame: encoded size {len(frame_bytes)} > 60KB UDP limit")
                continue
            frame_start = time.time()
            detections = send_frame_get_detections(sock, frame_bytes, server_addr)
            inference_time = time.time() - frame_start
            inference_times.append(inference_time)
            if detections is None:
                print("No response from server (timeout)")
                continue
            # Draw detections on the original frame
            draw_detections(frame, detections)
            avg_inference = np.mean(inference_times[-30:]) if inference_times else 0
            fps_actual = 1.0 / avg_inference if avg_inference > 0 else 0
            stats_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | FPS: {fps_actual:.1f} | Latency: {inference_time*1000:.0f}ms"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if display:
                cv2.imshow('WASM Object Detection (UDP)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit")
                    break
            if out:
                out.write(frame)
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
        sock.close()
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference time: {avg_inference*1000:.0f}ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_udp_video.py <video_path|0> [--no-display] [--save output.mp4]")
        print("Examples:")
        print("  python test_udp_video.py samples/walking_people_hd.mp4")
        print("  python test_udp_video.py 0  # Use webcam")
        print("  python test_udp_video.py samples/walking_people_hd.mp4 --save output.mp4")
        sys.exit(1)
    source = sys.argv[1]
    display = "--no-display" not in sys.argv
    save_output = None
    if "--save" in sys.argv:
        save_idx = sys.argv.index("--save")
        if save_idx + 1 < len(sys.argv):
            save_output = sys.argv[save_idx + 1]
    process_video(source, display, save_output)
