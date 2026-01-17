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

# Constants (default values)
HOST = "127.0.0.1"
PORT = 8081  # UDP port
CONFIDENCE_THRESHOLD = 0.5

# Colors for different classes (80 COCO classes)
np.random.seed(42)  # For consistent colors
COLORS = np.random.uniform(0, 255, size=(80, 3))

def send_frame_get_detections(sock, frame_bytes, server_addr, timeout=2.0):
    """Send frame to server using UDP fragmentation and receive detections"""
    import struct
    MTU = 1400  # Safe UDP payload size
    HEADER_FMT = '!IHH'  # frame_id:uint32, chunk_idx:uint16, total_chunks:uint16
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    MAX_PAYLOAD = MTU - HEADER_SIZE
    # Use a static frame_id counter
    if not hasattr(send_frame_get_detections, "frame_id"):
        send_frame_get_detections.frame_id = 1
    frame_id = send_frame_get_detections.frame_id
    send_frame_get_detections.frame_id += 1

    # Fragment frame_bytes
    chunks = [frame_bytes[i:i+MAX_PAYLOAD] for i in range(0, len(frame_bytes), MAX_PAYLOAD)]
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        header = struct.pack(HEADER_FMT, frame_id, idx, total_chunks)
        packet = header + chunk
        sock.sendto(packet, server_addr)

    # Receive response (unchanged)
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


def process_video(source, display=True, save_output=None, host="127.0.0.1", port=8081):
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
    print(f"Connecting to {host}:{port} (UDP)...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (host, port)
    print("Connected! Processing video...")
    frame_count = 0
    start_time = time.time()
    inference_times = []
    try:
        udp_size = (640, 640)
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Resize to 640x640 for server compatibility
            frame_udp = cv2.resize(frame, udp_size)
            # Try multiple JPEG qualities until frame fits under 60KB
            for quality in [35, 30, 25, 20]:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, buffer = cv2.imencode('.jpg', frame_udp, encode_param)
                frame_bytes = buffer.tobytes()
                if len(frame_bytes) <= 60000:
                    break
            else:
                print(f"[WARN] Skipping frame: encoded size {len(frame_bytes)} > 60KB UDP limit (even at lowest quality)")
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
            # Throttle to match original FPS
            elapsed_loop = time.time() - loop_start
            sleep_time = frame_interval - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
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
        print(f"Average inference time: {avg_inference*1000:.0f}ms")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UDP Video client for WASM Object Detection.")
    parser.add_argument("source", help="Video file path or 0 for webcam")
    parser.add_argument("--no-display", action="store_true", help="Disable video display window")
    parser.add_argument("--save", type=str, default=None, help="Save output video to file")
    parser.add_argument("--remote", nargs="?", const="auto", default=None, help="Test against Kubernetes app. Optionally specify NODE_IP (default: auto)")
    args = parser.parse_args()

    display = not args.no_display
    save_output = args.save
    source = args.source

    # If --remote is passed, set host and port for Kubernetes NodePort
    host = "127.0.0.1"
    port = 8081
    if args.remote is not None:
        port = 30081
        if args.remote == "auto":
            import os
            host = os.environ.get("K8S_NODE_IP", "127.0.0.1")
            print(f"[INFO] --remote: Using HOST={host}, PORT={port} (set K8S_NODE_IP env to override)")
        else:
            host = args.remote
            print(f"[INFO] --remote: Using HOST={host}, PORT={port}")

    process_video(source, display, save_output, host=host, port=port)
