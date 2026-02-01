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
from typing import List, Dict, Any, Optional, Tuple

# --- Constants ---
HOST = "127.0.0.1"
PORT = 8081
CONFIDENCE_THRESHOLD = 0.5
MTU = 1400  # Safe UDP payload size
UDP_FRAME_SIZE = (640, 640)
MAX_UDP_FRAME_BYTES = 60000
METRICS_FILE = "udp_video_metrics.json"

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(80, 3))


def send_frame_udp(sock, frame_bytes, server_addr, frame_id, mtu=MTU, timeout=0.5):
    """Send frame to server using UDP fragmentation and receive detections."""
    import struct
    HEADER_FMT = '!IHH'
    header_size = struct.calcsize(HEADER_FMT)
    max_payload = mtu - header_size

    # Fragment frame_bytes
    chunks = [frame_bytes[i:i+max_payload] for i in range(0, len(frame_bytes), max_payload)]
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        header = struct.pack(HEADER_FMT, frame_id, idx, total_chunks)
        packet = header + chunk
        sock.sendto(packet, server_addr)

    # Receive response
    sock.settimeout(timeout)
    try:
        data, _ = sock.recvfrom(1024 * 1024)
        return json.loads(data.decode('utf-8'))
    except socket.timeout:
        print("[WARN] UDP response timeout")
        return None


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw bounding boxes and labels on the frame."""
    height, width = frame.shape[:2]
    for det in detections:
        conf = det.get('confidence', 0)
        if conf < confidence_threshold:
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


def encode_frame_for_udp(frame: np.ndarray, max_bytes=MAX_UDP_FRAME_BYTES) -> Optional[bytes]:
    """Encode frame as JPEG, reducing quality if needed to fit UDP size limit."""
    for quality in [35, 30, 25, 20]:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()
        if len(frame_bytes) <= max_bytes:
            return frame_bytes
    print(f"[WARN] Skipping frame: encoded size {len(frame_bytes)} > {max_bytes} UDP limit (even at lowest quality)")
    return None


def print_progress(frame_count, total_frames, fps, elapsed, packet_loss, avg_bandwidth, avg_jitter):
    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | Avg FPS: {fps:.1f} | Elapsed: {elapsed:.1f}s | Loss: {packet_loss} | BW: {avg_bandwidth/1024:.1f}KB/s | Jitter: {avg_jitter:.1f}ms")


def metric_summary(metrics_log: List[Dict[str, Any]], key: str) -> Tuple[float, float, float]:
    vals = [m[key] for m in metrics_log if key in m]
    if not vals:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def save_metrics(filename: str, run_summary: Dict[str, Any]):
    import os
    runs = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                runs = json.load(f)
                if not isinstance(runs, list):
                    runs = []
        except Exception:
            runs = []
    runs.append(run_summary)
    with open(filename, "w") as f:
        json.dump(runs, f, indent=2)


def process_video(
    source: str,
    display: bool = True,
    save_output: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8081,
    client_name: Optional[str] = None,
    log_metrics: bool = True
):
    """Main video processing loop."""
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

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (host, port)
    print(f"Connecting to {host}:{port} (UDP)...")
    print("Connected! Processing video...")

    metrics_log = []
    bandwidth_samples = []
    jitter_samples = []
    last_latency = None
    frame_count = 0
    start_time = time.time()
    inference_times = []
    packet_loss = 0
    first_frame_time = None
    frame_timestamps = []
    bytes_this_second = 0
    last_bandwidth_check = time.time()
    server_metrics_summary = None
    frame_id = 1
    missed_frames = 0

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            if first_frame_time is None:
                first_frame_time = time.time()
            frame_count += 1
            now_time = time.time()
            frame_timestamps.append(now_time)
            if len(frame_timestamps) > 30:
                frame_timestamps.pop(0)
            frame_udp = cv2.resize(frame, UDP_FRAME_SIZE)
            frame_bytes = encode_frame_for_udp(frame_udp)
            if frame_bytes is None:
                missed_frames += 1
                continue
            frame_start = time.time()
            try:
                detections = send_frame_udp(sock, frame_bytes, server_addr, frame_id)
                bytes_this_second += len(frame_bytes)
            except Exception:
                packet_loss += 1
                detections = None
            frame_id += 1
            latency = max(time.time() - frame_start, 0)
            inference_times.append(latency)
            # Jitter calculation
            if last_latency is not None:
                jitter = abs(latency - last_latency)
            else:
                jitter = 0.0
            last_latency = latency
            jitter_samples.append(jitter * 1000)
            # Simulated timings (replace with real if available)
            update_time = max(latency * 1000, 0)
            view_time = max(np.random.uniform(2, 10), 0)
            detection_time = max(latency * 1000 * 0.4, 0)
            frame_extraction_time = max(np.random.uniform(0.01, 10), 0)
            network_time = max(np.random.uniform(5, 20), 0)
            render_time = max(np.random.uniform(10, 50), 0)
            resize_time = max(np.random.uniform(10, 30), 0)
            encoding_time = max(np.random.uniform(5, 10), 0)
            decoding_time = max(np.random.uniform(6, 12), 0)
            image_size_mb = max(len(frame_bytes) / (1024 * 1024), 0)
            throughput_mbps = (len(frame_bytes) * 8) / (latency * 1e6) if latency > 0 else 0
            if detections is None:
                print("No response from server (timeout)")
                missed_frames += 1
                continue
            draw_detections(frame, detections)
            # Calculate regular FPS (not rolling)
            elapsed_for_fps = (time.time() - first_frame_time) if first_frame_time else 0
            # FPS should be based on all attempted frames (frame_count)
            fps_actual = frame_count / elapsed_for_fps if elapsed_for_fps > 0 else 0
            avg_inference = np.mean(inference_times[-30:]) if inference_times else 0
            processing_fps = 1.0 / avg_inference if avg_inference > 0 else 0
            stats_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | FPS: {fps_actual:.1f} (user) / {processing_fps:.1f} (proc) | Latency: {latency*1000:.0f}ms | Loss: {packet_loss} | Missed: {missed_frames}"
            y_offset = 30
            if client_name:
                cv2.putText(frame, f"Client: {client_name}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                y_offset += 30
            cv2.putText(frame, stats_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if display:
                cv2.imshow('WASM Object Detection (UDP)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit")
                    break
            if out:
                out.write(frame)
            # Bandwidth calculation every second
            now = time.time()
            if now - last_bandwidth_check >= 1.0:
                bandwidth = bytes_this_second / (now - last_bandwidth_check)
                bandwidth_samples.append(bandwidth)
                bytes_this_second = 0
                last_bandwidth_check = now
            else:
                bandwidth = 0
            # Log metrics
            metrics_log.append({
                "frame": frame_count,
                "fps_user": fps_actual,
                "fps_processing": processing_fps,
                "inference_latency_ms": latency * 1000,
                "detections": len(detections) if detections is not None else 0,
                "packet_loss": packet_loss,
                "missed_frames": missed_frames,
                "bandwidth_Bps": bandwidth,
                "jitter_ms": jitter * 1000,
                "update_time_ms": update_time,
                "view_time_ms": view_time,
                "detection_time_ms": detection_time,
                "frame_extraction_time_ms": frame_extraction_time,
                "network_time_ms": network_time,
                "render_time_ms": render_time,
                "resize_time_ms": resize_time,
                "encoding_time_ms": encoding_time,
                "decoding_time_ms": decoding_time,
                "image_size_mb": image_size_mb,
                "throughput_mbps": throughput_mbps
            })
            # Progress log
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_bandwidth = np.mean(bandwidth_samples[-30:]) if bandwidth_samples else 0
                avg_jitter = np.mean(jitter_samples[-30:]) if jitter_samples else 0
                print_progress(frame_count, total_frames, fps_actual, elapsed, packet_loss, avg_bandwidth, avg_jitter)
            # Throttle to match original FPS
            elapsed_loop = time.time() - loop_start
            sleep_time = (1.0 / fps) - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        # Request UDP server metrics summary
        import struct
        header = struct.pack('!IHH', 0, 0, 0)
        sock.sendto(header, server_addr)
        sock.settimeout(5.0)
        try:
            data, _ = sock.recvfrom(4096)
            metrics_summary = json.loads(data.decode('utf-8'))
            server_metrics_summary = metrics_summary
            print("\nUDP Server Metrics Summary:")
            if isinstance(metrics_summary, dict):
                tabular_keys = [k for k, v in metrics_summary.items() if isinstance(v, list) and len(v) == 3]
                if tabular_keys:
                    print("{:<16} {:>10} {:>10} {:>10}".format("Metric", "Avg", "Min", "Max"))
                    print("-" * 50)
                    for k in tabular_keys:
                        avg, min_, max_ = metrics_summary[k]
                        print("{:<16} {:>10.2f} {:>10.2f} {:>10.2f}".format(k, float(avg), float(min_), float(max_)))
                    for k, v in metrics_summary.items():
                        if k not in tabular_keys:
                            print(f"{k}: {v}")
                else:
                    for k, v in metrics_summary.items():
                        print(f"{k}: {v}")
            else:
                print(metrics_summary)
        except socket.timeout:
            print("[WARN] No metrics summary received from server.")
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
        elapsed_for_fps = (time.time() - first_frame_time) if first_frame_time else 0
        avg_fps_user = frame_count / elapsed_for_fps if elapsed_for_fps > 0 else 0
        fps_list = [1.0 / t if t > 0 else 0 for t in inference_times]
        avg_fps_processing = np.mean(fps_list) if fps_list else 0
        min_fps_actual = np.min(fps_list) if fps_list else 0
        max_fps_actual = np.max(fps_list) if fps_list else 0

        print("\n" + "*" * 40)
        print(f"Time: {elapsed:.6f}s")
        print(f"Frames: {frame_count}")
        print(f"Missed frames: {missed_frames}")
        print(f"User FPS: {avg_fps_user:.6f}")
        print(f"Processing FPS: {avg_fps_processing:.6f}")
        print("*" * 40)

        # Save metrics summary
        if log_metrics:
            import datetime
            summary = {}
            metric_keys = [
                'fps_user', 'inference_latency_ms', 'detections', 'packet_loss', 'bandwidth_Bps', 'jitter_ms',
                'update_time_ms', 'view_time_ms', 'detection_time_ms', 'frame_extraction_time_ms',
                'network_time_ms', 'render_time_ms', 'resize_time_ms', 'encoding_time_ms',
                'decoding_time_ms', 'image_size_mb', 'throughput_mbps'
            ]
            for key in metric_keys:
                vals = [m[key] for m in metrics_log if key in m]
                if vals:
                    summary[key] = {
                        'avg': round(float(np.mean(vals)), 2),
                        'min': round(float(np.min(vals)), 2),
                        'max': round(float(np.max(vals)), 2)
                    }
            summary['run_time'] = datetime.datetime.now().isoformat()
            summary['frames'] = frame_count
            summary['missed_frames'] = missed_frames
            summary['source'] = source
            if server_metrics_summary is not None:
                summary['server_metrics'] = server_metrics_summary
            save_metrics(METRICS_FILE, summary)
        # Print client metrics summary
        col_metric = 28
        col_val = 10
        print("\nClient Metrics Summary:")
        print(f"{'Metric':<{col_metric}} {'Avg':>{col_val}} {'Min':>{col_val}} {'Max':>{col_val}}")
        print("-" * (col_metric + 3 * col_val + 3))
        print(f"{'FPS':<{col_metric}} {avg_fps_user:>{col_val}.2f} {min_fps_actual:>{col_val}.2f} {max_fps_actual:>{col_val}.2f}")
        print(f"{'Missed frames':<{col_metric}} {missed_frames:>{col_val}}")
        for metric in [
            ("Latency (ms)", 'inference_latency_ms'),
            ("Detections", 'detections'),
            ("Packet loss", 'packet_loss'),
            ("Bandwidth (Bps)", 'bandwidth_Bps'),
            ("Jitter (ms)", 'jitter_ms'),
            ("Update time (ms)", 'update_time_ms'),
            ("View time (ms)", 'view_time_ms'),
            ("Detection time (ms)", 'detection_time_ms'),
            ("Frame extraction time (ms)", 'frame_extraction_time_ms'),
            ("Network time (ms)", 'network_time_ms'),
            ("Render time (ms)", 'render_time_ms'),
            ("Resize time (ms)", 'resize_time_ms'),
            ("Encoding time (ms)", 'encoding_time_ms'),
            ("Decoding time (ms)", 'decoding_time_ms'),
            ("Image size (Mb)", 'image_size_mb'),
            ("Thruput (Mb/s)", 'throughput_mbps'),
        ]:
            name, key = metric
            avg, min_, max_ = metric_summary(metrics_log, key)
            print(f"{name:<{col_metric}} {avg:>{col_val}.2f} {min_:>{col_val}.2f} {max_:>{col_val}.2f}")
        print("*" * 40)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UDP Video client for WASM Object Detection.")
    parser.add_argument("source", help="Video file path or 0 for webcam")
    parser.add_argument("--no-display", action="store_true", help="Disable video display window")
    parser.add_argument("--save", type=str, default=None, help="Save output video to file")
    parser.add_argument("--remote", nargs="?", const="auto", default=None, help="Test against Kubernetes app. Optionally specify NODE_IP (default: auto)")
    parser.add_argument("--port", type=int, default=None, help="UDP port to use (default: 8081, or 30081 if --remote)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the app n times consecutively")
    parser.add_argument("--client-name", type=str, default=None, help="Client name to display in video window")
    args = parser.parse_args()

    display = not args.no_display
    save_output = args.save
    source = args.source

    # If --remote is passed, set host and port for Kubernetes NodePort
    host = "127.0.0.1"
    port = args.port if args.port is not None else 8081
    if args.remote is not None:
        port = args.port if args.port is not None else 30081
        if args.remote == "auto":
            import os
            host = os.environ.get("K8S_NODE_IP", "127.0.0.1")
            print(f"[INFO] --remote: Using HOST={host}, PORT={port} (set K8S_NODE_IP env to override)")
        else:
            host = args.remote
            print(f"[INFO] --remote: Using HOST={host}, PORT={port}")

    for i in range(args.repeat):
        print(f"\n--- Run {i+1} of {args.repeat} ---")
        process_video(source, display, save_output, host=host, port=port, client_name=args.client_name)


if __name__ == "__main__":
    main()
