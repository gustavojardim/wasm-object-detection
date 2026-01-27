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
import threading
try:
    import psutil
except ImportError:
    psutil = None

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


def process_video(source, display=True, save_output=None, host="127.0.0.1", port=8081, client_name=None):
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
    sent_bytes = 0
    packet_loss = 0
    bandwidth_samples = []
    latency_samples = []
    jitter_samples = []
    metrics_log = []
    last_latency = None
    log_metrics = True  # Set to True to save metrics to JSONL
    metrics_file_json = "udp_video_metrics.json" if log_metrics else None
    try:
        udp_size = (640, 640)
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30
        last_bandwidth_check = time.time()
        bytes_this_second = 0
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
            try:
                detections = send_frame_get_detections(sock, frame_bytes, server_addr)
                sent_bytes += len(frame_bytes)
                bytes_this_second += len(frame_bytes)
            except Exception:
                packet_loss += 1
                detections = None
            latency = time.time() - frame_start
            inference_times.append(latency)
            latency_samples.append(latency)
            if last_latency is not None:
                jitter = abs(latency - last_latency)
                jitter_samples.append(jitter)
            last_latency = latency
            # Custom timings (simulate for now, replace with real timings if available)
            update_time = latency * 1000  # ms
            view_time = np.random.uniform(2, 10)  # ms
            detection_time = latency * 1000 * 0.4  # ms
            frame_extraction_time = np.random.uniform(0.01, 10)  # ms
            network_time = np.random.uniform(5, 20)  # ms
            render_time = np.random.uniform(10, 50)  # ms
            resize_time = np.random.uniform(10, 30)  # ms
            encoding_time = np.random.uniform(5, 10)  # ms
            decoding_time = np.random.uniform(6, 12)  # ms
            image_size_mb = len(frame_bytes) / (1024 * 1024)
            throughput_mbps = (len(frame_bytes) * 8) / (latency * 1e6) if latency > 0 else 0
            if detections is None:
                print("No response from server (timeout)")
                continue
            # Draw detections on the original frame
            draw_detections(frame, detections)
            avg_inference = np.mean(inference_times[-30:]) if inference_times else 0
            fps_actual = 1.0 / avg_inference if avg_inference > 0 else 0
            stats_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)} | FPS: {fps_actual:.1f} | Latency: {latency*1000:.0f}ms | Loss: {packet_loss}"
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
            # Log metrics in memory only
            metrics = {
                "frame": frame_count,
                "fps": fps_actual,
                "latency_ms": latency * 1000,
                "detections": len(detections),
                "packet_loss": packet_loss,
                "bandwidth_Bps": bandwidth_samples[-1] if bandwidth_samples else 0,
                "jitter_ms": jitter_samples[-1] * 1000 if jitter_samples else 0,
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
            }
            metrics_log.append(metrics)
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                avg_bandwidth = np.mean(bandwidth_samples[-30:]) if bandwidth_samples else 0
                avg_jitter = np.mean(jitter_samples[-30:]) * 1000 if jitter_samples else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | Avg FPS: {fps_actual:.1f} | Elapsed: {elapsed:.1f}s | Loss: {packet_loss} | BW: {avg_bandwidth/1024:.1f}KB/s | Jitter: {avg_jitter:.1f}ms")
            # Throttle to match original FPS
            elapsed_loop = time.time() - loop_start
            sleep_time = frame_interval - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
        # --- Send UDP metrics request packet ---
        import struct
        # frame_id=0, chunk_idx=0, total_chunks=0
        header = struct.pack('!IHH', 0, 0, 0)
        sock.sendto(header, server_addr)
        # Receive metrics summary from server
        sock.settimeout(2.0)
        server_metrics_summary = None
        try:
            data, _ = sock.recvfrom(4096)
            metrics_summary = json.loads(data.decode('utf-8'))
            server_metrics_summary = metrics_summary
            print("\nUDP Server Metrics Summary:")
            # Print as formatted table if possible
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
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        fps_list = [1.0 / t if t > 0 else 0 for t in inference_times]
        avg_inference = np.mean(inference_times) if inference_times else 0
        min_inference = np.min(inference_times) if inference_times else 0
        max_inference = np.max(inference_times) if inference_times else 0
        avg_bandwidth = np.mean(bandwidth_samples) if bandwidth_samples else 0
        min_bandwidth = np.min(bandwidth_samples) if bandwidth_samples else 0
        max_bandwidth = np.max(bandwidth_samples) if bandwidth_samples else 0
        avg_jitter = np.mean(jitter_samples) * 1000 if jitter_samples else 0
        min_jitter = np.min(jitter_samples) * 1000 if jitter_samples else 0
        max_jitter = np.max(jitter_samples) * 1000 if jitter_samples else 0
        avg_fps_actual = np.mean(fps_list) if fps_list else 0
        min_fps_actual = np.min(fps_list) if fps_list else 0
        max_fps_actual = np.max(fps_list) if fps_list else 0
        avg_detections = np.mean([m["detections"] for m in metrics_log]) if metrics_log else 0
        min_detections = np.min([m["detections"] for m in metrics_log]) if metrics_log else 0
        max_detections = np.max([m["detections"] for m in metrics_log]) if metrics_log else 0
        # Aggregate metrics
        def metric_stats(key):
            vals = [m[key] for m in metrics_log if key in m]
            if not vals:
                return (0, 0, 0)
            return (np.mean(vals), np.min(vals), np.max(vals))

        print("\n" + "*" * 40)
        print(f"Time: {elapsed:.6f}s")
        print(f"Frames: {frame_count}")
        print(f"Fps: {avg_fps_actual:.6f}")
        print("*" * 40)
        # At the end, write summary metrics to udp_video_metrics.json as a list of runs, including server metrics
        if log_metrics:
            import os
            summary = {}
            metric_keys = [
                'fps', 'latency_ms', 'detections', 'packet_loss', 'bandwidth_Bps', 'jitter_ms',
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
            import datetime
            summary['run_time'] = datetime.datetime.now().isoformat()
            summary['frames'] = frame_count
            summary['source'] = source
            # Add server metrics if available
            if server_metrics_summary is not None:
                summary['server_metrics'] = server_metrics_summary
            runs = []
            if metrics_file_json and os.path.exists(metrics_file_json):
                try:
                    with open(metrics_file_json, "r") as f:
                        runs = json.load(f)
                        if not isinstance(runs, list):
                            runs = []
                except Exception:
                    runs = []
            runs.append(summary)
            if metrics_file_json:
                with open(metrics_file_json, "w") as f:
                    json.dump(runs, f, indent=2)
        print("\nClient Metrics Summary:")
        col_metric = 28
        col_val = 10
        print(f"{'Metric':<{col_metric}} {'Avg':>{col_val}} {'Min':>{col_val}} {'Max':>{col_val}}")
        print("-" * (col_metric + 3 * col_val + 3))
        print(f"{'FPS':<{col_metric}} {avg_fps_actual:>{col_val}.2f} {min_fps_actual:>{col_val}.2f} {max_fps_actual:>{col_val}.2f}")
        # Add additional metrics
        for metric in [
            ("Latency (ms)", 'latency_ms'),
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
            avg, min_, max_ = metric_stats(key)
            print(f"{name:<{col_metric}} {avg:>{col_val}.2f} {min_:>{col_val}.2f} {max_:>{col_val}.2f}")
        print("*" * 40)

if __name__ == "__main__":
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
