#!/usr/bin/env python3
"""
Simulate multiple clients deploying pods via kubectl apply, letting Kubernetes default scheduler assign nodes.
Collects metrics: success, error, time to ready, node assignment, etc.
"""
import subprocess
import threading
import time
import random
import uuid
import json
import os

# Configuration
BASE_POD_YAML = "kubernetes/pod.yaml"  # Path to base pod manifest
NUM_CLIENTS = 10
SPREAD_SECONDS = 15  # Spread requests over this many seconds
NODES = ["gjardim", "gspadotto", "worker1"]  # Used for mapping node names to IPs
NODE_IPS = {
    "gjardim": "192.168.0.105",
    "worker1": "192.168.0.113",
    "gspadotto": "192.168.0.102"
}
UDP_PORT = 30081
VIDEO_PATH = "samples/walking_people_hd.mp4"
POD_NAMESPACE = "default"

results = []

def deploy_client(client_id, delay):
    time.sleep(delay)
    pod_name = f"simclient-{client_id}-{uuid.uuid4().hex[:6]}"
    pod_yaml = f"/tmp/{pod_name}.yaml"
    try:
        # Step 1: Generate pod manifest with unique name
        with open(BASE_POD_YAML) as f:
            pod_spec = f.read().replace("POD_NAME_PLACEHOLDER", pod_name)
        with open(pod_yaml, "w") as f:
            f.write(pod_spec)
        # Step 2: kubectl apply
        apply_cmd = ["kubectl", "apply", "-f", pod_yaml]
        apply_proc = subprocess.run(apply_cmd, capture_output=True, text=True, timeout=30)
        if apply_proc.returncode != 0:
            print(f"[Client {client_id}] kubectl apply failed: {apply_proc.stderr}")
            results.append({"client_id": client_id, "status": "apply_failed", "error": apply_proc.stderr})
            return
        # Step 3: Wait for pod to be ready
        ready = False
        start = time.time()
        for _ in range(60):
            get_cmd = ["kubectl", "get", "pod", pod_name, "-n", POD_NAMESPACE, "-o", "json"]
            get_proc = subprocess.run(get_cmd, capture_output=True, text=True)
            if get_proc.returncode == 0:
                pod_info = json.loads(get_proc.stdout)
                phase = pod_info["status"].get("phase", "")
                if phase == "Running":
                    ready = True
                    break
            time.sleep(2)
        time_to_ready = time.time() - start
        if not ready:
            print(f"[Client {client_id}] Pod {pod_name} not ready after {time_to_ready:.2f}s")
            results.append({"client_id": client_id, "status": "not_ready", "pod_name": pod_name, "time_to_ready": time_to_ready})
            return
        # Step 4: Get node assignment
        get_cmd = ["kubectl", "get", "pod", pod_name, "-n", POD_NAMESPACE, "-o", "json"]
        get_proc = subprocess.run(get_cmd, capture_output=True, text=True)
        pod_info = json.loads(get_proc.stdout)
        node_name = pod_info["spec"].get("nodeName", None)
        if not node_name:
            print(f"[Client {client_id}] Could not determine node for pod {pod_name}")
            results.append({"client_id": client_id, "status": "no_node", "pod_name": pod_name, "time_to_ready": time_to_ready})
            return
        node_ip = NODE_IPS.get(node_name)
        if not node_ip:
            print(f"[Client {client_id}] Node {node_name} IP not found in mapping")
            results.append({"client_id": client_id, "status": "no_node_ip", "pod_name": pod_name, "node_name": node_name, "time_to_ready": time_to_ready})
            return
        # Step 5: Run UDP video test
        cmd = [
            "python3", "test-scripts/test_udp_video.py", VIDEO_PATH,
            "--remote", node_ip,
            "--port", str(UDP_PORT)
        ]
        try:
            udp_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            print(f"[Client {client_id}] test_udp_video result: {udp_result.stdout}")
            # Try to parse metrics from udp_result.stdout (expecting JSON or key metrics)
            udp_metrics = None
            try:
                udp_metrics = json.loads(udp_result.stdout)
            except Exception:
                udp_metrics = None
            results.append({
                "client_id": client_id,
                "status": "success",
                "pod_name": pod_name,
                "node_name": node_name,
                "node_ip": node_ip,
                "time_to_ready": time_to_ready,
                "udp_video_result": udp_result.stdout,
                "udp_video_error": udp_result.stderr,
                "udp_metrics": udp_metrics
            })
        except Exception as e:
            print(f"[Client {client_id}] test_udp_video failed: {e}")
            results.append({
                "client_id": client_id,
                "status": "success",
                "pod_name": pod_name,
                "node_name": node_name,
                "node_ip": node_ip,
                "time_to_ready": time_to_ready,
                "udp_video_result": None,
                "udp_video_error": str(e),
                "udp_metrics": None
            })
    except Exception as e:
        print(f"[Client {client_id}] Exception: {e}")
        results.append({"client_id": client_id, "status": "exception", "error": str(e)})
    finally:
        if os.path.exists(pod_yaml):
            os.remove(pod_yaml)

def main():
    threads = []
    for i in range(NUM_CLIENTS):
        delay = random.uniform(0, SPREAD_SECONDS)
        t = threading.Thread(target=deploy_client, args=(i, delay))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    # Aggregate and print summary
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    print(f"\n--- K8s Scheduler Deployment Summary ---")
    print(f"Total: {len(results)}, Success: {len(success)}, Failed: {len(failed)}")
    if success:
        avg_time = sum(r["time_to_ready"] for r in success if r.get("time_to_ready") is not None) / len(success)
        print(f"Average time to ready: {avg_time:.2f}s")
    # Save results
    with open("multi_client_k8s_scheduler_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
