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
from string import Template

# Configuration
POD_YAML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "pod_template_test.yaml")
def load_pod_template():
        with open(POD_YAML_TEMPLATE_PATH, "r") as f:
                return f.read()
NUM_CLIENTS = 10
NODES = ["gjardim", "gspadotto", "worker1"]  # Used for mapping node names to IPs
NODE_IPS = {
    "gjardim": "192.168.0.105",
    "worker1": "192.168.0.113",
    "gspadotto": "192.168.0.102"
}
UDP_PORT_RANGE = range(30081, 30101)  # 20 ports per node
port_pool = {node: {port: None for port in UDP_PORT_RANGE} for node in NODES}
VIDEO_PATH = "samples/walking_people_hd.mp4"
POD_NAMESPACE = "default"

results = []

def get_free_udp_port(node_name):
    for port, pod in port_pool[node_name].items():
        if pod is None:
            return port
    return None

def mark_port_used(node_name, port, pod_name):
    port_pool[node_name][port] = pod_name

def mark_port_free(node_name, port):
    port_pool[node_name][port] = None

def deploy_client(client_id, delay):
    time.sleep(delay)
    pod_name = f"simclient-{client_id}-{uuid.uuid4().hex[:6]}"
    pod_yaml = f"/tmp/{pod_name}.yaml"
    try:
        # Step 1: Wait for node assignment before generating pod manifest
        node_name = None
        while node_name is None:
            # Randomly pick a node for this client (simulate scheduler)
            node_name = random.choice(NODES)
            # Find a free UDP port for this node
            udp_port = get_free_udp_port(node_name)
            if udp_port is None:
                print(f"[Client {client_id}] No free UDP ports for node {node_name}, retrying...")
                node_name = None
        mark_port_used(node_name, udp_port, pod_name)
        # Step 2: Generate pod manifest YAML as a string
        pod_template = load_pod_template()
        pod_spec = Template(pod_template).substitute(POD_NAME=pod_name, UDP_PORT=udp_port)
        with open(pod_yaml, "w") as f:
            f.write(pod_spec)
        # Step 3: kubectl apply
        apply_cmd = ["kubectl", "apply", "-f", pod_yaml]
        apply_proc = subprocess.run(apply_cmd, capture_output=True, text=True, timeout=30)
        if apply_proc.returncode != 0:
            print(f"[Client {client_id}] kubectl apply failed: {apply_proc.stderr}")
            mark_port_free(node_name, udp_port)
            results.append({"client_id": client_id, "status": "apply_failed", "error": apply_proc.stderr})
            return
        # Step 4: Wait for pod to be ready
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
            mark_port_free(node_name, udp_port)
            results.append({"client_id": client_id, "status": "not_ready", "pod_name": pod_name, "time_to_ready": time_to_ready})
            return
        node_ip = NODE_IPS.get(node_name)
        if not node_ip:
            print(f"[Client {client_id}] Node {node_name} IP not found in mapping")
            mark_port_free(node_name, udp_port)
            results.append({"client_id": client_id, "status": "no_node_ip", "pod_name": pod_name, "node_name": node_name, "time_to_ready": time_to_ready})
            return
        # Step 5: Run UDP video test
        cmd = [
            "python3", "test-scripts/test_udp_video.py", VIDEO_PATH,
            "--remote", node_ip,
            "--port", str(udp_port)
        ]
        try:
            udp_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            print(f"[Client {client_id}] test_udp_video result: {udp_result.stdout}")
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
                "udp_port": udp_port,
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
                "udp_port": udp_port,
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
        mark_port_free(node_name, udp_port)

def main():
    threads = []
    client_id = 0
    # Step 1: Start with 3 clients
    for _ in range(3):
        t = threading.Thread(target=deploy_client, args=(client_id, 0))
        threads.append(t)
        t.start()
        client_id += 1
    # Step 2: Wait 5s, add 1 client
    time.sleep(5)
    if client_id < NUM_CLIENTS:
        t = threading.Thread(target=deploy_client, args=(client_id, 0))
        threads.append(t)
        t.start()
        client_id += 1
    # Step 3: Wait 5s, add 1 client
    time.sleep(5)
    if client_id < NUM_CLIENTS:
        t = threading.Thread(target=deploy_client, args=(client_id, 0))
        threads.append(t)
        t.start()
        client_id += 1
    # Step 4: Wait 5s, add 2 clients
    time.sleep(5)
    for _ in range(2):
        if client_id < NUM_CLIENTS:
            t = threading.Thread(target=deploy_client, args=(client_id, 0))
            threads.append(t)
            t.start()
            client_id += 1
    # Step 5: Wait 10s, add 2 clients
    time.sleep(10)
    for _ in range(2):
        if client_id < NUM_CLIENTS:
            t = threading.Thread(target=deploy_client, args=(client_id, 0))
            threads.append(t)
            t.start()
            client_id += 1
    # Step 6: Wait 15s, add 2 clients
    time.sleep(15)
    for _ in range(2):
        if client_id < NUM_CLIENTS:
            t = threading.Thread(target=deploy_client, args=(client_id, 0))
            threads.append(t)
            t.start()
            client_id += 1
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
    # Append results
    results_file = "multi_client_k8s_scheduler_results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                all_results = existing + results
            else:
                all_results = results
        except Exception:
            all_results = results
    else:
        all_results = results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
