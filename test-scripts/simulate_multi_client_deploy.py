#!/usr/bin/env python3
"""
Simulate multiple clients requesting deployments from the orchestrator at different times.
Collects metrics: success, error, time to ready, etc.
"""
import requests
import threading
import time
import random
import uuid
import json

ORCHESTRATOR_URL = "http://192.168.0.113:30500/deploy"
NODES = ["gjardim", "gspadotto", "worker1"]
NUM_CLIENTS = 10
SPREAD_SECONDS = 15  # Spread requests over this many seconds
LATENCY_THRESHOLD = 80

results = []

def deploy_client(client_id, delay):
    time.sleep(delay)
    request_id = str(uuid.uuid4())
    try:
        # Step 1: Get available nodes/probes from /request-deploy
        req_deploy_url = ORCHESTRATOR_URL.replace("/deploy", "/request-deploy")
        resp = requests.post(req_deploy_url, json={}, timeout=30)
        if resp.status_code != 200:
            print(f"[Client {client_id}] Failed to get probes: {resp.text}")
            results.append({"client_id": client_id, "status": "failed_request_deploy", "error": resp.text})
            return
        node_info = resp.json().get("nodes", [])
        # Step 2: Filter nodes by latency (<=20ms)
        latency_filtered_nodes = []
        for n in node_info:
            probe_url = n.get("probe")
            # Rewrite probe_url to use port 8080 (hostNetwork)
            if probe_url:
                # Replace :8080 or :30088 with :8080
                probe_url = probe_url.replace(":8080", ":8080").replace(":30088", ":8080")
            try:
                probe_start = time.perf_counter()
                probe_resp = requests.get(probe_url, timeout=2)
                probe_elapsed = (time.perf_counter() - probe_start) * 1000  # ms
                if probe_resp.status_code == 200 and probe_elapsed <= 200:
                    latency_filtered_nodes.append(n["name"])
                else:
                    print(f"[Client {client_id}] Node {n['name']} latency {probe_elapsed:.2f}ms (ignored)")
            except Exception as e:
                print(f"[Client {client_id}] Node {n['name']} probe failed: {e}")
        if not latency_filtered_nodes:
            print(f"[Client {client_id}] No nodes with latency <= 100ms.")
            results.append({"client_id": client_id, "status": "no_low_latency_nodes"})
            return
        # Step 3: Use all low-latency nodes for deployment
        payload = {"nodes": latency_filtered_nodes, "request_id": request_id, "client": f"client-{client_id}", "latency_threshold": LATENCY_THRESHOLD}
        # Step 3: Send deploy request
        start = time.time()
        resp = requests.post(ORCHESTRATOR_URL, json=payload, timeout=180)
        elapsed = time.time() - start
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        if resp.status_code == 200 and isinstance(data, dict):
            print(f"[Client {client_id}] Success: {data} (time: {elapsed:.2f}s)")
            node_ip = data.get("node_ip")
            udp_port = data.get("udp_port", 30081)
            if node_ip and udp_port:
                import subprocess
                video_path = "samples/walking_people_hd.mp4"
                cmd = [
                    "python3", "test-scripts/test_udp_video.py", video_path,
                    "--remote", node_ip,
                    "--port", str(udp_port),
                    "--client-name", f"client-{client_id}"
                ]
                try:
                    udp_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    print(f"[Client {client_id}] test_udp_video result: {udp_result.stdout}")
                    results.append({
                        "client_id": client_id,
                        "status": "success",
                        "deploy_time": data.get("time"),
                        "pod_name": data.get("pod"),
                        "node_ip": node_ip,
                        "udp_port": udp_port,
                        "elapsed": elapsed,
                        "udp_video_result": udp_result.stdout,
                        "udp_video_error": udp_result.stderr
                    })
                except Exception as e:
                    print(f"[Client {client_id}] test_udp_video failed: {e}")
                    results.append({
                        "client_id": client_id,
                        "status": "success",
                        "deploy_time": data.get("time"),
                        "pod_name": data.get("pod"),
                        "node_ip": node_ip,
                        "udp_port": udp_port,
                        "elapsed": elapsed,
                        "udp_video_result": None,
                        "udp_video_error": str(e)
                    })
            else:
                print(f"[Client {client_id}] Could not find node_ip or udp_port in orchestrator response")
                results.append({
                    "client_id": client_id,
                    "status": "success",
                    "deploy_time": data.get("time"),
                    "pod_name": data.get("pod"),
                    "node_ip": node_ip,
                    "udp_port": udp_port,
                    "elapsed": elapsed,
                    "udp_video_result": None,
                    "udp_video_error": "Node IP or UDP port not found in response"
                })
        else:
            print(f"[Client {client_id}] Error response: {data}")
            results.append({"client_id": client_id, "status": "failed", "error": str(data), "elapsed": elapsed})
    except Exception as e:
        print(f"[Client {client_id}] Exception: {e}")
        results.append({"client_id": client_id, "status": "exception", "error": str(e)})

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
    print(f"\n--- Deployment Summary ---")
    print(f"Total: {len(results)}, Success: {len(success)}, Failed: {len(failed)}")
    if success:
        avg_time = sum(r["deploy_time"] for r in success if r["deploy_time"] is not None) / len(success)
        print(f"Average deploy time: {avg_time:.2f}s")
    # Append results
    import os
    results_file = "multi_client_deploy_results.json"
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