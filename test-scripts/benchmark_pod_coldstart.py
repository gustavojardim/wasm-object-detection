#!/usr/bin/env python3
"""
Benchmark pod cold start and deploy time for a given pod manifest.
Usage: python3 benchmark_pod_coldstart.py -f <pod-yaml> --repeat N --output <results.json>
"""
import argparse
import subprocess
import time
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pod cold start and deploy time.")
    parser.add_argument("-f", "--file", required=True, help="Path to pod YAML manifest.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repetitions.")
    parser.add_argument("--output", default="pod_coldstart_results.json", help="Output JSON file.")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace.")
    return parser.parse_args()

def get_pod_name(yaml_path):
    # Extract pod name from YAML (assumes 'name:' is present and not indented in a list)
    with open(yaml_path) as f:
        for line in f:
            if line.strip().startswith("name:"):
                return line.strip().split(":", 1)[1].strip()
    raise ValueError("Could not find pod name in YAML.")

def main():
    args = parse_args()
    pod_yaml = args.file
    pod_name = get_pod_name(pod_yaml)
    results = []
    for i in range(args.repeat):
        print(f"\n--- Iteration {i+1}/{args.repeat} ---")
        # Delete pod if exists
        subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found", "--grace-period=0", "--force"], capture_output=True)
        # Apply pod
        t_apply_start = time.time()
        apply_proc = subprocess.run(["kubectl", "apply", "-f", pod_yaml, "-n", args.namespace], capture_output=True, text=True)
        t_apply_end = time.time()
        if apply_proc.returncode != 0:
            print(f"kubectl apply failed: {apply_proc.stderr}")
            results.append({"iteration": i+1, "status": "apply_failed", "error": apply_proc.stderr})
            continue
        # Wait for pod to be Running
        t_running_start = t_apply_end
        for _ in range(60):
            get_proc = subprocess.run(["kubectl", "get", "pod", pod_name, "-n", args.namespace, "-o", "json"], capture_output=True, text=True)
            if get_proc.returncode == 0:
                try:
                    pod_info = json.loads(get_proc.stdout)
                    phase = pod_info["status"].get("phase", "")
                    if phase == "Running":
                        t_running_end = time.time()
                        break
                except Exception:
                    pass
            time.sleep(1)
        else:
            print(f"Pod {pod_name} did not reach Running state in time.")
            results.append({"iteration": i+1, "status": "not_running", "apply_time": t_apply_end - t_apply_start})
            # Cleanup
            subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found"], capture_output=True)
            continue
        # Record times
        apply_time = t_apply_end - t_apply_start
        cold_start_time = t_running_end - t_apply_start
        print(f"Apply time: {apply_time:.2f}s, Cold start time: {cold_start_time:.2f}s")
        results.append({
            "iteration": i+1,
            "status": "success",
            "apply_time": apply_time,
            "cold_start_time": cold_start_time
        })
        # Delete pod
        subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", args.namespace, "--ignore-not-found"], capture_output=True)
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
