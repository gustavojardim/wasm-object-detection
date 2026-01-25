#!/usr/bin/env python3
"""
Aggregate and report deployment metrics from multi_client_deploy_results.json.
"""
import json

with open("multi_client_deploy_results.json") as f:
    results = json.load(f)

total = len(results)
success = [r for r in results if r["status"] == "success"]
failed = [r for r in results if r["status"] != "success"]

print(f"Total requests: {total}")
print(f"Success: {len(success)}")
print(f"Failed: {len(failed)}")

if success:
    avg_time = sum(r["deploy_time"] for r in success if r["deploy_time"] is not None) / len(success)
    min_time = min(r["deploy_time"] for r in success if r["deploy_time"] is not None)
    max_time = max(r["deploy_time"] for r in success if r["deploy_time"] is not None)
    print(f"Average deploy time: {avg_time:.2f}s")
    print(f"Min deploy time: {min_time:.2f}s")
    print(f"Max deploy time: {max_time:.2f}s")

if failed:
    print("\nFailures:")
    for r in failed:
        print(f"Client {r['client_id']}: {r.get('error','unknown error')}")
