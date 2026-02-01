import psutil
import time
import json
import sys
import os

 # GPU and VRAM statistics removed

def get_process_tree(parent):
    """
    Returns a dictionary {pid: process_obj} including the parent and all children.
    We use a dict to cache process objects so psutil can calculate CPU correctly over time.
    """
    try:
        children = parent.children(recursive=True)
        all_procs = [parent] + children
        return {p.pid: p for p in all_procs}
    except psutil.NoSuchProcess:
        return {}

def find_wasmtime_process(docker_mode=False):
    target_cmd = [
        'wasmtime', 'run', '-S', 'cli=y', '-S', 'nn=y', '-S', 'inherit-network=y', '-S', 'tcp=y',
        '--dir', '/app/models::/models', '/app/inference.wasm', '--device', 'cpu', '--udp'
    ]
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if docker_mode:
                cmdline = proc.info['cmdline']
                if cmdline == target_cmd:
                    return proc
            else:
                if 'wasmtime' in proc.info['name'] or \
                   any('wasmtime' in str(arg) for arg in proc.info['cmdline']):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor wasmtime or containerd-shim-wasmtime-v1 process resource usage.")
    parser.add_argument("--log-file", type=str, default="wasmtime_enhanced_log.jsonl", help="File to save logs (default: wasmtime_enhanced_log.jsonl)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--containerd", action="store_true", help="Monitor all containerd-shim-wasmtime-v1 processes (Kubernetes/Docker)")
    parser.add_argument("--docker", action="store_true", help="Strictly match the full wasmtime command line (for Docker)")
    args = parser.parse_args()

    interval = args.interval
    log_file = args.log_file
    use_containerd = args.containerd
    use_docker_mode = args.docker

    # Get System Core Count for Normalization
    logical_cores = psutil.cpu_count(logical=True)

    def find_containerd_wasmtime_procs():
        """
        Returns a list of psutil.Process objects for all containerd-shim-wasmtime-v1 processes.
        """
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'containerd-shim-wasmtime-v1' or \
                   (proc.info['cmdline'] and 'containerd-shim-wasmtime-v1' in ' '.join(proc.info['cmdline'])):
                    procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue
        return procs

    if use_containerd:
        procs = find_containerd_wasmtime_procs()
        if not procs:
            print("Error: No containerd-shim-wasmtime-v1 processes found.")
            sys.exit(1)
        print(f"Monitoring {len(procs)} containerd-shim-wasmtime-v1 processes | System Cores: {logical_cores}")
    else:
        proc = find_wasmtime_process(docker_mode=use_docker_mode)
        if not proc:
            print("Error: Wasmtime process not found.")
            sys.exit(1)
        print(f"Monitoring PID: {proc.pid} | System Cores: {logical_cores}")

    # Probe for GPU header
    gpu_header = False
    try:
        test_gpu = get_nvidia_gpu_stats()
        if test_gpu:
            gpu_header = True
    except Exception:
        pass
    print(f"{'Time':<10} | {'Core Load %':<12} | {'System %':<10} | {'Memory (MB)':<12}")
    print("-" * 48)

    known_procs = {}
    debug_interval = 30  # seconds between debug prints
    last_debug_time = time.time() + debug_interval  # suppress debug at startup
    with open(log_file, "a") as f:
        while True:
            try:
                # 1. Update the process tree (handle new workers spawning/dying)
                total_rss = 0
                total_cpu_cores = 0.0
                max_rss = 0
                per_proc_info = []
                if use_containerd:
                    shim_procs = find_containerd_wasmtime_procs()
                    for shim_proc in shim_procs:
                        try:
                            mem = shim_proc.memory_info().rss
                            cpu = shim_proc.cpu_percent(interval=None)
                            cmdline = ' '.join(shim_proc.cmdline())
                            if mem > max_rss:
                                max_rss = mem
                            total_cpu_cores += cpu
                            per_proc_info.append((shim_proc.pid, cmdline, mem, cpu))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    total_rss = max_rss
                else:
                    current_tree = get_process_tree(proc)
                    known_pids = set(known_procs.keys())
                    current_pids = set(current_tree.keys())
                    for pid in known_pids - current_pids:
                        del known_procs[pid]
                    for pid in current_pids - known_pids:
                        p = current_tree[pid]
                        try:
                            p.cpu_percent(interval=None)
                            known_procs[pid] = p
                        except:
                            pass
                    for p in known_procs.values():
                        try:
                            cmdline = ' '.join(p.cmdline())
                            mem = p.memory_info().rss
                            cpu = p.cpu_percent(interval=None)
                            if 'wasmtime' in cmdline and 'containerd-shim-wasmtime-v1' not in cmdline:
                                if mem > max_rss:
                                    max_rss = mem
                                total_cpu_cores += cpu
                            per_proc_info.append((p.pid, cmdline, mem, cpu))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    total_rss = max_rss

                # 4. Calculate Normalized Usage
                system_usage_percent = total_cpu_cores / logical_cores if logical_cores else 0

                # 5. Output
                timestamp = time.time()
                mem_mb = total_rss / (1024 * 1024)

                # Console Table (no GPU/VRAM)
                print(f"{time.strftime('%H:%M:%S', time.localtime(timestamp)):<10} | "
                      f"{total_cpu_cores:>6.1f}%      | "
                      f"{system_usage_percent:>6.1f}%    | "
                      f"{mem_mb:>8.1f} MB")

                # File Log
                log = {
                    "timestamp": timestamp,
                    "memory_bytes": total_rss,
                    "cpu_cores_sum": round(total_cpu_cores, 2),
                    "cpu_system_pct": round(system_usage_percent, 2)
                }
                f.write(json.dumps(log) + "\n")
                f.flush()

                time.sleep(interval)

            except psutil.NoSuchProcess:
                print("\nTarget process terminated.")
                break