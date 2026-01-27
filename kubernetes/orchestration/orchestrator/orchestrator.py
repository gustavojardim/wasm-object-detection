# Per-node pod limits
NODE_POD_LIMITS = {
    "gjardim": 2,
    "worker1": 1,
    "gspadotto": 2
}
from flask import Flask, request, jsonify
import logging
import time
from kubernetes import client, config
import os
import socket
import struct
import cv2
import numpy as np
import threading
# Port pool config (per node)
UDP_PORT_RANGE = range(30081, 30101)  # 20 ports per node
port_pool = {}  # {node_name: {port: pod_name or None}}

def get_free_udp_port(node_name):
    if node_name not in port_pool:
        port_pool[node_name] = {port: None for port in UDP_PORT_RANGE}
    for port, pod in port_pool[node_name].items():
        if pod is None:
            return port
    return None

def mark_port_used(node_name, port, pod_name):
    port_pool.setdefault(node_name, {p: None for p in UDP_PORT_RANGE})
    port_pool[node_name][port] = pod_name

def mark_port_free(node_name, port):
    if node_name in port_pool and port in port_pool[node_name]:
        port_pool[node_name][port] = None

def schedule_pod_deletion(pod_name, namespace, node_name, port, delay=120):
    def deleter():
        time.sleep(delay)
        try:
            v1.delete_namespaced_pod(pod_name, namespace)
            mark_port_free(node_name, port)
            logging.info(f"Pod {pod_name} deleted after TTL, port {port} freed.")
        except Exception as e:
            logging.error(f"Failed to delete pod {pod_name}: {e}")
    threading.Thread(target=deleter, daemon=True).start()

app = Flask(__name__)

# Updated Logging: Included placeholders for your context tags

# Custom logging filter to always provide a default request_id
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return True

log_formatter = logging.Formatter('%(asctime)s %(levelname)s [ID:%(request_id)s] %(message)s')
log_handler = logging.FileHandler('orchestrator_deployments.log')
log_handler.setFormatter(log_formatter)
log_handler.addFilter(RequestIdFilter())
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = []
root_logger.addHandler(log_handler)

# Configuration Setup
if os.getenv('KUBERNETES_SERVICE_HOST'):
    config.load_incluster_config()
    configuration = client.Configuration.get_default_copy()
    configuration.verify_ssl = False
    configuration.host = 'https://192.168.0.113:16443'
    v1 = client.CoreV1Api(client.ApiClient(configuration))
else:
    config.load_kube_config()
    v1 = client.CoreV1Api()

def parse_resource(val):
    """Handles Ki, Mi, Gi (binary) and m, k, M, G (decimal) units."""
    if not val: return 0
    val = str(val)
    units = {
        'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
        'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3, 'Ti': 1024**4
    }
    for unit, factor in units.items():
        if val.endswith(unit):
            return float(val[:-len(unit)]) * factor
    return float(val)

@app.route('/request-deploy', methods=['POST'])
def request_deploy():
    data = request.json or {}
    latency_threshold = data.get('latency_threshold', 100)  # ms
    nodes = v1.list_node().items
    probe_endpoints = []
    for n in nodes:
        if n.metadata.name == 'master': continue
        ip = next((addr.address for addr in n.status.addresses if addr.type == 'InternalIP'), None)
        if ip:
            probe_endpoints.append({
                'name': n.metadata.name, 
                'ip': ip, 
                'probe': f'http://{ip}:8080/ping'
            })
    logging.info(f"/request-deploy called, returning {len(probe_endpoints)} nodes, latency_threshold={latency_threshold}")
    return jsonify({'nodes': probe_endpoints, 'latency_threshold': latency_threshold})

@app.route('/deploy', methods=['POST'])
def filtered_nodes():
    data = request.json
    filtered = data.get('nodes', [])
    request_id = data.get('request_id', 'N/A')
    client_id = data.get('client', 'unknown')
    latency_threshold = data.get('latency_threshold', 50) # ms
    log_extra = {'request_id': request_id}
    deploy_start = time.time()

    try:
        # 1. Fetch Node Info
        all_nodes = v1.list_node().items
        # 2. Get IPs for filtered nodes
        node_ip_map = {}
        for n in all_nodes:
            if n.metadata.name in filtered:
                ip = next((addr.address for addr in n.status.addresses if addr.type == 'InternalIP'), None)
                if ip:
                    node_ip_map[n.metadata.name] = ip
        # 2.5. Count running inference pods per node and filter by per-node limit
        pods = v1.list_namespaced_pod(namespace='default', label_selector='app=wasm-inference')
        node_pod_counts = {}
        for pod in pods.items:
            node = pod.spec.node_name
            phase = pod.status.phase
            if node and phase not in ('Succeeded', 'Failed', 'Terminating', 'Error'):
                node_pod_counts[node] = node_pod_counts.get(node, 0) + 1
        eligible_nodes = []
        for name in node_ip_map:
            max_pods = NODE_POD_LIMITS.get(name, 1)
            running = node_pod_counts.get(name, 0)
            if running < max_pods:
                eligible_nodes.append(name)
            else:
                logging.info(f"Node {name} at pod limit ({running}/{max_pods}), skipping", extra=log_extra)
        if not eligible_nodes:
            logging.warning(f"No nodes available under pod limits. Current counts: {node_pod_counts}", extra=log_extra)
            return jsonify({'error': 'No nodes available under pod limits', 'status': 'failed', 'pod_counts': node_pod_counts}), 400
        # 3. Measure latency for each eligible node
        node_latencies = []
        for name in eligible_nodes:
            ip = node_ip_map[name]
            logging.info(f"[UDP TEST] Using {ip} for node {name}", extra=log_extra)
            try:
                latency = measure_qos_monitor_latency(ip)
                if latency is not None:
                    latency_ms = latency * 1000
                    node_latencies.append({'name': name, 'ip': ip, 'latency_ms': latency_ms})
                    logging.info(f"Node {name} latency: {latency_ms:.1f} ms", extra=log_extra)
                else:
                    logging.info(f"Node {name} did not respond to latency test", extra=log_extra)
            except Exception as e:
                logging.warning(f"Latency test failed for node {name}: {e}", extra=log_extra)
        logging.info(f"Measured latencies: {node_latencies}", extra=log_extra)
        # 4. Filter nodes with latency <= threshold
        eligible = [n for n in node_latencies if n['latency_ms'] <= latency_threshold]
        if not eligible:
            logging.warning(f"No nodes with latency <= {latency_threshold}ms. All latencies: {node_latencies}", extra=log_extra)
            return jsonify({'error': f'No nodes with latency <= {latency_threshold}ms', 'status': 'failed', 'latencies': node_latencies}), 400
        # 5. Select node with lowest latency
        best = min(eligible, key=lambda n: n['latency_ms'])
        logging.info(f"Selected node: {best['name']} (latency: {best['latency_ms']:.1f} ms)", extra=log_extra)

        # 6. Assign UDP port for this node
        udp_port = get_free_udp_port(best['name'])
        if udp_port is None:
            logging.error(f"No free UDP ports available for node {best['name']}", extra=log_extra)
            return jsonify({'error': f'No free UDP ports available for node {best["name"]}', 'status': 'failed'}), 400
        pod_name = f"wasm-inference-{client_id}-{int(time.time())}"
        mark_port_used(best['name'], udp_port, pod_name)

        # 7. Pod Definition (hostNetwork)
        pod_spec = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                labels={'app': 'wasm-inference', 'client': client_id},
                annotations={'module.wasm.image/variant': 'compat'}
            ),
            spec=client.V1PodSpec(
                runtime_class_name='wasmtime',
                node_name=best['name'],
                host_network=True,
                restart_policy='Never',
                containers=[client.V1Container(
                    name='inference',
                    image='192.168.0.113:32000/wasm-inference:latest',
                    command=['app.wasm', '--device', 'cpu', '--udp', '--port', str(udp_port)],
                    ports=[client.V1ContainerPort(container_port=udp_port, protocol='UDP')]
                )]
            )
        )
        v1.create_namespaced_pod(namespace='default', body=pod_spec)
        # 8. Wait for Readiness
        success = False
        for _ in range(60):
            status = v1.read_namespaced_pod_status(pod_name, 'default')
            if status.status.phase == 'Running':
                success = True
                break
            time.sleep(1)
        total_time = time.time() - deploy_start
        if success:
            # Fetch pod IP after it is running
            pod_status = v1.read_namespaced_pod_status(pod_name, 'default')
            pod_ip = pod_status.status.pod_ip if pod_status.status and pod_status.status.pod_ip else None
            node_ip = best['ip'] if 'ip' in best else node_ip_map.get(best['name'])
            logging.info(f"Deployment Successful: {pod_name} on {best['name']} in {total_time:.2f}s (pod_ip={pod_ip}, node_ip={node_ip}, udp_port={udp_port})", extra=log_extra)
            # Schedule pod deletion and port release
            schedule_pod_deletion(pod_name, 'default', best['name'], udp_port, delay=45)
            return jsonify({
                'status': 'success',
                'node': best['name'],
                'node_ip': node_ip,
                'pod': pod_name,
                'pod_ip': pod_ip,
                'udp_port': udp_port,
                'time': total_time,
                'latency_ms': best['latency_ms']
            })
    except Exception as e:
        logging.error(f"Deployment crashed: {str(e)}", extra=log_extra)
        return jsonify({'error': str(e)}, 500)

def measure_qos_monitor_latency(node_ip, udp_port=8081, image_path="/app/samples/image.png", timeout=5.0):
    """Send image.png to qos-monitor via UDP and return inference latency in seconds (or None on timeout)."""
    MTU = 1400
    HEADER_FMT = '!IHH'
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    MAX_PAYLOAD = MTU - HEADER_SIZE
    # Read and encode image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (640, 640))
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 30]
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    frame_bytes = buffer.tobytes()
    # Fragment
    chunks = [frame_bytes[i:i+MAX_PAYLOAD] for i in range(0, len(frame_bytes), MAX_PAYLOAD)]
    total_chunks = len(chunks)
    frame_id = 1  # static for test
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (node_ip, udp_port)
    logging.info(f"[UDP TEST] Sending to {node_ip}:{udp_port} - {total_chunks} fragments, frame size: {len(frame_bytes)} bytes")
    for idx, chunk in enumerate(chunks):
        header = struct.pack(HEADER_FMT, frame_id, idx, total_chunks)
        packet = header + chunk
        logging.info(f"[UDP TEST] Fragment {idx+1}/{total_chunks} to {node_ip}:{udp_port}, size: {len(packet)} bytes")
        sock.sendto(packet, server_addr)
    # Receive response and measure latency
    sock.settimeout(timeout)
    import time as _time
    start = _time.time()
    try:
        data, _ = sock.recvfrom(1024 * 1024)
        latency = _time.time() - start
        # Optionally parse detections: detections = json.loads(data.decode('utf-8'))
        return latency
    except socket.timeout:
        return None
    finally:
        sock.close()


# Ensure the Flask server runs when this script is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)