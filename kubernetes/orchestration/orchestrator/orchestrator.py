# Orchestrator for intelligent pod placement in Kubernetes
# Features:
# - Receives deployment requests from clients
# - Returns worker node probe endpoints
# - Accepts filtered node list from client
# - Selects best node based on resource metrics
# - Deploys inference pod to selected node

from flask import Flask, request, jsonify
from kubernetes import client, config
import os

app = Flask(__name__)

# Load Kubernetes config (in-cluster or local)
if os.getenv('KUBERNETES_SERVICE_HOST'):
    config.load_incluster_config()
    configuration = client.Configuration.get_default_copy()
    configuration.verify_ssl = False
    configuration.host = 'https://192.168.0.105:16443'
    v1 = client.CoreV1Api(client.ApiClient(configuration))
else:
    config.load_kube_config()
    v1 = client.CoreV1Api()

# 1. Receive deployment request
@app.route('/request-deploy', methods=['POST'])
def deploy():
    # 2. List worker nodes
    nodes = v1.list_node().items
    probe_endpoints = []
    for n in nodes:
        # Skip master node
        if n.metadata.name == 'master':
            continue
        # Get Internal IP
        ip = None
        for addr in n.status.addresses:
            if addr.type == 'InternalIP':
                ip = addr.address
                break
        if ip:
            probe_endpoints.append({'name': n.metadata.name, 'ip': ip, 'probe': f'http://{ip}:8080/ping'})
    # 3. Send node list to client
    return jsonify({'nodes': probe_endpoints})

# 4. Receive filtered node list
@app.route('/deploy', methods=['POST'])
def filtered_nodes():
    filtered = request.json['nodes']  # List of node names
    # 5. Collect resource metrics (CPU, memory, GPU)
    # For simplicity, use allocatable resources from node status
    nodes = v1.list_node().items
    metrics = []
    for n in nodes:
        if n.metadata.name in filtered:
            alloc = n.status.allocatable
            gpu = alloc.get('nvidia.com/gpu', '0')
            metrics.append({
                'name': n.metadata.name,
                'cpu': alloc.get('cpu', '0'),
                'memory': alloc.get('memory', '0'),
                'gpu': gpu
            })
    # 6. Select best node (prefer GPU, then most CPU/memory)
    gpu_nodes = [m for m in metrics if int(m['gpu']) > 0]
    if gpu_nodes:
        best = max(gpu_nodes, key=lambda m: (int(m['cpu']), int(m['memory'].rstrip('Ki'))))
    else:
        best = max(metrics, key=lambda m: (int(m['cpu']), int(m['memory'].rstrip('Ki'))))

    # 7. Deploy application
    pod_name = 'wasm-inference'
    namespace = 'default'
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name='wasm-inference',
            labels={'app': 'wasm-inference'},
            annotations={'module.wasm.image/variant': 'compat'}
        ),
        spec=client.V1PodSpec(
            runtime_class_name='wasmtime',
            restart_policy='Always',
            containers=[
                client.V1Container(
                    name='inference',
                    image='192.168.0.105:32000/wasm-inference:latest',
                    image_pull_policy='Always',
                    command=['app.wasm', '--tcp'],
                    ports=[
                        client.V1ContainerPort(container_port=8080, name='tcp', protocol='TCP'),
                        client.V1ContainerPort(container_port=8081, name='udp', protocol='UDP')
                    ],
                    resources=client.V1ResourceRequirements(
                        requests={'cpu': '500m', 'memory': '512Mi'},
                        limits={'cpu': '1', 'memory': '1Gi'}
                    )
                )
            ],
            node_name=best['name']
        )
    )

    v1.create_namespaced_pod(namespace=namespace, body=pod)
    
    return jsonify({'deployed_to': best['name']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
