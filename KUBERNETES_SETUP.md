# Kubernetes Setup (K3s Server + Worker Agents)

This guide explains how to use your existing K3s server and join machines as worker nodes.

Validated context:
- Existing K3s server already running on 5060 Ti machine
- Worker nodes on same network and reachable from server
- Date: 2026-02-27

---

## 1) On the server machine (K3s server)

Get the server IP and join token:

```bash
hostname -I
sudo cat /var/lib/rancher/k3s/server/node-token
```

You will use:
- `SERVER_IP` = internal IP of your 5060 Ti machine
- `NODE_TOKEN` = content of `/var/lib/rancher/k3s/server/node-token`

Confirm K3s server is healthy:

```bash
sudo systemctl status k3s --no-pager
sudo kubectl get nodes -o wide
```

---

## 2) Network and firewall requirements

Ensure nodes can reach the server:
- `6443/tcp` (Kubernetes API)
- `8472/udp` (Flannel VXLAN, default K3s CNI)

If you use a host firewall on the server, allow these ports.

Quick connectivity check from each worker node:

```bash
nc -zv <SERVER_IP> 6443
```

---

## 3) On each worker node machine (join as K3s agent)

From the project root on each node:

```bash
curl -sfL https://get.k3s.io | \
  INSTALL_K3S_VERSION=v1.34.4+k3s1 \
  K3S_URL=https://<SERVER_IP>:6443 \
  K3S_TOKEN=<SERVER_TOKEN> \
  INSTALL_K3S_EXEC="agent --node-name <NODE_NAME>" \
  sh -
```

Verify the agent service:

```bash
sudo systemctl status k3s-agent --no-pager
```

If needed, check logs:

```bash
sudo journalctl -u k3s-agent -f
```

---

## 4) Verify joined nodes from the server

On the 5060 Ti server:

```bash
sudo kubectl get nodes -o wide
sudo kubectl get nodes --show-labels
```

You should see each worker node in `Ready` state.

---

## 5) Use `kubectl` directly (without `k3s kubectl`)

On the K3s server (`gsjardim` control-plane), copy K3s kubeconfig to your user path:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$(id -u)":"$(id -g)" ~/.kube/config
chmod 600 ~/.kube/config
echo 'export KUBECONFIG=$HOME/.kube/config' >> ~/.bashrc
source ~/.bashrc
kubectl get nodes
```

Important:
- Do not use Markdown-style paths in shell commands (for example `[k3s.yaml](...)`).
- Use real filesystem path: `/etc/rancher/k3s/k3s.yaml`.

---

## 6) Enable local registry (NodePort `32000`)

This repository includes a registry manifest at `kubernetes/registry.yaml`.

### 6.1 Create namespace and deploy registry

```bash
kubectl create namespace container-registry --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f kubernetes/registry.yaml
kubectl -n container-registry get pods,svc
```

Note:
- `kubernetes/registry.yaml` currently pins `nodeName: gspadotto`.
- Change it to a real node name in your cluster (for example `gsjardim`) before applying.

### 6.2 Configure K3s/containerd on all nodes to trust the registry endpoint

On every node, create `/etc/rancher/k3s/registries.yaml`:

```yaml
mirrors:
  "<REGISTRY_HOST>:32000":
    endpoint:
      - "http://<REGISTRY_HOST>:32000"
```

For agent nodes, ensure the service points to that file via environment override:

```bash
sudo systemctl cat k3s-agent | grep -E "private-registry|EnvironmentFile"
sudo touch /etc/systemd/system/k3s-agent.service.env
echo 'K3S_PRIVATE_REGISTRY=/etc/rancher/k3s/registries.yaml' | sudo tee -a /etc/systemd/system/k3s-agent.service.env >/dev/null
sudo systemctl daemon-reload
```

If your distro layout exposes `/etc/rancher/node`, keep the canonical file at `/etc/rancher/k3s/registries.yaml` and use `K3S_PRIVATE_REGISTRY` as above so k3s-agent reads the correct path.

Then restart K3s service on each node:

- Server node:
```bash
sudo systemctl restart k3s
```

- Worker nodes:
```bash
sudo systemctl restart k3s-agent
```

### 6.3 Validate registry access

From any node:

```bash
curl http://<REGISTRY_HOST>:32000/v2/_catalog
```

Validate container runtime pull path (important):

```bash
sudo crictl pull <REGISTRY_HOST>:32000/wasm-inference:latest
```

If you see:

`http: server gave HTTP response to HTTPS client`

containerd is still using HTTPS; recheck section 6.2 and restart the service.

If you already have images pushed:

```bash
kubectl run reg-test --rm -it --image=<REGISTRY_HOST>:32000/wasm-inference:latest --restart=Never --command -- true
```

---

## 10) Common issues

### `kubectl` tries `localhost:8080`
- Usually `~/.kube/config` was not created or `KUBECONFIG` is unset.
- Re-run section 5 exactly.

### Cannot pull images from local registry
- Confirm registry service is up: `kubectl -n container-registry get svc registry`
- Confirm node containerd mirror config exists on every node: `/etc/rancher/k3s/registries.yaml`
- If pull shows `server gave HTTP response to HTTPS client`, set `K3S_PRIVATE_REGISTRY=/etc/rancher/k3s/registries.yaml` in `k3s-agent.service.env` and restart service.
- Restart K3s/K3s-agent after changing registry config.

---

## Quick command summary

Server token:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
sudo kubectl get nodes -o wide
```

Direct `kubectl` setup:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown "$(id -u)":"$(id -g)" ~/.kube/config
chmod 600 ~/.kube/config
echo 'export KUBECONFIG=$HOME/.kube/config' >> ~/.bashrc
source ~/.bashrc
kubectl get nodes
```

Registry enablement:

```bash
kubectl create namespace container-registry --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f kubernetes/registry.yaml
```
