# K8s Deployment Guide - WASM Object Detection

## Prerequisites

1. **K8s cluster with custom containerd-shim-wasmtime**
   - Your cluster already has the custom shim with wasi-nn + PyTorch support
   - RuntimeClass `wasmtime` configured

2. **Container registry**
   - Local registry at `192.168.0.118:32000`

3. **ORAS CLI** for pushing OCI-compliant WASM packages
   ```bash
   oras version
   ```

## Step 1: Package and Push WASM Bundle as OCI Image

### Option A: Using the Script (Recommended)

```bash
cd /home/gjardim/wasm-object-detection
./push-to-registry.sh

# Or with custom settings
REGISTRY=192.168.0.118:32000 IMAGE_TAG=v1.0 ./push-to-registry.sh
```

### Option B: Manual Steps

```bash
cd /home/gjardim/wasm-object-detection/bundle

# Create tar layer with both app.wasm and model
tar -cf wasm-bundle.tar -C apps app.wasm -C ../models yolov8n.torchscript
LAYER_SHA=$(sha256sum wasm-bundle.tar | awk '{print $1}')

# Create OCI config.json
cat > config.json <<EOF
{
  "created": "$(date -u +%FT%TZ)",
  "architecture": "wasm",
  "os": "wasip2",
  "rootfs": { "type": "layers", "diff_ids": ["sha256:${LAYER_SHA}"] },
  "config": { "Entrypoint": ["/app.wasm"] },
  "annotations": { "module.wasm.image/variant": "compat" }
}
EOF

# Push with ORAS
REG=192.168.0.118:32000
NAME=wasm-inference
TAG=latest

oras push --plain-http ${REG}/${NAME}:${TAG} \
  --config config.json:application/vnd.oci.image.config.v1+json \
  wasm-bundle.tar:application/vnd.oci.image.layer.v1.tar

# Verify
oras manifest fetch --plain-http ${REG}/${NAME}:${TAG} | jq .

# Cleanup
rm -f wasm-bundle.tar config.json
```

## Step 2: Deploy to K8s

```bash
# Apply the deployment
kubectl apply -f k8s-deployment.yaml

# Check pod status
kubectl get pods -l app=wasm-inference

# Check logs
kubectl logs wasm-inference -f

# Check service
kubectl get svc wasm-inference
```

## Step 3: Test the Deployment

```bash
# Get node IP (if using NodePort)
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

# Test with single image
python3 test_tcp_client.py samples/image.png $NODE_IP 30080

# Test with video
python3 test_tcp_video.py samples/walking_people_hd.mp4 --host $NODE_IP --port 30080
```

## Architecture

**WASM Component (wasm32-wasip2):**
- Single binary: `inference.wasm` (1.9MB)
- TCP server on port 8080
- wasi-nn for PyTorch inference
- YOLOv8n model (13MB)

**Runtime:**
- containerd-shim-wasmtime with wasi-nn support
- Wasmtime flags: `-S cli=y -S nn=y -S nn-graph=pytorch::/opt/wasm/models -S inherit-network=y -S tcp=y`

**Model:**
- YOLOv8n TorchScript
- Loaded via wasi-nn graph API
- CPU/GPU fallback support

## Troubleshooting

### Verify Registry Contents
```bash
# List tags
oras repo tags --plain-http 192.168.0.118:32000/wasm-inference

# View manifest
oras manifest fetch --plain-http 192.168.0.118:32000/wasm-inference:latest | jq .

# Fetch config blob (get digest from manifest first)
oras blob fetch --plain-http 192.168.0.118:32000/wasm-inference:latest sha256:<config-digest> | jq .
```

### Check Pod Status
```bash
kubectl describe pod wasm-inference
```

### View Logs
```bash
# Init container logs (ORAS pull)
kubectl logs wasm-inference -c oras-pull

# Main container logs
kubectl logs wasm-inference -c inference -f
```

### Exec into Pod (if needed)
```bash
# Note: Using scratch image, so no shell available
# Check from node directly:
crictl ps | grep wasm-inference
crictl logs <container-id>
```

### Common Issues

**Issue: Pod in CrashLoopBackOff**
- Check RuntimeClass is configured: `kubectl get runtimeclass wasmtime`
- Check containerd-shim-wasmtime is installed on nodes
- Check logs: `kubectl logs wasm-inference -c inference`

**Issue: ORAS pull fails**
- Verify registry is accessible: `curl http://192.168.0.118:32000/v2/_catalog`
- Check ORAS is installed on nodes: `oras version`
- Check bundle was pushed correctly

**Issue: Model loading fails**
- Check model file in bundle: Size should be ~13MB
- Check wasi-nn backend logs in pod
- Verify PyTorch is available in containerd-shim

**Issue: Cannot connect to service**
- Check service: `kubectl get svc wasm-inference`
- Check pod is running: `kubectl get pods`
- Test from within cluster first: `kubectl run test --rm -it --image=debian -- bash`
  ```bash
  apt update && apt install -y netcat-openbsd
  nc -zv wasm-inference 8080
  ```

## Performance

**Local Testing:**
- Single image: ~140ms inference time
- Video stream: 6-7 FPS (720p)

**Expected K8s Performance:**
- Similar to local (network overhead minimal for TCP)
- Can scale horizontally with multiple pods
- Consider using GPU nodes for better performance

## Scaling

```bash
# Create deployment instead of pod for scaling
kubectl create deployment wasm-inference --image=scratch --replicas=3 --dry-run=client -o yaml > deployment.yaml

# Then add RuntimeClass and other specs from k8s-deployment.yaml
```

## Updating

```bash
# Rebuild WASM
cd inference
cargo build --target wasm32-wasip2 --release

# Update bundle
cp ../target/wasm32-wasip2/release/inference.wasm ../bundle/apps/app.wasm

# Push new version
cd ..
IMAGE_TAG=v1.1 ./push-to-registry.sh

# Update deployment (if using versioned tag, update yaml first)
kubectl delete pod wasm-inference
kubectl apply -f k8s-deployment.yaml

# Or if using Deployment with rolling update:
kubectl set image deployment/wasm-inference inference=192.168.0.118:32000/wasm-inference:v1.1
```

## Bundle Contents

The OCI image contains:
- `/app.wasm` - WASM inference application (1.9MB, wasm32-wasip2)
- `/yolov8n.torchscript` - YOLOv8n model (13MB)

Total image size: ~15MB (much smaller than traditional container images!)

## Architecture Notes

**WASM Component (wasm32-wasip2):**
- Compiled with wit-bindgen using official wasi-nn WIT definitions
- Uses TCP sockets via WASI preview 2
- Loads model via wasi-nn graph API with PyTorch backend
- GPU/CPU fallback support

**Runtime:**
- containerd-shim-wasmtime with wasi-nn + PyTorch support
- Model loaded from `/yolov8n.torchscript` path
- Network inherited for TCP server on port 8080

**OCI Package:**
- Single tar layer with app.wasm + model
- `module.wasm.image/variant: "compat"` annotation for containerd
- Entrypoint: `/app.wasm`
- Architecture: `wasm`, OS: `wasip2`
