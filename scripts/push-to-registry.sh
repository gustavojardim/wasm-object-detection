#!/bin/bash
# Package and push WASM inference app as OCI image using ORAS

set -e

# Always run from project root
cd "$(dirname "$0")/.."

REG="${REGISTRY:-192.168.0.113:32000}"
NAME="${IMAGE_NAME:-wasm-inference}"
TAG="${IMAGE_TAG:-latest}"

echo "Packaging WASM bundle for ${REG}/${NAME}:${TAG}..."

# Create package directory at project root
PKGDIR="package"
mkdir -p "$PKGDIR"
cd "$PKGDIR"


# Find or build app.wasm
if [ -f ../bundle/apps/app.wasm ]; then
  cp ../bundle/apps/app.wasm app.wasm
elif [ -f ../target/wasm32-wasip2/release/inference.wasm ]; then
  echo "[INFO] Using ../target/wasm32-wasip2/release/inference.wasm as app.wasm"
  cp ../target/wasm32-wasip2/release/inference.wasm app.wasm
else
  echo "[ERROR] Could not find app.wasm (../bundle/apps/app.wasm or ../target/wasm32-wasip2/release/inference.wasm)"
  exit 1
fi

mkdir -p models
if [ -f ../models/yolov8n_cpu.torchscript ]; then
  cp ../models/yolov8n_cpu.torchscript models/
fi

if [ -f ../models/yolov8n_cuda.torchscript ]; then
  cp ../models/yolov8n_cuda.torchscript models/
fi


# Create a single-layer tar with app.wasm and all model files
echo "Creating tar layer..."
tar -cf wasm-bundle.tar app.wasm models/yolov8n_cpu.torchscript models/yolov8n_cuda.torchscript
LAYER_SHA=$(sha256sum wasm-bundle.tar | awk '{print $1}')

# Create OCI config.json
echo "Creating OCI config..."
cat > config.json <<EOF
{
  "created": "$(date -u +%FT%TZ)",
  "architecture": "wasm",
  "os": "linux",
  "rootfs": { "type": "layers", "diff_ids": ["sha256:${LAYER_SHA}"] },
  "config": { "Entrypoint": ["/app.wasm"] },
  "annotations": { 
    "module.wasm.image/variant": "compat",
    "org.opencontainers.image.description": "WASM Object Detection with wasi-nn and PyTorch"
  }
}
EOF

# Push with ORAS
echo "Pushing to registry..."
oras push --plain-http ${REG}/${NAME}:${TAG} \
  --config config.json:application/vnd.oci.image.config.v1+json \
  wasm-bundle.tar:application/vnd.oci.image.layer.v1.tar

# Verify
echo ""
echo "Verifying push..."
oras manifest fetch --plain-http ${REG}/${NAME}:${TAG} | jq .

echo ""
echo "✅ Successfully pushed ${REG}/${NAME}:${TAG}"
echo ""
echo "To deploy to K8s:"
echo "  kubectl apply -f k8s-deployment.yaml"
