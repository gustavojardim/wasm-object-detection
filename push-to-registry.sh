#!/bin/bash
# Package and push WASM inference app as OCI image using ORAS

set -e

cd "$(dirname "$0")"

REG="${REGISTRY:-192.168.0.105:32000}"
NAME="${IMAGE_NAME:-wasm-inference}"
TAG="${IMAGE_TAG:-latest}"

echo "Packaging WASM bundle for ${REG}/${NAME}:${TAG}..."

# Change to bundle directory
cd bundle

# Create a single-layer tar with both app.wasm and model
echo "Creating tar layer..."
tar -cf wasm-bundle.tar app.wasm models/yolov8n.torchscript
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

# Cleanup
rm -f wasm-bundle.tar config.json

echo ""
echo "✅ Successfully pushed ${REG}/${NAME}:${TAG}"
echo ""
echo "To deploy to K8s:"
echo "  kubectl apply -f k8s-deployment.yaml"
