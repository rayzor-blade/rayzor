#!/bin/bash
# Build the GPU crate as a WASM host module for browser @:jsImport
# Output: pkg/rayzor_gpu.js + rayzor_gpu_bg.wasm
#
# Prerequisites: wasm-pack (cargo install wasm-pack)
#
# Usage:
#   cd gpu/
#   ./build-wasm-host.sh
#
# The output in pkg/ can be:
#   1. Packed into an rpkg: rayzor rpkg pack --js-host rayzor-gpu=pkg/rayzor_gpu.js ...
#   2. Used directly in a project: [wasm] hosts = { "rayzor-gpu" = "gpu/pkg/rayzor_gpu.js" }

set -e

echo "Building rayzor-gpu WASM host module..."
wasm-pack build \
  --target web \
  --no-default-features \
  --features wasm-host \
  --out-dir pkg \
  --out-name rayzor_gpu

echo "Output:"
ls -la pkg/rayzor_gpu*.{js,wasm} 2>/dev/null
echo ""
echo "Pack into rpkg:"
echo "  rayzor rpkg pack --js-host rayzor-gpu=pkg/rayzor_gpu.js --haxe-dir ../compiler/haxe-std/rayzor/gpu/ -o rayzor-gpu.rpkg"
