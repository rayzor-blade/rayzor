#!/bin/bash
# Build the rayzor WASM runtime library.
# Output: target/wasm32-wasip1/release/rayzor_runtime_wasm.wasm
set -e

echo "Building rayzor-runtime-wasm for wasm32-wasip1..."
cargo build -p rayzor-runtime-wasm --target wasm32-wasip1 --release

WASM="target/wasm32-wasip1/release/rayzor_runtime_wasm.wasm"
if [ -f "../$WASM" ]; then
    SIZE=$(wc -c < "../$WASM" | tr -d ' ')
    echo "  wrote $WASM ($((SIZE / 1024)) KB)"
else
    echo "Error: $WASM not found"
    exit 1
fi
