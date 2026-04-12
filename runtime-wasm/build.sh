#!/bin/bash
# Build the rayzor WASM runtime library with shared memory support.
# Output: target/wasm32-wasip1-threads/release/rayzor_runtime_wasm.wasm
#
# Shared memory (SharedArrayBuffer) is required for real Thread.spawn
# parallelism in the browser: the compiled module declares its memory
# section with the `shared` flag, which is what makes the Web Worker
# pool actually run Rayzor threads concurrently via Atomics.wait/notify.
#
# Target: wasm32-wasip1-threads
#   Same WASI API surface as wasip1 but ships an atomics-enabled
#   wasi-libc sysroot, so `+atomics` target feature links cleanly.
set -e

echo "Building rayzor-runtime-wasm for wasm32-wasip1-threads (shared memory)..."
cd "$(dirname "$0")"
cargo +nightly build \
  --target wasm32-wasip1-threads \
  --release

WASM="target/wasm32-wasip1-threads/release/rayzor_runtime_wasm.wasm"
if [ -f "$WASM" ]; then
    SIZE=$(wc -c < "$WASM" | tr -d ' ')
    echo "  wrote runtime-wasm/$WASM ($((SIZE / 1024)) KB)"
else
    echo "Error: $WASM not found"
    exit 1
fi
