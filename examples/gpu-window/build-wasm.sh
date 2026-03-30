#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
RAYZOR="$ROOT/target/release/rayzor"
GPU_PKG="$ROOT/gpu/pkg"

# 1. Build the compiler (if needed)
if [ ! -f "$RAYZOR" ] || [ "$ROOT/src/main.rs" -nt "$RAYZOR" ]; then
  echo "Building rayzor compiler..."
  (cd "$ROOT" && cargo build --release)
fi

# 2. Build GPU wasm host (if needed)
if [ ! -f "$GPU_PKG/rayzor_gpu.js" ] || [ "$ROOT/gpu/src/wasm_exports.rs" -nt "$GPU_PKG/rayzor_gpu.js" ]; then
  echo "Building GPU wasm host module..."
  RUSTC="$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin/rustc"
  CARGO="$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin/cargo"
  (cd "$ROOT/gpu" && RUSTC="$RUSTC" "$CARGO" build --target wasm32-unknown-unknown --no-default-features --features wasm-host)
  wasm-bindgen --target web --out-dir "$GPU_PKG" "$ROOT/target/wasm32-unknown-unknown/debug/rayzor_gpu.wasm"
  echo "GPU host: $(grep -c 'export function' "$GPU_PKG/rayzor_gpu.js") exports"
fi

# 3. Compile Haxe → WASM
echo "Compiling Main.hx → WASM..."
"$RAYZOR" build --target wasm --browser Main.hx

# 4. Serve
BUILD=".rayzor/build"
echo ""
echo "Build output:"
ls -lh "$BUILD"/Main.core.wasm "$BUILD"/Main.js "$BUILD"/Main.html
echo ""
echo "Serving at http://localhost:8080/Main.html"
echo "Press Ctrl+C to stop"
cd "$BUILD" && python3 -m http.server 8080
