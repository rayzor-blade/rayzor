#!/bin/bash
# Build the rayzor-numa rpkg and run the demo.
#
# Mirrors examples/gpu-window/build-wasm.sh — keeps the built rpkg out of
# the workspace's git history while letting `rayzor run` resolve it via
# the manifest's [dependencies] table.

set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(cd ../.. && pwd)"
RAYZOR="$ROOT/target/release/rayzor"
NUMA_DYLIB="$ROOT/target/release/librayzor_numa.dylib"
NUMA_RPKG="$ROOT/numa/rayzor-numa.rpkg"

# 1. Build the compiler + numa crate if stale.
if [ ! -f "$RAYZOR" ] || [ "$ROOT/src/main.rs" -nt "$RAYZOR" ]; then
  echo "Building rayzor compiler..."
  (cd "$ROOT" && cargo build --release)
fi
if [ ! -f "$NUMA_DYLIB" ] || [ "$ROOT/numa/src/lib.rs" -nt "$NUMA_DYLIB" ]; then
  echo "Building rayzor-numa crate..."
  (cd "$ROOT" && cargo build --release -p rayzor-numa)
fi

# 2. Repack the rpkg if any input is newer.
needs_pack=0
if [ ! -f "$NUMA_RPKG" ]; then
  needs_pack=1
elif [ "$NUMA_DYLIB" -nt "$NUMA_RPKG" ]; then
  needs_pack=1
elif find "$ROOT/numa/haxe" -name '*.hx' -newer "$NUMA_RPKG" 2>/dev/null | grep -q .; then
  needs_pack=1
fi

if [ "$needs_pack" -eq 1 ]; then
  echo "Packing rayzor-numa.rpkg..."
  "$RAYZOR" rpkg pack \
    --haxe-dir "$ROOT/numa/haxe" \
    --dylib "$NUMA_DYLIB" \
    --output "$NUMA_RPKG" \
    --name rayzor-numa
fi

# 3. Run the demo.
echo
echo "Running numa-pool demo..."
"$RAYZOR" run Main.hx
