#!/bin/bash
# Build the rayzor-gpu crate and package it as an rpkg.
#
# Usage:
#   cd gpu && ./pack-rpkg.sh
#   rayzor rpkg install rayzor-gpu.rpkg
#
# This builds librayzor_gpu.dylib (or .so) with the webgpu backend,
# then bundles it with the Haxe stdlib files into rayzor-gpu.rpkg.

set -e

# Determine platform
case "$(uname -s)" in
    Darwin*) LIB_EXT="dylib" ;;
    Linux*)  LIB_EXT="so" ;;
    *)       echo "Unsupported platform"; exit 1 ;;
esac

# Build the GPU dylib
echo "Building rayzor-gpu with webgpu backend..."
cargo build -p rayzor-gpu --features webgpu-backend --release

DYLIB_PATH="../target/release/librayzor_gpu.${LIB_EXT}"
if [ ! -f "$DYLIB_PATH" ]; then
    echo "Error: $DYLIB_PATH not found"
    exit 1
fi

echo "Packaging rayzor-gpu.rpkg..."
# Use rayzor rpkg pack to create the package
../target/release/rayzor rpkg pack \
    --name rayzor-gpu \
    --dylib "$DYLIB_PATH" \
    --haxe-dir ../compiler/haxe-std/rayzor/gpu \
    --output rayzor-gpu.rpkg

echo "Done: rayzor-gpu.rpkg"
echo ""
echo "Install with:"
echo "  rayzor rpkg install rayzor-gpu.rpkg"
