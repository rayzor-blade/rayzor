#!/bin/bash
# Build rayzor-gpu for all supported platforms and package as a single rpkg.
#
# Usage:
#   cd gpu && ./pack-rpkg.sh              # current platform only (fast)
#   cd gpu && ./pack-rpkg.sh --cross      # all platforms via cross
#   rayzor rpkg install rayzor-gpu.rpkg

set -e
cd "$(dirname "$0")"

RAYZOR="../target/release/rayzor"
HAXE_DIR="../compiler/haxe-std/rayzor/gpu"
FEATURES="webgpu-backend"

if [ "$1" = "--cross" ]; then
    echo "=== Cross-building rayzor-gpu for all platforms ==="

    # macOS aarch64 (native — cross can't do macOS)
    echo "[1/4] macOS aarch64..."
    cargo build -p rayzor-gpu --features "$FEATURES" --release --target aarch64-apple-darwin
    MACOS_ARM="../target/aarch64-apple-darwin/release/librayzor_gpu.dylib"

    # macOS x86_64
    echo "[2/4] macOS x86_64..."
    cargo build -p rayzor-gpu --features "$FEATURES" --release --target x86_64-apple-darwin
    MACOS_X64="../target/x86_64-apple-darwin/release/librayzor_gpu.dylib"

    # Linux x86_64
    echo "[3/4] Linux x86_64..."
    cross build -p rayzor-gpu --features "$FEATURES" --release --target x86_64-unknown-linux-gnu
    LINUX_X64="../target/x86_64-unknown-linux-gnu/release/librayzor_gpu.so"

    # Windows x86_64
    echo "[4/4] Windows x86_64..."
    cross build -p rayzor-gpu --features "$FEATURES" --release --target x86_64-pc-windows-gnu
    WIN_X64="../target/x86_64-pc-windows-gnu/release/rayzor_gpu.dll"

    echo ""
    echo "Packaging rayzor-gpu.rpkg (all platforms)..."
    DYLIB_ARGS=""
    [ -f "$MACOS_ARM" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_ARM --os macos --arch aarch64"
    [ -f "$MACOS_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_X64 --os macos --arch x86_64"
    [ -f "$LINUX_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $LINUX_X64 --os linux --arch x86_64"
    [ -f "$WIN_X64" ]   && DYLIB_ARGS="$DYLIB_ARGS --dylib $WIN_X64 --os windows --arch x86_64"

    $RAYZOR rpkg pack \
        --name rayzor-gpu \
        $DYLIB_ARGS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-gpu.rpkg
else
    echo "=== Building rayzor-gpu (current platform) ==="
    cargo build -p rayzor-gpu --features "$FEATURES" --release

    case "$(uname -s)" in
        Darwin*) LIB_EXT="dylib" ;;
        Linux*)  LIB_EXT="so" ;;
        *)       echo "Unsupported platform"; exit 1 ;;
    esac

    DYLIB_PATH="../target/release/librayzor_gpu.${LIB_EXT}"
    [ ! -f "$DYLIB_PATH" ] && echo "Error: $DYLIB_PATH not found" && exit 1

    echo "Packaging rayzor-gpu.rpkg..."
    $RAYZOR rpkg pack \
        --name rayzor-gpu \
        --dylib "$DYLIB_PATH" \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-gpu.rpkg
fi

echo ""
echo "Done: rayzor-gpu.rpkg"
echo "Install: rayzor rpkg install rayzor-gpu.rpkg"
