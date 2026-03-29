#!/bin/bash
# Build rayzor-gpu for all supported platforms and package as a single rpkg.
# Includes native dylibs + WASM host module (via wasm-pack).
#
# Usage:
#   cd gpu && ./pack-rpkg.sh              # current platform only (fast)
#   cd gpu && ./pack-rpkg.sh --cross      # all platforms via cross
#   cd gpu && ./pack-rpkg.sh --wasm-only  # WASM host only (no native)
#   rayzor rpkg install rayzor-gpu.rpkg
#
# Prerequisites for WASM: cargo install wasm-pack

set -e
cd "$(dirname "$0")"

RAYZOR="../target/release/rayzor"
HAXE_DIR="../compiler/haxe-std/rayzor/gpu"
FEATURES="webgpu-backend"
WASM_HOST_JS=""

# Build WASM host module (if wasm-pack is available)
build_wasm_host() {
    if command -v wasm-pack &>/dev/null; then
        echo "=== Building WASM host module (wasm-pack) ==="
        wasm-pack build \
            --target web \
            --no-default-features \
            --features wasm-host \
            --out-dir pkg \
            --out-name rayzor_gpu
        WASM_HOST_JS="--js-host rayzor-gpu=pkg/rayzor_gpu.js"
        echo "  WASM host: pkg/rayzor_gpu.js + pkg/rayzor_gpu_bg.wasm"
    else
        echo "  (wasm-pack not found — skipping WASM host build)"
    fi
}

if [ "$1" = "--wasm-only" ]; then
    echo "=== WASM-only build ==="
    build_wasm_host
    [ -z "$WASM_HOST_JS" ] && echo "Error: wasm-pack required for --wasm-only" && exit 1

    echo "Packaging rayzor-gpu.rpkg (WASM only)..."
    $RAYZOR rpkg pack \
        --name rayzor-gpu \
        $WASM_HOST_JS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-gpu.rpkg

elif [ "$1" = "--cross" ]; then
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

    build_wasm_host

    echo ""
    echo "Packaging rayzor-gpu.rpkg (all platforms + WASM)..."
    DYLIB_ARGS=""
    [ -f "$MACOS_ARM" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_ARM --os macos --arch aarch64"
    [ -f "$MACOS_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_X64 --os macos --arch x86_64"
    [ -f "$LINUX_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $LINUX_X64 --os linux --arch x86_64"
    [ -f "$WIN_X64" ]   && DYLIB_ARGS="$DYLIB_ARGS --dylib $WIN_X64 --os windows --arch x86_64"

    $RAYZOR rpkg pack \
        --name rayzor-gpu \
        $DYLIB_ARGS \
        $WASM_HOST_JS \
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

    build_wasm_host

    echo "Packaging rayzor-gpu.rpkg..."
    $RAYZOR rpkg pack \
        --name rayzor-gpu \
        --dylib "$DYLIB_PATH" \
        $WASM_HOST_JS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-gpu.rpkg
fi

echo ""
echo "Done: rayzor-gpu.rpkg"
echo "Install: rayzor rpkg install rayzor-gpu.rpkg"
