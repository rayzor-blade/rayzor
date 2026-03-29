#!/bin/bash
# Build rayzor-window for all supported platforms and package as a single rpkg.
# Includes native dylibs + WASM host module (via wasm-pack).
#
# Usage:
#   cd window && ./pack-rpkg.sh              # current platform + WASM (fast)
#   cd window && ./pack-rpkg.sh --cross      # all platforms + WASM
#   cd window && ./pack-rpkg.sh --wasm-only  # WASM host only
#   rayzor rpkg install rayzor-window.rpkg
#
# Prerequisites for WASM: cargo install wasm-pack

set -e
cd "$(dirname "$0")"

RAYZOR="../target/release/rayzor"
HAXE_DIR="../compiler/haxe-std/rayzor/window"
WASM_HOST_JS=""

# Build WASM host module
build_wasm_host() {
    if command -v wasm-pack &>/dev/null; then
        echo "=== Building WASM host module (wasm-pack) ==="
        wasm-pack build \
            --target web \
            --no-default-features \
            --features wasm-host \
            --out-dir pkg \
            --out-name rayzor_window
        WASM_HOST_JS="--js-host rayzor-window=pkg/rayzor_window.js"
        echo "  WASM host: pkg/rayzor_window.js + pkg/rayzor_window_bg.wasm"
    else
        echo "  (wasm-pack not found — skipping WASM host build)"
    fi
}

if [ "$1" = "--wasm-only" ]; then
    echo "=== WASM-only build ==="
    build_wasm_host
    [ -z "$WASM_HOST_JS" ] && echo "Error: wasm-pack required for --wasm-only" && exit 1

    echo "Packaging rayzor-window.rpkg (WASM only)..."
    $RAYZOR rpkg pack \
        --name rayzor-window \
        $WASM_HOST_JS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-window.rpkg

elif [ "$1" = "--cross" ]; then
    echo "=== Cross-building rayzor-window for all platforms ==="

    echo "[1/4] macOS aarch64..."
    cargo build -p rayzor-window --release --target aarch64-apple-darwin
    MACOS_ARM="../target/aarch64-apple-darwin/release/librayzor_window.dylib"

    echo "[2/4] macOS x86_64..."
    cargo build -p rayzor-window --release --target x86_64-apple-darwin
    MACOS_X64="../target/x86_64-apple-darwin/release/librayzor_window.dylib"

    echo "[3/4] Linux x86_64..."
    cross build -p rayzor-window --release --target x86_64-unknown-linux-gnu
    LINUX_X64="../target/x86_64-unknown-linux-gnu/release/librayzor_window.so"

    echo "[4/4] Windows x86_64..."
    cross build -p rayzor-window --release --target x86_64-pc-windows-gnu
    WIN_X64="../target/x86_64-pc-windows-gnu/release/rayzor_window.dll"

    build_wasm_host

    echo ""
    echo "Packaging rayzor-window.rpkg (all platforms + WASM)..."
    DYLIB_ARGS=""
    [ -f "$MACOS_ARM" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_ARM --os macos --arch aarch64"
    [ -f "$MACOS_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $MACOS_X64 --os macos --arch x86_64"
    [ -f "$LINUX_X64" ] && DYLIB_ARGS="$DYLIB_ARGS --dylib $LINUX_X64 --os linux --arch x86_64"
    [ -f "$WIN_X64" ]   && DYLIB_ARGS="$DYLIB_ARGS --dylib $WIN_X64 --os windows --arch x86_64"

    $RAYZOR rpkg pack \
        --name rayzor-window \
        $DYLIB_ARGS \
        $WASM_HOST_JS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-window.rpkg
else
    echo "=== Building rayzor-window (current platform) ==="
    cargo build -p rayzor-window --release

    case "$(uname -s)" in
        Darwin*) LIB_EXT="dylib" ;;
        Linux*)  LIB_EXT="so" ;;
        *)       echo "Unsupported platform"; exit 1 ;;
    esac

    DYLIB_PATH="../target/release/librayzor_window.${LIB_EXT}"
    [ ! -f "$DYLIB_PATH" ] && echo "Error: $DYLIB_PATH not found" && exit 1

    build_wasm_host

    echo "Packaging rayzor-window.rpkg..."
    $RAYZOR rpkg pack \
        --name rayzor-window \
        --dylib "$DYLIB_PATH" \
        $WASM_HOST_JS \
        --haxe-dir "$HAXE_DIR" \
        --output rayzor-window.rpkg
fi

echo ""
echo "Done: rayzor-window.rpkg"
echo "Install: rayzor rpkg install rayzor-window.rpkg"
