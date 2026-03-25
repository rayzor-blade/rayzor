#!/bin/bash
set -e
case "$(uname -s)" in
    Darwin*) LIB_EXT="dylib" ;;
    Linux*)  LIB_EXT="so" ;;
    *)       echo "Unsupported platform"; exit 1 ;;
esac

echo "Building rayzor-window..."
cargo build -p rayzor-window --release

DYLIB_PATH="../target/release/librayzor_window.${LIB_EXT}"
[ ! -f "$DYLIB_PATH" ] && echo "Error: $DYLIB_PATH not found" && exit 1

echo "Packaging rayzor-window.rpkg..."
../target/release/rayzor rpkg pack \
    --name rayzor-window \
    --dylib "$DYLIB_PATH" \
    --haxe-dir ../compiler/haxe-std/rayzor/window \
    --output rayzor-window.rpkg

echo "Done: rayzor-window.rpkg"
echo "Install: rayzor rpkg install rayzor-window.rpkg"
