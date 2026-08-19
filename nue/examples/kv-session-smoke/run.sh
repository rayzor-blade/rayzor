#!/usr/bin/env bash
# Build and run the KV-session isolation check.
#
#   GGUF=/path/to/model.gguf ./run.sh
#
# Exits non-zero if two interleaved conversations diverge from the same two run
# one after the other — which is what sharing cache state between them looks
# like. Any causal model with a BPE tokenizer works; the assertion is about the
# cache, not the checkpoint.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPO=../../..
RAYZOR="$REPO/target/release/rayzor"
BUNDLE="${BUNDLE:-kv-session-smoke.rzb}"
case "$(uname -s)" in
  Linux) LIB="$REPO/target/release/libnue_plugins.so" ;;
  *) LIB="$REPO/target/release/libnue_plugins.dylib" ;;
esac

[[ -n "${GGUF:-}" ]] || { echo "error: set GGUF to a model file" >&2; exit 2; }

if [[ "${BUILD:-1}" == "1" ]]; then
  "$RAYZOR" bundle Main.hx -o "$BUNDLE" --no-cache >/dev/null || exit 2
fi

GGUF="$GGUF" exec "$RAYZOR" run "$BUNDLE" \
  --native-lib "$LIB" --preset server --release --llvm \
  --tier-thresholds 1/30/5/max --tier-promotion true --tier-start-interpreted false
