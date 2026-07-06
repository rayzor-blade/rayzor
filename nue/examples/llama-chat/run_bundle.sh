#!/usr/bin/env bash
# Run the precompiled llama-chat .rzb bundle as a standalone artifact —
# no rayzor.toml required. Everything is configured on the CLI:
#   --native-lib   points the runtime at the nue kernel dylib (Q8 KV + flash)
#   --preset       selects JIT tier pacing (application = auto-upgrade to LLVM)
#   --release      turns OFF stack-trace instrumentation (else ~19x slower)
#
# Rebuild the bundle with:  BUILD=1 ./run_bundle.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="${RAYZOR:-../../../target/release/rayzor}"
BUNDLE="${BUNDLE:-llama-chat.rzb}"
LIB="${LIB:-../../../target/release/libnue_plugins.dylib}"
GGUF="${GGUF:-/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"

# The long prompt (override with PROMPT=... ./run_bundle.sh).
PROMPT="${PROMPT:-Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation}"

MAX_TOKENS="${MAX_TOKENS:-5000}"
TEMP="${TEMP:-0.5}"
PRESET="${PRESET:-server}"
STATS="${STATS:-1}"   # set STATS=0 to hide the tier/beadie summary

# JIT tier thresholds: interpreter / warm / hot / blazing (call counts before
# each promotion). Tuned to warm=30, hot=5; blazing=max means no count-based
# LLVM promotion — the before-main auto-upgrade installs LLVM instead.
INTERP_THRESHOLD="${INTERP_THRESHOLD:-1}"
WARM_THRESHOLD="${WARM_THRESHOLD:-30}"
HOT_THRESHOLD="${HOT_THRESHOLD:-5}"
BLAZING_THRESHOLD="${BLAZING_THRESHOLD:-max}"

# nue serving config (kernel + KV + flash + lm_head requant + static bands).
export RAYZOR_HAXE_MATMUL="${RAYZOR_HAXE_MATMUL:-1}"
export RAYZOR_WORKERS="${RAYZOR_WORKERS:-1}"
export RAYZOR_HAXE_FLASH="${RAYZOR_HAXE_FLASH:-1}"
export RAYZOR_KV_Q8="${RAYZOR_KV_Q8:-1}"
export REQUANT_LM_HEAD="${REQUANT_LM_HEAD:-1}"
export RAYZOR_STATIC_BANDS="${RAYZOR_STATIC_BANDS:-1}"

# Optional rebuild.
if [[ "${BUILD:-0}" == "1" ]]; then
  echo ">> building $BUNDLE"
  "$RAYZOR" bundle Main.hx -o "$BUNDLE" --no-cache
fi

if [[ ! -f "$BUNDLE" ]]; then
  echo "error: $BUNDLE not found. Build it first: BUILD=1 ./run_bundle.sh" >&2
  exit 1
fi
if [[ ! -f "$LIB" ]]; then
  echo "error: native lib $LIB not found. Build it: cargo build --release -p nue-plugins" >&2
  exit 1
fi

# Clear any lingering runtime (an orphaned rayzor spins a core and skews timing).
pkill -9 -f target/release/rayzor 2>/dev/null || true
sleep 0.2

cmd=(
  "$RAYZOR" run "$BUNDLE"
  --native-lib "$LIB"
  --preset "$PRESET"
  --release
  --tier-thresholds "$INTERP_THRESHOLD/$WARM_THRESHOLD/$HOT_THRESHOLD/$BLAZING_THRESHOLD"
)
[[ "$STATS" == "1" ]] && cmd+=(--stats)
cmd+=(-- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$TEMP")

echo ">> ${cmd[*]}"
echo
exec "${cmd[@]}"
