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
DEFAULT_GGUF="/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if [[ -z "${GGUF:-}" && -f "$HOME/llama-q4.gguf" ]]; then
  DEFAULT_GGUF="$HOME/llama-q4.gguf"
fi
GGUF="${GGUF:-$DEFAULT_GGUF}"

# The long prompt (override with PROMPT=... ./run_bundle.sh).
PROMPT="${PROMPT:-Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation}"

MAX_TOKENS="${MAX_TOKENS:-5000}"
TEMP="${TEMP:-0.5}"
PRESET="${PRESET:-server}"
STATS="${STATS:-1}"   # set STATS=0 to hide the tier/beadie summary
USE_JEMALLOC="${USE_JEMALLOC:-auto}" # auto|1|0; Linux-only LD_PRELOAD when present

maybe_enable_jemalloc() {
  local mode="${USE_JEMALLOC:-auto}"
  case "$mode" in
    0|false|False|FALSE|no|No|NO|"") return 0 ;;
  esac
  [[ "$(uname -s)" == "Linux" ]] || return 0
  [[ "${LD_PRELOAD:-}" == *jemalloc* ]] && return 0
  local lib=""
  for candidate in \
    /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    /usr/lib64/libjemalloc.so.2 \
    /usr/lib/libjemalloc.so.2
  do
    if [[ -f "$candidate" ]]; then
      lib="$candidate"
      break
    fi
  done
  if [[ -z "$lib" ]] && command -v ldconfig >/dev/null 2>&1; then
    lib="$(ldconfig -p 2>/dev/null | awk '/libjemalloc\.so/ {print $NF; exit}')"
  fi
  if [[ -n "$lib" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$lib"
  elif [[ "$mode" != "auto" ]]; then
    echo "error: USE_JEMALLOC=$mode requested but libjemalloc was not found" >&2
    exit 1
  fi
}

# JIT tier thresholds: interpreter / warm / hot / blazing (call counts before
# each promotion). Tuned to warm=30, hot=5; blazing=max means no count-based
# LLVM promotion — the before-main auto-upgrade installs LLVM instead.
INTERP_THRESHOLD="${INTERP_THRESHOLD:-1}"
WARM_THRESHOLD="${WARM_THRESHOLD:-30}"
HOT_THRESHOLD="${HOT_THRESHOLD:-5}"
BLAZING_THRESHOLD="${BLAZING_THRESHOLD:-max}"

# nue serving config (kernel + KV + flash + lm_head requant + static bands).
export RAYZOR_HAXE_MATMUL="${RAYZOR_HAXE_MATMUL:-1}"
export RAYZOR_HAXE_FLASH="${RAYZOR_HAXE_FLASH:-1}"
export RAYZOR_KV_Q8="${RAYZOR_KV_Q8:-1}"
export RAYZOR_REQUANT_LM_HEAD="${RAYZOR_REQUANT_LM_HEAD:-${REQUANT_LM_HEAD:-1}}"
export RAYZOR_STATIC_BANDS="${RAYZOR_STATIC_BANDS:-1}"
# Avoid force-faulting the entire GGUF mmap before inference. Decode touches
# the active weight pages anyway, but skipping preload reduces startup thermal
# pressure and preserves the same peak/latency profile on the NUC.
export RAYZOR_NO_PRELOAD_MMAP="${RAYZOR_NO_PRELOAD_MMAP:-1}"

maybe_enable_jemalloc

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
  # --preset "$PRESET"
  --release
  --tier-thresholds "$INTERP_THRESHOLD/$WARM_THRESHOLD/$HOT_THRESHOLD/$BLAZING_THRESHOLD"
)
[[ "$STATS" == "1" ]] && cmd+=(--stats)
cmd+=(-- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$TEMP")

echo ">> ${cmd[*]}"
if [[ "${LD_PRELOAD:-}" == *jemalloc* ]]; then
  echo "allocator: jemalloc (${LD_PRELOAD})"
else
  echo "allocator: system"
fi
echo "mmap preload: $([[ "${RAYZOR_NO_PRELOAD_MMAP:-}" == "1" ]] && echo off || echo on)"
echo
exec "${cmd[@]}"
