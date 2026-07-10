#!/usr/bin/env bash
# Run the precompiled llama-chat .rzb bundle as a standalone artifact —
# no rayzor.toml required. Everything is configured on the CLI:
#   --native-lib   points the runtime at the nue kernel dylib (Q8 KV + flash)
#   --preset       selects JIT tier pacing
#   --release      turns OFF stack-trace instrumentation (else ~19x slower)
#
# Rebuild the bundle with:  BUILD=1 ./run_bundle.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="${RAYZOR:-../../../target/release/rayzor}"
BUNDLE="${BUNDLE:-llama-chat.rzb}"
if [[ -z "${LIB:-}" ]]; then
  case "$(uname -s)" in
    Darwin) LIB="../../../target/release/libnue_plugins.dylib" ;;
    Linux) LIB="../../../target/release/libnue_plugins.so" ;;
    *) LIB="../../../target/release/libnue_plugins.dylib" ;;
  esac
fi
DEFAULT_GGUF="/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if [[ -z "${GGUF:-}" && -f "$HOME/llama-q4.gguf" ]]; then
  DEFAULT_GGUF="$HOME/llama-q4.gguf"
fi
GGUF="${GGUF:-$DEFAULT_GGUF}"

# The long prompt (override with PROMPT=... ./run_bundle.sh).
PROMPT="${PROMPT:-Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation}"

MAX_TOKENS="${MAX_TOKENS:-5000}"
CTX="${CTX:-4096}"
TEMP="${TEMP:-0.5}"
PRESET="${PRESET:-server}"
STATS="${STATS:-0}"   # set STATS=1 to print tier/beadie diagnostics (measurable overhead)
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
# each promotion). Tuned to warm=30, hot=5; blazing=max keeps count-based
# P4 promotion off while still allowing explicit tier promotion.
INTERP_THRESHOLD="${INTERP_THRESHOLD:-1}"
WARM_THRESHOLD="${WARM_THRESHOLD:-30}"
HOT_THRESHOLD="${HOT_THRESHOLD:-5}"
BLAZING_THRESHOLD="${BLAZING_THRESHOLD:-max}"
TIER_PROMOTION="${TIER_PROMOTION:-true}"
START_INTERPRETED="${START_INTERPRETED:-false}"

# nue serving config (kernel + KV + flash + lm_head requant + static bands).
export RAYZOR_HAXE_MATMUL="${RAYZOR_HAXE_MATMUL:-1}"
export RAYZOR_HAXE_FLASH="${RAYZOR_HAXE_FLASH:-1}"
export RAYZOR_KV_Q8="${RAYZOR_KV_Q8:-1}"
export RAYZOR_REQUANT_LM_HEAD="${RAYZOR_REQUANT_LM_HEAD:-${REQUANT_LM_HEAD:-1}}"
export RAYZOR_PREFILL_LAST_LOGITS="${RAYZOR_PREFILL_LAST_LOGITS:-1}"
if [[ -n "${RAYZOR_HAXE_FUSED_MATMUL:-}" ]]; then
  export RAYZOR_HAXE_FUSED_MATMUL
fi
if [[ -n "${RAYZOR_HAXE_FLASH_POOL:-}" ]]; then
  export RAYZOR_HAXE_FLASH_POOL
fi
if [[ -n "${RAYZOR_STDOUT_FLUSH_MS:-${RAYZOR_LLAMA_STREAM_FLUSH_MS:-}}" ]]; then
  export RAYZOR_STDOUT_FLUSH_MS="${RAYZOR_STDOUT_FLUSH_MS:-$RAYZOR_LLAMA_STREAM_FLUSH_MS}"
fi
# mmap preload: let the runtime choose (default = preload ON) unless the
# caller sets RAYZOR_NO_PRELOAD_MMAP explicitly for an A/B. Forcing it off on
# Linux traded a small startup saving for 140-250ms page-fault stalls in the
# first decode steps (NUC: step p99 144ms -> 31.5ms and ttft 1.35s -> 0.88s
# with preload back on; 300-token avg equal or better).
if [[ -n "${RAYZOR_NO_PRELOAD_MMAP:-}" ]]; then
  export RAYZOR_NO_PRELOAD_MMAP
fi

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

# Optional cleanup for isolated benchmark boxes. Disabled by default because
# this script is often run while another llama-chat probe is active; killing
# every target/release/rayzor process makes concurrent runs fail with 137 and
# muddies the very latency numbers we are trying to compare.
if [[ "${KILL_RAYZOR:-0}" == "1" ]]; then
  pkill -9 -f target/release/rayzor 2>/dev/null || true
  sleep 0.2
fi

cmd=(
  "$RAYZOR" run "$BUNDLE"
  --native-lib "$LIB"
  --preset "$PRESET"
  --release
  --llvm
  --tier-thresholds "$INTERP_THRESHOLD/$WARM_THRESHOLD/$HOT_THRESHOLD/$BLAZING_THRESHOLD"
  --tier-promotion "$TIER_PROMOTION"
  --tier-start-interpreted "$START_INTERPRETED"
)
[[ "$STATS" == "1" ]] && cmd+=(--stats)
cmd+=(-- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$CTX" "$TEMP")

echo ">> ${cmd[*]}"
if [[ "${LD_PRELOAD:-}" == *jemalloc* ]]; then
  echo "allocator: jemalloc (${LD_PRELOAD})"
else
  echo "allocator: system"
fi
echo "mmap preload: $([[ "${RAYZOR_NO_PRELOAD_MMAP:-}" == "1" ]] && echo off || echo on)"
echo "kernels: haxe_matmul=${RAYZOR_HAXE_MATMUL} fused_matmul=${RAYZOR_HAXE_FUSED_MATMUL:-auto} workers=${RAYZOR_HAXE_MATMUL_WORKERS:-auto} haxe_flash=${RAYZOR_HAXE_FLASH} flash_pool=${RAYZOR_HAXE_FLASH_POOL:-auto} flash_shifted_q=${RAYZOR_HAXE_FLASH_SHIFTED_Q:-auto} flash_batch_max=${RAYZOR_HAXE_FLASH_BATCH_MAX:-auto} kv_q8=${RAYZOR_KV_Q8} lm_head_requant=${RAYZOR_REQUANT_LM_HEAD} prefill_last_logits=${RAYZOR_PREFILL_LAST_LOGITS}"
echo "pool: spins=${RAYZOR_HAXE_POOL_SPINS:-auto} relax=${RAYZOR_HAXE_POOL_RELAX:-auto} perf_affinity=${RAYZOR_PERF_AFFINITY:-off} recycle=${RAYZOR_POOL:-off}"
echo "tier: promotion=${TIER_PROMOTION} start_interpreted=${START_INTERPRETED} thresholds=${INTERP_THRESHOLD}/${WARM_THRESHOLD}/${HOT_THRESHOLD}/${BLAZING_THRESHOLD}"
echo "diag: spec_decode=${RAYZOR_SPEC_DECODE:-off} silent_stream=${RAYZOR_LLAMA_SILENT_STREAM:-0} stdout_flush_ms=${RAYZOR_STDOUT_FLUSH_MS:-auto} profile_decode=${RAYZOR_PROFILE_DECODE:-off} profile_pool=${RAYZOR_PROFILE_POOL:-off} dump_block_shapes=${RAYZOR_DUMP_BLOCK_SHAPES:-off}"
echo "generation: max_tokens=${MAX_TOKENS} ctx=${CTX} temp=${TEMP}"
echo
exec "${cmd[@]}"
