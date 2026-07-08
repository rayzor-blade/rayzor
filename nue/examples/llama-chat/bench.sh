#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="${RAYZOR:-../../../target/release/rayzor}"
DEFAULT_GGUF="/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
if [[ -z "${GGUF:-}" && -f "$HOME/llama-q4.gguf" ]]; then
  DEFAULT_GGUF="$HOME/llama-q4.gguf"
fi
GGUF="${GGUF:-$DEFAULT_GGUF}"
PROMPT="${PROMPT:-Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation}"
MAX_TOKENS="${MAX_TOKENS:-5000}"
TEMP="${TEMP:-0.7}"
RUNS="${RUNS:-6}"
TIMEOUT="${TIMEOUT:-240}"
COOLDOWN_MS="${COOLDOWN_MS:-15000}"
PRESET="${PRESET:-server}"
RELEASE="${RELEASE:-true}"
LLVM="${LLVM:-true}"
CACHE_SCRUB="${CACHE_SCRUB:-false}"
START_INTERPRETED="${START_INTERPRETED:-}"
TIER_PROMOTION="${TIER_PROMOTION:-}"
VARIANTS="${VARIANTS:-1/30/5}"
DECODE_PROFILE="${DECODE_PROFILE:-false}"
POOL_PROFILE="${POOL_PROFILE:-${RAYZOR_PROFILE_POOL:-false}}"
WORKER_VARIANTS="${WORKER_VARIANTS:-}"
SILENT_STREAM="${SILENT_STREAM:-true}"
USE_JEMALLOC="${USE_JEMALLOC:-auto}"
BUNDLE="${BUNDLE:-auto}"
BENCH_TARGET="${BENCH_TARGET:-}"

if [[ -n "$BENCH_TARGET" ]]; then
  BENCH_FILE="$BENCH_TARGET"
elif [[ "$BUNDLE" == "0" || "$BUNDLE" == "false" || "$BUNDLE" == "False" || "$BUNDLE" == "FALSE" || "$BUNDLE" == "source" ]]; then
  BENCH_FILE="Main.hx"
elif [[ "$BUNDLE" == "auto" ]]; then
  if [[ -f "$SCRIPT_DIR/llama-chat.rzb" ]]; then
    BENCH_FILE="$SCRIPT_DIR/llama-chat.rzb"
  else
    BENCH_FILE="Main.hx"
  fi
else
  BENCH_FILE="$BUNDLE"
fi

IS_BUNDLE=0
if [[ "$BENCH_FILE" == *.rzb ]]; then
  IS_BUNDLE=1
fi

DEFAULT_NATIVE_LIB="../../../target/release/libnue_plugins.dylib"
if [[ "$(uname -s)" == "Linux" && -f "../../../target/release/libnue_plugins.so" ]]; then
  DEFAULT_NATIVE_LIB="../../../target/release/libnue_plugins.so"
elif [[ "$(uname -s)" == "Darwin" && -f "../../../target/release/libnue_plugins.dylib" ]]; then
  DEFAULT_NATIVE_LIB="../../../target/release/libnue_plugins.dylib"
fi
NATIVE_LIB="${NATIVE_LIB:-${LIB:-$DEFAULT_NATIVE_LIB}}"

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

export RAYZOR_NO_PRELOAD_MMAP="${RAYZOR_NO_PRELOAD_MMAP:-1}"
maybe_enable_jemalloc
if [[ "$POOL_PROFILE" == "true" || "$POOL_PROFILE" == "1" || "$POOL_PROFILE" == "yes" ]]; then
  export RAYZOR_PROFILE_POOL=1
fi
export RAYZOR_HAXE_MATMUL="${RAYZOR_HAXE_MATMUL:-1}"
export RAYZOR_HAXE_FLASH="${RAYZOR_HAXE_FLASH:-1}"
export RAYZOR_KV_Q8="${RAYZOR_KV_Q8:-1}"
export RAYZOR_REQUANT_LM_HEAD="${RAYZOR_REQUANT_LM_HEAD:-${REQUANT_LM_HEAD:-1}}"
export RAYZOR_STATIC_BANDS="${RAYZOR_STATIC_BANDS:-1}"
if [[ "$SILENT_STREAM" == "true" || "$SILENT_STREAM" == "1" || "$SILENT_STREAM" == "yes" ]]; then
  export RAYZOR_LLAMA_SILENT_STREAM=1
else
  unset RAYZOR_LLAMA_SILENT_STREAM
fi

read -r -a VARIANT_LIST <<< "$VARIANTS"
if [[ -n "$WORKER_VARIANTS" ]]; then
  read -r -a WORKER_VARIANT_LIST <<< "$WORKER_VARIANTS"
else
  WORKER_VARIANT_LIST=("")
fi

if [[ "${LD_PRELOAD:-}" == *jemalloc* ]]; then
  echo "allocator: jemalloc (${LD_PRELOAD})"
else
  echo "allocator: system"
fi
echo "target: $BENCH_FILE"
if [[ "$IS_BUNDLE" == "1" ]]; then
  echo "native lib: $NATIVE_LIB"
fi
echo "mmap preload: $([[ "${RAYZOR_NO_PRELOAD_MMAP:-}" == "1" ]] && echo off || echo on)"
echo "preset: $PRESET"
echo "release: $RELEASE"
echo "stack traces: $([[ "$RELEASE" == "true" || "$RELEASE" == "1" || "$RELEASE" == "yes" ]] && echo off || echo manifest/preset)"
echo "llvm: $LLVM"
echo "cache scrub: $CACHE_SCRUB"
echo "silent stream: $SILENT_STREAM"
if [[ -n "$START_INTERPRETED" ]]; then
  echo "start interpreted override: $START_INTERPRETED"
fi
if [[ -n "$TIER_PROMOTION" ]]; then
  echo "tier promotion override: $TIER_PROMOTION"
fi
echo "pool profile: $([[ "${RAYZOR_PROFILE_POOL:-}" == "1" ]] && echo on || echo off)"
echo "haxe: matmul=$RAYZOR_HAXE_MATMUL flash=$RAYZOR_HAXE_FLASH kv_q8=$RAYZOR_KV_Q8 lm_head_requant=$RAYZOR_REQUANT_LM_HEAD"
if [[ -n "$WORKER_VARIANTS" ]]; then
  echo "worker variants: $WORKER_VARIANTS"
fi

run_bench() {
  local worker="$1"
  if [[ -n "$worker" ]]; then
    if [[ "$worker" == "auto" || "$worker" == "default" ]]; then
      unset RAYZOR_HAXE_MATMUL_WORKERS
    else
      export RAYZOR_HAXE_MATMUL_WORKERS="$worker"
    fi
    echo
    echo "=== worker variant: $worker ==="
  fi

  local cmd=(
    "$RAYZOR" debug bench "$BENCH_FILE"
    --metric tok-per-s
    -n "$RUNS"
    --timeout "$TIMEOUT"
    --preset "$PRESET"
  )
  if [[ "$IS_BUNDLE" == "1" ]]; then
    if [[ ! -f "$BENCH_FILE" ]]; then
      echo "error: bundle $BENCH_FILE not found. Set BUNDLE=0 to bench Main.hx or BUILD=1 ./run_bundle.sh first." >&2
      exit 1
    fi
    if [[ ! -f "$NATIVE_LIB" ]]; then
      echo "error: native lib $NATIVE_LIB not found. Build nue-plugins or set NATIVE_LIB=/path/to/libnue_plugins." >&2
      exit 1
    fi
    cmd+=(--native-lib "$NATIVE_LIB")
  fi
  if [[ "$RELEASE" == "true" || "$RELEASE" == "1" || "$RELEASE" == "yes" ]]; then
    cmd+=(--release)
  fi
  if [[ "$LLVM" == "true" || "$LLVM" == "1" || "$LLVM" == "yes" ]]; then
    cmd+=(--llvm)
  fi
  if [[ "$CACHE_SCRUB" != "true" && "$CACHE_SCRUB" != "1" && "$CACHE_SCRUB" != "yes" ]]; then
    cmd+=(--no-cache-scrub)
  fi
  if [[ -n "$START_INTERPRETED" ]]; then
    cmd+=(--tier-start-interpreted "$START_INTERPRETED")
  fi
  if [[ -n "$TIER_PROMOTION" ]]; then
    cmd+=(--tier-promotion "$TIER_PROMOTION")
  fi

  for variant in "${VARIANT_LIST[@]}"; do
    cmd+=(--tier-thresholds "$variant")
  done

  if [[ "$DECODE_PROFILE" == "true" || "$DECODE_PROFILE" == "1" || "$DECODE_PROFILE" == "yes" ]]; then
    cmd+=(--decode-profile)
  fi

  cmd+=(
    --cooldown-ms "$COOLDOWN_MS"
    -- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$TEMP"
  )

  "${cmd[@]}"
}

for worker in "${WORKER_VARIANT_LIST[@]}"; do
  run_bench "$worker"
done
