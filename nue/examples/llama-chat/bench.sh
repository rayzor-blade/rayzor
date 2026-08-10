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
CTX="${CTX:-4096}"
TEMP="${TEMP:-0.5}"
RUNS="${RUNS:-6}"
TIMEOUT="${TIMEOUT:-240}"
COOLDOWN_MS="${COOLDOWN_MS:-15000}"
PRESET="${PRESET:-server}"
BENCH_ENGINE="${BENCH_ENGINE:-direct}" # direct|debug
RELEASE="${RELEASE:-true}"
LLVM="${LLVM:-true}"
CACHE_SCRUB="${CACHE_SCRUB:-false}"
START_INTERPRETED="${START_INTERPRETED:-false}"
TIER_PROMOTION="${TIER_PROMOTION:-true}"
VARIANTS="${VARIANTS:-1/30/5}"
DECODE_PROFILE="${DECODE_PROFILE:-false}"
RUNTIME_STATS="${RUNTIME_STATS:-false}"
POOL_PROFILE="${POOL_PROFILE:-${RAYZOR_PROFILE_POOL:-false}}"
WORKER_VARIANTS="${WORKER_VARIANTS:-}"
SILENT_STREAM="${SILENT_STREAM:-true}"
USE_JEMALLOC="${USE_JEMALLOC:-auto}"
KILL_RAYZOR_BETWEEN_RUNS="${KILL_RAYZOR_BETWEEN_RUNS:-true}"
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

# The runtime gates preload on the variable's PRESENCE, like the rest of the
# RAYZOR_NO_* family: any value turns preload off. "On" means leaving it unset.
if [[ -z "${RAYZOR_NO_PRELOAD_MMAP:-}" ]]; then
  if [[ "$(uname -s)" != "Darwin" ]]; then
    export RAYZOR_NO_PRELOAD_MMAP=1
  fi
else
  export RAYZOR_NO_PRELOAD_MMAP
fi
maybe_enable_jemalloc
if [[ "$POOL_PROFILE" == "true" || "$POOL_PROFILE" == "1" || "$POOL_PROFILE" == "yes" ]]; then
  export RAYZOR_PROFILE_POOL=1
fi
export RAYZOR_HAXE_MATMUL="${RAYZOR_HAXE_MATMUL:-1}"
export RAYZOR_HAXE_FLASH="${RAYZOR_HAXE_FLASH:-1}"
export RAYZOR_KV_Q8="${RAYZOR_KV_Q8:-1}"
export RAYZOR_REQUANT_LM_HEAD="${RAYZOR_REQUANT_LM_HEAD:-${REQUANT_LM_HEAD:-1}}"
if [[ -n "${RAYZOR_STATIC_BANDS:-}" ]]; then
  export RAYZOR_STATIC_BANDS
fi
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
echo "mmap preload: $([[ -n "${RAYZOR_NO_PRELOAD_MMAP:-}" ]] && echo off || echo on)"
echo "preset: $PRESET"
echo "engine: $BENCH_ENGINE"
echo "release: $RELEASE"
echo "stack traces: $([[ "$RELEASE" == "true" || "$RELEASE" == "1" || "$RELEASE" == "yes" ]] && echo off || echo manifest/preset)"
echo "llvm: $LLVM"
echo "runtime stats: $RUNTIME_STATS"
echo "cache scrub: $CACHE_SCRUB"
echo "kill stale rayzor: $KILL_RAYZOR_BETWEEN_RUNS"
echo "silent stream: $SILENT_STREAM"
if [[ -n "$START_INTERPRETED" ]]; then
  echo "start interpreted override: $START_INTERPRETED"
fi
if [[ -n "$TIER_PROMOTION" ]]; then
  echo "tier promotion override: $TIER_PROMOTION"
fi
echo "pool profile: $([[ "${RAYZOR_PROFILE_POOL:-}" == "1" ]] && echo on || echo off)"
echo "haxe: matmul=$RAYZOR_HAXE_MATMUL flash=$RAYZOR_HAXE_FLASH flash_batch_max=${RAYZOR_HAXE_FLASH_BATCH_MAX:-auto} kv_q8=$RAYZOR_KV_Q8 lm_head_requant=$RAYZOR_REQUANT_LM_HEAD static_bands=${RAYZOR_STATIC_BANDS:-auto}"
echo "generation: max_tokens=$MAX_TOKENS ctx=$CTX temp=$TEMP"
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

  if [[ "$BENCH_ENGINE" == "direct" ]]; then
    run_direct_bench
    return
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
  if [[ "$RUNTIME_STATS" == "true" || "$RUNTIME_STATS" == "1" || "$RUNTIME_STATS" == "yes" ]]; then
    cmd+=(--runtime-stats)
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
    -- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$CTX" "$TEMP"
  )

  "${cmd[@]}"
}

run_direct_bench() {
  local tmp_dir="${TMPDIR:-/tmp}/rayzor-llama-chat-bench.$$"
  local results="$tmp_dir/results.tsv"
  mkdir -p "$tmp_dir"
  : > "$results"

  echo "=== rayzor direct bench ==="
  echo "file:    $BENCH_FILE"
  echo "runs:    $RUNS per variant ($((RUNS * ${#VARIANT_LIST[@]})) subprocesses)"
  echo "metric:  tok/s"
  echo "preset:  $PRESET"
  if [[ ${#VARIANT_LIST[@]} -gt 1 ]]; then
    local labels
    labels="$(printf "%s, " "${VARIANT_LIST[@]}")"
    echo "variants: ${labels%, }"
    echo "order:   round-robin"
  fi
  echo

  local ordinal=0
  for ((run_no = 1; run_no <= RUNS; run_no++)); do
    for variant in "${VARIANT_LIST[@]}"; do
      ordinal=$((ordinal + 1))
      if [[ "$CACHE_SCRUB" == "true" || "$CACHE_SCRUB" == "1" || "$CACHE_SCRUB" == "yes" ]]; then
        rm -rf .rayzor/cache .rayzor/blade/cache
      fi
      if [[ "$KILL_RAYZOR_BETWEEN_RUNS" == "true" || "$KILL_RAYZOR_BETWEEN_RUNS" == "1" || "$KILL_RAYZOR_BETWEEN_RUNS" == "yes" ]]; then
        pkill -9 -f target/release/rayzor 2>/dev/null || true
        sleep 0.2
      fi

      local cmd=("$RAYZOR" run "$BENCH_FILE" "--preset" "$PRESET")
      if [[ "$IS_BUNDLE" == "1" ]]; then
        cmd+=("--native-lib" "$NATIVE_LIB")
      fi
      if [[ "$RELEASE" == "true" || "$RELEASE" == "1" || "$RELEASE" == "yes" ]]; then
        cmd+=("--release")
      fi
      if [[ "$LLVM" == "true" || "$LLVM" == "1" || "$LLVM" == "yes" ]]; then
        cmd+=("--llvm")
      fi
      if [[ "$RUNTIME_STATS" == "true" || "$RUNTIME_STATS" == "1" || "$RUNTIME_STATS" == "yes" ]]; then
        cmd+=("--stats")
      fi
      if [[ -n "$START_INTERPRETED" ]]; then
        cmd+=("--tier-start-interpreted" "$START_INTERPRETED")
      fi
      if [[ -n "$TIER_PROMOTION" ]]; then
        cmd+=("--tier-promotion" "$TIER_PROMOTION")
      fi
      if [[ "$variant" != "toml" ]]; then
        cmd+=("--tier-thresholds" "$variant")
      fi
      cmd+=("--" "$GGUF" "$PROMPT" "$MAX_TOKENS" "$CTX" "$TEMP")

      local output status tps ttft label
      set +e
      if command -v timeout >/dev/null 2>&1; then
        output="$(timeout "$TIMEOUT" "${cmd[@]}" 2>&1)"
      else
        output="$("${cmd[@]}" 2>&1)"
      fi
      status=$?
      set -e
      tps="$(printf "%s\n" "$output" | sed -n 's/.*(\([0-9][0-9.]*\) tok\/s.*/\1/p' | tail -1)"
      ttft="$(printf "%s\n" "$output" | sed -n 's/.*ttft=\([0-9][0-9.]*\)s.*/\1/p' | tail -1)"
      if [[ "$status" == "0" && -n "$tps" ]]; then
        label="tok/s=$(printf "%.2f" "$tps")"
        if [[ -n "$ttft" ]]; then
          label="$label  ttft=$(printf "%.3f" "$ttft")s"
        fi
        printf "%s\t%s\tok\t%s\t%s\n" "$run_no" "$variant" "$tps" "$ttft" >> "$results"
        printf "  run %3d  %-14s ok       %s\n" "$run_no" "$variant" "$label"
      else
        printf "%s\t%s\tfail\t\t\n" "$run_no" "$variant" >> "$results"
        printf "  run %3d  %-14s FAIL     exit=%s no [done] metric\n" "$run_no" "$variant" "$status"
        printf "%s\n" "$output" | tail -8 | sed 's/^/               | /'
      fi

      if (( COOLDOWN_MS > 0 && ordinal < RUNS * ${#VARIANT_LIST[@]} )); then
        sleep "$(awk "BEGIN { printf \"%.3f\", $COOLDOWN_MS / 1000.0 }")"
      fi
    done
  done

  echo
  echo "=== summary ==="
  for variant in "${VARIANT_LIST[@]}"; do
    local total ok sorted
    total="$(awk -F'\t' -v v="$variant" '$2 == v { n++ } END { print n + 0 }' "$results")"
    ok="$(awk -F'\t' -v v="$variant" '$2 == v && $3 == "ok" { n++ } END { print n + 0 }' "$results")"
    echo "$variant         successful: $ok/$total"
    if [[ "$ok" != "0" ]]; then
      sorted="$tmp_dir/${variant//\//_}.sorted"
      awk -F'\t' -v v="$variant" '$2 == v && $3 == "ok" { print $4 }' "$results" | sort -n > "$sorted"
      awk '
        { a[++n] = $1; sum += $1; sumsq += $1 * $1 }
        END {
          min = a[1]; max = a[n];
          if (n % 2) median = a[(n + 1) / 2];
          else median = (a[n / 2] + a[n / 2 + 1]) / 2;
          mean = sum / n;
          variance = (sumsq / n) - (mean * mean);
          if (variance < 0) variance = 0;
          stddev = sqrt(variance);
          drift = a[n] - a[1];
          printf "               tok/s: min=%.2f  median=%.2f  mean=%.2f  max=%.2f  stddev=%.2f  drift=%.2f\n",
            min, median, mean, max, stddev, drift;
        }
      ' "$sorted"
      local ttft_sorted
      ttft_sorted="$tmp_dir/${variant//\//_}.ttft.sorted"
      awk -F'\t' -v v="$variant" '$2 == v && $3 == "ok" && $5 != "" { print $5 }' "$results" | sort -n > "$ttft_sorted"
      if [[ -s "$ttft_sorted" ]]; then
        awk '
          { a[++n] = $1; sum += $1 }
          END {
            if (n % 2) median = a[(n + 1) / 2];
            else median = (a[n / 2] + a[n / 2 + 1]) / 2;
            printf "               ttft:  min=%.3fs median=%.3fs mean=%.3fs max=%.3fs\n",
              a[1], median, sum / n, a[n];
          }
        ' "$ttft_sorted"
      fi
    fi
  done
  rm -rf "$tmp_dir"
}

for worker in "${WORKER_VARIANT_LIST[@]}"; do
  run_bench "$worker"
done
