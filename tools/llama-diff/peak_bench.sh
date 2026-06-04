#!/usr/bin/env bash
# Multi-run peak-memory bench for nue/llama-chat. Reports min/median/max
# of the tensor-data peak across successful runs (filters out SIGTRAP/SIGSEGV
# runs which don't reach the [tensor-data] dump line).
#
# Why "peak tensor-data" instead of RSS: macOS Activity Monitor's "Memory"
# column includes page cache (mmap'd model file pages count once per process
# but evicted-and-restored pages can double-count under pressure). The
# TrackingAllocator's tensor-data peak is the actual live tensor-data
# allocation seen across the run — deterministic across re-runs of the
# same code, suitable for comparing two commits.
#
# Usage:
#   tools/llama-diff/peak_bench.sh ["<prompt>"] [n_tokens] [runs]
#
# Defaults: prompt="A short story about Mark Zuckerberg and Facebook", n=200, runs=5

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Safety: refuse to run if REPO_ROOT didn't resolve to an actual rayzor
# checkout. Without this guard, running the script from /tmp or any other
# location outside the repo computes REPO_ROOT=/, and the cache-scrub
# `find "$REPO_ROOT" -name .rayzor -exec rm -rf {} +` line below sweeps
# the entire filesystem. Anchor to a file every rayzor checkout has.
if [[ ! -f "$REPO_ROOT/Cargo.toml" || ! -d "$REPO_ROOT/nue/examples/llama-chat" ]]; then
    echo "error: REPO_ROOT ($REPO_ROOT) doesn't look like a rayzor checkout" >&2
    echo "  expected $REPO_ROOT/Cargo.toml and $REPO_ROOT/nue/examples/llama-chat to exist" >&2
    echo "  refusing to run — would otherwise scan/scrub the filesystem from /" >&2
    exit 3
fi

LLAMA_CHAT_DIR="$REPO_ROOT/nue/examples/llama-chat"

RAYZOR="${RAYZOR:-$REPO_ROOT/target/release/rayzor}"
GGUF="${GGUF:-$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf 2>/dev/null | head -1)}"

PROMPT="${1:-A short story about Mark Zuckerberg and Facebook}"
N="${2:-200}"
RUNS="${3:-5}"

[[ -f "$GGUF" ]]   || { echo "error: GGUF not found at $GGUF" >&2; exit 1; }
[[ -x "$RAYZOR" ]] || { echo "error: rayzor binary not found/executable at $RAYZOR" >&2; exit 1; }

# Check if binary was built with --features profile (the [tensor-data]
# line only appears when the profile feature is enabled).
if [[ -z "$(nm "$RAYZOR" 2>/dev/null | grep _rayzor_dump_tensor_alloc_stats)" ]]; then
    echo "warning: rayzor binary lacks profile feature — [tensor-data] line"
    echo "  won't appear. Rebuild with:"
    echo "    cargo build --release --features profile -p rayzor"
    exit 2
fi

cd "$LLAMA_CHAT_DIR"

echo "=== peak bench ==="
echo "model:  $(basename "$GGUF")"
echo "prompt: \"$PROMPT\""
echo "tokens: $N"
echo "runs:   $RUNS"
echo

results=()
declare -i successful=0 failed=0
for ((i = 1; i <= RUNS; i++)); do
    find "$REPO_ROOT" -name .rayzor -type d -exec rm -rf {} + 2>/dev/null
    out=$(RAYZOR_DUMP_ALLOC_AT_EXIT=1 timeout 120 "$RAYZOR" run Main.hx -- \
        "$GGUF" "$PROMPT" "$N" 2>&1 | grep -E "tensor-data|done\b|sig[0-9]+-back" | tail -5)
    peak_mb=$(echo "$out" | grep tensor-data | sed -nE 's/.*peak=[0-9]+ \(([0-9.]+) MB\).*/\1/p')
    tok=$(echo "$out" | grep -E "^\[done\]" | sed -nE 's/^\[done\] ([0-9]+) tokens.*/\1/p')
    sig=$(echo "$out" | grep -oE "sig[0-9]+-back" | head -1)
    if [[ -n "${peak_mb:-}" && -n "${tok:-}" ]]; then
        successful=$((successful + 1))
        printf "  run %2d  peak=%8.1f MB  tokens=%s\n" "$i" "$peak_mb" "$tok"
        results+=("$peak_mb")
    else
        failed=$((failed + 1))
        printf "  run %2d  FAILED  %s\n" "$i" "${sig:-no-stats}"
    fi
done

echo
if [[ ${#results[@]} -eq 0 ]]; then
    echo "ALL RUNS FAILED ($failed/$RUNS). Nothing to average."
    exit 1
fi

# Compute min, median, max
sorted=($(printf "%s\n" "${results[@]}" | sort -n))
n=${#sorted[@]}
min=${sorted[0]}
max=${sorted[$((n - 1))]}
if (( n % 2 == 1 )); then
    median=${sorted[$((n / 2))]}
else
    a=${sorted[$((n / 2 - 1))]}
    b=${sorted[$((n / 2))]}
    median=$(awk -v a="$a" -v b="$b" 'BEGIN {printf "%.1f", (a+b)/2}')
fi

# Mean
sum=0
for v in "${results[@]}"; do
    sum=$(awk -v s="$sum" -v v="$v" 'BEGIN {printf "%.3f", s+v}')
done
mean=$(awk -v s="$sum" -v n="$n" 'BEGIN {printf "%.1f", s/n}')

printf "successful: %d/%d\n" "$successful" "$RUNS"
printf "peak (MB): min=%.1f median=%s mean=%s max=%.1f\n" "$min" "$median" "$mean" "$max"
