#!/usr/bin/env bash
# A/B peak-memory comparison between two git commits.
#
# Builds + runs peak_bench at the current HEAD, then at the given baseline
# commit, then compares medians. Restores the original git state on exit
# (success or failure) so an interrupted run doesn't strand the tree.
#
# Usage:
#   tools/llama-diff/peak_compare.sh <baseline_ref> ["<prompt>"] [n_tokens] [runs]
#
# Examples:
#   tools/llama-diff/peak_compare.sh HEAD~1       # current vs previous commit
#   tools/llama-diff/peak_compare.sh be797eb 200 7
#   tools/llama-diff/peak_compare.sh main "A story about Mark Zuckerberg" 600 5

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE="${1:?usage: $0 <baseline_ref> [\"<prompt>\"] [n_tokens] [runs]}"
PROMPT="${2:-A short story about Mark Zuckerberg and Facebook}"
N="${3:-200}"
RUNS="${4:-5}"

ORIG_REF=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)
restore() {
    echo
    echo "[restore] back to $ORIG_REF"
    git checkout "$ORIG_REF" --quiet
    cargo build --release --features profile -p rayzor 2>&1 | tail -1
}
trap restore EXIT

# Ensure clean tree so checkout works
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "error: working tree is dirty; commit or stash first" >&2
    exit 2
fi

run_bench() {
    local label=$1
    local ref=$2
    echo
    echo "=== building at $label ($ref) ==="
    git checkout "$ref" --quiet
    cargo build --release --features profile -p rayzor 2>&1 | tail -1
    "$SCRIPT_DIR/peak_bench.sh" "$PROMPT" "$N" "$RUNS"
}

HEAD_OUT=$(run_bench "HEAD" "$ORIG_REF")
BASE_OUT=$(run_bench "BASELINE" "$BASE")

extract_median() {
    echo "$1" | sed -nE 's/.*median=([0-9.]+).*/\1/p' | head -1
}
extract_success() {
    echo "$1" | sed -nE 's/.*successful: ([0-9]+).*/\1/p' | head -1
}

head_med=$(extract_median "$HEAD_OUT")
base_med=$(extract_median "$BASE_OUT")
head_suc=$(extract_success "$HEAD_OUT")
base_suc=$(extract_success "$BASE_OUT")

echo
echo "=== comparison ==="
echo "$HEAD_OUT"
echo
echo "$BASE_OUT"
echo
echo "=== summary ==="
printf "HEAD     (%s):  median=%s MB  successful=%s/%s\n" "$ORIG_REF" "$head_med" "$head_suc" "$RUNS"
printf "BASELINE (%s):  median=%s MB  successful=%s/%s\n" "$BASE"     "$base_med" "$base_suc" "$RUNS"
if [[ -n "$head_med" && -n "$base_med" ]]; then
    delta=$(awk -v h="$head_med" -v b="$base_med" 'BEGIN {printf "%.1f", h-b}')
    pct=$(awk -v h="$head_med" -v b="$base_med" 'BEGIN {printf "%+.1f", 100*(h-b)/b}')
    printf "delta:                  %s MB  (%s%%)\n" "$delta" "$pct"
fi
