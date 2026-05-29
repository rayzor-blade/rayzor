#!/usr/bin/env bash
# Resource benchmark for the same prompt+model on Rayzor's
# nue/llama-chat and llama.cpp. Captures:
#   - wall time
#   - peak RSS (resident memory)
#   - peak %CPU (sampled)
#   - user / sys CPU time
#   - tokens / second
#
# macOS implementation: `/usr/bin/time -l` for end-of-run totals
# (peak RSS in bytes, user+sys CPU time), plus a background sampler
# that polls `ps -o rss,pcpu` while the child is alive to capture
# instantaneous peaks.
#
# Usage:
#   tools/llama-diff/bench.sh "<prompt>" [n_tokens]
# Environment:
#   GGUF      — path to the .gguf model.
#   RAYZOR    — path to the rayzor binary.
#   N         — number of tokens to generate.
#   ONLY      — "rayzor" or "llama" to skip the other side.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CHAT_DIR="$REPO_ROOT/nue/examples/llama-chat"

RAYZOR="${RAYZOR:-$REPO_ROOT/target/release/rayzor}"
GGUF="${GGUF:-$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf 2>/dev/null | head -1)}"
ONLY="${ONLY:-both}"

PROMPT="${1:-What is the capital of France?}"
N="${N:-${2:-32}}"

[[ -f "$GGUF" ]]   || { echo "error: GGUF not found at $GGUF" >&2; exit 1; }
[[ -x "$RAYZOR" ]] || { echo "error: rayzor not found at $RAYZOR" >&2; exit 1; }
command -v llama-completion >/dev/null 2>&1 || \
    { echo "error: llama-completion not on PATH" >&2; exit 1; }
command -v /usr/bin/time >/dev/null 2>&1 || \
    { echo "error: /usr/bin/time not available" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "=== bench ==="
echo "model:  $(basename "$GGUF")"
echo "prompt: \"$PROMPT\""
echo "tokens: $N"
echo ""

# ----------------------------------------------------------------------
# Sampler: polls `ps -o rss,pcpu` for every descendant of $1 every
# 0.2s. macOS `/usr/bin/time -l` fork+execs the child, so $! gives us
# the time wrapper, not the actual binary — we have to walk the
# process tree to find the heavyweight worker.
# ----------------------------------------------------------------------
sample() {
    local time_pid="$1"
    local out="$2"
    while kill -0 "$time_pid" 2>/dev/null; do
        # Descendants of the /usr/bin/time wrapper. On macOS the child
        # is exec'd directly, so pgrep -P sees it as a child of time(1).
        local kids
        kids=$(pgrep -P "$time_pid" 2>/dev/null || true)
        for p in $kids; do
            ps -o rss=,pcpu= -p "$p" 2>/dev/null >> "$out"
        done
        sleep 0.2
    done
}

# ----------------------------------------------------------------------
# Reduce sample log → peaks + mean for CPU.
# ----------------------------------------------------------------------
summarize() {
    local label="$1"
    local time_log="$2"
    local sample_log="$3"
    local stdout_log="$4"

    local peak_rss_kb peak_pcpu mean_pcpu n_samples
    peak_rss_kb=$(awk 'NF>=2 && $1+0 > max {max=$1+0} END{print (max ? max : 0)}' "$sample_log")
    peak_pcpu=$(awk 'NF>=2 && $2+0 > max {max=$2+0} END{print (max ? max+0 : 0)}' "$sample_log")
    n_samples=$(wc -l < "$sample_log" | tr -d ' ')
    mean_pcpu=$(awk 'NF>=2 {sum+=$2+0; n++} END{print (n ? sum/n : 0)}' "$sample_log")

    local peak_rss_mb
    peak_rss_mb=$(awk -v kb="$peak_rss_kb" 'BEGIN{printf "%.1f", kb/1024}')

    # /usr/bin/time -l output lines we care about:
    #   "        N.NN real         N.NN user         N.NN sys"
    #   "        NNN  maximum resident set size"  (bytes on macOS)
    local wall user sys time_rss_b time_rss_mb
    wall=$(awk '/real/ && /user/ && /sys/ {print $1; exit}' "$time_log")
    user=$(awk '/real/ && /user/ && /sys/ {print $3; exit}' "$time_log")
    sys=$(awk  '/real/ && /user/ && /sys/ {print $5; exit}' "$time_log")
    time_rss_b=$(awk '/maximum resident set size/{print $1; exit}' "$time_log")
    time_rss_mb=$(awk -v b="$time_rss_b" 'BEGIN{printf "%.1f", b/1024/1024}')

    # Try to read tok/s. Rayzor llama-chat prints `[done] N tokens in T s (X tok/s)`;
    # llama.cpp prints `eval time = ... ( X.XX tokens per second)` (on stderr,
    # so we also grep the time_log which captured everything stderr).
    local tok_per_s
    tok_per_s=$(awk '/^\[done\]/ {
        # Match e.g. "[done] 24 tokens in 40.82s (0.59 tok/s)"
        for (i=1; i<=NF; i++) if ($i ~ /tok\/s/) {
            v = $(i-1); gsub(/[\(\)]/, "", v); print v; exit
        }
    }' "$stdout_log")
    if [[ -z "$tok_per_s" ]]; then
        tok_per_s=$(awk -F'[, ()]+' '/eval time/ && /tokens per second/ && !/prompt/ {
            # split on commas+parens+spaces; "tokens" follows the rate.
            for (i=1; i<=NF; i++) if ($i == "tokens" && $(i+1) == "per") { print $(i-1); exit }
        }' "$time_log" "$stdout_log")
    fi

    printf "  %-12s %s\n"   "wall"      "${wall:-?} s"
    printf "  %-12s %s\n"   "user CPU"  "${user:-?} s"
    printf "  %-12s %s\n"   "sys CPU"   "${sys:-?} s"
    printf "  %-12s %s\n"   "peak RSS"  "${time_rss_mb} MB (time) / ${peak_rss_mb} MB (sampled)"
    printf "  %-12s %s\n"   "peak %CPU" "${peak_pcpu}%"
    printf "  %-12s %.1f%%\n" "mean %CPU" "${mean_pcpu}"
    printf "  %-12s %s\n"   "samples"   "${n_samples}"
    printf "  %-12s %s\n"   "tok/s"     "${tok_per_s:-?}"
}

# ----------------------------------------------------------------------
# Run one binary under /usr/bin/time -l + the sampler.
# Returns via summarize() at the end.
# ----------------------------------------------------------------------
run_one() {
    local label="$1"
    local stdout_log="$OUT/$label.stdout"
    local time_log="$OUT/$label.time"
    local sample_log="$OUT/$label.sample"
    shift
    local workdir="$1"; shift

    echo "--- $label ---"
    (
        cd "$workdir"
        rm -rf .rayzor 2>/dev/null || true
        # `/usr/bin/time -l <cmd>` writes its own report to stderr.
        # We capture stdout separately. The sampler reads the
        # /usr/bin/time process's PID (not the child shell's) because
        # time wraps the child, but $! gives us the right tree.
        /usr/bin/time -l "$@" >"$stdout_log" 2>"$time_log" &
        local pid=$!
        # Wait one tick for the actual binary to spawn under time(1),
        # then sample by pgrep'ing children of that wrapper.
        sleep 0.1
        sample "$pid" "$sample_log" &
        local sampler=$!
        wait "$pid"
        local rc=$?
        kill "$sampler" 2>/dev/null || true
        wait "$sampler" 2>/dev/null || true
        exit "$rc"
    )
    local rc=$?

    summarize "$label" "$time_log" "$sample_log" "$stdout_log"
    if [[ "$rc" -ne 0 ]]; then
        echo "  exit:        $rc (non-zero — see $time_log)"
    fi
    echo ""
}

# ----------------------------------------------------------------------
# Rayzor
# ----------------------------------------------------------------------
if [[ "$ONLY" != "llama" ]]; then
    run_one "rayzor" "$LLAMA_CHAT_DIR" \
        "$RAYZOR" run Main.hx -- "$GGUF" "$PROMPT" "$N" 4096 0.01
fi

# ----------------------------------------------------------------------
# llama.cpp — both Metal (default) and a CPU-only run so Rayzor (pure
# CPU) can be fairly compared head-to-head against llama.cpp on the
# same backend.
# ----------------------------------------------------------------------
if [[ "$ONLY" != "rayzor" ]]; then
    run_one "llama.cpp (Metal)" "$REPO_ROOT" \
        llama-completion -m "$GGUF" -p "$PROMPT" -n "$N" --temp 0 --seed 42 \
            --no-display-prompt --jinja
fi
if [[ "$ONLY" != "rayzor" && "$ONLY" != "llama-metal" ]]; then
    run_one "llama.cpp (CPU)" "$REPO_ROOT" \
        llama-completion -m "$GGUF" -p "$PROMPT" -n "$N" --temp 0 --seed 42 \
            --no-display-prompt --jinja -ngl 0
fi
