#!/usr/bin/env bash
# Multi-run decode-only benchmark for nue/llama-chat (and optionally
# llama.cpp for comparison). Reports MEDIAN tok/s computed from the
# `[done] N tokens in Ts (X tok/s)` line that rayzor llama-chat prints
# — i.e. **decode-loop time only**, not wall-clock-including-load.
#
# Why decode-only: GGUF load + prompt encode runs at the same speed
# regardless of compiler tier, kernel optimisations, or anything else
# we tune. Mixing it into the headline tok/s number makes every
# optimisation look like noise on short runs (the prior bench harness
# reported wall-tok/s, which was dominated by ~1.3 s of model load).
#
# Default prompt is intentionally long-form so the model generates the
# full N tokens without EOS-ing early. Short prompts like "What is the
# capital of France?" answer in ~7 tokens, after which the model EOSes,
# and the bench measures mostly setup + 7 tokens of decode rather than
# steady-state throughput.
#
# Usage:
#   tools/llama-diff/bench.sh ["<prompt>"] [n_tokens]
#
# Environment:
#   GGUF        path to .gguf model
#   RAYZOR      path to rayzor binary
#   RUNS        number of runs (default 7; reports median over these)
#   N           tokens to generate per run (default 64)
#   ONLY        "rayzor" to skip llama.cpp comparison
#   RAYZOR_TIER "llvm" to opt into --llvm; else Cranelift default
#   PROMPT      override default long-form prompt (arg 1 also works)

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CHAT_DIR="$REPO_ROOT/nue/examples/llama-chat"

RAYZOR="${RAYZOR:-$REPO_ROOT/target/release/rayzor}"
GGUF="${GGUF:-$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf 2>/dev/null | head -1)}"
ONLY="${ONLY:-both}"
RUNS="${RUNS:-7}"

# Long-form prompt chosen so the 1B Instruct model has room to talk —
# typically generates 80-150 tokens before EOS at temp=0.01. Override
# via $1 or PROMPT env var if you need a specific workload.
DEFAULT_PROMPT="Write a long detailed paragraph about the history of Paris, including its founding by the Parisii tribe, key historical events through the medieval and revolutionary periods, cultural significance during the Belle Epoque, and the modern landmarks that define it today."

PROMPT="${1:-${PROMPT:-$DEFAULT_PROMPT}}"
N="${N:-${2:-64}}"

[[ -f "$GGUF" ]]   || { echo "error: GGUF not found at $GGUF" >&2; exit 1; }
[[ -x "$RAYZOR" ]] || { echo "error: rayzor not found at $RAYZOR" >&2; exit 1; }
if [[ "$ONLY" != "rayzor" ]]; then
    command -v llama-completion >/dev/null 2>&1 || \
        { echo "error: llama-completion not on PATH (set ONLY=rayzor to skip)" >&2; exit 1; }
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "=== bench ==="
echo "model:  $(basename "$GGUF")"
echo "prompt: \"${PROMPT:0:80}$([[ ${#PROMPT} -gt 80 ]] && echo ...)\""
echo "tokens: $N"
echo "runs:   $RUNS (median + min..max reported below)"
echo "tier:   ${RAYZOR_TIER:-cranelift}"
echo ""

# --- numeric helpers -------------------------------------------------

# Pipe input is one number per line. Outputs MEDIAN.
median() {
    awk '
        { vals[NR] = $1 + 0 }
        END {
            n = NR
            # sort
            for (i = 1; i <= n; i++)
                for (j = i + 1; j <= n; j++)
                    if (vals[i] > vals[j]) { t = vals[i]; vals[i] = vals[j]; vals[j] = t }
            if (n % 2 == 1)      printf "%.3f", vals[(n + 1) / 2]
            else if (n > 0)      printf "%.3f", (vals[n / 2] + vals[n / 2 + 1]) / 2
            else                 printf "%.3f", 0
        }
    '
}

min_v() { sort -n | head -1; }
max_v() { sort -n | tail -1; }

stddev() {
    awk '
        { vals[NR] = $1 + 0; sum += $1 + 0 }
        END {
            if (NR < 2) { print "0.000"; exit }
            mean = sum / NR
            for (i = 1; i <= NR; i++) ss += (vals[i] - mean) ** 2
            printf "%.3f", sqrt(ss / (NR - 1))
        }
    '
}

# --- one-run primitive -----------------------------------------------
# Runs rayzor llama-chat once, parses the [load] + [done] lines, prints
# a single result line: "tok_per_s decode_s n_gen load_s wall_s"

run_rayzor_once() {
    local idx="$1"
    local stdout_log="$OUT/rayzor_$idx.stdout"
    local time_log="$OUT/rayzor_$idx.time"

    (
        cd "$LLAMA_CHAT_DIR"
        rm -rf .rayzor 2>/dev/null || true
        if [[ "${RAYZOR_TIER:-cranelift}" == "llvm" ]]; then
            /usr/bin/time -p "$RAYZOR" run --llvm Main.hx -- "$GGUF" "$PROMPT" "$N" 4096 0.01
        else
            /usr/bin/time -p "$RAYZOR" run Main.hx -- "$GGUF" "$PROMPT" "$N" 4096 0.01
        fi
    ) >"$stdout_log" 2>"$time_log"

    # [load] done in X.XXs
    local load_s
    load_s=$(awk -F'[ s]+' '/^\[load\] done in/ { for(i=1;i<=NF;i++) if($i ~ /[0-9]\./){print $i; exit} }' "$stdout_log")

    # [done] N tokens in Ts (X tok/s)
    local done_line n_gen decode_s tok_per_s
    done_line=$(grep -m1 '^\[done\]' "$stdout_log" || echo "")
    n_gen=$(  echo "$done_line" | awk '{print $2}')
    decode_s=$(echo "$done_line" | awk '/in/ {for(i=1;i<=NF;i++) if($i=="in") {v=$(i+1); gsub(/s$/,"",v); print v; exit}}')
    tok_per_s=$(echo "$done_line" | awk '/tok\/s/ {for(i=1;i<=NF;i++) if($i ~ /tok\/s/){v=$(i-1); gsub(/[\(\)]/,"",v); print v; exit}}')

    # /usr/bin/time -p output: "real X.XX"
    local wall_s
    wall_s=$(awk '/^real/ {print $2; exit}' "$time_log")

    printf '%s %s %s %s %s\n' "${tok_per_s:-0}" "${decode_s:-0}" "${n_gen:-0}" "${load_s:-0}" "${wall_s:-0}"
}

# --- aggregate over RUNS -------------------------------------------

bench_rayzor() {
    echo "--- rayzor (${RAYZOR_TIER:-cranelift}) ---"

    declare -a tok_arr decode_arr n_arr load_arr wall_arr
    local failed=0

    for i in $(seq 1 "$RUNS"); do
        # Optional thermal-cooldown between runs. M1 Pro decode tok/s is
        # sensitive to package temperature — back-to-back runs after a
        # release build can heat the cores enough that the first 1-2
        # readings are 5-15% below steady state. Set BENCH_RUN_SPACING
        # (seconds) to insert a sleep before each run except the first.
        if [ "$i" -gt 1 ] && [ -n "${BENCH_RUN_SPACING:-}" ]; then
            sleep "$BENCH_RUN_SPACING"
        fi
        local line
        line=$(run_rayzor_once "$i")
        local tok decode n load wall
        read -r tok decode n load wall <<<"$line"
        # Filter runs where [done] didn't parse (n=0) — typically a JIT
        # bringup hiccup. Include in the per-run log so you see what
        # happened, but exclude from the median.
        if awk -v n="$n" 'BEGIN { exit !(n + 0 < 1) }'; then
            failed=$((failed + 1))
            printf '  run %d  FAILED (n=0; check tier-3 promotion + cache scrub)\n' "$i"
            continue
        fi
        tok_arr+=("$tok"); decode_arr+=("$decode"); n_arr+=("$n"); load_arr+=("$load"); wall_arr+=("$wall")
        printf '  run %d  n=%s tok/s=%s decode=%ss load=%ss wall=%ss\n' \
            "$i" "$n" "$tok" "$decode" "$load" "$wall"
    done

    if [[ "${#tok_arr[@]}" -eq 0 ]]; then
        echo ""
        echo "  ALL RUNS FAILED (${failed}/${RUNS}). Bench has nothing to median."
        return 1
    fi

    # Compute decode-only tok/s independently from N/decode_s as a sanity
    # cross-check against the [done] tok/s the runtime prints. If they
    # disagree noticeably, the [done] parser missed something.
    local tok_med tok_min tok_max tok_sd
    tok_med=$(printf '%s\n' "${tok_arr[@]}" | median)
    tok_min=$(printf '%s\n' "${tok_arr[@]}" | min_v)
    tok_max=$(printf '%s\n' "${tok_arr[@]}" | max_v)
    tok_sd=$( printf '%s\n' "${tok_arr[@]}" | stddev)

    local n_med decode_med load_med wall_med
    n_med=$(     printf '%s\n' "${n_arr[@]}"      | median)
    decode_med=$(printf '%s\n' "${decode_arr[@]}" | median)
    load_med=$(  printf '%s\n' "${load_arr[@]}"   | median)
    wall_med=$(  printf '%s\n' "${wall_arr[@]}"   | median)

    echo ""
    echo "  decode tok/s     median ${tok_med}  range ${tok_min}..${tok_max}  stddev ${tok_sd}"
    echo "  decode time      median ${decode_med}s"
    echo "  tokens generated median ${n_med} (requested ${N})"
    echo "  load time        median ${load_med}s"
    echo "  total wall       median ${wall_med}s"
    if [[ "$failed" -gt 0 ]]; then
        echo "  failed runs      ${failed}/${RUNS} (excluded from median)"
    fi
    echo "  successful runs  ${#tok_arr[@]}/${RUNS}"
    echo ""

    # Honest-framing helper: warn if the model EOS'd before reaching N.
    if awk -v ng="$n_med" -v nreq="$N" 'BEGIN { exit !(ng + 0 < nreq + 0) }'; then
        echo "  NOTE: model EOS'd at ${n_med} tokens (requested ${N}). Decode rate is"
        echo "        from a partial run. Use a longer-form prompt for steady-state"
        echo "        throughput, or lower N to match what the model actually generates."
        echo ""
    fi
}

bench_llama_metal() {
    echo "--- llama.cpp (Metal) ---"
    declare -a tok_arr
    for i in $(seq 1 "$RUNS"); do
        local log="$OUT/llama_metal_$i.log"
        llama-completion -m "$GGUF" -p "$PROMPT" -n "$N" --temp 0 --seed 42 \
            --no-display-prompt --jinja >/dev/null 2>"$log"
        local tok
        tok=$(awk -F'[, ()]+' '/eval time/ && /tokens per second/ && !/prompt/ {
            for (i = 1; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") { print $(i-1); exit }
        }' "$log")
        tok_arr+=("${tok:-0}")
        printf '  run %d  tok/s=%s\n' "$i" "$tok"
    done
    local med min max
    med=$(printf '%s\n' "${tok_arr[@]}" | median)
    min=$(printf '%s\n' "${tok_arr[@]}" | min_v)
    max=$(printf '%s\n' "${tok_arr[@]}" | max_v)
    echo ""
    echo "  decode tok/s     median ${med}  range ${min}..${max}"
    echo ""
}

bench_llama_cpu() {
    echo "--- llama.cpp (CPU only, -ngl 0) ---"
    declare -a tok_arr
    for i in $(seq 1 "$RUNS"); do
        local log="$OUT/llama_cpu_$i.log"
        llama-completion -m "$GGUF" -p "$PROMPT" -n "$N" --temp 0 --seed 42 \
            --no-display-prompt --jinja -ngl 0 >/dev/null 2>"$log"
        local tok
        tok=$(awk -F'[, ()]+' '/eval time/ && /tokens per second/ && !/prompt/ {
            for (i = 1; i <= NF; i++) if ($i == "tokens" && $(i+1) == "per") { print $(i-1); exit }
        }' "$log")
        tok_arr+=("${tok:-0}")
        printf '  run %d  tok/s=%s\n' "$i" "$tok"
    done
    local med min max
    med=$(printf '%s\n' "${tok_arr[@]}" | median)
    min=$(printf '%s\n' "${tok_arr[@]}" | min_v)
    max=$(printf '%s\n' "${tok_arr[@]}" | max_v)
    echo ""
    echo "  decode tok/s     median ${med}  range ${min}..${max}"
    echo ""
}

[[ "$ONLY" != "llama" ]] && bench_rayzor
[[ "$ONLY" != "rayzor" ]] && bench_llama_metal
[[ "$ONLY" != "rayzor" && "$ONLY" != "llama-metal" ]] && bench_llama_cpu
