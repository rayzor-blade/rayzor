#!/usr/bin/env bash
# Side-by-side per-layer-tensor diff between Rayzor's LlamaModel
# forward pass and llama.cpp's, on the same GGUF + same prompt.
#
# Workflow:
#   1. llama-eval-callback dumps `common_debug_cb_eval: <name> = ... sum = X`
#      for every op in the compute graph.
#   2. Rayzor's DumpLayers.hx (sibling) emits `[sum] <name> = X` at the
#      same milestones (embed, attn_norm, Q/K/V projections, attn_out,
#      ffn_inp, ffn_norm, ffn_out, l_out for each layer).
#   3. We normalise both into "name → sum" pairs and report the first
#      label whose values disagree by more than `--tol` (default 1e-3
#      relative).
#
# That label is the first stage where our pipeline diverges from the
# reference — RoPE, attention, FFN, RMSNorm, weight orientation,
# whatever. Fix it, re-run.
#
# Usage:
#   tools/llama-diff/layer-diff.sh "<prompt>"
# Environment:
#   GGUF    — path to the .gguf model.
#   RAYZOR  — path to the rayzor binary.
#   TOL     — relative tolerance for "match". Default 0.01 (1%).

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RAYZOR="${RAYZOR:-$REPO_ROOT/target/release/rayzor}"
GGUF="${GGUF:-$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf 2>/dev/null | head -1)}"
TOL="${TOL:-0.01}"

PROMPT="${1:-Hello}"

[[ -f "$GGUF" ]]     || { echo "error: GGUF not found at $GGUF" >&2; exit 1; }
[[ -x "$RAYZOR" ]]   || { echo "error: rayzor binary not found at $RAYZOR" >&2; exit 1; }
command -v llama-eval-callback >/dev/null 2>&1 \
    || { echo "error: llama-eval-callback not on PATH (brew install llama.cpp)" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "=== layer-diff ==="
echo "prompt: \"$PROMPT\""
echo "model:  $(basename "$GGUF")"
echo "tol:    $TOL (relative)"
echo ""

# ----------------------------------------------------------------------
# llama.cpp reference dump
# ----------------------------------------------------------------------
echo "[1/3] running llama-eval-callback ..."
llama-eval-callback -m "$GGUF" -p "$PROMPT" -n 1 --temp 0 \
    > "$OUT/llama.raw" 2>&1 \
    || { echo "error: llama-eval-callback failed; see $OUT/llama.raw" >&2; exit 1; }

# Pull `name = ...` from the `common_debug_cb_eval:` header lines and
# pair each with the next `sum = N` value. llama.cpp prints them in
# order so a stateful awk is enough.
awk '
    /common_debug_cb_eval:/ {
        line = $0
        sub(/.*common_debug_cb_eval: +/, "", line)
        name = line; sub(/ +=.*/, "", name)
        next
    }
    /sum = / {
        v = $0; sub(/.*sum = +/, "", v)
        if (name != "") {
            print name " " v
            name = ""
        }
    }
' "$OUT/llama.raw" > "$OUT/llama.kv"

# ----------------------------------------------------------------------
# rayzor dump
# ----------------------------------------------------------------------
echo "[2/3] running rayzor DumpLayers.hx ..."
(
    cd "$SCRIPT_DIR"
    rm -rf .rayzor  # cold cache so JIT compile is deterministic
    "$RAYZOR" run DumpLayers.hx -- "$GGUF" "$PROMPT"
) > "$OUT/rayzor.raw" 2>&1 \
    || { echo "error: rayzor DumpLayers failed; see $OUT/rayzor.raw" >&2; exit 1; }

awk '/^\[sum\]/ { print $2 " " $4 }' "$OUT/rayzor.raw" > "$OUT/rayzor.kv"

# ----------------------------------------------------------------------
# Diff
#   - join on label, walk in rayzor-emission order (which mirrors the
#     forward pass), report the first divergent row.
# ----------------------------------------------------------------------
echo "[3/3] diffing ..."
echo ""
printf "  %-22s %14s %14s %10s\n" "label" "llama.cpp" "rayzor" "rel.err"
printf "  %-22s %14s %14s %10s\n" "-----" "---------" "------" "-------"

FIRST_DIFF=""
TOLF="$TOL"
while read -r label rzv; do
    llv=$(awk -v lbl="$label" '$1 == lbl { print $2; exit }' "$OUT/llama.kv")
    if [[ -z "$llv" ]]; then
        printf "  %-22s %14s %14s %10s\n" "$label" "(missing)" "$rzv" "-"
        continue
    fi
    # Relative error (treat |llv| < 1 as absolute).
    rel=$(awk -v a="$llv" -v b="$rzv" 'BEGIN{
        d = a - b; if (d < 0) d = -d
        am = (a < 0 ? -a : a)
        denom = (am > 1 ? am : 1)
        printf "%.5f", d / denom
    }')
    over=$(awk -v r="$rel" -v t="$TOLF" 'BEGIN{print (r > t) ? 1 : 0}')
    if [[ "$over" == "1" ]]; then
        marker="<-- DIFF"
        if [[ -z "$FIRST_DIFF" ]]; then FIRST_DIFF="$label"; fi
    else
        marker=""
    fi
    printf "  %-22s %14s %14s %10s  %s\n" "$label" "$llv" "$rzv" "$rel" "$marker"
done < "$OUT/rayzor.kv"

echo ""
if [[ -z "$FIRST_DIFF" ]]; then
    echo "MATCH: every sum within tolerance ($TOL)"
else
    echo "FIRST DIVERGENCE: $FIRST_DIFF"
fi
