#!/usr/bin/env bash
# Side-by-side diff between Rayzor's nue/llama-chat and llama.cpp on the
# same GGUF + same prompt + same sampler (greedy, seed=42). When the two
# outputs disagree, the first divergent token tells us the model is
# fighting Rayzor's pipeline somewhere — embedding, attention, RoPE,
# RMSNorm, LM head, or the BPE tokenizer.
#
# Usage:
#   tools/llama-diff/compare.sh "<prompt>" [n_tokens]
#
# Environment:
#   GGUF      — path to the .gguf model. Default: Llama 3.2 1B Instruct
#               Q4_K_M from the local HF cache.
#   RAYZOR    — path to the rayzor binary. Default: target/release/rayzor.
#   N         — number of tokens to generate (overrides $2). Default: 24.
#   USE_CHAT  — 1 (default) wraps the prompt in the Llama-3 chat
#               template on both sides. 0 runs raw text completion.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CHAT_DIR="$REPO_ROOT/nue/examples/llama-chat"

RAYZOR="${RAYZOR:-$REPO_ROOT/target/release/rayzor}"
GGUF="${GGUF:-$(ls ~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf 2>/dev/null | head -1)}"
USE_CHAT="${USE_CHAT:-1}"

PROMPT="${1:-What is the capital of France?}"
N="${N:-${2:-24}}"

if [[ ! -f "$GGUF" ]]; then
    echo "error: GGUF model not found at: $GGUF" >&2
    echo "       set GGUF=<path> or place the file in the default HF cache." >&2
    exit 1
fi

if [[ ! -x "$RAYZOR" ]]; then
    echo "error: rayzor binary not found at: $RAYZOR" >&2
    echo "       run \`cargo build --release\` first." >&2
    exit 1
fi

if ! command -v llama-completion >/dev/null 2>&1; then
    echo "error: llama-completion not on PATH. install via \`brew install llama.cpp\`." >&2
    exit 1
fi

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

echo "=== llama-diff ==="
echo "model:  $(basename "$GGUF")"
echo "prompt: \"$PROMPT\""
echo "tokens: $N"
echo "mode:   $([[ "$USE_CHAT" == "1" ]] && echo "chat template" || echo "raw completion")"
echo ""

# ----------------------------------------------------------------------
# llama.cpp side
# ----------------------------------------------------------------------
echo "--- llama.cpp ---"
LLAMA_FLAGS=(-m "$GGUF" -n "$N" --temp 0 --seed 42 --no-display-prompt)
if [[ "$USE_CHAT" == "1" ]]; then
    LLAMA_FLAGS+=(--jinja)
fi
LLAMA_OUT="$OUT_DIR/llamacpp.txt"
{
    llama-completion "${LLAMA_FLAGS[@]}" -p "$PROMPT" 2>/dev/null \
        | sed -e 's/^> EOF by user$//' -e '/^[[:space:]]*$/d'
} > "$LLAMA_OUT"
cat "$LLAMA_OUT"
echo ""

# ----------------------------------------------------------------------
# Rayzor side
#
# Why temperature=0.01 instead of 0.0:
#   - 0.0 selects ArgmaxSampler, which currently SIGSEGVs partway
#     through generation when the chat template is applied
#     (separate compiler bug; see Main.hx comment block).
#   - 0.01 → LocalTempSampler with Top-K=50 and an almost-zero temp.
#     After temperature scaling, exp((logit-max)/0.01) is dominated
#     by the single top-1 logit (anything within 1.0 of max gets a
#     weight of exp(100) ≈ 2.6e43, the rest are negligible), so
#     multinomial draw effectively returns argmax. Deterministic for
#     a fixed seed, matches llama.cpp's --temp 0 within numerical
#     noise.
# ----------------------------------------------------------------------
echo "--- rayzor ---"
RAYZOR_OUT="$OUT_DIR/rayzor.txt"
RAYZOR_RAW="$OUT_DIR/rayzor.raw"
(
    cd "$LLAMA_CHAT_DIR"
    rm -rf .rayzor  # cold cache for deterministic JIT compile
    # Args: <gguf> [prompt] [max_tokens] [ctx] [temperature]
    "$RAYZOR" run Main.hx -- "$GGUF" "$PROMPT" "$N" 4096 0.01
) > "$RAYZOR_RAW" 2>&1
sed -n '/^\[gen\] streaming/,/^\[done\]/{//!p;}' "$RAYZOR_RAW" \
    | sed '/^[[:space:]]*$/d' > "$RAYZOR_OUT"
cat "$RAYZOR_OUT"
echo ""

# ----------------------------------------------------------------------
# Token-ID comparison: how each side tokenized the (chat-wrapped or
# raw) prompt. Divergence here means the BPE encoder disagrees with
# llama.cpp on byte-level rules — i.e. the model sees a different
# input. Always cheaper to fix tokenization first.
# ----------------------------------------------------------------------
echo ""
echo "--- prompt token IDs ---"

# Write the same string we feed Rayzor into a file so trailing
# newlines survive bash command-substitution stripping. Both sides
# tokenize byte-for-byte the same input.
TOK_FILE="$OUT_DIR/prompt.txt"
if [[ "$USE_CHAT" == "1" ]]; then
    printf '%s' "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

$PROMPT<|eot_id|><|start_header_id|>assistant<|end_header_id|>

" > "$TOK_FILE"
else
    printf '%s' "$PROMPT" > "$TOK_FILE"
fi

# llama.cpp prompt IDs — --no-bos so we don't double-count the BOS
# that the chat template already includes (raw mode has no BOS at
# all). Pipe via cat to preserve the trailing newlines that stdin
# from a here-string would lose.
LLAMA_IDS_FILE="$OUT_DIR/llama.ids"
cat "$TOK_FILE" | llama-tokenize -m "$GGUF" --stdin --ids --no-bos 2>/dev/null \
    | tr -d '[]' | tr ',' '\n' | awk 'NF{print $1}' > "$LLAMA_IDS_FILE"

# Rayzor prompt IDs — fished out of the `[dbg.prompt-id]` traces we
# added to Main.hx. Lines look like `[dbg.prompt-id] 0 128000`.
RAYZOR_IDS_FILE="$OUT_DIR/rayzor.ids"
grep '\[dbg\.prompt-id\]' "$RAYZOR_RAW" \
    | awk '{print $NF}' > "$RAYZOR_IDS_FILE" || true

LL_N=$(wc -l < "$LLAMA_IDS_FILE" | tr -d ' ')
RZ_N=$(wc -l < "$RAYZOR_IDS_FILE" | tr -d ' ')
echo "  llama.cpp tokens: $LL_N"
echo "  rayzor tokens:    $RZ_N"

# Side-by-side: paste columnar with `=` / `!=` marker per row
paste "$LLAMA_IDS_FILE" "$RAYZOR_IDS_FILE" \
    | awk 'BEGIN{print "  idx  llama   rayzor  status"}
           {
             status = ($1 == $2) ? "" : "  <-- DIFF"
             printf "  %3d  %5s   %5s%s\n", NR-1, $1, $2, status
           }'

# ----------------------------------------------------------------------
# Decoded-text diff
# ----------------------------------------------------------------------
echo ""
echo "--- decoded text diff ---"
if diff -u --label "llama.cpp" --label "rayzor" "$LLAMA_OUT" "$RAYZOR_OUT" > "$OUT_DIR/diff.txt"; then
    echo "MATCH: outputs are identical"
else
    cat "$OUT_DIR/diff.txt"
fi
