#!/usr/bin/env bash
# Mistral-7B head-to-head against llama.cpp, prefill and decode separately.
#
#   ./gate.sh                 # measure both, append a row
#   ./gate.sh --show
#   ./gate.sh --ours-only     # skip llama.cpp
#   ./gate.sh --force-rebuild
#
# Why a separate bench from llama-chat: 7B on a 16 GB box is memory-bound in a
# way the 0.5B model never is, and the comparison has protocol requirements the
# other harnesses do not.
#
# PROTOCOL -- both were got wrong before and each produced a fake result:
#
#   decode must be matched on n AND CONTEXT LENGTH. llama.cpp's tg128 generates
#   from a nearly empty context. Timing our decode after a 626-token prompt and
#   comparing it to tg128 once looked like a 4.7x regression; it was two
#   different measurements. Decode here uses a SHORT prompt.
#
#   prefill is measured in tokens of THIS model's tokenizer via llama-tokenize.
#   Do not eyeball a token count and do not reuse another model's -- the same
#   prompt is 626 tokens to Mistral and 653 to Llama.
#
# MEMORY: the model is ~4.1 GB on a 16 GB machine. If the box is already deep
# in swap, the load streams from disk and the run measures the SSD -- a 10
# minute timeout with no output is the symptom. The gate refuses to start below
# MIN_FREE_PCT rather than record a number that is really a page-fault trace.
#
# llama.cpp's build moves fast (their decode nearly doubled in 3 days once), so
# the baseline is re-measured on every run, never quoted from history.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="$REPO/nue/examples/llama-chat"
HISTORY="${HISTORY:-$SCRIPT_DIR/history.tsv}"

HEADER=$'date\tcommit\tprompt_tok\tttft_s\tprefill_tok_s\tdecode_tok_s\tcpp_pp\tcpp_tg\tprefill_ratio\tdecode_ratio\tfree_pct\tquiet\tmodel\tsubject'

FORCE_REBUILD="${FORCE_REBUILD:-0}"
OURS_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      [[ -f "$HISTORY" ]] && column -t -s $'\t' "$HISTORY" || echo "no history yet: $HISTORY"
      exit 0 ;;
    --ours-only) OURS_ONLY=1; shift ;;
    --force-rebuild|-f) FORCE_REBUILD=1; shift ;;
    -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

MODEL="${GGUF:-$HOME/models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
CTX="${CTX:-4096}"
THREADS="${THREADS:-8}"
MIN_FREE_PCT="${MIN_FREE_PCT:-45}"
BUNDLE_OUT="${BUNDLE_OUT:-/tmp/nue_bench_mistral.rzb}"

[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (set GGUF=)" >&2; exit 2; }

free_pct() {
  memory_pressure 2>/dev/null | awk -F: '/free percentage/ { gsub(/[^0-9]/, "", $2); print $2; exit }'
}

fp="$(free_pct)"
if [[ -n "$fp" && "$fp" -lt "$MIN_FREE_PCT" ]]; then
  echo "error: only ${fp}% memory free (need ${MIN_FREE_PCT}%). A 4 GB model on a" >&2
  echo "       swapping box measures the SSD, not the kernels. Free memory or set" >&2
  echo "       MIN_FREE_PCT to override." >&2
  sysctl vm.swapusage 2>/dev/null | sed 's/^/       /' >&2
  exit 3
fi

bench_provenance "$REPO"
echo ">> $BENCH_COMMIT$BENCH_DIRTY  $(basename "$MODEL")  free=${fp:-?}%"
bench_build_if_stale "$REPO" "$FORCE_REBUILD" || exit 2

if [[ "$FORCE_REBUILD" == "1" ]] \
   || bench_stale "$BUNDLE_OUT" "$APP_DIR" "$REPO/nue/nue" "$REPO/compiler/haxe-std" "$REPO/target/release/rayzor"; then
  echo ">> bundling"
  ( cd "$APP_DIR" && "$REPO/target/release/rayzor" bundle Main.hx -o "$BUNDLE_OUT" --no-cache >/dev/null 2>&1 ) \
    || { echo "error: bundle failed" >&2; exit 2; }
else
  echo ">> bundle up to date"
fi

# A long prompt, sized in THIS model's tokens.
LONG_PROMPT="$(awk 'BEGIN {
  s = "A B-tree is a self-balancing tree data structure that keeps data sorted and allows searches, sequential access, insertions and deletions in logarithmic time. High node fanout reduces tree height and therefore the number of disk pages a lookup must touch. ";
  out = "";
  for (i = 0; i < 14; i++) out = out s;
  printf "%sSummarise the passage above in one sentence.", out
}')"
PROMPT_TOK="?"
if command -v llama-tokenize >/dev/null 2>&1; then
  PROMPT_TOK="$(printf '%s' "$LONG_PROMPT" | llama-tokenize -m "$MODEL" --stdin 2>/dev/null | grep -c . || echo "?")"
fi
echo ">> prompt = ${PROMPT_TOK} tokens (this model's tokenizer)"

pkill -9 -f "[t]arget/release/rayzor" 2>/dev/null || true
bench_quiet_wait
bench_watch_start

# PREFILL: long prompt, few generated tokens -- ttft is the measurement.
echo ">> rayzor prefill"
pre_line="$( cd "$APP_DIR" && BUNDLE="$BUNDLE_OUT" BUILD=0 TEMP=0 REP_PENALTY=1.0 \
  MAX_TOKENS=8 CTX="$CTX" PROMPT="$LONG_PROMPT" GGUF="$MODEL" \
  timeout 1800 ./run_bundle.sh 2>/dev/null | grep -a "^\[done\]" | head -1 )"
ttft="$(echo "$pre_line" | grep -ao "ttft=[0-9.]*" | cut -d= -f2)"

# DECODE: short prompt, so the context matches llama.cpp's tg128.
echo ">> rayzor decode ($DECODE_TOKENS tokens, short context)"
dec_line="$( cd "$APP_DIR" && BUNDLE="$BUNDLE_OUT" BUILD=0 TEMP=0 REP_PENALTY=1.0 \
  MAX_TOKENS="$DECODE_TOKENS" CTX="$CTX" PROMPT="Count slowly." GGUF="$MODEL" \
  timeout 1800 ./run_bundle.sh 2>/dev/null | grep -a "^\[done\]" | head -1 )"
d_tok="$(echo "$dec_line" | sed -n 's/^\[done\] \([0-9][0-9]*\) tokens.*/\1/p')"
d_tot="$(echo "$dec_line" | sed -n 's/^\[done\] [0-9]* tokens in \([0-9.][0-9.]*\)s.*/\1/p')"
d_ttft="$(echo "$dec_line" | grep -ao "ttft=[0-9.]*" | cut -d= -f2)"
decode_tps="$(awk -v n="${d_tok:-0}" -v t="${d_tot:-0}" -v f="${d_ttft:-0}" \
  'BEGIN { g = t - f; if (g > 0 && n > 0) printf "%.2f", n / g; else print "NA" }')"
prefill_tps="$(awk -v n="${PROMPT_TOK:-0}" -v t="${ttft:-0}" \
  'BEGIN { if (t > 0 && n > 0) printf "%.2f", n / t; else print "NA" }')"

cpp_pp="NA"; cpp_tg="NA"
if [[ "$OURS_ONLY" -eq 0 ]] && command -v llama-bench >/dev/null 2>&1; then
  echo ">> llama.cpp baseline (re-measured, never quoted from history)"
  cpp_out="$(llama-bench -m "$MODEL" -ngl 0 -t "$THREADS" -p 512 -n 128 -r 3 2>/dev/null)"
  cpp_pp="$(echo "$cpp_out" | awk -F'|' '/pp512/ { gsub(/ /,"",$(NF-1)); split($(NF-1), a, "±"); print a[1]; exit }')"
  cpp_tg="$(echo "$cpp_out" | awk -F'|' '/tg128/ { gsub(/ /,"",$(NF-1)); split($(NF-1), a, "±"); print a[1]; exit }')"
fi

bench_watch_stop

ratio() { awk -v a="$1" -v b="$2" 'BEGIN { if (a+0>0 && b+0>0) printf "%.2f", b/a; else print "NA" }'; }
pre_ratio="$(ratio "$prefill_tps" "$cpp_pp")"   # >1 = they are faster
dec_ratio="$(ratio "$decode_tps" "$cpp_tg")"

bench_history_append "$HISTORY" "$HEADER" "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
  "$(date +%Y-%m-%d)" "$BENCH_COMMIT$BENCH_DIRTY" "$PROMPT_TOK" "${ttft:-NA}" "$prefill_tps" \
  "$decode_tps" "$cpp_pp" "$cpp_tg" "$pre_ratio" "$dec_ratio" "${fp:-NA}" "$BENCH_QUIET" \
  "$(basename "$MODEL")" "$BENCH_SUBJECT")"

echo
echo "  $BENCH_COMMIT$BENCH_DIRTY   $(basename "$MODEL")"
echo "    prefill    ttft=${ttft:-NA}s over ${PROMPT_TOK} tok = ${prefill_tps} tok/s   llama.cpp pp512=${cpp_pp}"
echo "    decode     ${decode_tps} tok/s over ${DECODE_TOKENS} tok, short ctx        llama.cpp tg128=${cpp_tg}"
echo "    ratios     prefill ${pre_ratio}x, decode ${dec_ratio}x  (>1 = llama.cpp ahead)"
echo "    machine    quiet=$BENCH_QUIET  free=${fp:-?}%"
bench_report "$BENCH_QUIET" ""
echo "  history: $HISTORY"
