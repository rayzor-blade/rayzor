#!/usr/bin/env bash
# Quality gate: teacher-forced perplexity, and an A/B that reports how often
# two configurations pick the same next token.
#
#   ./gate.sh                          # PPL for HEAD, append a row
#   ./gate.sh --show
#   ./gate.sh --ab NUE_FLASH_BATCH=512 # PPL off vs on + top-1 agreement
#   ./gate.sh --force-rebuild
#
# This exists because quality decisions were being made by generating two
# samples and judging whether both "looked coherent". That cannot tell a
# slightly lossy kernel from a broken one, and it is why the batched prefill
# attention stayed opt-in even after its arithmetic was proven bit-identical to
# the shipping decode path (nue/tests/flashbatch).
#
# Perplexity is chunked from an empty cache, llama.cpp style, so the PREFILL
# path is what gets exercised -- a token-at-a-time loop would measure decode
# only and say nothing about a prefill kernel.
#
# Reading the A/B: PPL is the headline, but on a small corpus a few hundredths
# is noise. Top-1 agreement is the sharper signal -- it counts positions where
# the two configurations would emit a different token, which is exactly what
# makes generated text diverge.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HISTORY="${HISTORY:-$SCRIPT_DIR/history.tsv}"
CORPUS="${CORPUS:-$SCRIPT_DIR/corpus.txt}"

HEADER=$'date\tcommit\tppl\tnll\tpositions\tchunk\tagreement_pct\tab_var\tab_ppl\tquiet\tmodel\tsubject'

FORCE_REBUILD="${FORCE_REBUILD:-0}"
AB_VAR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      [[ -f "$HISTORY" ]] && column -t -s $'\t' "$HISTORY" || echo "no history yet: $HISTORY"
      exit 0 ;;
    --ab) AB_VAR="$2"; shift 2 ;;
    --force-rebuild|-f) FORCE_REBUILD=1; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

MODEL="${GGUF:-$HOME/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf}"
CHUNK="${CHUNK:-256}"
MAX_CHUNKS="${MAX_CHUNKS:-6}"

[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (set GGUF=)" >&2; exit 2; }
[[ -f "$CORPUS" ]] || { echo "error: corpus not found: $CORPUS" >&2; exit 2; }

bench_provenance "$REPO"
echo ">> $BENCH_COMMIT$BENCH_DIRTY  $(basename "$MODEL")  chunk=$CHUNK x $MAX_CHUNKS"
bench_build_if_stale "$REPO" "$FORCE_REBUILD" || exit 2

pkill -9 -f "[t]arget/release/rayzor" 2>/dev/null || true
bench_quiet_wait
bench_watch_start

run_eval() { # <dump-path> ; env for the variant comes from the caller
  ( cd "$SCRIPT_DIR" && "$REPO/target/release/rayzor" run Main.hx --release --llvm \
      --safety-warnings off -- "$MODEL" "$CORPUS" --chunk "$CHUNK" \
      --max-chunks "$MAX_CHUNKS" --dump "$1" 2>/dev/null )
}

echo ">> scoring baseline"
base_out="$(run_eval /tmp/nue_eval_base.tsv)"
ppl="$(echo "$base_out" | grep -a "^PPL" | awk '{print $2}')"
nll="$(echo "$base_out" | grep -a "^NLL" | awk '{print $2}')"
positions="$(echo "$base_out" | grep -ao "positions=[0-9]*" | head -1 | cut -d= -f2)"

agreement="NA"; ab_ppl="NA"
if [[ -n "$AB_VAR" ]]; then
  echo ">> scoring variant $AB_VAR"
  ab_out="$( export "${AB_VAR?}"; run_eval /tmp/nue_eval_ab.tsv )"
  ab_ppl="$(echo "$ab_out" | grep -a "^PPL" | awk '{print $2}')"
  agreement="$(awk -F'\t' '
    NR == FNR { a[$1] = $2; next }
    ($1 in a) { n++; if (a[$1] == $2) same++ }
    END { if (n) printf "%.2f", 100 * same / n; else print "NA" }
  ' /tmp/nue_eval_base.tsv /tmp/nue_eval_ab.tsv)"
fi

bench_watch_stop

[[ -n "$ppl" ]] || { echo "error: no PPL in output" >&2; echo "$base_out" | tail -20 >&2; exit 2; }

bench_history_append "$HISTORY" "$HEADER" "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
  "$(date +%Y-%m-%d)" "$BENCH_COMMIT$BENCH_DIRTY" "$ppl" "${nll:-NA}" "${positions:-NA}" \
  "$CHUNK" "$agreement" "${AB_VAR:-none}" "$ab_ppl" "$BENCH_QUIET" \
  "$(basename "$MODEL")" "$BENCH_SUBJECT")"

echo
echo "  $BENCH_COMMIT$BENCH_DIRTY"
echo "    quality    PPL=$ppl  NLL=${nll:-NA}  over ${positions:-?} positions"
if [[ -n "$AB_VAR" ]]; then
  echo "    variant    $AB_VAR -> PPL=$ab_ppl"
  echo "    agreement  top-1 identical on ${agreement}% of positions"
fi
# PPL and agreement are deterministic: contention changes how long the scoring
# takes, not what it computes. The machine line is recorded for completeness,
# but a busy box does NOT invalidate a quality row the way it invalidates a
# throughput row.
echo "    machine    quiet=$BENCH_QUIET  (peak foreign ${BENCH_PEAK_CPU}% of one core)"
[[ "$BENCH_QUIET" != "yes" ]] && echo "               (timing only -- these numbers are deterministic)"
echo "  history: $HISTORY"
