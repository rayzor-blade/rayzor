#!/usr/bin/env bash
# Single-shot generation bench for nue/examples/llama-chat: the one that
# separates PREFILL from DECODE.
#
#   ./gate.sh                    # measure HEAD, append a row
#   ./gate.sh --show
#   ./gate.sh --force-rebuild
#   PROMPT_TOKENS=500 REPS=3 ./gate.sh
#
# The server bench reports end-to-end serving throughput, which is dominated by
# decode and hides prefill entirely. Attention work at seqQ>1 only shows up in
# ttft, so a prefill change can be a 3x win here and invisible there -- the
# batched attention kernel measured ttft 2.81s -> 0.93s while server tok/s
# barely moved.
#
# Prompt length is set in TOKENS, not characters, because that is what prefill
# cost scales with. The prompt is synthesised deterministically so the number
# is comparable across machines and dates.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="$REPO/nue/examples/llama-chat"
HISTORY="${HISTORY:-$SCRIPT_DIR/history.tsv}"

HEADER=$'date\tcommit\tttft_s\tdecode_tok_s\te2e_tok_s\tprefill_tok_s\tgen_tokens\tprompt_tokens\treps\tttft_spread_pct\tquiet\tflash_batch\tmodel\tsubject'

FORCE_REBUILD="${FORCE_REBUILD:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      [[ -f "$HISTORY" ]] && column -t -s $'\t' "$HISTORY" || echo "no history yet: $HISTORY"
      exit 0 ;;
    --force-rebuild|-f) FORCE_REBUILD=1; shift ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

MODEL="${GGUF:-$HOME/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf}"
PROMPT_TOKENS="${PROMPT_TOKENS:-500}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CTX="${CTX:-4096}"
REPS="${REPS:-3}"
BUNDLE_OUT="${BUNDLE_OUT:-/tmp/nue_bench_chat.rzb}"

[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (set GGUF=)" >&2; exit 2; }

# ~1.35 tokens per word for this prose on a BPE vocab; close enough that the
# reported prompt_tokens column carries the exact figure.
PROMPT="$(awk -v want="$PROMPT_TOKENS" 'BEGIN {
  s = "A B-tree is a self-balancing tree data structure that keeps data sorted and allows searches, sequential access, insertions and deletions in logarithmic time. High node fanout reduces tree height and therefore the number of disk pages a lookup must touch. ";
  n = int(want / 45) + 1;
  out = "";
  for (i = 0; i < n; i++) out = out s;
  printf "%sSummarise the passage above in one sentence.", out
}')"

bench_provenance "$REPO"
echo ">> $BENCH_COMMIT$BENCH_DIRTY  $(basename "$MODEL")  prompt~${PROMPT_TOKENS} tok"
bench_build_if_stale "$REPO" "$FORCE_REBUILD" || exit 2

if [[ "$FORCE_REBUILD" == "1" ]] \
   || bench_stale "$BUNDLE_OUT" "$APP_DIR" "$REPO/nue/nue" "$REPO/compiler/haxe-std" "$REPO/target/release/rayzor"; then
  echo ">> bundling"
  ( cd "$APP_DIR" && "$REPO/target/release/rayzor" bundle Main.hx -o "$BUNDLE_OUT" --no-cache >/dev/null 2>&1 ) \
    || { echo "error: bundle failed" >&2; exit 2; }
else
  echo ">> bundle up to date"
fi

pkill -9 -f "[t]arget/release/rayzor" 2>/dev/null || true
bench_quiet_wait
bench_watch_start

echo ">> benching ($REPS reps, $MAX_TOKENS generated tokens each)"
# The [done] line reports tokens / TOTAL seconds, which includes prefill -- on
# a long prompt that is mostly ttft and says nothing about decode. Recover the
# decode-only rate as tokens / (total - ttft) and report both.
ttfts=(); dtps=(); etps=()
for ((i = 1; i <= REPS; i++)); do
  line="$( cd "$APP_DIR" && BUNDLE="$BUNDLE_OUT" BUILD=0 TEMP=0 REP_PENALTY=1.0 \
    MAX_TOKENS="$MAX_TOKENS" CTX="$CTX" PROMPT="$PROMPT" GGUF="$MODEL" \
    timeout 900 ./run_bundle.sh 2>/dev/null | grep -a "^\[done\]" | head -1 )"
  t="$(echo "$line" | grep -ao "ttft=[0-9.]*" | cut -d= -f2)"
  ntok="$(echo "$line" | sed -n 's/^\[done\] \([0-9][0-9]*\) tokens.*/\1/p')"
  total="$(echo "$line" | sed -n 's/^\[done\] [0-9]* tokens in \([0-9.][0-9.]*\)s.*/\1/p')"
  e="$(echo "$line" | grep -aoE "\(([0-9.]+) tok/s" | grep -oE "[0-9.]+")"
  d="$(awk -v n="${ntok:-0}" -v tt="${total:-0}" -v f="${t:-0}" \
        'BEGIN{ g = tt - f; if (g > 0 && n > 0) printf "%.2f", n / g; else print "" }')"
  [[ -n "$t" ]] && ttfts+=("$t")
  [[ -n "$d" ]] && dtps+=("$d")
  [[ -n "$e" ]] && etps+=("$e")
  printf "   rep %d  ttft=%ss  decode=%s tok/s  e2e=%s tok/s\n" "$i" "${t:-NA}" "${d:-NA}" "${e:-NA}"
done

bench_watch_stop

[[ ${#ttfts[@]} -gt 0 ]] || { echo "error: no [done] line parsed from any rep" >&2; exit 2; }

med() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.3f", m}'; }
spread() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{ if(a[1]>0) printf "%.1f", 100*(a[NR]-a[1])/a[1]; else print "NA" }'; }

ttft_med="$(med "${ttfts[@]}")"
dtps_med="$(med "${dtps[@]}")"
etps_med="$(med "${etps[@]}")"
ttft_spread="$(spread "${ttfts[@]}")"
# Prompt tokens are what prefill scales with; report the rate it achieved.
prefill_tps="$(awk -v n="$PROMPT_TOKENS" -v t="$ttft_med" 'BEGIN{ if(t>0) printf "%.1f", n/t; else print "NA" }')"

bench_history_append "$HISTORY" "$HEADER" "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
  "$(date +%Y-%m-%d)" "$BENCH_COMMIT$BENCH_DIRTY" "$ttft_med" "$dtps_med" "$etps_med" "$prefill_tps" \
  "$MAX_TOKENS" "$PROMPT_TOKENS" "$REPS" "$ttft_spread" "$BENCH_QUIET" \
  "${NUE_FLASH_BATCH:-off}" "$(basename "$MODEL")" "$BENCH_SUBJECT")"

echo
echo "  $BENCH_COMMIT$BENCH_DIRTY"
echo "    prefill    ttft=${ttft_med}s  (~${prefill_tps} prompt tok/s over ${PROMPT_TOKENS} tokens)"
echo "    decode     ${dtps_med} tok/s over $MAX_TOKENS tokens (prefill excluded)"
echo "    end-to-end ${etps_med} tok/s including prefill -- falls with prompt length by design"
echo "    stability  ttft spread across $REPS reps = ${ttft_spread}%"
echo "    config     NUE_FLASH_BATCH=${NUE_FLASH_BATCH:-off}"
echo "    machine    quiet=$BENCH_QUIET  (peak foreign ${BENCH_PEAK_CPU}% of one core)"
bench_report "$BENCH_QUIET" ""
echo "  history: $HISTORY"
