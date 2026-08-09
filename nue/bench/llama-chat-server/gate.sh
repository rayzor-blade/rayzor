#!/usr/bin/env bash
# Serving bench for nue/examples/llama-chat-server, gated so the number is
# trustworthy and recorded so it can be compared later.
#
#   ./gate.sh                    # measure HEAD, append a row
#   ./gate.sh --show             # print the history and exit
#   ./gate.sh --force-rebuild    # rebuild even if nothing changed
#   RUNS=5 ./gate.sh             # tighter median
#   GGUF=/path/model.gguf ./gate.sh
#
# Run this after any commit touching a kernel, MIR lowering, the tiered
# backend or the tensor runtime. Twelve commits once landed in a single session
# with no throughput record; when a slowdown was reported there was no baseline
# to bisect against, and reconstructing one cost a day. The answer turned out
# to be that no regression existed.
#
# Reading the report: `quiet` is the trust signal, not stddev. stddev tells you
# something was competing; quiet tells you what and when. A row with
# quiet=NO(...) is not comparable to a clean one and should not be averaged
# with it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="$REPO/nue/examples/llama-chat-server"
HISTORY="${HISTORY:-$SCRIPT_DIR/history.tsv}"

HEADER=$'date\tcommit\ttok_s\tstddev\tmin\tmax\tttft_s\tready_s\tcold_ttft_s\tpeak_rss_mib\tquiet\truns\treq\tctx\tmax_tok\tmodel\tsubject'

FORCE_REBUILD="${FORCE_REBUILD:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      [[ -f "$HISTORY" ]] && column -t -s $'\t' "$HISTORY" || echo "no history yet: $HISTORY"
      exit 0 ;;
    --force-rebuild|-f) FORCE_REBUILD=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

MODEL="${GGUF:-$HOME/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf}"
RUNS="${RUNS:-3}"
REQUESTS="${REQUESTS:-3}"
CTX="${CTX:-4096}"
MAX_TOKENS="${MAX_TOKENS:-808}"
BUNDLE_OUT="${BUNDLE_OUT:-/tmp/nue_bench_server.rzb}"

[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL (set GGUF=)" >&2; exit 2; }

bench_provenance "$REPO"
echo ">> $BENCH_COMMIT$BENCH_DIRTY  $(basename "$MODEL")"
bench_build_if_stale "$REPO" "$FORCE_REBUILD" || exit 2

# A .rzb embeds MIR, so an artifact older than the compiler is a different
# program. Plugins are excluded -- they load at runtime via --native-lib.
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

echo ">> benching ($RUNS runs x $REQUESTS requests)"
out="$( cd "$APP_DIR" && MEMORY_PROFILE=1 BUILD=0 BUNDLE="$BUNDLE_OUT" RUNS="$RUNS" \
  REQUESTS="$REQUESTS" CTX="$CTX" MAX_TOKENS="$MAX_TOKENS" TEMP=0 \
  RZT_AMX_PREFILL=1 NUE_MATMUL=1 NUE_FLASH=1 STREAM=0 DECODE_PROFILE=false \
  NUE_PREFILL_WARM=1 NUE_REQUANT_LM_HEAD=1 NUE_HAXE_Q8_0=1 NUE_HAXE_INT8=1 \
  GGUF="$MODEL" timeout 1800 ./bench_server.sh 2>&1 )"

bench_watch_stop

pick() { echo "$out" | grep -ao "$1=[0-9.]*" | head -1 | cut -d= -f2; }
med="$(pick median)"; sd="$(pick stddev)"; mn="$(pick min)"; mx="$(pick max)"
ttft="$(echo "$out" | grep -ao "first_byte_med=[0-9.]*" | head -1 | cut -d= -f2)"
ready="$(echo "$out" | grep -ao "ready_med=[0-9.]*" | head -1 | cut -d= -f2)"
cold="$(echo "$out" | grep -ao "cold_ttft_med=[0-9.]*" | head -1 | cut -d= -f2)"
rss="$(echo "$out" | grep -ao "peak_rss=[0-9]*" | sort -t= -k2 -n | tail -1 | cut -d= -f2)"

if [[ -z "$med" ]]; then
  echo "error: no median in bench output; last lines:" >&2
  echo "$out" | tail -20 >&2
  exit 2
fi

bench_history_append "$HISTORY" "$HEADER" "$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
  "$(date +%Y-%m-%d)" "$BENCH_COMMIT$BENCH_DIRTY" "$med" "${sd:-NA}" "${mn:-NA}" "${mx:-NA}" \
  "${ttft:-NA}" "${ready:-NA}" "${cold:-NA}" "${rss:-NA}" "$BENCH_QUIET" \
  "$RUNS" "$REQUESTS" "$CTX" "$MAX_TOKENS" "$(basename "$MODEL")" "$BENCH_SUBJECT")"

echo
echo "  $BENCH_COMMIT$BENCH_DIRTY"
echo "    tok/s      median=$med  stddev=${sd:-NA}  min=${mn:-NA}  max=${mx:-NA}"
echo "    latency    ttft=${ttft:-NA}s  ready=${ready:-NA}s  cold_ttft=${cold:-NA}s"
echo "    memory     peak_rss=${rss:-NA}MiB"
echo "    machine    quiet=$BENCH_QUIET  (peak foreign ${BENCH_PEAK_CPU}% of one core)"
bench_report "$BENCH_QUIET" "$sd"
echo "  history: $HISTORY"
