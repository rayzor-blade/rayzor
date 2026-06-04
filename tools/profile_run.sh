#!/bin/sh
# tools/profile_run.sh — one-command profile of a rayzor program
#
# Scrubs caches, runs `rayzor run <file> [args]` with the full
# diagnostic env-var stack, then resolves PCs to Haxe file:line via
# tools/resolve_alloc_graph.py.
#
# Requires a profile-enabled build:
#   cargo build --release --features profile
#
# Usage:
#   tools/profile_run.sh nue/examples/llama-chat/Main.hx -- <gguf> "prompt" 60 4096 0.01
#
# Knobs (env vars):
#   RAYZOR_BIN              — path to the rayzor binary
#                             (default: target/release/rayzor)
#   PROFILE_DELAY_MS=2000   — skip first N ms (compile phase)
#   PROFILE_PERIOD_US=500   — SIGPROF interval in microseconds
#   PROFILE_DURATION_MS=0   — auto-disarm after N ms (0 = forever)
#   PROFILE_GRAPH_RATE=0    — also enable alloc-graph at 1-in-N
#                             sampling (0 = off; only CPU profile)
#   PROFILE_NO_SCRUB=1      — keep .rayzor caches across runs

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAYZOR_BIN="${RAYZOR_BIN:-$REPO/target/release/rayzor}"
DELAY_MS="${PROFILE_DELAY_MS:-2000}"
PERIOD_US="${PROFILE_PERIOD_US:-500}"
DURATION_MS="${PROFILE_DURATION_MS:-0}"
GRAPH_RATE="${PROFILE_GRAPH_RATE:-0}"

if [ ! -x "$RAYZOR_BIN" ]; then
  echo "error: rayzor binary not found at $RAYZOR_BIN" >&2
  echo "       build with: cargo build --release --features profile" >&2
  exit 2
fi

# Sanity: confirm profile build by checking for [cpu-profile] symbol
if ! nm "$RAYZOR_BIN" 2>/dev/null | grep -q 'rayzor_profile_start'; then
  echo "warning: $RAYZOR_BIN does not appear to be a profile-feature build" >&2
  echo "         (rayzor_profile_start symbol not found)" >&2
  echo "         rebuild with: cargo build --release --features profile" >&2
fi

# Scrub .rayzor BLADE caches + previous dump artifacts
if [ "${PROFILE_NO_SCRUB:-0}" != "1" ]; then
  find "$REPO" -name .rayzor -type d -exec rm -rf {} + 2>/dev/null || true
fi
rm -f /tmp/rayzor_jit_symbols.csv \
      /tmp/rayzor_file_table.csv \
      /tmp/rayzor_alloc_graph.csv \
      /tmp/rayzor_alloc_attributed.csv

# Build env-var stack
EXTRA_ENV=""
if [ "$GRAPH_RATE" -gt 0 ]; then
  EXTRA_ENV="RAYZOR_ALLOC_GRAPH=1 RAYZOR_ALLOC_GRAPH_RATE=$GRAPH_RATE"
fi

echo "[profile_run] DELAY_MS=$DELAY_MS PERIOD_US=$PERIOD_US DURATION_MS=$DURATION_MS GRAPH_RATE=$GRAPH_RATE"
echo "[profile_run] running: $RAYZOR_BIN run $*"
echo ""

env \
  RAYZOR_DUMP_JIT_MAP=1 \
  RAYZOR_DUMP_ALLOC_AT_EXIT=1 \
  RAYZOR_CPU_PROFILE=1 \
  RAYZOR_CPU_PROFILE_US="$PERIOD_US" \
  RAYZOR_CPU_PROFILE_DELAY_MS="$DELAY_MS" \
  RAYZOR_CPU_PROFILE_DURATION_MS="$DURATION_MS" \
  $EXTRA_ENV \
  "$RAYZOR_BIN" run "$@"

RUN_EXIT=$?
echo ""

if [ ! -f /tmp/rayzor_alloc_graph.csv ]; then
  echo "[profile_run] /tmp/rayzor_alloc_graph.csv not written (run failed or exited without dump)" >&2
  exit $RUN_EXIT
fi

python3 "$REPO/tools/resolve_alloc_graph.py" "$RAYZOR_BIN"

exit $RUN_EXIT
