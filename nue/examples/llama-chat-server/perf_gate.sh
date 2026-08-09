#!/usr/bin/env bash
# Record serving throughput and peak RSS for the current checkout, one row per
# run, appended to perf_history.tsv.
#
# Why this exists: twelve commits landed in one session with no per-commit
# throughput record. When a slowdown was reported there was no baseline to
# bisect against, and reconstructing one cost a day. Run this after any commit
# that touches a kernel, MIR lowering, the tiered backend or the tensor runtime.
#
#   ./perf_gate.sh                 # measure HEAD, append a row
#   ./perf_gate.sh --show          # print the history and exit
#   RUNS=5 ./perf_gate.sh          # tighter median
#   ./perf_gate.sh --force-rebuild # rebuild even if nothing changed
#
# The build and bundle are skipped when no source is newer than the artifacts,
# so a repeat run starts benching immediately. This is not only for speed: the
# cargo invocation itself wakes rust-analyzer, and the resulting rustc spike
# lands inside the very run being measured.
#
# Reading the output:
#   stddev is the trust signal. On a quiet machine this bench resolves to
#   ~0.25 tok/s. Anything above ~1 means something else was running -- find it
#   and re-run rather than averaging it away. The gate refuses to append a row
#   while a compiler is active, because that alone has turned a 135.46 +/- 0.25
#   measurement into 128.47 +/- 7.18 for the very same commit.
#
#   RSS is deterministic and nearly free, so it is the tight gate; tok/s is the
#   loose one. Note peak RSS includes the mmap'd model file (evictable), so
#   compare like-for-like models, and expect the curve to rise then fall.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HISTORY="${HISTORY:-$SCRIPT_DIR/perf_history.tsv}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --show)
      [[ -f "$HISTORY" ]] && column -t -s $'\t' "$HISTORY" || echo "no history yet: $HISTORY"
      exit 0
      ;;
    --force-rebuild|-f) FORCE_REBUILD=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${GGUF:-$HOME/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf}"
RUNS="${RUNS:-3}"
BUNDLE_OUT="${BUNDLE_OUT:-/tmp/perf_gate.rzb}"
TMP_PEAK="${TMPDIR:-/tmp}/perf_gate_peak.$$"

if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL (set GGUF=)" >&2
  exit 2
fi

commit="$(git -C "$REPO" rev-parse --short HEAD)"
subject="$(git -C "$REPO" log -1 --format=%s | cut -c1-60)"
dirty=""
[[ -n "$(git -C "$REPO" status --porcelain -- "$REPO/compiler" "$REPO/rayzor-tensors" "$REPO/nue" 2>/dev/null)" ]] && dirty="+dirty"

RAYZOR_BIN="$REPO/target/release/rayzor"
case "$(uname -s)" in
  Darwin) DYLIB_EXT="dylib" ;;
  *) DYLIB_EXT="so" ;;
esac
TENSORS_LIB="$REPO/target/release/librayzor_tensors.$DYLIB_EXT"
PLUGINS_LIB="$REPO/target/release/libnue_plugins.$DYLIB_EXT"

# Each artifact gets the sources it is ACTUALLY built from. One shared list
# does not work: cargo rightly never rebuilds rayzor-tensors when the compiler
# changes, so the dylib's mtime never advances past compiler/src and a combined
# list marks it stale on every single run -- which rebuilds everything, always.
HOST_SRC=(
  "$REPO/compiler" "$REPO/parser" "$REPO/diagnostics" "$REPO/source_map"
  "$REPO/runtime" "$REPO/runtime-core" "$REPO/plugin" "$REPO/src"
  "$REPO/Cargo.toml" "$REPO/Cargo.lock"
)
TENSORS_SRC=( "$REPO/rayzor-tensors" "$REPO/plugin" "$REPO/Cargo.toml" "$REPO/Cargo.lock" )
PLUGINS_SRC=( "$REPO/nue-plugins" "$REPO/rayzor-tensors" "$REPO/plugin" "$REPO/Cargo.toml" "$REPO/Cargo.lock" )
HAXE_SRC=( "$SCRIPT_DIR" "$REPO/nue/nue" "$REPO/compiler/haxe-std" )

# True when any source is newer than the artifact, or the artifact is missing.
# -print -quit stops at the first hit, so this stays cheap over the whole tree.
stale() {
  local artifact="$1"; shift
  [[ -f "$artifact" ]] || return 0
  local hit
  hit="$(find "$@" -type f -newer "$artifact" -not -path "*/.rayzor/*" -print -quit 2>/dev/null)"
  [[ -n "$hit" ]]
}

# The host and the plugins must be built in SEPARATE invocations: `--bin rayzor`
# filters targets, so naming the plugin packages in the same command builds
# nothing for them and they silently stay at whatever commit last built them,
# which then fails the ABI handshake.
built_something=0
if [[ "$FORCE_REBUILD" == "1" ]] || stale "$RAYZOR_BIN" "${HOST_SRC[@]}"; then
  echo ">> building host at $commit$dirty"
  CARGO_INCREMENTAL=0 cargo build --release --manifest-path "$REPO/Cargo.toml" -p rayzor --bin rayzor >/dev/null 2>&1 \
    || { echo "error: host build failed" >&2; exit 2; }
  built_something=1
fi
if [[ "$FORCE_REBUILD" == "1" ]] || stale "$TENSORS_LIB" "${TENSORS_SRC[@]}" || stale "$PLUGINS_LIB" "${PLUGINS_SRC[@]}"; then
  echo ">> building plugins at $commit$dirty"
  CARGO_INCREMENTAL=0 cargo build --release --manifest-path "$REPO/Cargo.toml" -p rayzor-tensors -p nue-plugins >/dev/null 2>&1 \
    || { echo "error: plugin build failed" >&2; exit 2; }
  built_something=1
fi
[[ "$built_something" -eq 0 ]] && echo ">> host and plugins up to date at $commit$dirty (--force-rebuild to rebuild)"

# A .rzb embeds MIR, so an artifact older than the compiler is a different
# program and may not even load -- the binary is a dependency of the bundle,
# not just the Haxe sources.
if [[ "$FORCE_REBUILD" == "1" ]] || stale "$BUNDLE_OUT" "${HAXE_SRC[@]}" "$RAYZOR_BIN"; then
  echo ">> bundling"
  "$RAYZOR_BIN" bundle Main.hx -o "$BUNDLE_OUT" --no-cache >/dev/null 2>&1 \
    || { echo "error: bundle failed" >&2; exit 2; }
else
  echo ">> bundle up to date ($BUNDLE_OUT)"
fi

pkill -9 -f "[t]arget/release/rayzor" 2>/dev/null || true

# Wait for compilers that are actually burning CPU, anywhere on the machine --
# a `cargo test --workspace` from an unrelated repo is what corrupts the number.
#
# Measured by CPU, not by process existence, for two reasons: matching command
# lines (pgrep -f) also hits the wrapper shells whose arguments merely contain
# "cargo", so a lingering shell blocks forever on a phantom; and rust-analyzer
# runs cargo check continuously in an open editor without meaningfully
# competing for cores. Load average is useless here -- a desktop with a browser
# never reaches load < 2.
# Thresholds are percent of ONE core, measured on this machine:
#   idle desktop (Chrome, VS Code, Slack, CleanMyMac, agents)  ~164%
#   a sibling `cargo test --workspace`                     385-754%
# 137 tok/s was recorded with all those apps open, so the gate must sit above
# the desktop floor -- anything tighter blocks forever on a box that is
# perfectly fine to bench on, which is the failure this gate keeps hitting.
FOREIGN_CPU_BUSY="${FOREIGN_CPU_BUSY:-300}"
COMPILER_CPU_BUSY="${COMPILER_CPU_BUSY:-50}"
COMPILER_WAIT_MAX="${COMPILER_WAIT_MAX:-900}"  # seconds before giving up

# CPU burned by everything that is NOT this benchmark. Watching only cargo and
# rustc proved too narrow: a run taken with 0% compiler CPU still came in 20
# tok/s low because the machine had been hammered for the preceding fifteen
# minutes (load average 12.77) and had not settled. `rayzor` is excluded
# because the process under test is supposed to be busy.
foreign_cpu() {
  # $NF, not $2: a process name can contain spaces ("Google Chrome Helper"),
  # which puts the cpu figure in the last field and a fragment of the name in
  # the second.
  { ps -Ao ucomm=,pcpu= 2>/dev/null || ps -Ao comm=,pcpu= 2>/dev/null; } | awk '
    { name = $1; sub(/.*\//, "", name)
      if (name == "rayzor" || name == "ps" || name == "awk") next
      total += $NF }
    END { printf "%d\n", total + 0 }'
}

# Kept for the message below: which part of the foreign load is a compiler.
compiler_cpu() {
  { ps -Ao ucomm=,pcpu= 2>/dev/null || ps -Ao comm=,pcpu= 2>/dev/null; } | awk '
    { name = $1; sub(/.*\//, "", name)
      if (name == "rustc" || name == "cargo") total += $NF }
    END { printf "%d\n", total + 0 }'
}

# The machine must be quiet AND have stayed quiet: a box coming off a heavy
# build is still slow for a while after the load itself is gone.
SETTLE_S="${SETTLE_S:-30}"
waited=0
quiet_for=0
while :; do
  cpu="$(foreign_cpu)"
  if [[ "$cpu" -lt "$FOREIGN_CPU_BUSY" ]]; then
    quiet_for=$((quiet_for + 5))
    [[ "$quiet_for" -ge "$SETTLE_S" ]] && break
    sleep 5
    waited=$((waited + 5))
    continue
  fi
  quiet_for=0
  if [[ $waited -eq 0 ]]; then
    ccpu="$(compiler_cpu)"
    echo ">> waiting for a quiet machine (${cpu}% of one core busy, ${ccpu}% of it compilers; SETTLE_S=$SETTLE_S, FOREIGN_CPU_BUSY=$FOREIGN_CPU_BUSY)"
  fi
  if [[ "$COMPILER_WAIT_MAX" -gt 0 && $waited -ge "$COMPILER_WAIT_MAX" ]]; then
    echo ">> WARNING: still ${cpu}% busy after ${waited}s -- benching anyway, treat the result as suspect"
    break
  fi
  sleep 15
  waited=$((waited + 15))
done

# Waiting for a quiet machine before the run is not enough: a sibling repo can
# start a test suite midway through, which is how a 5-run bench came back at
# 112.52 +/- 12.61 on a box that was idle when it started. Sample foreign CPU
# throughout and record the worst moment, so contamination is detected rather
# than inferred from a wide stddev after the fact.
peak_cpu_file="$TMP_PEAK"
: > "$peak_cpu_file"
(
  peak=0
  while :; do
    c="$(foreign_cpu)"
    [[ "$c" -gt "$peak" ]] && { peak="$c"; printf '%s\n' "$peak" > "$peak_cpu_file"; }
    sleep 5
  done
) &
cpu_watch_pid=$!

echo ">> benching ($RUNS runs)"
out="$(MEMORY_PROFILE=1 BUILD=0 BUNDLE="$BUNDLE_OUT" RUNS="$RUNS" TEMP=0 \
  RZT_AMX_PREFILL=1 NUE_MATMUL=1 NUE_FLASH=1 STREAM=0 DECODE_PROFILE=false \
  NUE_PREFILL_WARM=1 NUE_REQUANT_LM_HEAD=1 NUE_HAXE_Q8_0=1 NUE_HAXE_INT8=1 \
  GGUF="$MODEL" timeout 1800 ./bench_server.sh 2>&1)"

kill "$cpu_watch_pid" 2>/dev/null || true
wait "$cpu_watch_pid" 2>/dev/null || true
peak_cpu="$(cat "$peak_cpu_file" 2>/dev/null || echo 0)"
[[ -z "$peak_cpu" ]] && peak_cpu=0
rm -f "$peak_cpu_file"

med="$(echo "$out" | grep -ao "median=[0-9.]*" | head -1 | cut -d= -f2)"
sd="$(echo "$out" | grep -ao "stddev=[0-9.]*" | head -1 | cut -d= -f2)"
rss="$(echo "$out" | grep -ao "peak_rss=[0-9]*" | sort -t= -k2 -n | tail -1 | cut -d= -f2)"

if [[ -z "$med" ]]; then
  echo "error: no median in bench output; last lines:" >&2
  echo "$out" | tail -20 >&2
  exit 2
fi

# quiet= the worst compiler CPU seen DURING the run; a row with quiet>threshold
# is a contaminated sample and must not be compared against a clean one.
[[ -f "$HISTORY" ]] || printf "date\tcommit\ttok_s\tstddev\tpeak_rss_mib\tquiet\tmodel\tsubject\n" > "$HISTORY"
quiet_flag="yes"
[[ "$peak_cpu" -ge "$FOREIGN_CPU_BUSY" ]] && quiet_flag="NO(${peak_cpu}%)"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date +%Y-%m-%d)" "$commit$dirty" "$med" "${sd:-NA}" "${rss:-NA}" "$quiet_flag" \
  "$(basename "$MODEL")" "$subject" >> "$HISTORY"

echo
echo "  $commit$dirty  tok/s=$med  stddev=${sd:-NA}  peak_rss=${rss:-NA}MiB  quiet=$quiet_flag"
if [[ "$quiet_flag" != "yes" ]]; then
  echo "  CONTAMINATED: other processes hit ${peak_cpu}% of a core DURING this run."
  echo "                The number is not comparable to a quiet-machine result."
elif [[ -n "$sd" ]] && awk "BEGIN{exit !($sd > 1.0)}"; then
  echo "  WARNING: stddev $sd > 1.0 with the machine apparently quiet. Check whether"
  echo "           it had just come off a heavy build -- load takes time to settle."
fi
echo "  history: $HISTORY"
