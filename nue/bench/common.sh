#!/usr/bin/env bash
# Shared machinery for the nue benches. Source this; do not run it.
#
# Every bench in this directory has the same three obligations, and each one of
# them was learned by getting a wrong number and believing it:
#
#   1. Do not measure on a busy machine. A sibling repo's `cargo test` turned a
#      135.46 +/- 0.25 result into 128.47 +/- 7.18 for the same commit.
#   2. Do not measure a stale artifact, and do not rebuild when nothing changed
#      -- the cargo invocation itself wakes rust-analyzer, whose rustc spike
#      then lands inside the run being measured.
#   3. Record the result with enough context to compare it later. A row without
#      the commit, the model and whether the box was quiet is not evidence.
#
# Provides: bench_quiet_wait, bench_watch_start/stop, bench_stale,
#           bench_build_if_stale, bench_history_append, bench_report.

# --- machine quiet ----------------------------------------------------------
#
# Thresholds are percent of ONE core, measured on the reference Mac:
#   idle desktop (Chrome, VS Code, Slack, agents)   ~164%
#   a sibling `cargo test --workspace`           385-754%
# 137 tok/s was recorded with all those apps open, so gating tighter than the
# desktop floor blocks forever on a box that is perfectly fine to bench on.
BENCH_FOREIGN_BUSY="${BENCH_FOREIGN_BUSY:-300}"
BENCH_COMPILER_BUSY="${BENCH_COMPILER_BUSY:-50}"
BENCH_WAIT_MAX="${BENCH_WAIT_MAX:-900}"
BENCH_SETTLE_S="${BENCH_SETTLE_S:-30}"

# CPU burned by everything that is not the program under test.
# $NF, not $2: a process name can contain spaces ("Google Chrome Helper"),
# which puts the cpu figure in the last field and part of the name in $2.
# "Foreign" means CPU this benchmark did not cause. A head-to-head gate runs
# BOTH engines, so the llama.cpp baseline is as much under test as rayzor is --
# counting it made every comparison row flag itself as contaminated (measured:
# 879% foreign while llama-bench alone held 766%, against a ~150% idle floor).
# The phases are sequential, so neither engine is ever running during the
# other's measurement window.
bench_foreign_cpu() {
  { ps -Ao ucomm=,pcpu= 2>/dev/null || ps -Ao comm=,pcpu= 2>/dev/null; } | awk '
    { name = $1; sub(/.*\//, "", name)
      if (name == "rayzor" || name == "ps" || name == "awk") next
      if (name == "llama-bench" || name == "llama-cli" || name == "llama-server") next
      total += $NF }
    END { printf "%d\n", total + 0 }'
}

bench_compiler_cpu() {
  { ps -Ao ucomm=,pcpu= 2>/dev/null || ps -Ao comm=,pcpu= 2>/dev/null; } | awk '
    { name = $1; sub(/.*\//, "", name)
      if (name == "rustc" || name == "cargo") total += $NF }
    END { printf "%d\n", total + 0 }'
}

# Block until the machine is quiet AND has stayed quiet: a box coming off a
# heavy build is still slow after the load itself is gone.
bench_quiet_wait() {
  local waited=0 quiet_for=0 cpu ccpu
  while :; do
    cpu="$(bench_foreign_cpu)"
    if [[ "$cpu" -lt "$BENCH_FOREIGN_BUSY" ]]; then
      quiet_for=$((quiet_for + 5))
      [[ "$quiet_for" -ge "$BENCH_SETTLE_S" ]] && break
      sleep 5; waited=$((waited + 5)); continue
    fi
    quiet_for=0
    if [[ $waited -eq 0 ]]; then
      ccpu="$(bench_compiler_cpu)"
      echo ">> waiting for a quiet machine (${cpu}% of one core busy, ${ccpu}% of it compilers)"
    fi
    if [[ "$BENCH_WAIT_MAX" -gt 0 && $waited -ge "$BENCH_WAIT_MAX" ]]; then
      echo ">> WARNING: still ${cpu}% busy after ${waited}s -- benching anyway, treat as suspect"
      break
    fi
    sleep 15; waited=$((waited + 15))
  done
}

# Sample foreign CPU for the duration of the run and keep the worst reading.
# Waiting beforehand is not enough: a sibling build can start midway, and did.
BENCH_WATCH_PID=""
BENCH_PEAK_FILE=""
# Peak memory of one engine, in MB. On macOS RSS understates badly -- it
# excludes the compressor, which is exactly where a pressured box puts a model
# -- so prefer phys_footprint via footprint(1), which needs no privileges.
# Largest single matching process, not the newest -- an engine can have more
# than one, and summing would double-count the shared model mapping.
bench_proc_mem_mb() {
  local pids pid mb best=0
  pids="$(pgrep -x "$1" 2>/dev/null)"
  [[ -z "$pids" ]] && { echo 0; return 0; }
  for pid in $pids; do
    mb=""
    if command -v footprint >/dev/null 2>&1; then
      mb="$(footprint -p "$pid" 2>/dev/null | awk '
        /Footprint:/ { for (i = 1; i <= NF; i++) if ($i == "Footprint:") {
            v = $(i+1) + 0; u = $(i+2)
            if (u ~ /^GB/) printf "%d", v * 1024
            else if (u ~ /^MB/) printf "%d", v
            else if (u ~ /^KB/) printf "%d", v / 1024
            exit } }')"
    fi
    [[ -z "$mb" ]] && mb="$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%d", $1/1024}')"
    [[ -n "$mb" && "$mb" -gt "$best" ]] && best="$mb"
  done
  echo "$best"
}

bench_watch_start() {
  # Pressure BEFORE this run starts, so a level the run causes itself can be
  # told apart from a box that was already stressed.
  BENCH_START_PRESSURE="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)"
  BENCH_PEAK_FILE="${TMPDIR:-/tmp}/nue_bench_peak.$$"
  : > "$BENCH_PEAK_FILE"
  # Also watch MEMORY, not just CPU. A run can pass the pre-flight checks and
  # then degrade underneath itself: observed the compressor growing 5.7 -> 7.6 GB
  # and pressure flipping 1 -> 2 while the process under test was still idle at
  # 0.02 GB RSS, after which a 4 GB model never became resident and the run
  # thrashed for 10 minutes. Worst pressure level seen is recorded alongside the
  # CPU peak so the row says WHY it is untrustworthy.
  : > "$BENCH_PEAK_FILE.mem"
  : > "$BENCH_PEAK_FILE.ours"
  : > "$BENCH_PEAK_FILE.cpp"
  (
    peak=0
    worst=1
    mours=0
    mcpp=0
    while :; do
      c="$(bench_foreign_cpu)"
      [[ "$c" -gt "$peak" ]] && { peak="$c"; printf '%s\n' "$peak" > "$BENCH_PEAK_FILE"; }
      p="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)"
      [[ "$p" -gt "$worst" ]] && { worst="$p"; printf '%s\n' "$worst" > "$BENCH_PEAK_FILE.mem"; }
      # Each engine's own peak, so the row says what the model actually cost
      # rather than only what the box was doing around it.
      m="$(bench_proc_mem_mb rayzor)"
      [[ "$m" -gt "$mours" ]] && { mours="$m"; printf '%s\n' "$mours" > "$BENCH_PEAK_FILE.ours"; }
      m="$(bench_proc_mem_mb llama-bench)"
      [[ "$m" -gt "$mcpp" ]] && { mcpp="$m"; printf '%s\n' "$mcpp" > "$BENCH_PEAK_FILE.cpp"; }
      sleep 2
    done
  ) &
  BENCH_WATCH_PID=$!
}

# Sets BENCH_PEAK_CPU and BENCH_QUIET ("yes" or "NO(<peak>%)").
bench_watch_stop() {
  [[ -n "$BENCH_WATCH_PID" ]] && { kill "$BENCH_WATCH_PID" 2>/dev/null; wait "$BENCH_WATCH_PID" 2>/dev/null; }
  BENCH_PEAK_CPU="$(cat "$BENCH_PEAK_FILE" 2>/dev/null || echo 0)"
  [[ -z "$BENCH_PEAK_CPU" ]] && BENCH_PEAK_CPU=0
  rm -f "$BENCH_PEAK_FILE"
  BENCH_WORST_PRESSURE="$(cat "$BENCH_PEAK_FILE.mem" 2>/dev/null || echo 1)"
  [[ -z "$BENCH_WORST_PRESSURE" ]] && BENCH_WORST_PRESSURE=1
  rm -f "$BENCH_PEAK_FILE.mem"
  BENCH_PEAK_MEM_OURS="$(cat "$BENCH_PEAK_FILE.ours" 2>/dev/null || echo 0)"
  BENCH_PEAK_MEM_CPP="$(cat "$BENCH_PEAK_FILE.cpp" 2>/dev/null || echo 0)"
  [[ -z "$BENCH_PEAK_MEM_OURS" ]] && BENCH_PEAK_MEM_OURS=0
  [[ -z "$BENCH_PEAK_MEM_CPP" ]] && BENCH_PEAK_MEM_CPP=0
  rm -f "$BENCH_PEAK_FILE.ours" "$BENCH_PEAK_FILE.cpp"
  BENCH_QUIET="yes"
  [[ "$BENCH_PEAK_CPU" -ge "$BENCH_FOREIGN_BUSY" ]] && BENCH_QUIET="NO(${BENCH_PEAK_CPU}%)"
  # Memory degradation invalidates a throughput row just as surely as CPU
  # contention -- a model that cannot stay resident measures the SSD. But the
  # run causes some of it: a 7B wires ~4GB, which drives a 16GB box to WARN by
  # itself, and flagging that would mark every 7B row contaminated. So flag
  # WARN only when the box was ALREADY stressed before the run (the pressure is
  # then not ours), and flag CRITICAL unconditionally.
  local mem_bad=0
  [[ "$BENCH_WORST_PRESSURE" -ge 4 ]] && mem_bad=1
  [[ "${BENCH_START_PRESSURE:-1}" -gt 1 ]] && mem_bad=1
  if [[ "$mem_bad" -eq 1 ]]; then
    if [[ "$BENCH_QUIET" == "yes" ]]; then
      BENCH_QUIET="NO(mem-pressure-${BENCH_WORST_PRESSURE})"
    else
      BENCH_QUIET="${BENCH_QUIET}+mem${BENCH_WORST_PRESSURE}"
    fi
  fi
}

# --- memory pressure --------------------------------------------------------
#
# CPU quiet is NOT enough. macOS charges phys_footprint (not RSS), COMPRESSES
# anonymous pages before swapping, and evicts off a pressure LEVEL. A box can
# report "45% free" while sitting at WARN with 6 GB of RAM held by the
# compressor and 12 GB in swap -- which is exactly the state every measurement
# in this session was taken under, because the gate only watched CPU.
#
#   kern.memorystatus_vm_pressure_level: 1 = normal, 2 = warn, 4 = critical
BENCH_REQUIRE_PRESSURE="${BENCH_REQUIRE_PRESSURE:-1}"   # max acceptable level

bench_pressure_level() {
  sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1
}

# RAM held by the compressor, in MB. Invisible to RSS and to "free percentage".
bench_compressor_mb() {
  local pages psz
  pages="$(vm_stat 2>/dev/null | awk '/Pages occupied by compressor/ {gsub(/\./,"",$NF); print $NF}')"
  psz="$(pagesize 2>/dev/null || echo 4096)"
  [[ -z "$pages" ]] && { echo 0; return; }
  awk -v p="$pages" -v s="$psz" 'BEGIN { printf "%d", p * s / 1048576 }'
}

bench_swap_used_mb() {
  sysctl -n vm.swapusage 2>/dev/null | awk '{ for (i=1;i<=NF;i++) if ($i=="used") { gsub(/M/,"",$(i+2)); printf "%d", $(i+2); exit } }'
}

# Sets BENCH_PRESSURE, BENCH_COMPRESSOR_MB, BENCH_SWAP_MB and returns non-zero
# when the machine is above the acceptable pressure level.
bench_memory_state() {
  BENCH_PRESSURE="$(bench_pressure_level)"
  BENCH_COMPRESSOR_MB="$(bench_compressor_mb)"
  BENCH_SWAP_MB="$(bench_swap_used_mb)"
  [[ "${BENCH_PRESSURE:-1}" -le "$BENCH_REQUIRE_PRESSURE" ]]
}

# --- artifact staleness -----------------------------------------------------
#
# True when any source is newer than the artifact, or the artifact is missing.
# -print -quit stops at the first hit, so this stays cheap over a whole tree.
bench_stale() {
  local artifact="$1"; shift
  [[ -f "$artifact" ]] || return 0
  local hit
  hit="$(find "$@" -type f -newer "$artifact" -not -path "*/.rayzor/*" -print -quit 2>/dev/null)"
  [[ -n "$hit" ]]
}

# Build host and plugins only when their OWN sources moved. Each artifact needs
# its own list: cargo rightly never rebuilds rayzor-tensors when the compiler
# changes, so one shared list leaves that dylib permanently "stale" and
# rebuilds everything on every run.
#
# The host and the plugins must be built in SEPARATE cargo invocations --
# `--bin rayzor` filters targets, so naming the plugin packages in the same
# command builds nothing for them and they silently stay at the previous
# commit, then fail the ABI handshake.
bench_build_if_stale() {
  local repo="$1" force="${2:-0}"
  local ext=dylib
  [[ "$(uname -s)" != "Darwin" ]] && ext=so
  local bin="$repo/target/release/rayzor"
  local tensors="$repo/target/release/librayzor_tensors.$ext"
  local plugins="$repo/target/release/libnue_plugins.$ext"

  local host_src=(
    "$repo/compiler" "$repo/parser" "$repo/diagnostics" "$repo/source_map"
    "$repo/runtime" "$repo/runtime-core" "$repo/plugin" "$repo/src"
    "$repo/Cargo.toml" "$repo/Cargo.lock"
  )
  local tensors_src=( "$repo/rayzor-tensors" "$repo/plugin" "$repo/Cargo.toml" "$repo/Cargo.lock" )
  local plugins_src=( "$repo/nue-plugins" "$repo/rayzor-tensors" "$repo/plugin" "$repo/Cargo.toml" "$repo/Cargo.lock" )

  local did=0
  if [[ "$force" == "1" ]] || bench_stale "$bin" "${host_src[@]}"; then
    echo ">> building host"
    CARGO_INCREMENTAL=0 cargo build --release --manifest-path "$repo/Cargo.toml" -p rayzor --bin rayzor >/dev/null 2>&1 \
      || { echo "error: host build failed" >&2; return 2; }
    did=1
  fi
  if [[ "$force" == "1" ]] || bench_stale "$tensors" "${tensors_src[@]}" || bench_stale "$plugins" "${plugins_src[@]}"; then
    echo ">> building plugins"
    CARGO_INCREMENTAL=0 cargo build --release --manifest-path "$repo/Cargo.toml" -p rayzor-tensors -p nue-plugins >/dev/null 2>&1 \
      || { echo "error: plugin build failed" >&2; return 2; }
    did=1
  fi
  [[ $did -eq 0 ]] && echo ">> host and plugins up to date (--force-rebuild to rebuild)"
  return 0
}

# --- provenance and reporting ----------------------------------------------

# Sets BENCH_COMMIT, BENCH_SUBJECT, BENCH_DIRTY.
bench_provenance() {
  local repo="$1"
  BENCH_COMMIT="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  BENCH_SUBJECT="$(git -C "$repo" log -1 --format=%s 2>/dev/null | cut -c1-60)"
  BENCH_DIRTY=""
  [[ -n "$(git -C "$repo" status --porcelain -- "$repo/compiler" "$repo/rayzor-tensors" "$repo/nue" 2>/dev/null)" ]] \
    && BENCH_DIRTY="+dirty"
}

# bench_history_append <file> <tab-separated header> <tab-separated row>
bench_history_append() {
  local file="$1" header="$2" row="$3"
  [[ -f "$file" ]] || printf '%s\n' "$header" > "$file"
  printf '%s\n' "$row" >> "$file"
}

# bench_report <quiet> <stddev-or-empty> -- prints the trust verdict.
bench_report() {
  local quiet="$1" sd="${2:-}"
  if [[ "$quiet" != "yes" ]]; then
    # Name the condition that actually fired -- reporting a CPU peak that is
    # under the threshold sends people hunting a process that was never there.
    if [[ "$quiet" == *mem* ]]; then
      echo "  CONTAMINATED: memory pressure reached level ${BENCH_WORST_PRESSURE} during this run"
      echo "                (started at ${BENCH_START_PRESSURE:-1}; foreign CPU peaked ${BENCH_PEAK_CPU}%)."
      echo "                A model that cannot stay resident measures the SSD."
    fi
    if [[ "$BENCH_PEAK_CPU" -ge "$BENCH_FOREIGN_BUSY" ]]; then
      echo "  CONTAMINATED: other processes hit ${BENCH_PEAK_CPU}% of a core DURING this run."
      echo "                Not comparable to a quiet-machine result."
    fi
  elif [[ -n "$sd" ]] && awk "BEGIN{exit !($sd > 1.0)}"; then
    echo "  WARNING: stddev $sd > 1.0 with the machine apparently quiet. Check whether"
    echo "           it had just come off a heavy build -- load takes time to settle."
  fi
}
