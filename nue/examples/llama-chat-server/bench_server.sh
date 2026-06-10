#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="${RAYZOR:-../../../target/release/rayzor}"
MODEL_PATH="${RAYZOR_SERVER_MODEL:-${GGUF:-/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf}}"
PROMPTS="${PROMPTS:-Explain voronoi regions and their connection to delaunay computation.|||Describe vector graph database implementation.|||Give a compact Haxe example of an actor worker.}"
REQUESTS="${REQUESTS:-3}"
MAX_TOKENS="${MAX_TOKENS:-808}"
CTX="${CTX:-4096}"
TEMP="${TEMP:-0.0}"
RUNS="${RUNS:-6}"
TIMEOUT="${TIMEOUT:-360}"
COOLDOWN_MS="${COOLDOWN_MS:-15000}"
START_INTERPRETED="${START_INTERPRETED:-false}"
TIER_PROMOTION="${TIER_PROMOTION:-false}"
VARIANTS="${VARIANTS:-1/30/5}"
DECODE_PROFILE="${DECODE_PROFILE:-false}"
COLD_PROFILE="${COLD_PROFILE:-true}"
REQUEST_PROFILE="${REQUEST_PROFILE:-true}"
PREFILL_MORSELS="${PREFILL_MORSELS:-${RAYZOR_PREFILL_MORSELS:-false}}"
STREAM="${STREAM:-${RAYZOR_SERVER_STREAM:-false}}"
PRESET="${PRESET:-application}"
PORT_BASE="${PORT_BASE:-19890}"
NO_CACHE="${NO_CACHE:-true}"

if [[ -n "$VARIANTS" ]]; then
  read -r -a VARIANT_LIST <<< "$VARIANTS"
else
  VARIANT_LIST=("toml")
fi

TOTAL_RUNS=$((RUNS * ${#VARIANT_LIST[@]}))
TMP_DIR="${TMPDIR:-/tmp}/rayzor-llama-chat-server-bench.$$"
RESULTS="$TMP_DIR/results.tsv"
COLD_RESULTS="$TMP_DIR/cold.tsv"
mkdir -p "$TMP_DIR"
: > "$RESULTS"
: > "$COLD_RESULTS"

CURRENT_PID=""
cleanup() {
  if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill "$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sleep_ms() {
  local ms="$1"
  /usr/bin/python3 - "$ms" <<'PY'
import sys, time
time.sleep(int(sys.argv[1]) / 1000.0)
PY
}

now_s() {
  /usr/bin/python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
}

elapsed_s() {
  local start="$1"
  local end="$2"
  /usr/bin/python3 - "$start" "$end" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
}

wait_for_server() {
  local log="$1"
  local pid="$2"
  local deadline=$((SECONDS + TIMEOUT))
  while true; do
    if grep -q "\[server\] listening on" "$log" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep_ms 100
  done
}

send_requests() {
  local port="$1"
  PROMPTS="$PROMPTS" REQUESTS="$REQUESTS" PORT="$port" /usr/bin/python3 <<'PY'
import os, socket, sys, time

prompts = [p for p in os.environ.get("PROMPTS", "").split("|||") if p]
if not prompts:
    prompts = ["Hello"]
requests = int(os.environ["REQUESTS"])
port = int(os.environ["PORT"])

for i in range(requests):
    payload = prompts[i % len(prompts)].encode("utf-8")
    started = time.perf_counter()
    first_byte_s = None
    nbytes = 0
    with socket.create_connection(("127.0.0.1", port), timeout=30.0) as sock:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            if first_byte_s is None:
                first_byte_s = time.perf_counter() - started
            nbytes += len(chunk)
    elapsed = time.perf_counter() - started
    first = first_byte_s if first_byte_s is not None else elapsed
    print(
        f"[client-request] id={i + 1} seconds={elapsed:.6f} "
        f"first_byte_s={first:.6f} bytes={nbytes}",
        flush=True,
    )
PY
}

extract_metric() {
  local log="$1"
  /usr/bin/python3 - "$log" <<'PY'
import math, re, sys

patterns = [
    re.compile(r"\[done\].*\(([0-9]+(?:\.[0-9]+)?)\s+tok/s\)"),
    re.compile(r"\[server-done\].*\btok/s=([0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"\[request-done\].*\btok/s=([0-9]+(?:\.[0-9]+)?)"),
]

try:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    lines = []

for line in reversed(lines):
    for pattern in patterns:
        match = pattern.search(line)
        if not match:
            continue
        value = float(match.group(1))
        if math.isfinite(value):
            print(f"{value:.3f}")
            raise SystemExit(0)
PY
}

profile_label() {
  local log="$1"
  if [[ "$DECODE_PROFILE" != "true" && "$DECODE_PROFILE" != "1" && "$DECODE_PROFILE" != "yes" \
     && "$COLD_PROFILE" != "true" && "$COLD_PROFILE" != "1" && "$COLD_PROFILE" != "yes" ]]; then
    return 0
  fi
  /usr/bin/python3 - "$log" "$DECODE_PROFILE" "$COLD_PROFILE" <<'PY'
import re, statistics, sys
decode_on = sys.argv[2].lower() in ("1", "true", "yes")
cold_on = sys.argv[3].lower() in ("1", "true", "yes")
profiles = []
load = None
ready_s = None
client = []
for line in open(sys.argv[1], errors="replace"):
    if "[bench-cold]" in line:
        vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
        if "ready_s" in vals:
            ready_s = float(vals["ready_s"])
    if "[client-request]" in line:
        vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
        try:
            client.append((
                int(float(vals["id"])),
                float(vals["seconds"]),
                float(vals["first_byte_s"]),
            ))
        except KeyError:
            pass
    if "[profile-load]" in line:
        vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
        try:
            load = (
                float(vals["total_s"]),
                float(vals["tensors_s"]),
                float(vals["build_s"]),
                float(vals["tokenizer_s"]),
            )
        except KeyError:
            pass
    if not decode_on or "[profile-decode]" not in line:
        continue
    vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
    try:
        profiles.append((
            float(vals["step_p95_ms"]),
            float(vals["step_p99_ms"]),
            float(vals["step_max_ms"]),
            int(float(vals["step_max_i"])),
            float(vals.get("prefill_s", "nan")),
            float(vals.get("tokenize_s", "nan")),
            int(float(vals.get("prompt_tokens", "0"))),
        ))
    except KeyError:
        pass

extras = []
if decode_on and profiles:
    p95 = statistics.median(p[0] for p in profiles)
    p99 = statistics.median(p[1] for p in profiles)
    worst = max(profiles, key=lambda p: p[2])
    prefill = [p[4] * 1000.0 for p in profiles if p[4] == p[4]]
    tokenize = [p[5] * 1000.0 for p in profiles if p[5] == p[5]]
    prompt_tokens = [p[6] for p in profiles if p[6] > 0]
    if load is not None:
        extras.append(f"load={load[0]:.2f}s")
        extras.append(f"tensors={load[1]:.2f}s")
        extras.append(f"build={load[2]:.2f}s")
    if prefill:
        extras.append(f"prefill_med={statistics.median(prefill):.2f}ms")
    if tokenize:
        extras.append(f"tok_med={statistics.median(tokenize):.2f}ms")
    if prompt_tokens:
        extras.append(f"prompt_tok_med={statistics.median(prompt_tokens):.0f}")
    extras.insert(0, f"p95={p95:.2f}ms p99={p99:.2f}ms max={worst[2]:.2f}ms@{worst[3]}")
elif decode_on:
    if load is None:
        extras.append("profile=missing")
    else:
        extras.append(f"load={load[0]:.2f}s tensors={load[1]:.2f}s build={load[2]:.2f}s tokbuild={load[3]:.2f}s")

if cold_on:
    client.sort(key=lambda row: row[0])
    if ready_s is not None:
        extras.append(f"ready={ready_s:.2f}s")
    if client:
        first_req = client[0][1]
        first_byte = client[0][2]
        rtts = [row[1] for row in client]
        extras.append(f"first_rtt={first_req:.2f}s")
        extras.append(f"first_byte={first_byte:.2f}s")
        if ready_s is not None:
            extras.append(f"cold_first={ready_s + first_req:.2f}s")
        extras.append(f"client_med={statistics.median(rtts):.2f}s")

if extras:
    print("  " + " ".join(extras))
PY
}

request_label() {
  local log="$1"
  if [[ "$REQUEST_PROFILE" != "true" && "$REQUEST_PROFILE" != "1" && "$REQUEST_PROFILE" != "yes" ]]; then
    return 0
  fi
  /usr/bin/python3 - "$log" <<'PY'
import re, sys

requests = []
clients = {}
for line in open(sys.argv[1], errors="replace"):
    if "[request-done]" in line:
        match = re.search(
            r"id=(\d+).*tokens=([0-9.]+).*seconds=([0-9.]+).*tok/s=([0-9.]+)",
            line,
        )
        if match:
            requests.append((
                int(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            ))
    elif "[client-request]" in line:
        vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
        try:
            clients[int(float(vals["id"]))] = (
                float(vals["seconds"]),
                float(vals["first_byte_s"]),
                int(float(vals["bytes"])),
            )
        except KeyError:
            pass

requests.sort(key=lambda row: row[0])
if not requests:
    raise SystemExit(0)

req_tps = "/".join(f"{row[3]:.1f}" for row in requests)
req_s = "/".join(f"{row[2]:.2f}" for row in requests)
first_b = "/".join(
    f"{clients[row[0]][1]:.2f}" if row[0] in clients else "nan"
    for row in requests
)
bytes_s = "/".join(
    str(clients[row[0]][2]) if row[0] in clients else "nan"
    for row in requests
)
print(f"  req_tps={req_tps} req_s={req_s} fb={first_b} bytes={bytes_s}")
PY
}

extract_cold_metrics() {
  local log="$1"
  local ready_s="$2"
  /usr/bin/python3 - "$log" "$ready_s" <<'PY'
import math, re, statistics, sys

ready = float(sys.argv[2])
client = []
for line in open(sys.argv[1], errors="replace"):
    if "[client-request]" not in line:
        continue
    vals = dict(re.findall(r"([a-z0-9_]+)=([0-9.]+)", line))
    try:
        client.append((
            int(float(vals["id"])),
            float(vals["seconds"]),
            float(vals["first_byte_s"]),
        ))
    except KeyError:
        pass

client.sort(key=lambda row: row[0])
if client:
    first = client[0][1]
    first_byte = client[0][2]
    median = statistics.median(row[1] for row in client)
    print(
        f"{ready:.6f}\t{first:.6f}\t{first_byte:.6f}\t"
        f"{ready + first:.6f}\t{ready + first_byte:.6f}\t{median:.6f}"
    )
else:
    print(f"{ready:.6f}\tnan\tnan\tnan\tnan\tnan")
PY
}

run_one() {
  local run_no="$1"
  local variant="$2"
  local ordinal="$3"
  local port=$((PORT_BASE + ordinal))
  local log="$TMP_DIR/run-${run_no}-${variant//\//_}.log"
  local client_log="$TMP_DIR/client-${run_no}-${variant//\//_}.log"
  local cmd=("$RAYZOR" run)

  if [[ "$NO_CACHE" == "true" || "$NO_CACHE" == "1" || "$NO_CACHE" == "yes" ]]; then
    cmd+=("--no-cache")
  fi
  cmd+=("--safety-warnings" "off" "Main.hx" "--preset" "$PRESET" "--stats")
  if [[ "$variant" != "toml" ]]; then
    cmd+=("--tier-thresholds" "$variant")
  fi
  cmd+=("--tier-start-interpreted" "$START_INTERPRETED")
  cmd+=("--tier-promotion" "$TIER_PROMOTION")
  cmd+=("--" "--requests" "$REQUESTS" "--max" "$MAX_TOKENS" "--ctx" "$CTX" "--temp" "$TEMP" "--listen" "$port")

  local env_cmd=("env" "RAYZOR_SERVER_MODEL=$MODEL_PATH")
  if [[ "$DECODE_PROFILE" == "true" || "$DECODE_PROFILE" == "1" || "$DECODE_PROFILE" == "yes" ]]; then
    env_cmd+=("RAYZOR_PROFILE_DECODE=1")
  fi
  if [[ "$PREFILL_MORSELS" == "true" || "$PREFILL_MORSELS" == "1" || "$PREFILL_MORSELS" == "yes" ]]; then
    env_cmd+=("RAYZOR_PREFILL_MORSELS=1")
  fi
  if [[ "$STREAM" == "true" || "$STREAM" == "1" || "$STREAM" == "yes" ]]; then
    env_cmd+=("RAYZOR_SERVER_STREAM=1")
  fi
  local launch_start
  launch_start="$(now_s)"
  "${env_cmd[@]}" "${cmd[@]}" > "$log" 2>&1 &
  CURRENT_PID=$!

  if ! wait_for_server "$log" "$CURRENT_PID"; then
    kill "$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
    CURRENT_PID=""
    echo -e "${variant}\tFAIL" >> "$RESULTS"
    printf "  run %3d  %-14s FAIL     server did not reach listen\n" "$run_no" "$variant"
    return 0
  fi
  local ready_s
  ready_s="$(elapsed_s "$launch_start" "$(now_s)")"

  if ! send_requests "$port" > "$client_log" 2>&1; then
    kill "$CURRENT_PID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
    CURRENT_PID=""
    echo -e "${variant}\tFAIL" >> "$RESULTS"
    printf "  run %3d  %-14s FAIL     client request failed\n" "$run_no" "$variant"
    return 0
  fi

  local deadline=$((SECONDS + TIMEOUT))
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      kill "$CURRENT_PID" 2>/dev/null || true
      wait "$CURRENT_PID" 2>/dev/null || true
      CURRENT_PID=""
      echo -e "${variant}\tFAIL" >> "$RESULTS"
      printf "  run %3d  %-14s FAIL     server timeout after requests\n" "$run_no" "$variant"
      return 0
    fi
    sleep_ms 100
  done
  wait "$CURRENT_PID" 2>/dev/null || true
  CURRENT_PID=""

  {
    printf "[bench-cold] ready_s=%s\n" "$ready_s"
    if [[ -f "$client_log" ]]; then
      cat "$client_log"
    fi
  } >> "$log"

  local metric
  metric="$(extract_metric "$log")"
  if [[ -z "$metric" ]]; then
    echo -e "${variant}\tFAIL" >> "$RESULTS"
    printf "  run %3d  %-14s FAIL     no metric in server log (%s)\n" "$run_no" "$variant" "$log"
    return 0
  fi

  local profile
  profile="$(profile_label "$log")"
  local requests
  requests="$(request_label "$log")"
  echo -e "${variant}\t$metric" >> "$RESULTS"
  if [[ "$COLD_PROFILE" == "true" || "$COLD_PROFILE" == "1" || "$COLD_PROFILE" == "yes" ]]; then
    echo -e "${variant}\t$(extract_cold_metrics "$log" "$ready_s")" >> "$COLD_RESULTS"
  fi
  printf "  run %3d  %-14s ok       tok/s=%.2f%s%s\n" "$run_no" "$variant" "$metric" "$profile" "$requests"
}

print_summary() {
  /usr/bin/python3 - "$RESULTS" "$RUNS" "$COLD_RESULTS" "$COLD_PROFILE" "${VARIANT_LIST[@]}" <<'PY'
import math, statistics, sys

path = sys.argv[1]
runs = int(sys.argv[2])
cold_path = sys.argv[3]
cold_on = sys.argv[4].lower() in ("1", "true", "yes")
variants = sys.argv[5:]
data = {v: [] for v in variants}
failed = {v: 0 for v in variants}
cold = {v: [] for v in variants}

for line in open(path):
    label, value = line.rstrip("\n").split("\t", 1)
    if value == "FAIL":
        failed[label] = failed.get(label, 0) + 1
    else:
        data.setdefault(label, []).append(float(value))

if cold_on:
    for line in open(cold_path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 5:
            label = parts[0]
            try:
                ready, first, cold_first, client_med = (float(v) for v in parts[1:])
            except ValueError:
                continue
            cold.setdefault(label, []).append((ready, first, float("nan"), cold_first, float("nan"), client_med))
            continue
        if len(parts) != 7:
            continue
        label = parts[0]
        try:
            cold.setdefault(label, []).append(tuple(float(v) for v in parts[1:]))
        except ValueError:
            pass

print()
print("=== summary ===")
for label in variants:
    samples = data.get(label, [])
    ok = len(samples)
    print(f"{label:<14} successful: {ok}/{runs}")
    if not samples:
        continue
    sorted_samples = sorted(samples)
    mean = statistics.mean(sorted_samples)
    median = statistics.median(sorted_samples)
    stddev = math.sqrt(sum((x - mean) ** 2 for x in sorted_samples) / len(sorted_samples))
    drift = sorted_samples[-1] - sorted_samples[0] if len(sorted_samples) > 1 else float("nan")
    first_last_drift = samples[-1] - samples[0] if len(samples) > 1 else drift
    print(
        f"{'':14} tok/s: min={min(sorted_samples):.2f}  median={median:.2f}  "
        f"mean={mean:.2f}  max={max(sorted_samples):.2f}  stddev={stddev:.2f}  "
        f"drift={first_last_drift:.2f}"
    )
    cold_samples = cold.get(label, [])
    if cold_on and cold_samples:
        ready = [row[0] for row in cold_samples if math.isfinite(row[0])]
        first = [row[1] for row in cold_samples if math.isfinite(row[1])]
        first_byte = [row[2] for row in cold_samples if math.isfinite(row[2])]
        cold_first = [row[3] for row in cold_samples if math.isfinite(row[3])]
        cold_byte = [row[4] for row in cold_samples if math.isfinite(row[4])]
        client_med = [row[5] for row in cold_samples if math.isfinite(row[5])]
        if ready and first and cold_first and client_med:
            line = (
                f"{'':14} cold: ready_med={statistics.median(ready):.2f}s  "
                f"first_rtt_med={statistics.median(first):.2f}s  "
            )
            if first_byte and cold_byte:
                line += (
                    f"first_byte_med={statistics.median(first_byte):.2f}s  "
                    f"cold_ttft_med={statistics.median(cold_byte):.2f}s  "
                )
            line += (
                f"first_done_med={statistics.median(cold_first):.2f}s  "
                f"client_med={statistics.median(client_med):.2f}s"
            )
            print(line)
PY
}

echo "=== rayzor server bench ==="
echo "file:    Main.hx"
echo "runs:    $RUNS per variant ($TOTAL_RUNS subprocesses)"
echo "metric:  tok/s"
echo "preset:  $PRESET"
echo "mode:    TCP prompt requests"
echo "requests:$REQUESTS per subprocess"
if [[ ${#VARIANT_LIST[@]} -gt 1 ]]; then
  joined="$(printf "%s, " "${VARIANT_LIST[@]}")"
  echo "variants: ${joined%, }"
  echo "order:   round-robin"
fi
echo "start:   interpreted=$START_INTERPRETED"
echo "promote: $TIER_PROMOTION"
echo "prefill: morsels=$PREFILL_MORSELS"
echo "stream:  $STREAM"
echo "request: per-request profile=$REQUEST_PROFILE"
if [[ "$COLD_PROFILE" == "true" || "$COLD_PROFILE" == "1" || "$COLD_PROFILE" == "yes" ]]; then
  echo "cold:   launch/listen + client RTT enabled"
fi
if [[ "$DECODE_PROFILE" == "true" || "$DECODE_PROFILE" == "1" || "$DECODE_PROFILE" == "yes" ]]; then
  echo "decode:  tail-latency profile enabled"
fi
if (( COOLDOWN_MS > 0 )); then
  echo "cooldown: $COOLDOWN_MS ms between subprocesses"
fi
echo

ordinal=0
for run_no in $(seq 1 "$RUNS"); do
  for variant in "${VARIANT_LIST[@]}"; do
    ordinal=$((ordinal + 1))
    run_one "$run_no" "$variant" "$ordinal"
    if (( COOLDOWN_MS > 0 && ordinal < TOTAL_RUNS )); then
      sleep_ms "$COOLDOWN_MS"
    fi
  done
done

print_summary
