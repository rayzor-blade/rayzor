#!/usr/bin/env bash
# Bench the nue embed-server: throughput (sentences/s) + p50/p95 request latency.
#
#   ./bench_embed.sh [model.gguf]
#   REQUESTS=60 BATCH=16 WARMUP=8 PORT=8071 ./bench_embed.sh
#
# Loads the model once, then drives REQUESTS batches of BATCH sentences each
# through the TCP server. The first WARMUP requests are discarded (JIT tier
# promotion + cache warm-up) before the reported percentiles are computed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RAYZOR="$ROOT/target/release/rayzor"
CORPUS="$ROOT/nue/tests/bert/goldens/sentences.txt"

MODEL="${1:-$HOME/models/minilm/all-MiniLM-L6-v2.F32.gguf}"
PORT="${PORT:-8071}"
REQUESTS="${REQUESTS:-40}"
BATCH="${BATCH:-8}"
WARMUP="${WARMUP:-5}"

[ -f "$MODEL" ]  || { echo "model not found: $MODEL" >&2; exit 1; }
[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS" >&2; exit 1; }

# Benchmark hygiene: an orphaned server pins a core and inverts A/B.
pkill -9 -f "target/release/rayzor" 2>/dev/null || true
find "$ROOT" -name ".rayzor" -type d -exec rm -rf {} + 2>/dev/null || true

LOG="$(mktemp -t embsrv.XXXX.log)"
echo "[bench] model=$MODEL port=$PORT requests=$REQUESTS batch=$BATCH warmup=$WARMUP"
"$RAYZOR" run "$ROOT/nue/examples/embed-server/Main.hx" --llvm --release --no-cache -- \
    --model "$MODEL" --listen "$PORT" --requests "$REQUESTS" > "$LOG" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null || true; pkill -9 -f "target/release/rayzor" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do grep -aq "listening" "$LOG" && break; sleep 1; done
grep -aq "listening" "$LOG" || { echo "server failed to start"; tail -8 "$LOG"; exit 1; }
echo "[bench] server ready"

python3 - "$PORT" "$CORPUS" "$REQUESTS" "$BATCH" "$WARMUP" <<'PY'
import socket, sys, time, math
port, corpus, requests, batch, warmup = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
pool = [l.strip() for l in open(corpus) if l.strip()]
lat, sent = [], 0
for r in range(requests):
    sents = [pool[(r*batch + i) % len(pool)] for i in range(batch)]
    payload = ("\n".join(sents)).encode()
    t0 = time.perf_counter()
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall(payload); s.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
        c = s.recv(65536)
        if not c: break
        buf += c
    s.close()
    dt = time.perf_counter() - t0
    n = int(buf.split(b"\n", 1)[0].split()[0])
    if r >= warmup:
        lat.append(dt); sent += n
m = sorted(lat)
def pct(p): return m[min(len(m)-1, int(p/100*len(m)))]
wall = sum(lat)
print(f"[bench] measured {len(lat)} requests ({sent} sentences)")
print(f"[bench] throughput = {sent/wall:.1f} sentences/s")
print(f"[bench] latency/request  p50={pct(50)*1e3:.1f}ms  p95={pct(95)*1e3:.1f}ms  max={m[-1]*1e3:.1f}ms")
print(f"[bench] latency/sentence p50={pct(50)/batch*1e3:.2f}ms")
PY

echo "--- server summary ---"
grep -aE "^\[server-done\]" "$LOG" || true
