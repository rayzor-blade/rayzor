#!/usr/bin/env python3
"""Hold N GB of dirty anonymous memory for T seconds, then exit.

Reproduces a busy personal machine: anonymous pages cannot be dropped, so macOS
compresses (then swaps) them, which is the condition a 7B model has to survive.
Self-limiting -- always releases on its own even if the caller dies.

Usage: ballast.py GB SECONDS [churn]

Passive mode holds the pages once. That is NOT enough to hold the box at
pressure level 2: macOS simply swaps an idle ballast out and the pressure level
falls back to 1 (measured -- 8.5 GB passive ballast sat at level 1 with 13.8 GB
of swap and >=60% free). `churn` keeps re-touching the pages so they must stay
resident, which is what actually sustains contention against the model mapping.
"""
import sys, time

gb = float(sys.argv[1])
secs = int(sys.argv[2])
churn = len(sys.argv) > 3 and sys.argv[3] == "churn"

chunk = 64 << 20  # 64 MB
n = int(gb * (1 << 30) // chunk)
blocks = []
for i in range(n):
    b = bytearray(chunk)
    b[::4096] = b"\xa5" * len(b[::4096])  # touch every page -> genuinely dirty
    blocks.append(b)
print(f"ballast: holding {n * chunk / (1<<30):.2f} GB for {secs}s"
      f"{' (churning)' if churn else ''}", flush=True)

deadline = time.time() + secs
if not churn:
    time.sleep(secs)
else:
    v = 0
    while time.time() < deadline:
        for b in blocks:
            b[::4096] = b"\xa5" * len(b[::4096])
            v += b[0]
        time.sleep(0.05)  # leave the CPU mostly free -- this is a MEMORY load
