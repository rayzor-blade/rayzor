#!/usr/bin/env python3
"""Run a command under a wall-clock limit, exiting 124 if it overruns.

`timeout(1)` is not present on every platform the harness runs on, and a test
that hangs would otherwise stall the whole corpus.
"""
import subprocess
import sys

limit = float(sys.argv[1])
try:
    p = subprocess.run(sys.argv[2:], capture_output=True, timeout=limit)
except subprocess.TimeoutExpired as e:
    for chunk in (e.stdout, e.stderr):
        if chunk:
            sys.stdout.write(chunk.decode("utf-8", "replace"))
    sys.exit(124)
sys.stdout.write(p.stdout.decode("utf-8", "replace"))
sys.stdout.write(p.stderr.decode("utf-8", "replace"))
sys.exit(p.returncode)
