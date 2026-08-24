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
# A signal comes back negative here, and exiting with it wraps: SIGSEGV
# arrives as -11 and leaves as 245, which reads as an ordinary failure
# and files a crash under whatever bucket that number lands in. Report
# signals the way a shell does, so they stay recognisable as signals.
sys.exit(128 - p.returncode if p.returncode < 0 else p.returncode)
