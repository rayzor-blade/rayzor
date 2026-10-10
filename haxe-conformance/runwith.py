#!/usr/bin/env python3
"""Run a command under a wall-clock limit, exiting 124 if it overruns.

`timeout(1)` is not present on every platform the harness runs on, and a test
that hangs would otherwise stall the whole corpus.
"""
import json
import os
import subprocess
import sys

limit = float(sys.argv[1])
cmd = sys.argv[2:]
# RUNWITH_ARGS names a JSON list of program arguments, passed after `--`. A
# file rather than argv, so they survive bash 3.2 intact: empty strings,
# newlines and quotes included.
if os.environ.get("RUNWITH_ARGS"):
    with open(os.environ["RUNWITH_ARGS"], encoding="utf-8") as f:
        cmd += ["--"] + json.load(f)
try:
    p = subprocess.run(cmd, capture_output=True, timeout=limit)
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
