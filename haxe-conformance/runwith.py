#!/usr/bin/env python3
"""Run a command under a wall-clock limit, exiting 124 if it overruns.

`timeout(1)` is not present on every platform the harness runs on, and a test
that hangs would otherwise stall the whole corpus.
"""
import json
import os
import signal
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
# A timed-out native subprocess test can leave descendants holding its pipes.
# Give each case its own process group so the deadline covers the whole case.
with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      start_new_session=os.name == "posix") as p:
    try:
        stdout, stderr = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            p.kill()
        stdout, stderr = p.communicate()
        for chunk in (stdout, stderr):
            sys.stdout.write(chunk.decode("utf-8", "replace"))
        sys.exit(124)
sys.stdout.write(stdout.decode("utf-8", "replace"))
sys.stdout.write(stderr.decode("utf-8", "replace"))
# A signal comes back negative here, and exiting with it wraps: SIGSEGV
# arrives as -11 and leaves as 245, which reads as an ordinary failure
# and files a crash under whatever bucket that number lands in. Report
# signals the way a shell does, so they stay recognisable as signals.
sys.exit(128 - p.returncode if p.returncode < 0 else p.returncode)
