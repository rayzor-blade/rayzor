#!/usr/bin/env python3
"""Run focused compiler/runtime regressions with the same flags as conformance."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
binary = os.environ.get("RAYZOR", str(root / "target/release/rayzor"))
failed = []
for fixture in sorted((root / "haxe-conformance/regressions").glob("*.hx")):
    command = [binary, "run", "--release", "--no-cache", "--preset", "application"]
    command += sys.argv[1:]
    command.append(str(fixture))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        ok = result.returncode == 0 and "CONFORMANCE_OK" in output
    except subprocess.TimeoutExpired as error:
        output = str(error)
        ok = False
    print(f"{'PASS' if ok else 'FAIL'} {fixture.name}", flush=True)
    if not ok:
        failed.append(fixture.name)
        print(output, flush=True)
sys.exit(bool(failed))
