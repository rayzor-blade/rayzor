#!/usr/bin/env python3
"""Run focused compiler/runtime regressions with the same flags as conformance."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
binary = os.environ.get("RAYZOR", str(root / "target/release/rayzor"))
if os.sep in binary:
    binary = os.path.abspath(binary)
failed = []
regressions = root / "haxe-conformance/regressions"
# Projects supply class paths for regressions involving independently lowered modules.
fixtures = sorted(regressions.glob("*.hx")) + sorted(regressions.glob("projects/*/rayzor.toml"))
for fixture in fixtures:
    project = fixture.name == "rayzor.toml"
    label = fixture.parent.name if project else fixture.name
    command = [binary, "run", "--release", "--no-cache", "--preset", "application"]
    command += sys.argv[1:]
    if not project:
        command.append(str(fixture))
    try:
        result = subprocess.run(
            command, cwd=fixture.parent if project else None,
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        ok = result.returncode == 0 and "CONFORMANCE_OK" in output
    except subprocess.TimeoutExpired as error:
        output = str(error)
        ok = False
    print(f"{'PASS' if ok else 'FAIL'} {label}", flush=True)
    if not ok:
        failed.append(label)
        print(output, flush=True)
sys.exit(bool(failed))
