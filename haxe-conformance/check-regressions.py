#!/usr/bin/env python3
"""Run focused compiler/runtime regressions with the same flags as conformance."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
binary = os.environ.get("RAYZOR", str(root / "target/release/rayzor"))
if os.sep in binary:
    binary = os.path.abspath(binary)
failed = []
regressions = root / "haxe-conformance/regressions"
# Projects supply class paths for regressions involving independently lowered modules.
fixtures = sorted(regressions.glob("*.hx")) + sorted(regressions.glob("projects/*/rayzor.toml"))
# Every fixture runs without the cache. With REGRESSIONS_CACHE=1 it also runs
# with the cache on and an empty cache directory, the shipped default, where the
# embedded stdlib snapshot is what the program starts from.
# With REGRESSIONS_WARM=1 each fixture also runs twice against one cache
# directory, the entry cache removed between the runs, and the second run
# restores the first run's per-module artifacts: the incremental-build path.
modes = [("", ["--no-cache"])]
if os.environ.get("REGRESSIONS_CACHE"):
    modes.append((" [cache]", None))
if os.environ.get("REGRESSIONS_WARM"):
    modes.append((" [warm]", "warm"))
for fixture in fixtures:
    project = fixture.name == "rayzor.toml"
    name = fixture.parent.name if project else fixture.name
    for suffix, cache_flags in modes:
        label = name + suffix
        with tempfile.TemporaryDirectory() as scratch:
            warm = cache_flags == "warm"
            flags = ["--no-cache"] if cache_flags == ["--no-cache"] else ["--cache-dir", scratch]
            if warm:
                first = [binary, "run", "--release", *flags, "--preset", "application"]
                first += sys.argv[1:]
                if not project:
                    first.append(str(fixture))
                try:
                    subprocess.run(first, cwd=fixture.parent if project else None,
                                   capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired:
                    pass
                for entry in Path(scratch).glob("*.mir.cache"):
                    entry.unlink()
            command = [binary, "run", "--release", *flags, "--preset", "application"]
            command += sys.argv[1:]
            if not project:
                command.append(str(fixture))
            try:
                result = subprocess.run(
                    command, cwd=fixture.parent if project else None,
                    capture_output=True, text=True, timeout=60,
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
