#!/usr/bin/env python3
"""Check native AOT project configuration and runtime boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
binary = os.environ.get("RAYZOR", str(root / "target/release/rayzor"))
if os.sep in binary:
    binary = os.path.abspath(binary)
projects = root / "haxe-conformance/regressions/projects"
with tempfile.TemporaryDirectory(prefix="rayzor-aot-manifest-") as directory:
    work = Path(directory)
    entry = projects / "aot_manifest_source/ExternalEntry.hx"
    helpers = projects / "aot_manifest_entry/helpers"
    (work / "rayzor.toml").write_text(
        '[project]\nname = "external-entry"\nentry = ' + json.dumps(str(entry))
        + '\n[build]\nclass-paths = [' + json.dumps(str(helpers)) + ']\n'
    )
    output = work / ("external-entry.exe" if os.name == "nt" else "external-entry")
    cache = work / "disabled-cache"
    subprocess.run(
        [binary, "aot", "--no-cache", "--cache-dir", str(cache), "--strip", "-o", str(output)],
        cwd=work, check=True, timeout=120,
    )
    result = subprocess.run([str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONFORMANCE_OK" in result.stdout, result.stdout + result.stderr
    assert not cache.exists(), "--no-cache created a cache directory"
    print("PASS AOT external manifest entry")

for fixture in ["ProcessPipes", "StandardStreams", "EmptyNativeStrings", "FileStreams", "ArrayIterationSource"]:
    with tempfile.TemporaryDirectory(prefix="rayzor-aot-runtime-") as directory:
        work = Path(directory)
        output = work / (fixture + (".exe" if os.name == "nt" else ""))
        source = root / "haxe-conformance/regressions" / (fixture + ".hx")
        subprocess.run(
            [binary, "aot", str(source), "--no-cache", "--strip", "-o", str(output)],
            cwd=work, check=True, timeout=120,
        )
        result = subprocess.run(
            [sys.executable, str(root / "haxe-conformance/runwith.py"), "30", str(output)],
            capture_output=True, text=True, timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "CONFORMANCE_OK" in result.stdout, result.stdout + result.stderr
        print("PASS AOT " + fixture)
