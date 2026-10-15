#!/usr/bin/env python3
"""Check AOT manifest class paths when the entry is outside the project."""
import json
import os
from pathlib import Path
import subprocess
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
    result = subprocess.run([str(output)], capture_output=True, text=True, check=True, timeout=30)
    assert "CONFORMANCE_OK" in result.stdout, result.stdout + result.stderr
    assert not cache.exists(), "--no-cache created a cache directory"
    print("PASS AOT external manifest entry")
