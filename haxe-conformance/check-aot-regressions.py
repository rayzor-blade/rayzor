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

options = [["-O" + level] for level in ["0", "1", "2", "3", "s", "z"]]
options += [
    ["-O3", "--mir-opt-level", "0", "--codegen-opt-level", "0"],
    ["-O0", "--mir-opt-level", "3", "--codegen-opt-level", "3"],
]
with tempfile.TemporaryDirectory(prefix="rayzor-aot-optimization-") as directory:
    work = Path(directory)
    source = root / "haxe-conformance/regressions/AotOptimizationCalls.hx"
    output = work / ("calls.exe" if os.name == "nt" else "calls")
    for flags in options:
        subprocess.run(
            [binary, "aot", str(source), "--no-cache", *flags, "-o", str(output)],
            cwd=work, check=True, timeout=120,
        )
        result = subprocess.run([str(output)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (flags, result.stdout, result.stderr)
        assert result.stdout.strip() == "CONFORMANCE_OK 50", (flags, result.stdout, result.stderr)
        print("PASS AOT optimization " + " ".join(flags))
    cache = work / "blade"
    for state in ["cold", "warm"]:
        subprocess.run(
            [binary, "aot", str(source), "--cache-dir", str(cache),
             "-O3", "--codegen-opt-level", "0", "-o", str(output)],
            cwd=work, check=True, timeout=120,
        )
        result = subprocess.run([str(output)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (state, result.stdout, result.stderr)
        assert result.stdout.strip() == "CONFORMANCE_OK 50", (state, result.stdout, result.stderr)
        print("PASS AOT " + state + " cache optimization")
    ir_outputs = []
    for codegen in ["0", "3"]:
        ir = work / ("codegen-" + codegen + ".ll")
        subprocess.run(
            [binary, "aot", str(source), "--no-cache", "-O3", "--mir-opt-level", "0",
             "--codegen-opt-level", codegen, "--emit", "llvm-ir", "-o", str(ir)],
            cwd=work, check=True, timeout=120,
        )
        ir_outputs.append(ir.read_bytes())
    assert ir_outputs[0] == ir_outputs[1], "machine-code effort changed optimized LLVM IR"
    print("PASS AOT codegen effort preserves optimized LLVM IR")
    for flags, reason in [
        (["-O4"], "optimization level must be"),
        (["--mir-opt-level", "4"], "invalid value"),
        (["--codegen-opt-level", "4"], "invalid value"),
        (["--emit", "c", "--codegen-opt-level", "0"], "LLVM size/codegen options require"),
        (["--emit", "c", "-Oz"], "LLVM size/codegen options require"),
    ]:
        result = subprocess.run(
            [binary, "aot", str(source), *flags], cwd=work,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (flags, result.stdout, result.stderr)
        assert reason in result.stdout + result.stderr, (flags, result.stdout, result.stderr)
    print("PASS AOT invalid optimization controls")
