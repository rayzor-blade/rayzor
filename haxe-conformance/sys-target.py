#!/usr/bin/env python3
"""Supply Rayzor branches for upstream sys target adapters."""
import json
import os
from pathlib import Path
import sys


def replace_once(text, old, new):
    if old not in text:
        raise ValueError(f"upstream sys adapter changed: {old!r}")
    return text.replace(old, new, 1)


def adapt(path, runner):
    text = path.read_text(encoding="utf-8")
    binary = "UtilityProcess.exe" if os.name == "nt" else "UtilityProcess"
    if path.name == "UtilityProcess.hx":
        text = replace_once(text, '#if cpp\n', '#if rayzor\n\t\t"bin/rayzor";\n#elseif cpp\n')
        text = replace_once(text, '#if cpp\n', '#if rayzor\n\t\t' + json.dumps(binary) + ';\n#elseif cpp\n')
        text = replace_once(text, '#if (macro || interp)\n\t\tnew Process',
                            '#if rayzor\n\t\tnew Process(execFull, args);\n\t\t#elseif (macro || interp)\n\t\tnew Process')
        text = replace_once(text, '#if (macro || interp)\n\t\tSys.command',
                            '#if rayzor\n\t\tSys.command(execFull, args);\n\t\t#elseif (macro || interp)\n\t\tSys.command')
    elif path.name == "TestUnicode.hx":
        text = replace_once(text, '#if cpp\n', '#if rayzor\n\t\t"bin-rayzor";\n#elseif cpp\n')
    elif path.name == "TestSys.hx":
        text = replace_once(text, '#if interp\n', '#if rayzor\n\t\t\t' + json.dumps(Path(runner).name) + '\n\t\t#elseif interp\n')
    elif path.name == "gen_test_res.py":
        text = replace_once(text, '    for target, name in [\n',
                            '    for target, name in [\n        (' + repr('../../bin/rayzor/' + binary) + ', "bin-rayzor"),\n')
    else:
        return
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    for name in sys.argv[2:]:
        adapt(Path(name), sys.argv[1])
