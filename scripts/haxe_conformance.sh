#!/usr/bin/env bash
# Haxe conformance harness: run official HaxeFoundation/haxe tests/unit issue
# regressions through rayzor and categorise every outcome.
#
# Each IssueNNNN.hx is a class extending unit.Test with test*() methods. We copy
# it beside a minimal unit.Test shim, append a main() INSIDE the class (so the
# private test methods are legitimately reachable), compile and run.
#
# A test counts as PASS only if it exits 0 AND prints "CONFORMANCE_OK" AND emits
# no FAILCHECK line. Silence is never a pass.
set -uo pipefail

REPO="${REPO:-/Users/amaterasu/Vibranium/rayzor}"
SRC="${SRC:-/Users/amaterasu/Vibranium/haxe-tests/tests/unit/src/unit/issues}"
WORK="${WORK:-${TMPDIR:-/tmp}/rayzor_conformance}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-$WORK/report.tsv}"
RAYZOR="$REPO/target/release/rayzor"

mkdir -p "$WORK/run"
printf 'issue\tstatus\tdetail\n' > "$OUT"

n=0; pass=0; fail=0
for f in "$SRC"/*.hx; do
  base="$(basename "$f" .hx)"
  [[ "$LIMIT" != 0 && $n -ge $LIMIT ]] && break
  grep -q "extends unit.Test" "$f" || continue
  n=$((n+1))

  # test method names declared in this file
  methods=$(grep -oE '\bfunction[[:space:]]+(test[A-Za-z0-9_]*)[[:space:]]*\(' "$f" \
            | sed -E 's/function[[:space:]]+//; s/[[:space:]]*\(//' | sort -u)
  if [[ -z "$methods" ]]; then
    printf '%s\tSKIP\tno test methods\n' "$base" >> "$OUT"; n=$((n-1)); continue
  fi

  d="$WORK/run/$base"; rm -rf "$d"; mkdir -p "$d/unit/issues" "$d/unit"
  cp "$REPO/compiler/tests/conformance/unit/Test.hx" "$d/unit/Test.hx"
  cp "$REPO/compiler/tests/conformance/unit/ConfCheck.hx" "$d/unit/ConfCheck.hx"

  # Inject main() as the last member of the class; awk -v cannot carry newlines.
  python3 - "$f" "$d/unit/issues/$base.hx" "$base" $methods <<'PYGEN'
import sys
src, dst, cls = sys.argv[1], sys.argv[2], sys.argv[3]
methods = sys.argv[4:]
text = open(src, encoding='utf-8', errors='replace').read()
lines = text.split('\n')
# Find the closing brace of `class <cls>` specifically -- a file often declares
# private helper types after it, and the last brace belongs to one of those.
import re
start = next(i for i, l in enumerate(lines) if re.search(r'\bclass\s+%s\b' % cls, l))
depth = 0
opened = False
last = len(lines) - 1
for i in range(start, len(lines)):
    depth += lines[i].count('{') - lines[i].count('}')
    if '{' in lines[i]:
        opened = True
    if opened and depth <= 0:
        last = i
        break
main = ['    public static function main():Void {',
        '        var inst = new %s();' % cls]
main += ['        inst.%s();' % m for m in methods]
main += ['        unit.ConfCheck.summary();', '    }']
open(dst, 'w', encoding='utf-8').write('\n'.join(lines[:last] + main + lines[last:]))
PYGEN

  # Run as a project, not a lone file. `rayzor run <file>` compiles that file
  # and the standard library and nothing else: a class the test inherits from
  # -- `unit.Test`, which every one of them extends -- is known well enough to
  # declare a forward reference and never compiled, so the reference becomes a
  # trap stub and the test dies on SIGTRAP with nothing said. A manifest with a
  # class path compiles the siblings too.
  printf '[project]\nname = "conformance"\nentry = "unit/issues/%s.hx"\n\n[build]\nclass-paths = ["."]\n' \
    "$base" > "$d/rayzor.toml"
  out=$( cd "$d" && "$RAYZOR" run --release --no-cache 2>&1 )
  code=$?

  if [[ $code -ne 0 ]]; then
    det=$(printf '%s' "$out" | grep -oE '\[E[0-9]+\][^"]{0,80}' | head -1)
    [[ -z "$det" ]] && det=$(printf '%s' "$out" | grep -iE 'error|panic|signal' | head -1 | cut -c1-90)
    [[ -z "$det" ]] && det="exit $code"
    case "$code" in
      132|134|139|133) status="CRASH" ;;
      *) status="COMPILE_FAIL" ;;
    esac
    if [[ "$det" == *"No main function found"* ]]; then
      printf '%s\tSKIP\tharness could not inject main\n' "$base" >> "$OUT"; n=$((n-1))
    elif [[ "$det" == *"'utest'"* ]]; then
      printf '%s\tSKIP\tuses utest directly\n' "$base" >> "$OUT"; n=$((n-1))
    else
      printf '%s\t%s\t%s\n' "$base" "$status" "${det//$'\t'/ }" >> "$OUT"; fail=$((fail+1))
    fi
  elif printf '%s' "$out" | grep -q '^CONFORMANCE_OK'; then
    printf '%s\tPASS\t%s\n' "$base" "$(printf '%s' "$out" | grep -o 'CONFORMANCE_OK.*')" >> "$OUT"; pass=$((pass+1))
  elif printf '%s' "$out" | grep -q 'FAILCHECK\|CONFORMANCE_BAD'; then
    printf '%s\tWRONG_ANSWER\t%s\n' "$base" "$(printf '%s' "$out" | grep -m1 'FAILCHECK' | cut -c1-90)" >> "$OUT"; fail=$((fail+1))
  else
    printf '%s\tNO_OUTPUT\t%s\n' "$base" "$(printf '%s' "$out" | tail -1 | cut -c1-90)" >> "$OUT"; fail=$((fail+1))
  fi
done

echo "ran $n  pass $pass  fail $fail"
echo "report: $OUT"
