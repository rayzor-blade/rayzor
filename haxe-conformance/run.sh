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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
SRC="${SRC:-$HERE/corpus/tests/unit/src/unit/issues}"
WORK="${WORK:-${TMPDIR:-/tmp}/rayzor_conformance}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-$WORK/report.tsv}"
RAYZOR="${RAYZOR:-$REPO/target/release/rayzor}"

# Package roots that only exist on one Haxe target.
TARGET_PKGS="php|js|cs|java|python|lua|flash|neko|hl|eval|cpp|jvm"

mkdir -p "$WORK/run"
printf 'issue\tstatus\tdetail\n' > "$OUT"

n=0; pass=0; fail=0
for f in "$SRC"/*.hx; do
  base="$(basename "$f" .hx)"
  [[ "$LIMIT" != 0 && $n -ge $LIMIT ]] && break
  grep -q "extends unit.Test" "$f" || continue

  # A test that imports a target-language package is asserting about that
  # target's semantics, not about Haxe. Out of scope: rayzor is its own
  # target and will never satisfy it. Conditional compilation is NOT a
  # filter -- a `#if cpp ... #else ... #end` still has a branch that is
  # ours, so judging by `#if` would drop tests that legitimately apply.
  if grep -qE "^[[:space:]]*(import|using)[[:space:]]+($TARGET_PKGS)\." "$f"; then
    pkg=$(grep -oE "^[[:space:]]*(import|using)[[:space:]]+($TARGET_PKGS)\.[A-Za-z0-9_.]*" "$f" \
          | awk '{print $2}' | sort -u | tr '\n' ' ')
    printf '%s\tSKIP\ttargets %s\n' "$base" "${pkg% }" >> "$OUT"; continue
  fi

  n=$((n+1))

  d="$WORK/run/$base"; rm -rf "$d"; mkdir -p "$d/unit/issues" "$d/unit"
  cp "$HERE/shims/unit/Test.hx" "$d/unit/Test.hx"
  cp "$HERE/shims/unit/ConfCheck.hx" "$d/unit/ConfCheck.hx"

  # Inject main() as the last member of the class; awk -v cannot carry newlines.
  # Exits 3 when the class declares no test method that is live for us.
  python3 - "$f" "$d/unit/issues/$base.hx" "$base" "$TARGET_PKGS" <<'PYGEN'
import sys
src, dst, cls, target_pkgs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(src, encoding='utf-8', errors='replace').read()
lines = text.split('\n')
# Find the closing brace of `class <cls>` specifically -- a file often declares
# private helper types after it, and the last brace belongs to one of those.
import re

# Which lines are live for us. A `#if cpp` branch is dead here and its `#else`
# is live; a condition naming anything else (macro, sys, static, a version
# check) is left live, since guessing wrong would drop tests that do apply.
# Only method DISCOVERY uses this -- the source is copied through verbatim and
# rayzor's own preprocessor decides what to compile.
targets = set(target_pkgs.split('|'))
def branch_is_ours(cond):
    words = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', cond))
    return not (words and words <= targets)

live = [True] * len(lines)
stack = []            # (this_branch_live, any_branch_taken_yet)
for i, l in enumerate(lines):
    st = l.strip()
    m = re.match(r'#(if|elseif|else|end)\b(.*)', st)
    if m:
        kind, rest = m.group(1), m.group(2)
        if kind == 'if':
            ours = branch_is_ours(rest)
            stack.append([ours, ours])
        elif stack:
            if kind == 'end':
                stack.pop()
            elif kind == 'elseif':
                ours = branch_is_ours(rest)
                stack[-1][0] = ours and not stack[-1][1]
                stack[-1][1] = stack[-1][1] or ours
            else:                       # else
                stack[-1][0] = not stack[-1][1]
        live[i] = all(f[0] for f in stack) if stack else True
        continue
    live[i] = all(f[0] for f in stack) if stack else True

methods = []
for i, l in enumerate(lines):
    if not live[i]:
        continue
    m = re.search(r'\bfunction\s+(test[A-Za-z0-9_]*)\s*\(', l)
    if m and m.group(1) not in methods:
        methods.append(m.group(1))
if not methods:
    sys.exit(3)

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
  gen=$?
  if [[ $gen -eq 3 ]]; then
    printf '%s\tSKIP\tno test method live for this target\n' "$base" >> "$OUT"; n=$((n-1)); continue
  elif [[ $gen -ne 0 ]]; then
    printf '%s\tSKIP\tharness could not inject main\n' "$base" >> "$OUT"; n=$((n-1)); continue
  fi

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
    # A named uncompiled function is the most actionable thing a run can say:
    # it points at the exact construct the compiler could not build. Prefer it
    # over the generic exit code, which is what every one of these used to be.
    [[ -z "$det" ]] && det=$(printf '%s' "$out" | grep -oE 'rayzor: `[^`]+` was never compiled' \
                             | head -1 | sed -E 's/rayzor: `(.*)` was never compiled/uncompiled \1/')
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
