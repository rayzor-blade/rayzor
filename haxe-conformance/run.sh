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
TIMEOUT="${TIMEOUT:-60}"
OUT="${OUT:-$WORK/report.tsv}"
RAYZOR="${RAYZOR:-$REPO/target/release/rayzor}"

# Package roots that only exist on one Haxe target.
TARGET_PKGS="php|js|cs|java|python|lua|flash|neko|hl|eval|cpp|jvm"

# Fail loudly if the fixtures are missing. Without them every test compiles
# without unit.Test and the run reports a confident 0%, which looks like a
# catastrophic regression rather than a harness that cannot find its own
# files. A measurement that cannot be taken must not report a number.
for fixture in shims/unit/Test.hx shims/unit/ConfCheck.hx shims/utest/Assert.hx runwith.py; do
  [[ -f "$HERE/$fixture" ]] || { echo "harness incomplete: $HERE/$fixture is missing" >&2; exit 2; }
done
[[ -d "$SRC" ]] || { echo "no corpus at $SRC -- run ./fetch.sh, or set SRC" >&2; exit 2; }

mkdir -p "$WORK/run"
printf 'issue\tstatus\tdetail\n' > "$OUT"

# How many files are candidates, so progress has a denominator.
total=$(grep -lE "extends unit\.Test" "$SRC"/*.hx 2>/dev/null | wc -l | tr -d ' ')
[[ "$LIMIT" != 0 && $LIMIT -lt $total ]] && total=$LIMIT

seen=0
c_PASS=0; c_WRONG_ANSWER=0; c_NO_OUTPUT=0
c_COMPILE_FAIL=0; c_CRASH=0; c_TIMEOUT=0; c_SKIP=0

# A run over the whole corpus is minutes of silence otherwise, and silence
# looks the same as a hang. Every outcome goes through here: one line as it
# lands, a tally every 25, both on stderr so the TSV on stdout stays clean.
tally() {
  printf '  ---- %d/%d   pass %d  wrong %d  no_output %d  compile_fail %d  crash %d  timeout %d  skip %d\n' \
    "$seen" "$total" "$c_PASS" "$c_WRONG_ANSWER" "$c_NO_OUTPUT" \
    "$c_COMPILE_FAIL" "$c_CRASH" "$c_TIMEOUT" "$c_SKIP" >&2
}

record() {  # record <test> <status> <detail>
  local t="$1" st="$2" det="$3"
  printf '%s\t%s\t%s\n' "$t" "$st" "${det//$'\t'/ }" >> "$OUT"
  case "$st" in
    PASS)         c_PASS=$((c_PASS+1)) ;;
    WRONG_ANSWER) c_WRONG_ANSWER=$((c_WRONG_ANSWER+1)) ;;
    NO_OUTPUT)    c_NO_OUTPUT=$((c_NO_OUTPUT+1)) ;;
    COMPILE_FAIL) c_COMPILE_FAIL=$((c_COMPILE_FAIL+1)) ;;
    CRASH)        c_CRASH=$((c_CRASH+1)) ;;
    TIMEOUT)      c_TIMEOUT=$((c_TIMEOUT+1)) ;;
    SKIP)         c_SKIP=$((c_SKIP+1)) ;;
  esac
  seen=$((seen+1))
  printf '  %-18s %-13s %s\n' "$t" "$st" "$(printf '%s' "$det" | cut -c1-54)" >&2
  [[ $((seen % 25)) -eq 0 ]] && tally
  return 0
}

for f in "$SRC"/*.hx; do
  base="$(basename "$f" .hx)"
  [[ "$LIMIT" != 0 && $seen -ge $LIMIT ]] && break
  grep -q "extends unit.Test" "$f" || continue

  # A test that names a target-language package is asserting about that
  # target's semantics, not about Haxe. Out of scope: rayzor is its own
  # target and will never satisfy it. Both spellings count -- an `import
  # cpp.Pointer` and a bare `cpp.Reference<T>` in a type position say the
  # same thing, and only the first is an import.
  #
  # Conditional compilation is NOT a filter: `#if cpp ... #else ... #end`
  # still has a branch that is ours, so judging a file by its `#if`
  # directives would drop tests that legitimately apply to us.
  pkg=$( { grep -oE "^[[:space:]]*(import|using)[[:space:]]+($TARGET_PKGS)\.[A-Za-z0-9_.]*" "$f" \
             | awk '{print $2}' | sed 's/\.$//'
           grep -oE "\b($TARGET_PKGS)\.[A-Z][A-Za-z0-9_]*" "$f"
         } | sort -u | tr '\n' ' ')
  if [[ -n "$pkg" ]]; then
    record "$base" SKIP "targets ${pkg% }"; continue
  fi


  d="$WORK/run/$base"; rm -rf "$d"; mkdir -p "$d/unit/issues" "$d/unit" "$d/utest"
  # The corpus is not self-contained: tests reference siblings that upstream
  # ships beside them -- HelperMacros, MyClass, MyEnum, and the macros under
  # issues/misc. Without them a test fails on a missing type, which reads as
  # a resolution defect in the compiler rather than a file we did not provide.
  # Copied first so our shims below win where the names collide.
  cp "$SRC"/../*.hx "$d/unit/" 2>/dev/null || true
  if [[ -d "$SRC/misc" ]]; then
    mkdir -p "$d/unit/issues/misc"
    cp "$SRC"/misc/*.hx "$d/unit/issues/misc/" 2>/dev/null || true
  fi
  cp "$HERE/shims/unit/Test.hx" "$d/unit/Test.hx"
  cp "$HERE/shims/unit/ConfCheck.hx" "$d/unit/ConfCheck.hx"
  cp "$HERE/shims/utest/Assert.hx" "$d/utest/Assert.hx"

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
    record "$base" SKIP "no test method live for this target"; continue
  elif [[ $gen -ne 0 ]]; then
    record "$base" SKIP "harness could not inject main"; continue
  fi

  # Run as a project, not a lone file. `rayzor run <file>` compiles that file
  # and the standard library and nothing else: a class the test inherits from
  # -- `unit.Test`, which every one of them extends -- is known well enough to
  # declare a forward reference and never compiled, so the reference becomes a
  # trap stub and the test dies on SIGTRAP with nothing said. A manifest with a
  # class path compiles the siblings too.
  printf '[project]\nname = "conformance"\nentry = "unit/issues/%s.hx"\n\n[build]\nclass-paths = ["."]\n' \
    "$base" > "$d/rayzor.toml"
  # Bounded. A test that now compiles can also loop forever, and without a
  # limit one of those stalls the whole corpus -- in CI, until the job is
  # killed hours later. `timeout` is not on every platform we run this on, so
  # the watchdog is python3, which the harness already needs.
  out=$( cd "$d" && python3 "$HERE/runwith.py" "$TIMEOUT" "$RAYZOR" run --release --no-cache 2>&1 )
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
      124) status="TIMEOUT"; det="exceeded ${TIMEOUT}s" ;;
      132|134|139|133) status="CRASH" ;;
      *) status="COMPILE_FAIL" ;;
    esac
    # "No main function found" is only OUR fault if we failed to write one.
    # When the generated file plainly has a main and the compiler cannot see
    # it, that is a compiler defect -- in every case observed so far the RD
    # parser rejected a construct, fell back, and the fallback lost the class
    # body. Calling that a skip charges our own gaps to the harness and hides
    # them from the score.
    if [[ "$det" == *"No main function found"* ]] \
       && ! grep -q "static function main" "$d/unit/issues/$base.hx"; then
      record "$base" SKIP "harness could not inject main"
    elif [[ "$det" == *"'utest'"* ]]; then
      record "$base" SKIP "uses utest directly"
    elif [[ "$det" == *"No main function found"* ]]; then
      record "$base" COMPILE_FAIL "main present but not found: $(printf '%s' "$out" \
        | grep -oE "expected [^;]*" | head -1)"
    else
      record "$base" "$status" "$det"
    fi
  elif printf '%s' "$out" | grep -q '^CONFORMANCE_OK'; then
    record "$base" PASS "$(printf '%s' "$out" | grep -o 'CONFORMANCE_OK.*')"
  elif printf '%s' "$out" | grep -q 'FAILCHECK\|CONFORMANCE_BAD'; then
    record "$base" WRONG_ANSWER "$(printf '%s' "$out" | grep -m1 'FAILCHECK' | cut -c1-90)"
  else
    record "$base" NO_OUTPUT "$(printf '%s' "$out" | tail -1 | cut -c1-90)"
  fi
done

tally
scored=$((c_PASS + c_WRONG_ANSWER + c_NO_OUTPUT + c_COMPILE_FAIL + c_CRASH + c_TIMEOUT))
echo "scored $scored  pass $c_PASS  fail $((scored - c_PASS))  (skipped $c_SKIP)"
echo "report: $OUT"
