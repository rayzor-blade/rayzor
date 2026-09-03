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
# The issue files are regressions for individual upstream bugs. The feature
# suites one directory up -- TestBasetypes, TestCasts, TestGeneric, TestMatch,
# TestOps, TestReflect, TestType, ... -- are the systematic coverage, and
# scoring only the issues left every one of them unmeasured. Same base class,
# same shape, so they go through the identical path. FEATURES=0 to exclude.
FEATURE_SRC="${FEATURE_SRC:-$HERE/corpus/tests/unit/src/unit}"
FEATURES="${FEATURES:-1}"
# Not cases: Test is the base class we shim, and the two Main files are the
# upstream runner's entry points.
NOT_A_CASE=" Test TestMain TestMainNow Main ThreadTestBase "

# The other two suites a Haxe VM is normally held to. They are written against
# utest rather than unit.Test -- different base class, same idea -- and their
# cases are found and scored the same way.
SYS_SRC="${SYS_SRC:-$HERE/corpus/tests/sys/src}"
THREADS_SRC="${THREADS_SRC:-$HERE/corpus/tests/threads/src}"
SUITES="${SUITES:-unit,sys,threads}"

# What declares a case, in one place. unit.Test is the language suite's base;
# utest.Test is sys and threads'; TestCommandBase and ThreadTestBase are
# intermediate bases that cases inherit from without naming utest directly.
CASE_RE="extends[[:space:]]+(unit\\.)?Test\\b|extends[[:space:]]+utest\\.Test\\b|extends[[:space:]]+(TestCommandBase|ThreadTestBase)\\b"

suite_on() { case ",$SUITES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
WORK="${WORK:-${TMPDIR:-/tmp}/rayzor_conformance}"
LIMIT="${LIMIT:-0}"
TIMEOUT="${TIMEOUT:-60}"
OUT="${OUT:-$WORK/report.tsv}"
RAYZOR="${RAYZOR:-$REPO/target/release/rayzor}"
JOBS="${JOBS:-0}"
PRESET="${PRESET:-application}"
# LLVM=1 scores the corpus through the LLVM tier instead of letting Beadie/OSR
# promote only what a short test makes hot. Every preset sets
# auto_upgrade_to_llvm_after_main_entry:false, so a default run never calls
# upgrade_to_llvm at all -- the score is a Cranelift score. LLVM is not opt-in
# in production, so it should be measurable here.
LLVM="${LLVM:-0}"
# CRANELIFT=1 pins the corpus to Cranelift: start compiled at Baseline (P1) and
# never promote. The default run starts interpreted and climbs, so a test short
# enough never to get hot is scored mostly on the interpreter -- which is a
# third behaviour again, not "Cranelift". Tiers 0-3 are interpreter/Cranelift
# levels; LLVM is the separate upgrade, so this and LLVM=1 are disjoint.
CRANELIFT="${CRANELIFT:-0}"
LOGS="${LOGS:-$WORK/logs}"

# Keep enough parallelism to use the machine without letting a corpus run
# launch hundreds of compiler processes at once. JOBS=1 retains the old
# serial behaviour; JOBS=N makes performance experiments reproducible.
if [[ "$JOBS" == 0 ]]; then
  JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
  [[ "$JOBS" -gt 8 ]] && JOBS=8
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "JOBS must be a positive integer (got: $JOBS)" >&2
  exit 2
}

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

RUN_ROOT="$WORK/run"
SHARED="$WORK/shared-$$"
RESULTS="$WORK/results-$$"
# Hundreds of these tests crash, and a crash that writes a core costs far more
# than the crash itself: the kernel streams the whole image -- ~93 MB resident
# here -- and on a runner whose core_pattern pipes to a collector (apport, and
# GitHub's images ship it) that is paid per crash, on a 4-vCPU box, with four
# tests running at once. That is the difference between a crash reported in a
# second and one the 25s watchdog kills first.
#
# It shows up as a run where CRASH is exactly 0 and TIMEOUT is ~240 instead of
# ~35 -- the same tests, reclassified, and roughly an hour of extra wall clock.
# Observed on both AMD and Intel runners, so it is not the CPU.
ulimit -c 0 2>/dev/null || true

# A previous complete run must not authorize this run when WORK is reused for
# a pilot, a changed preset, or an interrupted retry.
rm -f "$WORK/COMPLETE"
mkdir -p "$RUN_ROOT" "$SHARED/unit/issues/misc" "$SHARED/utest" "$RESULTS" "$(dirname "$OUT")" "$LOGS" || {
  echo "cannot create conformance work/report directories" >&2
  exit 2
}
# WORK is reusable locally. Do not let raw output from an earlier run masquerade
# as evidence for this one when a test now takes a different path.
find "$LOGS" -maxdepth 1 -type f -name '*.log' -delete 2>/dev/null || true
if ! printf 'issue\tstatus\tdetail\tsuite\n' > "$OUT"; then
  echo "cannot write conformance report: $OUT" >&2
  exit 2
fi

# These files are identical for every issue. Keeping one class-path tree avoids
# copying ~74 sibling fixtures per test (roughly 76k copies for a full run).
# Tests only write beneath RUN_ROOT; SHARED is read-only input to every worker.
#
# The corpus is not self-contained in one directory: issues reference siblings
# under unit/ (HelperMacros, MyClass, ...), nested helper packages under
# unit/issues/misc/ (issue12259, issue12672, issue8543), and packages that live
# beside unit/ entirely -- scripthost/, misc/, and a few root-package helpers.
# A module any of these provides must be on the class path or the test fails as
# if the name never existed.
SRC_ROOT="$(cd "$SRC/../.." && pwd)"   # .../tests/unit/src
cp "$SRC"/../*.hx "$SHARED/unit/" 2>/dev/null || true
if [[ -d "$SRC/misc" ]]; then
  cp -R "$SRC"/misc/. "$SHARED/unit/issues/misc/" 2>/dev/null || true
fi
cp "$SRC_ROOT"/*.hx "$SHARED/" 2>/dev/null || true
[[ -d "$SRC_ROOT/scripthost" ]] && cp -R "$SRC_ROOT"/scripthost "$SHARED/" 2>/dev/null || true
[[ -d "$SRC_ROOT/misc" ]] && cp -R "$SRC_ROOT"/misc "$SHARED/" 2>/dev/null || true
# sys cases call into sibling helpers that upstream ships beside them --
# ExitCode, FileNames, UnicodeSequences, UtilityProcess -- and threads cases
# all extend ThreadTestBase. Main is excluded from both: it is upstream's own
# entry point, and a second main on the class path competes with the injected
# one.
if [[ -d "$SYS_SRC" ]]; then
  find "$SYS_SRC" -maxdepth 1 -name '*.hx' ! -name 'Main.hx' \
    -exec cp {} "$SHARED/" \; 2>/dev/null || true
fi
if [[ -d "$THREADS_SRC" ]]; then
  cp "$THREADS_SRC/ThreadTestBase.hx" "$SHARED/" 2>/dev/null || true
fi
cp "$HERE/shims/unit/Test.hx" "$SHARED/unit/Test.hx"
cp "$HERE/shims/unit/ConfCheck.hx" "$SHARED/unit/ConfCheck.hx"
cp "$HERE/shims/utest/Assert.hx" "$SHARED/utest/Assert.hx"
cp "$HERE/shims/utest/Test.hx" "$SHARED/utest/Test.hx"

# How many files are candidates, so progress has a denominator.
# Both spellings. A file in `package unit.issues` may name the base class
# `Test` or `unit.Test`, and matching only the qualified form dropped 714 of
# the 1165 issue files before they were ever compiled -- not skipped, not
# reported, just absent from the denominator.
# One definition of what counts as a case, used by both the denominator and
# the dispatch loop -- computing them separately is how a run ends up scoring
# a different set than it counted.
list_cases() {
  local p b
  {
    if suite_on unit; then
      ls "$SRC"/*.hx 2>/dev/null
      if [[ "$FEATURES" != 0 && -d "$FEATURE_SRC" ]]; then
        ls "$FEATURE_SRC"/*.hx 2>/dev/null
      fi
    fi
    if suite_on sys && [[ -d "$SYS_SRC" ]]; then
      find "$SYS_SRC" -name '*.hx' 2>/dev/null | sort
    fi
    if suite_on threads && [[ -d "$THREADS_SRC" ]]; then
      find "$THREADS_SRC" -name '*.hx' 2>/dev/null | sort
    fi
  } | while IFS= read -r p; do
    b="$(basename "$p" .hx)"
    case "$NOT_A_CASE" in *" $b "*) continue ;; esac
    grep -qE "$CASE_RE" "$p" && printf '%s\n' "$p"
  done
}

total=$(list_cases | wc -l | tr -d ' ')
[[ "$LIMIT" != 0 && $LIMIT -lt $total ]] && total=$LIMIT

seen=0
queued=0
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

record() {  # record <test> <status> <detail> <suite>
  local t="$1" st="$2" det="$3" sui="${4:-unit}"
  printf '%s\t%s\t%s\t%s\n' "$t" "$st" "${det//$'\t'/ }" "$sui" >> "$OUT"
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

emit_result() { # emit_result <path> <test> <status> <detail>
  local result="$1" t="$2" st="$3" det="$4"
  # Which suite the case came from. Without it the sys and threads results
  # disappear into a 1200-row aggregate and nobody can see they score zero.
  printf '%s\t%s\t%s\t%s\n' "$t" "$st" "${det//$'\t'/ }" "${suite:-unit}" > "$result"
}

process_one() { # process_one <source> <result-file>
  local f="$1" result="$2"
  local base pkg d gen out code det status suite
  base="$(basename "$f" .hx)"
  # Classified before anything can emit a row: the target-package skip below
  # returns early, and a result written without a suite lands in a phantom one.
  case "$f" in
    "$SYS_SRC"/*)     suite=sys ;;
    "$THREADS_SRC"/*) suite=threads ;;
    "$SRC"/*)         suite=issues ;;
    *)                suite=features ;;
  esac

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
    emit_result "$result" "$base" SKIP "targets ${pkg% }"
    return
  fi

  # Stage under the package the file declares rather than a fixed one: the
  # issue files are `package unit.issues`, the feature suites beside them are
  # `package unit`, and a class staged in the wrong directory does not resolve.
  local rel
  rel="$(grep -m1 -oE '^[[:space:]]*package[[:space:]]+[A-Za-z0-9_.]+' "$f" \
          | awk '{print $2}' | tr '.' '/')"
  # The sys suite declares no package at all, so an empty `rel` is the root of
  # the staging directory -- not a missing value to default away.
  d="$RUN_ROOT/$base"; rm -rf "$d"; mkdir -p "$d${rel:+/$rel}"
  # Haxe applies an `import.hx` at a class-path root to every module beneath
  # it. The threads suite puts its utest imports there, which is why its cases
  # call isTrue/same/pass unqualified. Staged per-test rather than into SHARED:
  # a wildcard static import visible to the whole corpus would change name
  # resolution for the other 1200 cases.
  case "$f" in
    "$THREADS_SRC"/*) cp "$THREADS_SRC/import.hx" "$d/" 2>/dev/null || true ;;
    "$SYS_SRC"/*)     cp "$SYS_SRC/import.hx" "$d/" 2>/dev/null || true ;;
  esac
  # The corpus is not self-contained: tests reference siblings that upstream
  # ships beside them -- HelperMacros, MyClass, MyEnum, and the macros under
  # issues/misc. Without them a test fails on a missing type, which reads as
  # a resolution defect in the compiler rather than a file we did not provide.
  # They and the harness shims live in SHARED, the second class path below.

  # Inject main() as the last member of the class; awk -v cannot carry newlines.
  # Exits 3 when the class declares no test method that is live for us.
  python3 - "$f" "$d${rel:+/$rel}/$base.hx" "$base" "$TARGET_PKGS" <<'PYGEN'
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
    emit_result "$result" "$base" SKIP "no test method live for this target"
    return
  elif [[ $gen -ne 0 ]]; then
    emit_result "$result" "$base" SKIP "harness could not inject main"
    return
  fi

  # Run as a project, not a lone file. `rayzor run <file>` compiles that file
  # and the standard library and nothing else: a class the test inherits from
  # -- `unit.Test`, which every one of them extends -- is known well enough to
  # declare a forward reference and never compiled, so the reference becomes a
  # trap stub and the test dies on SIGTRAP with nothing said. A manifest with a
  # class path compiles the siblings too.
  printf '[project]\nname = "conformance"\nentry = "%s%s.hx"\n\n[build]\nclass-paths = [".", "%s"]\n' \
    "${rel:+$rel/}" "$base" "$SHARED" > "$d/rayzor.toml"
  # Bounded. A test that now compiles can also loop forever, and without a
  # limit one of those stalls the whole corpus -- in CI, until the job is
  # killed hours later. `timeout` is not on every platform we run this on, so
  # the watchdog is python3, which the harness already needs.
  # Make the execution mode explicit. Application preserves the production
  # measurement by default; the preset remains overridable for backend probes.
  # Written without an array: `set -u` is on and bash 3.2, which is what macOS
  # ships, treats "${empty[@]}" as an unbound variable -- which fails EVERY
  # invocation and scores the whole corpus COMPILE_FAIL.
  if [[ "$LLVM" != 0 ]]; then
    out=$( cd "$d" && python3 "$HERE/runwith.py" "$TIMEOUT" "$RAYZOR" run --release --no-cache --preset "$PRESET" --llvm 2>&1 )
  elif [[ "$CRANELIFT" != 0 ]]; then
    out=$( cd "$d" && python3 "$HERE/runwith.py" "$TIMEOUT" "$RAYZOR" run --release --no-cache --preset "$PRESET" \
             --tier 1 --tier-start-interpreted false --tier-promotion false 2>&1 )
  else
    out=$( cd "$d" && python3 "$HERE/runwith.py" "$TIMEOUT" "$RAYZOR" run --release --no-cache --preset "$PRESET" 2>&1 )
  fi
  code=$?
  # The compact TSV is for scoring, not diagnosis. Preserve the complete output
  # for every issue so a CI-only crash/no-output result includes its tier event,
  # verifier error, and runner banner instead of only a truncated final line.
  local log="$LOGS/$base.log"
  printf '%s\n' "$out" > "$log"

  # Everything below reads the LOG, never `$out` through a pipe. With
  # `set -o pipefail`, `printf '%s' "$out" | grep -q PATTERN` reports FAILURE
  # when grep matches EARLY: grep -q exits at the first hit and closes the pipe,
  # the producer's next write takes SIGPIPE, and pipefail adopts that 141 as the
  # pipeline's status. A passing test prints CONFORMANCE_OK on line 1, so the
  # longer its remaining output, the likelier its own success killed the check
  # that was looking for it -- which is why this only bit in CI, where
  # RAYZOR_TIER_TRACE_STARTUP adds ~34 lines after that first one. Three cases
  # scored NO_OUTPUT against logs whose first line was CONFORMANCE_OK.
  # Reading the file has no producer to kill, and it makes the row and the log
  # come from one source, so they cannot disagree again.

  # A test that reported a failed assertion and THEN died computed a wrong
  # answer; the crash is downstream of it, usually while rendering the very
  # value that was wrong. Scoring it as a crash hides the wrong answer and
  # inflates the crash count.
  if [[ $code -ne 0 ]] && grep -q '^FAILCHECK' "$log"; then
    det=$(grep -m1 '^FAILVALUES' "$log" | cut -c1-90)
    [[ -z "$det" ]] && det=$(grep -m1 '^FAILCHECK' "$log" | cut -c1-90)
    emit_result "$result" "$base" WRONG_ANSWER "${det} (then exit $code)"
  elif [[ $code -ne 0 ]]; then
    det=$(grep -oE '\[E[0-9]+\][^"]{0,80}' "$log" | head -1)
    # A named uncompiled function is the most actionable thing a run can say:
    # it points at the exact construct the compiler could not build. Prefer it
    # over the generic exit code, which is what every one of these used to be.
    [[ -z "$det" ]] && det=$(grep -oE 'rayzor: `[^`]+` was never compiled' "$log" \
                             | head -1 | sed -E 's/rayzor: `(.*)` was never compiled/uncompiled \1/')
    [[ -z "$det" ]] && det=$(grep -iE 'error|panic|signal' "$log" | head -1 | cut -c1-90)
    [[ -z "$det" ]] && det="exit $code"
    # Any death by signal is a crash: the watchdog reports them the way a shell
    # does, as 128+signo. Naming individual numbers filed the ones nobody had
    # seen yet as compile failures -- SIGBUS is 138 on macOS and 135 on Linux,
    # so the same defect was a crash on one platform and a clean compile error
    # on the other, which flatters whichever platform is being quoted.
    case "$code" in
      124) status="TIMEOUT"; det="exceeded ${TIMEOUT}s" ;;
      *)
        if (( code >= 128 )); then
          status="CRASH"
          [[ "$det" == "exit $code" ]] && det="exit $code (signal $((code - 128)))"
        else
          status="COMPILE_FAIL"
        fi
        ;;
    esac
    # "No main function found" is only OUR fault if we failed to write one.
    # When the generated file plainly has a main and the compiler cannot see
    # it, that is a compiler defect -- in every case observed so far the RD
    # parser rejected a construct, fell back, and the fallback lost the class
    # body. Calling that a skip charges our own gaps to the harness and hides
    # them from the score.
    if [[ "$det" == *"No main function found"* ]] \
       && ! grep -q "static function main" "$d${rel:+/$rel}/$base.hx"; then
      emit_result "$result" "$base" SKIP "harness could not inject main"
    elif [[ "$det" == *"'utest'"* ]]; then
      emit_result "$result" "$base" SKIP "uses utest directly"
    elif [[ "$det" == *"No main function found"* ]]; then
      emit_result "$result" "$base" COMPILE_FAIL "main present but not found: $(printf '%s' "$out" \
        | grep -oE "expected [^;]*" | head -1)"
    else
      emit_result "$result" "$base" "$status" "$det"
    fi
  elif grep -q '^CONFORMANCE_OK' "$log"; then
    emit_result "$result" "$base" PASS "$(grep -o 'CONFORMANCE_OK.*' "$log" | head -1)"
  elif grep -q 'FAILCHECK\|CONFORMANCE_BAD' "$log"; then
    det=$(grep -m1 '^FAILVALUES' "$log" | cut -c1-90)
    [[ -z "$det" ]] && det=$(grep -m1 '^FAILCHECK' "$log" | cut -c1-90)
    emit_result "$result" "$base" WRONG_ANSWER "$det"
  else
    emit_result "$result" "$base" NO_OUTPUT "$(tail -1 "$log" | cut -c1-90)"
  fi
}

worker_pids=()
worker_results=()

reap_one() {
  local i pid result t st det sui
  while true; do
    for i in "${!worker_pids[@]}"; do
      pid="${worker_pids[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        result="${worker_results[$i]}"
        if [[ ! -f "$result" ]]; then
          echo "conformance worker $pid exited without a result" >&2
          exit 2
        fi
        IFS=$'\t' read -r t st det sui < "$result"
        record "$t" "$st" "$det" "$sui"
        unset 'worker_pids[$i]' 'worker_results[$i]'
        return
      fi
    done
    sleep 0.02
  done
}

while IFS= read -r f; do
  [[ "$LIMIT" != 0 && $queued -ge $LIMIT ]] && break
  base="$(basename "$f" .hx)"
  result="$RESULTS/$(printf '%05d' "$queued")-$base.tsv"
  process_one "$f" "$result" &
  worker_pids+=("$!")
  worker_results+=("$result")
  queued=$((queued+1))
  [[ ${#worker_pids[@]} -ge $JOBS ]] && reap_one
done < <(list_cases)
while [[ ${#worker_pids[@]} -gt 0 ]]; do
  reap_one
done

# Progress is reported in completion order, but the durable report must be
# stable so two runs can be diffed directly. Result names carry the source
# index, and shell glob order restores that order without serialising workers.
ordered_out="$OUT.ordered.$$"
printf 'issue\tstatus\tdetail\tsuite\n' > "$ordered_out"
for result in "$RESULTS"/*.tsv; do
  cat "$result" >> "$ordered_out"
done
mv "$ordered_out" "$OUT"

tally
scored=$((c_PASS + c_WRONG_ANSWER + c_NO_OUTPUT + c_COMPILE_FAIL + c_CRASH + c_TIMEOUT))
echo "scored $scored  pass $c_PASS  fail $((scored - c_PASS))  (skipped $c_SKIP)"
echo "report: $OUT"

# Written only after every test in a full corpus run has been recorded. A pilot
# is complete for its limit, but must not authorize a partial report as a score.
if [[ "$LIMIT" == 0 ]]; then
  : > "$WORK/COMPLETE"
fi
exit 0
