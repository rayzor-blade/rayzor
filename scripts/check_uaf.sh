#!/usr/bin/env bash
#
# Use-after-free oracle for the whole Haxe suite.
#
# The two oracles this repo had before are both LEAK-direction:
# check_composition_leak.sh compares peak footprint across iteration counts,
# and check_double_free.sh only sees HaxeString headers. Neither can observe a
# freed class instance, anon object or array header being read — the exact
# failure `insert_free` risks every time it decides something does not escape.
#
# macOS malloc can supply what is missing. MallocScribble fills freed blocks
# with 0x55 and MallocPreScribble fills fresh ones with 0xAA, so a read through
# a dangling pointer yields 0x5555... instead of the value that happened to
# survive. A program that passes normally and fails here read freed memory.
#
# KNOWN BLIND SPOT: a program whose output varies run to run is reported as
# non-reproducible rather than judged, and a use-after-free READ AT A BLOCK'S
# BASE produces exactly that -- the allocator's free-list link lives there and
# follows the address, so the corrupted value differs every run. Such a defect
# is skipped, not caught. The guard is still worth having: without it ten of
# twelve reports were the test's own nondeterminism, which buries the real
# ones. What it buys in precision it costs in recall, and the fixture is
# deliberately written to read PAST the header so the detector itself is
# testable.
#
#   ./check_uaf.sh                     whole suite
#   ./check_uaf.sh test_array_iterator single test
#   RZT_BENIGN_FREE=1 ./check_uaf.sh   with an experimental gate on
#
# Exits non-zero if any test behaves differently under poisoning.

set -uo pipefail
cd "$(dirname "$0")/.."

RAYZOR="$PWD/target/release/rayzor"
TESTS_DIR="compiler/tests/haxe"
[ -x "$RAYZOR" ] || { echo "no $RAYZOR — run scripts/build.sh"; exit 2; }

if [ $# -gt 0 ]; then
    FILES=()
    for n in "$@"; do FILES+=("$TESTS_DIR/${n%.hx}.hx"); done
else
    FILES=("$TESTS_DIR"/*.hx)   # the fixture lives in scripts/, not here
fi

# A run under poisoning is compared against the same program run without it, so
# a test that is already failing for its own reasons does not read as a UAF.
run() {
    local poison=$1 file=$2 out
    if [ "$poison" = yes ]; then
        out=$(MallocScribble=1 MallocPreScribble=1 \
              "$RAYZOR" run "$file" --llvm 2>&1)
    else
        out=$("$RAYZOR" run "$file" --llvm 2>&1)
    fi
    printf '%s\n--exit:%d' "$out" "$?"
}

# Prove the detector fires before trusting it to stay silent. uaf_fixture.hx
# frees a block and reads the payload back, so a working oracle must flag it.
# A green run means nothing if the check cannot come back red.
FIXTURE="$(dirname "$0")/uaf_fixture.hx"
if [ -f "$FIXTURE" ]; then
    run no "$FIXTURE" > /dev/null
    if [ "$(run no "$FIXTURE")" = "$(run yes "$FIXTURE")" ]; then
        echo "SELF-TEST FAILED: the detector did not flag a deliberate" >&2
        echo "use-after-free ($FIXTURE). Poisoning is not reaching the" >&2
        echo "program, so a silent result here proves nothing." >&2
        exit 2
    fi
    echo "self-test: detector fires on a known use-after-free"
else
    echo "SELF-TEST SKIPPED: $FIXTURE missing; results are unvalidated" >&2
fi

SUSPECT=0 FLAKY=0 CHECKED=0
for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .hx)
    CHECKED=$((CHECKED + 1))

    # The first run of a program COMPILES it and the rest read the cache, so
    # warnings and import errors appear only once. Warm the cache before
    # measuring or every diagnostic reads as a difference.
    run no "$f" > /dev/null

    # Several clean runs first. A program whose own output varies run to run --
    # hash iteration order, timings, thread interleaving -- cannot answer this
    # question, and reporting it as a memory bug buries the ones that can.
    # Two samples is not enough: a map with two iteration orders matches itself
    # half the time and then reads as a memory bug on the next line.
    clean=$(run no "$f")
    varied=no
    for _ in 1 2 3 4; do
        [ "$(run no "$f")" = "$clean" ] || { varied=yes; break; }
    done
    if [ "$varied" = yes ]; then
        FLAKY=$((FLAKY + 1))
        echo "--- $name: output is not reproducible; cannot judge"
        continue
    fi

    dirty=$(run yes "$f")
    if [ "$clean" = "$dirty" ]; then
        continue
    fi
    SUSPECT=$((SUSPECT + 1))
    echo "=== $name: reads memory that poisoning changed ==="
    echo "    0x55.. = freed and read again; 0xAA.. = allocated but never written"
    diff <(printf '%s' "$clean") <(printf '%s' "$dirty") | head -20
done

echo
echo "checked $CHECKED, suspect $SUSPECT, non-reproducible $FLAKY"
[ "$SUSPECT" -eq 0 ]
