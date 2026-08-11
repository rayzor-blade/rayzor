#!/usr/bin/env bash
# Run all .hx test files in compiler/tests/haxe/ through `rayzor run`
# Outputs results to test_results.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="$SCRIPT_DIR/target/release/rayzor"
TESTS_DIR="compiler/tests/haxe"
RESULTS_FILE="test_results.txt"

# Counters
PASSED=0
FAILED=0
CRASHED=0
TOTAL=0

# Build rayzor first
echo "Building rayzor..."
cargo build --release -p rayzor 2>&1 | tail -3

# Clear both BLADE and MIR caches. Stale MIR caches from a previous
# compiler revision can mask a fix (the test re-runs against the old
# MIR and shows the pre-fix failure), so we wipe both here.
rm -f .rayzor/blade/cache/*.blade 2>/dev/null || true
rm -f .rayzor/cache/*.mir.cache 2>/dev/null || true

echo ""
echo "Running all .hx tests in $TESTS_DIR"
echo "Results written to $RESULTS_FILE"
echo "========================================="

{
    echo "============================================"
    echo "  Rayzor Haxe Test Results"
    echo "  $(date)"
    echo "============================================"
    echo ""
} > "$RESULTS_FILE"

for hx_file in "$TESTS_DIR"/*.hx; do
    test_name="$(basename "$hx_file" .hx)"
    TOTAL=$((TOTAL + 1))

    printf "%-45s " "$test_name"

    output=""
    exit_code=0
    # Keep safety warnings on for tests that validate them
    safety_flag="--safety-warnings=off"
    case "$test_name" in
        test_safety_*) safety_flag="--safety-warnings=on" ;;
    esac
    output=$(timeout 30 "$RAYZOR" run $safety_flag "$hx_file" 2>&1) || exit_code=$?

    # A test can exit 0 and still have failed: the self-checking tests report
    # by printing. Only a marker at column 0 counts, so the indented "FAIL"
    # lines that test_rayzor_spec prints on purpose stay passing.
    self_reported_fail=0
    if printf '%s' "$output" | grep -qE '^(FAIL|FAILURES:)'; then
        self_reported_fail=1
    fi

    if [ $exit_code -eq 0 ] && [ $self_reported_fail -eq 0 ]; then
        status="PASS"
        PASSED=$((PASSED + 1))
        printf "\033[32mPASS\033[0m\n"
    elif [ $exit_code -eq 0 ]; then
        status="FAIL(self-reported)"
        FAILED=$((FAILED + 1))
        printf "\033[31mFAIL(self-reported)\033[0m\n"
    elif [ $exit_code -eq 124 ]; then
        status="TIMEOUT"
        CRASHED=$((CRASHED + 1))
        printf "\033[33mTIMEOUT\033[0m\n"
    elif [ $exit_code -eq 132 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 139 ]; then
        sig=""
        case $exit_code in
            132) sig="SIGILL" ;;
            134) sig="SIGABRT" ;;
            139) sig="SIGSEGV" ;;
        esac
        status="CRASH($exit_code/$sig)"
        CRASHED=$((CRASHED + 1))
        printf "\033[31mCRASH(%s)\033[0m\n" "$sig"
    else
        status="FAIL(exit=$exit_code)"
        FAILED=$((FAILED + 1))
        printf "\033[31mFAIL(exit=%d)\033[0m\n" "$exit_code"
    fi

    {
        echo "--- $test_name [$status] ---"
        echo "$output"
        echo ""
    } >> "$RESULTS_FILE"
done

# nue end-to-end smoke: compile AND run the llama-chat example against a
# small local model when one is present. The unit tests above never touch
# nue, so a compiler regression can pass 170/170 while inference is broken.
NUE_SMOKE_MODEL="${NUE_SMOKE_MODEL:-$HOME/models/qwen/Qwen2.5-0.5B-Instruct-Q6_K.gguf}"
if [ -f "$NUE_SMOKE_MODEL" ]; then
    TOTAL=$((TOTAL + 1))
    printf "%-50s" "nue_smoke_llama_chat"
    find . -name .rayzor -type d -exec rm -rf {} + 2>/dev/null
    smoke_out=$(cd nue/examples/llama-chat && RAYZOR_KV_Q8=1 timeout 300 \
        "$RAYZOR" run Main.hx --llvm --release --no-cache -- \
        "$NUE_SMOKE_MODEL" "Explain how a compiler works" 8 2048 0.0 2>&1)
    smoke_rc=$?
    if [ $smoke_rc -eq 0 ] && printf '%s' "$smoke_out" | grep -q "\[done\]"; then
        PASSED=$((PASSED + 1))
        printf "\033[32mPASS\033[0m\n"
        smoke_status="PASS"
    else
        CRASHED=$((CRASHED + 1))
        printf "\033[31mCRASH(exit=%d)\033[0m\n" "$smoke_rc"
        smoke_status="CRASH(exit=$smoke_rc)"
    fi
    {
        echo "--- nue_smoke_llama_chat [$smoke_status] ---"
        echo "$smoke_out" | tail -20
        echo ""
    } >> "$RESULTS_FILE"
fi

SUMMARY="
=========================================
  TOTAL:        $TOTAL
  PASSED:       $PASSED
  FAILED:       $FAILED
  CRASHED:      $CRASHED
========================================="

echo "$SUMMARY"
echo "$SUMMARY" >> "$RESULTS_FILE"
echo ""
echo "Full output: $RESULTS_FILE"
