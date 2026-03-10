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

# Clear blade cache
rm -f .rayzor/blade/cache/*.blade 2>/dev/null || true

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
    output=$(timeout 30 "$RAYZOR" run "$hx_file" 2>&1) || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        status="PASS"
        PASSED=$((PASSED + 1))
        printf "\033[32mPASS\033[0m\n"
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
