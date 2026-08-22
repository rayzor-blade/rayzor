#!/usr/bin/env bash
#
# Compile every benchmark through the whole-module LLVM path and report what
# fails to compile or verify.
#
# `rayzor run --llvm` does NOT do this. It compiles lazily, so a function the
# program never calls is never lowered and never verified -- which is why the
# tree benchmark runs clean there while the LLVM tier cannot build the module
# it lives in. The benchmark runner's rayzor-llvm target compiles everything,
# and that is the difference between "my program works" and "the tier works".
#
# This matters beyond a missing row. When the module fails to compile, the
# tiered targets cannot promote and silently keep running on Cranelift, so
# every rayzor row in the table is a Cranelift row wearing another name.
#
#   ./check_llvm_tier.sh              every benchmark
#   ./check_llvm_tier.sh binarytrees  one
#
# Exits non-zero if any benchmark's module fails.

set -uo pipefail
cd "$(dirname "$0")/.."

BENCHES=("$@")
if [ ${#BENCHES[@]} -eq 0 ]; then
    BENCHES=(binarytrees deltablue fibonacci mandelbrot nbody)
fi

RUNNER=(cargo run --release -q -p compiler --example benchmark_runner --)
FAILED=0

for bench in "${BENCHES[@]}"; do
    printf '%-14s ' "$bench"
    out=$(RAYZOR_LLVM_VERIFY_SURVEY=1 "${RUNNER[@]}" "$bench" -t rayzor-llvm 2>&1)
    if names=$(printf '%s' "$out" | grep -aoE "verification failed for [0-9]+ function\(s\)[^:]*: .*"); then
        echo "FAIL"
        printf '%s\n' "  $names" | head -2
        FAILED=$((FAILED + 1))
    elif printf '%s' "$out" | grep -qa "\[FAIL\]"; then
        echo "FAIL"
        printf '%s\n' "$out" | grep -a "\[FAIL\]" | head -1 | sed 's/^/  /'
        FAILED=$((FAILED + 1))
    else
        exec_ms=$(printf '%s' "$out" | grep -aoE "Execute: [0-9.]+ms" | head -1)
        echo "ok    ${exec_ms:-}"
    fi
done

echo
if [ "$FAILED" -ne 0 ]; then
    echo "$FAILED benchmark(s) cannot build under LLVM; tiered rows are Cranelift"
    exit 1
fi
echo "LLVM tier builds every benchmark"
