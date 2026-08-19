#!/usr/bin/env bash
#
# Fail if any string is released twice while running a program.
#
# Releasing a string twice frees its header twice, and the damage surfaces at
# some later allocation rather than at the release — so the crash names an
# innocent bystander and moves with the heap layout. That cost days once.
#
# This asks the runtime rather than the IR. `RZT_DBG_STRFREE` keeps released
# headers so a second release is recognised, which is exact: a static reading of
# the pass's placement over-reports, because two releases on branches that never
# run for the same allocation are correct and hard to tell apart in the CFG.
#
#   scripts/check_double_free.sh <program.hx> [-- args...]
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
rayzor="target/release/rayzor"
[[ -x "$rayzor" ]] || { echo "build first: scripts/build.sh" >&2; exit 2; }

program="${1:?usage: check_double_free.sh <program.hx> [-- args...]}"
shift || true

log="$(mktemp -t rayzor-double-free)"
trap 'rm -f "$log"' EXIT

RZT_DBG_STRFREE=1 "$rayzor" run "$program" --release --llvm --safety-warnings=off "$@" > "$log" 2>&1 || true

count=$(grep -ac 'DOUBLE FREE' "$log" || true)
if [[ "$count" -gt 0 ]]; then
    echo "$count string(s) released twice while running $program:" >&2
    grep -a 'DOUBLE FREE' "$log" | sort -u | head -20 >&2
    exit 1
fi
echo "no string released twice while running $program"
