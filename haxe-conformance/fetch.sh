#!/usr/bin/env bash
# Fetch the upstream Haxe regression corpus that run.sh scores against.
#
# The corpus is NOT vendored. Haxe's LICENSE says a file with no license header
# outside `std/` and `libs/` is GPL-2.0-or-later, and none of the 1165 issue
# files carries a header -- so copying them into this Apache-2.0 repository
# would mix incompatible licences. Fetching at run time keeps them out of our
# distribution while still pinning exactly which revision the numbers describe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$HERE/corpus}"
PIN="${PIN:-$(cat "$HERE/corpus.pin")}"
REMOTE="${REMOTE:-https://github.com/HaxeFoundation/haxe.git}"
# Three suites, matching what a Haxe VM is normally held to: the language
# itself, the sys stdlib, and threading. Scoring only tests/unit left the
# other two unmeasured -- not passing, not failing, simply never run.
SPARSE="${SPARSE:-tests/unit/src tests/sys tests/threads}"

if [[ -d "$DEST/.git" ]]; then
  # Re-apply the sparse pattern even when the pin matches: a previous clone may
  # have been made against a narrower path and be missing scripthost/, misc/,
  # or -- since sys and threads were added -- either of those two suites.
  git -C "$DEST" sparse-checkout set $SPARSE
  if [[ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null)" == "$PIN" ]]; then
    echo "corpus already at $PIN"; exit 0
  fi
fi

rm -rf "$DEST"
# Blobless + sparse: the full history of the Haxe compiler is ~200x the size of
# the directory we need.
git clone --filter=blob:none --sparse --no-checkout "$REMOTE" "$DEST"
# The issue corpus spans more than unit/: some issues import scripthost.*,
# misc.*, and root-package helpers that live beside unit/ under tests/unit/src.
git -C "$DEST" sparse-checkout set $SPARSE
git -C "$DEST" checkout --detach "$PIN"

issues=$(find "$DEST/tests/unit/src/unit/issues" -maxdepth 1 -name "*.hx" 2>/dev/null | wc -l | tr -d " ")
feat=$(find "$DEST/tests/unit/src/unit" -maxdepth 1 -name "*.hx" 2>/dev/null | wc -l | tr -d " ")
sys=$(find "$DEST/tests/sys/src" -name "*.hx" 2>/dev/null | wc -l | tr -d " ")
thr=$(find "$DEST/tests/threads/src" -name "*.hx" 2>/dev/null | wc -l | tr -d " ")
echo "corpus at $PIN  ($issues issue, $feat feature, $sys sys, $thr thread files)"
