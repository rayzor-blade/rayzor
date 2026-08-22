#!/usr/bin/env bash
#
# Build the CLI so that it carries a standard library matching itself.
#
# It takes two steps, and the order is not arbitrary. The generator links the
# compiler, so the compiler cannot produce the artifact it embeds in a single
# pass: step one builds the compiler with your changes and lowers the library
# with it, step two rebuilds the CLI, which picks the new artifact up because
# `build.rs` watches the file.
#
# Skipping this after a `compiler/src` change does not fail — the carried
# library is keyed on a hash of the compiler, so a stale one is quietly ignored
# and the library is lowered from source instead. That is slower, and it is a
# different code path, so a run can behave differently for reasons that have
# nothing to do with the change under test. This script exists so that cannot
# happen by omission.
#
#   scripts/build.sh              CLI only
#   scripts/build.sh --plugins    CLI plus the tensor and nue plugin dylibs
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CARGO_INCREMENTAL=0

# BSD mktemp treats -t as a bare prefix; GNU requires the template to end in
# XXXXXX and errors out otherwise, so the bare form fails on every Linux box.
log="$(mktemp -t rayzor-build.XXXXXX)"
trap 'rm -f "$log"' EXIT

echo ">> lowering the standard library with the current compiler"
cargo run --release -p snapshot-gen

echo ">> building the CLI"
cargo build --release -p rayzor --bin rayzor 2>&1 | tee "$log"

# `build.rs` reports an unusable artifact through cargo warnings, which scroll
# past in a long build. Treat them as the failure they are.
if grep -qE "no standard library snapshot at|was built by compiler|records no compiler id" "$log"; then
    echo >&2
    echo "!! the CLI does not carry a standard library matching it." >&2
    echo "   Re-run this script; if it repeats, the generator is failing." >&2
    exit 1
fi

if [[ "${1:-}" == "--plugins" ]]; then
    echo ">> building the plugin dylibs"
    cargo build --release -p rayzor-tensors -p nue-plugins
fi

echo ">> ready: target/release/rayzor"
