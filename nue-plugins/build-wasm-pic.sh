#!/usr/bin/env bash
#
# Build nue-plugins as a PIC wasm SIDE-MODULE — the portable Q8 KV-cache
# capability that the rayzor wasm-linker merges into the runtime (vs the
# native-only dylib that forces wasm onto the 4GB F32 cache).
#
# Why this is not just `cargo build --target wasm32-...`:
#   - A side-module must be position-independent (`-C relocation-model=pic
#     --experimental-pic -shared`) so its data/stack relocate via __memory_base
#     / GOT instead of colliding 1:1 with runtime-wasm's fixed 1MiB layout.
#   - PIC needs core/std recompiled with -fPIC → `-Z build-std`. The shipped
#     wasm std is NOT PIC.
#   - `-shared` is incompatible with the wasm32-wasip1-threads target's forced
#     `--export-memory`, which lives in the target's pre-link-args (unreachable
#     via `-C link-arg`). So we regenerate the target spec with that flag
#     stripped. We keep the threads lineage (+atomics/shared-memory) so the ABI
#     matches runtime-wasm's shared memory.
#   - The plugin is no_std on wasm (see lib.rs) so it links ONLY core+alloc and
#     drags in no wasi-libc (which ships non-PIC and would break the link).
#
# Output: target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm — a module
# with a `dylink.0` section, env.__memory_base/__table_base/__stack_pointer/
# __indirect_function_table imports, GOT.func.*/GOT.mem.* relocations, the 6
# env.rayzor_plugin_tensor_* seam imports, and the Q8 kernel + plugin_describe
# exports. The wasm-linker's (forthcoming) GOT-resolution pass consumes it.
#
# Requires the nightly toolchain + rust-src (`rustup component add rust-src
# --toolchain nightly`).
set -euo pipefail
cd "$(dirname "$0")"

SPEC=wasm32-wasip1-threads-pic.json

# Regenerate the stripped target spec from the CURRENT toolchain each run
# (rather than committing a frozen copy that drifts across rustc versions).
rustc -Z unstable-options --print target-spec-json --target wasm32-wasip1-threads \
  > /tmp/rayzor_threads_spec.json
python3 - "$SPEC" <<'PY'
import json, sys
spec = json.load(open('/tmp/rayzor_threads_spec.json'))
for flavor, args in spec.get('pre-link-args', {}).items():
    spec['pre-link-args'][flavor] = [a for a in args if 'export-memory' not in a]
json.dump(spec, open(sys.argv[1], 'w'), indent=2)
PY

SC="$(rustc --print sysroot)/lib/rustlib/wasm32-wasip1-threads/lib/self-contained"

RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C relocation-model=pic -L $SC \
-C link-arg=--experimental-pic -C link-arg=-shared \
-C link-arg=--unresolved-symbols=import-dynamic" \
  cargo +nightly build --release \
    --target "./$SPEC" \
    -Z build-std=std,panic_abort \
    -Z json-target-spec

echo "built: target/wasm32-wasip1-threads-pic/release/nue_plugins.wasm"
