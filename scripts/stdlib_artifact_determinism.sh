#!/usr/bin/env bash
# Acceptance oracle for a shippable stdlib snapshot.
#
# A stdlib module's lowered artifact must depend only on the stdlib, never on
# the program that happened to trigger its compilation. This compiles two
# deliberately different programs in separate working directories and compares
# the artifacts they produce for the modules they share. Any differing byte is
# a value from the producing context that leaked into the payload.
#
# Exit 0 when every shared module is byte-identical.

set -u

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RZ="${RZ:-$REPO/target/release/rayzor}"
WORK="${WORK:-$(mktemp -d)}"

if [ ! -x "$RZ" ]; then
    echo "rayzor binary not found at $RZ" >&2
    exit 2
fi

mkdir -p "$WORK/pa" "$WORK/pb"

cat > "$WORK/pa/A.hx" <<'EOF'
class A {
    static function main() {
        var s = "hi";
        var xs = [1, 2, 3];
        var t = 0;
        for (x in xs) t += x;
        trace(t + s.length + Std.int(Math.sqrt(9.0)));
    }
}
EOF

cat > "$WORK/pb/B.hx" <<'EOF'
class Helper {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
    public function doubled():Int { return v * 2; }
}
class B {
    static function extra(a:Int, b:Int):Int { return a + b; }
    static function main() {
        var f = Math.sqrt(25.0);
        var s = "longer string here";
        var h = new Helper(7);
        var xs = [4, 5, 6, 7];
        var t = extra(h.doubled(), Std.int(f));
        for (x in xs) t += x;
        trace(t + s.length);
    }
}
EOF

run_source() {
    local dir="$1" source="$2"
    if ! (cd "$dir" && rm -rf .rayzor && \
        RAYZOR_IGNORE_EMBEDDED_SNAPSHOT=1 "$RZ" run --plain "$source" >/dev/null 2>&1); then
        echo "FAIL: rayzor could not compile $dir/$source" >&2
        exit 2
    fi
}

# The oracle compares freshly produced artifacts. A valid release binary would
# otherwise restore its embedded snapshot and leave no local stdlib artifacts.
run_source "$WORK/pa" A.hx
run_source "$WORK/pb" B.hx

mkdir -p "$WORK/canonical/pa" "$WORK/canonical/pb"

# BLADE metadata records wall-clock timestamps. Remove those two variable-length
# fields before comparing so the check covers MIR and cached maps, not run time.
canonicalize() {
    python3 - "$1" "$2" <<'PY'
import sys

src, dst = sys.argv[1:]
data = bytearray(open(src, "rb").read())

def read_varint(pos):
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return pos, value
        shift += 7

pos = 4                              # magic
pos, _ = read_varint(pos)            # format version
for _ in range(2):                   # metadata name and source path
    pos, length = read_varint(pos)
    pos += length
pos, _ = read_varint(pos)             # source hash
source_timestamp_start = pos
source_timestamp_end, _ = read_varint(pos)
compile_timestamp_start = source_timestamp_end
compile_timestamp_end, _ = read_varint(compile_timestamp_start)

canonical = (
    data[:source_timestamp_start]
    + data[source_timestamp_end:compile_timestamp_start]
    + data[compile_timestamp_end:]
)
open(dst, "wb").write(canonical)
PY
}

shared=0
diverged=0
while IFS= read -r name; do
    fa=$(find "$WORK/pa/.rayzor" -name "$name" | head -1)
    fb=$(find "$WORK/pb/.rayzor" -name "$name" | head -1)
    [ -n "$fa" ] && [ -n "$fb" ] || continue
    shared=$((shared + 1))
    ca="$WORK/canonical/pa/$name"
    cb="$WORK/canonical/pb/$name"
    canonicalize "$fa" "$ca"
    canonicalize "$fb" "$cb"
    bytes=$(cmp -l "$ca" "$cb" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$bytes" != "0" ]; then
        diverged=$((diverged + 1))
        printf '  DIVERGED %-44s %s differing bytes\n' "${name%.blade}" "$bytes"
    else
        printf '  ok       %-44s\n' "${name%.blade}"
    fi
done < <(find "$WORK/pa/.rayzor" -name '*.blade' -exec basename {} \; | sort)

echo
if [ "$shared" = "0" ]; then
    echo "FAIL: the two programs shared no stdlib modules; the oracle proved nothing." >&2
    exit 2
fi
echo "$shared shared modules, $diverged program-dependent."
[ "$diverged" = "0" ]
