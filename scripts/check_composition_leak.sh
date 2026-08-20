#!/usr/bin/env bash
#
# A heap object stored into another object's field is never freed.
#
# The wrapper itself is freed; what it holds is not. `insert_free` reads the
# constructor's store of its parameter into a field as an escape, so the
# argument stops being freed at the caller, and nothing frees it when the
# wrapper dies.
#
# It is shape-sensitive, which is why ordinary programs hide it: when scalar
# replacement erases the wrapper the store disappears and the object is freed
# normally. SRA declines as soon as the wrapper pointer is an argument to a
# call the inliner refused — so the leak needs a wrapper that outlives its
# constructor AND a callee big enough not to inline.
#
# No annotation is involved. This is plain Haxe.
#
#   ./check_composition_leak.sh          2M and 4M iterations
#   ITERS=8000000 ./check_composition_leak.sh
#
# Reads PEAK MEMORY FOOTPRINT, not max RSS: on macOS only the former is
# meaningful. A flat footprint across the two sizes means no leak; one that
# tracks the iteration count is ~16 bytes per iteration never returned.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RAYZOR="${RAYZOR:-$PWD/target/release/rayzor}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

emit() { # $1 = class name, $2 = iterations, $3 = "wrap" | "plain"
    local name="$1" iters="$2" mode="$3"
    if [[ "$mode" == wrap ]]; then
        cat > "$WORK/$name.hx" <<HX
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class Box { public var c:Cell; public function new(c:Cell) { this.c = c; } }
class $name {
    // Four branches so the inliner declines and scalar replacement with it.
    static function kernel(b:Box, i:Int):Int {
        var s = 0;
        if ((i & 1) == 0) s += b.c.v; else s -= b.c.v;
        if ((i & 2) == 0) s += 2; else s -= 2;
        if ((i & 4) == 0) s += 3; else s -= 3;
        if ((i & 8) == 0) s += 5; else s -= 5;
        return s;
    }
    static function main():Void {
        var t = 0; var i = 0;
        while (i < $iters) { var c = new Cell(i); var b = new Box(c); t += kernel(b, i); i++; }
        Sys.println(t);
    }
}
HX
    else
        cat > "$WORK/$name.hx" <<HX
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class $name {
    static function kernel(c:Cell, i:Int):Int {
        var s = 0;
        if ((i & 1) == 0) s += c.v; else s -= c.v;
        if ((i & 2) == 0) s += 2; else s -= 2;
        if ((i & 4) == 0) s += 3; else s -= 3;
        if ((i & 8) == 0) s += 5; else s -= 5;
        return s;
    }
    static function main():Void {
        var t = 0; var i = 0;
        while (i < $iters) { var c = new Cell(i); t += kernel(c, i); i++; }
        Sys.println(t);
    }
}
HX
    fi
}

peak() { # $1 = class name -> peak footprint in bytes
    ( cd "$WORK" && rm -rf .rayzor &&
      /usr/bin/time -l "$RAYZOR" run "$1.hx" --release 2>&1 |
      grep -a "peak memory footprint" | awk '{print $1}' )
}

A="${ITERS:-2000000}"
B=$(( A * 2 ))

emit ctlA  "$A" plain ; emit ctlB  "$B" plain
emit wrapA "$A" wrap  ; emit wrapB "$B" wrap

pa=$(peak ctlA);  pb=$(peak ctlB)
wa=$(peak wrapA); wb=$(peak wrapB)

printf "  %-22s %12s %12s\n" "" "${A} iters" "${B} iters"
printf "  %-22s %12s %12s\n" "no wrapper (control)" "$pa" "$pb"
printf "  %-22s %12s %12s\n" "object held in a field" "$wa" "$wb"

growth=$(( wb - wa ))
ctl_growth=$(( pb - pa ))
echo
echo "  control growth: $ctl_growth bytes    wrapped growth: $growth bytes"
# The control moves by a few hundred KB of noise; a leak moves by ~16 bytes an
# iteration, so anything past a megabyte over A extra iterations is the bug.
if (( growth > 1000000 && growth > ctl_growth * 4 )); then
    echo "  LEAK: the object stored in the field is never freed."
    exit 1
fi
echo "  no leak."
