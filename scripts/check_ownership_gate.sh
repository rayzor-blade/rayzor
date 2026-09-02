#!/usr/bin/env bash
#
# Does an object built inside a loop body survive the loop?
#
# `var n = new Node(i)` makes the loop body's drop scope the OWNER of n. If the
# body frees it at scope exit while something outside still names it, malloc
# hands the identical block back next iteration and the whole structure
# collapses onto one self-referential node -- `head.next == head`, and a
# `while (p != null)` walk never ends.
#
# Ownership can leave the body by many routes, and each is a separate lowering
# path: assignment to a local, to a parameter, to a static field, through a
# ternary or a cast, into an array, as a call argument, through a capture cell.
# They are all here because fixing one says nothing about the others.
#
# The memory cases are the other half. A fix that stops freeing gets the
# ANSWERS right and leaks without bound, so every correctness case is paired
# with a footprint that must not track the iteration count. `drop_once` covers
# the opposite failure: freeing the same object twice.
#
# Expected answers are haxe 4.3.6's, taken from `haxe --run Main.hx`.
#
#   ./check_ownership_gate.sh
#   RAYZOR=/path/to/rayzor ./check_ownership_gate.sh
#
# Cases listed in KNOWN_FAIL are the open bugs this gate still tracks. Exit is
# non-zero only when something OUTSIDE that list fails, so the script is a
# regression gate today and a progress report as the list shrinks.
#
# Every case in that list now fails the same way: the object that escapes the loop
# body is freed there, so the structure collapses onto one self-referential node.
# mem_accumulator_dies is listed for the same reason seen from the other side --
# the walk over that collapsed list never terminates, so it yields no footprint at
# all rather than a wrong one. One defect, eight shapes; the list shrinks together
# or not at all.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RAYZOR="${RAYZOR:-$PWD/target/release/rayzor}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SMALL="${SMALL:-200000}"
LARGE="${LARGE:-800000}"
# A footprint that grows by more than this between the two sizes is a leak.
SLACK_MB="${SLACK_MB:-8}"
# Ceiling for one fixture. A fix that leaks per-iteration can reach tens of GB
# and take the machine down with it, so cap rather than trust the fixture.
CAP_MB="${CAP_MB:-2048}"
# Wall-clock ceiling for one fixture run. A collapsed object graph makes a list
# walk non-terminating, which no memory cap can catch.
RUN_LIMIT_S="${RUN_LIMIT_S:-120}"

KNOWN_FAIL="local_accumulator param_accumulator static_accumulator ternary_rhs array_push capture_cell shapes_guarded_while_nested mem_accumulator_dies"

pass=0; known=0; regressed=0; fixed=0

is_known() { case " $KNOWN_FAIL " in *" $1 "*) return 0;; *) return 1;; esac; }

# Run one fixture under BOTH a footprint cap and a wall-clock limit, echo its
# stdout. The time limit is not optional: a case whose object graph came out
# cyclic walks `while (p != null)` forever, and a memory cap never fires on a
# walk that allocates nothing. Without it the gate hangs instead of scoring the
# case as broken -- which is the failure it exists to report.
capped_run() {
    local dir="$1"; shift
    # `exec` so the backgrounded pid IS rayzor. Without it the pid is the
    # subshell, killing that orphans rayzor, the orphan holds the command
    # substitution's pipe open, and the caller blocks forever with a guard that
    # already fired.
    ( cd "$dir" && exec "$RAYZOR" run --release --no-cache Main.hx "$@" 2>&1 ) &
    local pid=$!
    (
        local waited=0
        while kill -0 "$pid" 2>/dev/null; do
            local rss
            rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
            [ -n "$rss" ] && [ "$rss" -gt $((CAP_MB * 1024)) ] && { kill -9 "$pid" 2>/dev/null; break; }
            sleep 0.3
            waited=$((waited + 1))
            [ "$waited" -gt $((RUN_LIMIT_S * 10 / 3)) ] && { kill -9 "$pid" 2>/dev/null; break; }
        done
    ) &
    local guard=$!
    wait "$pid" 2>/dev/null
    kill -9 "$guard" 2>/dev/null
}

report() {  # name outcome detail
    local name="$1" ok="$2" detail="${3:-}"
    if [ "$ok" = yes ]; then
        if is_known "$name"; then printf "  %-28s FIXED\n" "$name"; fixed=$((fixed + 1))
        else printf "  %-28s PASS\n" "$name"; pass=$((pass + 1)); fi
    else
        if is_known "$name"; then printf "  %-28s known-fail  %s\n" "$name" "$detail"; known=$((known + 1))
        else printf "  %-28s REGRESSED   %s\n" "$name" "$detail"; regressed=$((regressed + 1)); fi
    fi
}

answer_case() {  # name expected
    local name="$1" want="$2" got
    got=$(capped_run "$WORK/$name" | sed $'s/\033\[[0-9;]*m//g' | grep -v '^Running' | tr '\n' ' ' | sed 's/ *$//')
    [ "$got" = "$want" ] && report "$name" yes || report "$name" no "got[$got] want[$want]"
}

# /usr/bin/time reports its DIRECT child, so it must exec rayzor itself: wrapping
# the run in `timeout` measures the wrapper (~1MB) and hides everything.
peak_mb() {  # dir -> integer MB
    ( cd "$WORK/$1" && timeout -k 5 "$RUN_LIMIT_S" /usr/bin/time -l "$RAYZOR" run --release --no-cache Main.hx 2>&1 >/dev/null ) \
        | awk '/peak memory footprint/{printf "%d", $1/1048576}'
}

memory_case() {  # name -- compares the ${name}_small and ${name}_large fixtures
    local name="$1" a b sa sb
    # A fixture whose loop did not run proves nothing, so require the answer first.
    sa=$(capped_run "$WORK/${name}_small" | grep -c .)
    sb=$(capped_run "$WORK/${name}_large" | grep -c .)
    if [ "$sa" -eq 0 ] || [ "$sb" -eq 0 ]; then
        report "$name" no "fixture produced no output -- its loop did not run"
        return
    fi
    a=$(peak_mb "${name}_small")
    b=$(peak_mb "${name}_large")
    if [ -z "$a" ] || [ -z "$b" ]; then report "$name" no "no footprint reading"; return; fi
    if [ $((b - a)) -le "$SLACK_MB" ]; then report "$name" yes
    else report "$name" no "footprint ${a}MB -> ${b}MB from ${SMALL} to ${LARGE} iterations"; fi
}

new_case() { mkdir -p "$WORK/$1"; cat > "$WORK/$1/Main.hx"; }

# ---------------------------------------------------------------- correctness

new_case local_accumulator <<'HX'
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var head:Node = null;
    for (i in 0...3) { var n = new Node(i); n.next = head; head = n; }
    Sys.println(head.v + " " + head.next.v + " " + (head == head.next));
} }
HX

# The accumulator is a PARAMETER. Every loop lowerer filters parameters out of
# the loop-carried set, so nothing marks it as escaping the body.
new_case param_accumulator <<'HX'
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main {
    static function build(head:Node):Node {
        for (i in 0...3) { var n = new Node(i); n.next = head; head = n; }
        return head;
    }
    static function main() { var h = build(null); Sys.println(h.v + " " + h.next.v + " " + (h == h.next)); }
}
HX

# A static field write lowers to an lvalue whose symbol is global, which skips
# the local-variable ownership paths entirely.
new_case static_accumulator <<'HX'
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main {
    static var head:Node = null;
    static function main() {
        for (i in 0...3) { var n = new Node(i); n.next = head; head = n; }
        Sys.println(head.v + " " + head.next.v + " " + (head == head.next));
    }
}
HX

# Same assignment, wrapped so the right-hand side is no longer a bare variable.
new_case ternary_rhs <<'HX'
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var head:Node = null;
    for (i in 0...3) { var n = new Node(i); n.next = head; head = (i >= 0) ? n : head; }
    Sys.println(head.v + " " + head.next.v + " " + (head == head.next));
} }
HX

new_case field_store_cast <<'HX'
class Base { public var v:Int; public function new(v:Int) { this.v = v; } }
class Derived extends Base { public function new(v:Int) { super(v); } }
class Box { public var base:Base; public function new() {} }
class Main { static function main() {
    var box = new Box();
    var seen = 0;
    for (i in 0...3) { var n = new Derived(i); box.base = cast n; seen += box.base.v; }
    Sys.println(box.base.v + " " + seen);
} }
HX

# Ownership leaves through a call argument the callee retains.
new_case array_push <<'HX'
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var xs = [];
    for (i in 0...5) { var c = new Cell(i); xs.push(c); }
    var s = 0; for (c in xs) s += c.v;
    Sys.println(s + " " + xs.length + " " + (xs[0] != xs[1]));
} }
HX

new_case inline_new_arg <<'HX'
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class Keeper {
    public var items:Array<Cell>;
    public function new() { items = []; }
    public function add(c:Cell):Void { items.push(c); }
}
class Main { static function main() {
    var k = new Keeper();
    for (i in 0...5) k.add(new Cell(i));
    var s = 0; for (c in k.items) s += c.v;
    Sys.println(s + " " + k.items.length + " " + k.items[0].v);
} }
HX

# The accumulator is also captured, so it lives in a cell rather than a register
# and never gets a loop header phi.
new_case capture_cell <<'HX'
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var head:Node = null;
    var peek = function() { return head; };
    for (i in 0...3) { var n = new Node(i); n.next = head; head = n; }
    var p = peek();
    Sys.println(p.v + " " + p.next.v + " " + (p == p.next));
} }
HX

# The assignment sits behind an if, in a while, and in the inner loop of a nest.
new_case shapes_guarded_while_nested <<'HX'
class Cell { public var v:Int; public var next:Cell; public function new(v:Int) { this.v = v; } }
class Main {
    static function len(c:Cell):Int { var n = 0; var p = c; while (p != null && n < 100) { n++; p = p.next; } return n; }
    static function main() {
        var a:Cell = null;
        for (i in 0...6) if (i % 2 == 0) { var c = new Cell(i); c.next = a; a = c; }
        var b:Cell = null; var k = 0;
        while (k < 4) { var c = new Cell(k); c.next = b; b = c; k++; }
        var d:Cell = null;
        for (i in 0...3) for (j in 0...2) { var c = new Cell(j); c.next = d; d = c; }
        Sys.println(len(a) + " " + len(b) + " " + len(d));
    }
}
HX

echo "answers (haxe 4.3.6):"
answer_case local_accumulator           "2 1 false"
answer_case param_accumulator           "2 1 false"
answer_case static_accumulator          "2 1 false"
answer_case ternary_rhs                 "2 1 false"
answer_case field_store_cast            "2 3"
answer_case array_push                  "10 5 true"
answer_case inline_new_arg              "10 5 0"
answer_case capture_cell                "2 1 false"
answer_case shapes_guarded_while_nested "3 4 6"

# --------------------------------------------------------------------- memory

# The move happens on SOME iterations only. A fix that untracks the object when
# it lowers the assignment -- rather than when the branch is taken -- loses the
# free on every path, including the one that did not move anything.
#
# Bounds are literal on purpose. A bound read from Sys.args() makes the program
# print nothing and run no loop at all, which reads as a perfectly flat
# footprint and hides exactly the leak these cases exist to catch.
for pair in "small:$SMALL" "large:$LARGE"; do
    tag=${pair%%:*}; n=${pair##*:}

    new_case "mem_conditional_move_$tag" <<HX
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var best = new Cell(0);
    for (i in 1...$n) { var cand = new Cell(i % 7); if (cand.v > best.v) best = cand; }
    Sys.println("cond best=" + best.v);
} }
HX

    # Nothing escapes here at all. This is the one that catches a fix buying
    # correctness by simply freeing less.
    new_case "mem_unconditional_temp_$tag" <<HX
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
class Main { static function main() {
    var acc = 0;
    for (i in 1...$n) { var t = new Cell(i); acc += t.v & 1; }
    Sys.println("temp acc=" + acc);
} }
HX

    # A list built per call and dead on return. If nothing reclaims it the
    # footprint tracks the call count.
    new_case "mem_accumulator_dies_$tag" <<HX
class Node { public var v:Int; public var next:Node; public function new(v:Int) { this.v = v; } }
class Main {
    static function build():Int {
        var head:Node = null;
        for (i in 0...50) { var x = new Node(i); x.next = head; head = x; }
        var c = 0; var p = head;
        while (p != null) { c++; p = p.next; }
        return c;
    }
    static function main() {
        var t = 0;
        for (r in 0...$((n / 50))) t += build();
        Sys.println("acc total=" + t);
    }
}
HX
done

echo "footprint (must not track iteration count):"
memory_case mem_conditional_move
memory_case mem_unconditional_temp
memory_case mem_accumulator_dies

# ---------------------------------------------------------------- double free

# Reassigning the accumulator after the loop must not let two owners name one
# object. With @:derive(Drop) a second release is visible rather than silent.
new_case drop_once <<'HX'
@:derive(Drop)
class Node {
    public var v:Int; public var next:Node;
    public function new(v:Int) { this.v = v; }
    public function drop():Void { Sys.println("drop " + v); }
}
class Main { static function main() {
    var head:Node = null;
    for (i in 0...3) { var n = new Node(i); n.next = head; head = n; }
    var alias = head;
    Sys.println(alias.v);
    head = new Node(99);
    Sys.println(head.v);
} }
HX

echo "release count:"
dup=$(capped_run "$WORK/drop_once" | sed $'s/\033\[[0-9;]*m//g' | grep '^drop ' | sort | uniq -d | tr '\n' ' ')
[ -z "$dup" ] && report drop_once yes || report drop_once no "released twice: $dup"

echo
echo "  $pass pass, $fixed fixed, $known known-fail, $regressed regressed"
[ "$regressed" -eq 0 ] || echo "  a case outside KNOWN_FAIL broke"
exit $((regressed > 0 ? 1 : 0))
