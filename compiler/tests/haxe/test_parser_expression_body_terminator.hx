// Regression test for compiler fix #4 (partial):
// `function name():T return expr;` — the trailing `;` is the
// declaration terminator, not part of the expression. Pre-fix the
// RD parser left the `;` unconsumed; parse_class_body then tried to
// parse a new field starting at `;`, errored, the file accumulated
// a parse error, the RD parser returned Err, the nom fallback ran,
// and (for files containing newer syntax nom doesn't know like the
// parameterised arrow type form) silently produced an empty file —
// dropping the entire class declaration.
//
// After fix: the `;` is consumed by parse_function_decl directly.
// The class registers cleanly even when sibling methods use
// expression-body syntax.

class Counter {
    public function new() {}

    // Block body, no `;` involved.
    public function bumped(n:Int):Int { return n + 1; }

    // Expression body with trailing `;` — this was the bug trigger.
    public function answer():Int return 42;

    // Multiple expression-body methods in a row — all must register.
    public function half(n:Int):Int return Std.int(n / 2);
    public function negated(n:Int):Int return -n;
}

class Main {
    static function main() {
        var c = new Counter();
        trace(c.answer());         // 42
        trace(c.bumped(5));        // 6
        trace(c.half(8));          // 4
        trace(c.negated(7));       // -7
    }
}
