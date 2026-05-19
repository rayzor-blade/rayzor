// Regression test for nested generic close: `Array<Thread<Int>>` (the
// final `>>` is one `Shr` token from the lexer, but the parser needs
// to consume two consecutive `>` to close the two type-argument lists).
//
// Pre-fix the RD parser bailed with "expected type, found '>>'"; the
// nom fallback then silently produced an empty file when the rest of
// the file used newer syntax it didn't know (parameterised arrow type),
// dropping the class declaration entirely.
//
// After: TokenStream's `expect_closing_gt` virtually splits a `Shr`
// into two `Gt` tokens, claimed one per call. `at_closing_gt` returns
// true at `Gt` OR `Shr` so the loop terminates at the right spot.

import rayzor.concurrent.Thread;

class Box {
    public function new() {}

    // Single-level generic, no nesting — must continue to work.
    public function single():Array<Int> { return [1, 2, 3]; }

    // Two-level nested generic ending in `>>`.
    public function nested():Array<Array<Int>> { return [[1], [2, 3]]; }

    // Three-level nested generic ending in `>>>`. Lexer produces
    // `Ushr` (`>>>`); the parser splits it into three `Gt` tokens.
    public function triple():Array<Array<Array<Int>>> { return [[[1]]]; }

    // Mixed: nested generic in a method signature whose body also
    // uses the parameterised arrow type form — this combo is what
    // surfaced the original bug in NumaPool.
    public function parallel(items:Int, fn:(idx:Int, node:Int)->Void):Void {
        var threads = new Array<Thread<Int>>();
        // (no actual spawning needed for the parser test)
        for (i in 0...items) fn(i, 0);
    }
}

class Main {
    static function main() {
        var b = new Box();
        trace(b.single().length);              // 3
        trace(b.nested().length);              // 2
        trace(b.triple().length);              // 1
        b.parallel(2, function(i:Int, n:Int):Void { trace(i); });
    }
}
