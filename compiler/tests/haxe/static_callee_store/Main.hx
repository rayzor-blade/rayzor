// A static var assigned from a CALLEE must be visible to every later read.
//
// Guards a global-load cache that judges "read-only" from the loading function
// alone: a store made by a callee is then invisible, the caller reuses its
// pre-call load, and the static reads as two different values depending on
// which method looks at it.
//
// Needs all three together, each verified necessary: the writer in its own
// MODULE, a POINTER-typed static, and both reads inside ONE method called
// twice, so the reader holds two loads with the mutating call between them.

import pkg.Holder;

class Main {
    static var failures = 0;
    static function ok(what:String, cond:Bool):Void {
        if (!cond) { Sys.println("FAIL " + what); failures++; }
    }

    static function main() {
        Holder.entry(true);
        Holder.entry(false);

        ok("direct visible after in-function store", !Holder.directIsNull());
        ok("viaCall visible after callee store", !Holder.viaCallIsNull());

        if (failures == 0) Sys.println("PASS: statics written by a callee are visible to all readers");
        else { Sys.println("FAILURES: " + failures); Sys.exit(1); }
    }
}
