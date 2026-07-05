// Method calls on a receiver explicitly typed Null<C> must resolve against C
// (and behave like the ?param sugar). Regression: `sp.parallelRows(...)` on a
// `sp:Null<SpinPool>` helper param silently lowered to NOTHING — the branch
// body emitted only a line marker, so pooled kernels produced all-zero output.
class Counter {
    public var n:Int;
    public function new() { n = 0; }
    public function inc(by:Int):Void { n += by; }
    public function get():Int { return n; }
}

class Main {
    // Explicit Null<Counter> param — the failing shape.
    static function pokeExplicit(c:Null<Counter>, by:Int):Void {
        if (c != null) c.inc(by);
    }

    // Optional-param sugar — the shape that always worked.
    static function pokeSugar(?c:Counter, by:Int = 0):Void {
        if (c != null) c.inc(by);
    }

    static function main() {
        var a = new Counter();
        var b = new Counter();
        pokeExplicit(a, 7);
        pokeExplicit(a, 5);
        pokeSugar(b, 7);
        pokeSugar(b, 5);
        var la:Null<Counter> = a; // local explicitly Null-typed, call through it
        var viaLocal = (la != null) ? la.get() : -1;
        if (a.get() == 12 && b.get() == 12 && viaLocal == 12) {
            Sys.println("PASS null-receiver-method a=" + a.get() + " b=" + b.get());
        } else {
            Sys.println("FAIL a=" + a.get() + " b=" + b.get() + " viaLocal=" + viaLocal);
        }
    }
}
