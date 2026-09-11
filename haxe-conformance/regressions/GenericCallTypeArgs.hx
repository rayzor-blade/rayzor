// A call to a generic method must carry its inferred type argument all the way
// to the monomorphizer. Without it the template is never specialized, the
// backend installs a trap stub, and the stub cascades to every caller -- so the
// failure surfaces as "<caller> was never compiled", two levels away.
//
// T is only recoverable in the typer: Array<T> lowers to an opaque pointer, so
// by MIR both the signature and the argument registers say nothing.
private class Asserts {
    public function new() {}
    // T appears only INSIDE Array<T>
    function aeq<T>(expected:Array<T>, actual:Array<T>):Bool {
        if (expected.length != actual.length) return false;
        for (i in 0...expected.length) if (expected[i] != actual[i]) return false;
        return true;
    }
    // T is the whole parameter
    function eq<T>(a:T, b:T):Bool { return a == b; }
    public function run() {
        var ints = [1, 2, 3];
        if (!aeq([1, 2, 3], ints)) throw "aeq over Array<Int>";
        var strs = ["a", "b"];
        if (!aeq(["a", "b"], strs)) throw "aeq over Array<String>";
        if (!eq(1, 1)) throw "eq over Int";
        if (!eq("x", "x")) throw "eq over String";
        // second argument's element type is imprecise; the first still pins T
        var loose:Array<Dynamic> = [1, 2, 3];
        if (!aeq([1, 2, 3], loose)) throw "aeq with an imprecise argument";
    }
}
class GenericCallTypeArgs {
    static function main() {
        new Asserts().run();
        trace("CONFORMANCE_OK");
    }
}
