// Behavioural test: `obj.method` (without invocation) is a bound
// method reference — a callable value that captures `obj` and
// dispatches to `method` when invoked.

class Animal {
    public var name: String;
    public function new(n: String) { this.name = n; }
    public function describe(): String { return "Animal: " + this.name; }
    public function bumped(x: Int): Int { return x + 1; }
}

class Main {
    static function main() {
        var a = new Animal("Rex");

        // Direct invocation — regression guard.
        trace(a.describe());          // Animal: Rex

        // Bound method reference, no args.
        var d = a.describe;
        trace(d());                   // Animal: Rex

        // Bound method reference, with arg forwarding.
        var b = a.bumped;
        trace(b(99));                  // 100
    }
}
