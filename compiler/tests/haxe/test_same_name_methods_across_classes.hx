// Two classes whose methods share a bare name and parameter shape.
//
// `IrFunction.name` is the bare method name, so `Alpha.step` and `Beta.step`
// are both "step", and a backend that keys its declarations on that name can
// bind the second class's calls to the first class's body. Nothing here is
// reached through a pointer, so neither method carries a hidden environment
// parameter -- which is what used to keep the two signatures apart by
// accident, and is why this went unnoticed.
class Alpha {
    public var a:Int;
    public function new(a:Int) { this.a = a; }
    public function step(x:Int):Int { return this.a + x; }
}

class Beta {
    public var b:Int;
    public var c:Int;
    public function new(b:Int) { this.b = b; this.c = b * 5; }
    public function step(x:Int):Int { return this.b + this.c + x * 10; }
}

class TestSameNameMethodsAcrossClasses {
    static function check(label:String, got:Int, want:Int) {
        if (got == want) {
            Sys.println(label + " ok");
        } else {
            Sys.println("FAIL " + label + ": got " + got + " want " + want);
        }
    }

    static function main() {
        var p = new Alpha(10);
        var q = new Beta(10);
        check("alpha.step", p.step(3), 13);
        check("beta.b", q.b, 10);
        check("beta.c", q.c, 50);
        check("beta.step", q.step(3), 90);
    }
}
