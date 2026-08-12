// `cast expr` takes its type from the context: `var a:A = cast e` is a cast to
// A. Typed as Dynamic instead, the binding stays Dynamic and a later field read
// takes the Dynamic path — unboxing a value that was never boxed and
// dereferencing the result, which segfaults. A two-arg `cast(e, A)` always
// worked, so the two forms must agree.

class Animal {
    public var name:String;
    public function new(n:String) { name = n; }
}

class Dog extends Animal {
    public function new(n:String) { super(n); }
}

class Main {
    static var failures = 0;

    static function check(what:String, got:String, want:String) {
        if (got != want) {
            Sys.println("FAIL " + what + " -- expected " + want + " but got " + got);
            failures++;
        }
    }

    static function make():Dynamic {
        return new Dog("Rex");
    }

    static function main() {
        // The crashing shape: cast a call result, then read a field
        var a:Dog = cast make();
        check("one-arg cast of a call result", a.name, "Rex");

        // The two-arg form must agree
        var b:Dog = cast(make(), Dog);
        check("two-arg cast of a call result", b.name, "Rex");

        // Through the Type API, which returns Dynamic
        var c:Dog = cast Type.createInstance(Dog, ["Fido"]);
        check("one-arg cast of createInstance", c.name, "Fido");

        // Reading a field declared on the PARENT, through the cast binding
        var d:Animal = cast make();
        check("inherited field through cast", d.name, "Rex");

        Sys.println(failures == 0 ? "PASS onearg-cast" : "FAIL onearg-cast " + failures + " mismatches");
    }
}
