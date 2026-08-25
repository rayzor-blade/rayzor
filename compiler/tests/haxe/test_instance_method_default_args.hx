// Regression: resolved calls must bind skipped leading optionals and fill
// trailing defaults before MIR reaches the backend.
//
// This call path used to emit fewer arguments than the declared signature.
// Cranelift silently padded the missing values with zero, hiding both bugs:
// string defaults became null/empty, and a sole argument could occupy the
// leading optional parameter instead of the required parameter after it.

class Defaults {
    public function new() {}

    public function trailing(a:Int, b:String = "dflt"):String {
        return a + "/" + b;
    }

    public function leading(?a:Int, b:String):String {
        return a + "/" + b;
    }
}

class Main {
    static var failures = 0;

    static function expect(label:String, got:String, want:String):Void {
        if (got != want) {
            failures++;
            Sys.println("FAIL " + label + ": got " + got + " want " + want);
        }
    }

    static function main():Void {
        var methods = new Defaults();
        expect("trailing default", methods.trailing(2), "2/dflt");
        expect("explicit trailing", methods.trailing(2, "given"), "2/given");
        expect("skipped leading optional", methods.leading("value"), "0/value");
        expect("explicit leading optional", methods.leading(3, "value"), "3/value");

        if (failures == 0) {
            Sys.println("PASS instance-method default args");
        } else {
            Sys.println("FAILURES: " + failures);
            Sys.exit(1);
        }
    }
}
