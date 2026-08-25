// Reflect.compare's typed runtime accepts erased i64 slots. Float operands
// must enter those slots by bitcast, not numeric conversion: converting 0.1
// and 0.2 to integers makes both zero and destroys their ordering.

class Main {
    static var failures = 0;

    static function check(label:String, value:Bool):Void {
        if (!value) {
            failures++;
            Sys.println("FAIL " + label);
        }
    }

    static function genericCompare<T>(a:T, b:T):Int {
        return Reflect.compare(a, b);
    }

    static function main():Void {
        check("concrete less", Reflect.compare(0.1, 0.2) < 0);
        check("concrete greater", Reflect.compare(0.2, 0.1) > 0);
        check("concrete equal", Reflect.compare(0.125, 0.125) == 0);
        check("generic less", genericCompare(0.1, 0.2) < 0);
        check("generic greater", genericCompare(0.2, 0.1) > 0);

        if (failures == 0) {
            Sys.println("PASS Reflect.compare Float ordering");
        } else {
            Sys.println("FAILURES: " + failures);
            Sys.exit(1);
        }
    }
}
