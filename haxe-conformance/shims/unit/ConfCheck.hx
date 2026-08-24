package unit;

/** Assertion counters for the conformance harness, in the root package so the
    generated main() reaches them without a dotted static path. */
class ConfCheck {
    public static var checks:Int = 0;
    public static var failures:Int = 0;
    public static function fail(msg:String):Void {
        failures++;
        Sys.println("FAILCHECK " + msg);
    }

    public static function ok():Void { checks++; }
    public static function summary():Void {
        if (failures == 0) Sys.println("CONFORMANCE_OK " + checks);
        else Sys.println("CONFORMANCE_BAD " + failures);
    }
}
