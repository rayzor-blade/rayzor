class AbstractGenericFrom {
    static function relay<T>(value:DeferredValue<T>):DeferredValue<T> return value;

    static function check(label:String, condition:Bool) {
        if (!condition) throw label;
    }

    static function main() {
        check("Int", relay(7).get() == 7);
        check("Float", relay(2.5).get() == 2.5);
        var fractional:DeferredValue<Float> = relay(-3.75);
        check("typed Float", fractional.get() == -3.75);
        check("Bool", relay(false).get() == false);
        check("String", relay("payload").get() == "payload");
        var values = relay([11, 23]).get();
        check("Array", values.length == 2 && values[1] == 23);
        var instance = relay(new FromPayload(37)).get();
        check("Class", instance.value == 37);
        Sys.println("CONFORMANCE_OK");
    }
}

private abstract DeferredValue<T>(() -> T) {
    public function new(fn:() -> T) this = fn;
    public function get():T return this();

    @:from public static function constant<T>(value:T):DeferredValue<T> {
        return new DeferredValue(function() return value);
    }
}

private class FromPayload {
    public var value:Int;
    public function new(value:Int) this.value = value;
}
