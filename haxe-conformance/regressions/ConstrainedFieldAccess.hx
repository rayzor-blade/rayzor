private class FieldCarrier {
    public var iterator = 10;
    public var ratio = 1.5;
    public var name = "carrier";
    public var enabled = true;

    public function new() {}
    public function describe():String return name;
}

class ConstrainedFieldAccess {
    static function check(label:String, actual:String, expected:String):Void {
        if (actual != expected) throw label + ": " + actual + " != " + expected;
    }

    static function decrement<T:{iterator:Int}>(value:T):Void {
        value.iterator--;
        --value.iterator;
    }

    static function update<T:{ratio:Float, name:String, enabled:Bool}>(value:T):String {
        value.ratio += 0.5;
        value.name = value.name + "!";
        value.enabled = !value.enabled;
        return value.name;
    }

    static function method<T:{function describe():String;}>(value:T):() -> String {
        return value.describe;
    }

    static function copyName<T:{name:String}, U:{name:String}>(source:T, target:U):Void {
        target.name = source.name;
    }

    static function capturedField<T:FieldCarrier>(value:T):Int {
        var read = function() return value.iterator;
        return read();
    }

    static function main() {
        var record = {iterator:10, ratio:1.5, name:"record", enabled:true};
        decrement(record);
        check("record decrement", Std.string(record.iterator), "8");
        check("record update result", update(record), "record!");
        check("record float", Std.string(record.ratio), "2");
        check("record bool", Std.string(record.enabled), "false");

        var instance = new FieldCarrier();
        decrement(instance);
        check("class decrement", Std.string(instance.iterator), "8");
        check("captured nominal field", Std.string(capturedField(instance)), "8");
        check("class update result", update(instance), "carrier!");
        check("class float", Std.string(instance.ratio), "2");
        check("class bool", Std.string(instance.enabled), "false");
        copyName(record, instance);
        check("copied String", instance.name, "record!");
        var describe = method(instance);
        instance.name = "changed";
        check("bound method", describe(), "changed");
        Sys.println("CONFORMANCE_OK");
    }
}
