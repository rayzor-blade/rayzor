class DynamicObjectFields {
    static function check(label:String, actual:String, expected:String):Void {
        if (actual != expected) throw label + ": " + actual + " != " + expected;
    }

    static function object(value:Dynamic):{value:Dynamic} {
        return {value:value};
    }

    static function checkValue(label:String, value:Dynamic, expected:String):Void {
        var record = object(value);
        check(label + " direct", Std.string(record.value), expected);
        check(label + " reflected", Std.string(Reflect.field(record, "value")), expected);
        check(label + " copied", Std.string(Reflect.field(Reflect.copy(record), "value")), expected);
    }

    static function merge<A:{}, B:{}, C:A & B>(a:A, b:B):C {
        return cast {foo:(cast a).foo, bar:(cast b).bar};
    }

    static function main() {
        checkValue("Int", 5, "5");
        checkValue("Float", 1.5, "1.5");
        checkValue("Bool", true, "true");
        checkValue("String", "hello", "hello");
        checkValue("null", null, "null");
        checkValue("Array", [1, 2], "[1,2]");
        var merged = merge({foo:5}, {bar:"bar"});
        check("merged Int", Std.string(merged.foo), "5");
        check("merged String", merged.bar, "bar");
        var record = object(1);
        Reflect.setField(record, "value", "changed");
        check("mutated", Std.string(record.value), "changed");
        var nullable:Null<Int> = 23;
        Reflect.setField(record, "value", nullable);
        check("nullable write", Std.string(record.value), "23");
        var nullableText:Null<String> = "text";
        Reflect.setProperty(record, "value", nullableText);
        check("nullable property write", Std.string(record.value), "text");
        Reflect.setField(record, "value", [3, 4]);
        check("array write", Std.string(record.value), "[3,4]");
        Reflect.setField(record, "extra", true);
        Reflect.setField(record, "value", 42);
        check("promoted field", Std.string(Reflect.field(record, "value")), "42");
        var nested = object({name:"inner"});
        var reflected:Dynamic = Reflect.field(nested, "value");
        check("nested object", reflected.name, "inner");
        Sys.println("CONFORMANCE_OK");
    }
}
