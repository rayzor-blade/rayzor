private typedef Payload<T, C> = {value:T, items:C, name:String};

@:forward
private abstract Wrapper<T, C:Iterable<T>>(Payload<T, C>) from Payload<T, C> {
    @:op(a < b)
    public static inline function before<U, D:Iterable<U>>(a:Wrapper<U, D>, b:Wrapper<U, D>):Bool {
        return a.name.charCodeAt(0) < b.name.charCodeAt(0);
    }

    public static function rename<U, D:Iterable<U>>(a:Wrapper<U, D>, name:String):String {
        a.name = name;
        return a.name;
    }
}

class ForwardedGenericFields {
    static function check(label:String, actual:String, expected:String):Void {
        if (actual != expected) throw label + ": " + actual + " != " + expected;
    }

    static function main() {
        var first:Wrapper<Int, Array<Int>> = {value:11, items:[12, 13], name:"a"};
        var second:Wrapper<Int, Array<Int>> = {value:21, items:[22], name:"b"};
        check("operator", Std.string(first < second), "true");
        check("static method", Std.string(Wrapper.before(first, second)), "true");
        check("String field", first.name, "a");
        check("Int field", Std.string(first.value), "11");
        check("array field", first.items.join(","), "12,13");
        check("rename", Wrapper.rename(first, "z"), "z");
        check("shared mutation", first.name, "z");
        check("changed order", Std.string(first < second), "false");
        Sys.println("CONFORMANCE_OK");
    }
}
