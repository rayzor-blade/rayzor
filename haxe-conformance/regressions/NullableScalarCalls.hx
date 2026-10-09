class NullableScalarCalls {
    static inline function inlineNull<T>(value:Null<T>):Bool return value == null;
    static function callNull<T>(value:Null<T>):Bool return value == null;

    static function main() {
        var i:Int = 0;
        var u:UInt = 0;
        var l:haxe.Int64 = haxe.Int64.make(0, 0);
        var b = false;
        var f = 0.0;
        var a = new ScalarInt(0);
        if (inlineNull(i) || inlineNull(u) || inlineNull(l) || inlineNull(b) || inlineNull(f) || inlineNull(a)) throw "inline zero became null";
        if (callNull(i) || callNull(u) || callNull(l) || callNull(b) || callNull(f) || callNull(a)) throw "zero became null";
        var missing:Null<UInt> = null;
        var present:Null<UInt> = 0;
        var wide:Null<haxe.Int64> = l;
        if (!inlineNull(missing) || !callNull(missing)) throw "missing nullable value";
        if (inlineNull(present) || callNull(present) || inlineNull(wide) || callNull(wide)) throw "boxed nullable value";
        if (inlineNull("text") || callNull("text") || inlineNull([0]) || callNull([0])) throw "reference became null";
        var object = new NullableItem();
        if (inlineNull(object) || callNull(object)) throw "instance became null";
        var absent:NullableItem = null;
        if (!inlineNull(absent) || !callNull(absent)) throw "missing instance";
        Sys.println("CONFORMANCE_OK");
    }
}

abstract ScalarInt(Int) {
    public inline function new(value:Int) this = value;
}

class NullableItem {
    public function new() {}
}
