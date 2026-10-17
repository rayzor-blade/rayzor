private typedef Ints = Array<Int>;

private class Parent {
    public function new() {}
    public function name():String return "parent";
}

private class Child extends Parent {
    override public function name():String {
        final f = function() return super.name();
        return "child+" + f();
    }
}

private class Holder {
    public var callback:Null<Int->Int>;
    public function new() {
        callback = function(x) return x * 2;
    }
}

class ClosureSuperAndNullableFunctions {
    static function first():Int {
        return 1;
        return 2;
    }
    static function main() {
        if (first() != 1) throw "unreachable return replaced the live one";
        if (new Child().name() != "child+parent") throw "super inside a closure";
        if (new Holder().callback(21) != 42) throw "Null<Function> field call";
        final ints = new Ints();
        ints.push(3);
        if (ints.length != 1 || ints[0] != 3) throw "new on an Array typedef";
        Sys.println("CONFORMANCE_OK");
    }
}
