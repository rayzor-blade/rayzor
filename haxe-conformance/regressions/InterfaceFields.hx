private enum Payload { Empty; Number(value:Int); }
private interface ReadPayload { var payload:Payload; }
private class PayloadValue implements ReadPayload {
    public var payload:Payload;
    public function new(payload:Payload) { this.payload = payload; }
}
private enum Kind { First; Second; }
private interface ReadKind { var kind:Kind; }
private class KindValue implements ReadKind {
    public var kind:Kind;
    public function new(kind:Kind) { this.kind = kind; }
}
private interface ReadValue {
    var value(default, never):Float;
}
private interface WriteValue {
    var value(never, default):Int;
}
private interface ReadComputed { var computed(get, never):Int; }
private class Computed implements ReadComputed {
    var backing:Int = 23;
    public var computed(get, never):Int;
    function get_computed():Int { return backing; }
    public function new() {}
}
private interface ReadReferences {
    var label:String;
    var child:Parent;
}
private class References implements ReadReferences {
    public var label:String = "label";
    public var child:Parent = new Parent();
    public function new() {}
}
private class Parent {
    public var padding:Int = 91;
    public function new() {}
}
private class IntegerValue extends Parent implements ReadValue {
    public var value:Int = 12;
    public function new() { super(); }
}
private class FloatValue implements ReadValue implements WriteValue {
    public var value:Float = 2.5;
    public function new() {}
}
class InterfaceFields {
    static function payloadNumber(value:ReadPayload):Int {
        return switch (value.payload) { case Empty: 0; case Number(number): number; };
    }
    static function kindNumber(value:ReadKind):Int {
        return switch (value.kind) { case First: 1; case Second: 2; };
    }
    static function compare(expected:Float, actual:Float):Void {
        if (expected != actual) throw "interface field argument";
    }
    static function read(value:ReadValue):Float { return value.value; }
    static function write(value:WriteValue):Void { value.value = 7; }
    static function main() {
        var integer = new IntegerValue();
        var floating = new FloatValue();
        if (read(integer) != 12.0 || read(floating) != 2.5) throw "interface field read";
        var view:ReadValue = integer;
        compare(12.0, view.value);
        if (view.value != 12.0) throw "local interface field";
        integer.value = 15;
        if (read(integer) != 15.0) throw "interface field is a live view";
        write(floating);
        if (floating.value != 7.0 || read(floating) != 7.0) throw "interface field write";
        if (integer.padding != 91) throw "unrelated field modified";
        var computed:ReadComputed = new Computed();
        if (computed.computed != 23) throw "computed interface property";
        var references:ReadReferences = new References();
        if (references.label != "label" || references.child.padding != 91) throw "interface reference field";
        if (kindNumber(new KindValue(First)) != 1 || kindNumber(new KindValue(Second)) != 2)
            throw "interface enum field";
        if (payloadNumber(new PayloadValue(Number(41))) != 41 || payloadNumber(new PayloadValue(Empty)) != 0)
            throw "interface boxed enum field";
        trace("CONFORMANCE_OK");
    }
}
