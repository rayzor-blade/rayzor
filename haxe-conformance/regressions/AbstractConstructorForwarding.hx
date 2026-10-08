class AbstractConstructorForwarding {
    static function main() {
        var array = new ConstructorWrapper<Array<Int>>();
        if (array.value().length != 0) throw "array construction";
        var values = array.value();
        values.push(3);
        if (array.value()[0] != 3) throw "array storage";
        var regexp = new ConstructorWrapper<EReg>("abc", "");
        var pattern = regexp.value();
        if (!pattern.match("abc")) throw "regexp construction";
        var text = new ConstructorWrapper<ConstructorWrapper<String>>("hello");
        if (text.value().value() != "hello") throw "nested forwarding";
        var scalar = new ConstructorWrapper<ConstructedScalar>(2);
        if (scalar.value() != 4) throw "abstract constructor body";
        var instance = new ConstructorWrapper<ConstructedObject>(3);
        if (instance.value().number != 8 || ConstructedObject.calls != 1) throw "class constructor body";
        var float = new ConstructorWrapper<ConstructedFloat>(1.25);
        if (float.value() != 2.5) throw "float constructor body";
        Sys.println("CONFORMANCE_OK");
    }
}

@:forward.new
abstract ConstructorWrapper<T>(T) to T {
    public function value():T return this;
}

abstract ConstructedScalar(Int) to Int {
    public inline function new(value:Int) this = value * value;
}

abstract ConstructedFloat(Float) to Float {
    public inline function new(value:Float) this = value * 2;
}

class ConstructedObject {
    public static var calls = 0;
    public var number:Int;
    public function new(value:Int) {
        calls++;
        number = value + 5;
    }
}
