private abstract AlwaysTrue(Int) from Int {
    @:to public function asBool():Bool return true;
}

private abstract ConversionBase(Int) {
    public function new(value:Int) this = value;
    public function get():Int return this;
}

private abstract ConversionChild(Int) {
    public function new(value:Int) this = value;
    @:to public function asBase():ConversionBase return new ConversionBase(this * 2);
}

private abstract ConversionOp(Int) {
    public function new(value:Int) this = value;
    @:op(A + B) static function add(a:ConversionOp, b:ConversionBase):Int;
    @:op(A - B) function subtract(b:ConversionBase):Int return this - b.get();
}

private enum MapPayload {
    Text(value:String);
    Number(value:Int);
}

private abstract ConvertedMapValue(MapPayload) {
    @:from public static function text(value:String):ConvertedMapValue return cast Text(value);
    @:from public static function number(value:Int):ConvertedMapValue return cast Number(value);
    public function render():String {
        return switch this {
            case Text(value): value;
            case Number(value): "" + value;
        };
    }
}

class AbstractImplicitBoundaries {
    static function check(label:String, value:Bool) {
        if (!value) throw label;
    }

    static function take(value:ConversionBase):Int return value.get();

    static function main() {
        var truth:AlwaysTrue = 0;
        check("if", if (truth) true else false);
        check("guard", switch truth { case value if (value): true; case _: false; });
        var count = 0;
        while (truth) { count++; break; }
        check("while", count == 1);
        count = 0;
        do { count++; if (count == 2) break; } while (truth);
        check("do while", count == 2);
        var child = new ConversionChild(4);
        var base:ConversionBase = child;
        check("let", base.get() == 8);
        base = new ConversionChild(5);
        check("assignment", base.get() == 10);
        check("argument", take(child) == 8);
        check("bodyless operator", new ConversionOp(1) + child == 9);
        check("instance operator", new ConversionOp(1) - child == -7);
        var strings:Map<String, ConvertedMapValue> = ["text" => "payload", "number" => 12];
        check("StringMap text", strings["text"].render() == "payload");
        check("StringMap number", strings["number"].render() == "12");
        var ints:Map<Int, ConvertedMapValue> = [1 => "int key", 2 => 17];
        check("IntMap text", ints[1].render() == "int key");
        check("IntMap number", ints[2].render() == "17");
        var key = {};
        var objects:Map<{}, ConvertedMapValue> = [key => "object key"];
        check("ObjectMap", objects[key].render() == "object key");
        Sys.println("CONFORMANCE_OK");
    }
}
