private class ReflectionParent {
    public var value:Int = 4;
    public function new() {}
    public function inherited(v:Int):Int return value + v;
}
private class ReflectionChild extends ReflectionParent {
    public function new() super();
    function values(v:Array<Int>):Int return v[0] + value;
    function fraction(v:Float):Float return v + 0.5;
    public function check():Void {
        var names = ["values", "fraction", "inherited"];
        for (i in 0...names.length) {
            var method = Reflect.field(this, names[i]);
            if (!Reflect.isFunction(method)) throw "computed method";
            switch (i) {
                case 0: if (Reflect.callMethod(this, method, [[3]]) != 7) throw "Array call";
                case 1: if (Reflect.callMethod(this, method, [1.5]) != 2.0) throw "Float call";
                case 2: if (Reflect.callMethod(this, method, [6]) != 10) throw "inherited call";
            }
        }
        var fields = Reflect.fields(this);
        if (fields.length != 1 || fields[0] != "value") throw "stored instance fields";
        if (Type.getInstanceFields(ReflectionChild).indexOf("fraction") == -1) throw "declared instance methods";
    }
}
class ReflectComputedMethodNames {
    static function main() {
        new ReflectionChild().check();
        var integer = haxe.Int64.make(1, -1);
        var copied:haxe.Int64 = haxe.Unserializer.run(haxe.Serializer.run(integer));
        if (copied != integer) throw "Int64 field serialization";
        Sys.println("CONFORMANCE_OK");
    }
}
