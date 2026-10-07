abstract FloatSlots(Array<Float>) from Array<Float> to Array<Float> {
    public function get():String return "helper";
}
abstract Slots<T>(Array<T>) from Array<T> to Array<T> {}
typedef DecimalSlots = Slots<Float>;
abstract CustomSlots(Array<Float>) {
    public inline function new(values:Array<Float>) this = values;
    public function get():String return "custom helper";
    @:arrayAccess public inline function read(index:Int):Int return Std.int(this[index] * 10);
}

class AbstractArrayElements {
    static function number(value:Float):Float return value;
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        var floats:FloatSlots = [1.5, 2.25];
        check(number(floats[0]) == 1.5 && floats.get() == "helper", "underlying element over helper get");
        var typed:DecimalSlots = [3.5, 4.25];
        var value = typed[0];
        check(number(value) == 3.5 && typed[1] + 0.5 == 4.75, "generic abstract float elements");
        typed[0] = 6.5;
        check(typed[0] == 6.5, "typed element update");
        var integers:Slots<Int> = [0, -3];
        check(integers[0] == 0 && integers[1] == -3, "generic integer elements");
        var strings:Slots<String> = ["hello", ""];
        check(strings[0] == "hello" && strings[1] == "", "generic string elements");
        var custom = new CustomSlots([1.5]);
        check(custom[0] == 15 && custom.get() == "custom helper", "explicit named array accessor");
        Sys.println("CONFORMANCE_OK");
    }
}
