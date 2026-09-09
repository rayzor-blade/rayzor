private abstract Number(Int) from Int to Int {
    @:op(A + B) public static function add(a:Number, b:Number):Number { return (a:Int) + (b:Int) + 10; }
    @:op(A < B) public static function less(a:Number, b:Number):Bool { return (a:Int) < (b:Int); }
}
private abstract Bag(Array<String>) from Array<String> {
    @:op(A += B) public function append(value:String):Void { this.push(value); }
}
private abstract Clamped(Float) to Float {
    public inline function new(value:Float) { this = value < 0 ? 0.0 : (value > 1 ? 1.0 : value); }
    @:op(A *= B) public inline function multiply(value:Float):Clamped { return this = new Clamped(this * value); }
}
class AbstractOperators {
    static function main() {
        var a:Number = 2;
        var b:Number = 3;
        if ((a + b : Int) != 15) throw "static overload";
        if (!(a < b)) throw "Bool result";
        var values:Array<String> = [];
        var bag:Bag = values;
        bag += (1 == 2 ? "wrong" : "right");
        if (values.length != 1 || values[0] != "right") throw "compound overload";
        var fraction = new Clamped(0.5);
        fraction *= 3.0;
        if ((fraction:Float) != 1.0) throw "mutating scalar overload";
        trace("CONFORMANCE_OK");
    }
}
