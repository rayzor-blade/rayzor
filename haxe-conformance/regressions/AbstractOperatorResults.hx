abstract Measure(Int) from Int {
    @:op(A / B) static function divide(a:Measure, b:Measure):Float
        return (cast a:Int) / (cast b:Int);
    @:op(A + B) static function addInt(a:Measure, b:Int):Measure
        return (cast a:Int) + b;
    @:commutative @:op(A + B) static function addFloat(a:Measure, b:Float):Float
        return (cast a:Int) + b;
    @:op(A == B) static inline function equal<T:Float>(a:Measure, b:T):Bool
        return (cast a:Float) == b;
}

class AbstractOperatorResults {
    static var order = "";
    static function fraction():Float { order += "f"; return 0.5; }
    static function measure():Measure { order += "m"; return 3; }
    static function show<T>(a:T, b:T):String return Std.string(a) + "/" + Std.string(b);
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        var three:Measure = 3;
        var two:Measure = 2;
        var divided = three / two;
        check(show(divided, 1.5) == "1.5/1.5", "declared asymmetric result");
        var added = three + 0.5;
        check(show(added, 3.5) == "3.5/3.5", "floating-point overload");
        var integer:Measure = three + 2;
        check((cast integer:Int) == 5, "integer overload");
        check(fraction() + measure() == 3.5 && order == "fm", "commutative evaluation order");
        check(three == 3.0 && three == 3 && !(three == 3.5), "generic inline operator operands");
        order = "";
        check(measure() == fraction() + 2.5 && order == "mf", "generic operands evaluated once");
        var b:UInt = 50000;
        var a:UInt = b * b;
        check(show(a / b, 50000) == "50000/50000", "unsigned division result");
        check(a % b == 0 && a > b && b < a, "unsigned arithmetic");
        check(b == 50000.0 && a != 1.0, "unsigned floating-point equality");
        check(a > 1.0 && a >= 1.0 && !(a < -1.0) && !(a <= 1.0), "unsigned floating-point comparison");
        Sys.println("CONFORMANCE_OK");
    }
}
