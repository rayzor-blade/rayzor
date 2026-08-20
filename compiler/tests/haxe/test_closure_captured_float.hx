// A Float captured by a closure used to arrive as its own bit pattern read as
// an integer: 2.5 came out 4612811918334230528 and `* 2.0` saturated to
// i64::MAX, silently, on both backends. The environment stores every capture in
// an i64 slot, and floats were exempted from the read-back cast — so the body
// did integer arithmetic on the bits.
class test_closure_captured_float {
    static function run(f:Void->Void):Void { f(); }
    static function take(v:Float):Float { return v * 2.0; }

    static function fromParam(p:Float):Void {
        run(function():Void { Sys.println("  param        " + p + " *2=" + (p * 2.0)); });
    }
    static function main():Void {
        var lit:Float = 2.5;
        var expr:Float = 1.0 + 1.5;
        var sq:Float = Math.sqrt(6.25);
        var neg:Float = -0.75;
        fromParam(2.5);
        run(function():Void {
            Sys.println("  literal      " + lit + " *2=" + (lit * 2.0));
            Sys.println("  expression   " + expr + " *2=" + (expr * 2.0));
            Sys.println("  Math.sqrt    " + sq + " *2=" + (sq * 2.0));
            Sys.println("  negative     " + neg + " *2=" + (neg * 2.0));
            Sys.println("  via static   " + take(lit));
        });
    }
}
