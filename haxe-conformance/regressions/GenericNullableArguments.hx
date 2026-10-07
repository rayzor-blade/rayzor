typedef MaybeInteger = Null<Int>;
typedef MaybeNumber = Null<Float>;
class GenericNullableArguments {
    static var calls:Int = 0;
    static function integer():Null<Int> { calls++; return 2; }
    static function equal<T>(a:T, b:T):Bool return a == b;
    static function show<T>(a:T, b:T):String return Std.string(a) + "/" + Std.string(b);
    static function evaluate(callback:() -> Null<Float>):Null<Float> return callback();
    static function factory():() -> Null<Int> return () -> 0;
    static function check(value:Bool, label:String):Void { if (!value) throw label; }
    static function main():Void {
        var i:Null<Int> = 2;
        var f:Null<Float> = 2.0;
        check(equal(f, i), "mixed nullable numeric equality");
        check(show(f, i) == "2/2", "mixed nullable numeric formatting");
        check(equal(i, 2) && equal(f, 2.0), "nullable and concrete numeric equality");
        var other:Null<Float> = 2.0;
        check(equal(f, other) && show(f, other) == "2/2", "same nullable numeric type");
        var absent:Null<Int> = null;
        check(!equal(0.0, absent), "null differs from zero");
        check(equal((null:Null<Float>), absent), "mixed nullable null equality");
        other = 0;
        check(!equal((null:Null<Float>), other), "nullable null differs from nullable zero");
        var aliasInt:MaybeInteger = -3;
        var aliasFloat:MaybeNumber = -3.0;
        check(equal(aliasFloat, aliasInt), "nullable numeric aliases");
        check(equal(f, integer()) && calls == 1, "argument evaluated once");
        var fractional:Null<Float> = 2.5;
        check(!equal(fractional, i) && show(fractional, i) == "2.5/2", "fractional payload");
        var changes = 0;
        function local():Null<Int> { changes++; return 2; }
        f = local();
        check(equal(2.0, f) && changes == 1, "nullable local function return");
        var nullable:() -> Null<Float> = () -> 0.0;
        f = nullable();
        check(f != null && equal(f, 0.0), "nullable arrow result");
        nullable = () -> null;
        check(nullable() == null, "nullable arrow null");
        check(evaluate(() -> 1.5) == 1.5, "nullable callback result");
        var zero = factory()();
        check(zero != null && zero == 0, "returned nullable integer closure");
        var boolean:() -> Null<Bool> = () -> false;
        var value = boolean();
        check(value != null && value == false, "nullable boolean closure");
        Sys.println("CONFORMANCE_OK");
    }
}
