// An abstract argument converts through its @:to method when the parameter is
// the type that method returns, including a type parameter a sibling argument
// fixed, also on a method inherited from a parent class. An unannotated @:to
// toString returns String.
private abstract Meters(Float) from Float {
    @:to inline public function toString() { return '$this(m)'; }
}
private abstract A(Int) from Int to Int {
    @:to public function toString():String { return "A" + Std.string(this); }
}
private class Checker {
    public function new() {}
    public function same<T>(v:T, v2:T):Bool return v == v2;
}
private class SubChecker extends Checker {
    public function run(m:Meters):Bool return same("7(m)", m);
}
class AbstractToAtCallArgs {
    static function same<T>(v:T, v2:T):Bool return v == v2;
    static function takeS(s:String):String return "S:" + s;
    static function main() {
        var acc:Meters = 100.0;
        if (!same("100(m)", acc)) throw "generic sibling @:to";
        var a:A = 2;
        if (takeS(a) != "S:A2") throw "String parameter @:to";
        if (!same("A2", a)) throw "generic @:to on annotated method";
        var n:Int = a;
        if (n + 1 != 3) throw "to Int clause";
        if (!new SubChecker().run(7.0)) throw "inherited generic method @:to";
        trace("CONFORMANCE_OK");
    }
}
