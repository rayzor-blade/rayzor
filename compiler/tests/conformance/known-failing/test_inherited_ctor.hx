// A subclass that declares no constructor of its own must still be
// constructible -- `new Sub()` uses the inherited one. It SIGTRAPped with exit
// 133 and no message. This is idiomatic Haxe and the single biggest blocker to
// running the official Haxe test suite: all 451 of its unit.Test subclasses
// declare no constructor of their own.
class Base {
    public var tag:String;
    public function new() { tag = "base"; }
    public function twice(v:Int):Int { return v * 2; }
}

class WithCtor extends Base {
    public function new() { super(); }
    public function thrice(v:Int):Int { return v * 3; }
}

class NoCtor extends Base {
    public function quad(v:Int):Int { return v * 4; }
}

class Main {
    static function main():Void {
        var a = new WithCtor();
        if (a.thrice(3) != 9) Sys.println("FAIL withCtor own method");
        if (a.twice(3) != 6) Sys.println("FAIL withCtor inherited method");
        if (a.tag != "base") Sys.println("FAIL withCtor inherited field");

        var b = new NoCtor();
        if (b.quad(3) != 12) Sys.println("FAIL noCtor own method");
        if (b.twice(3) != 6) Sys.println("FAIL noCtor inherited method");
        if (b.tag != "base") Sys.println("FAIL noCtor inherited field");

        Sys.println("inherited ctor ok");
    }
}
