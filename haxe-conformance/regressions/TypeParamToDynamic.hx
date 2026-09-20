// A `T` handed to a Dynamic slot boxes as the instance's binding, so
// stringification, Std.isOfType and Type.typeof see the real value; a type
// named as a value (`Std.isOfType(v, Int)`) is the Int abstract, not an
// unknown type.
class Pt {
    public var x:Int;
    public function new(x) this.x = x;
    public function toString() return "Pt(" + x + ")";
}
class Box<T> {
    public var v:T;
    public function new(v:T) this.v = v;
    public function dyn():Dynamic return v;
    public function asDyn():Dynamic { var d:Dynamic = v; return d; }
    public function isInt():Bool return Std.isOfType(v, Int);
    public function isFloat():Bool return Std.isOfType(v, Float);
    public function isStr():Bool return Std.isOfType(v, String);
    public function typ():String return Std.string(Type.typeof(v));
    public function shown():String return TypeParamToDynamic.show(v);
}
class TypeParamToDynamic {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    public static function show(d:Dynamic):String return "<" + Std.string(d) + ">";
    static function toDyn<T>(x:T):Dynamic return x;
    static function main() {
        var i = new Box<Int>(7), f = new Box<Float>(2.5), s = new Box<String>("hi"), b = new Box<Bool>(true), p = new Box<Pt>(new Pt(3));
        check("dyn", i.dyn() + " " + f.dyn() + " " + s.dyn() + " " + b.dyn() + " " + p.dyn(), "7 2.5 hi true Pt(3)");
        check("asDyn", i.asDyn() + " " + f.asDyn() + " " + s.asDyn() + " " + b.asDyn() + " " + p.asDyn(), "7 2.5 hi true Pt(3)");
        check("isInt", i.isInt() + " " + f.isInt() + " " + s.isInt(), "true false false");
        check("isFloat", i.isFloat() + " " + f.isFloat() + " " + s.isFloat(), "true true false");
        check("isStr", i.isStr() + " " + f.isStr() + " " + s.isStr(), "false false true");
        check("typeof", i.typ() + " " + f.typ() + " " + b.typ(), "TInt TFloat TBool");
        check("arg", i.shown() + f.shown() + s.shown() + b.shown() + p.shown(), "<7><2.5><hi><true><Pt(3)>");
        check("static", toDyn(7) + " " + toDyn(2.5) + " " + toDyn("hi") + " " + toDyn(true) + " " + toDyn(new Pt(4)), "7 2.5 hi true Pt(4)");
        var d:Dynamic = i.dyn(); var n:Int = d;
        var e:Dynamic = f.dyn(); var g:Float = e;
        check("back", (n + 1) + " " + (g * 2), "8 5");
        var dy:Dynamic = 7, st:Dynamic = "x", fl:Dynamic = 2.5, pt:Dynamic = new Pt(1), ar:Dynamic = [1];
        check("isOfType Dynamic", Std.isOfType(dy, Int) + " " + Std.isOfType(dy, Float) + " " + Std.isOfType(dy, String) + " " + Std.isOfType(st, String) + " " + Std.isOfType(fl, Int) + " " + Std.isOfType(pt, Pt) + " " + Std.isOfType(ar, Array) + " " + Std.isOfType(ar, Pt), "true true false true false true true false");
        check("isOfType static", Std.isOfType(7, Int) + " " + Std.isOfType(7, Float) + " " + Std.isOfType(7, String) + " " + Std.isOfType("x", String) + " " + Std.isOfType("x", Int) + " " + Std.isOfType(true, Bool) + " " + Std.isOfType(true, Int), "true true false true false true false");
        Sys.println("CONFORMANCE_OK");
    }
}
