// A `Null<Int>` field's slot is a boxed pointer; a store to it must land on
// its own 8-byte slot, not at a byte offset (`gep *u8` was scaled by one),
// which trampled the header and the neighbouring fields. Also: the field's
// initializer runs, reflection sees array/anonymous/class fields as what
// they are, and a Dynamic field read reaches a typed parameter unboxed.
private class Child { public var v = 5; public function new() {} }
private class Holder {
    public var a = 11;
    public var b = 22;
    public var c = 33;
    public var n:Null<Int> = 44;
    public var kid = new Child();
    public var xs = [1, 2, 3];
    public var o = { p: 9 };
    public var s = "str";
    public function new() {}
}
class NullableFieldSlots {
    static function takeS(s:String):String return s + "!";
    static function takeA(a:Array<Int>):Int return a.length;
    static function takeO(o:{p:Int}):Int return o.p;
    static function main() {
        var h = new Holder();
        if (h.a != 11 || h.b != 22 || h.c != 33) throw "neighbours after ctor";
        if (h.n != 44) throw "Null<Int> initializer";
        h.n = 7;
        if (h.a != 11 || h.b != 22 || h.c != 33) throw "neighbours after store";
        if (h.n != 7) throw "Null<Int> store";
        h.n = null;
        if (h.a != 11 || h.n != null) throw "Null<Int> null store";
        if (h.kid.v != 5) throw "class field after Null<Int> stores";
        var d:Dynamic = h;
        if (takeS(d.s) != "str!") throw "Dynamic String field to a String param";
        if (takeA(d.xs) != 3) throw "Dynamic Array field to an Array param";
        if (takeO(d.o) != 9) throw "Dynamic anon field to a structural param";
        var k:Dynamic = Reflect.field(h, "kid");
        if (!Std.isOfType(k, Child)) throw "reflected class field is its class";
        var kc:Child = d.kid;
        if (kc.v != 5) throw "class field through a Dynamic read";
        if (!Std.isOfType(Reflect.field(h, "xs"), Array)) throw "reflected array field is an Array";
        var xs:Array<Int> = Reflect.field(h, "xs");
        if (xs[2] != 3) throw "reflected array field indexes";
        var o:{p:Int} = Reflect.field(h, "o");
        if (o.p != 9) throw "reflected anon field";
        Sys.println("CONFORMANCE_OK");
    }
}
