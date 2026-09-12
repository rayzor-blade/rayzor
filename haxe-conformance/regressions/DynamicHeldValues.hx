// A Dynamic is a box, and every reader tells what it holds by the box's tag:
// an array boxes as the runtime's TYPE_ARRAY (it boxed with a context-local
// id no reader knew), a value returned into a Dynamic return type is boxed
// (it came back raw), a field read goes through the tag (`d.length` read a
// class slot off whatever the name matched), and a cast from Dynamic
// unboxes (`cast(d, Foo)` reinterpreted the box as the object).
private class Foo { public var x:Int; public function new() { x = 7; } }
private typedef Tok = { var p:String; var s:Bool; }
class DynamicHeldValues {
    static function get():Dynamic { return [1, 2]; }
    static function typedLen(l:List<Tok>):Int return l.length;
    static function main() {
        var v:Dynamic = get();
        if (v.length != 2) throw "returned array .length";
        if (v.hasNext != null) throw "array has no hasNext";
        if (Reflect.field(v, "hasNext") != null) throw "Reflect.field on an array";
        if (Reflect.fields([1]).length != 0) throw "Reflect.fields on an array";
        var d:Dynamic = new Foo();
        if (cast(d, Foo).x != 7) throw "cast(d, Foo)";
        var g:Foo = cast d;
        if (g.x != 7) throw "unsafe cast to a class";
        if (d.x != 7) throw "field read through a class box";
        if (d.nosuch != null) throw "missing field is null";
        var da:Dynamic = [1, 2, 3];
        if (!Std.isOfType(da, Array)) throw "isOfType Array";
        var a:Array<Int> = cast(da, Array);
        if (a.length != 3) throw "cast(d, Array)";
        var w:Array<Int> = da;
        if (w.length != 3) throw "Dynamic -> Array let";
        var ds:Dynamic = "abc";
        var s2:String = ds;
        if (s2.length != 3) throw "Dynamic -> String let";
        if (ds.length != 3) throw "string .length through Dynamic";
        var o:Dynamic = {x: 9, name: "m"};
        if (o.x != 9 || o.name != "m") throw "anon field read through Dynamic";
        var ctx = {foo: "there"};
        var dc:Dynamic = ctx;
        if (Reflect.fields(dc).length != 1) throw "Reflect.fields on a boxed anon";
        if (Reflect.getProperty(dc, "foo") != "there") throw "getProperty on a boxed anon";
        if (Std.string(ctx) != "{foo: there}") throw "Std.string(anon): " + Std.string(ctx);
        var l = new List<Tok>(); l.add({p: "x", s: true});
        var di:Dynamic = 41;
        if (typedLen(l) != 1) throw "typed list";
        trace("CONFORMANCE_OK");
    }
}
