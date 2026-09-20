// haxe.ds.Vector is an abstract over rayzor.Vec through a typedef: its
// `@:op([])` accessors, overloaded constructors and erased element types
// all resolve from an importing module. Plain abstracts get constructors
// with bodies and by-arity overloads.
import haxe.ds.Vector;
abstract Doubled(Array<Int>) {
    public inline function new(n:Int) { this = []; for (i in 0...n) this.push(i * 2); }
    public function len():Int return this.length;
    public function at(i:Int):Int return this[i];
}
abstract Filled(Array<Int>) {
    extern overload public inline function new(n:Int) { this = []; for (i in 0...n) this.push(i); }
    extern overload public inline function new(n:Int, v:Int) { this = []; for (i in 0...n) this.push(v); }
    public function at(i:Int):Int return this[i];
    public function len():Int return this.length;
}
class VectorAndAbstractCtors {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var v = new Vector<Int>(3);
        v[0] = 5; v[1] = 1; v[2] = 3;
        check("length", Std.string(v.length), "3");
        check("index", Std.string(v[1]) + Std.string(v.get(2)), "13");
        check("toArray", Std.string(v.toArray()), "[5,1,3]");
        var c = v.copy(); c[0] = 9;
        check("copy", Std.string(c[0]) + Std.string(v[0]), "95");
        check("map", Std.string(v.map(x -> x * 10).toArray()), "[50,10,30]");
        var e = new Vector<Int>(3); e.fill(7);
        check("fill", Std.string(e.toArray()), "[7,7,7]");
        var w = new Vector<Int>(4); Vector.blit(v, 0, w, 1, 3);
        check("blit", Std.string(w[1]) + Std.string(w[3]), "53");
        var f = new Vector<Float>(2); f[1] = 2.5;
        check("float", Std.string(f[1] + 1), "3.5");
        var g = Vector.fromArrayCopy([1.5, 2.5]);
        check("fromArrayCopy", Std.string(g[1] + g.length), "4.5");
        var s = new Vector<String>(2, "z"); s[1] = "q";
        check("string", s[0] + s[1], "zq");
        var d = new Vector<Int>(2, 4);
        check("default", Std.string(d[0] + d[1]), "8");
        var o = new Vector<{x:Int}>(1); o[0] = {x: 42};
        check("object", Std.string(o[0].x), "42");
        check("ctor body", Std.string(new Doubled(3).at(2)) + Std.string(new Doubled(3).len()), "43");
        var one = new Filled(4); var two = new Filled(2, 9);
        check("overloads", Std.string(one.at(2)) + Std.string(one.len()) + Std.string(two.at(1)) + Std.string(two.len()), "2492");
        Sys.println("CONFORMANCE_OK");
    }
}
