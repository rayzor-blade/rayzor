// An abstract's `@:from` runs wherever a value of another type is bound to
// it: a static or instance field initializer, a field assignment, a local,
// an argument. A typedef of a generic abstract is that abstract with the
// use's arguments, for `new`, `[]` and its methods. A `null` element in an
// array pattern tests for null.
private abstract A(String) from String to String {
    @:from static function fromInt(v:Int):A return "abc" + v;
}
private class TC {
    static public var a:A = 1;
    public var b:A = 2;
    public var c:A;
    public function new() { c = 3; }
}
private typedef V<T> = haxe.ds.Vector<T>;
private typedef VI = haxe.ds.Vector<Int>;
private class X { public var s:String; public function new(s:String) this.s = s; }
class AbstractFromAndAliases {
    static var s:A = 4;
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function take(a:A):String return a;
    static function two(x1:Null<X>, x2:Null<X>) {
        return switch [x1, x2] {
            case [null, null]: "";
            case [a, null]: a.s;
            case [null, b]: b.s;
            case [a, b]: a.s + b.s;
        }
    }
    static function main() {
        var x:A = 5;
        var t = new TC();
        check("from", (x : String) + " " + take(6) + " " + (s : String) + " " + (t.b : String) + " " + (t.c : String) + " " + (TC.a : String), "abc5 abc6 abc4 abc2 abc3 abc1");
        var u = new VI(3); u[1] = 8;
        var v = new V<Int>(3); v[0] = 1;
        var q:V<String> = new V<String>(2); q[0] = "x";
        check("typedef of Vector", u[1] + " " + u.length + " " + v[0] + " " + v.length + " " + q[0], "8 3 1 3 x");
        check("null array pattern", two(null, null) + "|" + two(new X("a"), null) + "|" + two(null, new X("b")) + "|" + two(new X("a"), new X("b")), "|a|b|ab");
        Sys.println("CONFORMANCE_OK");
    }
}
