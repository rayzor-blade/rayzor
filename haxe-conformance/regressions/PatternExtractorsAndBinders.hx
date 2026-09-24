// Extractors (`expr => pattern`, chained and nested), binders
// (`name = pattern`), object patterns and constructors nested inside them;
// a return inside a switch types the function; array literals typed
// Array<Dynamic> hold boxes, nested literals included.
private enum K { Album(t:Array<String>); Single(b:String); }
private class C {
    var x(get, null):String;
    function get_x() return "foo";
    public function new() {}
    public static function viaPattern() {
        var m = new C();
        switch (m) {
            case {x: x = "foo"}: return x;
            case _: return null;
        }
    }
}
class PatternExtractorsAndBinders {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function nested(a:Array<Dynamic>) {
        return switch (a) {
            case [["a"]]: "match";
            default: "no match";
        }
    }
    static function main() {
        var s = "hello";
        check("extractor", (switch s { case _.length => 5: "five"; case _: "other"; }) + " " + (switch s { case _.toUpperCase() => up: up; }), "five HELLO");
        check("chain", switch 7 { case _ * 2 => _ + 1 => 15: "chain"; case _: "no"; }, "chain");
        check("binder", (switch (null) { case v = null: "null:" + v; case _: "no"; }) + " " + (switch 3 { case n = 1 | 3: "odd " + n; case _: "no"; }), "null:null odd 3");
        var o = {s: "abc", n: 2};
        check("object", (switch o { case {s: "abc"}: "const"; case _: "no"; }) + " " + (switch o { case {s: s, n: 2}: "cap " + s; case _: "no"; }) + " " + (switch o { case {s: _.length => 3, n: n}: "len " + n; case _: "no"; }), "const cap abc len 2");
        check("array", (switch [1, 2] { case [a, _ * 10 => 20]: "ok " + a; case _: "no"; }) + " " + (switch [2, 1] { case [a = 1 | 2, b = 2 | 1]: '$a+$b'; case _: "_"; }), "ok 1 2+1");
        var k = Album(["a"]);
        check("constructor", (switch {kind: k, y: 1960} { case {y: _ < 1970 => true, kind: kind = Album(tracks)}: "album " + tracks.length; case _: "no"; }) + " " + (switch k { case kk = Single(b): b; case _: "not single"; }), "album 1 not single");
        check("return in switch", C.viaPattern(), "foo");
        var single = Single("x");
        check("string argument", (switch single { case Single("x"): "yes"; case _: "no"; }) + " " + single.match(Single('y')), "yes false");
        var d:Array<Dynamic> = ["x", "yz"];
        var n:Array<Dynamic> = [[1, 2]];
        check("dynamic literals", d[1].length + " " + n[0].length + " " + nested([["a"]]) + " " + nested([["b"]]) + " " + nested([[]]), "2 2 match no match no match");
        Sys.println("CONFORMANCE_OK");
    }
}
