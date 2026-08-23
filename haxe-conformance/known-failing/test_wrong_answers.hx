// Three standard-Haxe behaviours where rayzor produces a WRONG ANSWER and
// exits 0. Verified against haxe 4.3.6 --interp as the oracle.
//
//   expression-bodied fn   rayzor: <empty>    haxe: shorthand=8
//   switch guard           rayzor: small99    haxe: big99
//   Std.string(anon)       rayzor: null       haxe: {x: 1}
//
// The loop-retained-allocation case is included as a CONTROL: it is correct,
// so it must stay correct.
enum E { Ax; Bx(i:Int); }
class Node { public var v:Int; public function new(v:Int) { this.v = v; } }

class Main {
    static function sh() return 8;

    static function main():Void {
        var a = sh();
        if (("" + a) != "8") Sys.println("FAIL expression-bodied fn: got '" + a + "' want 8");

        var ns = [];
        for (i in 0...5) ns.push(new Node(i));
        var s = "";
        for (n in ns) s += n.v + " ";
        if (s != "0 1 2 3 4 ") Sys.println("FAIL retained loop allocation: got '" + s + "'");

        var e = Bx(99);
        var r = switch (e) {
            case Bx(i) if (i > 10): "big" + i;
            case Bx(i): "small" + i;
            case Ax: "ax";
        };
        if (r != "big99") Sys.println("FAIL switch guard: got " + r + " want big99");

        var anon = Std.string({x:1});
        if (anon != "{x: 1}") Sys.println("FAIL Std.string(anon): got " + anon + " want {x: 1}");

        Sys.println("conformance wrong-answers ok");
    }
}
