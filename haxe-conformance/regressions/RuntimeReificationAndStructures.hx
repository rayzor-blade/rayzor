// `macro e` outside a macro builds the Expr at run time; qualified enum
// constructors and values; a local function names itself inside nested
// closures; a function literal applies its parameter defaults; generic
// calls infer type arguments through Iterable; object literals omitting a
// typedef's optional fields read back through the typedef; a string inside
// an interpolation.
import haxe.macro.Expr;

typedef Node = { var name:String; var ?meta:Array<String>; var ?kids:Array<Node>; }

class RuntimeReificationAndStructures {
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function show(e:Expr):String return switch e.expr {
        case EConst(CIdent(s)): s;
        case EField(o, f): show(o) + "." + f;
        case EMeta({name: ":implicitReturn"}, {expr: EReturn(inner)}): "implicit " + show(inner);
        case _: "?";
    }
    static function kids(n:Node) return n.kids == null ? 0 : n.kids.length;
    static function main() {
        var s = "q";
        check("reify", show(macro $i{s}) + " " + show(macro $e{macro $i{s}}.f), "q q.f");
        check("toFieldExpr", show(haxe.macro.MacroStringTools.toFieldExpr(["a", "b", "c"])), "a.b.c");
        var op = haxe.macro.Expr.Binop.OpAdd;
        var c = haxe.macro.Expr.ExprDef.EConst(haxe.macro.Expr.Constant.CIdent("z"));
        check("qualified", Std.string(op) + " " + show({expr: c, pos: null}), "OpAdd z");
        var inner:Expr = {expr: EConst(CIdent("x")), pos: null};
        check("object in constructor", show({expr: EMeta({name: ":implicitReturn", params: [], pos: null}, {expr: EReturn(inner), pos: null}), pos: null}), "implicit x");
        function loop(depth:Int, n:Int):Int {
            function step(x:Int) return loop(depth + 1, x);
            return n <= 0 ? depth : step(n - 1);
        }
        function fact(n:Int):Int return n <= 1 ? 1 : n * fact(n - 1);
        check("local recursion", loop(0, 3) + " " + fact(5), "3 120");
        var g = (a:Int = 1) -> a;
        var h = (a = 2) -> a;
        check("literal defaults", g() + " " + g(5) + " " + h(), "1 5 2");
        check("fold", Lambda.fold([1, 2, 3], function(x, acc) return acc + x, 0) + " "
            + Lambda.fold(["a", "b"], function(x, acc:String) return acc == null ? x : acc + "." + x, null), "6 a.b");
        check("typedef layout", kids({name: "r", kids: [{name: "k"}]}) + " " + kids({name: "leaf"}), "1 0");
        var foo = "hello";
        var nested = '${'${foo}'}';
        check("interpolation", nested, "hello");
        Sys.println("CONFORMANCE_OK");
    }
}
