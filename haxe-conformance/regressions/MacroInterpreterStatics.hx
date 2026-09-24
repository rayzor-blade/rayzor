// Macro-time statics persist across calls (each call site expands anew),
// including `#if macro` members; comprehensions, `.match(pattern)`,
// extractor/object patterns, Reflect and Type, and a local function that
// calls itself, inside macro bodies.
import haxe.macro.Expr;

class MacroInterpreterStatics {
    #if macro
    static var stored:Int = 0;
    #end
    static var counter = 10;

    static macro function bump() { counter++; return macro $v{counter}; }
    static macro function put() { stored = 42; return macro null; }
    static macro function take() { return macro $v{stored}; }
    static macro function squares() return macro $v{[for (i in 1...4) i * i].join(",")};
    static macro function isCall(e:Expr) return macro $v{e.expr.match(ECall(_, [_]))};
    static macro function fieldName(e:Expr) {
        return switch e.expr {
            case EField({expr: EConst(CIdent(base))}, name): macro $v{base + "." + name};
            case _: macro "?";
        }
    }
    static macro function reflected() {
        var o = {a: 1, b: 2};
        var names = Reflect.fields(o);
        names.sort(Reflect.compare);
        return macro $v{names.join(",") + ":" + Reflect.field(o, "b")};
    }
    static macro function depth() {
        function down(n:Int):Int return n == 0 ? 0 : 1 + down(n - 1);
        return macro $v{down(4)};
    }

    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function main() {
        var first = bump();
        var second = bump();
        check("statics", first + " " + second, "11 12");
        put();
        check("#if macro static", "" + take(), "42");
        check("comprehension", squares(), "1,4,9");
        check("match", isCall(trace(1)) + " " + isCall(x), "true false");
        check("patterns", fieldName(a.b), "a.b");
        check("reflect", reflected(), "a,b:2");
        check("local recursion", "" + depth(), "4");
        Sys.println("CONFORMANCE_OK");
    }
}
