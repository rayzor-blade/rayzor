import haxe.macro.Expr;
class MacroComprehensionRoundTrip {
    static macro function relocated(value:Expr):Expr {
        value.pos = (macro here).pos;
        return value;
    }
    static function main() {
        var values = relocated([for (i in 0...4) if (i % 2 == 0) i * 3]);
        if (values.join(",") != "0,6") throw "filtered array comprehension";
        var nested = relocated([for (i in 0...2) for (j in 0...2) i + j]);
        if (nested.join(",") != "0,1,1,2") throw "nested comprehension";
        var mapped = relocated([for (i in 0...4) if (i > 1) i => i * 3]);
        if (mapped.exists(0) || mapped.get(2) != 6 || mapped.get(3) != 9) throw "filtered map comprehension";
        Sys.println("CONFORMANCE_OK");
    }
}
