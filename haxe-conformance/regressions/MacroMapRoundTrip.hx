import haxe.macro.Expr;
class MacroMapRoundTrip {
    static macro function relocated(value:Expr):Expr {
        value.pos = (macro here).pos;
        return value;
    }
    static function main() {
        var values = relocated([1 => 2, 3 => 4]);
        if (values.get(1) != 2 || values.get(3) != 4) throw "map round trip";
        Sys.println("CONFORMANCE_OK");
    }
}
