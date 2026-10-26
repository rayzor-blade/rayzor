import haxe.macro.Expr;

class MacroInterpolationRoundTrip {
    static macro function relocated(value:Expr):Expr {
        value.pos = (macro here).pos;
        return value;
    }

    static function main() {
        var i = 10;
        if (relocated('foo$i') != "foo10") throw "identifier";
        if (relocated('answer ${i + 2}, $$literal') != "answer 12, $literal") throw "expression";
        if (relocated('line\n$i') != "line\n10") throw "escape";
        Sys.println("CONFORMANCE_OK");
    }
}
