import haxe.macro.Expr;

class MacroControlFlowSplices {
    #if macro
    static function values():Expr return macro [1, 2, 3];
    static function enabled():Expr return macro true;
    static function amount():Expr return macro 42;
    #end

    static macro function total():Expr {
        return macro {
            var sum = 0;
            for (value in ${values()}) sum += value;
            if (${enabled()}) sum += ${amount()};
            sum;
        };
    }

    static function main() {
        var result = total();
        if (result != 48) throw "quoted control flow: " + result;
        Sys.println("CONFORMANCE_OK");
    }
}
