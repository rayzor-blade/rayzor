class MacroTools {
    macro static function compileTimeAdd(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }

    macro static function doubled(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * 2;
    }
}

class Main {
    static function main() {
        trace(MacroTools.compileTimeAdd(100, 200));
        trace(MacroTools.doubled(21));
        trace("done");
    }
}
