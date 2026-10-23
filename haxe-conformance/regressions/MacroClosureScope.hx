private class Generator {
    public static macro function choose() return macro -100;
}

class MacroClosureScope {
    static function capture(choose:Void->Int):Void->Int {
        return () -> choose();
    }

    static function main() {
        var evaluate = (choose:Void->Int) -> choose();
        if (evaluate(() -> 42) != 42) throw "arrow parameter shadow";
        var other = function(choose:Void->Int) return choose();
        if (other(() -> 43) != 43) throw "function parameter shadow";
        if (capture(() -> 44)() != 44) throw "captured parameter shadow";
        {
            var choose = () -> 45;
            var captured = () -> choose();
            if (captured() != 45) throw "captured variable shadow";
        }
        if (Generator.choose() != -100) throw "qualified macro expansion";
        Sys.println("CONFORMANCE_OK");
    }
}
