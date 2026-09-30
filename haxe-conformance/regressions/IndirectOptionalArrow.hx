class IndirectOptionalArrow {
    static function main() {
        var plain: ?Int -> String -> Int = function (a:Int = 1, b:String) return a + b.length;
        if (plain("--") != 3) throw "function optional argument";

        var arrow: ?Int -> String -> Int = (a:Int = 1, b:String) -> a + b.length;
        if (arrow("--") != 3) throw "arrow optional argument";

        var castArrow: () -> Int = cast (() -> 1:() -> Int);
        if (castArrow() != 1) throw "cast arrow annotation";

        Sys.println("CONFORMANCE_OK");
    }
}
