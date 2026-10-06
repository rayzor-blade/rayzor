private enum Built {
    C(value:Int, text:String);
}
private enum abstract Constants(Int) {
    var C = 3;
}
private class OtherCallable {
    public static function C(value:Int):Int return value + 1;
}
class EnumConstructorNames {
    static function main() {
        var factory = Built.C;
        var direct = Built.C(7, "x");
        switch (direct) { case C(7, "x"): case _: throw "direct constructor"; }
        switch (factory(8, "y")) { case C(8, "y"): case _: throw "constructor value"; }
        if (OtherCallable.C(4) != 5) throw "static method";
        var method = OtherCallable.C;
        if (method(9) != 10) throw "static method value";
        if ((cast Constants.C : Int) != 3) throw "abstract constant";
        Sys.println("CONFORMANCE_OK");
    }
}
