private enum NumberValue { Shared(value:Int); }
private enum TextValue { Shared(value:String); }

class EnumConstructorIdentity {
    static function main() {
        var number = NumberValue.Shared(7);
        var text = TextValue.Shared("seven");
        if (Type.enumIndex(number) != 0 || Type.enumIndex(text) != 0) throw "constructor index";
        switch number {
            case Shared(value): if (value != 7) throw "integer payload";
        }
        switch text {
            case Shared(value): if (value != "seven") throw "string payload";
        }
        if (Type.getEnumConstructs(NumberValue).join(",") != "Shared") throw "number constructors";
        if (Type.getEnumConstructs(TextValue).join(",") != "Shared") throw "text constructors";
        var printer = new haxe.macro.Printer();
        if (printer.printComplexType(macro :Int -> String) != "Int -> String") throw "function type payload";
        Sys.println("CONFORMANCE_OK");
    }
}
