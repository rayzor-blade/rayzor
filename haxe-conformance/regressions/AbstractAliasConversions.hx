private abstract Cell<T>(T) from T to T {}
private typedef NumberCell = Cell<Int>;

private abstract Label(String) {
    @:from public static function ofString(value:String):Label {
        return cast ("[" + value + "]");
    }

    @:to public function asString():String {
        return this + "!";
    }
}

class AbstractAliasConversions {
    static function number():NumberCell return 7;
    static function raw(value:NumberCell):Int return value;
    static function label():Label return "return";
    static function text(value:Label):String return value;

    static function main() {
        var local:Label = "local";
        var boolean:Cell<Bool> = true;
        var unwrapped:Bool = boolean;
        if (raw(number()) != 7) throw "generic typedef conversions";
        if (!unwrapped) throw "generic local conversions";
        if (text(label()) != "[return]!") throw "return conversion methods";
        if (text(local) != "[local]!") throw "initializer conversion method";
        Sys.println("CONFORMANCE_OK");
    }
}
