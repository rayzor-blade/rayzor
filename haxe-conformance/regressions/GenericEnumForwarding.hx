import haxe.macro.Context;
using haxe.macro.TypeTools;

private enum Payload<T> {
    Text(value:String):Payload<String>;
    Integer(value:String):Payload<Int>;
    Number(value:String):Payload<Float>;
}
private enum Holder<T> {
    Hold(value:Payload<T>);
}
class GenericEnumForwarding {
    static function read<T>(value:Payload<T>):T {
        return switch (value) {
            case Text(text): text;
            case Integer(text): Std.parseInt(text);
            case Number(text): Std.parseFloat(text);
        };
    }
    static function forward<U>(value:Holder<U>):U {
        return switch (value) { case Hold(inner): read(inner); };
    }
    static function arrayValue<U>(values:Array<Payload<U>>):U {
        return read(values[0]);
    }
    static function recursive<V>(value:Payload<V>, depth:Int):V {
        return depth == 0 ? read(value) : recursive(value, depth - 1);
    }
    static function equal<T>(left:T, right:T):Bool {
        return left == right;
    }
    static function dynamicEqual<W>(boxed:Dynamic, value:W):Bool {
        return equal(boxed, value);
    }
    static macro function same(value:haxe.macro.Expr, expected:haxe.macro.Expr):haxe.macro.Expr {
        return macro $v{Context.typeof(value).toString() == Context.typeof(expected).toString()};
    }
    static function main() {
        var number = forward(Hold(Number("2.5")));
        if (!same(number, (null:Float)) || number != 2.5) throw "forwarded Float";
        var integer = forward(Hold(Integer("23")));
        if (!same(integer, (null:Int)) || integer != 23) throw "forwarded Int";
        var text = forward(Hold(Text("two")));
        if (!same(text, (null:String)) || text != "two") throw "forwarded String";
        if (arrayValue([Number("3.5")]) != 3.5) throw "nested array Float";
        if (arrayValue([Text("three")]) != "three") throw "nested array String";
        if (recursive(Number("4.5"), 3) != 4.5) throw "recursive Float";
        if (recursive(Text("four"), 3) != "four") throw "recursive String";
        if (!dynamicEqual((2.5:Dynamic), 2.5)) throw "Dynamic Float argument";
        if (!dynamicEqual(("five":Dynamic), "five")) throw "Dynamic String argument";
        var payload = Number("5.5");
        if (!dynamicEqual((payload:Dynamic), payload)) throw "Dynamic enum argument";
        trace("CONFORMANCE_OK");
    }
}
