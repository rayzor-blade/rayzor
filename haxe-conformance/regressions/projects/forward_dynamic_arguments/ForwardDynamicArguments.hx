import helpers.Early;
import helpers.Late;
class ForwardDynamicArguments {
    public static function suffix():String return "!";
    static function main() {
        var text:Dynamic = "forward";
        var number:Dynamic = 42;
        var decimal:Dynamic = 1.25;
        var enabled:Dynamic = true;
        if (Early.text(text) != "forward!") throw "String formal";
        if (Early.number(number) != 43) throw "Int formal";
        if (Early.decimal(decimal) != 2.5) throw "Float formal";
        if (!Early.enabled(enabled)) throw "Bool formal";
        if (Late.reverse(text) != "FORWARD") throw "reverse String formal";
        trace("CONFORMANCE_OK");
    }
}
