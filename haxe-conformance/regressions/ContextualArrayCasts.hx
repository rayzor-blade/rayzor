class ContextualArrayCasts {
    static function fraction(values:Array<Any>):Float return cast values[1];
    static function add(value:Float):Float return value + 1;

    static function main() {
        var values:Array<Any> = [12, 1.5, false, "text", {field:1}];
        var integer:Int = cast values[0];
        var number:Float = cast values[1];
        var flag:Bool = cast values[2];
        var word:String = cast values[3];
        if (integer != 12 || number != 1.5 || flag || word != "text") {
            throw "contextual array cast";
        }
        if (fraction(values) != 1.5) throw "return cast context";
        if (add(cast values[1]) != 2.5) throw "argument cast context";
        Sys.println("CONFORMANCE_OK");
    }
}
