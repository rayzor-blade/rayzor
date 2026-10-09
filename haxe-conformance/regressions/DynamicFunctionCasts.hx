class DynamicFunctionCasts {
    static function erased(value:Dynamic):Dynamic return value;
    static function shifted(value:Float):Float return value + 0.25;
    static function main() {
        var source = shifted;
        var converted:Int->Float = erased(source);
        if (converted(123) != 123.25) throw "integer to floating parameter";
        var again:Int->Float = erased(source);
        if (converted != again) throw "repeated cast identity";
        if (!Reflect.compareMethods(converted, source)) throw "original method identity";
        var roundTrip:Float->Float = erased(converted);
        if (roundTrip(12.5) != 12.75 || roundTrip != source) throw "round trip";
        var functions = [source, roundTrip, source];
        if (functions.indexOf(roundTrip) != 0 || functions.lastIndexOf(roundTrip) != 2) throw "array function identity";
        if (functions.indexOf(roundTrip, -2) != 1 || functions.lastIndexOf(roundTrip, -2) != 1) throw "array search bounds";
        if (!functions.contains(roundTrip) || !functions.remove(roundTrip) || functions.length != 2) throw "array search and removal";
        var unrelated:Float->Float = erased((value:Float) -> value);
        if (functions.contains(unrelated) || roundTrip == unrelated) throw "distinct functions";
        var dynamicView:Dynamic = converted;
        var originalView:Dynamic = source;
        if (dynamicView != originalView || dynamicView != source || source != dynamicView) throw "Dynamic identity";
        if (dynamicView(6) != 6.25) throw "cast back to Dynamic";

        var offset = 1.5;
        var captured:Int->Float = erased((value:Float) -> value + offset);
        if (captured(4) != 5.5) throw "captured environment";
        offset = 2.5;
        if (captured(4) != 6.5) throw "mutable environment";
        var bound:Int->Float = erased(new CastReceiver(3.5).add);
        if (bound(2) != 5.5) throw "bound receiver";

        var mixed:Int->Bool->String->String = erased((value:Float, enabled:Bool, text:String) -> enabled ? text + value : "disabled");
        if (mixed(2, true, "value=") != "value=2" || mixed(1, false, "") != "disabled") throw "mixed parameters";
        var strings:String->String = erased((text:String) -> text.toUpperCase());
        if (strings("cast") != "CAST") throw "String result";
        var arrays:Array<Int>->Array<Int> = erased((values:Array<Int>) -> values);
        var values = [3, 4];
        if (arrays(values) != values || arrays(values)[1] != 4) throw "Array identity and layout";
        var boolean:Int->Bool = erased((value:Float) -> value > 0);
        if (!boolean(2) || boolean(-2)) throw "Bool result";
        var calls = 0;
        var ignored:Int->Void = erased((value:Float) -> { calls++; return value; });
        ignored(9);
        if (calls != 1) throw "Void result";
        var empty:()->Float = erased(() -> 4.5);
        if (empty() != 4.5) throw "zero parameters";
        var missing:Int->Float = erased(null);
        if (missing != null) throw "null function";
        var originalMany = (a:Float,b:Float,c:Float,d:Float,e:Float,f:Float,g:Float,h:Float) -> a+b+c+d+e+f+g+h;
        var many:Int->Int->Int->Int->Int->Int->Int->Int->Float = erased(originalMany);
        if (many(1,2,3,4,5,6,7,8) != 36) throw "stack arguments";
        if (Reflect.callMethod(null, many, [1,2,3,4,5,6,7,8]) != 36) throw "reflected stack arguments";
        var integers = [1,2,3,4,5,6,7,8];
        if (Reflect.callMethod(null, originalMany, integers) != 36) throw "reflected integer slot layout";
        Sys.println("CONFORMANCE_OK");
    }
}

class CastReceiver {
    var offset:Float;
    public function new(offset:Float) this.offset = offset;
    public function add(value:Float):Float return value + offset;
}
