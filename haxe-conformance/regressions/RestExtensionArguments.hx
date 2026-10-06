using RestExtensionArguments.RestExtensions;

private class RestExtensions {
    public static function appendValues<T>(values:Array<T>, ...tail:T):Void {
        for (value in tail) values.push(value);
    }

    public static function sumAfter(values:Array<Float>, offset:Float, ...tail:Float):Float {
        var result = offset;
        for (value in values) result += value;
        for (value in tail) result += value;
        return result;
    }
}

class RestExtensionArguments {
    static var receiverCalls = 0;
    static var argumentCalls = 0;
    static var shared = [1];

    static function receiver():Array<Int> {
        receiverCalls++;
        return shared;
    }

    static function argument():Int {
        argumentCalls++;
        return 2;
    }

    static function check(label:String, actual:String, expected:String):Void {
        if (actual != expected) throw label + ": " + actual + " != " + expected;
    }

    static function main() {
        var integers = [1, 3, 4];
        integers.appendValues(5, 6, 7);
        check("generic extension", integers.join(","), "1,3,4,5,6,7");
        integers.appendValues();
        integers.appendValues(...[8, 9]);
        check("empty and spread", integers.join(","), "1,3,4,5,6,7,8,9");
        RestExtensions.appendValues(integers, 10, 11);
        check("static rest", integers.join(","), "1,3,4,5,6,7,8,9,10,11");

        var floats = [1.5];
        var integer = 2;
        floats.appendValues(integer, 3.25);
        check("float extension", floats.join(","), "1.5,2,3.25");
        check("fixed and rest floats", Std.string(floats.sumAfter(1, integer, 4)), "13.75");

        function rest<T>(...values:T):Array<T> return values.toArray();
        var inferred = rest(integer, 6.25, 7);
        check("generic numeric rest", inferred.join(","), "2,6.25,7");
        check("generic string rest", rest("a", "b").join(","), "a,b");

        receiver().appendValues(argument(), 3);
        check("receiver evaluated once", Std.string(receiverCalls), "1");
        check("argument evaluated once", Std.string(argumentCalls), "1");
        check("evaluation result", shared.join(","), "1,2,3");
        Sys.println("CONFORMANCE_OK");
    }
}
