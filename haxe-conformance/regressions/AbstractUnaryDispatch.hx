private class Calls {
    public static var count:Int = 0;
}
private abstract Counter(Int) from Int to Int {
    @:op(A++) public function increment():Counter {
        Calls.count++;
        return this + 10;
    }
    @:op(-A) public function negate():Int {
        Calls.count++;
        var value = this;
        return -value;
    }
}
private abstract StaticCounter(Int) from Int {
    @:op(-A) public static function negate(value:StaticCounter):Int {
        Calls.count++;
        return -(cast value:Int) - 1;
    }
}
class AbstractUnaryDispatch {
    static function main() {
        var value:Counter = 3;
        var result = value++;
        if (Calls.count != 1 || (result:Int) != 13 || (value:Int) != 3) throw "postfix overload";
        if (-value != -3 || Calls.count != 2) throw "instance unary overload";
        var other:StaticCounter = 4;
        if (-other != -5 || Calls.count != 3) throw "static unary overload";
        trace("CONFORMANCE_OK");
    }
}
