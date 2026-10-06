class ConcreteArithmeticResults {
    static function main() {
        var a:ConcretePoint = {x: 2.5};
        var b:ConcretePoint = {x: 1.0};
        var difference:ConcreteVector = a - b;
        if (difference.x != 1.5) throw "forwarded Float difference";
        var owner = new ConcreteMetricOwner();
        owner.width += 1.5;
        if (owner.width.dip != 1.5) throw "compound Float property";
        owner.width += 2.5;
        if (owner.width.dip != 4.0) throw "repeated Float property";
        var functionForm:Int->(Int->(Int->Int)) =
            function(a) return function(b) return function(c) return a + b + c;
        var arrowForm:Int->(Int->(Int->Int)) = a -> b -> c -> a + b + c;
        if (functionForm(1)(2)(3) != 6 || arrowForm(1)(2)(3) != 6) throw "curried scalar result";
        var curried = new ConcreteCurriedFunctions();
        curried.functionForm = function(a) return function(b) return function(c) return a + b + c;
        curried.arrowForm = a -> b -> c -> a + b + c;
        if (curried.functionForm(1)(2)(3) != 6 || curried.arrowForm(1)(2)(3) != 6) throw "assigned curried result";
        if (haxe.crypto.Sha1.encode("olá") != "3e16889d494632f3c528fb3a3756435f8aa88788") throw "integer hash arithmetic";
        Sys.println("CONFORMANCE_OK");
    }
}

private class ConcreteCurriedFunctions {
    public var functionForm:Int->(Int->(Int->Int));
    public var arrowForm:Int->(Int->(Int->Int));
    public function new() {}
}

@:structInit private final class ConcretePointData {
    public var x:Float;
}
@:forward private abstract ConcretePoint(ConcretePointData) from ConcretePointData to ConcretePointData {
    @:op(A - B) public function subtract(other:ConcretePoint):ConcreteVector {
        return {x: this.x - other.x};
    }
}
@:structInit private final class ConcreteVectorData {
    public var x:Float;
}
@:forward private abstract ConcreteVector(ConcreteVectorData) from ConcreteVectorData to ConcreteVectorData {
    @:op(A - B) public function subtract(other:ConcreteVector):ConcreteVector {
        return {x: this.x - other.x};
    }
}
private class ConcreteMetricOwner {
    public var width(get, set):ConcreteMetric;
    var stored:ConcreteMetric;
    public function new() stored = new ConcreteMetricData();
    function get_width() return stored;
    function set_width(value:ConcreteMetric) return stored = value;
}
@:forward private abstract ConcreteMetric(ConcreteMetricData) from ConcreteMetricData to ConcreteMetricData {
    @:from static function fromFloat(value:Float) {
        var result = new ConcreteMetricData();
        result.dip = value;
        return cast result;
    }
    @:to function toFloat() return this.dip;
    @:op(A += B) static function increment(value:ConcreteMetric, amount:Float) return value.dip += amount;
}
private class ConcreteMetricData {
    public var dip:Float = 0;
    public function new() {}
}
