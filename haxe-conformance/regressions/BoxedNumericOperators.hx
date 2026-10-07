typedef MaybeUnaryInt = Null<Int>;

private class NumericReceiver {
    public function new() {}
    public function number(value:Float):Float return value;
    public function integer(value:Int):Int return value;
}

private class NumericChild extends NumericReceiver {
    public function fromDynamic(value:Dynamic):Float return super.number(value);
    public function integerFromDynamic(value:Dynamic):Int return super.integer(value);
}

class BoxedNumericOperators {
    macro static function typeName(e:haxe.macro.Expr) {
        return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(e))};
    }
    static function check(value:Bool, label:String) {
        if (!value) throw label;
    }
    static function main() {
        #if rayzor
        #if !target.static
        throw "missing static target define";
        #end
        #end
        var integer:Null<Int> = 10;
        check(-integer == -10, "nullable negation");
        check(~integer == -11, "nullable complement");
        check(typeName(-integer) == "Int", "nullable negation type");
        check(typeName(~integer) == "Int", "nullable complement type");
        var zero:Null<Int> = 0;
        check(-zero == 0 && ~zero == -1, "nullable zero");
        var number:Null<Float> = 1.5;
        check(-number == -1.5, "nullable float negation");
        check(typeName(-number) == "Float", "nullable float result type");
        var dynamicInteger:Dynamic = 10;
        check(~dynamicInteger == -11, "dynamic complement");
        check(typeName(~dynamicInteger) == "Int", "dynamic complement type");
        check(typeName(-dynamicInteger) == "Float", "dynamic negation type");
        check(dynamicInteger << 1 == 20 && dynamicInteger >> 1 == 5
            && dynamicInteger >>> 1 == 5, "dynamic shifts");
        check((dynamicInteger & 15) == 10 && (dynamicInteger | 15) == 15
            && (dynamicInteger ^ 8) == 2, "dynamic bitwise operators");
        var dynamicShift:Dynamic = 1;
        check(10 << dynamicShift == 20 && 10 >> dynamicShift == 5
            && dynamicInteger >> dynamicShift == 5, "dynamic shift counts");
        var negative:Dynamic = -1;
        check(negative >>> dynamicShift == 2147483647, "unsigned dynamic shift");
        check(typeName(dynamicInteger << 1) == "Int", "dynamic shift type");
        var quotient:Float = dynamicInteger / 2;
        check(quotient == 5.0, "dynamic division value");
        var dynamicDivisor:Dynamic = 2;
        var boxedQuotient:Float = dynamicInteger / dynamicDivisor;
        check(boxedQuotient == 5.0, "two dynamic numeric operands");
        var receiver = new NumericChild();
        check(receiver.number(dynamicInteger / 2) == 5.0, "numeric method argument");
        check(receiver.number(dynamicInteger % 3) == 1.0, "numeric modulo argument");
        check(receiver.integer(dynamicInteger) == 10, "integer method argument");
        check(receiver.fromDynamic(dynamicInteger) == 10.0, "numeric super argument");
        check(receiver.integerFromDynamic(dynamicInteger) == 10, "integer super argument");
        var fields = {integer:integer, number:number};
        check(~fields.integer == -11 && -fields.number == -1.5, "nullable fields");
        var array:Array<Null<Int>> = [10];
        check(~array[0] == -11 && -array[0] == -10, "nullable array elements");
        var alias:MaybeUnaryInt = 4;
        check(~alias == -5 && -alias == -4, "nullable typedef");
        check(typeName(~alias) == "Int", "nullable typedef result type");
        var missing:Null<Int> = null;
        var otherMissing:Null<Int> = null;
        check(missing == otherMissing && !(missing != otherMissing), "null equality");
        check(missing != zero && zero != missing && !(missing == zero), "null and zero");
        #if target.static
        check(!(missing < zero) && !(missing <= zero) && !(missing > zero) && !(missing >= zero), "null ordering");
        check(!(zero < missing) && !(zero <= missing) && !(zero > missing) && !(zero >= missing), "reverse null ordering");
        #end
        var missingFloat:Null<Float> = null;
        var floatZero:Null<Float> = 0.0;
        check(floatZero != missingFloat && !(floatZero <= missingFloat) && !(missingFloat >= floatZero), "nullable float comparison");
        check(floatZero == zero && !(floatZero != zero), "mixed nullable numeric values");
        var missingBool:Null<Bool> = null;
        var falseBool:Null<Bool> = false;
        check(missingBool != falseBool && !(missingBool == falseBool), "null and false");
        Sys.println("CONFORMANCE_OK");
    }
}
