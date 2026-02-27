class Main {
    static function plusOne(x:Int):Int {
        return x + 1;
    }

    static function main() {
        var fn = plusOne;
        trace(Reflect.isFunction(fn)); // true
        trace(Type.typeof(fn)); // 5 (TFunction ordinal)
        trace(Reflect.isFunction(1)); // false
        trace(Reflect.isFunction("x")); // false
    }
}
