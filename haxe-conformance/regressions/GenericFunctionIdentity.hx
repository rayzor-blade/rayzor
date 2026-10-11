class GenericFunctionIdentity {
    static function alias<T>(value:T):T return value;
    static function same<T>(left:T, right:T):Bool return left == right;
    static function callback():Void {}
    static function main() {
        var callable:Dynamic = callback;
        var returned = alias(callable);
        if (!Reflect.isFunction(returned)) throw "generic function type";
        if (!same(callable, returned)) throw "generic function equality";
        if (!Reflect.compareMethods(callable, returned)) throw "generic function identity";
        returned();
        Sys.println("CONFORMANCE_OK");
    }
}
