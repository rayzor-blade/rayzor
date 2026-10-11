class AnyRestStringReturns {
    static function first(...args:Any):Any return args[0];
    static function equal<T>(left:T, right:T):Bool return Reflect.compare(left, right) == 0;
    static function main() {
        if (!equal("raw", first("raw"))) throw "raw rest string";
        var boxed:Dynamic = "boxed";
        if (!equal("boxed", first(boxed))) throw "dynamic rest string";
        Sys.println("CONFORMANCE_OK");
    }
}
