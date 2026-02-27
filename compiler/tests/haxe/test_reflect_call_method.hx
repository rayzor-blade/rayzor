class Main {
    static function add(a:Int, b:Int):Int {
        return a + b;
    }

    static function main() {
        trace(Reflect.callMethod(null, add, [20, 22])); // 42

        var scale = 3;
        var mul = function(v:Int):Int {
            return v * scale;
        };
        trace(Reflect.callMethod(null, mul, [7])); // 21

        var varFn:Dynamic = Reflect.makeVarArgs(function(args:Array<Dynamic>):Dynamic {
            return args;
        });
        trace(Reflect.isFunction(varFn)); // true
    }
}
