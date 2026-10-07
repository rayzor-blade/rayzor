class ReflectVarArgs {
    static function main() {
        var calls = 0;
        var count:Dynamic = Reflect.makeVarArgs(function(args:Array<Dynamic>):Int {
            calls++;
            return args.length;
        });
        if (!Reflect.isFunction(count)) throw "function tag";
        if (count() != 0 || count(1, 2) != 2 || count(0,1,2,3,4,5,6,7,8,9) != 10) throw "arity";
        if (calls != 3) throw "capture";
        var echo:Dynamic = Reflect.makeVarArgs((args:Array<Dynamic>) -> args[0]);
        if (echo("text") != "text" || echo(1.5) != 1.5 || echo(true) != true || echo(null) != null) throw "slots";
        if (Reflect.callMethod(null, echo, ["reflect"]) != "reflect") throw "reflect string";
        if (Reflect.callMethod(null, echo, [12]) != 12) throw "reflect int";
        if (Reflect.callMethod(null, echo, [1.25]) != 1.25) throw "reflect float";
        var floats:Array<Float> = [2.5];
        if (Reflect.callMethod(null, echo, floats) != 2.5) throw "reflect stored array";
        var integers:Int->Int = cast count;
        if (integers(5) != 1) throw "typed integer call";
        var floating:Float->Float = cast echo;
        if (floating(3.5) != 3.5) throw "typed float call";
        var text:String->String = cast echo;
        if (text("typed") != "typed") throw "typed string call";
        var noArgs:()->Int = cast count;
        if (noArgs() != 0) throw "typed zero argument call";
        var ignored:Int->Void = cast count;
        ignored(1);
        if (calls != 6) throw "typed Void result";
        var source = [7, 8];
        var result:Array<Int> = echo(source);
        if (result != source || result[1] != 8) throw "array identity";
        var ordinary = (args:Array<Int>) -> args.length;
        if (Reflect.callMethod(null, ordinary, [[1,2,3]]) != 3) throw "ordinary array parameter";
        Sys.println("CONFORMANCE_OK");
    }
}
