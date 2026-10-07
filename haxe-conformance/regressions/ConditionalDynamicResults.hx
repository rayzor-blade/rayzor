typedef ConditionalText = String;
class ConditionalDynamicResults {
    static function text(flag:Bool, value:Dynamic):ConditionalText return flag ? value : "empty";
    static function reversed(flag:Bool, value:Dynamic):String return flag ? "empty" : value;
    static function integer(flag:Bool, value:Dynamic):Int return flag ? value : 9;
    static function floating(flag:Bool, value:Dynamic):Float return flag ? value : 0.25;
    static function boolean(flag:Bool, value:Dynamic):Bool return flag ? value : false;
    static function array(flag:Bool, value:Dynamic):Array<Int> return flag ? value : [1, 2];
    static function object(flag:Bool, value:Dynamic):{value:Int} return flag ? value : {value:9};
    static function callback(flag:Bool, value:Dynamic):Int->Int return flag ? value : ((v:Int) -> v + 1);
    static function switched(value:Int):String return switch value {
        case 1: "first";
        case 2: "second";
        case _: "other";
    };
    static function switchedCallback(value:Int):Int->Int return switch value {
        case 1: (v:Int) -> v + 1;
        case _: (v:Int) -> v * 2;
    };
    static function main() {
        var choose = function(args:Array<Dynamic>):String {
            return args.length > 0 ? args[0] : "empty";
        };
        if (choose(["text"]) != "text" || choose([]) != "empty") throw "callback conditional";
        var arrow:Array<Dynamic>->String = (args:Array<Dynamic>) -> args.length > 0 ? args[0] : "empty";
        if (arrow(["arrow"]) != "arrow" || arrow([]) != "empty") throw "implicit callback return";
        var dynamicText:Dynamic = "direct";
        var direct:()->String = () -> dynamicText;
        if (direct() != "direct") throw "contextual Dynamic callback return";
        var wrapped:Dynamic = Reflect.makeVarArgs(choose);
        if (wrapped("varargs") != "varargs" || wrapped() != "empty") throw "varargs conditional";
        if (text(true, "value") != "value" || text(false, "value") != "empty") throw "string conditional";
        if (reversed(true, "value") != "empty" || reversed(false, "value") != "value") throw "else conditional";
        if (integer(true, 4) != 4 || integer(false, 4) != 9) throw "integer conditional";
        if (floating(true, 1.5) != 1.5 || floating(false, 1.5) != 0.25) throw "float conditional";
        if (!boolean(true, true) || boolean(false, true)) throw "boolean conditional";
        if (array(true, [4, 5])[1] != 5 || array(false, [4, 5])[1] != 2) throw "array conditional";
        if (object(true, {value:4}).value != 4 || object(false, {value:4}).value != 9) throw "anonymous conditional";
        if (callback(true, (v:Int) -> v * 2)(3) != 6 || callback(false, null)(3) != 4) throw "function conditional";
        if (switched(1) != "first" || switched(2) != "second" || switched(3) != "other") throw "switch string representation";
        if (switchedCallback(1)(3) != 4 || switchedCallback(2)(3) != 6) throw "switch function representation";
        Sys.println("CONFORMANCE_OK");
    }
}
