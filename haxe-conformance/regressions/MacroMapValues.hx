import haxe.macro.Context;
import haxe.macro.Expr;

class MacroMapValues {
    static macro function maps():Expr {
        var first = {x:1};
        var second = {x:1};
        var objects = [first => 2, second => 3, first => 4];
        if (objects.get(first) != 4 || objects[second] != 3) throw "macro object identity";
        objects[first] = 5;
        var count = 0;
        var sum = 0;
        for (key => value in objects) {
            count++;
            sum += key.x + value;
        }
        if (count != 2 || sum != 10) throw "macro map iteration";
        var position = Context.currentPos();
        var integers = Context.makeExpr([1 => 2, -3 => 4], position);
        var strings = Context.makeExpr(["first" => 6, "second" => 7], position);
        var objectValues = Context.makeExpr(objects, position);
        var generated = Context.makeExpr([for (index in 1...4) index => index * 2], position);
        return macro {
            integers:$integers,
            strings:$strings,
            objects:$objectValues,
            generated:$generated
        };
    }

    #if !macro
    static function main() {
        var values = maps();
        if (values.integers[1] != 2 || values.integers[-3] != 4) throw "integer map reification";
        if (values.strings["first"] != 6 || values.strings["second"] != 7) throw "string map reification";
        var count = 0;
        var sum = 0;
        for (key => value in values.objects) {
            count++;
            sum += key.x + value;
        }
        if (count != 2 || sum != 10) throw "object map reification";
        if (values.generated[1] != 2 || values.generated[3] != 6) throw "map comprehension reification";
        Sys.println("CONFORMANCE_OK");
    }
    #end
}
