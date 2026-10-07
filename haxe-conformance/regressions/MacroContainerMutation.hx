import haxe.macro.Context;
import haxe.macro.Expr;
class MacroContainerMutation {
    #if macro
    static var values = [0];
    static var labels:Map<Int, String> = new Map();
    static var record = {value:0};
    #end
    macro static function update():Expr {
        values[0] = values[0] + 1;
        labels[1] = "one";
        record.value = record.value + 1;
        if (values[0] != 1) Context.error("array mutation disappeared", Context.currentPos());
        if (labels[1] != "one") Context.error("map mutation disappeared", Context.currentPos());
        if (record.value != 1) Context.error("object mutation disappeared", Context.currentPos());
        return macro true;
    }
    static function main() {
        if (!update()) throw "macro containers";
        Sys.println("CONFORMANCE_OK");
    }
}
