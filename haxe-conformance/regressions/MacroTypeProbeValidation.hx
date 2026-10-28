import haxe.macro.Context;
import haxe.macro.Expr;

class MacroTypeProbeValidation {
    static macro function invalid(value:Expr):Expr {
        var failed = false;
        try Context.typeExpr(value) catch (_:Dynamic) failed = true;
        return macro $v{failed};
    }

    static function main() {
        var value:Int = 1;
        if (invalid(value = 2)) throw "valid assignment";
        if (!invalid(value = "wrong")) throw "invalid assignment";
        if (!invalid((42 : String))) throw "invalid annotation";
        if (!invalid({ var result:Int = "wrong"; result; })) throw "invalid block";
        if (invalid({ var result:Int = 42; result; })) throw "valid block";
        Sys.println("CONFORMANCE_OK");
    }
}
