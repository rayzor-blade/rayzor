import haxe.macro.Context;
import haxe.macro.Expr;

class MacroReadonlyViews {
    static macro function view(value:Expr):Expr {
        var syntax = Context.toComplexType(Context.typeof(value));
        return macro (null:$syntax);
    }
    static macro function rejects(value:Expr):Expr {
        var failed = false;
        try Context.typeof(value) catch (_:Dynamic) failed = true;
        return macro $v{failed};
    }
    static function main() {
        var original:{final value:Int;} = null;
        var copied = view(original);
        if (!rejects(copied.value = 1)) throw "final view";
        var readable:{var value(default, null):Int;} = null;
        var readView = view(readable);
        if (!rejects(readView.value = 1)) throw "property view";
        if (rejects(readView.value)) throw "readable view";
        var writable:{var value(null, default):Int;} = null;
        if (!rejects(writable.value)) throw "hidden getter";
        if (rejects(writable.value = 1)) throw "writable property";
        if (rejects(@:privateAccess writable.value)) throw "private getter access";
        Sys.println("CONFORMANCE_OK");
    }
}
