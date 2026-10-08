import haxe.macro.Context;
import haxe.macro.Expr;

class ProbeOwner {
    static var number:Int;
    static function accept(value:Int):Void {}
    public static function visible():Int return 1;
    public static function ownAccess():Void {
        if (MacroProbeAccess.errorText(number) != null) throw "own private field";
        if (MacroProbeAccess.errorText(accept(number)) != null) throw "own private method";
    }
}

class ProbeChild extends ProbeOwner {
    public static function inheritedAccess():Void {
        if (MacroProbeAccess.errorText(ProbeOwner.number) != null) throw "inherited private field";
    }
}

class MacroProbeAccess {
    @:access(haxe.macro.Error.childErrors)
    public macro static function errorText(expression:Expr):Expr {
        var message = try {
            Context.typeof(expression);
            null;
        } catch (error:haxe.macro.Expr.Error) {
            var text = error.message;
            if (error.childErrors != null) for (child in error.childErrors) text += "\n" + child.message;
            text;
        };
        return macro $v{message};
    }
    #if !macro
    static function main() {
        if (errorText(ProbeOwner.accept) != "Cannot access private field accept") throw "private method";
        if (errorText(ProbeOwner.number) != "Cannot access private field number") throw "private field";
        if (errorText(@:privateAccess ProbeOwner.accept(ProbeOwner.number)) != null) throw "private access";
        var nested = errorText(@:privateAccess ProbeOwner.accept(@:noPrivateAccess ProbeOwner.number));
        if (nested == null || !StringTools.startsWith(nested, "Cannot access private field number\n")
            || !StringTools.endsWith(nested, "For function argument 'value'")) throw "nested private access: " + nested;
        if (errorText(ProbeOwner.visible()) != null) throw "public method";
        if (errorText(Std.random(1)) != null) throw "extern method";
        if (errorText(Math.PI) != null) throw "extern field";
        ProbeOwner.ownAccess();
        ProbeChild.inheritedAccess();
        var value:Any = 12;
        if (errorText(value.field) != "Any has no field field") throw "Any field";
        if (errorText(value[0]) != "Array access is not allowed on Any") throw "Any index";
        if (errorText(value > 1) != "Cannot compare Any and Int") throw "Any comparison";
        if (errorText((value:Int) > 1) != null) throw "promoted Any";
        var dynamicValue:Dynamic = {field:1};
        if (errorText(dynamicValue.field) != null) throw "Dynamic field";
        Sys.println("CONFORMANCE_OK");
    }
    #end
}
