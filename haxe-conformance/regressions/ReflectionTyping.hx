import haxe.macro.Context;
import haxe.macro.Expr;
import Type as ReflectionType;

class ReflectionTyping {
    public static macro function typeName(expression:Expr):Expr {
        var name = haxe.macro.TypeTools.toString(Context.typeof(expression));
        return macro $v{name};
    }

    #if !macro
    static function main() {
        if (typeName(Type.typeof(12)) != "ValueType") throw "reflection return type";
        if (typeName(ReflectionType.typeof(12)) != "ValueType") throw "imported reflection type";
        if (typeName(Type.enumEq(Type.typeof(12), Type.ValueType.TInt)) != "Bool") {
            throw "enum comparison type";
        }
        Sys.println("CONFORMANCE_OK");
    }
    #end
}
