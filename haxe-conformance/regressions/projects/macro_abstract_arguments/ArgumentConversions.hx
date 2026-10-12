import haxe.macro.Expr;
import haxe.macro.Context;
import haxe.macro.Type;
using haxe.macro.TypeTools;

abstract ArgumentTypeName(String) to String {
    public inline function new(value:String) this = value;
    @:from public static macro function fromExpression(value:Expr) {
        return macro new ArgumentTypeName($v{Context.typeof(value).toString()});
    }
}

abstract ArgumentArity(Int) to Int {
    public inline function new(value:Int) this = value;
    @:from public static macro function fromExpression(value:Expr) {
        return switch (Context.typeof(value).follow()) {
            case TFun(arguments, _): macro new ArgumentArity($v{arguments.length});
            case _: Context.error("Expected a function", value.pos);
        }
    }
}

abstract ArgumentLength(Int) to Int {
    public inline function new(value:Int) this = value;
    @:from public static macro function fromExpression(value:Expr) {
        return macro new ArgumentLength(($value).length);
    }
}

abstract ArgumentListener<T>(String) to String {
    public inline function new(value:String) this = value;
    @:from public static macro function fromExpression(value:Expr) {
        var typed = Context.typeExpr(value);
        var constructor = switch (typed.expr) {
            case TField(_, FEnum(_, field)): field;
            case _: Context.error("Expected enum constructor", value.pos);
        };
        var expected = switch (Context.getExpectedType().follow()) {
            case TAbstract(_, [type]): type;
            case _: Context.error("Expected listener", value.pos);
        };
        var result = Context.typeof(macro (null : Void));
        var signature = switch (typed.t.follow()) {
            case TFun(arguments, _): TFun(arguments, result);
            case TEnum(_): TFun([], result);
            case _: Context.error("Expected enum constructor type", value.pos);
        };
        Context.unify(expected, signature);
        return macro new ArgumentListener($v{constructor.name});
    }
}
