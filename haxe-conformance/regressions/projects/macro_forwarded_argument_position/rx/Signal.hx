package rx;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;
#end

abstract Signal<T>(Dynamic) {
	macro public static function make(initial:Expr):Expr {
		var kind = haxe.macro.TypeTools.toString(Context.follow(Context.typeof(initial)));
		return macro new rx.Box(Std.string($initial), $v{kind});
	}
}
