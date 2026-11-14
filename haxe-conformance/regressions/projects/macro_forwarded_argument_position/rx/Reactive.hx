package rx;

#if macro
import haxe.macro.Expr;
#end

// Forwards its argument to a typing macro at the argument's own position.
macro function signal(value:Expr):Expr
	return macro @:pos(value.pos) rx.Signal.make($value);
