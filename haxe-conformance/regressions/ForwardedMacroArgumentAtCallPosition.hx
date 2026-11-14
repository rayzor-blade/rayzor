// A macro forwarding its argument to a typing macro at the argument's own
// position: the argument is typed and lowered once, as itself.
private class Holder {
	public static var made:Array<String> = [];

	public final value:String;

	public function new(value:String) {
		this.value = value;
		made.push(value);
	}
}

class ForwardedMacroArgumentAtCallPosition {
	macro static function forward(value:haxe.macro.Expr) {
		return macro @:pos(value.pos) ForwardedMacroArgumentAtCallPosition.make($value);
	}

	macro static function forwardType(value:haxe.macro.Expr) {
		return macro @:pos(value.pos) ForwardedMacroArgumentAtCallPosition.typeName($value);
	}

	macro static function typeName(value:haxe.macro.Expr) {
		return macro $v{haxe.macro.TypeTools.toString(haxe.macro.Context.typeof(value))};
	}

	macro static function make(value:haxe.macro.Expr) {
		var ct = haxe.macro.Context.toComplexType(haxe.macro.Context.typeof(value));
		return macro(cast new Holder(cast $value) : Holder);
	}

	static function main() {
		var b = "ab";
		var joined = forward([for (i in 0...3) b].join("-"));
		var direct = make([for (i in 0...2) b].join(""));
		var called = forward(StringTools.trim("  x  "));
		var noArgs = forward(b.toUpperCase());
		var types = forwardType([b, b].join("")) + "," + forwardType(b.toUpperCase());
		var got = joined.value + "|" + direct.value + "|" + called.value + "|" + noArgs.value + "|" + Holder.made.join(",") + "|" + types;
		if (got != "ab-ab-ab|abab|x|AB|ab-ab-ab,abab,x,AB|String,String")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
