// A macro returning `$a{entries}` whose every entry is `key => value` builds a
// map literal, as the same entries written in source do.
#if macro
import haxe.macro.Expr;
#end

class MacroSplicedMapLiteral {
	#if !macro
	static final names:Map<Int, String> = all();
	#end

	static macro function all():Expr {
		var entries:Array<Expr> = [];
		for (i in 0...5)
			entries.push(macro $v{i * 10} => $v{"p" + i});
		return macro $a{entries};
	}

	static macro function list():Expr {
		var entries:Array<Expr> = [for (i in 0...3) macro $v{i + 1}];
		return macro $a{entries};
	}

	static function main() {
		#if !macro
		var plain:Array<Int> = list();
		var got = [names.get(30), names.get(0), Std.string(names.get(7)), Std.string(names.exists(40)), Std.string(plain.length)].join(",");
		if (got != "p3,p0,null,true,3")
			throw got;
		Sys.println("CONFORMANCE_OK");
		#end
	}
}
