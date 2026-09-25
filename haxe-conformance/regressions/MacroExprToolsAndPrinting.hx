// Macro-time ExprTools: `e.toString()` prints as haxe.macro.Printer does,
// `e.map(f)` rebuilds through `f`, `${expr}` splices a computed Expr, and
// expression metadata annotates the whole expression that follows it.
// Context.getLocalImports lists the module's imports, latest first.
import haxe.macro.Expr;
import haxe.ds.StringMap as SM;
using haxe.macro.Tools;

class MacroExprToolsAndPrinting {
	static macro function show(e:Expr) return macro $v{e.toString()};

	static macro function paren(e:Expr) {
		function add(e:Expr) return macro (${e.map(add)});
		return macro $v{add(e).toString()};
	}

	static macro function imports() {
		return macro $v{haxe.macro.Context.getLocalImports().map(i -> Std.string(i.mode) + ":" + i.path.map(p -> p.name).join("."))};
	}

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		check("binop", show(a + b * c), "a + b * c");
		check("block", show({ var x = 1; x++; }), "{\n\tvar x = 1;\n\tx++;\n}");
		check("switch", show(switch x { case 1: "a"; case _: "b"; }), "switch x {\n\tcase 1:\"a\";\n\tcase _:\"b\";\n}");
		check("arrow", show(x -> x * 2), "x -> x * 2");
		check("object", show({a: 1, b: [1, 2]}), "{ a : 1, b : [1, 2] }");
		check("map", paren(@:test x = 1), "(@:test ((x) = (1)))");
		check("imports", imports().join(","), "IAsName(SM):haxe.ds.StringMap,INormal:haxe.macro.Expr");
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
